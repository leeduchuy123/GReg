"""Thin-Plate-Spline Motion Model (Zhao & Zhang, CVPR 2022) -- self-contained port.

GReg uses TPSMM as its flow estimator: "ResNet-18 backbone, 50 keypoints in
groups of 5 -> 10 TPS transforms, a single occlusion mask".  Here the
keypoint detector and the dense-motion network see the *normal maps*
(N_s, N_t); the inpainting network (only used to pre-train the flow) warps the
RGB source image.

Module names follow upstream (``kp_detector``, ``dense_motion_network``,
``inpainting_network``) but the state dicts are not guaranteed to be
interchangeable with upstream checkpoints.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tvmodels

from greg.models.blocks import (AntiAliasInterpolation2d, DownBlock2d, Hourglass, ResBlock2d, SameBlock2d,
                                UpBlock2d, kp2gaussian, make_coordinate_grid)


# ----------------------------------------------------------------------------- keypoints
class KPDetector(nn.Module):
    """ResNet-18 regressing ``num_tps * 5`` keypoints in [-1, 1]."""

    def __init__(self, num_tps: int = 10, num_channels: int = 3, pretrained_backbone: bool = False):
        super().__init__()
        self.num_tps = num_tps
        weights = tvmodels.ResNet18_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        self.fg_encoder = tvmodels.resnet18(weights=weights)
        if num_channels != 3:
            self.fg_encoder.conv1 = nn.Conv2d(num_channels, 64, 7, stride=2, padding=3, bias=False)
        self.fg_encoder.fc = nn.Linear(self.fg_encoder.fc.in_features, num_tps * 5 * 2)

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        kp = torch.sigmoid(self.fg_encoder(image)) * 2 - 1
        return {"fg_kp": kp.view(image.shape[0], self.num_tps * 5, 2)}


# ----------------------------------------------------------------------------------- TPS
class TPS:
    """Thin-plate-spline warps.  ``mode='kp'``: one TPS per group of 5 keypoints
    mapping driving -> source.  ``mode='random'``: random affine+TPS for the
    equivariance loss."""

    def __init__(self, mode: str, bs: int, **kwargs):
        self.bs = bs
        self.mode = mode
        if mode == "random":
            noise = torch.normal(mean=0, std=kwargs["sigma_affine"] * torch.ones([bs, 2, 3]))
            self.theta = noise + torch.eye(2, 3).view(1, 2, 3)
            self.control_points = make_coordinate_grid((kwargs["points_tps"], kwargs["points_tps"]))
            self.control_points = self.control_points.unsqueeze(0)
            self.control_params = torch.normal(mean=0, std=kwargs["sigma_tps"] * torch.ones([bs, 1, kwargs["points_tps"] ** 2]))
        elif mode == "kp":
            kp_1, kp_2 = kwargs["kp_1"], kwargs["kp_2"]          # (B, G, 5, 2) driving, source
            device, dt = kp_1.device, kp_1.dtype
            self.gs = kp_1.shape[1]
            n = kp_1.shape[2]
            K = torch.norm(kp_1[:, :, :, None] - kp_1[:, :, None, :], dim=4, p=2) ** 2
            K = K * torch.log(K + 1e-9)
            one1 = torch.ones(bs, self.gs, n, 1, device=device, dtype=dt)
            kp_1p = torch.cat([kp_1, one1], 3)
            zero = torch.zeros(bs, self.gs, 3, 3, device=device, dtype=dt)
            P = torch.cat([kp_1p, zero], 2)
            L = torch.cat([K, kp_1p.permute(0, 1, 3, 2)], 2)
            L = torch.cat([L, P], 3)
            zero = torch.zeros(bs, self.gs, 3, 2, device=device, dtype=dt)
            Y = torch.cat([kp_2, zero], 2)
            L = L + torch.eye(L.shape[2], device=device, dtype=dt).expand(L.shape) * 0.01
            param = torch.matmul(torch.inverse(L), Y)
            self.theta = param[:, :, n:, :].permute(0, 1, 3, 2)   # (B, G, 2, 3)
            self.control_points = kp_1
            self.control_params = param[:, :, :n, :]
        else:
            raise ValueError(mode)

    def transform_frame(self, frame: torch.Tensor) -> torch.Tensor:
        grid = make_coordinate_grid(frame.shape[2:], frame.dtype, frame.device).unsqueeze(0)
        grid = grid.view(1, frame.shape[2] * frame.shape[3], 2)
        shape = [self.bs, frame.shape[2], frame.shape[3], 2]
        if self.mode == "kp":
            shape.insert(1, self.gs)
        return self.warp_coordinates(grid).view(*shape)

    def warp_coordinates(self, coordinates: torch.Tensor) -> torch.Tensor:
        theta = self.theta.to(coordinates.dtype).to(coordinates.device)
        control_points = self.control_points.to(coordinates.dtype).to(coordinates.device)
        control_params = self.control_params.to(coordinates.dtype).to(coordinates.device)
        if self.mode == "kp":
            transformed = torch.matmul(theta[:, :, :, :2], coordinates.permute(0, 2, 1)) + theta[:, :, :, 2:]
            distances = coordinates.view(coordinates.shape[0], 1, 1, -1, 2) - control_points.view(self.bs, self.gs, -1, 1, 2)
            distances = (distances ** 2).sum(-1)
            result = distances * torch.log(distances + 1e-9)
            result = torch.matmul(result.permute(0, 1, 3, 2), control_params)
            transformed = transformed.permute(0, 1, 3, 2) + result
        else:  # random
            theta = theta.unsqueeze(1)
            transformed = torch.matmul(theta[:, :, :, :2], coordinates.unsqueeze(-1)) + theta[:, :, :, 2:]
            transformed = transformed.squeeze(-1)
            ances = coordinates.view(coordinates.shape[0], -1, 1, 2) - control_points.view(1, 1, -1, 2)
            distances = (ances ** 2).sum(-1)
            result = distances * torch.log(distances + 1e-9)
            result = result * control_params
            result = result.sum(dim=2).view(self.bs, coordinates.shape[1], 1)
            transformed = transformed + result
        return transformed


# ---------------------------------------------------------------------------- dense motion
class DenseMotionNetwork(nn.Module):
    """Predicts the backward sampling grid ("deformation", in [-1, 1]) and
    occlusion mask(s) from source image/normal + keypoints of source & driving."""

    def __init__(self, block_expansion=64, num_blocks=5, max_features=1024, num_tps=10, num_channels=3,
                 scale_factor=0.25, multi_mask=False, kp_variance=0.01):
        super().__init__()
        self.num_tps = num_tps
        self.kp_variance = kp_variance
        self.scale_factor = scale_factor
        self.multi_mask = multi_mask
        if scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, scale_factor)
        self.hourglass = Hourglass(block_expansion, in_features=num_channels * (num_tps + 1) + num_tps * 5 + 1,
                                   num_blocks=num_blocks, max_features=max_features)
        self.maps = nn.Conv2d(self.hourglass.out_filters, num_tps + 1, kernel_size=7, padding=3)
        if multi_mask:
            self.up_nums = int(math.log(1 / scale_factor, 2))
            self.occlusion_num = 4
            channel = [self.hourglass.out_filters // (2 ** i) for i in range(self.up_nums)]
            self.up = nn.ModuleList([UpBlock2d(channel[i], channel[i] // 2) for i in range(self.up_nums)])
            lc = self.hourglass.level_channels
            channel = [lc[-i - 1] for i in range(self.occlusion_num - self.up_nums)][::-1]
            channel[-1] = self.hourglass.out_filters  # last level after conv/norm
            channel += [self.hourglass.out_filters // (2 ** (i + 1)) for i in range(self.up_nums)]
            self.occlusion = nn.ModuleList([nn.Conv2d(c, 1, kernel_size=7, padding=3) for c in channel])
        else:
            self.occlusion = nn.ModuleList([nn.Conv2d(self.hourglass.out_filters, 1, kernel_size=7, padding=3)])

    def create_heatmap_representations(self, source, kp_driving, kp_source):
        spatial = source.shape[2:]
        heat = kp2gaussian(kp_driving["fg_kp"], spatial, self.kp_variance) - kp2gaussian(kp_source["fg_kp"], spatial, self.kp_variance)
        zeros = torch.zeros(heat.shape[0], 1, *spatial, dtype=heat.dtype, device=heat.device)
        return torch.cat([zeros, heat], dim=1)

    def create_transformations(self, source, kp_driving, kp_source):
        bs, _, h, w = source.shape
        kp_1 = kp_driving["fg_kp"].view(bs, -1, 5, 2)
        kp_2 = kp_source["fg_kp"].view(bs, -1, 5, 2)
        trans = TPS(mode="kp", bs=bs, kp_1=kp_1, kp_2=kp_2)
        driving_to_source = trans.transform_frame(source)                       # (B, G, h, w, 2)
        identity = make_coordinate_grid((h, w), kp_1.dtype, kp_1.device).view(1, 1, h, w, 2).repeat(bs, 1, 1, 1, 1)
        return torch.cat([identity, driving_to_source], dim=1)                 # (B, G+1, h, w, 2)

    def create_deformed_source_image(self, source, transformations):
        bs, _, h, w = source.shape
        rep = source.unsqueeze(1).repeat(1, self.num_tps + 1, 1, 1, 1).view(bs * (self.num_tps + 1), -1, h, w)
        grid = transformations.view(bs * (self.num_tps + 1), h, w, -1)
        deformed = F.grid_sample(rep, grid, align_corners=True)
        return deformed.view(bs, self.num_tps + 1, -1, h, w)

    @staticmethod
    def dropout_softmax(x, p):
        drop = (torch.rand(x.shape[0], x.shape[1], device=x.device) < (1 - p)).to(x.dtype)
        drop[..., 0] = 1
        drop = drop.unsqueeze(-1).unsqueeze(-1)
        x = x.exp()
        x = x * drop
        return x / (x.sum(1, keepdim=True) + 1e-6)

    def forward(self, source, kp_driving, kp_source, dropout_flag=False, dropout_p=0.0) -> Dict[str, torch.Tensor]:
        if self.scale_factor != 1:
            source = self.down(source)
        bs, _, h, w = source.shape
        out = {}
        heat = self.create_heatmap_representations(source, kp_driving, kp_source)
        transformations = self.create_transformations(source, kp_driving, kp_source)
        deformed = self.create_deformed_source_image(source, transformations)
        out["deformed_source"] = deformed
        inp = torch.cat([heat, deformed.view(bs, -1, h, w)], dim=1)
        prediction = self.hourglass(inp)
        contribution = self.maps(prediction[-1])
        contribution = self.dropout_softmax(contribution, dropout_p) if dropout_flag else F.softmax(contribution, dim=1)
        out["contribution_maps"] = contribution
        deformation = (transformations.permute(0, 1, 4, 2, 3) * contribution.unsqueeze(2)).sum(1)
        out["deformation"] = deformation.permute(0, 2, 3, 1)      # (B, h, w, 2) sampling grid
        occlusion = []
        if self.multi_mask:
            for i in range(self.occlusion_num - self.up_nums):
                occlusion.append(torch.sigmoid(self.occlusion[i](prediction[self.up_nums - self.occlusion_num + i])))
            pred = prediction[-1]
            for i in range(self.up_nums):
                pred = self.up[i](pred)
                occlusion.append(torch.sigmoid(self.occlusion[i + self.occlusion_num - self.up_nums](pred)))
        else:
            occlusion.append(torch.sigmoid(self.occlusion[0](prediction[-1])))
        out["occlusion_map"] = occlusion
        return out


# ------------------------------------------------------------------------------- warping
def resize_deformation(deformation: torch.Tensor, size) -> torch.Tensor:
    """(B, h, w, 2) -> (B, H, W, 2) bilinear."""
    if tuple(deformation.shape[1:3]) == tuple(size):
        return deformation
    d = deformation.permute(0, 3, 1, 2)
    d = F.interpolate(d, size=size, mode="bilinear", align_corners=True)
    return d.permute(0, 2, 3, 1)


def warp(x: torch.Tensor, deformation: torch.Tensor, padding_mode: str = "zeros") -> torch.Tensor:
    """Backward-warp (B, C, H, W) with a (B, h, w, 2) sampling grid (resized to H x W)."""
    grid = resize_deformation(deformation, x.shape[-2:])
    return F.grid_sample(x, grid, mode="bilinear", padding_mode=padding_mode, align_corners=True)


def resize_occlusion(occlusion: torch.Tensor, size) -> torch.Tensor:
    if tuple(occlusion.shape[-2:]) == tuple(size):
        return occlusion
    return F.interpolate(occlusion, size=size, mode="bilinear", align_corners=True)


# ---------------------------------------------------------------------------- inpainting
class InpaintingNetwork(nn.Module):
    """TPSMM generator used only to pre-train the flow estimator."""

    def __init__(self, num_channels=3, block_expansion=64, max_features=512, num_down_blocks=3, multi_mask=False):
        super().__init__()
        self.num_down_blocks = num_down_blocks
        self.multi_mask = multi_mask
        self.first = SameBlock2d(num_channels, block_expansion, kernel_size=7, padding=3)
        down, up, res = [], [], []
        in_f = [min(max_features, block_expansion * (2 ** i)) for i in range(num_down_blocks)]
        out_f = [min(max_features, block_expansion * (2 ** (i + 1))) for i in range(num_down_blocks)]
        for i in range(num_down_blocks):
            down.append(DownBlock2d(in_f[i], out_f[i]))
        self.down_blocks = nn.ModuleList(down)
        up_in = out_f[::-1]                         # deepest first
        up_out = in_f[::-1]
        for i in range(num_down_blocks):
            in_ch = up_in[i] if i == 0 else up_in[i] * 2
            up.append(UpBlock2d(in_ch, up_out[i]))
            res.append(ResBlock2d(in_ch))
            res.append(ResBlock2d(in_ch))
        self.up_blocks = nn.ModuleList(up)
        self.resblock = nn.ModuleList(res)
        self.final = nn.Conv2d(block_expansion * 2, num_channels, kernel_size=7, padding=3)

    def forward(self, source_image, dense_motion) -> Dict[str, torch.Tensor]:
        out = self.first(source_image)
        enc = [out]
        for b in self.down_blocks:
            out = b(out)
            enc.append(out)
        deformation = dense_motion["deformation"]
        occ = dense_motion["occlusion_map"]
        n = len(enc)
        warped_maps = []
        out = warp(enc[-1], deformation)
        o = resize_occlusion(occ[0], out.shape[-2:])
        out = out * o
        warped_maps.append(warp(enc[-1].detach(), deformation) * o.detach())
        for i in range(self.num_down_blocks):
            out = self.resblock[2 * i](out)
            out = self.resblock[2 * i + 1](out)
            out = self.up_blocks[i](out)
            skip = enc[n - 2 - i]
            idx = (i + 1) if self.multi_mask else 0
            o = resize_occlusion(occ[idx], skip.shape[-2:])
            skip_w = warp(skip, deformation) * o
            warped_maps.append(warp(skip.detach(), deformation) * o.detach())
            out = torch.cat([out, skip_w], dim=1)
        deformed_source = warp(source_image, deformation)
        o_last = resize_occlusion(occ[-1], out.shape[-2:])
        pred = torch.sigmoid(self.final(out))
        pred = pred * (1 - o_last) + deformed_source * o_last
        return {"prediction": pred, "deformed": deformed_source, "warped_encoder_maps": warped_maps,
                "occlusion_map": occ, "deformation": deformation}

    def encode(self, image) -> List[torch.Tensor]:
        out = self.first(image)
        enc = [out]
        for b in self.down_blocks:
            out = b(out)
            enc.append(out)
        return enc


# ----------------------------------------------------------------------------- estimator
class FlowEstimator(nn.Module):
    """KPDetector + DenseMotionNetwork on normal maps -> (deformation, occlusion).

    ``forward(normal_source, normal_driving)`` returns a dict with
    ``deformation`` (B, h, w, 2), ``occlusion`` (B, 1, h, w) (last/only mask),
    plus the raw dense-motion output.
    """

    def __init__(self, num_tps=10, num_channels=3, dense_motion: Optional[dict] = None, multi_mask=False,
                 pretrained_backbone=False):
        super().__init__()
        dm = dict(block_expansion=64, num_blocks=5, max_features=1024, scale_factor=0.25)
        dm.update(dense_motion or {})
        self.kp_detector = KPDetector(num_tps, num_channels, pretrained_backbone)
        self.dense_motion_network = DenseMotionNetwork(num_tps=num_tps, num_channels=num_channels,
                                                       multi_mask=multi_mask, **dm)

    def forward(self, normal_source, normal_driving, dropout_flag=False, dropout_p=0.0):
        kp_s = self.kp_detector(normal_source)
        kp_d = self.kp_detector(normal_driving)
        dm = self.dense_motion_network(normal_source, kp_d, kp_s, dropout_flag, dropout_p)
        return {"deformation": dm["deformation"], "occlusion": dm["occlusion_map"][-1], "dense_motion": dm,
                "kp_source": kp_s, "kp_driving": kp_d}

    def freeze(self):
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()
        return self
