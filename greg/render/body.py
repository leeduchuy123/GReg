"""SMPL-X body wrapper: 182-d SignAvatars vector -> vertices / joints / 2D projections.

The camera model follows SignAvatars ``vis.py`` (verified, see
:mod:`greg.data.signavatars`)::

    X_cam = X + cam_trans           # OpenCV axes: x right, y down, z forward
    u = X_cam.x / X_cam.z * fx + cx
    v = X_cam.y / X_cam.z * fy + cy
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch

from greg.data.signavatars import SMPLX_SLICES

# SMPL-X joint indices (smplx package ordering, 55 skeleton joints + extras).
HEAD_JOINTS = (15, 22, 23, 24)                 # head, jaw, left eye, right eye
LEFT_HAND_JOINTS = (20,) + tuple(range(25, 40))    # left wrist + 15 finger joints
RIGHT_HAND_JOINTS = (21,) + tuple(range(40, 55))   # right wrist + 15 finger joints
# Extra joints appended by smplx: 55 nose, 56 reye, 57 leye, 58 rear, 59 lear,
# 60-65 toes/heels, 66-70 left fingertips, 71-75 right fingertips, 76+ face landmarks.
LEFT_FINGERTIPS = tuple(range(66, 71))
RIGHT_FINGERTIPS = tuple(range(71, 76))
FACE_LANDMARKS = tuple(range(76, 144))         # 51 landmarks + 17 contour (use_face_contour=True)
NOSE = 55

PART_BACKGROUND, PART_BODY, PART_FACE, PART_LHAND, PART_RHAND = 0, 1, 2, 3, 4
PART_NAMES = ("background", "body", "face", "lhand", "rhand")


@dataclass
class Camera:
    """Per-frame pinhole intrinsics (pixels) and SMPL-X -> camera translation."""
    focal: torch.Tensor      # (B, 2)
    princpt: torch.Tensor    # (B, 2)
    cam_trans: torch.Tensor  # (B, 3)

    def cropped(self, x0, y0, side, out_size: int) -> "Camera":
        """Intrinsics for a square crop ``[x0, x0+side) x [y0, y0+side)`` resized to ``out_size``."""
        s = out_size / torch.as_tensor(side, dtype=self.focal.dtype, device=self.focal.device)
        s = s.view(-1, 1)
        x0 = torch.as_tensor(x0, dtype=self.focal.dtype, device=self.focal.device).view(-1, 1)
        y0 = torch.as_tensor(y0, dtype=self.focal.dtype, device=self.focal.device).view(-1, 1)
        offset = torch.cat([x0, y0], dim=1)
        return Camera(focal=self.focal * s, princpt=(self.princpt - offset) * s, cam_trans=self.cam_trans)

    def to(self, device) -> "Camera":
        return Camera(self.focal.to(device), self.princpt.to(device), self.cam_trans.to(device))


def project(points: torch.Tensor, cam: Camera, eps: float = 1e-4) -> torch.Tensor:
    """(B, N, 3) SMPL-X-space points -> (B, N, 2) pixel coordinates."""
    xc = points + cam.cam_trans[:, None, :]
    z = xc[..., 2:3] + eps
    uv = xc[..., :2] / z * cam.focal[:, None, :] + cam.princpt[:, None, :]
    return uv


def to_camera(points: torch.Tensor, cam: Camera) -> torch.Tensor:
    return points + cam.cam_trans[:, None, :]


class SMPLXBody:
    """Thin wrapper over ``smplx.SMPLX`` configured like SignAvatars.

    ``model_dir`` must contain ``smplx/SMPLX_NEUTRAL.npz`` (MPI v1.1 release).
    """

    def __init__(self, model_dir: str | Path, gender: str = "neutral", device="cpu",
                 batch_size: int = 1):
        import smplx  # local import so the data layer does not require smplx

        self.model = smplx.create(str(model_dir), "smplx", gender=gender, use_pca=False,
                                  use_face_contour=True, num_betas=10, num_expression_coeffs=10,
                                  batch_size=batch_size).to(device)
        self.device = torch.device(device)
        self.faces = torch.as_tensor(self.model.faces.astype(np.int64), device=self.device)  # (F, 3)
        self.vertex_part = self._vertex_parts()

    # ------------------------------------------------------------------ parts
    def _vertex_parts(self) -> torch.Tensor:
        """Per-vertex part label from the arg-max of the skinning weights.

        The hand labels coincide exactly with MANO_SMPLX_vertex_ids (778 per hand).
        """
        w = self.model.lbs_weights  # (V, 55)
        amax = w.argmax(1)
        part = torch.full_like(amax, PART_BODY)
        part[torch.isin(amax, torch.tensor(HEAD_JOINTS, device=amax.device))] = PART_FACE
        part[torch.isin(amax, torch.tensor(LEFT_HAND_JOINTS, device=amax.device))] = PART_LHAND
        part[torch.isin(amax, torch.tensor(RIGHT_HAND_JOINTS, device=amax.device))] = PART_RHAND
        return part

    # ---------------------------------------------------------------- forward
    @torch.no_grad()
    def forward(self, smplx_vec: torch.Tensor, betas_override: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:
        """Run SMPL-X on a (B, 182) vector.

        Returns ``vertices`` (B, V, 3) and ``joints`` (B, J, 3) in SMPL-X space
        (before ``cam_trans``), plus the camera translation.
        """
        v = smplx_vec.to(self.device).float()
        b = v.shape[0]
        p = {k: v[:, sl] for k, sl in SMPLX_SLICES.items()}
        betas = p["betas"] if betas_override is None else betas_override.to(self.device).expand(b, -1)
        zeros = torch.zeros(b, 3, device=self.device)
        # every parameter is passed explicitly, so the model's internal
        # batch_size=1 buffers (e.g. transl) simply broadcast
        out = self.model(global_orient=p["global_orient"], body_pose=p["body_pose"],
                         left_hand_pose=p["left_hand_pose"], right_hand_pose=p["right_hand_pose"],
                         jaw_pose=p["jaw_pose"], betas=betas, expression=p["expression"],
                         leye_pose=zeros, reye_pose=zeros, transl=None)
        return {"vertices": out.vertices, "joints": out.joints, "cam_trans": p["cam_trans"]}

    __call__ = forward


def region_centers_from_joints(joints_2d: torch.Tensor) -> torch.Tensor:
    """(B, J, 2) projected joints -> (B, 3, 2) centres of face / left hand / right hand.

    Face: mean of the 68 face landmarks (+ nose).  Hands: mean of wrist, finger
    joints and fingertips.  Order matches ``REGION_NAMES``.
    """
    face_idx = list(FACE_LANDMARKS) + [NOSE]
    lh_idx = list(LEFT_HAND_JOINTS) + list(LEFT_FINGERTIPS)
    rh_idx = list(RIGHT_HAND_JOINTS) + list(RIGHT_FINGERTIPS)
    face = joints_2d[:, face_idx].mean(1)
    lh = joints_2d[:, lh_idx].mean(1)
    rh = joints_2d[:, rh_idx].mean(1)
    return torch.stack([face, lh, rh], dim=1)


def region_extents_from_joints(joints_2d: torch.Tensor) -> torch.Tensor:
    """(B, J, 2) -> (B, 3) max side length of the bounding box of each region's joints."""
    out = []
    for idx in (list(FACE_LANDMARKS) + [NOSE],
                list(LEFT_HAND_JOINTS) + list(LEFT_FINGERTIPS),
                list(RIGHT_HAND_JOINTS) + list(RIGHT_FINGERTIPS)):
        pts = joints_2d[:, idx]
        ext = (pts.max(1).values - pts.min(1).values).max(-1).values
        out.append(ext)
    return torch.stack(out, dim=1)


REGION_NAMES: Sequence[str] = ("face", "lhand", "rhand")
