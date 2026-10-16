"""SignAvatars ``.pkl`` reader (HamNoSys / How2Sign / PHOENIX releases).

Each file is a dict (torch tensors or numpy arrays) with

    width, height          video size
    focal, princpt         (N, 2) per-frame pinhole intrinsics in pixels
    smplx                  (N, 182) smoothed SMPL-X parameters, layout below
    unsmooth_smplx         (N, 169) (not used)
    2d                     (N, 106, 3) 2D keypoints (x, y, score) in video pixels
    pred_2d                (N, 106, 2) projected keypoints of the fit
    total_valid_index      (N,) video frame index of every row
    left_valid, right_valid, bb2img_trans

182-d layout (SignAvatars README / vis.py)::

    [0:3]    global_orient   [3:66]   body_pose (21x3)   [66:111]  left_hand_pose (15x3)
    [111:156] right_hand_pose (15x3)  [156:159] jaw_pose  [159:169] betas
    [169:179] expression     [179:182] cam_trans (SMPL-X -> camera translation; transl = 0)

Camera (verified against the 2D keypoints, docs/decisions.md Sec. 1)::

    X_cam = X + cam_trans;  u = X_cam.x / X_cam.z * fx + cx;  v = X_cam.y / X_cam.z * fy + cy
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

SMPLX_DIM = 182
SMPLX_SLICES: Dict[str, slice] = {
    "global_orient": slice(0, 3),
    "body_pose": slice(3, 66),
    "left_hand_pose": slice(66, 111),
    "right_hand_pose": slice(111, 156),
    "jaw_pose": slice(156, 159),
    "betas": slice(159, 169),
    "expression": slice(169, 179),
    "cam_trans": slice(179, 182),
}
NUM_KEYPOINTS_2D = 106


def _to_numpy(x: Any) -> np.ndarray:
    """torch tensor / list / scalar -> numpy (CPU)."""
    if hasattr(x, "detach"):          # torch.Tensor without importing torch here
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _scalar(x: Any) -> int:
    return int(np.asarray(_to_numpy(x)).reshape(-1)[0])


def _per_frame(x: Any, n: int, last: int, name: str) -> np.ndarray:
    """Broadcast (last,) or (1, last) -> (n, last); check (n, last)."""
    a = _to_numpy(x).astype(np.float32)
    if a.ndim == 1 or (a.ndim == 2 and a.shape[0] == 1):
        a = np.broadcast_to(a.reshape(1, last), (n, last)).copy()
    if a.shape != (n, last):
        raise ValueError(f"{name}: expected ({n}, {last}), got {a.shape}")
    return a


class _CPUUnpickler(pickle.Unpickler):
    """The released files hold *CUDA* tensors; map their storages to the CPU while unpickling."""

    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            import io
            import torch

            return lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
        return super().find_class(module, name)


def load_pkl(path: str | Path) -> dict:
    """Unpickle a SignAvatars annotation.  Needs ``torch`` importable (the files hold tensors)."""
    with open(path, "rb") as f:
        try:
            return _CPUUnpickler(f).load()
        except ModuleNotFoundError as e:  # pragma: no cover
            raise RuntimeError(f"{path}: the pickle references {e.name}; install it (torch) to read the file") from e


@dataclass
class ClipAnnotation:
    key: str
    width: int
    height: int
    smplx: np.ndarray                 # (N, 182) float32
    focal: np.ndarray                 # (N, 2)
    princpt: np.ndarray               # (N, 2)
    frame_index: np.ndarray           # (N,) int64 = total_valid_index
    keypoints_2d: Optional[np.ndarray] = None    # (N, 106, 3) ground-truth 2D (x, y, score)
    pred_keypoints_2d: Optional[np.ndarray] = None
    left_valid: Optional[np.ndarray] = None
    right_valid: Optional[np.ndarray] = None
    extra: Dict[str, Any] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------ views
    @property
    def num_rows(self) -> int:
        return int(self.smplx.shape[0])

    @property
    def cam_trans(self) -> np.ndarray:
        return self.smplx[:, SMPLX_SLICES["cam_trans"]]

    @property
    def betas(self) -> np.ndarray:
        return self.smplx[:, SMPLX_SLICES["betas"]]

    def param(self, name: str) -> np.ndarray:
        return self.smplx[:, SMPLX_SLICES[name]]

    def params(self, rows=None) -> Dict[str, np.ndarray]:
        v = self.smplx if rows is None else self.smplx[rows]
        return {k: v[:, sl] for k, sl in SMPLX_SLICES.items()}

    # ---------------------------------------------------------------- loading
    @classmethod
    def from_dict(cls, d: dict, key: str = "") -> "ClipAnnotation":
        smplx = _to_numpy(d["smplx"]).astype(np.float32)
        if smplx.ndim != 2 or smplx.shape[1] != SMPLX_DIM:
            raise ValueError(f"{key}: smplx must be (N, {SMPLX_DIM}), got {smplx.shape}")
        n = smplx.shape[0]
        if "total_valid_index" in d:
            frame_index = _to_numpy(d["total_valid_index"]).astype(np.int64).reshape(-1)
        else:
            frame_index = np.arange(n, dtype=np.int64)
        if frame_index.shape[0] != n:
            raise ValueError(f"{key}: total_valid_index has {frame_index.shape[0]} entries for {n} rows")

        def opt(name):
            return _to_numpy(d[name]) if name in d and d[name] is not None else None

        kp = opt("2d")
        pkp = opt("pred_2d") if "pred_2d" in d else opt("pred2d")      # (N, 106, 2) in the HamNoSys release
        used = {"width", "height", "focal", "princpt", "smplx", "total_valid_index", "2d", "pred_2d", "pred2d",
                "left_valid", "right_valid"}
        return cls(key=key, width=_scalar(d["width"]), height=_scalar(d["height"]), smplx=smplx,
                   focal=_per_frame(d["focal"], n, 2, "focal"), princpt=_per_frame(d["princpt"], n, 2, "princpt"),
                   frame_index=frame_index,
                   keypoints_2d=None if kp is None else kp.astype(np.float32),
                   pred_keypoints_2d=None if pkp is None else pkp.astype(np.float32),
                   left_valid=opt("left_valid"), right_valid=opt("right_valid"),
                   extra={k: v for k, v in d.items() if k not in used})

    @classmethod
    def from_pkl(cls, path: str | Path, key: Optional[str] = None) -> "ClipAnnotation":
        path = Path(path)
        return cls.from_dict(load_pkl(path), key=key if key is not None else path.stem)
