"""Atomic checkpoint save / load."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import torch


def save_checkpoint(state: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str | Path, map_location="cpu") -> Dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)


def load_module_state(module: torch.nn.Module, state: Dict[str, Any], strict: bool = True) -> None:
    """Load a state dict, tolerating a ``module.`` prefix from DataParallel."""
    if any(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}
    module.load_state_dict(state, strict=strict)
