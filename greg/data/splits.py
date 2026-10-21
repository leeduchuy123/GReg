"""Random clip-level train/test split (GReg protocol: 8:2, seed recorded)."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def source_of_key(key: str) -> str:
    """Clip key -> HamNoSys source corpus.

    Same rule as the download notebook: ``pjm_*`` PJM (slownikpjm), ``gsl_*`` GSL
    (Dicta-Sign), numeric ids DGS (korpusdict), everything else (French words in
    capitals) LSF (Dicta-Sign).
    """
    k = key.lower()
    if k.startswith("pjm"):
        return "PJM"
    if k.startswith("gsl"):
        return "GSL"
    if key.isdigit():
        return "DGS"
    return "LSF"


def make_random_split(keys: Iterable[str], train_ratio: float = 0.8, seed: int = 0) -> Dict[str, List[str]]:
    """Deterministic shuffle of the sorted keys -> ``{"train": [...], "test": [...]}`` (both sorted)."""
    keys = sorted(set(keys))
    rng = random.Random(seed)
    rng.shuffle(keys)
    n_train = int(round(len(keys) * float(train_ratio)))
    return {"train": sorted(keys[:n_train]), "test": sorted(keys[n_train:])}


def make_val_split(train_keys: Iterable[str], val_ratio: float = 0.1, seed: int = 0) -> Dict[str, List[str]]:
    """Hold out ``val_ratio`` of the *train* clips for hyper-parameter selection.

    Returns ``{"train_fit": [...], "val": [...]}``; the test split is never touched.
    ``seed + 1`` is used so the shuffle is independent of the train/test one.
    """
    keys = sorted(set(train_keys))
    rng = random.Random(seed + 1)
    rng.shuffle(keys)
    n_val = max(1, int(round(len(keys) * float(val_ratio)))) if keys and val_ratio > 0 else 0
    return {"train_fit": sorted(keys[n_val:]), "val": sorted(keys[:n_val])}


def pick_keys(keys: Iterable[str], n: Optional[int], seed: int = 0) -> List[str]:
    """Deterministic random subset of ``n`` keys (all keys, sorted, if ``n`` is None or too large).

    Sorted keys cluster by corpus (numeric ids = DGS first), so ``keys[:n]`` would be biased.
    """
    keys = sorted(set(keys))
    if not n or n >= len(keys):
        return keys
    return sorted(random.Random(seed).sample(keys, n))


def write_split(split: Dict[str, List[str]], out_dir: str | Path, info: Optional[dict] = None) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, ks in split.items():
        with open(out / f"{name}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(ks) + ("\n" if ks else ""))
    if info is not None:
        with open(out / "split_info.json", "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)


def read_split(path: str | Path) -> List[str]:
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]
