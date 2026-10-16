"""Video probing / frame extraction and the ``.pkl`` <-> frame alignment rule.

Rule (verified on 12 clips of all 5 sources, 2026-10-14, see docs/decisions.md):
* videos above 30 fps (DGS korpusdict, 50 fps) are resampled with ``ffmpeg -r 25``;
  everything else keeps its native rate;
* row ``i`` of the ``.pkl`` corresponds to extracted frame ``total_valid_index[i]``
  (0-based).  Some clips drop a few leading / trailing frames.

``ffmpeg`` is taken from PATH or from the ``imageio-ffmpeg`` wheel; ``ffprobe``
from PATH when available (otherwise the stream info is parsed from ``ffmpeg -i``).
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

FPS_RESAMPLE_ABOVE = 30.0
TARGET_FPS = 25


def ffmpeg_rate_args(fps: float) -> List[str]:
    """``["-r", "25"]`` for high-frame-rate videos, ``[]`` otherwise."""
    return ["-r", str(TARGET_FPS)] if float(fps) > FPS_RESAMPLE_ABOVE else []


def check_alignment(num_frames: int, frame_index: np.ndarray) -> str:
    """Classify how the ``.pkl`` rows map onto ``num_frames`` extracted frames.

    * ``exact``: one row per frame, starting at frame 0;
    * ``subset_by_total_valid_index``: every index fits inside the video;
    * ``MISMATCH``: some index points past the last frame.
    """
    idx = np.asarray(frame_index).astype(np.int64)
    if idx.size == 0 or num_frames <= 0:
        return "MISMATCH"
    if idx.max() >= num_frames or idx.min() < 0:
        return "MISMATCH"
    if idx.size == num_frames and idx.min() == 0 and np.all(np.diff(idx) == 1):
        return "exact"
    return "subset_by_total_valid_index"


# ----------------------------------------------------------------------------- binaries
def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:  # pragma: no cover
        raise RuntimeError("ffmpeg not found on PATH and imageio-ffmpeg is not installed") from e


def ffprobe_exe() -> Optional[str]:
    exe = shutil.which("ffprobe")
    if exe:
        return exe
    # ffprobe usually sits next to a system ffmpeg
    ff = shutil.which("ffmpeg")
    if ff:
        cand = Path(ff).with_name("ffprobe" + Path(ff).suffix)
        if cand.exists():
            return str(cand)
    return None


def _run(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")


def _parse_rate(s: str) -> float:
    if "/" in s:
        num, den = s.split("/")
        return float(num) / float(den) if float(den) else 0.0
    return float(s)


# ------------------------------------------------------------------------------- probe
def probe_video(path: str | Path) -> Dict[str, float]:
    """-> ``{"width", "height", "fps", "num_frames"}`` of the first video stream."""
    path = str(path)
    probe = ffprobe_exe()
    if probe:
        r = _run([probe, "-v", "error", "-select_streams", "v:0", "-count_packets",
                  "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate,nb_read_packets",
                  "-of", "json", path])
        if r.returncode == 0 and r.stdout.strip():
            s = json.loads(r.stdout)["streams"][0]
            fps = _parse_rate(s.get("avg_frame_rate") or "0/1") or _parse_rate(s.get("r_frame_rate") or "0/1")
            return {"width": int(s["width"]), "height": int(s["height"]), "fps": float(fps),
                    "num_frames": int(s.get("nb_read_packets", 0))}
    # fallback: parse `ffmpeg -i`
    r = _run([ffmpeg_exe(), "-hide_banner", "-i", path])
    m = re.search(r"Video:.*?(\d{2,5})x(\d{2,5}).*?(\d+(?:\.\d+)?) fps", r.stderr, re.S)
    if not m:
        raise RuntimeError(f"could not probe {path}: {r.stderr[-400:]}")
    n = count_frames(path, fps=float(m.group(3)), resample=False)
    return {"width": int(m.group(1)), "height": int(m.group(2)), "fps": float(m.group(3)), "num_frames": n}


def count_frames(path: str | Path, fps: float, resample: bool = True) -> int:
    """Number of frames ffmpeg would output (after the fps rule when ``resample``)."""
    rate = ffmpeg_rate_args(fps) if resample else []
    r = _run([ffmpeg_exe(), "-v", "error", "-i", str(path), *rate, "-f", "null", "-", "-progress", "pipe:1"])
    frames = [l for l in r.stdout.splitlines() if l.startswith("frame=")]
    return int(frames[-1].split("=")[1]) if frames else 0


# ----------------------------------------------------------------------------- extract
def extract_frames(video: str | Path, out_dir: str | Path, fps: float, quality: int = 2,
                   overwrite: bool = False, ext: str = "jpg") -> int:
    """Write ``out_dir/000000.ext, 000001.ext, ...`` (0-based) applying the fps rule; returns the count."""
    out = Path(out_dir)
    if out.exists() and overwrite:
        shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    existing = sorted(out.glob(f"*.{ext}"))
    if existing and not overwrite:
        return len(existing)
    cmd = [ffmpeg_exe(), "-v", "error", "-y", "-i", str(video), *ffmpeg_rate_args(fps),
           "-start_number", "0"]
    if ext.lower() in ("jpg", "jpeg"):
        cmd += ["-q:v", str(int(quality))]
    cmd.append(str(out / f"%06d.{ext}"))
    r = _run(cmd)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {video}: {r.stderr[-400:]}")
    return len(list(out.glob(f"*.{ext}")))


def frame_path(frames_dir: str | Path, index: int, ext: str = "jpg") -> Path:
    return Path(frames_dir) / f"{int(index):06d}.{ext}"


__all__ = ["ffmpeg_rate_args", "check_alignment", "probe_video", "count_frames", "extract_frames",
           "frame_path", "ffmpeg_exe", "ffprobe_exe", "FPS_RESAMPLE_ABOVE", "TARGET_FPS"]
