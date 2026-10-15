"""Build run_greg_pipeline.ipynb (Colab / Kaggle runner for the whole GReg pipeline)."""
import json
from pathlib import Path

CELLS = []


def md(t):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": t.strip("\n")})


def code(t):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.strip("\n")})


md("""
# GReg pipeline on Colab (HamNoSys)

Expects on Google Drive (paths in Cell 1):
- the GReg code folder (zip it locally: `GReg/` without `models_smplx_v1_1`, upload as `GReg_code.zip`),
- `.pkl` annotations (optimized shape) and the videos from `download_hamnosys_videos.ipynb`,
- SMPL-X model files `models/smplx/SMPLX_NEUTRAL.npz`.

Run the cells in order. Each stage writes to Drive so a disconnected session can resume.
""")

code(r'''
# Cell 1: Drive + paths
from google.colab import drive
drive.mount('/content/drive')
import os, glob, shutil, subprocess, sys

DRIVE = '/content/drive/MyDrive'
CODE_ZIP   = f'{DRIVE}/GReg_code.zip'                                  # zip of the GReg repo (code only)
PKL_DIR    = glob.glob(f'{DRIVE}/**/hamnosys_pkls_cropFalse_shapeTrue/**/10010.pkl', recursive=True)
PKL_DIR    = os.path.dirname(PKL_DIR[0]) if PKL_DIR else None
VIDEO_DIR  = f'{DRIVE}/Sign Language Datasets/HamNoSys videos'
SMPLX_DIR  = f'{DRIVE}/models_smplx_v1_1/models'                       # contains smplx/SMPLX_NEUTRAL.npz
WORK       = f'{DRIVE}/GReg_runs'                                      # checkpoints / logs / eval on Drive
CACHE_TAR  = f'{WORK}/hamnosys_cache.tar'                              # preprocessed cache archive on Drive
CACHE      = '/content/hamnosys_cache'                                  # local copy used for training
os.makedirs(WORK, exist_ok=True)
print('PKL_DIR', PKL_DIR); print('videos', len(glob.glob(f'{VIDEO_DIR}/*.mp4')))
assert os.path.exists(f'{SMPLX_DIR}/smplx/SMPLX_NEUTRAL.npz'), 'SMPL-X model files missing'

shutil.rmtree('/content/GReg', ignore_errors=True)
subprocess.run(['unzip', '-q', CODE_ZIP, '-d', '/content/GReg_unzip'], check=True)
inner = [d for d in glob.glob('/content/GReg_unzip/**/greg', recursive=True)][0]
shutil.move(os.path.dirname(inner), '/content/GReg')
%cd /content/GReg
!pip -q install smplx lpips imageio-ffmpeg
!python -m pytest -q -x 2>&1 | tail -2
''')

code(r'''
# Cell 2: preprocess (frames + normal maps at 256). ~1-2 h on a T4 for all 5.7k clips.
# Resume-safe: already-done clips are skipped. Set LIMIT for a quick trial.
LIMIT = None   # e.g. 50
if os.path.exists(CACHE_TAR) and not os.path.isdir(CACHE):
    !tar -xf "{CACHE_TAR}" -C /content
extra = f'--limit {LIMIT}' if LIMIT else ''
!python scripts/02_preprocess_hamnosys.py --pkl-dir "{PKL_DIR}" --video-dir "{VIDEO_DIR}" --out "{CACHE}" \
    --smplx-dir "{SMPLX_DIR}" --device cuda --batch 32 --tmp /content/tmp_frames {extra}
!tar -cf "{CACHE_TAR}" -C /content hamnosys_cache && echo "cache archived to Drive"
''')

code(r'''
# Cell 3: alignment check (Fig. 1 style) + split
!python scripts/03_overlay_check.py --cache "{CACHE}" --pkl-dir "{PKL_DIR}" --out "{WORK}/overlay" --num 12
from IPython.display import Image, display
display(Image(f'{WORK}/overlay/overlay_grid.png', width=900))
!python scripts/04_make_splits.py --cache "{CACHE}" --seed 0
''')

code(r'''
# Cell 4: stage 1 - TPSMM flow estimator on normal maps (resume-safe)
FLOW_OUT = f'{WORK}/flow'
resume = f'--resume {FLOW_OUT}/checkpoints/last.pth' if os.path.exists(f'{FLOW_OUT}/checkpoints/last.pth') else ''
!python scripts/05_train_flow.py --config configs/hamnosys_flow.yaml \
    --override data.cache="{CACHE}" output_dir="{FLOW_OUT}" train.batch_size=8 train.num_workers=2 train.amp=false {resume}
''')

code(r'''
# Cell 5: stage 2 - GReg with the flow frozen (resume-safe)
GREG_OUT = f'{WORK}/greg'
flow_ck = f'{FLOW_OUT}/checkpoints/epoch_100.pth'
if not os.path.exists(flow_ck): flow_ck = f'{FLOW_OUT}/checkpoints/last.pth'
resume = f'--resume {GREG_OUT}/checkpoints/last.pth' if os.path.exists(f'{GREG_OUT}/checkpoints/last.pth') else f'--flow-checkpoint {flow_ck}'
!python scripts/06_train_greg.py --config configs/hamnosys_greg.yaml \
    --override data.cache="{CACHE}" output_dir="{GREG_OUT}" train.batch_size=4 train.num_workers=2 {resume}
''')

code(r'''
# Cell 6: reconstruction evaluation (+ frames for AKD / FVD)
ck = f'{GREG_OUT}/checkpoints/epoch_100.pth'
if not os.path.exists(ck): ck = f'{GREG_OUT}/checkpoints/last.pth'
!python scripts/07_eval_reconstruction.py --checkpoint "{ck}" --cache "{CACHE}" --split test \
    --out "{GREG_OUT}/eval_recon" --lpips --save-frames --batch-size 16
!python scripts/10_export_fvd_clips.py --eval-dir "{GREG_OUT}/eval_recon" --out "{GREG_OUT}/fvd_clips.npz"
''')

code(r'''
# Cell 7 (optional): AKD with HRNet-W32 COCO-WholeBody via MMPose (same pins as the PGMM repo)
!pip -q install openmim mmengine==0.10.7 mmcv-lite==2.1.0 mmdet==3.3.0 mmpose==1.1.0 xtcocotools
!mim download mmpose --config td-hm_hrnet-w32_8xb64-210e_coco-wholebody-256x192 --dest /content/hrnet
cfg = glob.glob('/content/hrnet/*.py')[0]; pth = glob.glob('/content/hrnet/*.pth')[0]
!python scripts/09_eval_akd.py --eval-dir "{GREG_OUT}/eval_recon" --config "{cfg}" --checkpoint "{pth}"
''')

code(r'''
# Cell 8: cross-person generation (driving poses re-rendered with the source person's shape)
!python scripts/08_eval_cross_person.py --checkpoint "{ck}" --cache "{CACHE}" --pkl-dir "{PKL_DIR}" \
    --smplx-dir "{SMPLX_DIR}" --out "{GREG_OUT}/eval_cross" --num-pairs 30 --shape-from source
display(Image(glob.glob(f'{GREG_OUT}/eval_cross/*/_grid.png')[0], width=900))
''')

nb = {"cells": CELLS, "metadata": {"colab": {"provenance": [], "gpuType": "T4"}, "accelerator": "GPU",
                                   "kernelspec": {"name": "python3", "display_name": "Python 3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
out = Path(__file__).with_name("run_greg_pipeline.ipynb")
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", out)
