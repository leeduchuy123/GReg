# GReg re-implementation (SignAvatars HamNoSys)

Re-implementation of **GReg: Geometry-Aware Region Refinement for Sign Language Video
Generation** (Shi et al., ICCV 2025) on the SignAvatars HamNoSys subset.
All choices for details the paper does not specify are listed in `docs/decisions.md`.

```
greg/
  data/      .pkl reader (SMPL-X layout + camera), fps rule, gradient map (Eq. 2),
             key regions (boxes / Gaussians / crop & paste), cache datasets, splits
  render/    SMPL-X wrapper, pure-PyTorch rasteriser (+ optional nvdiffrast), normal maps
  models/    TPSMM port (flow), style encoder + modulated conv, Regional Experts,
             gradient encoder + GRM, geometry-aware generator, regional PatchGANs, VGG loss
  losses.py  GReg objective (Eq. 5-10)
  eval/      metrics (L1/PSNR/SSIM/LPIPS/region-L1/AKD/TCD), checkpoint runner
scripts/     02 preprocess · 03 overlay check · 04 splits · 05 train flow · 06 train GReg ·
             07 reconstruction eval · 08 cross-person · 09 AKD (MMPose) · 10 FVD export
configs/     hamnosys_flow.yaml, hamnosys_greg.yaml, local_debug_*.yaml
colab/       download_hamnosys_videos.ipynb, run_greg_pipeline.ipynb
tests/       CPU unit tests (pytest)
```

## Data you need

1. SignAvatars HamNoSys `.pkl` files (optimized shape) – `hamnosys_pkls_cropFalse_shapeTrue/*.pkl`.
2. The RGB videos as `<key>.mp4` – `colab/download_hamnosys_videos.ipynb`.
3. SMPL-X v1.1 model files – `models_smplx_v1_1/models/smplx/SMPLX_{NEUTRAL,MALE,FEMALE}.npz`
   (registered download from smpl-x.is.tue.mpg.de; do not commit them).

## Pipeline

```bash
pip install -r requirements.txt           # torch/torchvision already installed on Colab/Kaggle

# 1. cache: 256x256 frames + normal maps + part maps + projected joints (GPU recommended)
python scripts/02_preprocess_hamnosys.py --pkl-dir PKLS --video-dir VIDEOS --out data/hamnosys_cache \
       --smplx-dir models_smplx_v1_1/models --device cuda
python scripts/03_overlay_check.py --cache data/hamnosys_cache --pkl-dir PKLS --out outputs/overlay
python scripts/04_make_splits.py --cache data/hamnosys_cache --seed 0

# 2. stage 1: TPSMM flow on normal maps
python scripts/05_train_flow.py --config configs/hamnosys_flow.yaml

# 3. stage 2: GReg (flow frozen)
python scripts/06_train_greg.py --config configs/hamnosys_greg.yaml --flow-checkpoint outputs/flow/checkpoints/epoch_100.pth

# 4. evaluation
python scripts/07_eval_reconstruction.py --checkpoint outputs/greg/checkpoints/epoch_100.pth \
       --cache data/hamnosys_cache --out outputs/greg/eval_recon --lpips --save-frames
python scripts/09_eval_akd.py --eval-dir outputs/greg/eval_recon --config <mmpose cfg> --checkpoint <hrnet pth>
python scripts/10_export_fvd_clips.py --eval-dir outputs/greg/eval_recon --out outputs/greg/fvd.npz   # -> PGMM/scripts/10_eval_fvd.py
python scripts/08_eval_cross_person.py --checkpoint ... --cache ... --pkl-dir PKLS --smplx-dir models_smplx_v1_1/models --out outputs/greg/eval_cross
```

Ablations (Table 2): `--override model.ablation.no_rpg=true`, `model.ablation.no_gr=true`,
`loss.weights.adv=0`, `loss.struct_target=grayscale|rgb`.

Local smoke test (CPU, tiny models): `configs/local_debug_*.yaml`, `python -m pytest`.

## Verified so far (2026-10-14, CPU)

* `.pkl` ↔ video frame alignment rule and per-frame camera (12 clips, all sources).
* Normal maps overlay the real body like Fig. 1 (`scripts/03_overlay_check.py`, ~2 px joint error at 256²).
* Preprocessing, both training stages (forward/backward/checkpoint/resume), reconstruction and
  cross-person evaluation run end-to-end on a 4-clip cache; 20 unit tests pass.
* Not yet run: full training on GPU, AKD (needs MMPose), FVD (PGMM evaluator).
