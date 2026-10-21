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

Hyper-parameters the paper leaves open (Hr×Wr, σ, λ's, `detach_prior`): `04_make_splits.py` also writes
`val.txt` (10 % of the train clips) and `train_fit.txt`; the sweep trains stage 2 on `train_fit` with a short schedule
(`configs/sweep_greg.yaml`, one stage-1 checkpoint shared by all runs) and ranks the runs on the validation clips:

```bash
python scripts/11_sweep.py --sweep configs/sweeps/hamnosys_sweep.yaml --flow-checkpoint outputs/flow/checkpoints/epoch_100.pth --out-root outputs/sweep
```

Training logs `sec_per_step` / `vram_gb` (`logs/loss.csv`) for the GPU budget; `train.val_every` adds a validation curve
(`logs/val.csv`). Region-L1 is always measured on fixed 64×64 boxes (`--eval-region-size`) so runs with different
`model.region_size` stay comparable. FVD can be computed in-repo: `python scripts/12_eval_fvd.py --eval-dir ... --i3d i3d_torchscript.pt --bootstrap 10`
(weights not included; cross-check against the PGMM evaluator first, see `greg/eval/fvd.py`). AKD: `09_eval_akd.py --real-cache DIR`
reuses the real-frame keypoints between runs.

Visual analysis (look at the errors, not only the numbers):
`python scripts/13_visualize.py --checkpoint ... --cache ... --split test --out outputs/greg/vis` ranks clips, renders the
worst / median / best ones as `clip.gif` (N_t | target | output | |error|) plus labelled panels (warping, occlusion O, G_R,
regional priors, generator output, error map, region close-ups with the RE prior vs target gradient) and writes `report.html`.
During stage-2 training `train.val_every` also saves `samples/fixed_epXXX.jpg` (the same validation clips every time, so epochs
can be compared); `08_eval_cross_person.py` writes a `clip.gif` per pair.

Ablations (Table 2): `--override model.ablation.no_rpg=true`, `model.ablation.no_gr=true`,
`loss.weights.adv=0`, `loss.struct_target=grayscale|rgb`.

Local smoke test (CPU, tiny models): `configs/local_debug_*.yaml`, `python -m pytest`.

## Verified so far (2026-10-17, CPU)

* `greg/data/` (pkl reader, frame alignment, gradient map, regions, datasets, splits) was rewritten on
  2026-10-17: the original package had never been committed because `.gitignore`'s `data/` rule also
  matched it (now `/data/`). The released `.pkl` files hold **CUDA** tensors; the reader maps them to the
  CPU while unpickling, so preprocessing also works on CPU-only machines.
* Re-verified end-to-end on 5 real clips (DGS 25/50 fps, GSL with dropped leading frames, LSF, PJM 720p):
  preprocess → overlay (1.9 px mean joint error @256²) → split → flow → GReg → reconstruction eval →
  FVD export → cross-person; 20 unit tests pass.

* `.pkl` ↔ video frame alignment rule and per-frame camera (12 clips, all sources).
* Normal maps overlay the real body like Fig. 1 (`scripts/03_overlay_check.py`, ~2 px joint error at 256²).
* Preprocessing, both training stages (forward/backward/checkpoint/resume), reconstruction and
  cross-person evaluation run end-to-end on a 4-clip cache; 20 unit tests pass.
* Not yet run: full training on GPU, AKD (needs MMPose), FVD (PGMM evaluator).
