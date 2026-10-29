# GReg re-implementation: decisions for everything the paper leaves open

Paper: Shi et al., *GReg: Geometry-Aware Region Refinement for Sign Language Video Generation*, ICCV 2025.
No official code was available when this was written (2026-10-14). Everything below is a
choice we made; each one is a config switch so it can be revisited.

## 1. Normal-map rendering (risk #1 – resolved)

| Item | Decision | Evidence |
|---|---|---|
| Camera | Pinhole, OpenCV axes, `X_cam = X + cam_trans`, `u = X/Z·f + c`, **per-frame** `focal/princpt/cam_trans` from the `.pkl` | Projected SMPL-X joints vs. the `.pkl` 2D keypoints: 1.6–5.7 px error at native resolution on 12/12 clips, better than `vis.py`'s frame-0 shortcut on every clip |
| Renderer | Pure-PyTorch tiled rasteriser (`greg/render/rasterize.py`, backend `torch`); `nvdiffrast` optional | No compilation; CPU ≈0.5 s/frame at native res, GPU much faster; unit-tested |
| Normals | Area-weighted vertex normals interpolated with barycentrics, camera space, encoded `(n+1)/2`, background 0, mask in alpha | Smooth shading like Fig. 1 |
| Crop | One square crop per clip from the projected body clipped to the frame (+15 % margin), resized to 256; intrinsics adjusted (`Camera.cropped`) so the normal map is rendered directly at 256² | `scripts/02_preprocess_hamnosys.py` |
| Shape | Optimized-shape release (`betas` from the `.pkl`); default-shape release only for cross-person experiments | Fig. 1a requires the mesh silhouette to match the real body |
| Frames | fps rule: `-r 25` only for >30 fps videos, otherwise native; row *i* ↔ frame `total_valid_index[i]` | Verified on 12 clips, see memory/notebook Cell 5 |

## 2. Key-region boxes, Hr×Wr, σ

* Centres = mean of the **projected SMPL-X joints** of each region (face landmarks + nose; wrist + finger joints + fingertips).
  Not HRNet: at inference there is no target image to run HRNet on, and the SMPL-X joints are
  exactly aligned with the normal map the generator sees. HRNet is used only for the AKD metric.
* Boxes: fixed **64×64** squares at 256², centred and clamped into the frame (no resampling, so RE
  outputs paste back "at their original positions"). `model.region_size`.
* σ (Eq. 1) = **32 px** = region_size / 2. `model.gaussian_sigma`.
* Regions whose centre leaves the frame (hands down) get weight 0 in the regional losses
  (`region_visibility`).
* SAM: not used. The SMPL-X mesh already yields a pixel-exact hand/face mask for free
  (`parts/*.png`, from the LBS skinning weights – identical to the MANO vertex ids). If a mask-based
  loss is wanted later, use that; SAM would add a heavy dependency and a second source of
  misalignment.
* Hr×Wr are the same for face and hands (the paper uses one symbol for all three).

## 3. What the Regional Expert predicts

`R_r` is a **1-channel map trained to match the gradient map (Eq. 2) of the grayscale target RGB
region**, conditioned on the style code of the source image. Reasons: the text says REs are
"constrained by the target regional gradient"; the RE also receives the normal crop, so predicting
the gradient of the *normal* would be trivial and need no style; Table 2's "Grayscale/RGB"
rows are the alternatives (`loss.struct_target: grayscale | rgb`). Eq. 8's `I_{t,r}` is read as a
typo for the gradient target. The three priors are pasted into blank canvases → `R ∈ R^{3×H×W}`.

## 4. Losses

`L = λ_per L_per + λ_rper L_r-per + λ_struct L_struct + λ_adv L_adv` with `λ_prior ≡ λ_struct`.

* No λ values in the paper. Defaults (`configs/hamnosys_greg.yaml`): per 1, r-per 1 (VGG layers
  already weighted ×10 as in TPSMM), struct 10, adv 1. **Placeholders – tune on a validation split.**
* Regional perceptual loss on RGB crops upsampled to 128² (so VGG sees enough pixels).
* Adversarial: 3 PatchGAN discriminators (spectral norm), input = RGB crop ⊕ its gradient map.
  Fake pair uses the gradient of the **generated** crop (Eq. 10's form, assumed also for Eq. 9).
  GAN loss `bce` (paper's log form, non-saturating for G); `hinge`/`lsgan` available.
* RE is trained by `L_struct` only: the prior is **detached** before entering the generator
  (`model.detach_prior`, our reading of "we separately train RE"). Set to `false` for joint training.

## 5. GRM placement / attention size

GRMs at decoder resolutions **32, 64, 128** (three blocks, as drawn in Fig. 3) using
`F.scaled_dot_product_attention` (no explicit N×N matrix on fused backends). Single head (Eq. 4).
An output 1×1 conv initialised to zero makes each GRM start as identity (`gr.zero_init_out`).
Gradient encoder = small CNN pyramid producing `X_g` at each GRM resolution.

## 6. Training order

* Stage 1 (`05_train_flow.py`): TPSMM with the keypoint detector and dense-motion network on
  **normal maps**, inpainting network reconstructing the RGB target from the warped RGB source;
  losses = pyramid perceptual + keypoint equivariance (on normals) + warp loss (TPSMM defaults,
  dropout schedule included). 100 epochs, Adam 2e-4, ×0.1 at 70/90.
* Stage 2 (`06_train_greg.py`): flow **frozen** for all 100 epochs (`model.freeze_flow: true`);
  RE + style encoder + gradient encoder + generator + 3 D's trained jointly with the loss above.
* Style encoder: 5-layer strided CNN + GAP + FC, `C_style = 256`.
* Split: random 8:2 over clips, `seed 0` (`scripts/04_make_splits.py`, recorded in `split_info.json`).
* Batch size: placeholder 8 (flow) / 4 (GReg) — set to what the GPU allows.

## 7. AKD

HRNet-W32 COCO-WholeBody through MMPose with the full-frame box, exactly like the PGMM repo
(`scripts/09_eval_akd.py`). AKD = mean Euclidean distance of face (23–90) and hand (91–132)
keypoints normalised by the image size, counted where the real-frame confidence ≥ 0.3.

## 8. Fade-in / fade-out frames (GSL)

The Dicta-Sign Greek (GSL) videos fade in from black and out to black, and SignAvatars'
`total_valid_index` keeps those frames. In the first full run the reconstruction source (row 0) of
59 % of the GSL test clips was almost black, so the whole clip was regenerated from a dark image:
GSL L1 0.076 vs 0.014–0.018 for the other sources (14 % of the test frames, but most of the error).
`scripts/02b_mark_valid_rows.py` stores `valid_range` per clip (leading / trailing rows darker than
0.9 × the clip median brightness; rows in the middle of a clip are never dropped), and
`HamNoSysCache.valid_rows` restricts training pairs, the reconstruction source (first frame after
the fade-in) and the evaluated targets to it. No other source is affected (checked by the script).
The paper does not say how it handled this, so this is a protocol choice to report.

## 9. Improvements beyond the paper: A1 (mesh flow) and B1 (fitted regions)

Not part of the paper; off by default (`configs/hamnosys_greg.yaml` still reproduces the paper), on in
`configs/hamnosys_greg_mesh.yaml`. Full description, formulas and verification: `docs/A1_B1_kien_truc.md`.

* **A1** `model.flow.type: mesh`: TPSMM is replaced by the analytic SMPL-X correspondence
  (`greg/geometry/mesh_flow.py`): full-resolution flow, z-buffer visibility V with back-face and
  same-surface tests, stretch confidence C, O = V·C plus a zero-initialised learned residual
  (`model.occlusion_refine`, regularised by `loss.weights.occ`). No stage 1. Needs `scripts/02c_mesh_buffers.py`.
* **B1** `model.regions.mode: adaptive` + `loss.region_mask_mode: multiply`: hand boxes fitted to the
  hand joints (face box unchanged), regional VGG / discriminator inputs multiplied by the mesh part mask,
  `L_struct` weighted 1 on the region / `struct_bg_weight` elsewhere. Works with TPSMM too (masks from `parts/*.png`).
* Protocol option `data.source_first_prob`: share of training pairs whose source is the first valid row
  (the evaluation's rest-pose source). Report separately from the paper protocol.
* Evaluation adds `hand_l1`, `face_l1`, `arm_l1`, `disocc_frac` and per-motion-bin means
  (`wrist_disp` < 10 / 10-40 / > 40 px); `region_l1` on fixed 64x64 boxes is kept for comparability.

## Other

* Image-level L1/PSNR/SSIM/LPIPS and region-level L1 in `greg/eval/metrics.py`; FVD through the
  PGMM repo's evaluator via `10_export_fvd_clips.py` (16 uniform frames, NTHWC uint8).
* Cross-person: driving poses from clip B re-rendered with A's `betas` and A's camera
  (`08_eval_cross_person.py --shape-from source`), which is why both SignAvatars shape releases are useful.
* Ablations of Table 2 that are implemented: w/o RPG, w/o GR, w/o adv, Grayscale/RGB prior targets.
  Skeleton/heatmap representations and NTED/SignGAN baselines are not implemented here
  (PGMM lives in the sibling repo).
