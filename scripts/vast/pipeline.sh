#!/bin/bash
# Overnight pipeline on a vast.ai instance (resume-safe: every stage leaves /workspace/state/<stage>.done).
#   fetch -> preprocess (sharded) -> fade trimming -> splits -> mesh buffers (A1) -> overlay check -> stage 1 (flow)
#   -> E0 diagnostic (TPSMM vs mesh warp) -> A1/B1 ablation (stage 2, short) -> GIFs / panels of base vs A1+B1
#   [-> old hyper-parameter sweep if HP_SWEEP=1]
# docs/A1_B1_kien_truc.md explains the A1/B1 stages.
# BASE_CKPT_DRIVE=<Drive path>: re-use a finished 'base' run of the ablation (run5's sweep_clean/ref) instead of training it.
# FLOW_CKPT=<path>: use an existing stage-1 checkpoint (e.g. fetched from another run's Drive folder, kept OUTSIDE
# /workspace/outputs so it is not synced again) instead of training TPSMM; only base / b1 / E0 use it.
# Layout: /workspace/GReg (code), /workspace/data/raw/{pkls,videos,smplx_dl}, /workspace/outputs, /workspace/state.
# Inputs come from Google Drive via scripts/vast/fetch_data.sh, results are mirrored back by scripts/vast/sync_loop.sh
# (Drive paths for both in scripts/vast/drive.env).
set -uo pipefail
cd /workspace/GReg
RAW=/workspace/data/raw
PKL=$RAW/pkls/hamnosys_pkls_cropFalse_shapeTrue
VID=$RAW/videos
SMPLX=$RAW/smplx_dl/models
CACHE=/workspace/data/hamnosys_cache
OUT=/workspace/outputs
ST=/workspace/state
# every preprocessing shard renders on the GPU (~0.6 GB each): 28 shards on a 16 GB card ran out of memory and
# 94% of the clips failed, so the default is derived from the GPU memory (24 GB -> 14 as before, 16 GB -> 8)
GPU_GB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | awk '{print int($1/1024)}')
SHARDS=${SHARDS:-$(( ${GPU_GB:-24} * 10 / 17 < 14 ? ${GPU_GB:-24} * 10 / 17 : 14 ))}
MESH_SHARDS=${MESH_SHARDS:-4}
WORKERS=${WORKERS:-8}            # DataLoader workers of the stage-2 runs (4 left the GPU idle on a busy host)
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
HP_SWEEP=${HP_SWEEP:-0}
TRAINED_FLOW=$OUT/flow/checkpoints/epoch_100.pth
FLOW_CKPT=${FLOW_CKPT:-$TRAINED_FLOW}
mkdir -p $ST $OUT /workspace/logs

stage() {  # stage <name> <command...>
    local name=$1; shift
    if [ -f "$ST/$name.done" ]; then echo "[skip] $name"; return 0; fi
    echo "[$(date -u +%FT%TZ)] START $name"
    if "$@"; then touch "$ST/$name.done"; echo "[$(date -u +%FT%TZ)] DONE  $name"
    else echo "[$(date -u +%FT%TZ)] FAILED $name"; return 1; fi
}

preprocess() {
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    local pids=()
    for i in $(seq 0 $((SHARDS - 1))); do
        python scripts/02_preprocess_hamnosys.py --pkl-dir $PKL --video-dir $VID --out $CACHE --smplx-dir $SMPLX \
            --device cuda --batch 32 --tmp /workspace/tmp_frames/s$i --shard $i/$SHARDS > /workspace/logs/pre_shard$i.log 2>&1 &
        pids+=($!)
    done
    local rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
    python scripts/02_preprocess_hamnosys.py --pkl-dir $PKL --video-dir $VID --out $CACHE --smplx-dir $SMPLX --merge
    # 02_preprocess records a failing clip and goes on; a run on a fraction of the data must not pass silently
    python - "$CACHE/meta.json" <<'PYEOF' || return 1
import collections, json, sys
clips = json.load(open(sys.argv[1]))["clips"]
st = collections.Counter(v.get("status", "?").split(":")[0] for v in clips.values())
err = st.get("error", 0)
print(f"preprocess check: {dict(st)}")
if err > 0.02 * len(clips):
    sys.exit(f"{err}/{len(clips)} clips failed (see meta.json 'error: ...' entries and logs/pre_shard*.log)")
PYEOF
    return $rc
}

flow() {
    if [ "$FLOW_CKPT" != "$TRAINED_FLOW" ]; then   # reuse a stage-1 checkpoint, no TPSMM training
        [ -f "$FLOW_CKPT" ] || { echo "FLOW_CKPT=$FLOW_CKPT does not exist"; return 1; }
        echo "using the existing stage-1 checkpoint $FLOW_CKPT"; return 0
    fi
    local resume=""; [ -f $OUT/flow/checkpoints/last.pth ] && resume="--resume $OUT/flow/checkpoints/last.pth"
    python scripts/05_train_flow.py --config configs/hamnosys_flow.yaml \
        --override data.cache=$CACHE output_dir=$OUT/flow train.batch_size=8 train.num_workers=8 train.amp=true $resume
}

sweep() {
    python scripts/11_sweep.py --sweep configs/sweeps/hamnosys_sweep.yaml --flow-checkpoint $OUT/flow/checkpoints/epoch_100.pth \
        --out-root $OUT/sweep --cache $CACHE
}

meshbuf() {   # A1: face-index maps + SMPL-X vectors for the analytic mesh flow (no video decoding)
    local pids=()
    for i in $(seq 0 $((MESH_SHARDS - 1))); do
        python scripts/02c_mesh_buffers.py --cache $CACHE --pkl-dir $PKL --smplx-dir $SMPLX --device cuda --batch 32 \
            --shard $i/$MESH_SHARDS --check > /workspace/logs/meshbuf_shard$i.log 2>&1 &
        pids+=($!)
    done
    local rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
    return $rc
}

diag_e0() {   # E0 v2: TPSMM vs mesh warp on the visible arm / hand pixels, per motion bin, with the limb-edge
              # confidence (edge_px 4) and O_geo-weighted errors (v1 results stay in $OUT/diag_e0)
    python scripts/14_diag_mesh_flow.py --cache $CACHE --flow-checkpoint $FLOW_CKPT \
        --smplx-dir $SMPLX --out $OUT/diag_e0_v2 --split val --max-clips 50 --max-rows 16 --edge-px 4
}

reuse_base() {  # 'base' of ablation v2 = run5's sweep_clean/ref (same config, data, split, seed): copy its checkpoint,
                # 11_sweep then resumes it at its last epoch (no training) and only re-evaluates it with the new metrics
    [ -n "${BASE_CKPT_DRIVE:-}" ] || { echo "BASE_CKPT_DRIVE not set: base is trained"; return 0; }
    source scripts/vast/drive.env
    mkdir -p $OUT/ablation_v2/base/checkpoints
    [ -f $OUT/ablation_v2/base/checkpoints/last.pth ] || \
        rclone copyto "$DRIVE$BASE_CKPT_DRIVE" $OUT/ablation_v2/base/checkpoints/last.pth --log-level ERROR || return 1
    python -c "import torch; ck = torch.load('$OUT/ablation_v2/base/checkpoints/last.pth', map_location='cpu', weights_only=False); \
m = ck['config']['model']; assert m['gr'].get('qk_norm') is True, 'base checkpoint without qk_norm'; \
assert ck['config']['loss']['weights']['regional_perceptual'] == 3; \
print('reused base: epoch', ck['epoch'], 'step', ck['step'], 'qk_norm', m['gr']['qk_norm'])"
}

ablation() {  # v2: base (= run5 ref) / a1b1_res / b1 / a1b1 / a1_res, run5's config, short schedule, same seed
    python scripts/11_sweep.py --sweep configs/sweeps/a1_b1_ablation_v2.yaml --flow-checkpoint $FLOW_CKPT \
        --out-root $OUT/ablation_v2 --cache $CACHE --common model.flow.smplx_dir=$SMPLX train.num_workers=$WORKERS
}

visualize() {  # clip.gif (N_t | target | output | |error|) + labelled panels + report.html per run; the same
               # largest-motion val clips (from E0) are rendered for every run so they can be compared side by side
    local keys=""; [ -f $OUT/diag_e0_v2/largest_motion_keys.txt ] && keys=$(tr -d '\r' < $OUT/diag_e0_v2/largest_motion_keys.txt | tr '\n' ' ')
    local rc=0
    for run in base a1b1_res b1 a1b1; do
        local ck=$OUT/ablation_v2/$run/checkpoints/last.pth
        [ -f $ck ] || { echo "no checkpoint for $run, skipped"; continue; }
        python scripts/13_visualize.py --checkpoint $ck --cache $CACHE --split val --out $OUT/vis_v2/$run \
            --rank-clips 60 --worst 4 --median 2 --best 2 --max-frames 96 ${keys:+--keys $keys} || rc=1
    done
    return $rc
}

stage fetch bash scripts/vast/fetch_data.sh || exit 1
stage preprocess preprocess || exit 1
stage validrows python scripts/02b_mark_valid_rows.py --cache $CACHE --workers $(nproc) || exit 1
stage splits python scripts/04_make_splits.py --cache $CACHE --seed 0 || exit 1
stage meshbuf meshbuf || exit 1
stage overlay python scripts/03_overlay_check.py --cache $CACHE --pkl-dir $PKL --out $OUT/overlay --num 12 || true
stage flow flow || exit 1
# v2 stage names: the v1 stages (diag_e0 / ablation / visualize) of an instance that ran v1 are not re-run
stage diag_e0_v2 diag_e0 || true
stage reuse_base reuse_base || exit 1
stage ablation_v2 ablation || exit 1
stage visualize_v2 visualize || true
if [ "$HP_SWEEP" = 1 ]; then stage sweep sweep || exit 1; fi
echo "[$(date -u +%FT%TZ)] PIPELINE_DONE"
