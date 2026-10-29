#!/bin/bash
# Overnight pipeline on a vast.ai instance (resume-safe: every stage leaves /workspace/state/<stage>.done).
#   fetch -> preprocess (sharded) -> fade trimming -> splits -> mesh buffers (A1) -> overlay check -> stage 1 (flow)
#   -> E0 diagnostic (TPSMM vs mesh warp) -> A1/B1 ablation (stage 2, short)  [-> old hyper-parameter sweep if HP_SWEEP=1]
# docs/A1_B1_kien_truc.md explains the A1/B1 stages.
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
SHARDS=${SHARDS:-14}
MESH_SHARDS=${MESH_SHARDS:-4}
HP_SWEEP=${HP_SWEEP:-0}
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
    return $rc
}

flow() {
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

diag_e0() {   # E0: is the warp the bottleneck? TPSMM vs mesh warp on the visible arm / hand pixels, per motion bin
    python scripts/14_diag_mesh_flow.py --cache $CACHE --flow-checkpoint $OUT/flow/checkpoints/epoch_100.pth \
        --smplx-dir $SMPLX --out $OUT/diag_e0 --split val --max-clips 50 --max-rows 16
}

ablation() {  # base / A1 / B1 / A1+B1 (+2 secondary), short schedule, same seed
    python scripts/11_sweep.py --sweep configs/sweeps/a1_b1_ablation.yaml --flow-checkpoint $OUT/flow/checkpoints/epoch_100.pth \
        --out-root $OUT/ablation_a1b1 --cache $CACHE --common model.flow.smplx_dir=$SMPLX
}

stage fetch bash scripts/vast/fetch_data.sh || exit 1
stage preprocess preprocess || exit 1
stage validrows python scripts/02b_mark_valid_rows.py --cache $CACHE --workers $(nproc) || exit 1
stage splits python scripts/04_make_splits.py --cache $CACHE --seed 0 || exit 1
stage meshbuf meshbuf || exit 1
stage overlay python scripts/03_overlay_check.py --cache $CACHE --pkl-dir $PKL --out $OUT/overlay --num 12 || true
stage flow flow || exit 1
stage diag_e0 diag_e0 || true
stage ablation ablation || exit 1
if [ "$HP_SWEEP" = 1 ]; then stage sweep sweep || exit 1; fi
echo "[$(date -u +%FT%TZ)] PIPELINE_DONE"
