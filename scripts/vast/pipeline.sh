#!/bin/bash
# Overnight pipeline on a vast.ai instance (resume-safe: every stage leaves /workspace/state/<stage>.done).
#   fetch -> preprocess (sharded) -> fade trimming -> splits -> overlay check -> stage 1 (flow) -> hyper-parameter sweep (stage 2, short)
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

stage fetch bash scripts/vast/fetch_data.sh || exit 1
stage preprocess preprocess || exit 1
stage validrows python scripts/02b_mark_valid_rows.py --cache $CACHE --workers $(nproc) || exit 1
stage splits python scripts/04_make_splits.py --cache $CACHE --seed 0 || exit 1
stage overlay python scripts/03_overlay_check.py --cache $CACHE --pkl-dir $PKL --out $OUT/overlay --num 12 || true
stage flow flow || exit 1
stage sweep sweep || exit 1
echo "[$(date -u +%FT%TZ)] PIPELINE_DONE"
