#!/bin/bash
# Shared stages of the diffusion branches (docs/diffusion_6_huong_ke_hoach.md Sec. 2) on a vast.ai instance.
# Resume-safe like pipeline.sh: every stage leaves /workspace/state/<stage>.done.
#
#   fetch -> cache (restore from Drive, else preprocess + validrows + splits + meshbuf, then archive to Drive)
#   -> fetch_ckpts (run5 GReg epoch_040 / epoch_100 + stage-1 flow) -> export_run5_{val,test,train}
#   -> eval_baseline (21_eval_frames on the test export: must reproduce run5's Table-1 numbers)
#   -> the branch's own stages (DIFF_STAGES="d1 ..." are functions defined in scripts/vast/stages_<name>.sh)
#
# Layout: /workspace/GReg (code), /workspace/data/{raw,hamnosys_cache}, /workspace/ckpt, /workspace/outputs,
# /workspace/state.  Results are mirrored by sync_loop.sh (DRIVE_DEST_DIR, e.g. GReg_runs/diff_common).
set -uo pipefail
cd /workspace/GReg
source scripts/vast/drive.env
RAW=/workspace/data/raw
PKL=$RAW/pkls/hamnosys_pkls_cropFalse_shapeTrue
VID=$RAW/videos
SMPLX=$RAW/smplx_dl/models
export CACHE=${CACHE:-/workspace/data/hamnosys_cache}
OUT=/workspace/outputs
CK=/workspace/ckpt
ST=/workspace/state
GPU_GB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | awk '{print int($1/1024)}')
SHARDS=${SHARDS:-$(( ${GPU_GB:-24} * 10 / 17 < 14 ? ${GPU_GB:-24} * 10 / 17 : 14 ))}
MESH_SHARDS=${MESH_SHARDS:-4}
WORKERS=${WORKERS:-8}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
RUN5=${RUN5:-GReg_runs/run5/outputs}
EXPORT=${EXPORT:-$OUT/export}
TRAIN_ROWS=${TRAIN_ROWS:-uniform:8}       # rows per train clip exported from epoch_040 (+ their predecessors)
VAL_ROWS=${VAL_ROWS:-uniform:16}
DIFF_STAGES=${DIFF_STAGES:-}
mkdir -p $ST $OUT $CK /workspace/logs

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

meshbuf() {
    local pids=()
    for i in $(seq 0 $((MESH_SHARDS - 1))); do
        python scripts/02c_mesh_buffers.py --cache $CACHE --pkl-dir $PKL --smplx-dir $SMPLX --device cuda --batch 32 \
            --shard $i/$MESH_SHARDS --check > /workspace/logs/meshbuf_shard$i.log 2>&1 &
        pids+=($!)
    done
    local rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
    return $rc
}

cache_stage() {   # restore the archived cache, or build it and archive it
    if bash scripts/vast/cache_archive.sh restore; then
        [ -f $CACHE/splits/test.txt ] || python scripts/04_make_splits.py --cache $CACHE --seed 0 || return 1
        return 0
    fi
    stage preprocess preprocess || return 1
    stage validrows python scripts/02b_mark_valid_rows.py --cache $CACHE --workers $(nproc) || return 1
    stage splits python scripts/04_make_splits.py --cache $CACHE --seed 0 || return 1
    stage meshbuf meshbuf || return 1
    bash scripts/vast/cache_archive.sh archive || echo "WARNING: cache archive to Drive failed (the run goes on)"
}

fetch_ckpts() {
    local rc=0
    for f in greg_final/checkpoints/epoch_040.pth greg_final/checkpoints/epoch_100.pth flow/checkpoints/epoch_100.pth; do
        local dst=$CK/run5_$(echo $f | tr '/' '_')
        [ -f "$dst" ] || rclone copyto "$DRIVE$RUN5/$f" "$dst" --log-level ERROR || rc=1
    done
    ls -la $CK
    python -c "import torch,glob; [print(p, torch.load(p, map_location='cpu', weights_only=False)['epoch']) for p in sorted(glob.glob('$CK/*.pth'))]" || rc=1
    return $rc
}

export_run5() {   # <split> <checkpoint> <rows> [extra args]
    local split=$1 ck=$2 rows=$3; shift 3
    python scripts/20_export_greg.py --checkpoint $ck --cache $CACHE --split $split --rows $rows --with-prev \
        --smplx-dir $SMPLX --out $EXPORT/run5_$(basename $ck .pth | sed 's/run5_greg_final_checkpoints_//')/$split \
        --batch-size 16 --num-workers $WORKERS "$@"
}

eval_baseline() {   # the frame-directory path must give run5's own numbers (Table 1: L1 0.01735, LPIPS 0.04837 on test)
    python scripts/21_eval_frames.py --frames $EXPORT/run5_epoch_100/test --cache $CACHE --split test \
        --out $OUT/eval/run5_epoch_100_test --lpips --smplx-dir $SMPLX --num-workers $WORKERS --all-rows
}

stage fetch bash scripts/vast/fetch_data.sh || exit 1
stage cache cache_stage || exit 1
stage fetch_ckpts fetch_ckpts || exit 1
stage export_run5_val export_run5 val $CK/run5_greg_final_checkpoints_epoch_100.pth $VAL_ROWS || exit 1
stage export_run5_test export_run5 test $CK/run5_greg_final_checkpoints_epoch_100.pth all || exit 1
stage eval_baseline eval_baseline || exit 1
stage export_run5_train export_run5 train $CK/run5_greg_final_checkpoints_epoch_040.pth $TRAIN_ROWS || exit 1
for s in $DIFF_STAGES; do
    source scripts/vast/stages_$s.sh || exit 1          # defines run_$s
    stage $s run_$s || exit 1
done
echo "[$(date -u +%FT%TZ)] PIPELINE_DONE"
