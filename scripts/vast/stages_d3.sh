#!/bin/bash
# D3 stages for scripts/vast/pipeline_diff.sh (DIFF_STAGES="d3").  Sourced: uses CACHE, OUT, CK, SMPLX, WORKERS, DRIVE.
D3=$OUT/d3
FLOW=$CK/run5_flow_checkpoints_epoch_100.pth

d3_sub() {
    local name=d3_$1; shift
    if [ -f "$ST/$name.done" ]; then echo "[skip] $name"; return 0; fi
    echo "[$(date -u +%FT%TZ)] START $name"
    if "$@"; then touch "$ST/$name.done"; echo "[$(date -u +%FT%TZ)] DONE  $name"
    else echo "[$(date -u +%FT%TZ)] FAILED $name"; return 1; fi
}

d3_base() {   # 'base' = run5 sweep_clean/ref; reuse the D2 copy when present (already re-evaluated there)
    mkdir -p $D3/sweep/base/checkpoints
    if [ -f $OUT/d2/sweep/base/val_metrics.json ]; then
        cp -n $OUT/d2/sweep/base/checkpoints/last.pth $D3/sweep/base/checkpoints/ 2>/dev/null
        cp -n $OUT/d2/sweep/base/val_metrics.json $D3/sweep/base/ 2>/dev/null; return 0
    fi
    [ -f $D3/sweep/base/checkpoints/last.pth ] || rclone copyto \
        "${DRIVE}GReg_runs/run5/outputs/sweep_clean/ref/checkpoints/last.pth" $D3/sweep/base/checkpoints/last.pth --log-level ERROR
}

d3_sweep() {
    python scripts/11_sweep.py --sweep configs/sweeps/d3_diffusion_loss.yaml --flow-checkpoint $FLOW --out-root $D3/sweep \
        --cache $CACHE --common model.flow.smplx_dir=$SMPLX train.num_workers=$WORKERS
}

run_d3() {
    d3_sub base d3_base || return 1
    d3_sub sweep d3_sweep || return 1
}
