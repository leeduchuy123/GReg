#!/bin/bash
# D2 stages for scripts/vast/pipeline_diff.sh (DIFF_STAGES="d2").  Sourced: uses CACHE, OUT, CK, SMPLX, WORKERS, DRIVE.
D2=$OUT/d2
FLOW=$CK/run5_flow_checkpoints_epoch_100.pth
D2_STEPS=${D2_STEPS:-30000}

d2_sub() {
    local name=d2_$1; shift
    if [ -f "$ST/$name.done" ]; then echo "[skip] $name"; return 0; fi
    echo "[$(date -u +%FT%TZ)] START $name"
    if "$@"; then touch "$ST/$name.done"; echo "[$(date -u +%FT%TZ)] DONE  $name"
    else echo "[$(date -u +%FT%TZ)] FAILED $name"; return 1; fi
}

d2_re() {
    python scripts/26_train_region_diffusion.py --cache $CACHE --out $D2/re --steps $D2_STEPS \
        --regression-checkpoint $CK/run5_greg_final_checkpoints_epoch_100.pth --num-workers $WORKERS
}

d2_base() {   # run5 sweep_clean/ref = 'base' of the sweep (same config, data, split, seed); 11_sweep re-evaluates it
    mkdir -p $D2/sweep/base/checkpoints
    [ -f $D2/sweep/base/checkpoints/last.pth ] || \
        rclone copyto "${DRIVE}GReg_runs/run5/outputs/sweep_clean/ref/checkpoints/last.pth" $D2/sweep/base/checkpoints/last.pth \
        --log-level ERROR
}

d2_sweep() {
    python scripts/11_sweep.py --sweep configs/sweeps/d2_diffusion_re.yaml --flow-checkpoint $FLOW --out-root $D2/sweep \
        --cache $CACHE --common model.flow.smplx_dir=$SMPLX train.num_workers=$WORKERS model.rpg.checkpoint=$D2/re/last.pth
}

run_d2() {
    d2_sub re d2_re || return 1
    d2_sub base d2_base || return 1
    d2_sub sweep d2_sweep || return 1
}
