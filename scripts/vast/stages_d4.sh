#!/bin/bash
# D4 stages for scripts/vast/pipeline_diff.sh (DIFF_STAGES="d4").  Sourced: uses CACHE, OUT, CK, SMPLX, WORKERS, DRIVE.
#   backgrounds: 600 SD-1.5 images; 0-499 for training, 500-599 held out for the unseen-background test
#   matte:       person mattes of the train_fit + val clips
#   sweep:       base (run5 ref) / fresh (fixed-pairs fix only) / aug (fix + backgrounds + colour jitter)
#   unseen:      reconstruction on val clips composited onto the held-out backgrounds, every run
D4=$OUT/d4
FLOW=$CK/run5_flow_checkpoints_epoch_100.pth
BG=/workspace/data/backgrounds
BG_TRAIN=/workspace/data/backgrounds_train
BG_HELD=/workspace/data/backgrounds_heldout

d4_sub() {
    local name=d4_$1; shift
    if [ -f "$ST/$name.done" ]; then echo "[skip] $name"; return 0; fi
    echo "[$(date -u +%FT%TZ)] START $name"
    if "$@"; then touch "$ST/$name.done"; echo "[$(date -u +%FT%TZ)] DONE  $name"
    else echo "[$(date -u +%FT%TZ)] FAILED $name"; return 1; fi
}

d4_backgrounds() {
    python scripts/27_make_backgrounds.py --out $BG --num 600 || return 1
    mkdir -p $BG_TRAIN $BG_HELD
    for i in $(seq 0 599); do
        f=$(printf "%04d.png" $i)
        if [ $i -lt 500 ]; then ln -sf $BG/$f $BG_TRAIN/$f; else ln -sf $BG/$f $BG_HELD/$f; fi
    done
}

d4_base() {
    mkdir -p $D4/sweep/base/checkpoints
    for d in $OUT/d2/sweep/base $OUT/d3/sweep/base; do
        if [ -f $d/val_metrics.json ]; then cp -n $d/checkpoints/last.pth $D4/sweep/base/checkpoints/; cp -n $d/val_metrics.json $D4/sweep/base/; return 0; fi
    done
    [ -f $D4/sweep/base/checkpoints/last.pth ] || rclone copyto \
        "${DRIVE}GReg_runs/run5/outputs/sweep_clean/ref/checkpoints/last.pth" $D4/sweep/base/checkpoints/last.pth --log-level ERROR
}

d4_sweep() {
    python scripts/11_sweep.py --sweep configs/sweeps/d4_augment.yaml --flow-checkpoint $FLOW --out-root $D4/sweep \
        --cache $CACHE --common model.flow.smplx_dir=$SMPLX train.num_workers=$WORKERS data.augment.background_dir=$BG_TRAIN
}

d4_unseen() {
    for run in base fresh aug; do
        [ -f $D4/unseen/$run/summary.json ] && continue
        python scripts/29_eval_unseen_backgrounds.py --checkpoint $D4/sweep/$run/checkpoints/last.pth --cache $CACHE \
            --bg-dir $BG_HELD --out $D4/unseen/$run --lpips --num-workers $WORKERS || return 1
    done
}

run_d4() {
    d4_sub backgrounds d4_backgrounds || return 1
    d4_sub matte python scripts/28_matte_cache.py --cache $CACHE --keys-file $CACHE/splits/train_fit.txt \
        $CACHE/splits/val.txt --check || return 1
    d4_sub base d4_base || return 1
    d4_sub sweep d4_sweep || return 1
    d4_sub unseen d4_unseen || return 1
}
