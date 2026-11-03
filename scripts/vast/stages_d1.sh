#!/bin/bash
# D1 stages for scripts/vast/pipeline_diff.sh (DIFF_STAGES="d1").  Sourced: uses CACHE, OUT, EXPORT, SMPLX, WORKERS.
# Every sub-step is resume-safe (its own state file), so a re-run continues where it stopped.
#   run0: training-free SD1.5 + ControlNet-Tile (strength 0.3 / 0.5) on 40 val clips x 4 rows
#   run1: refiner, GReg image only          run2: + normal + masks, no temporal loss
#   run3: + temporal loss (= the full config)
#   eval: every run on val (uniform:16 export), blends soft / hard / freq / none for the full run;
#   test: the best run on the full test export (scores the whole Table-1 protocol)
D1=$OUT/d1
VAL_EXPORT=$EXPORT/run5_epoch_100/val
TEST_EXPORT=$EXPORT/run5_epoch_100/test
COMMON="data.cache=$CACHE data.export_train=$EXPORT/run5_epoch_040/train data.export_val=$VAL_EXPORT data.smplx_dir=$SMPLX train.num_workers=$WORKERS"
D1_STEPS=${D1_STEPS:-5000}
D1_ABL_STEPS=${D1_ABL_STEPS:-3000}

d1_sub() {  # d1_sub <name> <command...>
    local name=d1_$1; shift
    if [ -f "$ST/$name.done" ]; then echo "[skip] $name"; return 0; fi
    echo "[$(date -u +%FT%TZ)] START $name"
    if "$@"; then touch "$ST/$name.done"; echo "[$(date -u +%FT%TZ)] DONE  $name"
    else echo "[$(date -u +%FT%TZ)] FAILED $name"; return 1; fi
}

d1_eval() {  # d1_eval <frames dir> <out dir>
    python scripts/21_eval_frames.py --frames $1 --cache $CACHE --split val --out $2 --lpips --smplx-dir $SMPLX \
        --num-workers $WORKERS
}

d1_run0() {
    for s in 0.3 0.5; do
        local o=$D1/run0_s$s
        [ -f $o/timing.json ] || python scripts/24_refine_training_free.py --export $VAL_EXPORT --cache $CACHE --out $o \
            --strength $s --blends soft none --max-clips 40 --max-items-per-clip 4 || return 1
        for bm in soft none; do d1_eval $o/$bm $o/eval_$bm || return 1; done
    done
    # the GReg baseline on exactly the same 40 x 4 rows
    python scripts/21_eval_frames.py --frames $VAL_EXPORT --index $D1/run0_s0.3/soft/index.json --cache $CACHE \
        --split val --out $D1/run0_baseline_greg --lpips --smplx-dir $SMPLX --num-workers $WORKERS
}

d1_train() {  # d1_train <run> <overrides...>
    local run=$1; shift
    python scripts/22_train_refiner.py --config configs/diffusion/d1_refiner.yaml \
        --override $COMMON output_dir=$D1/$run train.steps=$D1_STEPS "$@"
}

d1_infer_eval() {  # d1_infer_eval <run> <blends...>
    local run=$1; shift
    python scripts/23_refine_frames.py --checkpoint $D1/$run/checkpoints/final.pt --export $VAL_EXPORT --cache $CACHE \
        --out $D1/$run/frames_val --blends "$@" --num-workers $WORKERS || return 1
    for bm in "$@"; do d1_eval $D1/$run/frames_val/$bm $D1/$run/eval_val_$bm || return 1; done
}

run_d1() {
    d1_sub smoke d1_train smoke train.steps=20 train.val_every=20 train.log_every=5 data.val_clips=4 || return 1
    d1_sub baseline_val d1_eval $VAL_EXPORT $D1/baseline_greg_val || return 1
    d1_sub run0 d1_run0 || return 1
    d1_sub run3_train d1_train run3 || return 1
    d1_sub run3_eval d1_infer_eval run3 soft hard freq none || return 1
    d1_sub run1_train d1_train run1 train.steps=$D1_ABL_STEPS model.cond_normal=false model.num_masks=0 loss.weights.temp=0 data.pairs=false || return 1
    d1_sub run1_eval d1_infer_eval run1 soft || return 1
    d1_sub run2_train d1_train run2 train.steps=$D1_ABL_STEPS loss.weights.temp=0 data.pairs=false || return 1
    d1_sub run2_eval d1_infer_eval run2 soft || return 1
    python scripts/25_collect_results.py --root $D1 --baseline $D1/baseline_greg_val --out $D1/results_val.md || true
}
