#!/bin/bash
# D5 stages for scripts/vast/pipeline_diff.sh (DIFF_STAGES="d5").  Sourced: uses CACHE, OUT, EXPORT, SMPLX, WORKERS.
#   train the three condition variants (k / n / ng) with identical data / steps / seed, sample a fixed 200-clip x
#   16-row test subset, and score GReg run5 on exactly the same rows.
D5=$OUT/d5
D5_STEPS=${D5_STEPS:-12000}

d5_sub() {
    local name=d5_$1; shift
    if [ -f "$ST/$name.done" ]; then echo "[skip] $name"; return 0; fi
    echo "[$(date -u +%FT%TZ)] START $name"
    if "$@"; then touch "$ST/$name.done"; echo "[$(date -u +%FT%TZ)] DONE  $name"
    else echo "[$(date -u +%FT%TZ)] FAILED $name"; return 1; fi
}

d5_train() {  # d5_train <variant> [overrides]
    local v=$1; shift
    python scripts/30_train_frame_ldm.py --config configs/diffusion/d5_frame_ldm.yaml --override model.variant=$v \
        data.cache=$CACHE data.smplx_dir=$SMPLX output_dir=$D5/$v train.steps=$D5_STEPS train.num_workers=$WORKERS "$@"
}

d5_sample_eval() {  # d5_sample_eval <variant>
    local v=$1
    python scripts/31_sample_frame_ldm.py --checkpoint $D5/$v/checkpoints/final.pt --cache $CACHE --split test \
        --max-clips 200 --rows 16 --out $D5/$v/frames_test --smplx-dir $SMPLX --num-workers $WORKERS || return 1
    python scripts/21_eval_frames.py --frames $D5/$v/frames_test --cache $CACHE --split test --out $D5/$v/eval_test \
        --lpips --smplx-dir $SMPLX --num-workers $WORKERS
}

d5_greg() {   # GReg run5 on the same rows (its test export holds every row)
    python scripts/21_eval_frames.py --frames $EXPORT/run5_epoch_100/test --index $D5/ng/frames_test/index.json \
        --cache $CACHE --split test --out $D5/greg_run5_eval_test --lpips --smplx-dir $SMPLX --num-workers $WORKERS
}

run_d5() {
    d5_sub smoke d5_train ng train.steps=10 train.preview_every=10 train.log_every=5 output_dir=$D5/smoke || return 1
    for v in ng n k; do
        d5_sub train_$v d5_train $v || return 1
        d5_sub eval_$v d5_sample_eval $v || return 1
    done
    d5_sub greg d5_greg || return 1
    python scripts/25_collect_results.py --root $D5 --baseline $D5/greg_run5_eval_test --out $D5/results_test.md || true
}
