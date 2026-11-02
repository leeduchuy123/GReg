#!/bin/bash
# D6 stages for scripts/vast/pipeline_diff.sh (DIFF_STAGES="d6").  Sourced: uses CACHE, OUT, EXPORT, SMPLX, WORKERS.
# 320^2 x 49 frames on the 24 GB card (docs/code_notes_dwm_wan.md Sec. 7.3: ~6-9 GB at batch 1).
#   weights -> text -> test windows (the same 200 test clips as D5) -> ablation 0 (zero-shot, checks the mask sign and
#   the weight conversion) -> train windows -> ng / n / k / ng_ones -> sample + evaluate on the D5 rows
D6=$OUT/d6
WAN=/workspace/hf/wan_fun_inp_13b
D6_STEPS=${D6_STEPS:-3000}
COMMON6="data.cache_dir=$D6/cache model.checkpoint_dir=$WAN"
D5_INDEX=${D5_INDEX:-$OUT/d5/ng/frames_test/index.json}

d6_sub() {
    local name=d6_$1; shift
    if [ -f "$ST/$name.done" ]; then echo "[skip] $name"; return 0; fi
    echo "[$(date -u +%FT%TZ)] START $name"
    if "$@"; then touch "$ST/$name.done"; echo "[$(date -u +%FT%TZ)] DONE  $name"
    else echo "[$(date -u +%FT%TZ)] FAILED $name"; return 1; fi
}

d6_weights() {
    HF_HOME=/workspace/hf python - <<'PYEOF'
from huggingface_hub import snapshot_download
snapshot_download("alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP", allow_patterns=["diffusion_pytorch_model.safetensors", "config.json"],
                  local_dir="/workspace/hf/wan_fun_inp_13b")
snapshot_download("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", allow_patterns=["vae/*", "tokenizer/*", "text_encoder/*", "model_index.json"])
snapshot_download("Wan-AI/Wan2.1-I2V-14B-480P-Diffusers", allow_patterns=["image_encoder/*", "image_processor/*"])
PYEOF
    python -c "import sys; sys.path.insert(0,'.'); from greg.diffusion.wan_video import load_fun_inp_transformer as L; m=L('$WAN'); print('converted', sum(p.numel() for p in m.parameters())/1e9, 'B params, in', m.config.in_channels)"
}

d6_eval() {  # d6_eval <frames dir> <out>
    local idx=""; [ -f "$D5_INDEX" ] && idx="--index $D5_INDEX"
    python scripts/21_eval_frames.py --frames $1 $idx --cache $CACHE --split test --out $2 --lpips --smplx-dir $SMPLX \
        --num-workers $WORKERS
}

d6_variant() {  # d6_variant <variant>
    local v=$1
    [ -f $D6/$v/checkpoints/final.pt ] || python scripts/33_train_wan.py --config configs/diffusion/d6_wan.yaml \
        --override $COMMON6 model.variant=$v output_dir=$D6/$v train.steps=$D6_STEPS || return 1
    [ -f $D6/$v/frames_test/timing.json ] || python scripts/34_sample_wan.py --config configs/diffusion/d6_wan.yaml \
        --checkpoint $D6/$v/checkpoints/final.pt --windows $D6/cache/test --out $D6/$v/frames_test || return 1
    d6_eval $D6/$v/frames_test $D6/$v/eval_test
}

run_d6() {
    export HF_HOME=/workspace/hf
    d6_sub weights d6_weights || return 1
    d6_sub text python scripts/32_cache_wan_windows.py text --out $D6/cache || return 1
    d6_sub cache_test python scripts/32_cache_wan_windows.py windows --cache $CACHE --split test --max-clips 200 \
        --all-windows --smplx-dir $SMPLX --out $D6/cache/test || return 1
    d6_sub zeroshot bash -c "python scripts/34_sample_wan.py --config configs/diffusion/d6_wan.yaml --variant base --zero-shot \
        --windows $D6/cache/test --out $D6/zeroshot/frames_test --max-windows 60 && \
        python scripts/21_eval_frames.py --frames $D6/zeroshot/frames_test --cache $CACHE --split test \
        --out $D6/zeroshot/eval_test --lpips --smplx-dir $SMPLX --num-workers $WORKERS" || return 1
    d6_sub cache_train python scripts/32_cache_wan_windows.py windows --cache $CACHE --split train_fit \
        --windows-per-clip 1 --smplx-dir $SMPLX --out $D6/cache/train || return 1
    d6_sub smoke python scripts/33_train_wan.py --config configs/diffusion/d6_wan.yaml --override $COMMON6 \
        output_dir=$D6/smoke train.steps=20 train.log_every=5 || return 1
    for v in ng n k ng_ones; do d6_sub $v d6_variant $v || return 1; done
    python scripts/25_collect_results.py --root $D6 --out $D6/results_test.md || true
}
