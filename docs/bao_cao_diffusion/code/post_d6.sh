#!/bin/bash
# Post-D6 (user-approved 2026-11-05): runs when tmux 'chain' (D6) ends.
#   A  D6 hands composited into GReg run5 / D1 frames (pose-derived boxes), every finished D6 variant, 200-clip rows
#   B' D1 on the same 200-clip rows (free: subset of its test u16 scores) -> inside C
#   C  paired clip bootstrap CIs of every comparison
set -u
export HF_HOME=/workspace/hf
CACHE=/workspace/data/hamnosys_cache; SMPLX=/workspace/data/raw/smplx_dl/models; O=/workspace/outputs
P=$O/post; mkdir -p $P/compose; LOG=/workspace/logs/chain.log
IDX=$O/d5/ng/frames_test/index.json
while tmux has-session -t chain 2>/dev/null; do sleep 60; done
echo "[$(date -u +%FT%TZ)] === post_d6 (A compose, C bootstrap)" >> $LOG
cd /workspace
fail=0
for v in ng n k ng_ones; do
    [ -f $O/d6/$v/frames_test/timing.json ] || { echo "[post] skip $v (no frames)"; continue; }
    for b in greg:$O/export/run5_epoch_100/test d1:$O/final/refined_run5/freq; do
        n=${b%%:*} d=${b#*:}
        [ -f $P/compose/${v}_on_$n/summary.json ] && continue
        python compose_hands.py --base $d --hands $O/d6/$v/frames_test --index $IDX --cache $CACHE --smplx-dir $SMPLX \
            --out $P/compose/${v}_on_$n --num-workers 12 || fail=1
    done
done
G200=$O/d5/greg_run5_eval_test/per_frame.csv
U=$O/final/eval_test_u16
pairs=(
  "fulltest_run5_vs_d4aug100=$O/eval/run5_epoch_100_test/per_frame.csv:$O/final/d4_aug_100/eval_test/per_frame.csv"
  "u16_greg_vs_d1=$U/greg_run5/per_frame.csv:$U/d1_run5/per_frame.csv"
  "u16_greg_vs_d4aug=$U/greg_run5/per_frame.csv:$U/greg_d4aug/per_frame.csv"
  "u16_greg_vs_d1_on_d4aug=$U/greg_run5/per_frame.csv:$U/d1_d4aug/per_frame.csv"
  "s200_greg_vs_d1=$G200:$U/d1_run5/per_frame.csv"
  "s200_greg_vs_d5ng=$G200:$O/d5/ng/eval_test/per_frame.csv"
)
for v in ng n k ng_ones; do
    [ -f $O/d6/$v/eval_test/per_frame.csv ] && pairs+=("s200_greg_vs_d6_$v=$G200:$O/d6/$v/eval_test/per_frame.csv")
    for n in greg d1; do
        [ -f $P/compose/${v}_on_$n/per_frame.csv ] && pairs+=("s200_greg_vs_d6${v}_hands_on_$n=$G200:$P/compose/${v}_on_$n/per_frame.csv")
    done
done
[ -f $P/compose/ng_on_d1/per_frame.csv ] && pairs+=("s200_d1_vs_d6ng_hands_on_d1=$U/d1_run5/per_frame.csv:$P/compose/ng_on_d1/per_frame.csv")
python bootstrap_ci.py --out $P/bootstrap.md "${pairs[@]}" || fail=1
if [ $fail -eq 0 ]; then echo "[$(date -u +%FT%TZ)] post_d6 done" >> $LOG; else echo "[$(date -u +%FT%TZ)] post_d6 FAILED" >> $LOG; fi
