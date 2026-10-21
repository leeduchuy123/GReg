#!/bin/bash
# Mirror /workspace/outputs (+ logs, state) to Google Drive every 10 minutes: GReg_runs/<RUN_NAME>/.
# Only new/changed files are copied; the optimizer-carrying checkpoints are included on purpose
# (they are what lets a dead instance be resumed).
RUN=${RUN_NAME:-run1}
while true; do
    rclone copy /workspace/outputs "gdrive:GReg_runs/$RUN/outputs" --update --transfers 8 --exclude "smoke*/**" -q 2>/dev/null
    rclone copy /workspace/logs "gdrive:GReg_runs/$RUN/logs" --update -q 2>/dev/null
    rclone copy /workspace/state "gdrive:GReg_runs/$RUN/state" --update -q 2>/dev/null
    date -u +%FT%TZ > /workspace/state/last_sync.txt
    sleep ${SYNC_EVERY:-600}
done
