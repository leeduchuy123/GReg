#!/bin/bash
# Mirror /workspace/outputs (+ logs, state) to Google Drive every 10 minutes: <project folder>/GReg_runs/<RUN_NAME>/
# (see drive.env). Only new/changed files are copied; the optimizer-carrying checkpoints are included on purpose
# (they are what lets a dead instance be resumed). rclone errors go to /workspace/logs/sync.log, and
# last_sync.txt is only rewritten after a fully successful round.
source "$(dirname "$0")/drive.env"
RUN=${RUN_NAME:-run1}
DEST="$DRIVE$DRIVE_RUNS/$RUN"
LOG=/workspace/logs/sync.log
while true; do
    ok=1
    rclone copy /workspace/outputs "$DEST/outputs" --update --transfers 8 --exclude "smoke*/**" --log-level ERROR --log-file $LOG || ok=0
    rclone copy /workspace/logs "$DEST/logs" --update --log-level ERROR --log-file $LOG || ok=0
    rclone copy /workspace/state "$DEST/state" --update --log-level ERROR --log-file $LOG || ok=0
    if [ $ok = 1 ]; then
        date -u +%FT%TZ > /workspace/state/last_sync.txt
    else
        echo "[$(date -u +%FT%TZ)] sync to $DEST FAILED" >> $LOG
    fi
    sleep ${SYNC_EVERY:-600}
done
