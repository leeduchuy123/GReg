#!/bin/bash
# Download the raw inputs from Google Drive (paths in drive.env) into the layout pipeline.sh expects:
#   /workspace/data/raw/pkls/hamnosys_pkls_cropFalse_shapeTrue/*.pkl
#   /workspace/data/raw/videos/*.mp4
#   /workspace/data/raw/smplx_dl/models/smplx/SMPLX_*.npz   (smplx.create loads the .npz; the .pkl copies are skipped)
# Needs rclone with a Google Drive remote named $GDRIVE_REMOTE (default "gdrive") in ~/.config/rclone/rclone.conf.
# Re-runnable: rclone only copies missing/changed files, and each source is checked against Drive afterwards.
set -euo pipefail
source "$(dirname "$0")/drive.env"
RAW=/workspace/data/raw
LOGS=/workspace/logs
mkdir -p $RAW $LOGS

command -v rclone >/dev/null || { echo "rclone not installed: curl https://rclone.org/install.sh | bash"; exit 1; }
rclone listremotes | grep -qx "${GDRIVE_REMOTE}:" || { echo "no rclone remote '${GDRIVE_REMOTE}:' (copy rclone.conf to ~/.config/rclone/)"; exit 1; }

fetch() {  # fetch <drive path> <local dir> <log name> [filter args...]
    local src=$1 dst=$2 log=$3; shift 3
    echo "[$(date -u +%FT%TZ)] fetch '$src' -> $dst"
    rclone copy "$DRIVE$src" "$dst" "$@" --transfers 16 --checkers 16 --stats 30s --log-level INFO --log-file "$LOGS/$log.log"
    rclone check "$DRIVE$src" "$dst" "$@" --one-way --size-only --log-level ERROR
}

fetch "$DRIVE_SMPLX" $RAW/smplx_dl dl_smplx --include "models/smplx/*.npz"
fetch "$DRIVE_PKLS" $RAW/pkls dl_pkl
fetch "$DRIVE_VIDEOS" $RAW/videos dl_vid

[ -f $RAW/smplx_dl/models/smplx/SMPLX_NEUTRAL.npz ] || { echo "SMPLX_NEUTRAL.npz missing"; exit 1; }
echo "pkls:   $(find $RAW/pkls/hamnosys_pkls_cropFalse_shapeTrue -name '*.pkl' | wc -l)"
echo "videos: $(find $RAW/videos -name '*.mp4' | wc -l)"
echo "smplx:  $(ls $RAW/smplx_dl/models/smplx)"
