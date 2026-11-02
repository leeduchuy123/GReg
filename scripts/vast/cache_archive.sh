#!/bin/bash
# Archive the preprocessed cache to Google Drive as a few big tar files, or restore it from there.
# Re-preprocessing costs 3-4 GPU hours per fresh instance; streaming tars through rclone needs no local
# scratch space and a restore takes ~20-30 minutes.
#
#   bash scripts/vast/cache_archive.sh archive   # <project folder>/GReg_cache/<CACHE_TAG>/{meta,frames_i,...}.tar
#   bash scripts/vast/cache_archive.sh restore   # into $CACHE (skips when the cache already has meta.json)
#   bash scripts/vast/cache_archive.sh exists    # exit 0 when the archive is complete on Drive
#
# CACHE_TAG distinguishes caches (hamnosys_256_v1 = 256^2, fade-trimmed, mesh buffers); DONE marker = _complete.
set -uo pipefail
source "$(dirname "$0")/drive.env"
CACHE=${CACHE:-/workspace/data/hamnosys_cache}
CACHE_TAG=${CACHE_TAG:-hamnosys_256_v1}
DEST="${DRIVE}GReg_cache/${CACHE_TAG}"
PARTS=${CACHE_PARTS:-8}
SUBDIRS="frames normals parts faces"

archive() {
    cd "$CACHE" || exit 1
    [ -f meta.json ] || { echo "no meta.json in $CACHE"; exit 1; }
    if rclone lsf "$DEST/_complete" >/dev/null 2>&1; then echo "archive already complete at $DEST"; return 0; fi
    echo "[$(date -u +%FT%TZ)] archiving $CACHE -> $DEST"
    tar -cf - meta.json splits joints mesh 2>/dev/null | rclone rcat "$DEST/meta.tar" --log-level ERROR || return 1
    for sub in $SUBDIRS; do
        [ -d "$sub" ] || continue
        ls "$sub" | sort > /tmp/_keys_$sub.txt
        local n; n=$(wc -l < /tmp/_keys_$sub.txt)
        for i in $(seq 0 $((PARTS - 1))); do
            if rclone lsf "$DEST/${sub}_$i.tar" >/dev/null 2>&1; then echo "[skip] ${sub}_$i.tar"; continue; fi
            awk -v i=$i -v p=$PARTS 'NR % p == i' /tmp/_keys_$sub.txt | sed "s#^#$sub/#" > /tmp/_list_${sub}_$i.txt
            echo "[$(date -u +%FT%TZ)] ${sub}_$i.tar ($(wc -l < /tmp/_list_${sub}_$i.txt) of $n clips)"
            tar -cf - -T /tmp/_list_${sub}_$i.txt | rclone rcat "$DEST/${sub}_$i.tar" --log-level ERROR || return 1
        done
    done
    echo "$(date -u +%FT%TZ) $(du -sh "$CACHE" | cut -f1)" | rclone rcat "$DEST/_complete" --log-level ERROR
    echo "[$(date -u +%FT%TZ)] archive done"
}

restore() {
    if [ -f "$CACHE/meta.json" ] && [ -d "$CACHE/mesh" ]; then echo "cache present at $CACHE, nothing to restore"; return 0; fi
    rclone lsf "$DEST/_complete" >/dev/null 2>&1 || { echo "no complete archive at $DEST"; return 1; }
    mkdir -p "$CACHE" && cd "$CACHE" || exit 1
    echo "[$(date -u +%FT%TZ)] restoring $DEST -> $CACHE"
    for f in $(rclone lsf "$DEST" | grep '\.tar$' | sort); do
        echo "[$(date -u +%FT%TZ)] $f"
        rclone cat "$DEST/$f" --log-level ERROR | tar -xf - || return 1
    done
    [ -f meta.json ] || { echo "restore produced no meta.json"; return 1; }
    echo "[$(date -u +%FT%TZ)] restore done: $(ls frames | wc -l) clips"
}

case "${1:-}" in
    archive) archive ;;
    restore) restore ;;
    exists) rclone lsf "$DEST/_complete" >/dev/null 2>&1 ;;
    *) echo "usage: $0 archive|restore|exists"; exit 2 ;;
esac
