#!/bin/bash
# One-time setup of a fresh vast.ai instance (image pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime) for pipeline.sh.
#
#   apt-get update && apt-get install -y git && git clone -b <branch> https://github.com/leeduchuy123/GReg.git /workspace/GReg
#   scp rclone.conf root@<host>:/root/.config/rclone/rclone.conf       # Google Drive remote "gdrive" (drive.env)
#   cd /workspace/GReg && FLOW_CKPT_DRIVE=GReg_runs/run5/outputs/flow/checkpoints/epoch_100.pth bash scripts/vast/setup.sh
#
# FLOW_CKPT_DRIVE (optional): a stage-1 checkpoint inside the Drive project folder, copied to /workspace/flow_ckpt/
# (outside /workspace/outputs, so sync_loop.sh does not upload it again); then run pipeline.sh with
# FLOW_CKPT=/workspace/flow_ckpt/epoch_100.pth.  Re-runnable; leaves /workspace/state/setup.done.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p /workspace/logs /workspace/state /workspace/flow_ckpt
source scripts/vast/drive.env

echo "[$(date -u +%FT%TZ)] apt"
apt-get update -qq > /workspace/logs/apt.log 2>&1
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg git tmux curl unzip >> /workspace/logs/apt.log 2>&1

echo "[$(date -u +%FT%TZ)] rclone"
command -v rclone >/dev/null || curl -fsSL https://rclone.org/install.sh | bash > /workspace/logs/rclone_install.log 2>&1
rclone listremotes | grep -qx "${GDRIVE_REMOTE}:" || { echo "no rclone remote '${GDRIVE_REMOTE}:' in ~/.config/rclone/rclone.conf"; exit 1; }

echo "[$(date -u +%FT%TZ)] pip"
pip install -q -r requirements.txt > /workspace/logs/pip.log 2>&1
python -c "import torch, smplx, lpips, scipy; assert torch.cuda.is_available(), 'no CUDA'; print('torch', torch.__version__, torch.cuda.get_device_name(0))"
ffmpeg -version | head -1

if [ -n "${FLOW_CKPT_DRIVE:-}" ]; then
    echo "[$(date -u +%FT%TZ)] stage-1 checkpoint $FLOW_CKPT_DRIVE"
    rclone copyto "$DRIVE$FLOW_CKPT_DRIVE" /workspace/flow_ckpt/epoch_100.pth --log-level ERROR
    python -c "import torch; ck = torch.load('/workspace/flow_ckpt/epoch_100.pth', map_location='cpu', weights_only=False); \
print('flow checkpoint epoch', ck.get('epoch'), 'keys', sorted(ck)[:6])"
fi
touch /workspace/state/setup.done
echo "[$(date -u +%FT%TZ)] setup done"
