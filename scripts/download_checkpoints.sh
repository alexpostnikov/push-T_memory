#!/usr/bin/env bash
#
# download_checkpoints.sh
#
# Helper script to download pretrained Push-T baseline checkpoints (ACT and Diffusion)
# for this project. Downloads files from HuggingFace if not present in ./checkpoints/.
#
# Usage:
#   bash scripts/download_checkpoints.sh
#
# Requirements:
#   - wget (preferred) or curl
#
# The script will:
#   - Create a 'checkpoints/' directory if it does not exist.
#   - Download pusht_act.ckpt (~80MB) and pusht_diffusion.ckpt (~120MB) if missing or empty.
#   - Skip download if file exists and is non-empty.
#

set -e

ACT_URL="https://huggingface.co/lerobot/pusht-act-checkpoint/resolve/main/pusht_act.ckpt"
DIFF_URL="https://huggingface.co/lerobot/pusht-diffusion-checkpoint/resolve/main/pusht_diffusion.ckpt"
CKPT_DIR="checkpoints"
ACT_FILE="$CKPT_DIR/pusht_act.ckpt"
DIFF_FILE="$CKPT_DIR/pusht_diffusion.ckpt"

# Check for wget or curl
if command -v wget &>/dev/null; then
    DL_CMD="wget"
elif command -v curl &>/dev/null; then
    DL_CMD="curl"
else
    echo "Error: Neither 'wget' nor 'curl' is installed. Please install one to proceed." >&2
    exit 1
fi

mkdir -p "$CKPT_DIR"

download_if_needed() {
    local url="$1"
    local dest="$2"
    local label="$3"

    if [[ -s "$dest" ]]; then
        echo "✓ $label checkpoint already exists and is non-empty: $dest"
        return 0
    fi

    echo "→ Downloading $label checkpoint to $dest ..."
    if [[ "$DL_CMD" == "wget" ]]; then
        wget -O "$dest" "$url"
    else
        curl -L -o "$dest" "$url"
    fi

    if [[ -s "$dest" ]]; then
        echo "✓ Downloaded $label checkpoint successfully."
        return 0
    else
        echo "✗ Failed to download $label checkpoint to $dest." >&2
        return 1
    fi
}

ALL_OK=0

download_if_needed "$ACT_URL" "$ACT_FILE" "ACT" || ALL_OK=1
download_if_needed "$DIFF_URL" "$DIFF_FILE" "Diffusion" || ALL_OK=1

if [[ $ALL_OK -eq 0 ]]; then
    echo "✅ All requested checkpoints are present in '$CKPT_DIR/'."
    exit 0
else
    echo "⚠️  One or more checkpoints failed to download. Please check your connection and try again."
    exit 2
fi