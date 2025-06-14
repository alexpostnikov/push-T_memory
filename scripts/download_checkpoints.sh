#!/usr/bin/env bash
#
# download_checkpoints.sh
#
# Helper script to download pretrained Push-T baseline checkpoints (ACT and Diffusion)
# for this project. Downloads files from Hugging Face if not present in ./checkpoints/.
#
# Usage:
#   bash scripts/download_checkpoints.sh
#
# Requirements:
#   - wget (preferred) or curl
#
# The script will:
#   - Create a 'checkpoints/' directory if it does not exist.
#   - Download pusht_act.safetensors (~210MB) and pusht_diffusion.safetensors (~1GB) if missing or empty.
#   - Skip download if file exists and is non-empty.
#

set -e

# Updated checkpoint sources (2024):
#   - ACT policy:     https://huggingface.co/pepijn223/act-pusht/resolve/main/model.safetensors (~210MB)
#   - Diffusion:      https://huggingface.co/lerobot/diffusion_pusht/resolve/main/model.safetensors (~1GB)
ACT_URL="https://huggingface.co/pepijn223/act-pusht/resolve/main/model.safetensors"
DIFF_URL="https://huggingface.co/lerobot/diffusion_pusht/resolve/main/model.safetensors"
CKPT_DIR="checkpoints"
ACT_FILE="$CKPT_DIR/pusht_act.safetensors"
DIFF_FILE="$CKPT_DIR/pusht_diffusion.safetensors"
# for this project. Downloads files from Hugging Face if not present in ./checkpoints/.
#
# Usage:
#   bash scripts/download_checkpoints.sh
#
# Requirements:
#   - wget (preferred) or curl
#
# The script will:
#   - Create a 'checkpoints/' directory if it does not exist.
#   - Download pusht_act.safetensors (~210MB) and pusht_diffusion.safetensors (~1GB) if missing or empty.
#   - Skip download if file exists and is non-empty.
#

set -e

# Updated checkpoint sources (2024):
#   - ACT policy:     https://huggingface.co/pepijn223/act-pusht/resolve/main/model.safetensors (~210MB)
#   - Diffusion:      https://huggingface.co/lerobot/diffusion_pusht/resolve/main/model.safetensors (~1GB)
ACT_URL="https://huggingface.co/pepijn223/act-pusht/resolve/main/model.safetensors"
DIFF_URL="https://huggingface.co/lerobot/diffusion_pusht/resolve/main/model.safetensors"
CKPT_DIR="checkpoints"
ACT_FILE="$CKPT_DIR/pusht_act.safetensors"
DIFF_FILE="$CKPT_DIR/pusht_diffusion.safetensors"

# Check for wget or curl
if command -v wget >/dev/null 2>&1; then
    DL_CMD="wget"
elif command -v curl >/dev/null 2>&1; then
    DL_CMD="curl"
else
    echo "Error: Neither 'wget' nor 'curl' is installed. Please install one to proceed." >&2
    exit 1
fi

mkdir -p "$CKPT_DIR"

# Helper: test if a variable is set and non-empty
is_set() {
    [ "${1+set}" = "set" ] && [ -n "$1" ]
}

download_if_needed() {
    url="$1"
    dest="$2"
    label="$3"

    if [ -s "$dest" ]; then
        echo "✓ $label checkpoint already exists and is non-empty: $dest"
        return 0
    fi

    echo "→ Downloading $label checkpoint to $dest ..."
    HTTP_CODE=""
    if is_set "$HF_TOKEN"; then
        echo "Using Hugging Face token for authentication (HF_TOKEN)."
        if [ "$DL_CMD" = "wget" ]; then
            wget --header="Authorization: Bearer $HF_TOKEN" -O "$dest" "$url" 2>&1 | tee /tmp/dl_log_$
            WGET_STATUS="${PIPESTATUS[0]}"
            HTTP_CODE=$(grep -o "HTTP/[0-9.]* 401" /tmp/dl_log_$ || true)
            rm -f /tmp/dl_log_$
            if [ "$WGET_STATUS" -ne 0 ]; then
                echo "✗ Failed to download $label checkpoint to $dest." >&2
                return 1
            fi
        else
            # curl: -f (fail on >=400), -s (silent), -S (show errors), -L (follow redirects)
            HTTP_STATUS=$(curl -s -w "%{http_code}" -H "Authorization: Bearer $HF_TOKEN" -L -o "$dest" "$url")
            if [ "$HTTP_STATUS" = "401" ]; then
                echo "✗ Unauthorized (401) when downloading $label checkpoint. Check your HF_TOKEN." >&2
                return 1
            elif [ "$HTTP_STATUS" -ge 400 ]; then
                echo "✗ Failed to download $label checkpoint to $dest (HTTP $HTTP_STATUS)." >&2
                return 1
            fi
        fi
    else
        if [ "$DL_CMD" = "wget" ]; then
            wget -O "$dest" "$url" 2>&1 | tee /tmp/dl_log_$
            WGET_STATUS="${PIPESTATUS[0]}"
            HTTP_CODE=$(grep -o "HTTP/[0-9.]* 401" /tmp/dl_log_$ || true)
            rm -f /tmp/dl_log_$
            if [ -n "$HTTP_CODE" ]; then
                echo "✗ Unauthorized (401) when downloading $label checkpoint."
                echo "If this checkpoint is private, provide an access token:"
                echo "    export HF_TOKEN=hf_your_token && bash scripts/download_checkpoints.sh"
                return 1
            elif [ "$WGET_STATUS" -ne 0 ]; then
                echo "✗ Failed to download $label checkpoint to $dest." >&2
                return 1
            fi
        else
            HTTP_STATUS=$(curl -s -w "%{http_code}" -L -o "$dest" "$url")
            if [ "$HTTP_STATUS" = "401" ]; then
                echo "✗ Unauthorized (401) when downloading $label checkpoint."
                echo "If this checkpoint is private, provide an access token:"
                echo "    export HF_TOKEN=hf_your_token && bash scripts/download_checkpoints.sh"
                return 1
            elif [ "$HTTP_STATUS" -ge 400 ]; then
                echo "✗ Failed to download $label checkpoint to $dest (HTTP $HTTP_STATUS)." >&2
                return 1
            fi
        fi
    fi

    if [ -s "$dest" ]; then
        echo "✓ Downloaded $label checkpoint successfully."
        return 0
    else
        echo "✗ Failed to download $label checkpoint to $dest." >&2
        return 1
    fi
}

ALL_OK=0

download_if_needed "$ACT_URL" "$ACT_FILE" "ACT (~210MB)" || ALL_OK=1
download_if_needed "$DIFF_URL" "$DIFF_FILE" "Diffusion (~1GB)" || ALL_OK=1

if [ "$ALL_OK" -eq 0 ]; then
    echo "✅ All requested checkpoints are present in '$CKPT_DIR/'."
    exit 0
else
    echo "⚠️  One or more checkpoints failed to download. Please check your connection and try again."
    exit 2
fi