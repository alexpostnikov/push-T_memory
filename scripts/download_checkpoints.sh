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

# No global DL_CMD; we'll attempt wget and/or curl inside the download logic

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

    DL_SUCCESS=1

    # 1. Try wget if available
    if command -v wget >/dev/null 2>&1; then
        if is_set "$HF_TOKEN"; then
            echo "Using Hugging Face token for authentication (HF_TOKEN)."
            TMP_LOG=$(mktemp)
            wget --header="Authorization: Bearer $HF_TOKEN" -O "$dest" "$url" -q 2>"$TMP_LOG" 1>/dev/null
            WGET_STATUS=$?
            HTTP_CODE=$(grep -o "HTTP/[0-9.]* 401" "$TMP_LOG" || true)
            rm -f "$TMP_LOG"
            if [ -n "$HTTP_CODE" ]; then
                echo "✗ Unauthorized (401) when downloading $label checkpoint."
                echo "If this checkpoint is private, provide an access token:"
                echo "    export HF_TOKEN=hf_your_token && bash scripts/download_checkpoints.sh"
                DL_SUCCESS=1
            elif [ "$WGET_STATUS" -eq 0 ] && [ -s "$dest" ]; then
                echo "✓ Downloaded $label checkpoint successfully."
                return 0
            else
                DL_SUCCESS=1
            fi
        else
            TMP_LOG=$(mktemp)
            wget -O "$dest" "$url" -q 2>"$TMP_LOG" 1>/dev/null
            WGET_STATUS=$?
            HTTP_CODE=$(grep -o "HTTP/[0-9.]* 401" "$TMP_LOG" || true)
            rm -f "$TMP_LOG"
            if [ -n "$HTTP_CODE" ]; then
                echo "✗ Unauthorized (401) when downloading $label checkpoint."
                echo "If this checkpoint is private, provide an access token:"
                echo "    export HF_TOKEN=hf_your_token && bash scripts/download_checkpoints.sh"
                DL_SUCCESS=1
            elif [ "$WGET_STATUS" -eq 0 ] && [ -s "$dest" ]; then
                echo "✓ Downloaded $label checkpoint successfully."
                return 0
            else
                DL_SUCCESS=1
            fi
        fi
    else
        DL_SUCCESS=1
    fi

    # If wget failed or file empty, try curl if available
    if [ ! -s "$dest" ] && command -v curl >/dev/null 2>&1; then
        echo "wget failed, trying curl…"
        if is_set "$HF_TOKEN"; then
            echo "Using Hugging Face token for authentication (HF_TOKEN)."
            HTTP_STATUS=$(curl -s -w "%{http_code}" -H "Authorization: Bearer $HF_TOKEN" -L -o "$dest" "$url")
            if [ "$HTTP_STATUS" = "401" ]; then
                echo "✗ Unauthorized (401) when downloading $label checkpoint. Check your HF_TOKEN." >&2
                DL_SUCCESS=1
            elif [ "$HTTP_STATUS" -ge 400 ]; then
                echo "✗ Failed to download $label checkpoint to $dest (HTTP $HTTP_STATUS)." >&2
                DL_SUCCESS=1
            elif [ -s "$dest" ]; then
                echo "✓ Downloaded $label checkpoint successfully."
                return 0
            else
                DL_SUCCESS=1
            fi
        else
            HTTP_STATUS=$(curl -s -w "%{http_code}" -L -o "$dest" "$url")
            if [ "$HTTP_STATUS" = "401" ]; then
                echo "✗ Unauthorized (401) when downloading $label checkpoint."
                echo "If this checkpoint is private, provide an access token:"
                echo "    export HF_TOKEN=hf_your_token && bash scripts/download_checkpoints.sh"
                DL_SUCCESS=1
            elif [ "$HTTP_STATUS" -ge 400 ]; then
                echo "✗ Failed to download $label checkpoint to $dest (HTTP $HTTP_STATUS)." >&2
                DL_SUCCESS=1
            elif [ -s "$dest" ]; then
                echo "✓ Downloaded $label checkpoint successfully."
                return 0
            else
                DL_SUCCESS=1
            fi
        fi
    fi

    # If curl not available, or both attempts failed
    if [ -s "$dest" ]; then
        echo "✓ Downloaded $label checkpoint successfully."
        return 0
    else
        if ! command -v wget >/dev/null 2>&1 && ! command -v curl >/dev/null 2>&1; then
            echo "Error: Neither 'wget' nor 'curl' is installed. Please install one to proceed." >&2
        else
            echo "✗ Failed to download $label checkpoint to $dest." >&2
        fi
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