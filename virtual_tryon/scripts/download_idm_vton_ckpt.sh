#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_DIR="$ROOT_DIR/models/idm_vton/ckpt"
TMP_DIR="${TMPDIR:-/tmp}/idm_vton_ckpt_download_$$"

required_files=(
  "densepose/model_final_162be9.pkl"
  "humanparsing/parsing_atr.onnx"
  "humanparsing/parsing_lip.onnx"
  "openpose/ckpts/body_pose_model.pth"
)

cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

mkdir -p "$TMP_DIR" "$CKPT_DIR"

echo "Downloading IDM-VTON ckpt files from Hugging Face space yisol/IDM-VTON"
echo "Target checkpoint dir: $CKPT_DIR"

if command -v hf >/dev/null 2>&1; then
  hf download yisol/IDM-VTON \
    --type space \
    --include "ckpt/**" \
    --local-dir "$TMP_DIR"
elif command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download yisol/IDM-VTON \
    --repo-type space \
    --include "ckpt/**" \
    --local-dir "$TMP_DIR"
else
  echo "ERROR: neither huggingface-cli nor hf is installed." >&2
  echo "Install project runtime tools with: uv sync --extra idm" >&2
  exit 2
fi

if [ ! -d "$TMP_DIR/ckpt" ]; then
  echo "ERROR: download finished but ckpt/ was not found in temp folder: $TMP_DIR" >&2
  find "$TMP_DIR" -maxdepth 3 -type f | sort >&2 || true
  exit 3
fi

mkdir -p "$CKPT_DIR"
cp -a "$TMP_DIR/ckpt/." "$CKPT_DIR/"

missing=0
echo
echo "Verifying required IDM-VTON preprocessing checkpoints:"
for rel in "${required_files[@]}"; do
  path="$CKPT_DIR/$rel"
  if [ ! -s "$path" ]; then
    echo "MISSING $rel"
    missing=1
  else
    size="$(wc -c < "$path" | tr -d ' ')"
    echo "OK      $rel ($size bytes)"
  fi
done

if [ "$missing" -ne 0 ]; then
  echo
  echo "ERROR: IDM-VTON checkpoint download is incomplete." >&2
  echo "Expected files under: $CKPT_DIR" >&2
  exit 4
fi

echo
echo "IDM-VTON checkpoint verification complete."
