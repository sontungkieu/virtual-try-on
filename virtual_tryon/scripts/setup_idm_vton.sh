#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$ROOT_DIR/third_party/IDM-VTON"
CKPT_DIR="$ROOT_DIR/models/idm_vton/ckpt"
IDM_REPO_URL="${IDM_REPO_URL:-https://github.com/yisol/IDM-VTON.git}"

if [ ! -d "$REPO_DIR/.git" ]; then
  mkdir -p "$(dirname "$REPO_DIR")"
  git clone --depth 1 "$IDM_REPO_URL" "$REPO_DIR"
else
  git -C "$REPO_DIR" fetch --depth 1 origin
fi

mkdir -p \
  "$CKPT_DIR/densepose" \
  "$CKPT_DIR/humanparsing" \
  "$CKPT_DIR/openpose/ckpts"

required_files=(
  "densepose/model_final_162be9.pkl"
  "humanparsing/parsing_atr.onnx"
  "humanparsing/parsing_lip.onnx"
  "openpose/ckpts/body_pose_model.pth"
)

echo "IDM-VTON repo: $REPO_DIR"
echo "Checkpoint dir: $CKPT_DIR"
echo

missing=0
for rel in "${required_files[@]}"; do
  path="$CKPT_DIR/$rel"
  if [ ! -s "$path" ]; then
    echo "MISSING $rel"
    missing=1
  else
    echo "OK      $rel"
  fi
done

cat <<'MSG'

Place preprocessing checkpoints under models/idm_vton/ckpt:
  densepose/model_final_162be9.pkl
  humanparsing/parsing_atr.onnx
  humanparsing/parsing_lip.onnx
  openpose/ckpts/body_pose_model.pth

The IDM-VTON diffusion weights are loaded from configs/models.yaml idm_vton.model_name,
which defaults to yisol/IDM-VTON.
MSG

exit "$missing"
