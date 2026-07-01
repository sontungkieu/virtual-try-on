#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p \
  "$ROOT_DIR/models/idm_vton" \
  "$ROOT_DIR/models/flux2" \
  "$ROOT_DIR/models/catvton" \
  "$ROOT_DIR/models/loras" \
  "$ROOT_DIR/third_party"

cat <<'MSG'
Model directories are ready.

Place checkpoints using this convention:
  models/idm_vton/                 IDM-VTON weights and config files
  models/flux2/                    Optional local FLUX.2 cache/checkpoints
  models/catvton/                  CatVTON baseline weights
  models/loras/flux-klein-tryon.safetensors

Then update configs/models.yaml if your repo path or entrypoint differs.
This script intentionally does not auto-download gated models or private weights.
MSG
