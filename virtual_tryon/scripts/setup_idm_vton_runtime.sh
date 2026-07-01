#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

sync_args=(--locked --extra idm --no-dev --no-install-project)
run_args=(--no-sync --extra idm)
pip_args=()

if [[ "${TRYON_IDM_SYNC_TENSORRT:-false}" == "true" ]]; then
  sync_args+=(--extra tensorrt)
  run_args+=(--extra tensorrt)
fi

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  sync_args+=(--active --inexact)
  run_args+=(--active)
  pip_args+=(--python "$VIRTUAL_ENV/bin/python")
fi

uv sync "${sync_args[@]}"

# IDM-VTON's reference environment does not require PEFT. Newer PEFT releases
# import APIs missing from accelerate==0.25.0 and can break diffusers loading.
uv pip uninstall "${pip_args[@]}" -y peft >/dev/null 2>&1 || true

uv run "${run_args[@]}" python - <<'PY'
import accelerate
import diffusers
import einops
import torch
import torchvision
import transformers

print("IDM-VTON runtime ready")
print(f"torch={torch.__version__}")
print(f"torchvision={torchvision.__version__}")
print(f"diffusers={diffusers.__version__}")
print(f"transformers={transformers.__version__}")
print(f"accelerate={accelerate.__version__}")
print(f"einops={einops.__version__}")
PY
