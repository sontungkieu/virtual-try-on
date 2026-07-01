#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
PROJECT_ROOT="${VTON_PROJECT_ROOT:-$WORKSPACE/Project_Phase2/virtual_tryon}"
COMFY_ROOT="${COMFY_ROOT:-$WORKSPACE/ComfyUI}"
VENV_PYTHON="${VENV_PYTHON:-$WORKSPACE/venvs/project_phase2/bin/python}"
UV_BIN="${UV_BIN:-uv}"

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "Missing Python venv at $VENV_PYTHON" >&2
  exit 1
fi

if [[ ! -d "$COMFY_ROOT" ]]; then
  git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git "$COMFY_ROOT"
fi

"$UV_BIN" pip install --python "$VENV_PYTHON" -r "$COMFY_ROOT/requirements.txt"

mkdir -p "$COMFY_ROOT/custom_nodes"
rm -rf "$COMFY_ROOT/custom_nodes/vton_phase2_nodes"
ln -s "$PROJECT_ROOT/comfyui_nodes/vton_phase2_nodes" "$COMFY_ROOT/custom_nodes/vton_phase2_nodes"

cat > "$COMFY_ROOT/run_vton_phase2_comfyui.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
export VTON_PROJECT_ROOT="${VTON_PROJECT_ROOT:-/workspace/Project_Phase2/virtual_tryon}"
export VTON_KLEIN_MODEL_DIR="${VTON_KLEIN_MODEL_DIR:-/workspace/Project_Phase2/virtual_tryon/models/flux2-klein-9b}"
export VTON_KLEIN_LORA_PATH="${VTON_KLEIN_LORA_PATH:-/workspace/hf-cache/hub/models--fal--flux-klein-9b-virtual-tryon-lora/snapshots/8b078b15c6d958ce48892b9ef31b66aa7587d792/flux-klein-tryon.safetensors}"
export VTON_COMFY_OUTPUT_ROOT="${VTON_COMFY_OUTPUT_ROOT:-/workspace/Project_Phase2/virtual_tryon/data/outputs/comfyui_runs}"
cd /workspace/ComfyUI
export VIRTUAL_ENV="${VIRTUAL_ENV:-/workspace/venvs/project_phase2}"
export PATH="/root/.local/bin:$VIRTUAL_ENV/bin:$PATH"
exec uv run --active --no-sync --no-project python main.py --listen 0.0.0.0 --port "${COMFY_PORT:-8188}"
SH
chmod +x "$COMFY_ROOT/run_vton_phase2_comfyui.sh"

echo "ComfyUI root: $COMFY_ROOT"
echo "Custom nodes: $COMFY_ROOT/custom_nodes/vton_phase2_nodes"
echo "Launch: $COMFY_ROOT/run_vton_phase2_comfyui.sh"
