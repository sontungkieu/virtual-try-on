#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/workspace/Project_Phase2}"
VENV_DIR="${VENV_DIR:-/workspace/venvs/project_phase2}"
HF_HOME_DIR="${HF_HOME:-/workspace/hf-cache}"

mkdir -p "$PROJECT_DIR" "$HF_HOME_DIR" /workspace/venvs

export HF_HOME="$HF_HOME_DIR"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export DIFFUSERS_CACHE="$HF_HOME/diffusers"

python3 - <<'PY'
import sys
print("python:", sys.version)
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
except Exception as exc:
    print("torch_check_error:", repr(exc))
PY

if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv --system-site-packages "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade \
  "numpy<3" \
  pillow \
  opencv-python-headless \
  tqdm \
  rich \
  requests \
  huggingface_hub \
  safetensors \
  accelerate \
  transformers \
  diffusers \
  sentencepiece \
  protobuf \
  peft \
  bitsandbytes \
  gradio \
  fastapi \
  "uvicorn[standard]" \
  python-multipart

cat > "$PROJECT_DIR/env.sh" <<'EOF'
export HF_HOME=/workspace/hf-cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export DIFFUSERS_CACHE=$HF_HOME/diffusers
source /workspace/venvs/project_phase2/bin/activate
cd /workspace/Project_Phase2
EOF

grep -q "Project_Phase2/env.sh" ~/.bashrc || {
  echo "" >> ~/.bashrc
  echo "# Project_Phase2 environment" >> ~/.bashrc
  echo "[ -f /workspace/Project_Phase2/env.sh ] && source /workspace/Project_Phase2/env.sh" >> ~/.bashrc
}

python - <<'PY'
import importlib
mods = ["torch", "diffusers", "transformers", "accelerate", "gradio", "fastapi"]
for name in mods:
    mod = importlib.import_module(name)
    print(f"{name}: {getattr(mod, '__version__', 'ok')}")
import torch
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

nvidia-smi
echo "Bootstrap complete. Run: source /workspace/Project_Phase2/env.sh"
