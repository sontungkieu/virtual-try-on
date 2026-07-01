# Model Setup

## Directory Convention

```text
virtual_tryon/models/
  idm_vton/
    ckpt/
      densepose/
        model_final_162be9.pkl
      humanparsing/
        parsing_atr.onnx
        parsing_lip.onnx
      openpose/
        ckpts/
          body_pose_model.pth
  flux2/
  catvton/
  loras/
    flux-klein-tryon.safetensors

virtual_tryon/third_party/
  IDM-VTON/
  CatVTON/
```

Run:

```bash
cd virtual_tryon
uv sync
bash scripts/setup_idm_vton.sh
bash scripts/setup_idm_vton_runtime.sh
bash scripts/download_idm_vton_ckpt.sh
```

Runtime Python dependencies are declared in `pyproject.toml` and locked by `uv.lock`. Use `uv add` for new backend/script dependencies. Real IDM-VTON runtime dependencies live in the optional `idm` extra, while TensorRT packages live in the optional `tensorrt` extra:

```bash
cd virtual_tryon
uv sync --extra idm
uv sync --extra idm --extra tensorrt
```

`scripts/setup_idm_vton_runtime.sh` runs the locked `idm` sync and verifies the import set. On an existing RunPod venv that already contains ComfyUI or IDM runtime packages, use an inexact active sync so UV installs locked extras without removing shared third-party packages:

```bash
cd virtual_tryon
VIRTUAL_ENV=/workspace/venvs/project_phase2 \
PATH=/root/.local/bin:/workspace/venvs/project_phase2/bin:$PATH \
uv sync --locked --extra idm --extra tensorrt --inexact --active --no-install-project
```

The `torch-tensorrt==2.12.1` metadata is overridden in `pyproject.toml` to avoid pulling `executorch`, which currently forces `numpy>=2`; this project pins `numpy<2` for backend/runtime compatibility.

`setup_idm_vton.sh` clones the official IDM-VTON implementation into `third_party/IDM-VTON` and creates checkpoint folders.

`setup_idm_vton_runtime.sh` syncs the IDM-VTON-compatible runtime pins from the `idm` extra. It also removes `peft` from the target venv through `uv pip uninstall` because newer PEFT releases can conflict with `accelerate==0.25.0` during IDM-VTON pipeline loading.

`download_idm_vton_ckpt.sh` downloads `ckpt/**` from the Hugging Face space `yisol/IDM-VTON`, copies it into `models/idm_vton/ckpt`, and verifies the four required preprocessing checkpoint files. It uses a temporary folder and does not write tokens to the repo.

## Token And Model Safety

- Never commit Hugging Face tokens, `.env` files, shell history, checkpoints, model weights, generated outputs, or third-party source mirrors.
- Never expose model or checkpoint folders through HTTP; only `data/outputs` is served at `/artifacts`.
- Keep `HF_TOKEN` in the shell environment only for the current session, or use the Hugging Face CLI login cache outside the repository.
- If a token is pasted into a chat, terminal transcript, issue, or any external system, revoke or rotate it in Hugging Face settings before continuing production work.
- `.gitignore` must keep `virtual_tryon/models/`, `virtual_tryon/data/outputs/`, `virtual_tryon/data/temp/`, and `virtual_tryon/third_party/` out of Git.
- Before commit, run a quick safety check such as `git status --short` and `rg -n "hf_[A-Za-z0-9_]+" .`.

## Config

Edit:

```text
configs/models.yaml
```

Important fields:

- `idm_vton.checkpoint_dir`
- `idm_vton.repo_path`
- `idm_vton.entrypoint`
- `idm_vton.model_name`
- `flux_refiner.backend`
- `flux_refiner.model_name`
- `flux_refiner.model_path`
- `flux_refiner.checkpoint_dir`
- `flux_refiner.api_url_env`
- `flux_refiner.api_key_env`
- `klein_tryon_lora.backend`
- `klein_tryon_lora.fal_endpoint`
- `klein_tryon_lora.lora_repo`
- `klein_tryon_lora.lora_weight_api`
- `klein_tryon_lora.lora_path`

`flux_refiner.backend` supports:

- `disabled`
- `flux2_dev`
- `flux2_klein`
- `flux2_api`
- `fal_tryon_lora`

For API backends, set credentials through environment variables only:

```bash
export FLUX_REFINER_API_URL="https://example.invalid/refine"
export FLUX_REFINER_API_KEY="..."
```

Do not store API keys or Hugging Face tokens in the repo. If a token was pasted into an external system, revoke or rotate it.

## Check IDM-VTON

```bash
cd virtual_tryon
uv run python scripts/run_idm_vton.py --check
```

## Real CLI Run

```bash
cd virtual_tryon
uv run python scripts/run_tryon.py \
  --person data/examples/person_001.jpg \
  --garment data/examples/top_001.jpg \
  --category upper_body \
  --prompt "replace the shirt with the reference garment, preserve face, pose, and body shape" \
  --output data/outputs/real_idm_test.png
```

Do not pass `--mock` for the real engine. The adapter creates a one-sample VITON-HD-style dataset under:

```text
data/outputs/{job_id}/idm_vton_dataset/
```

Debug files include:

```text
core_output.png
idm_vton_command.txt
idm_vton_stdout.txt
idm_vton_stderr.txt
idm_vton_worker_request.json
idm_vton_worker_response.json
idm_vton_resident_error.txt
mask_preview.png
mask_metadata.json
mask_innerwear_shape.png
mask_body_silhouette.png
agnostic.png
cloth_mask.png
densepose.png or densepose_placeholder.png
```

`idm_vton_worker_request.json`, `idm_vton_worker_response.json`, and `idm_vton_resident_error.txt` are present only when `idm_vton.resident_worker=true` or when the worker path is attempted.
`mask_innerwear_shape.png` and `mask_body_silhouette.png` are written for adult innerwear categories; the silhouette file is present only when foreground body estimation is available.

## Resident IDM-VTON Worker

`configs/models.yaml` enables `idm_vton.resident_worker` for deployed IDM-VTON runs. The backend starts `scripts/idm_vton_resident_worker.py` on the first real IDM job, the worker loads `yisol/IDM-VTON` once, then later jobs reuse that in-memory pipeline instead of launching `accelerate` and reloading the model each time.

Useful toggles:

```yaml
idm_vton:
  resident_worker: true
  resident_worker_fallback: true
  resident_worker_entrypoint: "./scripts/idm_vton_resident_worker.py"
  resident_worker_optimization: "eager"
```

Set `TRYON_IDM_RESIDENT_WORKER=false` before starting the backend to disable the worker without editing YAML. Set `TRYON_IDM_WORKER_OPTIMIZATION=torch_compile` for an experimental compiled worker. With fallback enabled, a worker startup/runtime failure writes `idm_vton_resident_error.txt` and then runs the previous subprocess adapter.

## Baseline Suite

After IDM-VTON is available, preserve a fixed baseline before adding refiners:

```bash
cd virtual_tryon
uv run python scripts/run_idm_baseline_suite.py
```

The script discovers paired `data/examples/person_*` and `data/examples/top_*` files, runs IDM-VTON without FLUX or repair, and writes:

```text
data/outputs/baseline_suite/{sample_id}/
  input_person.png
  input_garment.png
  core_output.png
  mask_preview.png
  idm_vton_command.txt
  idm_vton_stdout.txt
  idm_vton_stderr.txt
  run_metadata.json
data/outputs/baseline_suite/baseline_summary.json
```

If only one or two paired samples exist, the script still runs and prints guidance to add more examples.

## Real API Test

```bash
curl -X POST http://127.0.0.1:8000/tryon \
  -F "person_image=@data/examples/person_001.jpg" \
  -F "garment_top=@data/examples/top_001.jpg" \
  -F "category=upper_body" \
  -F "prompt=replace the shirt with the reference garment, preserve face, pose, body shape" \
  -F "use_refiner=false" \
  -F "repair_mode=false"
```

## Common Errors

`IDM-VTON is not available. missing checkpoint: densepose/model_final_162be9.pkl; ...`

The checkpoint folder is missing required files. Place preprocessing checkpoints under `models/idm_vton/ckpt` or update `configs/models.yaml`.

`entrypoint not found: .../third_party/IDM-VTON/inference.py`

Run `bash scripts/setup_idm_vton.sh` or update `idm_vton.entrypoint`.

`checkpoint looks incomplete: ...`

The file exists but is probably a Git LFS pointer or placeholder. Replace it with the real checkpoint file.

`CatVTON checkpoint not found at ...`

CatVTON is a baseline engine. Keep it disabled until weights and an entrypoint are configured.

## CatVTON Baseline

CatVTON is benchmark-only by default:

```yaml
catvton:
  enabled: false
  repo_path: "./third_party/CatVTON"
  checkpoint_dir: "./models/catvton"
  entrypoint: null
```

Clone or install CatVTON under `third_party/CatVTON`, place weights under `models/catvton`, and set `entrypoint` before enabling it. Both `third_party/` and `models/` are ignored by Git.

## Klein Try-On LoRA Baseline

Klein Try-On LoRA is also benchmark-only by default:

```yaml
klein_tryon_lora:
  enabled: false
  backend: "diffusers_local"
  base_model: "black-forest-labs/FLUX.2-klein-9B"
  model_path: "./models/flux2-klein-9b"
  lora_repo: "fal/flux-klein-9b-virtual-tryon-lora"
  lora_weight_api: "flux-klein-tryon.safetensors"
  fal_endpoint: "fal-ai/flux-2/klein/9b/base/edit/lora"
  lora_path: "./models/loras/flux-klein-tryon.safetensors"
```

`enabled: false` keeps IDM-VTON as the production default. Selecting `klein_lora` in the benchmark, ablation script, or optional API `engine_mode` explicitly opts into the experimental adapter.

The default backend is local Diffusers. It runs through
`scripts/klein_diffusers_local_worker.py`, so keep the new Diffusers stack in a
Klein-specific Python environment and point `TRYON_KLEIN_PYTHON` at it:

```bash
/root/.local/bin/uv venv /workspace/venvs/project_phase2_klein --python 3.11
/root/.local/bin/uv pip install --python /workspace/venvs/project_phase2_klein/bin/python \
  "diffusers @ git+https://github.com/huggingface/diffusers.git" \
  "transformers>=4.56" "accelerate>=1.0" "peft>=0.17" "safetensors>=0.4" pillow numpy
export TRYON_KLEIN_PYTHON=/workspace/venvs/project_phase2_klein/bin/python
$TRYON_KLEIN_PYTHON scripts/download_klein_local_models.py
```

`download_klein_local_models.py` downloads:

- `black-forest-labs/FLUX.2-klein-9B` into `models/flux2-klein-9b`.
- `fal/flux-klein-9b-virtual-tryon-lora/flux-klein-tryon.safetensors` into `models/loras/`.

The 9B model is gated on Hugging Face. Accept the license on the model page and
login with `huggingface-cli login`, or set `HF_TOKEN`/`HUGGINGFACE_HUB_TOKEN`
only in the shell environment. Do not commit or print tokens.

`idm` and `klein-local` are declared as conflicting extras because IDM-VTON pins
older `diffusers`/`transformers`, while `Flux2KleinPipeline` requires the new
Diffusers implementation. The worker split lets one backend process serve both
UI modes without upgrading the IDM runtime environment.

For the optional fal.ai backend, set credentials only in the shell environment
and override the backend:

```bash
export TRYON_KLEIN_BACKEND="fal_api"
export FAL_KEY="..."
```

On RunPod, account Settings may only expose SSH keys. For a running pod, keep
private credentials in a workspace file outside the Git repo:

```bash
umask 077
mkdir -p /workspace/secrets
printf 'FAL_KEY=%q\n' "$FAL_KEY" > /workspace/secrets/virtual_tryon.env
```

Then start or restart the API with `scripts/run_backend.sh`, which sources
`/workspace/secrets/virtual_tryon.env` with auto-export enabled. You can point
to another private file by setting `VIRTUAL_TRYON_ENV_FILE`.

Do not write `FAL_KEY`, `HF_TOKEN`, or API keys into `.env`, docs, logs, outputs, or committed files. The adapter sanitizes request and response JSON before saving artifacts.

The LoRA endpoint expects three images: person, top garment, and bottom garment. For upper-body samples without a bottom garment reference, the adapter defaults to:

```yaml
bottom_strategy: "crop_from_person"
```

This crops the lower-body region from the person image, saves `auto_bottom_reference.png`, and records that the bottom reference was auto-cropped. `blank_placeholder` and `skip` are available for controlled experiments.

Benchmark logs include the final `TRYON ...` prompt. If credentials, dependencies, model access, or the execution backend are missing, the benchmark row is marked unavailable instead of crashing.
