# Virtual Try-On Backend

FastAPI backend for the Virtual Try-On pipeline.

## Run

```bash
cd virtual_tryon
uv sync
uv run uvicorn app.main:app --app-dir backend --reload
```

On RunPod, `scripts/run_backend.sh` also reads exported variables from
`/workspace/secrets/virtual_tryon.env` or from `VIRTUAL_TRYON_ENV_FILE` when it
is set. Use that file for private runtime overrides such as Hugging Face tokens
or the optional fal.ai backend:

```bash
umask 077
mkdir -p /workspace/secrets
printf 'HF_TOKEN=%q\n' "$HF_TOKEN" > /workspace/secrets/virtual_tryon.env
bash scripts/run_backend.sh
```

Do not write API keys into tracked files or logs.

## Important Behavior

- Core try-on engine defaults to `idm_vton`.
- Supported API categories are `upper_body`, `lower_body`, `dress`, `full_outfit`, `men_underwear`, `women_underwear`, and `women_bra`.
- Adult innerwear categories use anatomy-shaped masks instead of outerwear rectangles: `men_underwear` and `women_underwear` read `garment_bottom`; `women_bra` reads `garment_top`.
- Mask generation rejects edge-touching foreground body boxes, extends upper-body masks lower for shirt hems, and places underwear masks on the lower pelvis rather than the abdomen.
- `/tryon` accepts per-job `output_width`, `output_height`, and `steps` overrides for fast preview versus quality runs.
- IDM-VTON uses the resident worker when `idm_vton.resident_worker=true`, keeping the model loaded between jobs and falling back to the subprocess runner if configured. `idm_vton.resident_worker_optimization` selects `eager` or experimental compiler modes.
- Missing core checkpoints return a clear failed job message.
- Mock engine is available for tests and pipeline validation via `TRYON_ENGINE=mock`.
- FLUX refiner and ADetailer-like repair are post-processing modules only.
- Klein LoRA defaults to `diffusers_local`; it runs through `scripts/klein_diffusers_local_worker.py` and should use `TRYON_KLEIN_PYTHON` pointing to a Klein-specific venv with the `klein-local` dependency stack. It also requires the local `models/flux2-klein-9b` snapshot and `models/loras/flux-klein-tryon.safetensors`. The old fal.ai path is available with `TRYON_KLEIN_BACKEND=fal_api` and `FAL_KEY`.
- Local Klein stops the resident IDM worker before loading so GPU memory is released during model switching.
- Every job writes intermediates to `data/outputs/{job_id}/`.
- `/tryon/history` lists retained job folders with inputs, config, output, runtime, status, and per-stage timings.
- `use_refiner=true` is best-effort: FLUX load/OOM/runtime failures are logged and the job falls back to `core_output.png`.
- Every completed core job writes `quality_report.json`, `mask_metadata.json`, and refine mask overlays for debugging.
- `/health` returns detailed model status strings; benchmark baselines can be unavailable without affecting the default IDM-VTON API.
