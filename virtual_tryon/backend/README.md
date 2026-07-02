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
- `/tryon` accepts per-job `output_width`, `output_height`, `steps`, `seed`, and `deterministic` overrides for fast preview versus reproducible runs. Deterministic mode is best-effort because CUDA/TensorRT kernels may still be non-bit-exact.
- IDM-VTON uses the resident worker when `idm_vton.resident_worker=true`, keeping the model loaded between jobs and falling back to the subprocess runner if configured. `idm_vton.resident_worker_optimization` selects `eager`, `torch_compile`, or `tensorrt`; TensorRT supports `TRYON_TRT_PROFILE=stable` and the tested full block-wise profile `TRYON_TRT_PROFILE=full_safe`.
- Missing core checkpoints return a clear failed job message.
- Mock engine is available for tests and pipeline validation via `TRYON_ENGINE=mock`.
- FLUX refiner and ADetailer-like repair are post-processing modules only.
- `engine_mode=flux_redux_catvton` is an explicit ComfyUI bridge: it sends the backend dynamic mask, person image, and garment reference to the local Flux Fill + Redux + CatVTON graph on port `8188`, then writes the ComfyUI workflow/history/output artifacts into the same job folder.
- Klein LoRA defaults to `diffusers_local`; it runs through `scripts/klein_diffusers_local_worker.py` and should use `TRYON_KLEIN_PYTHON` pointing to a Klein-specific venv with the `klein-local` dependency stack. It also requires the local `models/flux2-klein-9b` snapshot and `models/loras/flux-klein-tryon.safetensors`. The default placement is `TRYON_KLEIN_DEVICE_MAP=cpu_offload`. For the tested all-GPU path on a 24 GB GPU, use `TRYON_KLEIN_DEVICE_MAP=cuda`, `TRYON_KLEIN_QUANTIZATION=bnb_4bit`, and `TRYON_KLEIN_QUANTIZE_COMPONENTS=transformer,text_encoder`. Klein TensorRT is separate from IDM TensorRT: `TRYON_KLEIN_TRT_PROFILE=vae_decode` compiles only VAE decode, while transformer/full profiles are debug-only and reject bnb-quantized transformer weights. The old fal.ai path is available with `TRYON_KLEIN_BACKEND=fal_api` and `FAL_KEY`.
- Local Klein stops the resident IDM worker before loading so GPU memory is released during model switching.
- Every job writes production artifacts to `data/outputs/{job_id}/`; request `save_intermediates=true` to also write full debug masks, overlays, crops, and densepose placeholders.
- `core_output_raw.png` is the raw model output. `core_output.png` and `result.png` are mask-composited against the original person image, preserving unmasked pixels even when a global engine generates outside the target region.
- `/tryon/history` lists retained job folders with inputs, config, seed, output, runtime, status, and per-stage timings.
- The web backend does not queue ComfyUI by default; the `flux_redux_catvton` engine mode is the explicit exception. ComfyUI JSON workflows are also packaged separately for demo/batch runs.
- `use_refiner=true` is best-effort: FLUX load/OOM/runtime failures are logged and the job falls back to `core_output.png`.
- Every completed core job writes `quality_report.json` and `mask_metadata.json`. Refine mask overlays and detailed mask/crop images are written when intermediate saving is enabled.
- `/health` returns detailed model status strings; benchmark baselines can be unavailable without affecting the default IDM-VTON API. Disabled engines return a lightweight disabled status and do not spawn local worker runtime checks.
