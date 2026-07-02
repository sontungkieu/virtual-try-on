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
- `/health` reports `active_engine_mode`, `loaded_engine_mode`, and `default_engine_mode`. The web UI uses this on startup to select the already loaded resident model instead of immediately switching models; if none is loaded, it defaults to `klein_bnb_4bit`.
- `/tryon/model/prepare` preloads the selected `engine_mode` before generation. The web UI disables Generate while this request is loading so model startup is not hidden inside the generation timer.
- During generation, the web UI timer follows the active pipeline stage such as `Loading model` or `Generating` instead of counting only from the original button click.
- IDM-VTON uses the resident worker when `idm_vton.resident_worker=true`, keeping the model loaded between jobs and falling back to the subprocess runner if configured. `idm_vton.resident_worker_optimization` selects `eager`, `torch_compile`, or `tensorrt`; TensorRT supports `TRYON_TRT_PROFILE=stable` and the tested full block-wise profile `TRYON_TRT_PROFILE=full_safe`.
- Missing core checkpoints return a clear failed job message.
- Mock engine is available for tests and pipeline validation via `TRYON_ENGINE=mock`.
- FLUX refiner and ADetailer-like repair are post-processing modules only.
- Klein LoRA defaults to `diffusers_local`; it runs through a resident `scripts/klein_diffusers_local_worker.py` process and should use `TRYON_KLEIN_PYTHON` pointing to a Klein-specific venv with the `klein-local` dependency stack. The worker loads the current base model + LoRA + quantization preset once, keeps it resident, and reloads only when that model cache key changes. Innerwear-bottom requests use the backend agnostic person image as the Klein preserve/top reference when available, so the old underwear is removed from conditioning before Klein sees the reference set. It also requires the local `models/flux2-klein-9b` snapshot and `models/loras/flux-klein-tryon.safetensors`. The default placement is `TRYON_KLEIN_DEVICE_MAP=cpu_offload`. For the tested all-GPU path on a 24 GB GPU, use `TRYON_KLEIN_DEVICE_MAP=cuda`, `TRYON_KLEIN_QUANTIZATION=bnb_4bit`, and `TRYON_KLEIN_QUANTIZE_COMPONENTS=transformer,text_encoder`. Klein TensorRT is separate from IDM TensorRT: `TRYON_KLEIN_TRT_PROFILE=vae_decode` compiles only VAE decode, while transformer/full profiles are debug-only and reject bnb-quantized transformer weights. The old fal.ai path is available with `TRYON_KLEIN_BACKEND=fal_api` and `FAL_KEY`.
- Model switching is explicit: preparing/running Klein releases IDM resident workers; preparing/running IDM releases Klein resident workers. Hybrid cannot keep IDM and Klein pinned together on the 24 GB target GPU, so one phase may still load during the job.
- Every job writes production artifacts to `data/outputs/{job_id}/`; request `save_intermediates=true` to also write full debug masks, overlays, crops, and densepose placeholders.
- `core_output_raw.png` is the raw model output. `core_output.png` and `result.png` are mask-composited against the original person image, preserving unmasked pixels even when a global engine generates outside the target region.
- `/tryon/history` lists retained job folders with inputs, config, seed, output, runtime, status, and per-stage timings. History displays the requested `engine_mode` when present, so `klein_lora` and `klein_bnb_4bit` remain distinguishable.
- The web backend does not queue ComfyUI by default. ComfyUI JSON workflows are packaged separately for demo/batch reproduction and can call the backend through the custom node.
- `use_refiner=true` is best-effort: FLUX load/OOM/runtime failures are logged and the job falls back to `core_output.png`.
- Every completed core job writes `quality_report.json` and `mask_metadata.json`. Refine mask overlays and detailed mask/crop images are written when intermediate saving is enabled.
- `/health` returns detailed model status strings plus the currently loaded resident engine mode when one can be detected; benchmark baselines can be unavailable without affecting the default UI mode. Disabled engines return a lightweight disabled status and do not spawn local worker runtime checks.
