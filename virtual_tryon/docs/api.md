# API

## GET /health

Returns service status, detected device, and model availability.

```json
{
  "status": "ok",
  "device": "cuda:NVIDIA GeForce RTX 3090 Ti",
  "models": {
    "idm_vton": "available",
    "flux_refiner": "unavailable: license/access not accepted or model is private",
    "comfyui_flux_redux": "available",
    "catvton": "unavailable: catvton.enabled is false"
  }
}
```

Model status strings may include detailed skip reasons. When the resident IDM-VTON worker is enabled, the `idm_vton` status also includes `resident_worker=not_started`, `running pid=...`, or an exited return code. Disabled optional engines return a lightweight disabled status instead of running local worker dependency checks. IDM-VTON is the default core API engine; CatVTON, Klein LoRA, and the ComfyUI Flux Redux bridge are benchmark or ablation paths unless explicitly selected.

## POST /tryon/model/prepare

Preloads the selected `engine_mode` before generation. The frontend calls this
when the engine dropdown changes and keeps Generate disabled until the response
is `ready`.

```json
{
  "engine_mode": "klein_bnb_4bit"
}
```

Response:

```json
{
  "status": "ready",
  "engine": "klein_tryon_lora",
  "engine_mode": "klein_bnb_4bit",
  "runtime_seconds": 39.2,
  "metadata": {
    "resident_worker": "running pid=1234",
    "worker": {
      "cached": false,
      "quantization": "bnb_4bit"
    }
  },
  "message": null
}
```

Preparing IDM releases a resident Klein worker; preparing Klein releases a
resident IDM worker. This keeps model switching explicit on the 24 GB target
GPU. Hybrid modes may return a `message` because IDM and Klein cannot reliably
stay pinned together in VRAM; one phase can still load during the job and will
show up as the `loading_model` stage.

## POST /tryon

Multipart form fields:

- `person_image`: required image file.
- `garment_top`: optional image file.
- `garment_bottom`: optional image file.
- `garment_dress`: optional image file.
- `category`: `upper_body`, `lower_body`, `dress`, `full_outfit`, `men_underwear`, `women_underwear`, or `women_bra`.
- `prompt`: optional text.
- `use_refiner`: boolean, default `true`.
- `repair_mode`: boolean, default `true`.
- `run_mode`: optional `sync` or `async`; defaults to `configs/pipeline.yaml`.
- `engine_mode`: optional `idm_vton`, `idm_mask_expanded`, `idm_vton_flux`, `idm_mask_expanded_flux`, `flux_redux_catvton`, `klein_lora`, `klein_bnb_4bit`, `idm_klein_hybrid`, `idm_klein_hybrid_pro`, or `catvton`; default is the configured core engine.
- `auto_prompt`: optional boolean, default `false`.
- `testcase_id`: optional `tc1` through `tc15`; required when `auto_prompt=true`.
- `prompt_variant`: optional `default`, `strong_remove_old_garment`, or `identity_strict`.
- `seed`: optional integer.
- `output_width` and `output_height`: optional output resolution override; provide both, use multiples of 8, each side 384-1536px, and at most 1572864 total pixels.
- `steps`: optional IDM-VTON inference step override, 4-50.

Response:

```json
{
  "job_id": "abc",
  "status": "completed",
  "current_stage": "completed",
  "stages": [
    {"key": "queued", "label": "Queued", "status": "completed", "runtime_seconds": 0.0},
    {"key": "running", "label": "Preprocess", "status": "completed", "runtime_seconds": 1.2},
    {"key": "loading_model", "label": "Loading model", "status": "skipped", "runtime_seconds": 0.0},
    {"key": "generating", "label": "Generating", "status": "completed", "runtime_seconds": 42.5},
    {"key": "refining", "label": "Refining", "status": "skipped", "runtime_seconds": 0.0},
    {"key": "completed", "label": "Completed", "status": "completed", "runtime_seconds": 0.1}
  ],
  "result_url": "/artifacts/abc/result.png",
  "debug": {
    "mask_url": "/artifacts/abc/mask_preview.png",
    "mask_urls": ["/artifacts/abc/mask_preview.png"],
    "agnostic_url": "/artifacts/abc/agnostic.png",
    "core_output_url": "/artifacts/abc/core_output.png",
    "refined_output_url": "/artifacts/abc/refined_output.png",
    "quality_report_url": "/artifacts/abc/quality_report.json",
    "refine_mask_url": "/artifacts/abc/safe_refine_mask_overlay.png",
    "mask_metadata_url": "/artifacts/abc/mask_metadata.json",
    "prompt_core_url": "/artifacts/abc/prompt_core.txt",
    "prompt_refine_url": "/artifacts/abc/prompt_refine.txt",
    "prompt_metadata_url": "/artifacts/abc/prompt_metadata.json"
  },
  "seed": 123
}
```

When `run_mode=async`, `POST /tryon` returns quickly:

```json
{
  "job_id": "abc",
  "status": "queued",
  "current_stage": "queued",
  "stages": [
    {"key": "queued", "label": "Queued", "status": "running"},
    {"key": "running", "label": "Preprocess", "status": "pending"},
    {"key": "loading_model", "label": "Loading model", "status": "pending"},
    {"key": "generating", "label": "Generating", "status": "pending"},
    {"key": "refining", "label": "Refining", "status": "pending"},
    {"key": "completed", "label": "Completed", "status": "pending"}
  ],
  "result_url": null,
  "debug": {}
}
```

Poll `GET /tryon/{job_id}` until `completed` or `failed`.

If the core model is missing, the job returns `status: failed` and a clear `error` string.

Prompt behavior:

- Manual prompt with `auto_prompt=false` preserves the existing API behavior.
- `auto_prompt=true` builds an engine-specific prompt from testcase metadata.
- `engine_mode=klein_lora`, `klein_bnb_4bit`, `idm_klein_hybrid`, and `idm_klein_hybrid_pro` normalize the prompt so it starts with `TRYON`.
- `engine_mode=idm_vton_flux` or `idm_mask_expanded_flux` can save both core and refine prompts.
- `engine_mode=flux_redux_catvton` uses the local ComfyUI Flux Fill + Redux + CatVTON graph as the core generator and marks the backend FLUX refiner stage as skipped.
- Prompt artifacts are served through the same `/artifacts` route as images and reports.

Generation overrides:

- Resolution and step overrides apply to the current job only.
- The default remains `768x1024` with 30 IDM-VTON steps.
- For faster previews, use `512x768` with 12-18 steps or `640x896` with 18-22 steps.
- Higher resolutions and step counts are slower and increase GPU memory use.

Category upload mapping:

- `upper_body` and `women_bra` use `garment_top`.
- `lower_body`, `men_underwear`, and `women_underwear` use `garment_bottom`.
- `dress` uses `garment_dress`.
- `full_outfit` uses the dress reference first, then top, then bottom if multiple references are uploaded.

Masks are built from a dynamic body estimate for the uploaded person image. The backend first tries foreground separation, then adaptive row/column background estimation and center-saliency estimation; only then does it fall back to a conservative body box. The adult innerwear categories use anatomy-shaped masks rather than broad outerwear rectangles. Underwear bottom masks target the pelvis/brief region with leg openings; `women_bra` targets cup, bridge, band, and strap regions. Repeated jobs with the same person image, category, resolution, and mask config reuse cached masks from `data/temp/mask_cache`. Mask source, cache hit/miss, bbox, area, and warnings are recorded in `mask_metadata.json`. These categories are intended for adult-only, non-sexual product try-on workflows.

Generation reproducibility is controlled with form fields `seed` and
`deterministic`. If `seed` is omitted, the backend generates one and returns it
in the job response. If `deterministic=true`, the backend and resident/local
workers enable deterministic PyTorch settings on a best-effort basis; CUDA,
TensorRT, and some attention kernels may still prevent exact bit-for-bit output.
The web API normally runs Python engines/workers directly. The explicit
`engine_mode=flux_redux_catvton` exception queues the local ComfyUI server on
port `8188` with the backend-generated dynamic mask and saves the ComfyUI
workflow/history artifacts into the same job output folder. Packaged ComfyUI
workflow files remain available for separate demo/batch reproduction.

Core model output is mask-composited before it becomes the final image. The
backend saves `core_output_raw.png` for inspection, then pastes only the
soft-mask region from that raw model image onto the original person image and
writes the constrained result to `core_output.png` and `result.png`. This is
especially important for global engines such as Klein LoRA, which do not accept
a hard inpaint mask internally.

`engine_mode=klein_lora` is experimental. It opts into the Klein Try-On LoRA adapter for that request only, requires a top garment, and uses the configured bottom-reference strategy when no bottom garment is uploaded. The default backend is local Diffusers and requires `models/flux2-klein-9b`, `models/loras/flux-klein-tryon.safetensors`, and the `klein-local` dependency set. Local Klein runs through a resident worker; the engine stops the resident IDM worker before loading Klein so VRAM is released at model switch time. If local model files, dependencies, or model access are missing, the job returns `status: failed` with `error_code: ENGINE_UNAVAILABLE` and no raw stack trace. Local placement is controlled by `TRYON_KLEIN_DEVICE_MAP`; default `cpu_offload` is safest, while `cuda` attempts to place the full pipe on GPU. On 24 GB GPUs, `engine_mode=klein_bnb_4bit` forces the tested all-GPU mode: `device_map=cuda`, `quantization=bnb_4bit`, and `quantize_components=transformer,text_encoder`. This still loads `flux-klein-tryon.safetensors` and sets the Try-On LoRA adapter; bnb 4-bit only changes how the base Klein pipeline components are loaded. `engine_mode=idm_klein_hybrid` runs IDM first, runs Klein second, and fuses Klein detail inside the IDM delta mask. `engine_mode=idm_klein_hybrid_pro` uses the same hybrid fusion but forces IDM resident `torch_compile` plus the Klein bnb 4-bit preset. Optional Klein TensorRT uses `TRYON_KLEIN_TRT_PROFILE=vae_decode` and records TensorRT metadata in artifacts; transformer/full TensorRT profiles are debug-only and reject bnb-quantized transformer weights. `TRYON_KLEIN_BACKEND=fal_api` keeps the old fal.ai path and requires `FAL_KEY`.

`engine_mode=flux_redux_catvton` is experimental. It requires ComfyUI to be
running and reachable through `TRYON_COMFYUI_URL` or the default
`http://127.0.0.1:8188`, with Flux Fill FP8, Redux, SigCLIP, CLIP/T5, VAE, and
CatVTON LoRA files installed under `/workspace/ComfyUI/models`. It writes
`comfyui_flux_redux_workflow.json`, `comfyui_flux_redux_history.json`, and
`comfyui_flux_redux_output.png` for inspection, then mask-composites the raw
ComfyUI output into `core_output.png` and `result.png`.

If `use_refiner=true` and the FLUX refiner is missing, incompatible, or runs out of memory, the job still returns `status: completed` with `result_url` pointing to the IDM-VTON core output. The output folder includes `flux_refiner_error.txt` and `quality_report.json` explaining the fallback. Raw stack traces are not returned in the API response. Repair runs only after a refined output is created and accepted by the quality gate.

Important output files:

```text
core_output_raw.png
core_output.png
result.png
quality_report.json
mask_metadata.json
mask_innerwear_shape.png
mask_body_silhouette.png
garment_refine_mask.png
boundary_refine_mask.png
safe_refine_mask.png
garment_refine_mask_overlay.png
boundary_refine_mask_overlay.png
safe_refine_mask_overlay.png
idm_vton_command.txt
idm_vton_stdout.txt
idm_vton_stderr.txt
idm_vton_worker_request.json
idm_vton_worker_response.json
idm_vton_resident_error.txt
```

`quality_report.json` includes `engine_status`, `final_choice`, and `final_choice_reason`. For model comparison across CatVTON/Klein baselines, use `scripts/benchmark_pipeline.py` instead of the default `/tryon` API.

## GET /artifacts/{path}

Serves files under `data/outputs` only:

```text
/artifacts/{job_id}/result.png
/artifacts/{job_id}/core_output.png
/artifacts/{job_id}/quality_report.json
```

Path traversal and files outside `data/outputs` return clean 404 responses. Models, checkpoints, `third_party`, `.env`, and tokens are never served by this route.

Example missing-model response:

```json
{
  "job_id": "abc",
  "status": "failed",
  "result_url": null,
  "error": "IDM-VTON is not available. missing checkpoint: densepose/model_final_162be9.pkl"
}
```

## GET /tryon/{job_id}

Returns the stored job status. Job metadata is also written to:

```text
data/outputs/{job_id}/job.json
```

`job.json` includes `queued`, `running`, `completed`, or `failed`, timestamps, clean error text, result URL, debug URLs, engine status, `current_stage`, and per-stage timings. Stages are `queued`, `running`, `loading_model`, `generating`, `refining`, and `completed`; `refining` is marked `skipped` when refinement is disabled or unavailable, and `loading_model` is marked `skipped` when the selected resident worker is already warm.

## GET /tryon/history

Returns recent jobs from `data/outputs` with input artifact URLs, result URL, category, engine, output resolution, step count, seed, timestamps, runtime, per-stage timings, quality summary, and engine status.

```text
/tryon/history?limit=20
```

The frontend history panel uses this endpoint so completed jobs remain visible after a browser refresh as long as their output folders are retained.

## DELETE /tryon/{job_id}

Cancels a queued job. Running jobs are marked with `cancel_requested`; the current local executor does not kill an active IDM-VTON subprocess.

## POST /tryon/refine

Multipart form fields:

- `image`: required image file.
- `mask`: optional image file.
- `prompt`: required or default prompt.
- `seed`: optional integer.
