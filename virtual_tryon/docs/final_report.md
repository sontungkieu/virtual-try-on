# Virtual Try-On Final Report

## Problem Statement

This project generates a new image of a person wearing a reference garment while preserving the person's identity, pose, body shape, and background. The primary engineering challenge is to transfer garment color, texture, and structure without turning the task into unrestricted image editing.

## Input And Output

The pipeline accepts a person image, one or more garment images, a garment category, and an optional text prompt. Supported categories in the evaluation schema are `upper_body`, `lower_body`, `dress`, `full_outfit`, `men_underwear`, `women_underwear`, and `women_bra`, subject to engine capability and checkpoint availability. Innerwear categories are scoped to adult, non-sexual product try-on and use narrower target masks than outerwear categories.

Each job writes isolated artifacts under `data/outputs/{job_id}/`. Important outputs include `core_output.png`, the selected `result.png`, `quality_report.json`, `artifact_manifest.json`, masks, run metadata, and engine stdout/stderr logs.

## Architecture

The system contains a FastAPI backend, a React/TypeScript frontend, engine adapters, preprocessing modules, an asynchronous job service, quality evaluation, observability endpoints, and release scripts. Runtime configuration lives under `configs/`, while model weights and third-party source remain local and untracked.

```text
person + garment + category + prompt
  -> validation and normalization
  -> parsing, DensePose, and mask preparation
  -> IDM-VTON core inference
  -> optional FLUX refinement
  -> quality gate and final selection
  -> artifacts, report, metrics, and frontend result
```

## IDM-VTON Core

IDM-VTON is the default garment-conditioned core. Its adapter stages a one-sample VITON-HD-style dataset, invokes the isolated official implementation through `accelerate`, captures command/stdout/stderr evidence, and copies the generated image to `core_output.png`. Health checks report missing repositories, dependencies, or checkpoints before inference starts.

## Optional FLUX Refiner

FLUX is an optional post-processing stage for garment boundaries, folds, lighting, and occlusion artifacts. It operates on constrained masks and is not the core try-on model. If it is unavailable or rejected by the quality gate, the completed IDM-VTON output remains the final result.

## Async Job System

`POST /tryon` supports asynchronous execution so GPU inference does not hold the request open. Jobs move through `queued`, `running`, and terminal states, persist status in job artifacts, serialize local GPU work, support retry, expose cooperative cancellation, and return clean API errors instead of raw stack traces.

## Artifact Serving

Artifacts are served only from the configured output root and only for allowlisted image, JSON, CSV, HTML, and text extensions. Path traversal and model/code file extensions are blocked. Public serving can be disabled through configuration, and every completed job receives an `artifact_manifest.json`.

## Quality Report

`quality_report.json` records core and refined scores, engine status, notes, the final choice, and the reason for that choice. Current automated checks cover background preservation, garment change, over-editing, and basic artifact heuristics. Missing parser-dependent signals are represented as `null` with explanatory notes.

## Benchmark And Evaluation Set

The committed evaluation set contains one valid upper-body sample for repeatable smoke validation. `validate_eval_set.py` checks metadata and required images. `benchmark_pipeline.py` can compare `idm`, `idm_flux`, and optional baselines, producing JSON/CSV summaries, a grid, an HTML review gallery, and a manual rating sheet.

## Frontend Demo

The frontend provides person and garment upload, category and prompt controls, asynchronous status polling, a timed job timeline, final image display, quality summary, artifact links, and job ID copy. API errors are rendered from the backend's structured error schema.

## Observability

`/health` reports service and engine readiness. `/system` reports Python, PyTorch, CUDA, GPU, and commit information. `/metrics` exports Prometheus-compatible job counts, runtime summaries, failure counts, GPU memory, queue size, and artifact bytes. Structured job events provide a traceable execution history.

## Security And Safety

The API enforces upload size, MIME type, extension, and decodable-image checks. CORS, queue behavior, timeout, retries, public artifacts, and retention are configurable. Secrets, checkpoints, third-party clones, runtime outputs, temporary data, frontend dependencies, and generated browser artifacts are excluded from version control.

GitHub Actions intentionally runs mock-only backend tests and the frontend build. Hosted runners do not have the required GPU, IDM-VTON source, or checkpoints; real inference remains a documented RunPod release check.

## Limitations

- Logos, typography, fine patterns, unusual poses, and strong hand or hair occlusion remain difficult.
- Safe masks do not yet use complete face, hair, and hand exclusion signals.
- FLUX can over-edit identity or garment details and therefore remains optional.
- Active GPU cancellation is cooperative, and timeout handling does not yet terminate the subprocess immediately.
- The demo has no authentication, tenant isolation, quota system, or private per-user artifact authorization.
- Performance and reproducibility depend on external model versions, licenses, and local checkpoints.

## Future Work

Future work should expand the evaluation set, add parser-aware exclusion masks, introduce perceptual and identity metrics, implement process-level cancellation, and add authenticated artifact access. Broader lower-body, dress, and innerwear support should be claimed only after engine-specific evaluation. Deployment work should also add a durable queue, external artifact storage, retention automation, and GPU telemetry dashboards.
