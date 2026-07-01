# Virtual Try-On v0.1.0-rc1

Release candidate for the GPU-backed Virtual Try-On demo built around IDM-VTON with an optional FLUX refinement stage.

## Features

- FastAPI API with validated image uploads and structured error responses.
- Real IDM-VTON adapter with VITON-HD-style staging, checkpoint readiness checks, and captured inference logs.
- Optional FLUX refinement with mask constraints, quality gating, and core-output fallback.
- Synchronous and asynchronous jobs with persisted lifecycle, retry, cooperative cancellation, timeout reporting, and serialized GPU execution.
- Allowlisted artifact serving with per-job manifests and configurable public access.
- Quality reports, evaluation-set validation, benchmark summaries, and review gallery generation.
- React/TypeScript demo with uploads, status timeline, result viewer, quality summary, artifacts, and job ID copy.
- `/health`, `/system`, and Prometheus-compatible `/metrics` endpoints.
- Lightweight GitHub Actions CI for mock backend tests and the frontend production build.

## Verification Results

- Backend mock test suite: `61 passed, 3 warnings`.
- Frontend production build: passed.
- Playwright Chromium E2E: `1 passed`.
- Evaluation-set validation: one valid upper-body sample at medium difficulty.
- Real IDM-VTON API E2E on an RTX 3090 Ti: passed in `49.104s` with `use_refiner=false`.
- `/health`: IDM-VTON available.
- `/system`: CUDA available with RTX 3090 Ti, PyTorch 2.8.0+cu128, and Python 3.12.3.
- Required `/metrics` series: present.

GitHub CI is mock-only. It does not download checkpoints or run real IDM-VTON, FLUX, Playwright against a real backend, benchmarks, or GPU tests.

## Known Limitations

- Fine logos, typography, patterns, unusual poses, and hand or hair occlusion can reduce output fidelity.
- FLUX refinement depends on local model availability and can be rejected for over-editing.
- Cancellation and timeout enforcement do not forcibly terminate an active model subprocess.
- The current committed evaluation set is intentionally small and is not a production quality benchmark.
- Authentication, multi-tenant isolation, durable distributed queues, and private artifact authorization are not included.

## Setup Requirements

- Linux GPU environment with CUDA and sufficient VRAM; the verified environment used an RTX 3090 Ti.
- Python backend dependencies from `pyproject.toml` and `uv.lock`.
- Node LTS and frontend dependencies from `frontend/package-lock.json`.
- IDM-VTON source under `third_party/IDM-VTON` and the required local checkpoints under `models/idm_vton/`.
- RunPod users should load `/workspace/Project_Phase2/env.sh` before starting services.

## Non-Committed Assets

The release source does not include model weights or checkpoints, third-party repositories, generated outputs, temporary data, environment files or tokens, `node_modules`, frontend `dist`, or Playwright reports, traces, videos, and screenshots. These assets must be provisioned locally according to `docs/model_setup.md` and the deployment instructions.
