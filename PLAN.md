# Virtual Try-On Implementation Plan

## Current Repository

- Git remote: `https://github.com/NguyenNgocMinh30012005/Project_Phase2.git`
- Tracked files before this implementation: `README.md`
- Existing local working files kept untouched:
  - RunPod setup helpers: `runpod_bootstrap.sh`, `runpod_connect_setup.ps1`, `runpod_verify.py`
  - Reference PDFs for FLUX architecture/model notes

## Current Stack

- No application stack existed yet.
- Target stack being introduced:
  - Backend: FastAPI, Pydantic, Pillow, NumPy, PyYAML
  - Test: pytest, FastAPI TestClient
  - Frontend MVP: React + TypeScript + Vite-style structure
  - Runtime target: RunPod RTX 3090 Ti with CUDA/PyTorch

## Missing Pieces Found

- No backend API.
- No model configuration layer.
- No pipeline abstraction.
- No try-on engine interface.
- No preprocessing/mask utilities.
- No storage/job service.
- No frontend.
- No documentation for model setup, API, or architecture.
- No tests.

## Modules To Add

- FastAPI backend with `/health`, `/tryon`, `/tryon/{job_id}`, and `/tryon/refine`.
- Config loader reading `configs/*.yaml` without hard-coded absolute paths.
- Engine interfaces and adapters:
  - IDM-VTON core engine.
  - FLUX refiner.
  - CatVTON baseline.
  - FLUX Klein Try-On LoRA baseline.
  - ADetailer-like repair module.
  - Mock engines for tests and early end-to-end validation.
- Preprocessing:
  - Image loading/normalization.
  - Human parser and DensePose stubs with clear warnings.
  - Agnostic mask generation.
  - Garment segmentation.
  - Mask utilities.
- Storage and job services with debug intermediates saved under `data/outputs/{job_id}/`.
- Quality checks with a scoring object.
- CLI scripts for single-run try-on and benchmarking.
- React frontend MVP.
- Docs and tests.

## Implementation Notes

- Production default uses `idm_vton` as the configured core engine.
- If IDM-VTON checkpoints are missing, API/job errors must be explicit, e.g. `IDM-VTON checkpoint not found at ...`.
- Mock engine is available through config/env/tests, but real production paths must not silently fall back to mock.
- FLUX/repair modules are post-processing/refinement only, not core try-on.
