# Final Demo Checklist

## Before The Session

- [ ] RunPod pod is running and the project volume is mounted at `/workspace/Project_Phase2`.
- [ ] Environment is loaded with `source /workspace/Project_Phase2/env.sh`.
- [ ] Git commit matches the intended release candidate.
- [ ] IDM-VTON repository, runtime dependencies, and checkpoints are available locally.
- [ ] No token, checkpoint, output, temp file, or generated frontend artifact is staged.

## Start Services

- [ ] Backend is running with `bash scripts/run_backend.sh` from `virtual_tryon/`.
- [ ] Frontend is running with `VITE_API_BASE_URL=http://127.0.0.1:8000 bash scripts/run_frontend.sh`.
- [ ] `/health` returns HTTP 200 and reports `idm_vton: available`.
- [ ] `/system` reports CUDA and the expected GPU.
- [ ] `/metrics` returns Prometheus-compatible text.

## Demonstrate The Workflow

- [ ] Upload the sample person image.
- [ ] Upload the sample reference garment.
- [ ] Select the correct category and submit an asynchronous job.
- [ ] Show the job moving from queued/running to completed.
- [ ] Confirm the generated result is visible in the frontend.
- [ ] Open the quality summary and explain the final-choice reason.
- [ ] Show the artifact list, including `core_output.png`, `quality_report.json`, and `artifact_manifest.json`.
- [ ] Copy the job ID and locate `data/outputs/{job_id}/` on the backend.

## Release Evidence

- [ ] `bash scripts/release_check.sh` ends with `FINAL RESULT: PASS`.
- [ ] Real IDM-VTON smoke has passed with `use_refiner=false`.
- [ ] Frontend build and Playwright evidence are available.
- [ ] Known limitations and the optional status of FLUX are stated clearly.
