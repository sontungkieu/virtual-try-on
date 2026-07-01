# Demo Runbook

## RunPod Backend

```bash
cd /workspace/Project_Phase2
source env.sh
cd virtual_tryon
bash scripts/run_backend.sh
```

The backend listens on port `8000`.

Health:

```bash
curl http://127.0.0.1:8000/health
```

## Frontend

Install/build once:

```bash
cd /workspace/Project_Phase2/virtual_tryon
bash scripts/setup_frontend_runtime.sh
cd frontend
npm install
npm run build
```

Run dev server:

```bash
cd /workspace/Project_Phase2/virtual_tryon
VITE_API_BASE_URL=http://127.0.0.1:8000 bash scripts/run_frontend.sh
```

The frontend listens on port `5173`.

## Expose Ports

On RunPod, expose/proxy:

- backend: `8000`
- frontend: `5173`

Open the frontend URL, upload `data/eval_set/sample_001/person.jpg` and `garment_top.jpg`, select `Top`, and generate.

## Artifacts

Every job writes files under:

```text
data/outputs/{job_id}/
```

The backend serves them as:

```text
/artifacts/{job_id}/result.png
/artifacts/{job_id}/core_output.png
/artifacts/{job_id}/quality_report.json
```

Only `data/outputs` is served. Models, checkpoints, third-party repos, temp files, and environment files are not exposed.

## E2E Smoke Test

With the backend running:

```bash
cd /workspace/Project_Phase2/virtual_tryon
uv run python scripts/e2e_smoke_test.py \
  --api-base http://127.0.0.1:8000 \
  --sample data/eval_set/sample_001 \
  --use-refiner false \
  --timeout 900
```

The script uploads a sample, polls if the job is async, fetches `result_url` and `quality_report_url`, and writes a report under `data/outputs`.

## Async Mode

Default is sync. To enable async:

```bash
export TRYON_API_RUN_MODE=async
bash scripts/run_backend.sh
```

Or send form field `run_mode=async`. `POST /tryon` returns `queued`; poll `GET /tryon/{job_id}`.

## Cleanup

```bash
uv run python scripts/cleanup_outputs.py --older-than-hours 24 --keep-latest 5 --dry-run
uv run python scripts/cleanup_outputs.py --older-than-hours 24 --keep-latest 5
```

Cleanup only removes generated folders under `data/outputs`.

## Debug

- Core errors: check `data/outputs/{job_id}/job.json`.
- IDM-VTON runtime: check `idm_vton_command.txt`, `idm_vton_stdout.txt`, `idm_vton_stderr.txt`.
- Refiner fallback: check `flux_refiner_error.txt` and `quality_report.json`.
- Frontend API URL: set `VITE_API_BASE_URL`.
