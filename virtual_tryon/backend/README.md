# Virtual Try-On Backend

FastAPI backend for the Virtual Try-On pipeline.

## Run

```bash
cd virtual_tryon
uv sync
uv run uvicorn app.main:app --app-dir backend --reload
```

## Important Behavior

- Core try-on engine defaults to `idm_vton`.
- Supported API categories are `upper_body`, `lower_body`, `dress`, `full_outfit`, `men_underwear`, `women_underwear`, and `women_bra`.
- Adult innerwear categories use anatomy-shaped masks instead of outerwear rectangles: `men_underwear` and `women_underwear` read `garment_bottom`; `women_bra` reads `garment_top`.
- `/tryon` accepts per-job `output_width`, `output_height`, and `steps` overrides for fast preview versus quality runs.
- IDM-VTON uses the resident worker when `idm_vton.resident_worker=true`, keeping the model loaded between jobs and falling back to the subprocess runner if configured. `idm_vton.resident_worker_optimization` selects `eager` or experimental compiler modes.
- Missing core checkpoints return a clear failed job message.
- Mock engine is available for tests and pipeline validation via `TRYON_ENGINE=mock`.
- FLUX refiner and ADetailer-like repair are post-processing modules only.
- Every job writes intermediates to `data/outputs/{job_id}/`.
- `/tryon/history` lists retained job folders with inputs, config, output, runtime, and status.
- `use_refiner=true` is best-effort: FLUX load/OOM/runtime failures are logged and the job falls back to `core_output.png`.
- Every completed core job writes `quality_report.json`, `mask_metadata.json`, and refine mask overlays for debugging.
- `/health` returns detailed model status strings; benchmark baselines can be unavailable without affecting the default IDM-VTON API.
