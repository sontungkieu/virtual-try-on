# Release Candidate Checklist

- [ ] `git status` is clean except ignored local model/runtime files.
- [ ] `PYTHONPATH=. TRYON_ENGINE=mock pytest` passes.
- [ ] `uv run python scripts/validate_eval_set.py --eval-set data/eval_set` passes.
- [ ] `npm run build` passes in `frontend/`.
- [ ] API E2E smoke passes against the running backend.
- [ ] Playwright E2E passes against running backend and frontend services.
- [ ] Real IDM-VTON smoke test passes with `use_refiner=false`.
- [ ] Benchmark evaluation set completes and produces review artifacts.
- [ ] `/health`, `/system`, and `/metrics` return HTTP 200.
- [ ] Artifact cleanup dry-run completes without deleting active jobs.
- [ ] No token, `.env`, checkpoint, model weight, output, temp file, `node_modules`, or `dist` is committed.

Run the automated subset:

```bash
cd /workspace/Project_Phase2/virtual_tryon
bash scripts/release_check.sh
```

To include API smoke testing:

```bash
TRYON_BACKEND_URL=http://127.0.0.1:8000 bash scripts/release_check.sh
```
