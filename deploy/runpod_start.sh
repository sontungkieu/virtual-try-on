#!/usr/bin/env bash
set -euo pipefail

cd /workspace/Project_Phase2/virtual_tryon

echo "Start backend:"
echo "  bash scripts/run_backend.sh"
echo
echo "Start frontend in another terminal:"
echo "  VITE_API_BASE_URL=http://127.0.0.1:8000 bash scripts/run_frontend.sh"
echo
echo "Run E2E smoke test:"
echo "  uv run python scripts/e2e_smoke_test.py --api-base http://127.0.0.1:8000 --sample data/eval_set/sample_001 --use-refiner false --timeout 900"
