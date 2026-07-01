#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$REPO_ROOT/.." && pwd)"

if [ -f "$PROJECT_ROOT/env.sh" ]; then
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/env.sh"
fi

cd "$REPO_ROOT/backend"
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
