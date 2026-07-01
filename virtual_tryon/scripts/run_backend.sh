#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$REPO_ROOT/.." && pwd)"

source_exported_env() {
  local env_file="$1"
  if [ -f "$env_file" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$env_file"
    set +a
  fi
}

source_exported_env "$PROJECT_ROOT/env.sh"

if [ -n "${VIRTUAL_TRYON_ENV_FILE:-}" ]; then
  source_exported_env "$VIRTUAL_TRYON_ENV_FILE"
else
  source_exported_env "/workspace/secrets/virtual_tryon.env"
fi

cd "$REPO_ROOT/backend"
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
