#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if command -v tmux >/dev/null 2>&1; then
  session="virtual_tryon_demo"
  tmux new-session -d -s "$session" "cd '$REPO_ROOT' && bash scripts/run_backend.sh"
  tmux split-window -h -t "$session" "cd '$REPO_ROOT' && bash scripts/run_frontend.sh"
  tmux attach -t "$session"
else
  echo "tmux is not installed. Run these in two terminals:"
  echo "  cd $REPO_ROOT && bash scripts/run_backend.sh"
  echo "  cd $REPO_ROOT && VITE_API_BASE_URL=http://127.0.0.1:8000 bash scripts/run_frontend.sh"
fi
