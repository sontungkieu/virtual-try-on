#!/usr/bin/env bash
set -euo pipefail

if ! command -v node >/dev/null 2>&1; then
  echo "node is not installed. Install Node.js LTS, then rerun:"
  echo "  cd virtual_tryon/frontend && npm install && npm run build"
  exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "npm is not installed. Install Node.js LTS with npm, then rerun:"
  echo "  cd virtual_tryon/frontend && npm install && npm run build"
  exit 1
fi

echo "node=$(node -v)"
echo "npm=$(npm -v)"
echo "Frontend runtime is available."
