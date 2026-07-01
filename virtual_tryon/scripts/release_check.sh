#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

failures=0
warnings=0
summary_labels=()
summary_states=()
UV_BIN="${TRYON_UV_BIN:-uv}"
UV_RUN=("$UV_BIN" run --locked --no-sync)
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  UV_RUN+=("--active")
fi

record_summary() {
  summary_labels+=("$1")
  summary_states+=("$2")
}

warn_check() {
  local label="$1"
  local message="$2"
  echo "WARN: $message"
  warnings=$((warnings + 1))
  record_summary "$label" "WARN"
}

run_required() {
  local label="$1"
  shift
  echo
  echo "==> $label"
  if "$@"; then
    echo "PASS: $label"
    record_summary "$label" "PASS"
  else
    echo "FAIL: $label"
    failures=$((failures + 1))
    record_summary "$label" "FAIL"
  fi
}

check_forbidden_files() {
  local tracked_forbidden staged_forbidden
  tracked_forbidden="$({
    git ls-files \
      | grep -E '(^|/)(models|third_party|node_modules|dist|playwright-report|test-results|blob-report)(/|$)|(^|/)data/(outputs|temp)/|(^|/)\.env($|\.)' \
      | grep -Ev '^(virtual_tryon/)?data/(outputs|temp)/\.gitkeep$'
  } || true)"
  staged_forbidden="$({
    git diff --cached --name-only \
      | grep -E '(^|/)(models|third_party|node_modules|dist|playwright-report|test-results|blob-report)(/|$)|(^|/)data/(outputs|temp)/|(^|/)\.env($|\.)' \
      | grep -Ev '^(virtual_tryon/)?data/(outputs|temp)/\.gitkeep$'
  } || true)"

  if [[ -n "$tracked_forbidden" || -n "$staged_forbidden" ]]; then
    [[ -n "$tracked_forbidden" ]] && printf 'Forbidden tracked paths:\n%s\n' "$tracked_forbidden"
    [[ -n "$staged_forbidden" ]] && printf 'Forbidden staged paths:\n%s\n' "$staged_forbidden"
    return 1
  fi
  return 0
}

check_release_docs() {
  local required_docs=(
    docs/final_report.md
    docs/final_demo_checklist.md
    docs/final_presentation_script.md
    docs/release_notes_v0.1.0-rc1.md
    docs/eval_set_expansion_guide.md
    docs/tag_release.md
  )
  local missing=0
  local path
  for path in "${required_docs[@]}"; do
    if [[ ! -s "$path" ]]; then
      echo "Missing or empty: $path"
      missing=1
    fi
  done
  return "$missing"
}

echo "========================================"
echo " Virtual Try-On Release Candidate Check"
echo "========================================"

commit="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "Current commit: $commit"
record_summary "Current commit" "${commit:0:12}"

git_status="$(git status --short 2>/dev/null || true)"
if [[ -n "$git_status" ]]; then
  echo "Git status: dirty"
  printf '%s\n' "$git_status"
  warn_check "Git worktree" "worktree contains tracked or untracked changes; review them before tagging"
else
  echo "Git status: clean"
  record_summary "Git worktree" "PASS"
fi

run_required "Forbidden file safety" check_forbidden_files

ignored_count="$(git status --short --ignored 2>/dev/null | awk '$1 == "!!" {count++} END {print count+0}')"
if (( ignored_count > 0 )); then
  warn_check "Ignored local files" "$ignored_count ignored path(s) are present locally; confirm generated assets stay untracked"
else
  record_summary "Ignored local files" "PASS"
fi

run_required "Release documentation" check_release_docs

if [[ -d backend && -f pyproject.toml && -f uv.lock ]]; then
  if command -v "$UV_BIN" >/dev/null 2>&1; then
    run_required "Backend mock tests" env PYTHONPATH=. TRYON_ENGINE=mock PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "${UV_RUN[@]}" python -m pytest
    run_required "Evaluation set validation" "${UV_RUN[@]}" python scripts/validate_eval_set.py --eval-set data/eval_set
  else
    warn_check "Backend mock tests" "uv executable '$UV_BIN' is unavailable; backend checks skipped"
    warn_check "Evaluation set validation" "uv executable '$UV_BIN' is unavailable; validation skipped"
  fi
else
  warn_check "Backend mock tests" "backend project or uv lockfile is missing; backend checks skipped"
  warn_check "Evaluation set validation" "backend project is missing; validation skipped"
fi

if [[ ! -f frontend/package.json ]]; then
  warn_check "Frontend build" "frontend/package.json is missing; frontend build skipped"
elif command -v npm >/dev/null 2>&1; then
  run_required "Frontend build" npm --prefix frontend run build
else
  warn_check "Frontend build" "npm is unavailable; frontend build skipped"
fi

if [[ -n "${TRYON_BACKEND_URL:-}" ]]; then
  run_required \
    "API E2E smoke" \
    "${UV_RUN[@]}" python scripts/e2e_smoke_test.py \
      --api-base "$TRYON_BACKEND_URL" \
      --sample data/eval_set/sample_001 \
      --use-refiner false \
      --timeout "${TRYON_E2E_TIMEOUT:-900}"
else
  warn_check "Optional API E2E smoke" "TRYON_BACKEND_URL is not set; API E2E smoke skipped"
fi

echo
echo "--------------- Summary ----------------"
for index in "${!summary_labels[@]}"; do
  printf '%-30s %s\n' "${summary_labels[$index]}" "${summary_states[$index]}"
done
echo "----------------------------------------"
echo "Manual evidence still required before tagging:"
echo "  [ ] Real IDM-VTON smoke"
echo "  [ ] Playwright E2E"
echo "  [ ] /health, /system, and /metrics"
echo "  [ ] Benchmark/review evidence as required"

if (( failures > 0 )); then
  echo "FINAL RESULT: FAIL ($failures required check(s) failed, $warnings warning(s))"
  exit 1
fi
echo "FINAL RESULT: PASS ($warnings warning(s))"
