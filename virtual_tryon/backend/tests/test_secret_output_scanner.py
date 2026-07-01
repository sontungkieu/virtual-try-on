from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_secret_scanner_passes_clean_output(tmp_path):
    (tmp_path / "response_sanitized.json").write_text('{"status":"ok"}', encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "scan_outputs_for_secrets.py"),
            "--path",
            str(tmp_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "No secret-like strings found" in completed.stdout


def test_secret_scanner_reports_without_printing_secret(tmp_path):
    secret = "fal-secret-value-that-should-not-print"
    (tmp_path / "response_sanitized.json").write_text(f'{{"FAL_KEY":"{secret}"}}', encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "scan_outputs_for_secrets.py"),
            "--path",
            str(tmp_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 1
    assert "response_sanitized.json:1" in completed.stdout
    assert "FAL_KEY" in completed.stdout
    assert secret not in completed.stdout
