from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _write_eval_sample(root: Path) -> Path:
    sample_dir = root / "sample_001"
    sample_dir.mkdir(parents=True)
    Image.new("RGB", (128, 192), (180, 180, 180)).save(sample_dir / "person.jpg")
    Image.new("RGB", (128, 192), (20, 80, 210)).save(sample_dir / "garment_top.jpg")
    (sample_dir / "metadata.json").write_text(
        json.dumps(
            {
                "sample_id": "sample_001",
                "category": "upper_body",
                "difficulty": "easy",
                "expected_focus": ["identity"],
                "notes": "",
            }
        ),
        encoding="utf-8",
    )
    return root


def test_benchmark_skips_unavailable_baseline_engines(tmp_path):
    eval_set = _write_eval_sample(tmp_path / "eval_set")
    output_dir = tmp_path / "benchmark_test"
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "benchmark_pipeline.py"),
            "--eval-set",
            str(eval_set),
            "--modes",
            "idm,catvton,klein_lora",
            "--limit",
            "1",
            "--output",
            str(output_dir),
            "--mock",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    rows = summary["rows"]
    assert len(rows) == 3
    by_mode = {row["mode"]: row for row in rows}
    assert by_mode["idm"]["status"] == "completed"
    assert by_mode["catvton"]["status"] == "unavailable"
    assert by_mode["klein_lora"]["status"] == "unavailable"
    assert (output_dir / "index.html").exists()
    assert (output_dir / "manual_ratings.csv").exists()


def test_benchmark_klein_lora_skips_when_unavailable(tmp_path, monkeypatch):
    monkeypatch.delenv("FAL_KEY", raising=False)
    eval_set = _write_eval_sample(tmp_path / "eval_set")
    output_dir = tmp_path / "benchmark_klein"
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "benchmark_pipeline.py"),
            "--eval-set",
            str(eval_set),
            "--modes",
            "klein_lora",
            "--limit",
            "1",
            "--output",
            str(output_dir),
            "--mock",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    row = summary["rows"][0]
    assert row["mode"] == "klein_lora"
    assert row["status"] == "unavailable"
    assert row["error_code"] == "ENGINE_UNAVAILABLE"
    assert (output_dir / "sample_001" / "klein_lora" / "status.json").exists()
