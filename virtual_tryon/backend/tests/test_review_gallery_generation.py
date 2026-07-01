from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_review_gallery import build_gallery  # noqa: E402


def test_review_gallery_generation_handles_skipped_modes(tmp_path):
    summary = {
        "rows": [
            {
                "sample_id": "sample_001",
                "mode": "idm",
                "status": "completed",
                "runtime_seconds": 1.2,
                "output_path": None,
                "input_person_path": None,
                "input_garment_path": None,
                "background_preservation_score": 0.9,
                "face_preservation_score": None,
                "garment_change_score": 0.2,
                "over_edit_score": 0.05,
                "final_choice": "core",
                "notes": "ok",
            },
            {
                "sample_id": "sample_001",
                "mode": "catvton",
                "status": "unavailable",
                "runtime_seconds": 0.0,
                "output_path": None,
                "input_person_path": None,
                "input_garment_path": None,
                "final_choice": None,
                "notes": "checkpoint missing",
            },
        ]
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    index_path = build_gallery(tmp_path)

    assert index_path.exists()
    assert (tmp_path / "manual_ratings.csv").exists()
    html = index_path.read_text(encoding="utf-8")
    assert "catvton" in html
    assert "skipped" in html
