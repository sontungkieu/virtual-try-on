from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from validate_eval_set import discover_eval_samples  # noqa: E402


def _write_sample(root: Path, sample_id: str = "sample_001") -> Path:
    sample_dir = root / sample_id
    sample_dir.mkdir(parents=True)
    Image.new("RGB", (64, 96), (180, 180, 180)).save(sample_dir / "person.jpg")
    Image.new("RGB", (64, 96), (20, 80, 210)).save(sample_dir / "garment_top.jpg")
    (sample_dir / "metadata.json").write_text(
        json.dumps(
            {
                "sample_id": sample_id,
                "category": "upper_body",
                "difficulty": "easy",
                "expected_focus": ["identity", "garment_texture"],
                "notes": "",
            }
        ),
        encoding="utf-8",
    )
    return sample_dir


def test_eval_set_validation_accepts_valid_sample(tmp_path):
    _write_sample(tmp_path)
    samples, issues = discover_eval_samples(tmp_path)
    assert not issues
    assert len(samples) == 1
    assert samples[0].sample_id == "sample_001"
    assert samples[0].garment_top is not None


def test_eval_set_validation_warns_on_empty_folder(tmp_path):
    samples, issues = discover_eval_samples(tmp_path)
    assert samples == []
    assert issues
    assert "empty" in issues[0]["errors"][0]
