from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from app.core.config import MaskExperimentConfig, PreprocessingConfig, load_settings
from app.preprocessing.agnostic_mask import create_agnostic_mask
from app.preprocessing.mask_utils import mask_area
from scripts.run_mask_ablation import (
    DEFAULT_VARIANTS,
    SUMMARY_COLUMNS,
    run_ablation,
    run_flux_local_variant,
)
from scripts.validate_eval_set import validate_sample


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DIR = PROJECT_ROOT / "data" / "eval_set" / "sample_001"


@pytest.fixture(scope="module")
def mock_ablation_output(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("mask_ablation")
    summary = run_ablation(
        sample_dir=SAMPLE_DIR,
        seed=123,
        variants=list(DEFAULT_VARIANTS),
        output_dir=output_dir,
        mock=True,
    )
    return output_dir, summary


def test_upper_body_mask_expansion_flag_off_keeps_default():
    image = Image.new("RGB", (256, 384), (180, 180, 180))
    default = create_agnostic_mask(image, "upper_body", PreprocessingConfig())
    disabled = create_agnostic_mask(
        image,
        "upper_body",
        PreprocessingConfig(),
        MaskExperimentConfig(enabled=False),
    )
    assert default.raw_mask.tobytes() == disabled.raw_mask.tobytes()
    assert default.soft_mask.tobytes() == disabled.soft_mask.tobytes()
    assert disabled.expanded_upper_body_mask is None


def test_upper_body_mask_expansion_increases_lower_torso_area():
    image = Image.new("RGB", (256, 384), (180, 180, 180))
    default = create_agnostic_mask(image, "upper_body", PreprocessingConfig())
    expanded = create_agnostic_mask(
        image,
        "upper_body",
        PreprocessingConfig(),
        MaskExperimentConfig(enabled=True),
    )
    assert mask_area(expanded.raw_mask) > mask_area(default.raw_mask)
    diff = np.array(expanded.diff_upper_body_mask, dtype=np.uint8)
    assert diff[int(image.height * 0.68) :, :].sum() > 0


def test_upper_body_mask_preserves_face_hair_hands_regions():
    image = Image.new("RGB", (256, 384), (180, 180, 180))
    draw = ImageDraw.Draw(image)
    skin_color = (190, 140, 110)
    draw.rectangle((42, 140, 66, 330), fill=skin_color)
    draw.rectangle((190, 140, 214, 330), fill=skin_color)
    default = create_agnostic_mask(image, "upper_body", PreprocessingConfig())
    expanded = create_agnostic_mask(
        image,
        "upper_body",
        PreprocessingConfig(),
        MaskExperimentConfig(enabled=True),
    )
    mask = np.array(expanded.soft_mask, dtype=np.uint8)
    default_mask = np.array(default.soft_mask, dtype=np.uint8)
    face_bottom = int(image.height * 0.22)
    assert np.array_equal(mask[:face_bottom, :], default_mask[:face_bottom, :])
    assert np.array_equal(
        mask[150:320, 46:62],
        default_mask[150:320, 46:62],
    )
    assert np.array_equal(
        mask[150:320, 194:210],
        default_mask[150:320, 194:210],
    )


def test_mask_ablation_outputs_summary_schema(mock_ablation_output):
    output_dir, summary = mock_ablation_output
    assert summary["production_default_changed"] is False
    assert {row["variant"] for row in summary["rows"]} == set(DEFAULT_VARIANTS)
    for row in summary["rows"]:
        assert set(SUMMARY_COLUMNS).issubset(row)
    assert (output_dir / "comparison_grid.png").exists()
    assert (output_dir / "comparison_index.html").exists()
    assert (output_dir / "summary.csv").exists()
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "manual_ratings_mask_ablation.csv").exists()


def test_manual_rating_not_written_to_quality_report(mock_ablation_output):
    output_dir, _ = mock_ablation_output
    quality_path = output_dir / "idm_original" / "quality_report.json"
    report = json.loads(quality_path.read_text(encoding="utf-8"))
    serialized = json.dumps(report)
    assert "identity_1_5" not in serialized
    assert "garment_fidelity_1_5" not in serialized
    assert "original pink shirt remains visible" not in serialized


def test_manual_observations_not_written_to_quality_report(mock_ablation_output):
    output_dir, _ = mock_ablation_output
    quality_path = output_dir / "idm_mask_expanded" / "quality_report.json"
    report = json.loads(quality_path.read_text(encoding="utf-8"))
    serialized = json.dumps(report).lower()
    assert "old_garment_removed_1_5" not in serialized
    assert "pink shirt remains visible" not in serialized
    assert "subjective" not in serialized


def test_flux_local_refine_skips_cleanly_when_unavailable(tmp_path):
    sample, issues = validate_sample(SAMPLE_DIR)
    assert sample is not None, issues
    expanded_dir = tmp_path / "idm_mask_expanded"
    expanded_dir.mkdir()
    for filename, mode in [
        ("core_output.png", "RGB"),
        ("person.png", "RGB"),
        ("garment_normalized.png", "RGB"),
        ("safe_refine_mask.png", "L"),
    ]:
        color = 255 if mode == "L" else (100, 120, 140)
        Image.new(mode, (64, 96), color).save(expanded_dir / filename)

    class UnavailableRefiner:
        def is_available(self) -> bool:
            return False

        def status(self) -> str:
            return "unavailable: test backend"

    row = run_flux_local_variant(
        sample,
        tmp_path,
        load_settings(),
        expanded_dir,
        seed=123,
        refiner_factory=lambda _settings: UnavailableRefiner(),
    )
    assert row["status"] == "skipped"
    assert row["final_choice"] == "core"
    status = json.loads(
        (tmp_path / "idm_mask_expanded_flux_local" / "refiner_status.json").read_text(
            encoding="utf-8"
        )
    )
    assert status["status"] == "skipped"
