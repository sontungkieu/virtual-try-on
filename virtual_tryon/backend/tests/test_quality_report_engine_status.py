from __future__ import annotations

from PIL import Image, ImageDraw

from app.core.config import load_settings
from app.evaluation.quality_checks import build_quality_report


def _mask() -> Image.Image:
    mask = Image.new("L", (128, 192), 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle((36, 54, 92, 140), fill=255)
    return mask


def test_quality_report_includes_engine_status_and_reason():
    settings = load_settings()
    person = Image.new("RGB", (128, 192), (180, 180, 180))
    core = Image.new("RGB", (128, 192), (120, 140, 180))
    report = build_quality_report(
        person,
        core,
        None,
        _mask(),
        settings.quality,
        engine_status={
            "idm_vton": "success",
            "flux_refiner": "skipped",
            "catvton": "skipped",
            "klein_lora": "skipped",
        },
    )

    assert report["engine_status"]["idm_vton"] == "success"
    assert report["engine_status"]["flux_refiner"] == "skipped"
    assert report["final_choice"] == "core"
    assert report["final_choice_reason"]
    assert "outside_mask_delta" in report["core"]
    assert "blank_or_corrupt_check" in report["core"]
