from __future__ import annotations

import json

from PIL import Image

from app.core.config import load_settings
from app.services.storage_service import StorageService
from app.services.tryon_pipeline import PipelineRequest, TryOnPipeline


def _settings(tmp_path):
    settings = load_settings()
    settings.pipeline.engine = "mock"
    settings.storage.inputs_dir = tmp_path / "inputs"
    settings.storage.outputs_dir = tmp_path / "outputs"
    settings.storage.temp_dir = tmp_path / "temp"
    settings.repair.enabled = False
    return settings


def _request(job_id: str) -> PipelineRequest:
    return PipelineRequest(
        job_id=job_id,
        person_image=Image.new("RGB", (256, 384), (180, 180, 180)),
        garment_top=Image.new("RGB", (256, 384), (20, 80, 210)),
        garment_bottom=None,
        garment_dress=None,
        category="upper_body",
        prompt="preserve identity and pose",
        use_refiner=False,
        repair_mode=False,
        seed=123,
    )


def test_quality_report_schema_is_always_written(tmp_path):
    settings = _settings(tmp_path)
    storage = StorageService(settings.storage)

    response = TryOnPipeline(settings, storage).run(_request("quality_schema"))

    job_dir = settings.storage.outputs_dir / "quality_schema"
    report_path = job_dir / "quality_report.json"
    assert response.debug.quality_report_url
    assert response.debug.refine_mask_url
    assert report_path.exists()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert set(report) >= {"core", "refined", "final_choice"}
    assert report["final_choice"] in {"core", "refined"}
    assert set(report["core"]) >= {
        "background_preservation_score",
        "face_preservation_score",
        "garment_change_score",
        "over_edit_score",
        "artifact_heuristic_score",
        "needs_refine",
        "notes",
    }
    assert report["core"]["face_preservation_score"] is None
    assert any("face parser" in note.lower() for note in report["core"]["notes"])
    assert report["refined"]["accepted"] is False
