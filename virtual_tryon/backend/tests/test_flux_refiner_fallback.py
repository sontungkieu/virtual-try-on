from __future__ import annotations

import json

from PIL import Image

import app.services.tryon_pipeline as pipeline_module
from app.core.config import load_settings
from app.services.storage_service import StorageService
from app.services.tryon_pipeline import PipelineRequest, TryOnPipeline
from app.utils.errors import ModelUnavailableError


class UnavailableRefiner:
    name = "unavailable_flux_refiner"

    def refine(self, image, mask, prompt, references=None, seed=None):
        raise ModelUnavailableError("FLUX test checkpoint is not available.")


def _settings(tmp_path):
    settings = load_settings()
    settings.pipeline.engine = "mock"
    settings.storage.inputs_dir = tmp_path / "inputs"
    settings.storage.outputs_dir = tmp_path / "outputs"
    settings.storage.temp_dir = tmp_path / "temp"
    settings.refinement.enabled = True
    settings.flux_refiner.enabled = True
    settings.repair.enabled = False
    return settings


def _request(job_id: str, *, use_refiner: bool = True) -> PipelineRequest:
    return PipelineRequest(
        job_id=job_id,
        person_image=Image.new("RGB", (256, 384), (180, 180, 180)),
        garment_top=Image.new("RGB", (256, 384), (20, 80, 210)),
        garment_bottom=None,
        garment_dress=None,
        category="upper_body",
        prompt="preserve identity and pose",
        use_refiner=use_refiner,
        repair_mode=False,
        seed=123,
    )


def test_flux_refiner_unavailable_falls_back_to_core(monkeypatch, tmp_path):
    settings = _settings(tmp_path)
    storage = StorageService(settings.storage)
    monkeypatch.setattr(pipeline_module, "create_refiner", lambda _settings: UnavailableRefiner())

    response = TryOnPipeline(settings, storage).run(_request("flux_fallback"))

    job_dir = settings.storage.outputs_dir / "flux_fallback"
    assert response.status == "completed"
    assert (job_dir / "core_output.png").exists()
    assert (job_dir / "result.png").exists()
    assert not (job_dir / "refined_output.png").exists()
    assert (job_dir / "flux_refiner_error.txt").exists()

    report = json.loads((job_dir / "quality_report.json").read_text(encoding="utf-8"))
    assert report["final_choice"] == "core"
    assert report["refined"]["accepted"] is False
    assert "FLUX test checkpoint" in " ".join(report["refined"]["notes"])
