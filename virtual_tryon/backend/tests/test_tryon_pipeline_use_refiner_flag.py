from __future__ import annotations

from PIL import Image, ImageEnhance

import app.services.tryon_pipeline as pipeline_module
from app.core.config import load_settings
from app.engines.base import RefineResult
from app.services.storage_service import StorageService
from app.services.tryon_pipeline import PipelineRequest, TryOnPipeline


class CountingRefiner:
    name = "counting_refiner"

    def __init__(self) -> None:
        self.calls = 0

    def refine(self, image, mask, prompt, references=None, seed=None):
        self.calls += 1
        base = image.convert("RGB")
        enhanced = ImageEnhance.Contrast(base).enhance(1.08)
        out = base.copy()
        out.paste(enhanced, mask=mask.convert("L"))
        return RefineResult(out, {"engine": self.name, "calls": self.calls})


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


def _request(job_id: str, use_refiner: bool) -> PipelineRequest:
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


def test_use_refiner_false_skips_refiner(monkeypatch, tmp_path):
    settings = _settings(tmp_path)
    storage = StorageService(settings.storage)
    refiner = CountingRefiner()
    monkeypatch.setattr(pipeline_module, "create_refiner", lambda _settings: refiner)

    response = TryOnPipeline(settings, storage).run(_request("refiner_disabled", use_refiner=False))

    assert response.status == "completed"
    assert refiner.calls == 0
    assert not (settings.storage.outputs_dir / "refiner_disabled" / "refined_output.png").exists()


def test_use_refiner_true_runs_refiner(monkeypatch, tmp_path):
    settings = _settings(tmp_path)
    storage = StorageService(settings.storage)
    refiner = CountingRefiner()
    monkeypatch.setattr(pipeline_module, "create_refiner", lambda _settings: refiner)

    response = TryOnPipeline(settings, storage).run(_request("refiner_enabled", use_refiner=True))

    assert response.status == "completed"
    assert refiner.calls == 1
    assert (settings.storage.outputs_dir / "refiner_enabled" / "refined_output.png").exists()
