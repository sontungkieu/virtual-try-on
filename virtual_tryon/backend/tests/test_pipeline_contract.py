from __future__ import annotations

import json

import pytest
from PIL import Image

import app.services.tryon_pipeline as pipeline_module
from app.core.config import load_settings
from app.engines.base import TryOnResult
from app.schemas.tryon import TryOnResponse
from app.services.storage_service import StorageService
from app.services.job_service import JobService
from app.services.tryon_pipeline import PipelineRequest, TryOnPipeline
from app.utils.errors import ModelUnavailableError


def make_request(job_id: str = "test_job") -> PipelineRequest:
    return PipelineRequest(
        job_id=job_id,
        person_image=Image.new("RGB", (256, 384), (180, 180, 180)),
        garment_top=Image.new("RGB", (256, 384), (40, 90, 220)),
        garment_bottom=None,
        garment_dress=None,
        category="upper_body",
        prompt="preserve identity and pose",
        use_refiner=True,
        repair_mode=True,
        seed=123,
    )


def configure_temp_storage(settings, tmp_path):
    settings.storage.inputs_dir = tmp_path / "inputs"
    settings.storage.outputs_dir = tmp_path / "outputs"
    settings.storage.temp_dir = tmp_path / "temp"
    return settings


def test_pipeline_returns_tryon_result(tmp_path):
    settings = configure_temp_storage(load_settings(), tmp_path)
    settings.pipeline.engine = "mock"
    storage = StorageService(settings.storage)
    pipeline = TryOnPipeline(settings, storage)
    response = pipeline.run(make_request())
    assert isinstance(response, TryOnResponse)
    assert response.status == "completed"
    assert response.result_url


def test_pipeline_selects_innerwear_upload_slots(tmp_path):
    settings = configure_temp_storage(load_settings(), tmp_path)
    settings.pipeline.engine = "mock"
    storage = StorageService(settings.storage)
    pipeline = TryOnPipeline(settings, storage)

    bottom_request = make_request("men_innerwear_job")
    bottom_request.category = "men_underwear"
    bottom_request.garment_top = None
    bottom_request.garment_bottom = Image.new("RGB", (256, 384), (80, 40, 220))
    assert pipeline.run(bottom_request).status == "completed"
    bottom_dir = settings.storage.outputs_dir / "men_innerwear_job"
    assert (bottom_dir / "mask_innerwear_shape.png").exists()
    assert (bottom_dir / "mask_metadata.json").exists()

    bra_request = make_request("bra_innerwear_job")
    bra_request.category = "women_bra"
    bra_request.garment_top = Image.new("RGB", (256, 384), (220, 40, 120))
    bra_request.garment_bottom = None
    assert pipeline.run(bra_request).status == "completed"
    bra_dir = settings.storage.outputs_dir / "bra_innerwear_job"
    assert (bra_dir / "mask_innerwear_shape.png").exists()
    assert (bra_dir / "mask_metadata.json").exists()


def test_pipeline_reuses_mask_cache_for_same_person_and_category(tmp_path):
    settings = configure_temp_storage(load_settings(), tmp_path)
    settings.pipeline.engine = "mock"
    storage = StorageService(settings.storage)
    pipeline = TryOnPipeline(settings, storage)

    first = make_request("mask_cache_first")
    first.category = "women_bra"
    first.garment_top = Image.new("RGB", (256, 384), (220, 40, 120))
    assert pipeline.run(first).status == "completed"

    second = make_request("mask_cache_second")
    second.category = "women_bra"
    second.garment_top = Image.new("RGB", (256, 384), (60, 180, 220))
    assert pipeline.run(second).status == "completed"

    first_metadata = json.loads((settings.storage.outputs_dir / "mask_cache_first" / "mask_metadata.json").read_text())
    second_metadata = json.loads((settings.storage.outputs_dir / "mask_cache_second" / "mask_metadata.json").read_text())
    assert first_metadata["cache"]["enabled"] is True
    assert first_metadata["cache"]["hit"] is False
    assert second_metadata["cache"]["hit"] is True
    assert second_metadata["cache"]["key"] == first_metadata["cache"]["key"]
    assert (settings.storage.temp_dir / "mask_cache" / second_metadata["cache"]["key"] / "metadata.json").exists()


def test_missing_model_gives_clear_error(tmp_path):
    settings = configure_temp_storage(load_settings(), tmp_path)
    settings.pipeline.engine = "idm_vton"
    settings.idm_vton.checkpoint_dir = tmp_path / "missing_idm_vton"
    storage = StorageService(settings.storage)
    pipeline = TryOnPipeline(settings, storage)
    with pytest.raises(ModelUnavailableError, match="IDM-VTON checkpoint not found"):
        pipeline.run(make_request("missing_model"))


def test_debug_paths_are_created(tmp_path):
    settings = configure_temp_storage(load_settings(), tmp_path)
    settings.pipeline.engine = "mock"
    storage = StorageService(settings.storage)
    pipeline = TryOnPipeline(settings, storage)
    response = pipeline.run(make_request("debug_job"))
    job_dir = settings.storage.outputs_dir / "debug_job"
    assert (job_dir / "mask_preview.png").exists()
    assert (job_dir / "agnostic.png").exists()
    assert (job_dir / "core_output.png").exists()
    assert response.debug.core_output_url


class GlobalChangingEngine:
    name = "klein_tryon_lora"

    def run(self, inputs):
        return TryOnResult(Image.new("RGB", inputs.person_image.size, (250, 20, 20)), {"engine": self.name})


def test_pipeline_masks_global_engine_output(monkeypatch, tmp_path):
    settings = configure_temp_storage(load_settings(), tmp_path)
    settings.pipeline.engine = "klein_tryon_lora"
    settings.refinement.enabled = False
    settings.repair.enabled = False
    storage = StorageService(settings.storage)
    monkeypatch.setattr(pipeline_module, "create_tryon_engine", lambda _settings: GlobalChangingEngine())

    request = make_request("masked_global_output")
    request.use_refiner = False
    request.repair_mode = False
    response = TryOnPipeline(settings, storage).run(request)

    job_dir = settings.storage.outputs_dir / "masked_global_output"
    person = Image.open(job_dir / "person.png").convert("RGB")
    raw = Image.open(job_dir / "core_output_raw.png").convert("RGB")
    core = Image.open(job_dir / "core_output.png").convert("RGB")
    result = Image.open(job_dir / "result.png").convert("RGB")

    assert response.status == "completed"
    assert raw.getpixel((0, 0)) == (250, 20, 20)
    assert core.getpixel((0, 0)) == person.getpixel((0, 0))
    assert result.getpixel((0, 0)) == person.getpixel((0, 0))


class HybridFakeEngine:
    def __init__(self, color: tuple[int, int, int], name: str) -> None:
        self.color = color
        self.name = name

    def run(self, inputs):
        return TryOnResult(Image.new("RGB", inputs.person_image.size, self.color), {"engine": self.name})


def test_idm_klein_hybrid_uses_idm_delta_for_klein_detail(monkeypatch, tmp_path):
    from app.engines.idm_klein_hybrid_engine import IDMKleinHybridEngine

    settings = configure_temp_storage(load_settings(), tmp_path)
    storage = StorageService(settings.storage)
    engine = IDMKleinHybridEngine(settings)
    engine.idm_engine = HybridFakeEngine((40, 40, 40), "idm_vton")  # type: ignore[assignment]
    engine.klein_engine = HybridFakeEngine((220, 40, 120), "klein_tryon_lora")  # type: ignore[assignment]
    monkeypatch.setattr(pipeline_module, "create_tryon_engine", lambda _settings: engine)

    request = make_request("hybrid_output")
    request.engine_mode = "idm_klein_hybrid"
    request.use_refiner = False
    request.repair_mode = False
    response = TryOnPipeline(settings, storage).run(request)

    job_dir = settings.storage.outputs_dir / "hybrid_output"
    result = Image.open(job_dir / "result.png").convert("RGB")
    delta_mask = Image.open(job_dir / "hybrid_idm_delta_mask.png").convert("L")

    assert response.status == "completed"
    assert (job_dir / "hybrid_idm_base.png").exists()
    assert (job_dir / "hybrid_klein_detail.png").exists()
    assert max(delta_mask.getextrema()) > 0
    assert bytes((220, 40, 120)) in result.tobytes()


def test_pipeline_applies_resolution_and_step_overrides(tmp_path):
    settings = configure_temp_storage(load_settings(), tmp_path)
    settings.pipeline.engine = "mock"
    storage = StorageService(settings.storage)
    pipeline = TryOnPipeline(settings, storage)
    request = make_request("resolution_override")
    request.output_width = 512
    request.output_height = 768
    request.steps = 12
    response = pipeline.run(request)
    result_path = storage.file_path_from_public_url(response.result_url)
    assert Image.open(result_path).size == (512, 768)
    metadata = (settings.storage.outputs_dir / "resolution_override" / "metadata.json").read_text(encoding="utf-8")
    assert '"output_width": 512' in metadata
    assert '"output_height": 768' in metadata
    assert '"steps": 12' in metadata


def test_pipeline_real_engine_fallback_returns_failed_job(tmp_path):
    settings = configure_temp_storage(load_settings(), tmp_path)
    settings.pipeline.engine = "idm_vton"
    settings.idm_vton.checkpoint_dir = tmp_path / "missing_idm_vton"
    storage = StorageService(settings.storage)
    service = JobService(TryOnPipeline(settings, storage))
    response = service.create_tryon_job(make_request("real_engine_missing"))
    assert response.status == "failed"
    assert response.error
    assert "IDM-VTON" in response.error
