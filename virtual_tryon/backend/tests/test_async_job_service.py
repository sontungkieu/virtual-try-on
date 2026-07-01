from __future__ import annotations

import json

from PIL import Image

from app.core.config import load_settings
from app.services.job_service import JobService
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


def test_async_job_service_queued_running_completed(tmp_path):
    settings = _settings(tmp_path)
    storage = StorageService(settings.storage)
    service = JobService(TryOnPipeline(settings, storage), storage)
    request = _request("async_ok")

    queued = service.queue_tryon_job(request)
    assert queued.status == "queued"
    assert queued.current_stage == "queued"
    assert {stage.key: stage.status for stage in queued.stages}["queued"] == "running"

    service.run_queued_job(request)
    completed = service.get_job("async_ok")

    assert completed is not None
    assert completed.status == "completed"
    assert completed.result_url
    assert completed.current_stage == "completed"
    stages = {stage.key: stage for stage in completed.stages}
    assert stages["queued"].status == "completed"
    assert stages["generating"].runtime_seconds is not None
    assert stages["refining"].status == "skipped"
    job_json = settings.storage.outputs_dir / "async_ok" / "job.json"
    assert job_json.exists()
    payload = json.loads(job_json.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["stages"]
    assert payload["started_at"]
    assert payload["finished_at"]


def test_async_job_service_failed_job_has_clean_error(tmp_path):
    settings = _settings(tmp_path)
    settings.pipeline.engine = "idm_vton"
    settings.idm_vton.checkpoint_dir = tmp_path / "missing"
    storage = StorageService(settings.storage)
    service = JobService(TryOnPipeline(settings, storage), storage)
    request = _request("async_fail")

    service.queue_tryon_job(request)
    service.run_queued_job(request)
    failed = service.get_job("async_fail")

    assert failed is not None
    assert failed.status == "failed"
    assert failed.error
    assert "Traceback" not in failed.error
    assert (settings.storage.outputs_dir / "async_fail" / "job.json").exists()
