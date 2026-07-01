from __future__ import annotations

from PIL import Image
import pytest

from app.core.config import load_settings
from app.services.job_service import JobService
from app.services.storage_service import StorageService
from app.services.tryon_pipeline import PipelineRequest, TryOnPipeline
from app.utils.errors import QueueFullError


def _service(tmp_path) -> JobService:
    settings = load_settings()
    settings.pipeline.engine = "mock"
    settings.storage.inputs_dir = tmp_path / "inputs"
    settings.storage.outputs_dir = tmp_path / "outputs"
    settings.storage.temp_dir = tmp_path / "temp"
    settings.repair.enabled = False
    storage = StorageService(settings.storage)
    return JobService(TryOnPipeline(settings, storage), storage)


def _request(job_id: str) -> PipelineRequest:
    return PipelineRequest(
        job_id=job_id,
        person_image=Image.new("RGB", (128, 192), "gray"),
        garment_top=Image.new("RGB", (128, 192), "blue"),
        garment_bottom=None,
        garment_dress=None,
        category="upper_body",
        prompt=None,
        use_refiner=False,
        repair_mode=False,
        seed=1,
    )


def test_queue_full_reject_policy(tmp_path):
    service = _service(tmp_path)
    service.pipeline.settings.api.queue_policy = "reject"
    service.pipeline.settings.api.max_concurrent_jobs = 1
    service.queue_tryon_job(_request("first"))
    with pytest.raises(QueueFullError):
        service.queue_tryon_job(_request("second"))


def test_queued_job_can_be_cancelled(tmp_path):
    service = _service(tmp_path)
    service.queue_tryon_job(_request("cancel-me"))
    cancelled = service.cancel_job("cancel-me")
    assert cancelled is not None
    assert cancelled.status == "cancelled"
    assert cancelled.error_code == "CANCELLED"
    assert cancelled.artifact_manifest is not None


def test_job_timeout_is_marked_cleanly(tmp_path):
    service = _service(tmp_path)
    service.pipeline.settings.api.max_job_runtime_seconds = 0
    completed = service.create_tryon_job(_request("timeout"))
    assert completed.status == "failed"
    assert completed.error_code == "TIMEOUT"
    assert "runtime limit" in (completed.error or "")
    assert completed.artifact_manifest is not None
