from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone
from uuid import uuid4

from app.observability.metrics import metrics
from app.schemas.tryon import TryOnStatusResponse
from app.services.artifact_service import write_artifact_manifest
from app.services.storage_service import StorageService
from app.services.tryon_pipeline import PipelineRequest, TryOnPipeline
from app.utils.errors import (
    EngineExecutionError,
    InputValidationError,
    ModelUnavailableError,
    QueueFullError,
    TryOnError,
)


logger = logging.getLogger(__name__)


class JobService:
    def __init__(self, pipeline: TryOnPipeline, storage: StorageService | None = None) -> None:
        self.pipeline = pipeline
        self.storage = storage or pipeline.storage
        self.jobs: dict[str, TryOnStatusResponse] = {}
        self._jobs_guard = threading.RLock()
        self._slots = threading.BoundedSemaphore(max(1, pipeline.settings.api.max_concurrent_jobs))
        self._metrics_finalized_jobs: set[str] = set()

    @staticmethod
    def new_job_id() -> str:
        return uuid4().hex

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @property
    def _engine(self) -> str:
        return self.pipeline.settings.pipeline.engine

    def _job_json_path(self, job_id: str):
        return self.storage.outputs_dir / job_id / "job.json"

    def _queue_size(self) -> int:
        with self._jobs_guard:
            return sum(job.status == "queued" for job in self.jobs.values())

    def _active_size(self) -> int:
        with self._jobs_guard:
            return sum(job.status in {"queued", "running", "cancel_requested"} for job in self.jobs.values())

    def _log_event(
        self,
        job_id: str,
        stage: str,
        status: str,
        *,
        runtime_seconds: float | None = None,
        error_code: str | None = None,
    ) -> None:
        logger.info(
            "job_event %s",
            json.dumps(
                {
                    "job_id": job_id,
                    "engine": self._engine,
                    "stage": stage,
                    "runtime_seconds": runtime_seconds,
                    "status": status,
                    "error_code": error_code,
                },
                separators=(",", ":"),
            ),
        )

    def _save_job(self, job: TryOnStatusResponse) -> TryOnStatusResponse:
        with self._jobs_guard:
            self.jobs[job.job_id] = job
            self.storage.save_json(job.job_id, "job.json", job.model_dump(mode="json"))
            metrics.set_queue_size(self._queue_size())
        return job

    def _load_job_from_disk(self, job_id: str) -> TryOnStatusResponse | None:
        path = self._job_json_path(job_id)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            job = TryOnStatusResponse(**payload)
        except Exception:
            return None
        with self._jobs_guard:
            self.jobs[job_id] = job
        return job

    def _attach_engine_status(self, job: TryOnStatusResponse) -> TryOnStatusResponse:
        quality_report = self.storage.job_dir(job.job_id) / "quality_report.json"
        if quality_report.exists():
            try:
                report = json.loads(quality_report.read_text(encoding="utf-8"))
                if isinstance(report.get("engine_status"), dict):
                    job.engine_status = report["engine_status"]
            except Exception:
                pass
        return job

    def _finalize(self, job: TryOnStatusResponse, runtime_seconds: float) -> TryOnStatusResponse:
        job = self._attach_engine_status(job)
        self._save_job(job)
        _, manifest = write_artifact_manifest(
            job.job_id,
            self.storage.job_dir(job.job_id),
            self.pipeline.settings.storage.public_outputs_prefix,
        )
        job.artifact_manifest = manifest
        self._save_job(job)
        if job.job_id not in self._metrics_finalized_jobs:
            self._metrics_finalized_jobs.add(job.job_id)
            metrics.record_job(job.status, self._engine)
            metrics.observe_runtime(runtime_seconds)
            metrics.add_artifact_bytes(sum(int(item["size_bytes"]) for item in manifest["files"]))
            if job.error_code:
                metrics.record_failure(self._engine, job.error_code)
        self._log_event(
            job.job_id,
            "finalize",
            job.status,
            runtime_seconds=runtime_seconds,
            error_code=job.error_code,
        )
        return job

    @staticmethod
    def _error_code(exc: Exception) -> str:
        if isinstance(exc, ModelUnavailableError):
            return "ENGINE_UNAVAILABLE"
        if isinstance(exc, InputValidationError):
            return "INVALID_IMAGE"
        if isinstance(exc, EngineExecutionError):
            return "ENGINE_EXECUTION_FAILED"
        return "INTERNAL_ERROR"

    def _run_attempts(
        self,
        request: PipelineRequest,
        *,
        created_at: str,
        started_at: str,
    ) -> TryOnStatusResponse:
        max_retries = max(0, self.pipeline.settings.api.max_retries)
        started = time.monotonic()
        last_error: Exception | None = None

        for attempt in range(max_retries + 1):
            current = self.get_job(request.job_id)
            if current and current.cancel_requested:
                finished = self._now()
                return self._finalize(
                    TryOnStatusResponse(
                        job_id=request.job_id,
                        status="cancelled",
                        error="Job cancelled.",
                        error_code="CANCELLED",
                        created_at=created_at,
                        started_at=started_at,
                        finished_at=finished,
                        updated_at=finished,
                        cancel_requested=True,
                        retry_count=attempt,
                    ),
                    time.monotonic() - started,
                )
            try:
                response = self.pipeline.run(request)
                runtime = time.monotonic() - started
                current = self.get_job(request.job_id)
                if current and current.cancel_requested:
                    finished = self._now()
                    return self._finalize(
                        TryOnStatusResponse(
                            job_id=request.job_id,
                            status="cancelled",
                            error="Job cancelled.",
                            error_code="CANCELLED",
                            created_at=created_at,
                            started_at=started_at,
                            finished_at=finished,
                            updated_at=finished,
                            cancel_requested=True,
                            retry_count=attempt,
                        ),
                        runtime,
                    )
                if runtime > self.pipeline.settings.api.max_job_runtime_seconds:
                    finished = self._now()
                    return self._finalize(
                        TryOnStatusResponse(
                            job_id=request.job_id,
                            status="failed",
                            error=(
                                "Job exceeded the configured runtime limit of "
                                f"{self.pipeline.settings.api.max_job_runtime_seconds} seconds."
                            ),
                            error_code="TIMEOUT",
                            created_at=created_at,
                            started_at=started_at,
                            finished_at=finished,
                            updated_at=finished,
                            retry_count=attempt,
                        ),
                        runtime,
                    )
                now = self._now()
                payload = response.model_dump()
                return self._finalize(
                    TryOnStatusResponse(
                        **payload,
                        created_at=created_at,
                        started_at=started_at,
                        finished_at=now,
                        updated_at=now,
                        retry_count=attempt,
                    ),
                    runtime,
                )
            except (TryOnError, InputValidationError) as exc:
                last_error = exc
                self._log_event(
                    request.job_id,
                    f"attempt_{attempt + 1}",
                    "retrying" if attempt < max_retries else "failed",
                    runtime_seconds=time.monotonic() - started,
                    error_code=self._error_code(exc),
                )
                if attempt < max_retries:
                    continue
            except Exception as exc:
                logger.exception("Unexpected try-on job failure for %s", request.job_id)
                last_error = exc
                break

        runtime = time.monotonic() - started
        error_code = self._error_code(last_error or RuntimeError("Unknown job failure"))
        message = (
            str(last_error)
            if isinstance(last_error, TryOnError)
            else "The try-on job failed unexpectedly. Check server logs for details."
        )
        finished = self._now()
        return self._finalize(
            TryOnStatusResponse(
                job_id=request.job_id,
                status="failed",
                error=message,
                error_code=error_code,
                created_at=created_at,
                started_at=started_at,
                finished_at=finished,
                updated_at=finished,
                retry_count=max_retries,
            ),
            runtime,
        )

    def _acquire_slot(self) -> None:
        reject = self.pipeline.settings.api.queue_policy == "reject"
        if not self._slots.acquire(blocking=not reject):
            raise QueueFullError()

    def create_tryon_job(self, request: PipelineRequest) -> TryOnStatusResponse:
        self._acquire_slot()
        now = self._now()
        running = TryOnStatusResponse(
            job_id=request.job_id,
            status="running",
            created_at=now,
            started_at=now,
            updated_at=now,
        )
        self._save_job(running)
        self._log_event(request.job_id, "pipeline", "running")
        try:
            return self._run_attempts(request, created_at=now, started_at=now)
        finally:
            self._slots.release()

    def queue_tryon_job(self, request: PipelineRequest) -> TryOnStatusResponse:
        settings = self.pipeline.settings.api
        if settings.queue_policy == "reject" and self._active_size() >= settings.max_concurrent_jobs:
            raise QueueFullError()
        now = self._now()
        queued = TryOnStatusResponse(job_id=request.job_id, status="queued", created_at=now, updated_at=now)
        self._log_event(request.job_id, "queue", "queued")
        return self._save_job(queued)

    def run_queued_job(self, request: PipelineRequest) -> None:
        job = self.get_job(request.job_id)
        if job is None or job.status == "cancelled":
            return
        if job.cancel_requested:
            self.cancel_job(request.job_id)
            return

        self._slots.acquire()
        try:
            job = self.get_job(request.job_id) or job
            if job.cancel_requested or job.status == "cancelled":
                self.cancel_job(request.job_id)
                return
            job.status = "running"
            job.started_at = self._now()
            job.updated_at = job.started_at
            self._save_job(job)
            self._log_event(request.job_id, "pipeline", "running")
            self._run_attempts(
                request,
                created_at=job.created_at or job.started_at,
                started_at=job.started_at,
            )
        finally:
            self._slots.release()

    def cancel_job(self, job_id: str) -> TryOnStatusResponse | None:
        job = self.get_job(job_id)
        if job is None:
            return None
        if job.status == "queued":
            job.status = "cancelled"
            job.cancel_requested = True
            job.error = "Job cancelled before start."
            job.error_code = "CANCELLED"
            job.finished_at = self._now()
            job.updated_at = job.finished_at
            return self._finalize(job, 0.0)
        if job.status == "running":
            job.status = "cancel_requested"
            job.cancel_requested = True
            job.updated_at = self._now()
            self._log_event(job_id, "cancel", "cancel_requested")
        return self._save_job(job)

    def get_job(self, job_id: str) -> TryOnStatusResponse | None:
        with self._jobs_guard:
            job = self.jobs.get(job_id)
        if job is not None:
            return job
        return self._load_job_from_disk(job_id)
