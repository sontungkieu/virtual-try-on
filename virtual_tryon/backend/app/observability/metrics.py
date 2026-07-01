from __future__ import annotations

import threading
from collections import defaultdict


class MetricsRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs_total: dict[tuple[str, str], int] = defaultdict(int)
        self._engine_failures_total: dict[tuple[str, str], int] = defaultdict(int)
        self._runtime_count = 0
        self._runtime_sum = 0.0
        self._queue_size = 0
        self._artifact_bytes_total = 0

    def reset(self) -> None:
        with self._lock:
            self._jobs_total.clear()
            self._engine_failures_total.clear()
            self._runtime_count = 0
            self._runtime_sum = 0.0
            self._queue_size = 0
            self._artifact_bytes_total = 0

    def record_job(self, status: str, engine: str) -> None:
        with self._lock:
            self._jobs_total[(status, engine)] += 1

    def record_failure(self, engine: str, error_code: str) -> None:
        with self._lock:
            self._engine_failures_total[(engine, error_code)] += 1

    def observe_runtime(self, runtime_seconds: float) -> None:
        with self._lock:
            self._runtime_count += 1
            self._runtime_sum += max(0.0, runtime_seconds)

    def set_queue_size(self, size: int) -> None:
        with self._lock:
            self._queue_size = max(0, size)

    def add_artifact_bytes(self, size_bytes: int) -> None:
        with self._lock:
            self._artifact_bytes_total += max(0, size_bytes)

    def render(self, gpu_memory_used_mb: float = 0.0) -> str:
        with self._lock:
            jobs = dict(self._jobs_total)
            failures = dict(self._engine_failures_total)
            runtime_count = self._runtime_count
            runtime_sum = self._runtime_sum
            queue_size = self._queue_size
            artifact_bytes = self._artifact_bytes_total

        lines = [
            "# HELP tryon_jobs_total Total try-on jobs by terminal status and engine.",
            "# TYPE tryon_jobs_total counter",
        ]
        for (status, engine), value in sorted(jobs.items()):
            lines.append(f'tryon_jobs_total{{status="{status}",engine="{engine}"}} {value}')
        lines.extend(
            [
                "# HELP tryon_job_runtime_seconds Job runtime summary.",
                "# TYPE tryon_job_runtime_seconds summary",
                f"tryon_job_runtime_seconds_count {runtime_count}",
                f"tryon_job_runtime_seconds_sum {runtime_sum:.6f}",
                "# HELP tryon_engine_failures_total Engine failures by error code.",
                "# TYPE tryon_engine_failures_total counter",
            ]
        )
        for (engine, error_code), value in sorted(failures.items()):
            lines.append(
                f'tryon_engine_failures_total{{engine="{engine}",error_code="{error_code}"}} {value}'
            )
        lines.extend(
            [
                "# HELP tryon_gpu_memory_used_mb GPU memory currently allocated by PyTorch.",
                "# TYPE tryon_gpu_memory_used_mb gauge",
                f"tryon_gpu_memory_used_mb {max(0.0, gpu_memory_used_mb):.3f}",
                "# HELP tryon_queue_size Jobs waiting to run.",
                "# TYPE tryon_queue_size gauge",
                f"tryon_queue_size {queue_size}",
                "# HELP tryon_artifact_bytes_total Bytes written to public job artifacts.",
                "# TYPE tryon_artifact_bytes_total counter",
                f"tryon_artifact_bytes_total {artifact_bytes}",
            ]
        )
        return "\n".join(lines) + "\n"


metrics = MetricsRegistry()
