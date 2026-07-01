from __future__ import annotations

from fastapi.testclient import TestClient


def test_metrics_returns_prometheus_text(client):
    response = TestClient(client).get("/metrics")
    assert response.status_code == 200
    assert "tryon_jobs_total" in response.text
    assert "tryon_job_runtime_seconds_count" in response.text
    assert "tryon_gpu_memory_used_mb" in response.text
    assert "tryon_queue_size" in response.text
    assert "tryon_artifact_bytes_total" in response.text


def test_system_returns_expected_keys_without_requiring_cuda(client):
    response = TestClient(client).get("/system")
    assert response.status_code == 200
    payload = response.json()
    assert {
        "device",
        "cuda_available",
        "gpu_name",
        "gpu_memory_total_mb",
        "gpu_memory_allocated_mb",
        "python_version",
        "torch_version",
        "commit",
    } <= set(payload)
