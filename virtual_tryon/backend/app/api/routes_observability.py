from __future__ import annotations

import platform
import subprocess

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from app.core.config import PROJECT_ROOT, get_settings
from app.observability.metrics import metrics


router = APIRouter(tags=["observability"])


def _torch_system_info() -> dict:
    info = {
        "cuda_available": False,
        "gpu_name": None,
        "gpu_memory_total_mb": 0.0,
        "gpu_memory_allocated_mb": 0.0,
        "torch_version": "unavailable",
    }
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_total_mb"] = round(
                torch.cuda.get_device_properties(0).total_memory / (1024**2),
                2,
            )
            info["gpu_memory_allocated_mb"] = round(torch.cuda.memory_allocated(0) / (1024**2), 2)
    except Exception:
        pass
    return info


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.strip()
    except Exception:
        return "unknown"


@router.get("/metrics", response_class=PlainTextResponse)
def prometheus_metrics() -> PlainTextResponse:
    gpu_memory = float(_torch_system_info()["gpu_memory_allocated_mb"])
    return PlainTextResponse(metrics.render(gpu_memory), media_type="text/plain; version=0.0.4")


@router.get("/system")
def system_info() -> dict:
    settings = get_settings()
    torch_info = _torch_system_info()
    detected_device = "cuda" if torch_info["cuda_available"] else "cpu"
    return {
        "device": detected_device if settings.runtime.device == "cuda" else settings.runtime.device,
        **torch_info,
        "python_version": platform.python_version(),
        "commit": _git_commit(),
    }
