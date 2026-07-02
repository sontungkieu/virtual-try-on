from __future__ import annotations

from fastapi import APIRouter

from app.core.config import get_settings
from app.engines.factory import loaded_engine_state, model_statuses
from app.schemas.tryon import HealthResponse


router = APIRouter(tags=["health"])


def _detect_device(configured_device: str) -> str:
    try:
        import torch

        if configured_device == "cuda" and torch.cuda.is_available():
            return f"cuda:{torch.cuda.get_device_name(0)}"
        return "cpu"
    except Exception:
        return configured_device


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok",
        device=_detect_device(settings.runtime.device),
        models=model_statuses(settings),
        **loaded_engine_state(settings),
    )
