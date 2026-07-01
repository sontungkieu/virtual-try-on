from __future__ import annotations

from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings


def test_cors_origins_are_loaded_from_config(client):
    settings = get_settings()
    middleware = next(item for item in client.user_middleware if item.cls is CORSMiddleware)
    assert middleware.kwargs["allow_origins"] == settings.api.cors_origins
    assert "*" not in middleware.kwargs["allow_origins"]
