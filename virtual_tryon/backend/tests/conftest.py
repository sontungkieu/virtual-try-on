from __future__ import annotations

import os
from io import BytesIO

import pytest
from PIL import Image


os.environ.setdefault("TRYON_ENGINE", "mock")


@pytest.fixture()
def client():
    from app.core.config import clear_settings_cache
    from app.services.container import clear_container_cache

    clear_settings_cache()
    clear_container_cache()
    from app.main import app

    return app


def image_bytes(color: tuple[int, int, int] = (180, 80, 80), size: tuple[int, int] = (256, 384)) -> bytes:
    image = Image.new("RGB", size, color)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture()
def png_file():
    def _make(name: str, color: tuple[int, int, int] = (180, 80, 80)):
        return (name, BytesIO(image_bytes(color)), "image/png")

    return _make
