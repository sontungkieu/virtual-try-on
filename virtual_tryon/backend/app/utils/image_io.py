from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps


def open_rgb(path: str | Path) -> Image.Image:
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def save_image(image: Image.Image, path: str | Path, *, format: str | None = None) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    image.save(resolved, format=format)
    return resolved


def copy_or_save(image: Image.Image, path: str | Path) -> Path:
    return save_image(image.convert("RGB"), path)
