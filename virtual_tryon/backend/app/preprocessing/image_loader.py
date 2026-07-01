from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import BinaryIO

from PIL import Image, ImageOps

from app.utils.errors import InputValidationError


ALLOWED_IMAGE_MIME = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def validate_mime(
    content_type: str | None,
    filename: str | None = None,
    *,
    allowed_mime_types: set[str] | None = None,
) -> None:
    allowed = allowed_mime_types or ALLOWED_IMAGE_MIME
    if content_type and content_type not in allowed:
        raise InputValidationError(f"Unsupported image MIME type: {content_type}")
    if filename and Path(filename).suffix.lower() not in ALLOWED_SUFFIXES:
        raise InputValidationError(f"Unsupported image file extension: {filename}")


def normalize_image(image: Image.Image, max_side: int) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def load_image_from_bytes(data: bytes, *, max_side: int) -> Image.Image:
    try:
        verifier = Image.open(BytesIO(data))
        verifier.verify()
        image = Image.open(BytesIO(data))
        image.load()
        return normalize_image(image, max_side)
    except Exception as exc:
        raise InputValidationError(f"Invalid image file: {exc}") from exc


def load_image_from_file(file_obj: BinaryIO, *, max_side: int) -> Image.Image:
    return load_image_from_bytes(file_obj.read(), max_side=max_side)


def load_image_from_path(path: str | Path, *, max_side: int) -> Image.Image:
    try:
        image = Image.open(path)
        return normalize_image(image, max_side)
    except Exception as exc:
        raise InputValidationError(f"Could not load image at {path}: {exc}") from exc


def fit_to_canvas(image: Image.Image, width: int, height: int, fill: tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    image = image.convert("RGB")
    scale = min(width / image.width, height / image.height)
    new_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    resized = image.resize(new_size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), fill)
    x = (width - resized.width) // 2
    y = (height - resized.height) // 2
    canvas.paste(resized, (x, y))
    return canvas
