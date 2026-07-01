from __future__ import annotations

from pathlib import Path

from PIL import Image

from .mask_morphology import binarize


def load_manual_mask(mask_path: str | Path, expected_size: tuple[int, int]) -> Image.Image:
    path = Path(mask_path)
    if not path.exists():
        raise FileNotFoundError(f"mask_image does not exist: {path}")

    mask = Image.open(path).convert("L")
    if mask.size != expected_size:
        mask = mask.resize(expected_size, Image.Resampling.NEAREST)
    mask = binarize(mask)
    if mask.getbbox() is None:
        raise ValueError(f"mask_image is empty after thresholding: {path}")
    return mask
