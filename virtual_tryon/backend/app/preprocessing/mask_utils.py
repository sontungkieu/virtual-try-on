from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter


def to_l_mask(mask: Image.Image) -> Image.Image:
    return mask.convert("L")


def threshold(mask: Image.Image, cutoff: int = 1) -> Image.Image:
    arr = np.array(to_l_mask(mask))
    out = np.where(arr >= cutoff, 255, 0).astype(np.uint8)
    return Image.fromarray(out, mode="L")


def dilate(mask: Image.Image, px: int) -> Image.Image:
    if px <= 0:
        return threshold(mask)
    size = px * 2 + 1
    return threshold(mask).filter(ImageFilter.MaxFilter(size=size))


def erode(mask: Image.Image, px: int) -> Image.Image:
    if px <= 0:
        return threshold(mask)
    size = px * 2 + 1
    return threshold(mask).filter(ImageFilter.MinFilter(size=size))


def blur(mask: Image.Image, radius: int) -> Image.Image:
    if radius <= 0:
        return to_l_mask(mask)
    return to_l_mask(mask).filter(ImageFilter.GaussianBlur(radius=radius))


def invert(mask: Image.Image) -> Image.Image:
    arr = 255 - np.array(to_l_mask(mask))
    return Image.fromarray(arr.astype(np.uint8), mode="L")


def merge(*masks: Image.Image) -> Image.Image:
    if not masks:
        raise ValueError("At least one mask is required.")
    arrays = [np.array(to_l_mask(mask), dtype=np.uint8) for mask in masks]
    merged = np.maximum.reduce(arrays)
    return Image.fromarray(merged, mode="L")


def bbox_from_mask(mask: Image.Image) -> tuple[int, int, int, int] | None:
    arr = np.array(to_l_mask(mask))
    ys, xs = np.where(arr > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def mask_area(mask: Image.Image) -> int:
    return int((np.array(to_l_mask(mask)) > 0).sum())


def overlay_mask_preview(image: Image.Image, mask: Image.Image, color: tuple[int, int, int] = (36, 140, 255)) -> Image.Image:
    base = image.convert("RGB")
    mask_arr = np.array(to_l_mask(mask), dtype=np.float32) / 255.0
    base_arr = np.array(base, dtype=np.float32)
    color_arr = np.array(color, dtype=np.float32)
    alpha = (mask_arr * 0.45)[..., None]
    out = base_arr * (1.0 - alpha) + color_arr * alpha
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")


@dataclass(frozen=True)
class MaskBundle:
    raw_mask: Image.Image
    dilated_mask: Image.Image
    soft_mask: Image.Image
    preview: Image.Image
