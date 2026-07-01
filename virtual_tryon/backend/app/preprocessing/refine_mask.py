from __future__ import annotations

from dataclasses import dataclass, field

from PIL import Image
import numpy as np

from app.core.config import RefinementConfig
from app.preprocessing.mask_utils import blur, dilate, erode, overlay_mask_preview, to_l_mask


@dataclass(frozen=True)
class RefineMaskBundle:
    garment_refine_mask: Image.Image
    boundary_refine_mask: Image.Image
    safe_refine_mask: Image.Image
    garment_overlay: Image.Image
    boundary_overlay: Image.Image
    safe_overlay: Image.Image
    notes: list[str] = field(default_factory=list)


def _subtract_masks(outer: Image.Image, inner: Image.Image) -> Image.Image:
    outer_l = to_l_mask(outer)
    inner_l = to_l_mask(inner).resize(outer_l.size)
    result = np.maximum(0, np.array(outer_l, dtype=np.int16) - np.array(inner_l, dtype=np.int16))
    return Image.fromarray(result.astype(np.uint8), mode="L")


def build_refine_masks(
    person_image: Image.Image,
    garment_mask: Image.Image,
    config: RefinementConfig,
) -> RefineMaskBundle:
    notes: list[str] = []
    garment = blur(to_l_mask(garment_mask), config.soft_blur_radius)

    dilated = dilate(garment_mask, config.boundary_dilation_px)
    eroded = erode(garment_mask, config.boundary_erosion_px)
    boundary = blur(_subtract_masks(dilated, eroded), config.soft_blur_radius)

    safe = garment
    if config.preserve_face or config.preserve_hands or config.preserve_hair:
        notes.append("Safe refine mask is using garment-mask fallback because face/hair/hands parser is not wired yet.")

    return RefineMaskBundle(
        garment_refine_mask=garment,
        boundary_refine_mask=boundary,
        safe_refine_mask=safe,
        garment_overlay=overlay_mask_preview(person_image, garment, (0, 130, 220)),
        boundary_overlay=overlay_mask_preview(person_image, boundary, (255, 122, 38)),
        safe_overlay=overlay_mask_preview(person_image, safe, (21, 153, 105)),
        notes=notes,
    )


def select_refine_mask(bundle: RefineMaskBundle, mode: str) -> Image.Image:
    if mode == "boundary":
        return bundle.boundary_refine_mask
    if mode == "garment":
        return bundle.garment_refine_mask
    return bundle.safe_refine_mask
