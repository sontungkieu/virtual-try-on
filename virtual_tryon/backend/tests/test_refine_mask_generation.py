from __future__ import annotations

from PIL import Image, ImageDraw

from app.core.config import load_settings
from app.preprocessing.mask_utils import mask_area
from app.preprocessing.refine_mask import build_refine_masks, select_refine_mask


def _garment_mask() -> Image.Image:
    mask = Image.new("L", (128, 192), 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle((36, 54, 92, 140), fill=255)
    return mask


def test_refine_mask_bundle_generates_expected_masks():
    settings = load_settings()
    person = Image.new("RGB", (128, 192), (180, 180, 180))
    bundle = build_refine_masks(person, _garment_mask(), settings.refinement)

    assert bundle.garment_refine_mask.mode == "L"
    assert bundle.boundary_refine_mask.mode == "L"
    assert bundle.safe_refine_mask.mode == "L"
    assert bundle.garment_overlay.mode == "RGB"
    assert bundle.boundary_overlay.mode == "RGB"
    assert bundle.safe_overlay.mode == "RGB"
    assert mask_area(bundle.garment_refine_mask) > 0
    assert mask_area(bundle.boundary_refine_mask) > 0
    assert bundle.notes


def test_select_refine_mask_modes():
    settings = load_settings()
    person = Image.new("RGB", (128, 192), (180, 180, 180))
    bundle = build_refine_masks(person, _garment_mask(), settings.refinement)

    assert select_refine_mask(bundle, "garment") is bundle.garment_refine_mask
    assert select_refine_mask(bundle, "boundary") is bundle.boundary_refine_mask
    assert select_refine_mask(bundle, "safe") is bundle.safe_refine_mask
    assert select_refine_mask(bundle, "unknown") is bundle.safe_refine_mask
