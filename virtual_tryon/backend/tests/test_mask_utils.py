from __future__ import annotations

from PIL import Image, ImageDraw

from app.preprocessing.mask_utils import bbox_from_mask, dilate, erode, mask_area, merge


def square_mask() -> Image.Image:
    mask = Image.new("L", (64, 64), 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle((24, 24, 40, 40), fill=255)
    return mask


def test_dilation_increases_area():
    mask = square_mask()
    assert mask_area(dilate(mask, 4)) > mask_area(mask)


def test_erosion_decreases_area():
    mask = square_mask()
    assert mask_area(erode(mask, 3)) < mask_area(mask)


def test_merge_works():
    first = square_mask()
    second = Image.new("L", (64, 64), 0)
    draw = ImageDraw.Draw(second)
    draw.rectangle((2, 2, 8, 8), fill=255)
    merged = merge(first, second)
    assert mask_area(merged) > mask_area(first)


def test_bbox_from_mask_works():
    assert bbox_from_mask(square_mask()) == (24, 24, 41, 41)
