from __future__ import annotations

from PIL import Image

from app.core.config import PreprocessingConfig
from app.preprocessing.agnostic_mask import create_agnostic_mask
from app.preprocessing.garment_segmenter import GarmentSegmenter
from app.preprocessing.mask_utils import mask_area


def test_agnostic_mask_has_expected_outputs():
    image = Image.new("RGB", (256, 384), (180, 180, 180))
    result = create_agnostic_mask(image, "upper_body", PreprocessingConfig())
    assert result.raw_mask.size == image.size
    assert result.dilated_mask.size == image.size
    assert result.soft_mask.size == image.size
    assert mask_area(result.dilated_mask) >= mask_area(result.raw_mask)


def test_innerwear_masks_are_targeted_regions():
    image = Image.new("RGB", (256, 384), (180, 180, 180))
    upper = create_agnostic_mask(image, "upper_body", PreprocessingConfig())
    lower = create_agnostic_mask(image, "lower_body", PreprocessingConfig())
    bra = create_agnostic_mask(image, "women_bra", PreprocessingConfig())
    brief = create_agnostic_mask(image, "women_underwear", PreprocessingConfig())

    assert mask_area(bra.raw_mask) < mask_area(upper.raw_mask)
    assert mask_area(brief.raw_mask) < mask_area(lower.raw_mask)
    assert mask_area(bra.raw_mask) > 0
    assert mask_area(brief.raw_mask) > 0
    assert bra.innerwear_shape_mask is not None
    assert brief.innerwear_shape_mask is not None
    assert bra.mask_source == "geometric_body_fallback"
    assert brief.mask_source == "geometric_body_fallback"


def test_innerwear_mask_uses_foreground_body_when_available():
    image = Image.new("RGB", (256, 384), (245, 245, 245))
    body = Image.new("RGB", (96, 342), (70, 70, 80))
    image.paste(body, (80, 22))

    result = create_agnostic_mask(image, "men_underwear", PreprocessingConfig())

    assert result.mask_source.startswith("foreground_body")
    assert result.body_bbox_xyxy is not None
    x0, y0, x1, y1 = result.body_bbox_xyxy
    assert 70 <= x0 <= 85
    assert 15 <= y0 <= 30
    assert 171 <= x1 <= 186
    assert 355 <= y1 <= 370
    assert result.body_silhouette_mask is not None


def test_outerwear_mask_tracks_shifted_foreground_body():
    left_image = Image.new("RGB", (256, 384), (245, 245, 245))
    right_image = Image.new("RGB", (256, 384), (245, 245, 245))
    body = Image.new("RGB", (78, 330), (70, 70, 80))
    left_image.paste(body, (42, 28))
    right_image.paste(body, (136, 28))

    left = create_agnostic_mask(left_image, "upper_body", PreprocessingConfig())
    right = create_agnostic_mask(right_image, "upper_body", PreprocessingConfig())

    left_bbox = left.raw_mask.getbbox()
    right_bbox = right.raw_mask.getbbox()
    assert left_bbox is not None
    assert right_bbox is not None
    assert left.mask_source.startswith("foreground_body")
    assert right.mask_source.startswith("foreground_body")
    assert right_bbox[0] - left_bbox[0] > 70


def test_innerwear_masks_have_anatomy_specific_extents():
    image = Image.new("RGB", (256, 384), (180, 180, 180))
    bra = create_agnostic_mask(image, "women_bra", PreprocessingConfig())
    underwear = create_agnostic_mask(image, "women_underwear", PreprocessingConfig())

    bra_bbox = bra.raw_mask.getbbox()
    underwear_bbox = underwear.raw_mask.getbbox()
    assert bra_bbox is not None
    assert underwear_bbox is not None
    assert bra_bbox[1] < int(image.height * 0.36)
    assert bra_bbox[3] < int(image.height * 0.50)
    assert underwear_bbox[1] > int(image.height * 0.40)
    assert underwear_bbox[3] < int(image.height * 0.75)


def test_garment_segmenter_returns_normalized_crop():
    image = Image.new("RGB", (128, 128), (255, 255, 255))
    garment = Image.new("RGB", (80, 100), (20, 80, 200))
    image.paste(garment, (24, 14))
    result = GarmentSegmenter().segment(image, (256, 384))
    assert result.normalized_crop.size == (256, 384)
    assert result.cloth_mask.size == image.size
