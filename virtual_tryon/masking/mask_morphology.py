from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter


TARGET_REGIONS = {"upper", "lower", "dress", "shoes", "hat", "accessory"}


@dataclass(frozen=True)
class MaskPostprocessConfig:
    dilation_px: int = 10
    erosion_px: int = 0
    feather_px: int = 6
    threshold: int = 8
    remove_face_hair: bool = True
    remove_hands: bool = True
    overlay_color: tuple[int, int, int] = (36, 140, 255)


@dataclass(frozen=True)
class MaskArtifacts:
    raw_mask: Image.Image
    processed_mask: Image.Image
    overlay: Image.Image
    mask_area_ratio: float
    warnings: list[str]


def to_l(mask: Image.Image) -> Image.Image:
    return mask.convert("L")


def binarize(mask: Image.Image, threshold: int = 8) -> Image.Image:
    arr = np.array(to_l(mask), dtype=np.uint8)
    out = np.where(arr >= threshold, 255, 0).astype(np.uint8)
    return Image.fromarray(out, mode="L")


def dilate(mask: Image.Image, px: int) -> Image.Image:
    mask = binarize(mask)
    if px <= 0:
        return mask
    return mask.filter(ImageFilter.MaxFilter(px * 2 + 1))


def erode(mask: Image.Image, px: int) -> Image.Image:
    mask = binarize(mask)
    if px <= 0:
        return mask
    return mask.filter(ImageFilter.MinFilter(px * 2 + 1))


def feather(mask: Image.Image, px: int) -> Image.Image:
    if px <= 0:
        return binarize(mask)
    return to_l(mask).filter(ImageFilter.GaussianBlur(px))


def mask_area_ratio(mask: Image.Image) -> float:
    arr = np.array(to_l(mask), dtype=np.uint8)
    return float((arr > 0).sum()) / float(arr.size)


def overlay_mask(
    image: Image.Image,
    mask: Image.Image,
    color: tuple[int, int, int] = (36, 140, 255),
    alpha: float = 0.45,
) -> Image.Image:
    base = image.convert("RGB")
    mask_arr = np.array(to_l(mask), dtype=np.float32) / 255.0
    base_arr = np.array(base, dtype=np.float32)
    color_arr = np.array(color, dtype=np.float32)
    a = (mask_arr * alpha)[..., None]
    out = base_arr * (1.0 - a) + color_arr * a
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")


def _skin_mask(image: Image.Image) -> Image.Image:
    ycbcr = np.array(image.convert("YCbCr"), dtype=np.uint8)
    cb = ycbcr[:, :, 1]
    cr = ycbcr[:, :, 2]
    skin = (cb >= 77) & (cb <= 127) & (cr >= 133) & (cr <= 173)
    return Image.fromarray((skin.astype(np.uint8) * 255), mode="L")


def protected_region_mask(
    image: Image.Image,
    target_region: str,
    *,
    remove_face_hair: bool,
    remove_hands: bool,
) -> Image.Image:
    width, height = image.size
    protected = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(protected)

    if remove_face_hair and target_region not in {"hat", "accessory"}:
        draw.rectangle((0, 0, width, int(height * 0.20)), fill=255)

    if remove_hands and target_region in {"upper", "dress", "accessory"}:
        yy, xx = np.indices((height, width))
        side_region = (
            ((xx < width * 0.34) | (xx > width * 0.66))
            & (yy > height * 0.28)
            & (yy < height * 0.86)
        )
        skin = np.array(_skin_mask(image), dtype=np.uint8) > 0
        hand = Image.fromarray(((skin & side_region).astype(np.uint8) * 255), mode="L")
        hand = hand.filter(ImageFilter.MaxFilter(21)).filter(ImageFilter.GaussianBlur(4))
        protected = ImageChops.lighter(protected, hand)

    return protected


def remove_protected_regions(
    image: Image.Image,
    mask: Image.Image,
    target_region: str,
    config: MaskPostprocessConfig,
    protected_mask: Image.Image | None = None,
) -> Image.Image:
    if protected_mask is not None:
        protected = binarize(protected_mask.resize(image.size, Image.Resampling.NEAREST), config.threshold)
    else:
        protected = protected_region_mask(
            image,
            target_region,
            remove_face_hair=config.remove_face_hair,
            remove_hands=config.remove_hands,
        )
    mask_arr = np.array(to_l(mask), dtype=np.uint8)
    protected_arr = np.array(to_l(protected), dtype=np.uint8)
    mask_arr[protected_arr > 0] = 0
    return Image.fromarray(mask_arr, mode="L")


def postprocess_mask(
    person_image: Image.Image,
    raw_mask: Image.Image,
    target_region: str,
    config: MaskPostprocessConfig | None = None,
    protected_mask: Image.Image | None = None,
) -> MaskArtifacts:
    if target_region not in TARGET_REGIONS:
        raise ValueError(f"Unsupported target_region '{target_region}'. Expected one of {sorted(TARGET_REGIONS)}")
    config = config or MaskPostprocessConfig()
    warnings: list[str] = []

    raw = binarize(raw_mask.resize(person_image.size, Image.Resampling.NEAREST), config.threshold)
    processed = raw
    if config.erosion_px > 0:
        processed = erode(processed, config.erosion_px)
    if config.dilation_px > 0:
        processed = dilate(processed, config.dilation_px)
    processed = remove_protected_regions(person_image, processed, target_region, config, protected_mask)
    if config.feather_px > 0:
        processed = feather(processed, config.feather_px)

    ratio = mask_area_ratio(binarize(processed))
    if ratio < 0.01:
        warnings.append(f"mask too small: area_ratio={ratio:.4f}")
    if ratio > 0.55:
        warnings.append(f"mask too large: area_ratio={ratio:.4f}")

    return MaskArtifacts(
        raw_mask=raw,
        processed_mask=processed,
        overlay=overlay_mask(person_image, processed, config.overlay_color),
        mask_area_ratio=ratio,
        warnings=warnings,
    )
