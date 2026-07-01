from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

from app.core.config import MaskExperimentConfig, PreprocessingConfig
from app.preprocessing.mask_utils import MaskBundle, blur, dilate, mask_area, overlay_mask_preview
from app.schemas.tryon import INNERWEAR_BOTTOM_CATEGORIES, INNERWEAR_TOP_CATEGORIES, TryOnCategory


UPPER_BODY_LIKE_CATEGORIES = {"upper_body", *INNERWEAR_TOP_CATEGORIES}
LOWER_BODY_LIKE_CATEGORIES = {"lower_body", *INNERWEAR_BOTTOM_CATEGORIES}
INNERWEAR_CATEGORIES = {*INNERWEAR_BOTTOM_CATEGORIES, *INNERWEAR_TOP_CATEGORIES}


@dataclass(frozen=True)
class AgnosticMaskResult:
    raw_mask: Image.Image
    dilated_mask: Image.Image
    soft_mask: Image.Image
    preview: Image.Image
    agnostic_image: Image.Image
    mask_source: str = "geometric_bbox"
    mask_warnings: tuple[str, ...] = ()
    body_bbox_xyxy: tuple[int, int, int, int] | None = None
    body_silhouette_mask: Image.Image | None = None
    innerwear_shape_mask: Image.Image | None = None
    original_upper_body_mask: Image.Image | None = None
    expanded_upper_body_mask: Image.Image | None = None
    diff_upper_body_mask: Image.Image | None = None
    original_upper_body_overlay: Image.Image | None = None
    expanded_upper_body_overlay: Image.Image | None = None
    diff_upper_body_overlay: Image.Image | None = None


def _region_bbox(width: int, height: int, category: TryOnCategory) -> tuple[int, int, int, int]:
    if category == "upper_body":
        return int(width * 0.16), int(height * 0.22), int(width * 0.84), int(height * 0.72)
    if category == "women_bra":
        return int(width * 0.22), int(height * 0.25), int(width * 0.78), int(height * 0.52)
    if category == "lower_body":
        return int(width * 0.20), int(height * 0.50), int(width * 0.80), int(height * 0.95)
    if category == "men_underwear":
        return int(width * 0.24), int(height * 0.46), int(width * 0.76), int(height * 0.72)
    if category == "women_underwear":
        return int(width * 0.25), int(height * 0.47), int(width * 0.75), int(height * 0.70)
    if category == "dress":
        return int(width * 0.16), int(height * 0.22), int(width * 0.84), int(height * 0.95)
    return int(width * 0.15), int(height * 0.20), int(width * 0.85), int(height * 0.95)


def _expand_bbox(
    bbox: tuple[int, int, int, int],
    size: tuple[int, int],
    *,
    x_pad: int = 0,
    y_pad: int = 0,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    width, height = size
    return (
        max(0, x0 - x_pad),
        max(0, y0 - y_pad),
        min(width, x1 + x_pad),
        min(height, y1 + y_pad),
    )


def _fallback_body_bbox(size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = size
    return (
        int(width * 0.24),
        int(height * 0.08),
        int(width * 0.76),
        int(height * 0.96),
    )


def _bbox_from_foreground(
    foreground: np.ndarray,
    size: tuple[int, int],
    source: str,
    warnings: list[str],
) -> tuple[tuple[int, int, int, int] | None, Image.Image | None]:
    width, height = size
    area_ratio = float(foreground.mean())
    if area_ratio < 0.03 or area_ratio > 0.75:
        warnings.append(f"{source} rejected: area_ratio={area_ratio:.3f}.")
        return None, None

    silhouette = Image.fromarray((foreground.astype(np.uint8) * 255), mode="L")
    silhouette = silhouette.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(3))
    foreground = np.array(silhouette, dtype=np.uint8) > 0

    row_min = max(3, int(width * 0.035))
    col_min = max(3, int(height * 0.025))
    ys = np.where(foreground.sum(axis=1) >= row_min)[0]
    xs = np.where(foreground.sum(axis=0) >= col_min)[0]
    if len(xs) == 0 or len(ys) == 0:
        warnings.append(f"{source} rejected: empty projected bbox.")
        return None, None

    bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    bx0, by0, bx1, by1 = bbox
    bw = bx1 - bx0
    bh = by1 - by0
    edge_margin = max(2, int(width * 0.02))
    if bx0 <= edge_margin or bx1 >= width - edge_margin:
        warnings.append(f"{source} rejected: bbox touches horizontal image edge: bbox={bbox}.")
        return None, None
    if bh < height * 0.42 or bw < width * 0.14 or bw > width * 0.94:
        warnings.append(f"{source} rejected: bbox={bbox}.")
        return None, None

    bbox = _expand_bbox(
        bbox,
        size,
        x_pad=max(3, width // 80),
        y_pad=max(3, height // 120),
    )
    return bbox, silhouette


def _adaptive_rowcol_foreground(rgb: np.ndarray, border_px: int) -> np.ndarray:
    height, width, _ = rgb.shape
    left_right = np.concatenate([rgb[:, :border_px, :], rgb[:, -border_px:, :]], axis=1)
    top_bottom = np.concatenate([rgb[:border_px, :, :], rgb[-border_px:, :, :]], axis=0)
    row_bg = np.median(left_right, axis=1)
    col_bg = np.median(top_bottom, axis=0)

    row_dist = np.sqrt(np.square(rgb - row_bg[:, None, :]).sum(axis=2))
    col_dist = np.sqrt(np.square(rgb - col_bg[None, :, :]).sum(axis=2))
    row_border_dist = np.sqrt(np.square(left_right - row_bg[:, None, :]).sum(axis=2)).reshape(-1)
    col_border_dist = np.sqrt(np.square(top_bottom - col_bg[None, :, :]).sum(axis=2)).reshape(-1)
    row_cutoff = max(20.0, float(np.percentile(row_border_dist, 90)) + 10.0)
    col_cutoff = max(20.0, float(np.percentile(col_border_dist, 90)) + 10.0)

    strict = (row_dist > row_cutoff) & (col_dist > col_cutoff)
    if strict.mean() >= 0.03:
        return strict
    return (row_dist > row_cutoff) | (col_dist > col_cutoff)


def _center_saliency_foreground(rgb: np.ndarray) -> np.ndarray:
    height, width, _ = rgb.shape
    gray = rgb.mean(axis=2)
    saturation = rgb.max(axis=2) - rgb.min(axis=2)
    border_px = max(2, min(width, height) // 28)
    border_gray = np.concatenate(
        [
            gray[:border_px, :].reshape(-1),
            gray[-border_px:, :].reshape(-1),
            gray[:, :border_px].reshape(-1),
            gray[:, -border_px:].reshape(-1),
        ]
    )
    contrast = np.abs(gray - float(np.median(border_gray)))

    yy, xx = np.indices((height, width))
    x_weight = 1.0 - np.clip(np.abs(xx - width * 0.5) / max(1.0, width * 0.48), 0.0, 1.0)
    y_weight = 1.0 - np.clip(np.abs(yy - height * 0.52) / max(1.0, height * 0.55), 0.0, 1.0)
    center_weight = 0.35 + 0.65 * x_weight * y_weight
    score = (contrast + saturation * 0.45) * center_weight
    threshold = max(float(np.percentile(score, 82)), float(score.mean() + score.std() * 0.35), 18.0)
    return score > threshold


def _estimate_body_from_foreground(
    person_image: Image.Image,
) -> tuple[tuple[int, int, int, int], Image.Image | None, str, list[str]]:
    """Estimate a coarse person envelope from border/background contrast.

    This is not semantic parsing. It only keeps innerwear geometry anchored to
    the detected person instead of the whole canvas when the photo background is
    simple enough. Complex backgrounds fall back to the conservative body bbox.
    """

    width, height = person_image.size
    fallback = _fallback_body_bbox(person_image.size)
    warnings: list[str] = []
    if width < 32 or height < 48:
        return fallback, None, "geometric_body_fallback", ["image too small for foreground body estimation."]

    rgb = np.array(person_image.convert("RGB"), dtype=np.int16)
    border_px = max(2, min(width, height) // 32)
    border = np.concatenate(
        [
            rgb[:border_px, :, :].reshape(-1, 3),
            rgb[-border_px:, :, :].reshape(-1, 3),
            rgb[:, :border_px, :].reshape(-1, 3),
            rgb[:, -border_px:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    bg = np.median(border, axis=0)
    dist = np.sqrt(np.square(rgb - bg).sum(axis=2))
    border_dist = np.sqrt(np.square(border - bg).sum(axis=1))
    cutoff = max(24.0, float(np.percentile(border_dist, 92)) + 12.0)
    foreground = dist > cutoff
    bbox, silhouette = _bbox_from_foreground(foreground, person_image.size, "global foreground body estimate", warnings)
    if bbox is not None:
        return bbox, silhouette, "foreground_body", warnings

    adaptive = _adaptive_rowcol_foreground(rgb, border_px)
    bbox, silhouette = _bbox_from_foreground(adaptive, person_image.size, "adaptive row/column body estimate", warnings)
    if bbox is not None:
        return bbox, silhouette, "adaptive_foreground_body", warnings

    saliency = _center_saliency_foreground(rgb)
    bbox, silhouette = _bbox_from_foreground(saliency, person_image.size, "center saliency body estimate", warnings)
    if bbox is not None:
        return bbox, silhouette, "center_saliency_body", warnings

    return fallback, None, "geometric_body_fallback", warnings


def _body_region_bbox(
    size: tuple[int, int],
    category: TryOnCategory,
    body: tuple[int, int, int, int] | None,
) -> tuple[int, int, int, int]:
    if body is None:
        return _region_bbox(*size, category)
    width, height = size
    bx0, by0, bx1, by1 = body
    bw = max(1, bx1 - bx0)
    bh = max(1, by1 - by0)

    if category == "upper_body":
        bbox = (bx0 + int(bw * 0.02), by0 + int(bh * 0.16), bx1 - int(bw * 0.02), by0 + int(bh * 0.72))
    elif category == "lower_body":
        bbox = (bx0 + int(bw * 0.10), by0 + int(bh * 0.48), bx1 - int(bw * 0.10), by0 + int(bh * 0.98))
    elif category == "dress":
        bbox = (bx0 + int(bw * 0.02), by0 + int(bh * 0.18), bx1 - int(bw * 0.02), by0 + int(bh * 0.98))
    elif category == "full_outfit":
        bbox = (bx0, by0 + int(bh * 0.15), bx1, by0 + int(bh * 0.98))
    else:
        bbox = _region_bbox(width, height, category)
    x0, y0, x1, y1 = bbox
    return max(0, x0), max(0, y0), min(width, x1), min(height, y1)


def _create_outerwear_raw_mask(
    person_image: Image.Image,
    category: TryOnCategory,
    config: PreprocessingConfig,
) -> tuple[Image.Image, Image.Image | None, str, tuple[int, int, int, int] | None, tuple[str, ...]]:
    body_bbox, silhouette, source, warnings = _estimate_body_from_foreground(person_image)
    raw = Image.new("L", person_image.size, 0)
    draw = ImageDraw.Draw(raw)
    bbox = _body_region_bbox(person_image.size, category, body_bbox)
    draw.rounded_rectangle(bbox, radius=max(8, person_image.size[0] // 30), fill=255)

    if config.innerwear_use_silhouette_clip and silhouette is not None:
        clip = dilate(silhouette, config.innerwear_silhouette_clip_dilation_px)
        clipped = _intersect_masks(raw, clip)
        raw_area = max(1, mask_area(raw))
        clipped_area = mask_area(clipped)
        if clipped_area >= raw_area * 0.40:
            raw = clipped
            source = f"{source}_clipped"
        else:
            warnings.append(f"body silhouette clip rejected for outerwear mask: clipped_area_ratio={clipped_area / raw_area:.3f}.")

    return raw, silhouette, source, body_bbox, tuple(warnings)


def _intersect_masks(left: Image.Image, right: Image.Image) -> Image.Image:
    left_values = np.array(left.convert("L"), dtype=np.uint8) > 0
    right_values = np.array(right.convert("L").resize(left.size, Image.Resampling.NEAREST), dtype=np.uint8) > 0
    return Image.fromarray(((left_values & right_values).astype(np.uint8) * 255), mode="L")


def _draw_bra_mask(size: tuple[int, int], body: tuple[int, int, int, int]) -> Image.Image:
    width, _ = size
    bx0, by0, bx1, by1 = body
    bw = max(1, bx1 - bx0)
    bh = max(1, by1 - by0)
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)

    cup_rx = max(8, int(bw * 0.17))
    cup_ry = max(8, int(bh * 0.075))
    left_cx = int(bx0 + bw * 0.40)
    right_cx = int(bx0 + bw * 0.60)
    cup_cy = int(by0 + bh * 0.275)
    band_y0 = int(by0 + bh * 0.335)
    band_y1 = int(by0 + bh * 0.405)
    band_x0 = int(bx0 + bw * 0.22)
    band_x1 = int(bx0 + bw * 0.78)

    radius = max(5, width // 60)
    draw.rounded_rectangle((band_x0, band_y0, band_x1, band_y1), radius=radius, fill=255)
    draw.ellipse((left_cx - cup_rx, cup_cy - cup_ry, left_cx + cup_rx, cup_cy + cup_ry), fill=255)
    draw.ellipse((right_cx - cup_rx, cup_cy - cup_ry, right_cx + cup_rx, cup_cy + cup_ry), fill=255)
    bridge_w = max(4, int(bw * 0.05))
    bridge_x0 = left_cx + cup_rx - bridge_w
    bridge_x1 = right_cx - cup_rx + bridge_w
    if bridge_x1 < bridge_x0:
        bridge_cx = (left_cx + right_cx) // 2
        bridge_x0 = bridge_cx - bridge_w
        bridge_x1 = bridge_cx + bridge_w
    draw.rounded_rectangle(
        (bridge_x0, cup_cy - cup_ry // 3, bridge_x1, band_y1),
        radius=max(3, bridge_w // 2),
        fill=255,
    )

    strap_w = max(3, width // 90)
    shoulder_y = int(by0 + bh * 0.17)
    draw.line(
        [(left_cx - cup_rx // 2, cup_cy - cup_ry), (int(bx0 + bw * 0.32), shoulder_y)],
        fill=255,
        width=strap_w,
    )
    draw.line(
        [(right_cx + cup_rx // 2, cup_cy - cup_ry), (int(bx0 + bw * 0.68), shoulder_y)],
        fill=255,
        width=strap_w,
    )
    return mask.filter(ImageFilter.MaxFilter(3))


def _draw_underwear_mask(
    size: tuple[int, int],
    body: tuple[int, int, int, int],
    category: TryOnCategory,
) -> Image.Image:
    width, _ = size
    bx0, by0, bx1, by1 = body
    bw = max(1, bx1 - bx0)
    bh = max(1, by1 - by0)
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)

    if category == "men_underwear":
        waist_y = int(by0 + bh * 0.58)
        hip_y = int(by0 + bh * 0.68)
        hem_y = int(by0 + bh * 0.82)
        x0 = int(bx0 + bw * 0.18)
        x1 = int(bx0 + bw * 0.82)
        hip_x0 = int(bx0 + bw * 0.12)
        hip_x1 = int(bx0 + bw * 0.88)
        crotch_x0 = int(bx0 + bw * 0.40)
        crotch_x1 = int(bx0 + bw * 0.60)
        draw.polygon(
            [
                (x0, waist_y),
                (x1, waist_y),
                (hip_x1, hip_y),
                (int(bx0 + bw * 0.70), hem_y),
                (crotch_x1, hem_y),
                (crotch_x0, hem_y),
                (int(bx0 + bw * 0.30), hem_y),
                (hip_x0, hip_y),
            ],
            fill=255,
        )
        draw.rounded_rectangle((x0, waist_y, x1, int(waist_y + bh * 0.045)), radius=max(4, width // 80), fill=255)
    else:
        waist_y = int(by0 + bh * 0.62)
        side_y = int(by0 + bh * 0.72)
        crotch_y = int(by0 + bh * 0.86)
        draw.polygon(
            [
                (int(bx0 + bw * 0.18), waist_y),
                (int(bx0 + bw * 0.82), waist_y),
                (int(bx0 + bw * 0.74), side_y),
                (int(bx0 + bw * 0.58), crotch_y),
                (int(bx0 + bw * 0.42), crotch_y),
                (int(bx0 + bw * 0.26), side_y),
            ],
            fill=255,
        )
        draw.rounded_rectangle(
            (int(bx0 + bw * 0.18), waist_y, int(bx0 + bw * 0.82), int(waist_y + bh * 0.04)),
            radius=max(4, width // 90),
            fill=255,
        )

    holes = Image.new("L", size, 0)
    hole_draw = ImageDraw.Draw(holes)
    leg_rx = int(bw * (0.16 if category == "men_underwear" else 0.13))
    leg_ry = int(bh * (0.08 if category == "men_underwear" else 0.10))
    leg_y = int(by0 + bh * (0.80 if category == "men_underwear" else 0.82))
    hole_draw.ellipse(
        (int(bx0 + bw * 0.30) - leg_rx, leg_y - leg_ry, int(bx0 + bw * 0.30) + leg_rx, leg_y + leg_ry),
        fill=255,
    )
    hole_draw.ellipse(
        (int(bx0 + bw * 0.70) - leg_rx, leg_y - leg_ry, int(bx0 + bw * 0.70) + leg_rx, leg_y + leg_ry),
        fill=255,
    )
    mask = ImageChops.subtract(mask, holes)
    return mask.filter(ImageFilter.MaxFilter(5))


def _create_innerwear_raw_mask(
    person_image: Image.Image,
    category: TryOnCategory,
    config: PreprocessingConfig,
) -> tuple[Image.Image, Image.Image | None, Image.Image | None, str, tuple[int, int, int, int], tuple[str, ...]]:
    body_bbox, silhouette, source, warnings = _estimate_body_from_foreground(person_image)
    if category == "women_bra":
        shape = _draw_bra_mask(person_image.size, body_bbox)
    else:
        shape = _draw_underwear_mask(person_image.size, body_bbox, category)

    raw = shape
    if config.innerwear_use_silhouette_clip and silhouette is not None:
        clip = dilate(silhouette, config.innerwear_silhouette_clip_dilation_px)
        clipped = _intersect_masks(shape, clip)
        shape_area = max(1, mask_area(shape))
        clipped_area = mask_area(clipped)
        if clipped_area >= shape_area * 0.35:
            raw = clipped
            source = f"{source}_clipped"
        else:
            warnings.append(
                f"silhouette clip rejected for innerwear mask: clipped_area_ratio={clipped_area / shape_area:.3f}."
            )

    if raw.getbbox() is None:
        warnings.append("innerwear anatomy mask was empty; falling back to tight bbox mask.")
        raw = Image.new("L", person_image.size, 0)
        draw = ImageDraw.Draw(raw)
        draw.rounded_rectangle(_region_bbox(*person_image.size, category), radius=max(8, person_image.size[0] // 30), fill=255)
        source = "tight_bbox_fallback"

    return raw, silhouette, shape, source, body_bbox, tuple(warnings)


def _protected_region_mask(
    person_image: Image.Image,
    category: TryOnCategory,
    *,
    preserve_face: bool,
    preserve_hair: bool,
    preserve_hands: bool,
) -> Image.Image:
    width, height = person_image.size
    protected = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(protected)
    if preserve_face or preserve_hair:
        face_clearance = int(height * 0.22)
        draw.rectangle((0, 0, width, face_clearance), fill=255)
    if preserve_hands and category in {*UPPER_BODY_LIKE_CATEGORIES, "dress", "full_outfit"}:
        ycbcr = np.array(person_image.convert("YCbCr"), dtype=np.uint8)
        cb = ycbcr[:, :, 1]
        cr = ycbcr[:, :, 2]
        skin = (cb >= 77) & (cb <= 127) & (cr >= 133) & (cr <= 173)
        yy, xx = np.indices((height, width))
        side_region = (
            ((xx < width * 0.32) | (xx > width * 0.68))
            & (yy > height * 0.28)
            & (yy < height * 0.92)
        )
        hand_mask = Image.fromarray((skin & side_region).astype(np.uint8) * 255, mode="L")
        hand_mask = hand_mask.filter(ImageFilter.MaxFilter(21))
        protected = ImageChops.lighter(protected, hand_mask)
    return protected


def _clear_protected(mask: Image.Image, protected: Image.Image) -> Image.Image:
    values = np.array(mask.convert("L"), dtype=np.uint8)
    protected_values = np.array(protected.convert("L"), dtype=np.uint8)
    values[protected_values > 0] = 0
    return Image.fromarray(values, mode="L")


def _expand_upper_body_hem(
    raw_mask: Image.Image,
    config: MaskExperimentConfig,
) -> Image.Image:
    width, height = raw_mask.size
    x0, _, x1, y1 = _region_bbox(width, height, "upper_body")
    extension = max(0, int(round(height * config.torso_down_extension_ratio)))
    bottom = min(height - 1, y1 + extension)
    waist_extra = max(0, config.waist_extra_dilation_px)
    waist_x0 = max(0, x0 - waist_extra)
    waist_x1 = min(width - 1, x1 + waist_extra)

    expanded = raw_mask.copy()
    draw = ImageDraw.Draw(expanded)
    overlap = max(2, min(waist_extra // 2, height // 32))
    draw.rounded_rectangle(
        (waist_x0, max(0, y1 - overlap), waist_x1, bottom),
        radius=max(8, width // 30),
        fill=255,
    )
    return expanded


def create_agnostic_mask(
    person_image: Image.Image,
    category: TryOnCategory,
    config: PreprocessingConfig,
    upper_body_experiment: MaskExperimentConfig | None = None,
) -> AgnosticMaskResult:
    width, height = person_image.size
    mask_source = "geometric_bbox"
    mask_warnings: tuple[str, ...] = ()
    body_bbox_xyxy: tuple[int, int, int, int] | None = None
    body_silhouette_mask: Image.Image | None = None
    innerwear_shape_mask: Image.Image | None = None
    if category in INNERWEAR_CATEGORIES:
        (
            original_raw,
            body_silhouette_mask,
            innerwear_shape_mask,
            mask_source,
            body_bbox_xyxy,
            mask_warnings,
        ) = _create_innerwear_raw_mask(person_image, category, config)
        draw = ImageDraw.Draw(original_raw)
    else:
        (
            original_raw,
            body_silhouette_mask,
            mask_source,
            body_bbox_xyxy,
            mask_warnings,
        ) = _create_outerwear_raw_mask(person_image, category, config)
        draw = ImageDraw.Draw(original_raw)

    if category not in INNERWEAR_CATEGORIES and (config.preserve_face or config.preserve_hair):
        face_clearance = int(height * 0.22)
        draw.rectangle((0, 0, width, face_clearance), fill=0)

    if config.preserve_hands and category in {*UPPER_BODY_LIKE_CATEGORIES, "dress", "full_outfit"}:
        hand_w = int(width * 0.13)
        hand_y0 = int(height * 0.38)
        hand_y1 = int(height * 0.72)
        draw.rectangle((0, hand_y0, hand_w, hand_y1), fill=0)
        draw.rectangle((width - hand_w, hand_y0, width, hand_y1), fill=0)

    experiment_enabled = bool(
        category == "upper_body"
        and upper_body_experiment is not None
        and upper_body_experiment.enabled
    )
    selected_raw = original_raw
    debug_expanded = None
    debug_diff = None
    original_overlay = None
    expanded_overlay = None
    diff_overlay = None
    protected = None
    if experiment_enabled and upper_body_experiment is not None:
        protected = _protected_region_mask(
            person_image,
            category,
            preserve_face=upper_body_experiment.preserve_face,
            preserve_hair=upper_body_experiment.preserve_hair,
            preserve_hands=upper_body_experiment.preserve_hands,
        )
        candidate = _expand_upper_body_hem(original_raw, upper_body_experiment)
        added_region = _clear_protected(
            ImageChops.subtract(candidate, original_raw),
            protected,
        )
        debug_expanded = ImageChops.lighter(original_raw, added_region)
        debug_diff = added_region
        selected_raw = debug_expanded
        if upper_body_experiment.save_debug_overlays:
            original_overlay = overlay_mask_preview(person_image, original_raw, (59, 130, 246))
            expanded_overlay = overlay_mask_preview(person_image, debug_expanded, (16, 185, 129))
            diff_overlay = overlay_mask_preview(person_image, debug_diff, (239, 68, 68))

    if protected is not None and debug_diff is not None:
        base_expanded = dilate(original_raw, config.dilation_px)
        added_expanded = _clear_protected(
            dilate(debug_diff, config.dilation_px),
            protected,
        )
        expanded = ImageChops.lighter(base_expanded, added_expanded)
        base_soft = blur(base_expanded, config.blur_radius)
        added_soft = _clear_protected(
            blur(added_expanded, config.blur_radius),
            protected,
        )
        soft = ImageChops.lighter(base_soft, added_soft)
    elif category in INNERWEAR_CATEGORIES:
        expanded = dilate(selected_raw, config.innerwear_dilation_px)
        soft = blur(expanded, config.innerwear_blur_radius)
    else:
        expanded = dilate(selected_raw, config.dilation_px)
        soft = blur(expanded, config.blur_radius)
    preview = overlay_mask_preview(person_image, expanded)

    agnostic = person_image.convert("RGB").copy()
    overlay = Image.new("RGB", person_image.size, (224, 224, 224))
    agnostic.paste(overlay, mask=soft)
    return AgnosticMaskResult(
        selected_raw,
        expanded,
        soft,
        preview,
        agnostic,
        mask_source=mask_source,
        mask_warnings=mask_warnings,
        body_bbox_xyxy=body_bbox_xyxy,
        body_silhouette_mask=body_silhouette_mask,
        innerwear_shape_mask=innerwear_shape_mask,
        original_upper_body_mask=original_raw if experiment_enabled else None,
        expanded_upper_body_mask=debug_expanded,
        diff_upper_body_mask=debug_diff,
        original_upper_body_overlay=original_overlay,
        expanded_upper_body_overlay=expanded_overlay,
        diff_upper_body_overlay=diff_overlay,
    )


def bundle_from_result(result: AgnosticMaskResult) -> MaskBundle:
    return MaskBundle(
        raw_mask=result.raw_mask,
        dilated_mask=result.dilated_mask,
        soft_mask=result.soft_mask,
        preview=result.preview,
    )
