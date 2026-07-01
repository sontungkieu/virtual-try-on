from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .mask_morphology import binarize, mask_area_ratio


TargetRegion = Literal["upper", "lower", "dress", "shoes", "hat", "accessory"]


# ATR labels emitted by the IDM-VTON parser used in this project.
ATR_HEAD_LABELS = [1, 2, 3, 11, 18]
ATR_BODY_LABELS = [2, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 18]
ATR_TORSO_LABELS = [4, 5, 6, 7, 8, 18]
ATR_LEG_LABELS = [5, 6, 9, 10, 12, 13]
ATR_LEFT_LEG_LABELS = [9, 12]
ATR_RIGHT_LEG_LABELS = [10, 13]
ATR_ARM_LABELS = [14, 15]


MIN_EXTENT_FALLBACK_AREA: dict[str, float] = {
    "upper": 0.065,
    "lower": 0.050,
    "dress": 0.120,
    "shoes": 0.012,
    "hat": 0.015,
    "accessory": 0.006,
}


@dataclass(frozen=True)
class TargetExtentMaskResult:
    mask: Image.Image
    source: str
    bbox_xyxy: tuple[int, int, int, int] | None
    warnings: list[str]


def should_use_target_extent_fallback(
    target_region: str,
    area_ratio: float,
    warnings: list[str] | None = None,
) -> bool:
    threshold = MIN_EXTENT_FALLBACK_AREA.get(target_region, 0.01)
    if area_ratio < threshold:
        return True
    return any("mask too small" in warning.lower() for warning in (warnings or []))


def _load_semantic_map(path: str | Path, size: tuple[int, int]) -> np.ndarray:
    semantic = Image.open(path)
    if semantic.mode not in {"P", "L", "I", "I;16"}:
        semantic = semantic.convert("L")
    if semantic.size != size:
        semantic = semantic.resize(size, Image.Resampling.NEAREST)
    return np.array(semantic)


def _mask_from_labels(semantic: np.ndarray, labels: list[int], size: tuple[int, int]) -> Image.Image:
    if semantic.size == 0:
        return Image.new("L", size, 0)
    selected = np.isin(semantic, np.array(labels, dtype=semantic.dtype))
    return Image.fromarray((selected.astype(np.uint8) * 255), mode="L")


def _bbox(mask: Image.Image) -> tuple[int, int, int, int] | None:
    return binarize(mask).getbbox()


def _expand_bbox(
    bbox: tuple[int, int, int, int] | None,
    size: tuple[int, int],
    *,
    x_pad: int = 0,
    y_pad: int = 0,
) -> tuple[int, int, int, int] | None:
    if bbox is None:
        return None
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
    return (int(width * 0.24), int(height * 0.08), int(width * 0.76), int(height * 0.96))


def _semantic_bboxes(
    person_size: tuple[int, int],
    semantic_map_path: str | Path | None,
) -> dict[str, tuple[int, int, int, int] | None]:
    if semantic_map_path is None or not Path(semantic_map_path).exists():
        return {
            "body": None,
            "head": None,
            "torso": None,
            "legs": None,
            "left_leg": None,
            "right_leg": None,
            "arms": None,
        }

    semantic = _load_semantic_map(semantic_map_path, person_size)
    return {
        "body": _bbox(_mask_from_labels(semantic, ATR_BODY_LABELS, person_size)),
        "head": _bbox(_mask_from_labels(semantic, ATR_HEAD_LABELS, person_size)),
        "torso": _bbox(_mask_from_labels(semantic, ATR_TORSO_LABELS, person_size)),
        "legs": _bbox(_mask_from_labels(semantic, ATR_LEG_LABELS, person_size)),
        "left_leg": _bbox(_mask_from_labels(semantic, ATR_LEFT_LEG_LABELS, person_size)),
        "right_leg": _bbox(_mask_from_labels(semantic, ATR_RIGHT_LEG_LABELS, person_size)),
        "arms": _bbox(_mask_from_labels(semantic, ATR_ARM_LABELS, person_size)),
    }


def _draw_lower(draw: ImageDraw.ImageDraw, body: tuple[int, int, int, int], size: tuple[int, int]) -> None:
    width, _ = size
    bx0, by0, bx1, by1 = body
    bw = bx1 - bx0
    bh = by1 - by0
    x0 = int(bx0 + bw * 0.12)
    x1 = int(bx1 - bw * 0.12)
    y0 = int(by0 + bh * 0.42)
    y1 = int(by0 + bh * 0.76)
    draw.rounded_rectangle((x0, y0, x1, y1), radius=max(8, width // 38), fill=255)


def _draw_upper(draw: ImageDraw.ImageDraw, body: tuple[int, int, int, int], size: tuple[int, int]) -> None:
    width, _ = size
    bx0, by0, bx1, by1 = body
    bw = bx1 - bx0
    bh = by1 - by0
    x0 = int(bx0 + bw * 0.05)
    x1 = int(bx1 - bw * 0.05)
    y0 = int(by0 + bh * 0.17)
    y1 = int(by0 + bh * 0.55)
    draw.rounded_rectangle((x0, y0, x1, y1), radius=max(8, width // 36), fill=255)


def _draw_dress(draw: ImageDraw.ImageDraw, body: tuple[int, int, int, int]) -> None:
    bx0, by0, bx1, by1 = body
    bw = bx1 - bx0
    bh = by1 - by0
    shoulder_y = int(by0 + bh * 0.16)
    waist_y = int(by0 + bh * 0.43)
    hem_y = int(by0 + bh * 0.86)
    top_l = int(bx0 + bw * 0.20)
    top_r = int(bx1 - bw * 0.20)
    waist_l = int(bx0 + bw * 0.12)
    waist_r = int(bx1 - bw * 0.12)
    hem_l = int(bx0 - bw * 0.06)
    hem_r = int(bx1 + bw * 0.06)
    draw.polygon(
        [
            (top_l, shoulder_y),
            (top_r, shoulder_y),
            (waist_r, waist_y),
            (hem_r, hem_y),
            (hem_l, hem_y),
            (waist_l, waist_y),
        ],
        fill=255,
    )


def _draw_hat(
    draw: ImageDraw.ImageDraw,
    head: tuple[int, int, int, int] | None,
    body: tuple[int, int, int, int],
    size: tuple[int, int],
) -> None:
    width, height = size
    if head is None:
        bx0, by0, bx1, by1 = body
        cx = (bx0 + bx1) // 2
        hw = max(width // 10, (bx1 - bx0) // 4)
        head = (cx - hw, by0, cx + hw, int(by0 + height * 0.18))
    hx0, hy0, hx1, hy1 = head
    hw = hx1 - hx0
    hh = max(1, hy1 - hy0)
    cx = (hx0 + hx1) // 2
    y0 = max(0, int(hy0 - hh * 0.55))
    y1 = min(height, int(hy0 + hh * 0.42))
    x0 = max(0, int(cx - hw * 0.85))
    x1 = min(width, int(cx + hw * 0.85))
    draw.ellipse((x0, y0, x1, y1), fill=255)
    brim_y0 = max(0, int(hy0 + hh * 0.15))
    brim_y1 = min(height, int(hy0 + hh * 0.45))
    draw.rounded_rectangle(
        (max(0, int(cx - hw * 1.15)), brim_y0, min(width, int(cx + hw * 1.15)), brim_y1),
        radius=max(4, hw // 10),
        fill=255,
    )


def _draw_shoes(
    draw: ImageDraw.ImageDraw,
    bboxes: dict[str, tuple[int, int, int, int] | None],
    body: tuple[int, int, int, int],
    size: tuple[int, int],
) -> None:
    width, _ = size
    bx0, by0, bx1, by1 = body
    bw = bx1 - bx0
    bh = by1 - by0
    default_y0 = int(by0 + bh * 0.84)
    default_y1 = by1
    leg_boxes = [bboxes.get("left_leg"), bboxes.get("right_leg")]
    if not any(leg_boxes):
        draw.rounded_rectangle(
            (int(bx0 + bw * 0.08), default_y0, int(bx1 - bw * 0.08), default_y1),
            radius=max(4, width // 48),
            fill=255,
        )
        return
    for leg in leg_boxes:
        if leg is None:
            continue
        lx0, ly0, lx1, ly1 = leg
        lw = max(lx1 - lx0, int(bw * 0.18))
        cx = (lx0 + lx1) // 2
        y0 = max(default_y0, int(ly1 - bh * 0.10))
        y1 = min(size[1], int(ly1 + bh * 0.025))
        draw.rounded_rectangle(
            (
                max(0, int(cx - lw * 0.85)),
                y0,
                min(size[0], int(cx + lw * 0.85)),
                y1,
            ),
            radius=max(4, width // 55),
            fill=255,
        )


def _draw_accessory(
    draw: ImageDraw.ImageDraw,
    arms: tuple[int, int, int, int] | None,
    body: tuple[int, int, int, int],
    size: tuple[int, int],
) -> None:
    width, height = size
    bx0, by0, bx1, by1 = body
    bw = bx1 - bx0
    bh = by1 - by0
    if arms is None:
        centers = [
            (int(bx0 + bw * 0.04), int(by0 + bh * 0.56)),
            (int(bx1 - bw * 0.04), int(by0 + bh * 0.56)),
        ]
    else:
        ax0, ay0, ax1, ay1 = arms
        centers = [
            (int(ax0 + (ax1 - ax0) * 0.10), int(ay0 + (ay1 - ay0) * 0.72)),
            (int(ax1 - (ax1 - ax0) * 0.10), int(ay0 + (ay1 - ay0) * 0.72)),
        ]
    rx = max(8, int(width * 0.045))
    ry = max(8, int(height * 0.035))
    for cx, cy in centers:
        draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=255)


def create_target_extent_mask(
    person_image: Image.Image,
    target_region: TargetRegion,
    *,
    semantic_map_path: str | Path | None = None,
) -> TargetExtentMaskResult:
    person = person_image.convert("RGB")
    size = person.size
    bboxes = _semantic_bboxes(size, semantic_map_path)
    body = _expand_bbox(bboxes.get("body"), size, x_pad=max(4, size[0] // 60), y_pad=max(4, size[1] // 120))
    warnings: list[str] = []
    if body is None:
        body = _fallback_body_bbox(size)
        warnings.append("target extent used geometric body fallback because semantic body bbox was unavailable.")

    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    if target_region == "upper":
        _draw_upper(draw, body, size)
    elif target_region == "lower":
        _draw_lower(draw, body, size)
    elif target_region == "dress":
        _draw_dress(draw, body)
    elif target_region == "shoes":
        _draw_shoes(draw, bboxes, body, size)
    elif target_region == "hat":
        _draw_hat(draw, bboxes.get("head"), body, size)
    elif target_region == "accessory":
        _draw_accessory(draw, bboxes.get("arms"), body, size)
    else:
        raise ValueError(f"Unsupported target_region '{target_region}'")

    mask = binarize(mask.filter(ImageFilter.GaussianBlur(0.25)))
    if mask.getbbox() is None:
        raise ValueError(f"Target extent mask is empty for target_region='{target_region}'")
    ratio = mask_area_ratio(mask)
    if ratio < MIN_EXTENT_FALLBACK_AREA.get(target_region, 0.01):
        warnings.append(f"target extent mask remains small: area_ratio={ratio:.4f}")
    return TargetExtentMaskResult(
        mask=mask,
        source="target_extent",
        bbox_xyxy=_bbox(mask),
        warnings=warnings,
    )
