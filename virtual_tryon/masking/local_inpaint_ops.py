from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageOps


TARGET_REGIONS = ("upper", "lower", "dress", "shoes", "hat", "accessory")


@dataclass
class MaskMorphologyConfig:
    grow_px: int = 16
    blur_px: int = 10
    threshold: int = 8
    invert: bool = False
    fill_holes: bool = True
    keep_largest_component: bool = False


@dataclass
class BBoxCropConfig:
    padding_ratio: float = 0.22
    min_padding_px: int = 32
    max_padding_px: int = 160
    force_square: bool = False
    target_multiple: int = 16
    min_crop_size: int = 256
    target_region: str = "upper"
    min_mask_area_ratio: float = 0.0005


@dataclass
class PasteBackConfig:
    feather_px: int = 12
    color_match: bool = True
    preserve_outside_mask: bool = True
    debug: bool = True


@dataclass
class FitCanvasConfig:
    width: int = 768
    height: int = 1024
    background: str = "white"


@dataclass
class LocalInpaintResult:
    image: Image.Image | None = None
    mask: Image.Image | None = None
    bbox_json: str | None = None
    status: dict[str, Any] = field(default_factory=dict)
    overlay: Image.Image | None = None


def ensure_rgb(image: Image.Image) -> Image.Image:
    return ImageOps.exif_transpose(image.convert("RGB"))


def ensure_mask(mask: Image.Image, size: tuple[int, int] | None = None, threshold: int = 8) -> Image.Image:
    out = mask.convert("L")
    if size is not None and out.size != size:
        out = out.resize(size, Image.Resampling.NEAREST)
    return out.point(lambda p: 255 if p > threshold else 0)


def mask_area_ratio(mask: Image.Image, threshold: int = 8) -> float:
    arr = np.asarray(mask.convert("L"))
    return float((arr > threshold).mean())


def _neighbors(x: int, y: int, width: int, height: int):
    if x > 0:
        yield x - 1, y
    if x + 1 < width:
        yield x + 1, y
    if y > 0:
        yield x, y - 1
    if y + 1 < height:
        yield x, y + 1


def _fill_holes(binary: np.ndarray) -> np.ndarray:
    height, width = binary.shape
    background = ~binary
    visited = np.zeros_like(background, dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    for x in range(width):
        if background[0, x]:
            queue.append((x, 0))
        if background[height - 1, x]:
            queue.append((x, height - 1))
    for y in range(height):
        if background[y, 0]:
            queue.append((0, y))
        if background[y, width - 1]:
            queue.append((width - 1, y))

    while queue:
        x, y = queue.popleft()
        if visited[y, x] or not background[y, x]:
            continue
        visited[y, x] = True
        for nx, ny in _neighbors(x, y, width, height):
            if not visited[ny, nx] and background[ny, nx]:
                queue.append((nx, ny))

    holes = background & ~visited
    return binary | holes


def _keep_largest_component(binary: np.ndarray) -> np.ndarray:
    height, width = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    best: list[tuple[int, int]] = []

    for y in range(height):
        for x in range(width):
            if visited[y, x] or not binary[y, x]:
                continue
            component: list[tuple[int, int]] = []
            queue: deque[tuple[int, int]] = deque([(x, y)])
            visited[y, x] = True
            while queue:
                cx, cy = queue.popleft()
                component.append((cx, cy))
                for nx, ny in _neighbors(cx, cy, width, height):
                    if not visited[ny, nx] and binary[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((nx, ny))
            if len(component) > len(best):
                best = component

    out = np.zeros_like(binary, dtype=bool)
    for x, y in best:
        out[y, x] = True
    return out


def mask_morphology(mask: Image.Image, config: MaskMorphologyConfig) -> LocalInpaintResult:
    """Grow/erode, optionally hole-fill, keep largest component, and feather a mask."""
    warnings: list[str] = []
    binary = ensure_mask(mask, threshold=int(config.threshold))
    if config.invert:
        binary = ImageChops.invert(binary)

    arr = np.asarray(binary) > 8
    if config.fill_holes:
        arr = _fill_holes(arr)
    if config.keep_largest_component and arr.any():
        arr = _keep_largest_component(arr)
    elif config.keep_largest_component:
        warnings.append("keep_largest_component requested but mask is empty.")

    processed = Image.fromarray((arr.astype(np.uint8) * 255), mode="L")
    grow = int(config.grow_px)
    if grow > 0:
        processed = processed.filter(ImageFilter.MaxFilter(grow * 2 + 1))
    elif grow < 0:
        processed = processed.filter(ImageFilter.MinFilter(abs(grow) * 2 + 1))
    if config.blur_px > 0:
        processed = processed.filter(ImageFilter.GaussianBlur(float(config.blur_px)))

    status = {
        "node": "mask_morphology",
        "grow_px": int(config.grow_px),
        "blur_px": int(config.blur_px),
        "threshold": int(config.threshold),
        "invert": bool(config.invert),
        "fill_holes": bool(config.fill_holes),
        "keep_largest_component": bool(config.keep_largest_component),
        "mask_area_ratio": round(mask_area_ratio(processed), 6),
        "warnings": warnings,
    }
    return LocalInpaintResult(mask=processed, status=status)


def fallback_bbox(size: tuple[int, int], target_region: str) -> tuple[int, int, int, int]:
    width, height = size
    regions = {
        "upper": (0.20, 0.12, 0.80, 0.62),
        "lower": (0.20, 0.40, 0.80, 0.96),
        "dress": (0.15, 0.12, 0.85, 0.94),
        "shoes": (0.12, 0.72, 0.88, 1.00),
        "hat": (0.20, 0.00, 0.80, 0.34),
        "accessory": (0.10, 0.22, 0.90, 0.76),
    }
    x0r, y0r, x1r, y1r = regions.get(target_region, regions["upper"])
    return (
        max(0, int(width * x0r)),
        max(0, int(height * y0r)),
        min(width, int(width * x1r)),
        min(height, int(height * y1r)),
    )


def _expand_bbox(
    bbox: tuple[int, int, int, int],
    image_size: tuple[int, int],
    padding_ratio: float,
    min_padding_px: int,
    max_padding_px: int,
    min_crop_size: int,
    force_square: bool,
    target_multiple: int,
) -> tuple[int, int, int, int]:
    width, height = image_size
    x0, y0, x1, y1 = bbox
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    pad = int(max(min_padding_px, min(max_padding_px, max(bw, bh) * padding_ratio)))
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    crop_w = max(bw + pad * 2, min_crop_size)
    crop_h = max(bh + pad * 2, min_crop_size)
    if force_square:
        side = max(crop_w, crop_h)
        crop_w = crop_h = side
    if target_multiple and target_multiple > 1:
        crop_w = int(np.ceil(crop_w / target_multiple) * target_multiple)
        crop_h = int(np.ceil(crop_h / target_multiple) * target_multiple)

    x0 = int(round(cx - crop_w / 2))
    y0 = int(round(cy - crop_h / 2))
    x1 = x0 + int(crop_w)
    y1 = y0 + int(crop_h)

    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x1 > width:
        shift = x1 - width
        x0 = max(0, x0 - shift)
        x1 = width
    if y1 > height:
        shift = y1 - height
        y0 = max(0, y0 - shift)
        y1 = height
    return (int(x0), int(y0), int(x1), int(y1))


def bbox_crop(image: Image.Image, mask: Image.Image, config: BBoxCropConfig) -> LocalInpaintResult:
    """Crop image and mask around the target mask with explicit fallback behavior."""
    person = ensure_rgb(image)
    binary = ensure_mask(mask, person.size)
    warnings: list[str] = []
    bbox = binary.getbbox()
    area = mask_area_ratio(binary)
    used_fallback = False
    if bbox is None or area < config.min_mask_area_ratio:
        bbox = fallback_bbox(person.size, config.target_region)
        used_fallback = True
        warnings.append(
            f"Mask empty or too small (area_ratio={area:.6f}); used {config.target_region} fallback bbox."
        )

    expanded = _expand_bbox(
        bbox,
        person.size,
        float(config.padding_ratio),
        int(config.min_padding_px),
        int(config.max_padding_px),
        int(config.min_crop_size),
        bool(config.force_square),
        int(config.target_multiple),
    )
    cropped_image = person.crop(expanded).convert("RGB")
    cropped_mask = binary.crop(expanded).convert("L")
    if used_fallback and cropped_mask.getbbox() is None:
        cropped_mask = Image.new("L", cropped_image.size, 255)
        warnings.append("Fallback bbox had empty cropped mask; crop mask was filled to allow local edit.")

    overlay = person.convert("RGBA")
    layer = Image.new("RGBA", person.size, (36, 140, 255, 0))
    alpha = binary.point(lambda p: int(p * 0.35))
    layer.putalpha(alpha)
    overlay = Image.alpha_composite(overlay, layer)
    draw = ImageDraw.Draw(overlay)
    draw.rectangle(expanded, outline=(255, 80, 40, 255), width=max(3, person.size[0] // 180))

    x0, y0, x1, y1 = expanded
    payload = {
        "target_region": config.target_region,
        "bbox_xyxy": [x0, y0, x1, y1],
        "bbox_width": x1 - x0,
        "bbox_height": y1 - y0,
        "original_width": person.width,
        "original_height": person.height,
        "mask_area_ratio": round(area, 6),
        "cropped_mask_area_ratio": round(mask_area_ratio(cropped_mask), 6),
        "used_fallback": used_fallback,
        "padding_ratio": float(config.padding_ratio),
        "min_padding_px": int(config.min_padding_px),
        "max_padding_px": int(config.max_padding_px),
        "force_square": bool(config.force_square),
        "target_multiple": int(config.target_multiple),
        "min_crop_size": int(config.min_crop_size),
        "warnings": warnings,
    }
    return LocalInpaintResult(
        image=cropped_image,
        mask=cropped_mask,
        bbox_json=json.dumps(payload, ensure_ascii=False),
        status={"node": "bbox_crop", **payload},
        overlay=overlay.convert("RGB"),
    )


def _parse_bbox_json(bbox_json: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(bbox_json, str):
        return json.loads(bbox_json)
    return bbox_json


def _color_match_crop(generated: Image.Image, original: Image.Image, mask: Image.Image) -> Image.Image:
    edge = mask.filter(ImageFilter.MaxFilter(15))
    inner = mask.filter(ImageFilter.MinFilter(7))
    band = ImageChops.subtract(edge, inner)
    band_arr = np.asarray(band) > 8
    if band_arr.sum() < 64:
        band_arr = np.asarray(mask) > 8
    if band_arr.sum() < 64:
        return generated

    gen = np.asarray(generated.convert("RGB")).astype(np.float32)
    ori = np.asarray(original.convert("RGB")).astype(np.float32)
    gen_mean = gen[band_arr].mean(axis=0)
    ori_mean = ori[band_arr].mean(axis=0)
    adjusted = gen + (ori_mean - gen_mean) * 0.35
    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
    return Image.fromarray(adjusted, mode="RGB")


def fit_canvas_with_meta(image: Image.Image, config: FitCanvasConfig) -> LocalInpaintResult:
    """Fit an image onto a fixed canvas and keep exact placement metadata."""
    source = ensure_rgb(image)
    canvas_size = (int(config.width), int(config.height))
    fitted = ImageOps.contain(source, canvas_size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", canvas_size, config.background)
    x0 = (canvas_size[0] - fitted.width) // 2
    y0 = (canvas_size[1] - fitted.height) // 2
    x1 = x0 + fitted.width
    y1 = y0 + fitted.height
    canvas.paste(fitted, (x0, y0))
    payload = {
        "node": "fit_canvas_with_meta",
        "original_width": source.width,
        "original_height": source.height,
        "canvas_width": canvas_size[0],
        "canvas_height": canvas_size[1],
        "content_xyxy": [x0, y0, x1, y1],
        "content_width": fitted.width,
        "content_height": fitted.height,
        "background": config.background,
    }
    return LocalInpaintResult(image=canvas, bbox_json=json.dumps(payload, ensure_ascii=False), status=payload)


def extract_fitted_canvas_region(generated_canvas: Image.Image, fit_meta_json: str | dict[str, Any]) -> LocalInpaintResult:
    """Undo fit_canvas_with_meta: crop the generated canvas content and resize to original crop size."""
    meta = _parse_bbox_json(fit_meta_json)
    x0, y0, x1, y1 = [int(v) for v in meta["content_xyxy"]]
    generated = ensure_rgb(generated_canvas)
    content = generated.crop((x0, y0, x1, y1)).convert("RGB")
    original_size = (int(meta["original_width"]), int(meta["original_height"]))
    if content.size != original_size:
        content = content.resize(original_size, Image.Resampling.LANCZOS)
    status = {
        "node": "extract_fitted_canvas_region",
        "content_xyxy": [x0, y0, x1, y1],
        "output_width": content.width,
        "output_height": content.height,
    }
    return LocalInpaintResult(image=content, status=status)


def masked_paste_back(
    original_image: Image.Image,
    generated_crop: Image.Image,
    cropped_mask: Image.Image,
    bbox_json: str | dict[str, Any],
    config: PasteBackConfig,
) -> LocalInpaintResult:
    """Paste a generated local crop back into the original image using the crop mask."""
    original = ensure_rgb(original_image)
    meta = _parse_bbox_json(bbox_json)
    x0, y0, x1, y1 = [int(v) for v in meta["bbox_xyxy"]]
    bbox_size = (max(1, x1 - x0), max(1, y1 - y0))
    original_crop = original.crop((x0, y0, x1, y1)).convert("RGB")
    generated = ensure_rgb(generated_crop).resize(bbox_size, Image.Resampling.LANCZOS)
    mask = cropped_mask.convert("L").resize(bbox_size, Image.Resampling.NEAREST)
    if config.color_match:
        generated = _color_match_crop(generated, original_crop, ensure_mask(mask, bbox_size))
    if config.feather_px > 0:
        paste_mask = mask.filter(ImageFilter.GaussianBlur(float(config.feather_px)))
    else:
        paste_mask = ensure_mask(mask, bbox_size)
    if config.preserve_outside_mask:
        composite_crop = Image.composite(generated, original_crop, paste_mask)
    else:
        composite_crop = generated

    final = original.copy()
    final.paste(composite_crop, (x0, y0))
    overlay = original.convert("RGBA")
    crop_layer = Image.new("RGBA", original.size, (255, 90, 30, 0))
    full_mask = Image.new("L", original.size, 0)
    full_mask.paste(paste_mask, (x0, y0))
    crop_layer.putalpha(full_mask.point(lambda p: int(p * 0.42)))
    overlay = Image.alpha_composite(overlay, crop_layer).convert("RGB")
    draw = ImageDraw.Draw(overlay)
    draw.rectangle((x0, y0, x1, y1), outline=(255, 70, 20), width=max(3, original.width // 180))

    status = {
        "node": "masked_paste_back",
        "bbox_xyxy": [x0, y0, x1, y1],
        "feather_px": int(config.feather_px),
        "color_match": bool(config.color_match),
        "preserve_outside_mask": bool(config.preserve_outside_mask),
        "paste_mask_area_ratio": round(mask_area_ratio(full_mask), 6),
        "warnings": [],
    }
    return LocalInpaintResult(image=final, mask=full_mask, status=status, overlay=overlay)


def make_debug_sheet(
    *,
    person: Image.Image,
    garment: Image.Image,
    raw_mask: Image.Image,
    processed_mask: Image.Image,
    overlay: Image.Image,
    crop_image: Image.Image,
    crop_mask: Image.Image,
    generated_crop: Image.Image,
    final_image: Image.Image,
    status_text: str,
) -> LocalInpaintResult:
    """Create a compact contact sheet for one local masked inpaint pass."""
    labels = [
        ("person", person),
        ("garment", garment),
        ("raw mask", raw_mask.convert("RGB")),
        ("processed mask", processed_mask.convert("RGB")),
        ("overlay", overlay),
        ("crop", crop_image),
        ("crop mask", crop_mask.convert("RGB")),
        ("generated crop", generated_crop),
        ("final", final_image),
    ]
    cell = (220, 280)
    header_h = 28
    cols = 3
    rows = 3
    sheet = Image.new("RGB", (cell[0] * cols, (cell[1] + header_h) * rows + 92), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, (label, image) in enumerate(labels):
        col = idx % cols
        row = idx // cols
        x = col * cell[0]
        y = row * (cell[1] + header_h)
        draw.rectangle((x, y, x + cell[0], y + header_h), fill=(236, 240, 245), outline=(211, 217, 225))
        draw.text((x + 8, y + 8), label, fill=(20, 24, 32))
        thumb = ImageOps.contain(ensure_rgb(image), (cell[0] - 12, cell[1] - 12), Image.Resampling.LANCZOS)
        sheet.paste(thumb, (x + (cell[0] - thumb.width) // 2, y + header_h + (cell[1] - thumb.height) // 2))
    footer_y = (cell[1] + header_h) * rows
    draw.rectangle((0, footer_y, sheet.width, sheet.height), fill=(248, 250, 252))
    for line_idx, line in enumerate(status_text.splitlines()[:4]):
        draw.text((10, footer_y + 10 + line_idx * 18), line[:130], fill=(62, 70, 82))
    return LocalInpaintResult(image=sheet, status={"node": "debug_sheet", "items": len(labels)})
