from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

from .manual_mask_loader import load_manual_mask
from .mask_morphology import MaskArtifacts, MaskPostprocessConfig, binarize, mask_area_ratio, postprocess_mask
from .sam_mask import SAMMaskConfig, SAMMasker


TargetRegion = Literal["upper", "lower", "dress", "shoes", "hat", "accessory"]
MaskSource = Literal["semantic", "manual", "sam_box"]


# LIP/SCHP-style human parsing labels:
# 0 background, 1 hat, 2 hair, 3 glove, 4 sunglasses, 5 upper-clothes,
# 6 dress, 7 coat, 8 socks, 9 pants, 10 jumpsuits, 11 scarf, 12 skirt,
# 13 face, 14 left-arm, 15 right-arm, 16 left-leg, 17 right-leg,
# 18 left-shoe, 19 right-shoe.
LIP_TARGET_LABELS: dict[str, list[int]] = {
    "upper": [5, 7],
    "lower": [9, 12],
    "dress": [6, 10, 12],
    "shoes": [18, 19],
    "hat": [1],
    "accessory": [1, 3, 4, 11],
}

LIP_PROTECT_LABELS: dict[str, list[int]] = {
    "upper": [1, 2, 3, 4, 13, 14, 15],
    "lower": [1, 2, 3, 4, 5, 6, 7, 13, 14, 15, 18, 19],
    "dress": [1, 2, 3, 4, 13, 14, 15],
    "shoes": [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 14, 15, 16, 17],
    "hat": [2, 3, 4, 13],
    "accessory": [2, 13],
}


# ATR-style labels used by the IDM-VTON ONNX human parser:
# 0 background, 1 hat, 2 hair, 3 sunglasses, 4 upper-clothes, 5 skirt,
# 6 pants, 7 dress, 8 belt, 9 left-shoe, 10 right-shoe, 11 face,
# 12 left-leg, 13 right-leg, 14 left-arm, 15 right-arm, 16 bag,
# 17 scarf, 18 neck (added by the IDM parser postprocess).
ATR_TARGET_LABELS: dict[str, list[int]] = {
    "upper": [4],
    "lower": [5, 6],
    "dress": [7],
    "shoes": [9, 10],
    "hat": [1],
    "accessory": [1, 3, 8, 16, 17],
}

ATR_PROTECT_LABELS: dict[str, list[int]] = {
    "upper": [1, 2, 3, 11, 14, 15, 18],
    "lower": [1, 2, 3, 4, 7, 9, 10, 11, 14, 15, 18],
    "dress": [1, 2, 3, 11, 14, 15, 18],
    "shoes": [1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 15, 18],
    "hat": [2, 3, 11, 18],
    "accessory": [2, 11, 18],
}


@dataclass(frozen=True)
class HybridMaskConfig:
    """Production-oriented mask builder for virtual try-on.

    The default contract follows:
    semantic human parsing -> optional SAM boundary refine -> protect-mask
    subtraction -> morphology/feather -> validation.

    This module is intentionally an adapter layer. It does not pretend to run
    SCHP/SAM2 when those dependencies/checkpoints are missing; callers can pass
    an existing semantic map, manual mask, or SAM box and receive explicit
    warnings when optional refinement is unavailable.
    """

    target_label_map: dict[str, list[int]] = field(default_factory=lambda: dict(LIP_TARGET_LABELS))
    protect_label_map: dict[str, list[int]] = field(default_factory=lambda: dict(LIP_PROTECT_LABELS))
    postprocess: MaskPostprocessConfig = field(default_factory=MaskPostprocessConfig)
    sam: SAMMaskConfig = field(default_factory=SAMMaskConfig)
    prefer_semantic: bool = True
    refine_with_sam: bool = True
    sam_bbox_padding_px: int = 12
    intersect_sam_with_semantic_envelope: bool = True
    semantic_envelope_dilation_px: int = 16
    sam_accept_min_iou: float = 0.55
    sam_accept_min_area_ratio: float = 0.65
    sam_accept_max_area_ratio: float = 1.60
    subtract_semantic_protect: bool = True
    min_area_ratio: float = 0.005
    max_area_ratio: float = 0.55

    @classmethod
    def atr(
        cls,
        *,
        postprocess: MaskPostprocessConfig | None = None,
        sam: SAMMaskConfig | None = None,
        **overrides: object,
    ) -> "HybridMaskConfig":
        values = {
            "target_label_map": dict(ATR_TARGET_LABELS),
            "protect_label_map": dict(ATR_PROTECT_LABELS),
            "postprocess": postprocess or MaskPostprocessConfig(),
            "sam": sam or SAMMaskConfig(),
            **overrides,
        }
        return cls(**values)


@dataclass(frozen=True)
class HybridMaskResult:
    source: MaskSource
    raw_mask: Image.Image
    processed_mask: Image.Image
    overlay: Image.Image
    protect_mask: Image.Image
    boundary_refined_mask: Image.Image | None
    mask_area_ratio: float
    bbox_xyxy: tuple[int, int, int, int] | None
    warnings: list[str]

    @property
    def artifacts(self) -> MaskArtifacts:
        return MaskArtifacts(
            raw_mask=self.raw_mask,
            processed_mask=self.processed_mask,
            overlay=self.overlay,
            mask_area_ratio=self.mask_area_ratio,
            warnings=list(self.warnings),
        )


@dataclass(frozen=True)
class GarmentCleanupResult:
    garment_image: Image.Image
    cutout_mask: Image.Image
    normalized_reference: Image.Image
    bbox_xyxy: tuple[int, int, int, int] | None
    warnings: list[str]


def _load_semantic_map(path: Path, size: tuple[int, int]) -> np.ndarray:
    semantic = Image.open(path)
    if semantic.mode == "P":
        pass
    elif semantic.mode not in {"L", "I", "I;16"}:
        semantic = semantic.convert("L")
    if semantic.size != size:
        semantic = semantic.resize(size, Image.Resampling.NEAREST)
    return np.array(semantic)


def _labels_to_mask(semantic: np.ndarray, labels: list[int]) -> Image.Image:
    selected = np.isin(semantic, np.array(labels, dtype=semantic.dtype))
    return Image.fromarray((selected.astype(np.uint8) * 255), mode="L")


def _bbox_from_mask(mask: Image.Image, padding_px: int = 0) -> tuple[int, int, int, int] | None:
    bbox = binarize(mask).getbbox()
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    width, height = mask.size
    return (
        max(0, x0 - padding_px),
        max(0, y0 - padding_px),
        min(width, x1 + padding_px),
        min(height, y1 + padding_px),
    )


def _subtract(base: Image.Image, remove: Image.Image) -> Image.Image:
    base_arr = np.array(binarize(base), dtype=np.uint8)
    remove_arr = np.array(binarize(remove.resize(base.size, Image.Resampling.NEAREST)), dtype=np.uint8)
    base_arr[remove_arr > 0] = 0
    return Image.fromarray(base_arr, mode="L")


def _intersect(left: Image.Image, right: Image.Image) -> Image.Image:
    left_arr = np.array(binarize(left), dtype=np.uint8) > 0
    right_arr = np.array(binarize(right.resize(left.size, Image.Resampling.NEAREST)), dtype=np.uint8) > 0
    return Image.fromarray(((left_arr & right_arr).astype(np.uint8) * 255), mode="L")


def _mask_iou(left: Image.Image, right: Image.Image) -> float:
    left_arr = np.array(binarize(left), dtype=np.uint8) > 0
    right_arr = np.array(binarize(right.resize(left.size, Image.Resampling.NEAREST)), dtype=np.uint8) > 0
    union = left_arr | right_arr
    if not union.any():
        return 0.0
    return float((left_arr & right_arr).sum()) / float(union.sum())


def _dilate(mask: Image.Image, px: int) -> Image.Image:
    if px <= 0:
        return binarize(mask)
    return binarize(mask).filter(ImageFilter.MaxFilter(px * 2 + 1))


def _skin_protect_mask(image: Image.Image, target_region: str) -> Image.Image:
    """Fallback protect mask when no semantic labels are available.

    This is intentionally conservative. It protects likely face/hands/skin from
    broad clothing masks and should not be used as the only semantic source.
    """

    width, height = image.size
    protected = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(protected)
    if target_region != "hat":
        draw.rectangle((0, 0, width, int(height * 0.20)), fill=255)

    ycbcr = np.array(image.convert("YCbCr"), dtype=np.uint8)
    cb = ycbcr[:, :, 1]
    cr = ycbcr[:, :, 2]
    skin = (cb >= 77) & (cb <= 127) & (cr >= 133) & (cr <= 173)
    yy, xx = np.indices((height, width))
    if target_region in {"upper", "dress", "accessory"}:
        side_body = ((xx < width * 0.35) | (xx > width * 0.65)) & (yy > height * 0.25) & (yy < height * 0.86)
        protected = ImageChops.lighter(
            protected,
            Image.fromarray(((skin & side_body).astype(np.uint8) * 255), mode="L")
            .filter(ImageFilter.MaxFilter(19))
            .filter(ImageFilter.GaussianBlur(3)),
        )
    if target_region == "hat":
        draw.rectangle((int(width * 0.32), int(height * 0.12), int(width * 0.68), int(height * 0.34)), fill=255)
    return binarize(protected)


def create_semantic_target_mask(
    semantic_map_path: str | Path,
    person_size: tuple[int, int],
    target_region: TargetRegion,
    *,
    target_label_map: dict[str, list[int]] | None = None,
) -> Image.Image:
    label_map = target_label_map or LIP_TARGET_LABELS
    labels = label_map.get(target_region)
    if not labels:
        raise ValueError(f"No semantic labels configured for target_region='{target_region}'")
    semantic = _load_semantic_map(Path(semantic_map_path), person_size)
    mask = _labels_to_mask(semantic, labels)
    if mask.getbbox() is None:
        raise ValueError(f"Semantic target mask is empty for target_region='{target_region}'")
    return mask


def create_semantic_protect_mask(
    semantic_map_path: str | Path,
    person_size: tuple[int, int],
    target_region: TargetRegion,
    *,
    protect_label_map: dict[str, list[int]] | None = None,
) -> Image.Image:
    label_map = protect_label_map or LIP_PROTECT_LABELS
    labels = label_map.get(target_region, [])
    if not labels:
        return Image.new("L", person_size, 0)
    semantic = _load_semantic_map(Path(semantic_map_path), person_size)
    return _labels_to_mask(semantic, labels)


def refine_mask_with_sam(
    person_image: Image.Image,
    mask: Image.Image,
    config: HybridMaskConfig,
) -> tuple[Image.Image | None, list[str]]:
    warnings: list[str] = []
    bbox = _bbox_from_mask(mask, config.sam_bbox_padding_px)
    if bbox is None:
        return None, ["SAM refine skipped because input mask is empty."]

    masker = SAMMasker(config.sam)
    sam_mask = masker.create_mask_from_box(person_image, bbox)
    if sam_mask is None:
        reason = masker.unavailable_reason or "SAM unavailable."
        return None, [f"SAM refine skipped: {reason}"]

    refined = binarize(sam_mask)
    if config.intersect_sam_with_semantic_envelope:
        envelope = _dilate(mask, config.semantic_envelope_dilation_px)
        refined = _intersect(refined, envelope)
        if refined.getbbox() is None:
            warnings.append("SAM/envelope intersection was empty; keeping pre-SAM mask.")
            return None, warnings
    original_area = mask_area_ratio(mask)
    refined_area = mask_area_ratio(refined)
    area_factor = refined_area / max(original_area, 1e-8)
    iou = _mask_iou(mask, refined)
    if iou < config.sam_accept_min_iou or area_factor < config.sam_accept_min_area_ratio or area_factor > config.sam_accept_max_area_ratio:
        warnings.append(
            "SAM refine rejected; keeping parser/manual mask "
            f"(iou={iou:.3f}, area_factor={area_factor:.3f})."
        )
        return None, warnings
    return refined, warnings


def build_hybrid_vton_mask(
    person_image: Image.Image,
    target_region: TargetRegion,
    config: HybridMaskConfig | None = None,
    *,
    semantic_map_path: str | Path | None = None,
    manual_mask_path: str | Path | None = None,
    sam_box_xyxy: tuple[int, int, int, int] | None = None,
) -> HybridMaskResult:
    config = config or HybridMaskConfig()
    warnings: list[str] = []
    person = person_image.convert("RGB")
    source: MaskSource

    raw_mask: Image.Image | None = None
    protect_mask = Image.new("L", person.size, 0)

    if config.prefer_semantic and semantic_map_path is not None and Path(semantic_map_path).exists():
        raw_mask = create_semantic_target_mask(
            semantic_map_path,
            person.size,
            target_region,
            target_label_map=config.target_label_map,
        )
        if config.subtract_semantic_protect:
            protect_mask = create_semantic_protect_mask(
                semantic_map_path,
                person.size,
                target_region,
                protect_label_map=config.protect_label_map,
            )
            raw_mask = _subtract(raw_mask, protect_mask)
        source = "semantic"
    elif manual_mask_path is not None:
        raw_mask = load_manual_mask(manual_mask_path, person.size)
        if config.subtract_semantic_protect:
            if semantic_map_path is not None and Path(semantic_map_path).exists():
                protect_mask = create_semantic_protect_mask(
                    semantic_map_path,
                    person.size,
                    target_region,
                    protect_label_map=config.protect_label_map,
                )
            else:
                protect_mask = _skin_protect_mask(person, target_region)
            raw_mask = _subtract(raw_mask, protect_mask)
        source = "manual"
        warnings.append("Using manual mask fallback; semantic human parsing was not available.")
    elif sam_box_xyxy is not None:
        masker = SAMMasker(config.sam)
        raw_mask = masker.create_mask_from_box(person, sam_box_xyxy)
        if raw_mask is None:
            reason = masker.unavailable_reason or "SAM unavailable."
            raise RuntimeError(f"SAM box mask requested but unavailable: {reason}")
        protect_mask = _skin_protect_mask(person, target_region)
        raw_mask = _subtract(raw_mask, protect_mask)
        source = "sam_box"
        warnings.append("Using SAM box fallback; no semantic human parsing mask was available.")
    else:
        raise ValueError(
            "No usable mask source. Provide semantic_map_path, manual_mask_path, or sam_box_xyxy. "
            "Rectangle masks are intentionally not part of the production hybrid masking module."
        )

    raw_mask = binarize(raw_mask)
    if raw_mask.getbbox() is None:
        raise ValueError(f"Hybrid mask is empty after protection subtraction for target_region='{target_region}'")

    boundary_refined: Image.Image | None = None
    if config.refine_with_sam and source in {"semantic", "manual"}:
        boundary_refined, sam_warnings = refine_mask_with_sam(person, raw_mask, config)
        warnings.extend(sam_warnings)
    raw_for_postprocess = boundary_refined if boundary_refined is not None else raw_mask

    artifacts = postprocess_mask(person, raw_for_postprocess, target_region, config.postprocess, protect_mask)
    warnings.extend(artifacts.warnings)
    ratio = mask_area_ratio(binarize(artifacts.processed_mask))
    if ratio < config.min_area_ratio:
        warnings.append(f"mask area below hybrid minimum: area_ratio={ratio:.4f}")
    if ratio > config.max_area_ratio:
        warnings.append(f"mask area above hybrid maximum: area_ratio={ratio:.4f}")

    return HybridMaskResult(
        source=source,
        raw_mask=raw_mask,
        processed_mask=artifacts.processed_mask,
        overlay=artifacts.overlay,
        protect_mask=protect_mask,
        boundary_refined_mask=boundary_refined,
        mask_area_ratio=ratio,
        bbox_xyxy=_bbox_from_mask(artifacts.processed_mask),
        warnings=warnings,
    )


def clean_garment_reference(
    garment_image: Image.Image,
    target_size: tuple[int, int] = (768, 768),
    *,
    diff_threshold: int = 12,
    feather_px: int = 2,
) -> GarmentCleanupResult:
    """Lightweight garment reference cleanup for Flux Redux inputs.

    This is a local deterministic fallback for alpha/background cutout. It is not
    a replacement for BiRefNet/U2-Net, but it provides the same module contract so
    a stronger cutout model can be swapped in later.
    """

    warnings: list[str] = []
    image = garment_image.convert("RGBA")
    if "A" in image.getbands() and np.array(image.getchannel("A")).min() < 255:
        mask = binarize(image.getchannel("A"))
    else:
        rgb = image.convert("RGB")
        bg = Image.new("RGB", rgb.size, rgb.getpixel((0, 0)))
        diff = ImageChops.difference(rgb, bg).convert("L")
        mask = binarize(diff, diff_threshold)
        if mask.getbbox() is None:
            warnings.append("garment cutout fallback could not separate background; using full image.")
            mask = Image.new("L", rgb.size, 255)

    if feather_px > 0:
        soft_mask = mask.filter(ImageFilter.GaussianBlur(feather_px))
    else:
        soft_mask = mask
    bbox = _bbox_from_mask(mask)
    crop = image.convert("RGB").crop(bbox) if bbox else image.convert("RGB")
    crop_mask = soft_mask.crop(bbox) if bbox else soft_mask
    crop.thumbnail(target_size, Image.Resampling.LANCZOS)
    crop_mask.thumbnail(target_size, Image.Resampling.LANCZOS)

    normalized = Image.new("RGB", target_size, (255, 255, 255))
    x = (target_size[0] - crop.width) // 2
    y = (target_size[1] - crop.height) // 2
    normalized.paste(crop, (x, y), crop_mask)

    cutout = Image.new("RGB", image.size, (255, 255, 255))
    cutout.paste(image.convert("RGB"), (0, 0), soft_mask)
    return GarmentCleanupResult(
        garment_image=cutout,
        cutout_mask=mask,
        normalized_reference=normalized,
        bbox_xyxy=bbox,
        warnings=warnings,
    )
