from __future__ import annotations

import json
import shutil
import sys
import tempfile
import argparse
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from virtual_tryon.final_demo import DEFAULT_FINAL_EVAL_ROOT, GARMENT_FILE_BY_REGION
from virtual_tryon.masking import (
    HybridMaskConfig,
    HybridMaskResult,
    MaskPostprocessConfig,
    SAMMaskConfig,
    build_hybrid_vton_mask,
    create_target_extent_mask,
    should_use_target_extent_fallback,
)


PROJECT_ROOT = Path("/workspace/Project_Phase2")
EVAL_ROOT = DEFAULT_FINAL_EVAL_ROOT
OUT_ROOT = PROJECT_ROOT / "virtual_tryon/data/outputs/schp_sam_masks_20260626"
TEMP_ROOT = PROJECT_ROOT / "virtual_tryon/data/temp/schp_sam_masks"
MANUAL_MASK_ROOT = PROJECT_ROOT / "virtual_tryon/data/temp/precise_target_masks"
PARSER_ROOT = PROJECT_ROOT / "virtual_tryon/third_party/IDM-VTON/preprocess/humanparsing"
SAM_CHECKPOINT = PROJECT_ROOT / "virtual_tryon/models/sam/sam_vit_b_01ec64.pth"
REAL_PARSER_CKPT = PROJECT_ROOT / "virtual_tryon/models/idm_vton/ckpt/humanparsing"
IDM_PARSER_CKPT = PROJECT_ROOT / "virtual_tryon/third_party/IDM-VTON/ckpt/humanparsing"

REGION_BY_ITEM = {file_name: region for region, file_name in GARMENT_FILE_BY_REGION.items()}


def ensure_parser_checkpoint_links() -> None:
    IDM_PARSER_CKPT.mkdir(parents=True, exist_ok=True)
    for name in ["parsing_atr.onnx", "parsing_lip.onnx"]:
        source = REAL_PARSER_CKPT / name
        target = IDM_PARSER_CKPT / name
        if not source.exists() or source.stat().st_size < 1_000_000:
            raise FileNotFoundError(f"Real human parsing checkpoint is missing or invalid: {source}")
        if target.exists() or target.is_symlink():
            if target.stat().st_size >= 1_000_000:
                continue
            target.unlink()
        target.symlink_to(source)


def run_idm_parser(person_path: Path, sample_id: str) -> Path:
    ensure_parser_checkpoint_links()
    sys.path.insert(0, str(PARSER_ROOT))
    from run_parsing import Parsing  # type: ignore

    semantic_dir = TEMP_ROOT / sample_id
    semantic_dir.mkdir(parents=True, exist_ok=True)
    semantic_path = semantic_dir / f"{sample_id}_semantic_atr.png"
    if semantic_path.exists():
        return semantic_path

    with tempfile.TemporaryDirectory(prefix=f"{sample_id}_parser_") as tmp:
        tmp_dir = Path(tmp)
        shutil.copy2(person_path, tmp_dir / person_path.name)
        parser = Parsing(0)
        parsed_image, _ = parser(tmp_dir.as_posix())
        parsed_image.save(semantic_path)
    return semantic_path


def save_label_debug(semantic_path: Path, out_dir: Path) -> dict[str, int]:
    semantic = Image.open(semantic_path)
    arr = semantic if semantic.mode == "P" else semantic.convert("L")
    counts: dict[str, int] = {}
    hist = arr.histogram()
    for label, count in enumerate(hist):
        if count:
            counts[str(label)] = int(count)
    semantic.save(out_dir / "semantic_atr.png")
    return counts


def build_grid(sample_id: str, rows: list[dict[str, Any]], out_dir: Path) -> Path:
    headers = ["person", "semantic", "protect", "raw", "processed", "overlay"]
    cell_w, cell_h, header_h = 230, 280, 38
    canvas = Image.new("RGB", (cell_w * len(headers), header_h + cell_h * len(rows)), "white")
    draw = ImageDraw.Draw(canvas)
    for c, header in enumerate(headers):
        draw.rectangle((c * cell_w, 0, (c + 1) * cell_w, header_h), fill=(235, 238, 242), outline=(210, 215, 220))
        draw.text((c * cell_w + 8, 12), header, fill=(20, 24, 32))
    for r, row in enumerate(rows):
        y0 = header_h + r * cell_h
        draw.text((8, y0 + 8), f"{sample_id}/{row['region']} ratio={row['ratio']:.4f}", fill=(0, 0, 0))
        for c, key in enumerate(["person", "semantic", "protect", "raw", "processed", "overlay"]):
            path = Path(row[key])
            x0 = c * cell_w
            draw.rectangle((x0, y0, x0 + cell_w, y0 + cell_h), outline=(222, 226, 232))
            if not path.exists():
                draw.text((x0 + 10, y0 + 48), "missing", fill=(180, 0, 0))
                continue
            image = Image.open(path).convert("RGB")
            image.thumbnail((cell_w - 20, cell_h - 46), Image.Resampling.LANCZOS)
            canvas.paste(image, (x0 + (cell_w - image.width) // 2, y0 + 36 + (cell_h - 46 - image.height) // 2))
    path = out_dir / "mask_grid.png"
    canvas.save(path)
    return path


def infer_regions(sample_dir: Path) -> list[str]:
    regions: list[str] = []
    metadata_path = sample_dir / "metadata.json"
    item_names: list[str] = []
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        item_names = [str(name) for name in metadata.get("items", [])]
    if not item_names:
        item_names = [name for name in REGION_BY_ITEM if (sample_dir / name).exists()]
    for item_name in item_names:
        region = REGION_BY_ITEM.get(item_name)
        if region and region not in regions:
            regions.append(region)
    if not regions:
        category = ""
        if metadata_path.exists():
            category = str(json.loads(metadata_path.read_text(encoding="utf-8")).get("category", ""))
        if category == "upper_body":
            regions.append("upper")
        elif category == "lower_body":
            regions.append("lower")
        elif category == "dress":
            regions.append("dress")
    return regions


def find_manual_fallback(sample_dir: Path, sample_id: str, region: str) -> Path | None:
    candidates: list[Path] = []
    if region == "hat":
        candidates.append(MANUAL_MASK_ROOT / sample_id / f"{sample_id}_hat_strict_mask.png")
    candidates.append(MANUAL_MASK_ROOT / sample_id / f"{sample_id}_{region}_mask.png")
    candidates.append(sample_dir / f"{region}_mask.png")
    candidates.append(sample_dir / f"mask_{region}.png")
    if region not in {"hat", "accessory", "shoes"}:
        candidates.append(sample_dir / "mask.png")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def create_heuristic_fallback_mask(person: Image.Image, region: str) -> Image.Image:
    width, height = person.size
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    if region == "upper":
        draw.rounded_rectangle(
            (int(width * 0.24), int(height * 0.20), int(width * 0.76), int(height * 0.58)),
            radius=max(8, width // 30),
            fill=255,
        )
    elif region == "lower":
        draw.rounded_rectangle(
            (int(width * 0.26), int(height * 0.43), int(width * 0.74), int(height * 0.84)),
            radius=max(8, width // 30),
            fill=255,
        )
    elif region == "dress":
        draw.polygon(
            [
                (int(width * 0.30), int(height * 0.20)),
                (int(width * 0.70), int(height * 0.20)),
                (int(width * 0.82), int(height * 0.82)),
                (int(width * 0.18), int(height * 0.82)),
            ],
            fill=255,
        )
    elif region == "shoes":
        draw.rounded_rectangle(
            (int(width * 0.20), int(height * 0.82), int(width * 0.80), int(height * 0.99)),
            radius=max(4, width // 40),
            fill=255,
        )
    elif region == "hat":
        draw.ellipse(
            (int(width * 0.28), int(height * 0.01), int(width * 0.72), int(height * 0.25)),
            fill=255,
        )
    elif region == "accessory":
        draw.ellipse((int(width * 0.14), int(height * 0.45), int(width * 0.32), int(height * 0.64)), fill=255)
        draw.ellipse((int(width * 0.68), int(height * 0.45), int(width * 0.86), int(height * 0.64)), fill=255)
    else:
        draw.rectangle((int(width * 0.25), int(height * 0.25), int(width * 0.75), int(height * 0.75)), fill=255)
    return mask


def _with_extra_warnings(result: HybridMaskResult, warnings: list[str]) -> HybridMaskResult:
    return result.__class__(
        source=result.source,
        raw_mask=result.raw_mask,
        processed_mask=result.processed_mask,
        overlay=result.overlay,
        protect_mask=result.protect_mask,
        boundary_refined_mask=result.boundary_refined_mask,
        mask_area_ratio=result.mask_area_ratio,
        bbox_xyxy=result.bbox_xyxy,
        warnings=[*warnings, *result.warnings],
    )


def _fallback_config(base: HybridMaskConfig, region: str) -> HybridMaskConfig:
    return HybridMaskConfig.atr(
        postprocess=base.postprocess,
        sam=base.sam,
        refine_with_sam=base.refine_with_sam,
        sam_bbox_padding_px=base.sam_bbox_padding_px,
        intersect_sam_with_semantic_envelope=base.intersect_sam_with_semantic_envelope,
        semantic_envelope_dilation_px=base.semantic_envelope_dilation_px,
        prefer_semantic=False,
        subtract_semantic_protect=region not in {"hat", "accessory", "shoes"},
        max_area_ratio=base.max_area_ratio,
    )


def _build_manual_or_extent_fallback(
    *,
    person: Image.Image,
    sample_id: str,
    region: str,
    semantic_path: Path,
    manual_fallback: Path | None,
    temp_dir: Path,
    config: HybridMaskConfig,
    reason: str,
) -> HybridMaskResult:
    fallback_warnings: list[str] = []
    fallback_path = manual_fallback
    fallback_label = "manual"
    if fallback_path is None:
        try:
            extent = create_target_extent_mask(person, region, semantic_map_path=semantic_path)  # type: ignore[arg-type]
            fallback_path = temp_dir / f"{sample_id}_{region}_target_extent_mask.png"
            extent.mask.save(fallback_path)
            fallback_label = "target-extent"
            fallback_warnings.extend(extent.warnings)
        except Exception as exc:
            fallback_path = temp_dir / f"{sample_id}_{region}_heuristic_fallback.png"
            create_heuristic_fallback_mask(person, region).save(fallback_path)
            fallback_label = "legacy-heuristic"
            fallback_warnings.append(f"Target extent fallback failed ({type(exc).__name__}: {exc}); used legacy heuristic.")

    result = build_hybrid_vton_mask(
        person,
        region,  # type: ignore[arg-type]
        _fallback_config(config, region),
        semantic_map_path=semantic_path,
        manual_mask_path=fallback_path,
    )
    return _with_extra_warnings(
        result,
        [
            f"{reason}; used {fallback_label} fallback mask {fallback_path.as_posix()}.",
            *fallback_warnings,
        ],
    )


def process_sample(sample_id: str, regions: list[str] | None = None) -> dict[str, Any]:
    sample_dir = EVAL_ROOT / sample_id
    person_path = sample_dir / "person.png"
    person = Image.open(person_path).convert("RGB")
    regions = regions or infer_regions(sample_dir)
    if not regions:
        raise ValueError(f"Could not infer target regions for {sample_id}")

    out_dir = OUT_ROOT / sample_id
    temp_dir = TEMP_ROOT / sample_id
    out_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(person_path, out_dir / "person.png")

    semantic_path = run_idm_parser(person_path, sample_id)
    label_counts = save_label_debug(semantic_path, out_dir)
    config = HybridMaskConfig.atr(
        postprocess=MaskPostprocessConfig(dilation_px=2, erosion_px=0, feather_px=2, remove_face_hair=True, remove_hands=True),
        sam=SAMMaskConfig(checkpoint_path=SAM_CHECKPOINT, model_type="vit_b", device="cuda"),
        refine_with_sam=True,
        sam_bbox_padding_px=8,
        intersect_sam_with_semantic_envelope=True,
        semantic_envelope_dilation_px=14,
        max_area_ratio=0.50,
    )

    rows: list[dict[str, Any]] = []
    masks_meta: dict[str, Any] = {}
    for region in regions:
        manual_fallback = find_manual_fallback(sample_dir, sample_id, region)
        try:
            result = build_hybrid_vton_mask(
                person,
                region,  # type: ignore[arg-type]
                config,
                semantic_map_path=semantic_path,
            )
            if should_use_target_extent_fallback(region, result.mask_area_ratio, result.warnings):
                result = _build_manual_or_extent_fallback(
                    person=person,
                    sample_id=sample_id,
                    region=region,
                    semantic_path=semantic_path,
                    manual_fallback=None,
                    temp_dir=temp_dir,
                    config=config,
                    reason=(
                        "Semantic SCHP/ATR mask was available but too small for the desired try-on target "
                        f"(area_ratio={result.mask_area_ratio:.4f})"
                    ),
                )
        except ValueError as exc:
            result = _build_manual_or_extent_fallback(
                person=person,
                sample_id=sample_id,
                region=region,
                semantic_path=semantic_path,
                manual_fallback=manual_fallback,
                temp_dir=temp_dir,
                config=config,
                reason=f"Semantic target unavailable ({exc})",
            )
        region_dir = out_dir / region
        region_temp = temp_dir
        region_dir.mkdir(parents=True, exist_ok=True)
        raw_path = region_dir / "mask_raw.png"
        processed_path = region_temp / f"{sample_id}_{region}_mask.png"
        processed_preview_path = region_dir / "mask_processed.png"
        overlay_path = region_dir / "mask_overlay.png"
        protect_path = region_dir / "mask_protect.png"
        refined_path = region_dir / "mask_sam_refined.png"
        result.raw_mask.save(raw_path)
        result.processed_mask.save(processed_path)
        result.processed_mask.save(processed_preview_path)
        result.overlay.save(overlay_path)
        result.protect_mask.save(protect_path)
        if result.boundary_refined_mask is not None:
            result.boundary_refined_mask.save(refined_path)
        masks_meta[region] = {
            "source": result.source,
            "semantic_map_path": semantic_path.as_posix(),
            "raw_mask": raw_path.as_posix(),
            "mask_path": processed_path.as_posix(),
            "preview_mask_path": processed_preview_path.as_posix(),
            "overlay_path": overlay_path.as_posix(),
            "protect_mask": protect_path.as_posix(),
            "sam_refined_mask": refined_path.as_posix() if result.boundary_refined_mask is not None else None,
            "area_ratio": round(result.mask_area_ratio, 4),
            "bbox_xyxy": result.bbox_xyxy,
            "warnings": result.warnings,
        }
        rows.append(
            {
                "region": region,
                "ratio": result.mask_area_ratio,
                "person": out_dir / "person.png",
                "semantic": out_dir / "semantic_atr.png",
                "protect": protect_path,
                "raw": raw_path,
                "processed": processed_preview_path,
                "overlay": overlay_path,
            }
        )
    grid_path = build_grid(sample_id, rows, out_dir)
    meta = {
        "sample_id": sample_id,
        "person_path": person_path.as_posix(),
        "semantic_map_path": semantic_path.as_posix(),
        "label_counts": label_counts,
        "grid_path": grid_path.as_posix(),
        "masks": masks_meta,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def write_html_report(all_meta: dict[str, Any], out_root: Path) -> Path:
    rows: list[str] = []
    for sample_id in sorted(k for k in all_meta if k.startswith("sample_")):
        sample_meta = all_meta[sample_id]
        grid_path = Path(sample_meta["grid_path"])
        grid_rel = grid_path.relative_to(out_root).as_posix()
        masks = sample_meta.get("masks", {})
        mask_rows: list[str] = []
        for region, info in masks.items():
            warnings = info.get("warnings") or []
            status = "ok" if info.get("source") else "missing"
            if warnings:
                status = "warning" if status == "ok" else "missing"
            mask_rows.append(
                "<tr>"
                f"<td>{region}</td>"
                f"<td>{info.get('source') or '-'}</td>"
                f"<td>{info.get('area_ratio', '-')}</td>"
                f"<td>{status}</td>"
                f"<td>{'<br>'.join(warnings) if warnings else ''}</td>"
                "</tr>"
            )
        rows.append(
            "<section class='sample'>"
            f"<h2>{sample_id}</h2>"
            f"<a href='{grid_rel}'><img src='{grid_rel}' alt='{sample_id} mask grid'></a>"
            "<table><thead><tr><th>Region</th><th>Source</th><th>Area</th><th>Status</th><th>Warnings</th></tr></thead>"
            f"<tbody>{''.join(mask_rows)}</tbody></table>"
            "</section>"
        )
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>SCHP + SAM Mask Report</title>"
        "<style>"
        "body{font-family:Arial,sans-serif;margin:24px;background:#f7f8fa;color:#17202a}"
        "h1{margin:0 0 8px} .meta{color:#566573;margin-bottom:20px}"
        ".sample{background:#fff;border:1px solid #d8dee8;border-radius:8px;padding:16px;margin:0 0 18px}"
        ".sample img{max-width:100%;border:1px solid #d8dee8;background:#fff}"
        "table{border-collapse:collapse;width:100%;margin-top:12px;font-size:14px}"
        "th,td{border:1px solid #d8dee8;padding:8px;vertical-align:top} th{background:#eef2f7;text-align:left}"
        "</style></head><body>"
        "<h1>SCHP + SAM Hybrid Mask Report</h1>"
        "<div class='meta'>Human parser checkpoint + SAM ViT-B checkpoint. Accessory/hat/shoes require item-specific masks when absent from source parsing.</div>"
        f"{''.join(rows)}"
        "</body></html>"
    )
    report_path = out_root / "mask_report.html"
    report_path.write_text(html, encoding="utf-8")
    return report_path


def main() -> int:
    global EVAL_ROOT, OUT_ROOT, TEMP_ROOT

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, default=EVAL_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--temp-root", type=Path, default=TEMP_ROOT)
    args = parser.parse_args()

    EVAL_ROOT = args.eval_root
    OUT_ROOT = args.output_root
    TEMP_ROOT = args.temp_root

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    all_meta: dict[str, Any] = {
        "sam_checkpoint": SAM_CHECKPOINT.as_posix(),
        "human_parser_checkpoints": [
            (PROJECT_ROOT / "virtual_tryon/models/idm_vton/ckpt/humanparsing/parsing_atr.onnx").as_posix(),
            (PROJECT_ROOT / "virtual_tryon/models/idm_vton/ckpt/humanparsing/parsing_lip.onnx").as_posix(),
        ],
    }
    for sample_dir in sorted(EVAL_ROOT.glob("sample_*")):
        if not (sample_dir / "person.png").exists():
            continue
        all_meta[sample_dir.name] = process_sample(sample_dir.name)
    (OUT_ROOT / "metadata.json").write_text(json.dumps(all_meta, indent=2), encoding="utf-8")
    write_html_report(all_meta, OUT_ROOT)
    print(json.dumps(all_meta, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
