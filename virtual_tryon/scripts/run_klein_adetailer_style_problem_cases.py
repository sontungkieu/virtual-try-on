from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VTON_ROOT = PROJECT_ROOT / "virtual_tryon"
KLEIN_ROOT = VTON_ROOT / "data/outputs/vton_phase2_extra_cases_20260623/klein9b_lora"
MASK_ROOT = VTON_ROOT / "data/outputs/schp_sam_masks_extent_v2_20260626"
OUT_ROOT = VTON_ROOT / "data/outputs/klein_local_mask_composite_problem_cases_20260626"

PROBLEM_CASES: dict[str, list[str]] = {
    "sample_010": ["lower"],
    "sample_012": ["lower", "hat"],
    "sample_013": ["hat", "accessory"],
    "sample_014": ["upper", "lower", "hat"],
    "sample_015": ["dress", "shoes", "hat"],
}

VARIANTS: dict[str, Path] = {
    "klein_4step": Path("klein9b_lora_4step/result.png"),
    "klein_28": Path("klein9b_lora_28_default/result.png"),
    "klein_28_strong": Path("klein9b_lora_28_strong/result.png"),
}


def open_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def open_mask(path: Path, size: tuple[int, int]) -> Image.Image:
    mask = Image.open(path).convert("L").resize(size, Image.Resampling.NEAREST)
    return mask.point(lambda p: 255 if p > 8 else 0)


def mask_area_ratio(mask: Image.Image) -> float:
    arr = np.asarray(mask.convert("L"))
    return float((arr > 8).mean())


def postprocess_mask(mask: Image.Image, dilation: int = 4, blur: int = 6) -> Image.Image:
    out = mask.convert("L").point(lambda p: 255 if p > 8 else 0)
    if dilation > 0:
        out = out.filter(ImageFilter.MaxFilter(dilation * 2 + 1))
    if blur > 0:
        out = out.filter(ImageFilter.GaussianBlur(blur))
    return out


def union_masks(paths: Iterable[Path], size: tuple[int, int]) -> Image.Image:
    union = Image.new("L", size, 0)
    for path in paths:
        union = ImageChops.lighter(union, open_mask(path, size))
    return postprocess_mask(union)


def overlay(image: Image.Image, mask: Image.Image, color: tuple[int, int, int] = (36, 140, 255)) -> Image.Image:
    base = image.convert("RGBA")
    layer = Image.new("RGBA", base.size, (*color, 0))
    alpha = mask.convert("L").point(lambda p: int(p * 0.42))
    layer.putalpha(alpha)
    return Image.alpha_composite(base, layer).convert("RGB")


def seam_mask(mask: Image.Image, dilation: int = 8, erosion: int = 3, blur: int = 6) -> Image.Image:
    binary = mask.convert("L").point(lambda p: 255 if p > 8 else 0)
    dilated = binary.filter(ImageFilter.MaxFilter(dilation * 2 + 1)) if dilation > 0 else binary
    eroded = binary.filter(ImageFilter.MinFilter(erosion * 2 + 1)) if erosion > 0 else binary
    edge = ImageChops.subtract(dilated, eroded)
    if blur > 0:
        edge = edge.filter(ImageFilter.GaussianBlur(blur))
    return edge


def local_seam_repair(image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
    edge = seam_mask(mask)
    enhanced = ImageEnhance.Sharpness(image).enhance(1.12)
    enhanced = ImageEnhance.Contrast(enhanced).enhance(1.03)
    repaired = Image.composite(enhanced, image, edge)
    return repaired, edge


def placeholder(text: str, size: tuple[int, int] = (260, 340)) -> Image.Image:
    image = Image.new("RGB", size, (245, 245, 245))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, size[0] - 1, size[1] - 1), outline=(160, 160, 160))
    draw.multiline_text((12, 14), text, fill=(40, 40, 40), spacing=4)
    return image


def fit_thumb(path: Path | None, size: tuple[int, int], label: str = "") -> Image.Image:
    if path is None or not path.exists():
        return placeholder(f"missing\n{label}", size)
    image = open_rgb(path)
    image.thumbnail((size[0] - 16, size[1] - 40), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(image, ((size[0] - image.width) // 2, 30 + (size[1] - 40 - image.height) // 2))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), label, fill=(0, 0, 0))
    return canvas


def build_contact_sheet(rows: list[dict], sheet_path: Path) -> None:
    columns = [
        ("person", "person"),
        ("candidate", "klein candidate"),
        ("mask_overlay", "mask overlay"),
        ("output_base", "localized composite"),
        ("output_refined", "local seam repaired"),
    ]
    cell = (250, 340)
    label_w = 190
    sheet = Image.new("RGB", (label_w + cell[0] * len(columns), cell[1] * (len(rows) + 1)), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 10), "sample / variant", fill=(0, 0, 0))
    for col_idx, (_, title) in enumerate(columns):
        draw.text((label_w + col_idx * cell[0] + 8, 10), title, fill=(0, 0, 0))
    for row_idx, row in enumerate(rows, start=1):
        y = row_idx * cell[1]
        draw.rectangle((0, y, label_w - 1, y + cell[1] - 1), outline=(220, 220, 220))
        draw.multiline_text(
            (8, y + 12),
            f"{row['sample_id']}\n{row['variant']}\nregions: {', '.join(row['regions'])}",
            fill=(0, 0, 0),
            spacing=5,
        )
        for col_idx, (key, title) in enumerate(columns):
            thumb = fit_thumb(Path(row[key]) if row.get(key) else None, cell, title)
            sheet.paste(thumb, (label_w + col_idx * cell[0], y))
    sheet.save(sheet_path)


def write_html(rows: list[dict], out_dir: Path) -> None:
    html_rows = []
    for row in rows:
        cells = [
            f"<td><b>{html.escape(row['sample_id'])}</b><br>{html.escape(row['variant'])}<br>"
            f"regions: {html.escape(', '.join(row['regions']))}<br>"
            f"mask area: {row['mask_area_ratio']:.4f}</td>"
        ]
        for key in ["person", "candidate", "mask_overlay", "output_base", "output_refined"]:
            rel = Path(row[key]).relative_to(out_dir).as_posix()
            cells.append(f'<td><a href="{rel}"><img src="{rel}"></a></td>')
        html_rows.append("<tr>" + "".join(cells) + "</tr>")
    page = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>Klein + local mask composite problem case report</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; }}
table {{ border-collapse: collapse; }}
td, th {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
img {{ max-width: 220px; max-height: 300px; }}
.note {{ color: #444; max-width: 980px; }}
</style></head><body>
<h1>Klein + mask-guided local composite report</h1>
<p class="note">This experiment reuses existing Klein candidates, composites only the SCHP/SAM target mask region back onto the original person, then applies a non-diffusion local seam/detail repair. It is not ADetailer.</p>
<p><a href="klein_local_mask_composite_contact_sheet.png">Open contact sheet</a></p>
<table>
<tr><th>case</th><th>person</th><th>candidate</th><th>mask</th><th>composite</th><th>refined</th></tr>
{''.join(html_rows)}
</table>
</body></html>"""
    (out_dir / "report.html").write_text(page, encoding="utf-8")


def run_case(sample_id: str, regions: list[str], out_dir: Path) -> list[dict]:
    sample_dir = KLEIN_ROOT / sample_id
    person_path = sample_dir / "person_reference.png"
    if not person_path.exists():
        raise FileNotFoundError(f"Missing person image for {sample_id}: {person_path}")
    person = open_rgb(person_path)
    case_dir = out_dir / sample_id
    case_dir.mkdir(parents=True, exist_ok=True)
    person.save(case_dir / "person.png")

    mask_paths = [MASK_ROOT / sample_id / region / "mask_processed.png" for region in regions]
    missing_masks = [str(path) for path in mask_paths if not path.exists()]
    if missing_masks:
        raise FileNotFoundError(f"Missing masks for {sample_id}: {missing_masks}")
    mask = union_masks(mask_paths, person.size)
    mask_path = case_dir / "mask_union_processed.png"
    mask_overlay_path = case_dir / "mask_overlay.png"
    mask.save(mask_path)
    overlay(person, mask).save(mask_overlay_path)

    rows: list[dict] = []
    for variant, rel_candidate in VARIANTS.items():
        variant_dir = case_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        candidate_path = sample_dir / rel_candidate
        if not candidate_path.exists():
            rows.append(
                {
                    "sample_id": sample_id,
                    "variant": variant,
                    "regions": regions,
                    "person": str(case_dir / "person.png"),
                    "candidate": "",
                    "mask_overlay": str(mask_overlay_path),
                    "output_base": "",
                    "output_refined": "",
                    "mask_area_ratio": mask_area_ratio(mask),
                    "flags": ["missing candidate"],
                }
            )
            continue

        candidate = open_rgb(candidate_path)
        if candidate.size != person.size:
            candidate = candidate.resize(person.size, Image.Resampling.LANCZOS)
        candidate_out = variant_dir / "candidate_klein.png"
        candidate.save(candidate_out)
        base = Image.composite(candidate, person, mask)
        refined, edge = local_seam_repair(base, mask)
        base_path = variant_dir / "output_base.png"
        refined_path = variant_dir / "output_refined.png"
        edge_path = variant_dir / "seam_detail_mask.png"
        base.save(base_path)
        refined.save(refined_path)
        edge.save(edge_path)
        overlay(base, edge, color=(255, 90, 30)).save(variant_dir / "seam_detail_overlay.png")
        row = {
            "sample_id": sample_id,
            "variant": variant,
            "regions": regions,
            "person": str(case_dir / "person.png"),
            "candidate": str(candidate_out),
            "mask_union": str(mask_path),
            "mask_overlay": str(mask_overlay_path),
            "output_base": str(base_path),
            "output_refined": str(refined_path),
            "seam_detail_mask": str(edge_path),
            "mask_area_ratio": mask_area_ratio(mask),
            "flags": [],
        }
        (variant_dir / "metadata.json").write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
        rows.append(row)

    metadata = {
        "sample_id": sample_id,
        "regions": regions,
        "person": str(person_path),
        "mask_paths": [str(path) for path in mask_paths],
        "mask_union": str(mask_path),
        "mask_area_ratio": mask_area_ratio(mask),
        "variants": [row["variant"] for row in rows],
    }
    (case_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Klein + mask-guided local composite localization tests.")
    parser.add_argument("--out-dir", type=Path, default=OUT_ROOT)
    parser.add_argument(
        "--samples",
        nargs="*",
        default=list(PROBLEM_CASES.keys()),
        help="Sample ids to process. Default: known problem cases.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    failures: list[dict] = []
    for sample_id in args.samples:
        regions = PROBLEM_CASES.get(sample_id)
        if not regions:
            failures.append({"sample_id": sample_id, "error": "No target regions configured for this script."})
            continue
        try:
            rows.extend(run_case(sample_id, regions, out_dir))
        except Exception as exc:
            failures.append({"sample_id": sample_id, "error": f"{type(exc).__name__}: {exc}"})

    build_contact_sheet(rows, out_dir / "klein_local_mask_composite_contact_sheet.png")
    write_html(rows, out_dir)
    summary = {
        "method": "klein_candidate_mask_composite_local_seam_repair",
        "out_dir": str(out_dir),
        "case_count": len(set(row["sample_id"] for row in rows)),
        "row_count": len(rows),
        "failures": failures,
        "contact_sheet": str(out_dir / "klein_local_mask_composite_contact_sheet.png"),
        "report": str(out_dir / "report.html"),
        "note": "Fast localization/debug experiment. It does not rerun Klein diffusion; it post-processes existing Klein candidates using target masks. It is not ADetailer.",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
