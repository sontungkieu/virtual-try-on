from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VTON_ROOT = PROJECT_ROOT / "virtual_tryon"
OUT_ROOT = VTON_ROOT / "data/outputs/final_project_output"
ARTIFACT_ROOT = OUT_ROOT / "_artifacts"

SCHP_ROOT = VTON_ROOT / "data/outputs/full15_archived_grid_schp_sam_tryon_outputs_20260626"
MASK_ROOT = VTON_ROOT / "data/outputs/schp_sam_masks_extent_v2_20260626"
EXTRA_KLEIN_ROOT = VTON_ROOT / "data/outputs/vton_phase2_extra_cases_20260623/klein9b_lora"
LOCAL_MASKED_ROOT = VTON_ROOT / "data/outputs/klein_local_masked_tryon_full15_20260626"
EVAL_ROOT = VTON_ROOT / "data/temp/full15_archived_grid_eval_set"
ARCHIVED_GRID = VTON_ROOT / "data/outputs/vton_phase2_full_15_cases_20260625/full_15_test_cases_grid.png"

SAMPLE_IDS = [f"sample_{index:03d}" for index in range(1, 16)]
METHOD_LABELS = [
    ("Input image", "input"),
    ("Garment / reference", "garment"),
    ("SCHP/SAM\nFlux Fill + CatVTON", "schp_sam_flux_catvton"),
    ("Klein 28", "klein_28"),
    ("Klein 28 strong", "klein_28_strong"),
    ("Klein+LoRA\nlocal masked inpaint", "klein_local_masked_inpaint"),
]
TILE_SIZE = (260, 360)
HEADER_H = 72
TITLE_H = 54
FOOTER_H = 42


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def fit_tile(image: Image.Image, size: tuple[int, int] = TILE_SIZE) -> Image.Image:
    image = ImageOps.exif_transpose(image.convert("RGB"))
    contained = ImageOps.contain(image, (size[0] - 14, size[1] - 16), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(contained, ((size[0] - contained.width) // 2, (size[1] - contained.height) // 2))
    return canvas


def placeholder(label: str, size: tuple[int, int] = TILE_SIZE) -> Image.Image:
    image = Image.new("RGB", size, (246, 247, 249))
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, size[0] - 8, size[1] - 8), outline=(190, 196, 205), width=2)
    lines = label.split("\n")
    y = size[1] // 2 - (len(lines) - 1) * 13
    for line in lines:
        draw.text((size[0] // 2, y), line, fill=(85, 92, 104), font=font(15, bold=True), anchor="mm")
        y += 28
    return image


def read_rgb(path: Path | None) -> Image.Image | None:
    if path and path.exists():
        return Image.open(path).convert("RGB")
    return None


def read_mask(path: Path, size: tuple[int, int]) -> Image.Image:
    mask = Image.open(path).convert("L").resize(size, Image.Resampling.NEAREST)
    return mask.point(lambda p: 255 if p > 8 else 0)


def mask_area(mask: Image.Image) -> float:
    array = np.asarray(mask.convert("L"))
    return float((array > 8).mean())


def union_masks(sample_id: str, size: tuple[int, int]) -> tuple[Image.Image, list[str]]:
    sample_mask_root = MASK_ROOT / sample_id
    mask_paths = sorted(sample_mask_root.glob("*/mask_processed.png"))
    union = Image.new("L", size, 0)
    for path in mask_paths:
        union = ImageChops.lighter(union, read_mask(path, size))
    union = union.filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.GaussianBlur(6))
    return union, [path.as_posix() for path in mask_paths]


def seam_mask(mask: Image.Image) -> Image.Image:
    binary = mask.convert("L").point(lambda p: 255 if p > 8 else 0)
    dilated = binary.filter(ImageFilter.MaxFilter(17))
    eroded = binary.filter(ImageFilter.MinFilter(7))
    return ImageChops.subtract(dilated, eroded).filter(ImageFilter.GaussianBlur(6))


def local_mask_composite(person: Image.Image, candidate: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
    if candidate.size != person.size:
        candidate = candidate.resize(person.size, Image.Resampling.LANCZOS)
    base = Image.composite(candidate, person, mask)
    edge = seam_mask(mask)
    enhanced = ImageEnhance.Sharpness(base).enhance(1.12)
    enhanced = ImageEnhance.Contrast(enhanced).enhance(1.03)
    refined = Image.composite(enhanced, base, edge)
    return refined, edge


def crop_archived_grid_cell(old_grid: Image.Image, sample_id: str, column: str) -> Image.Image | None:
    number = int(sample_id.split("_")[1])
    if number > 9:
        return None
    x_ranges = {
        "klein_28": (1512, 1731),
        "klein_28_strong": (1762, 1981),
    }
    x0, x1 = x_ranges[column]
    y0 = 75 + (number - 1) * 334
    y1 = y0 + 292
    return old_grid.crop((x0, y0, x1, y1)).convert("RGB")


def klein_image(old_grid: Image.Image, sample_id: str, variant: str) -> tuple[Image.Image | None, str | None]:
    number = int(sample_id.split("_")[1])
    if number <= 9:
        crop = crop_archived_grid_cell(old_grid, sample_id, variant)
        return crop, f"{ARCHIVED_GRID.as_posix()}#crop:{sample_id}:{variant}" if crop is not None else None
    variant_dir = "klein9b_lora_28_default" if variant == "klein_28" else "klein9b_lora_28_strong"
    path = EXTRA_KLEIN_ROOT / sample_id / variant_dir / "result.png"
    return read_rgb(path), path.as_posix() if path.exists() else None


def sample_pass_dirs(sample_id: str) -> list[Path]:
    sample_dir = SCHP_ROOT / sample_id
    return sorted([path for path in sample_dir.glob("pass_*") if path.is_dir()])


def sample_person(sample_id: str) -> tuple[Image.Image | None, str | None]:
    eval_path = EVAL_ROOT / sample_id / "person.png"
    if eval_path.exists():
        return read_rgb(eval_path), eval_path.as_posix()
    pass_dirs = sample_pass_dirs(sample_id)
    if pass_dirs:
        path = pass_dirs[0] / "input_person.png"
        return read_rgb(path), path.as_posix() if path.exists() else None
    fallback = SCHP_ROOT / sample_id / "final_output.png"
    return read_rgb(fallback), fallback.as_posix() if fallback.exists() else None


def garment_canvas(sample_id: str) -> tuple[Image.Image | None, list[str]]:
    eval_sample = EVAL_ROOT / sample_id
    eval_garments = [
        ("top", eval_sample / "garment_top.png"),
        ("bottom", eval_sample / "garment_bottom.png"),
        ("dress", eval_sample / "garment_dress.png"),
        ("shoes", eval_sample / "accessory_shoes.png"),
        ("hat", eval_sample / "accessory_hat.png"),
        ("watch", eval_sample / "accessory_watch.png"),
    ]
    available = [(label, path) for label, path in eval_garments if path.exists()]
    if len(available) == 1:
        label, path = available[0]
        return read_rgb(path), [path.as_posix()]
    if len(available) > 1:
        cell_w = 240
        cell_h = max(150, 330 // len(available))
        canvas = Image.new("RGB", (cell_w, cell_h * len(available)), "white")
        draw = ImageDraw.Draw(canvas)
        small = font(15, bold=True)
        sources: list[str] = []
        for idx, (label, path) in enumerate(available):
            image = Image.open(path).convert("RGB")
            tile = fit_tile(image, (cell_w, cell_h))
            y = idx * cell_h
            canvas.paste(tile, (0, y))
            draw.rectangle((4, y + 4, 105, y + 28), fill=(255, 255, 255))
            draw.text((10, y + 8), label, fill=(20, 24, 32), font=small)
            sources.append(path.as_posix())
        return canvas, sources

    pass_dirs = sample_pass_dirs(sample_id)
    garments: list[tuple[str, Image.Image, str]] = []
    for pass_dir in pass_dirs:
        path = pass_dir / "input_garment.png"
        if not path.exists():
            continue
        region = pass_dir.name.split("_", 2)[-1]
        garments.append((region, Image.open(path).convert("RGB"), path.as_posix()))
    if not garments:
        return None, []
    if len(garments) == 1:
        return garments[0][1], [garments[0][2]]

    cell_w = 240
    cell_h = max(190, 300 // max(1, len(garments)))
    canvas = Image.new("RGB", (cell_w, cell_h * len(garments)), "white")
    draw = ImageDraw.Draw(canvas)
    small = font(15, bold=True)
    sources: list[str] = []
    for idx, (region, image, source) in enumerate(garments):
        tile = fit_tile(image, (cell_w, cell_h))
        y = idx * cell_h
        canvas.paste(tile, (0, y))
        draw.rectangle((4, y + 4, 110, y + 28), fill=(255, 255, 255))
        draw.text((10, y + 8), region, fill=(20, 24, 32), font=small)
        sources.append(source)
    return canvas, sources


def schp_output(sample_id: str) -> tuple[Image.Image | None, str | None]:
    path = SCHP_ROOT / sample_id / "final_output.png"
    return read_rgb(path), path.as_posix() if path.exists() else None


def local_masked_output(sample_id: str) -> tuple[Image.Image | None, str | None]:
    path = LOCAL_MASKED_ROOT / sample_id / "final_output.png"
    return read_rgb(path), path.as_posix() if path.exists() else None


def save_artifact(image: Image.Image, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path.as_posix()


def draw_multiline_center(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, text_font: ImageFont.ImageFont) -> None:
    lines = text.split("\n")
    line_h = 19
    y = (box[1] + box[3]) // 2 - (len(lines) - 1) * line_h // 2
    for line in lines:
        draw.text(((box[0] + box[2]) // 2, y), line, fill=(25, 29, 36), font=text_font, anchor="mm")
        y += line_h


def build_sample_grid(sample_id: str, old_grid: Image.Image) -> dict[str, Any]:
    sample_artifacts = ARTIFACT_ROOT / sample_id
    sample_artifacts.mkdir(parents=True, exist_ok=True)

    person, person_source = sample_person(sample_id)
    garment, garment_sources = garment_canvas(sample_id)
    schp, schp_source = schp_output(sample_id)
    klein_28, klein_28_source = klein_image(old_grid, sample_id, "klein_28")
    klein_strong, klein_strong_source = klein_image(old_grid, sample_id, "klein_28_strong")

    method4, method4_source = local_masked_output(sample_id)

    tiles = {
        "input": fit_tile(person) if person is not None else placeholder("missing\ninput"),
        "garment": fit_tile(garment) if garment is not None else placeholder("missing\ngarment"),
        "schp_sam_flux_catvton": fit_tile(schp) if schp is not None else placeholder("missing\nSCHP/SAM"),
        "klein_28": fit_tile(klein_28) if klein_28 is not None else placeholder("missing\nKlein 28"),
        "klein_28_strong": fit_tile(klein_strong) if klein_strong is not None else placeholder("missing\nKlein strong"),
        "klein_local_masked_inpaint": (
            fit_tile(method4) if method4 is not None else placeholder("missing\nlocal masked\ninpaint")
        ),
    }

    width = TILE_SIZE[0] * len(METHOD_LABELS)
    height = TITLE_H + HEADER_H + TILE_SIZE[1] + FOOTER_H
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    title_font = font(22, bold=True)
    header_font = font(15, bold=True)
    small_font = font(12)

    draw.rectangle((0, 0, width, TITLE_H), fill=(248, 250, 252))
    draw.text((18, TITLE_H // 2), f"{sample_id} - final project output comparison", fill=(20, 24, 32), font=title_font, anchor="lm")
    for idx, (label, key) in enumerate(METHOD_LABELS):
        x0 = idx * TILE_SIZE[0]
        draw.rectangle((x0, TITLE_H, x0 + TILE_SIZE[0], TITLE_H + HEADER_H), fill=(236, 240, 245), outline=(211, 217, 225))
        draw_multiline_center(draw, (x0, TITLE_H, x0 + TILE_SIZE[0], TITLE_H + HEADER_H), label, header_font)
        sheet.paste(tiles[key], (x0, TITLE_H + HEADER_H))
        draw.rectangle((x0, TITLE_H + HEADER_H, x0 + TILE_SIZE[0] - 1, TITLE_H + HEADER_H + TILE_SIZE[1] - 1), outline=(218, 224, 232))

    footer_y = TITLE_H + HEADER_H + TILE_SIZE[1]
    draw.rectangle((0, footer_y, width, height), fill=(248, 250, 252))
    note = "Method 4 is a real ComfyUI pass: SCHP/SAM mask -> crop -> Klein 9B + LoRA generation -> masked paste-back."
    draw.text((18, footer_y + 20), note, fill=(75, 83, 96), font=small_font, anchor="lm")

    output_path = OUT_ROOT / f"{sample_id}_grid.png"
    sheet.save(output_path)
    return {
        "sample_id": sample_id,
        "grid": output_path.as_posix(),
        "sources": {
            "input": person_source,
            "garment": garment_sources,
            "schp_sam_flux_catvton": schp_source,
            "klein_28": klein_28_source,
            "klein_28_strong": klein_strong_source,
            "klein_local_masked_inpaint": method4_source,
        },
    }


def write_index(rows: list[dict[str, Any]]) -> None:
    items = []
    for row in rows:
        name = Path(row["grid"]).name
        items.append(f'<section><h2>{html.escape(row["sample_id"])}</h2><a href="{name}"><img src="{name}"></a></section>')
    page = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>Final Project Output</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; background: #f6f7f9; color: #1f2328; }}
section {{ margin-bottom: 28px; }}
img {{ max-width: 100%; border: 1px solid #d8dee8; background: white; }}
</style></head><body>
<h1>Final Project Output - 15 Test Cases / 4 Methods</h1>
<p>Each image contains: input, garment/reference, SCHP/SAM Flux Fill + CatVTON, Klein 28, Klein 28 strong, and Klein+LoRA local masked inpaint.</p>
{''.join(items)}
</body></html>"""
    (OUT_ROOT / "index.html").write_text(page, encoding="utf-8")


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    if not ARCHIVED_GRID.exists():
        raise FileNotFoundError(f"Missing archived grid: {ARCHIVED_GRID}")
    old_grid = Image.open(ARCHIVED_GRID).convert("RGB")
    rows = [build_sample_grid(sample_id, old_grid) for sample_id in SAMPLE_IDS]
    write_index(rows)
    metadata = {
        "output_dir": OUT_ROOT.as_posix(),
        "sample_count": len(rows),
        "expected_grid_count": 15,
        "methods": [label for label, key in METHOD_LABELS if key not in {"input", "garment"}],
        "rows": rows,
    }
    (OUT_ROOT / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output_dir": OUT_ROOT.as_posix(), "grid_count": len(rows)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
