from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps


PROJECT_ROOT = Path("/workspace/Project_Phase2")
VTON_ROOT = PROJECT_ROOT / "virtual_tryon"
EVAL_ROOT = VTON_ROOT / "data/temp/full15_archived_grid_eval_set"
SCHP_ROOT = VTON_ROOT / "data/outputs/full15_archived_grid_schp_sam_tryon_outputs_20260626"
EXTRA_KLEIN_ROOT = VTON_ROOT / "data/outputs/vton_phase2_extra_cases_20260623/klein9b_lora"
FULL_15_GRID = VTON_ROOT / "data/outputs/vton_phase2_full_15_cases_20260625/full_15_test_cases_grid.png"
OUT_ROOT = VTON_ROOT / "data/outputs/full15_schp_sam_comparison_20260626"


SAMPLE_IDS = [f"sample_{index:03d}" for index in range(1, 16)]
TILE_SIZE = (230, 305)
HEADER_H = 58
ROW_LABEL_W = 118
ROW_H = 328


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def fit_tile(image: Image.Image, size: tuple[int, int] = TILE_SIZE) -> Image.Image:
    image = ImageOps.exif_transpose(image.convert("RGB"))
    contained = ImageOps.contain(image, (size[0] - 10, size[1] - 10), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(contained, ((size[0] - contained.width) // 2, (size[1] - contained.height) // 2))
    return canvas


def placeholder(label: str, size: tuple[int, int] = TILE_SIZE) -> Image.Image:
    image = Image.new("RGB", size, (246, 247, 249))
    draw = ImageDraw.Draw(image)
    font = load_font(16, bold=True)
    small = load_font(12)
    draw.rectangle((8, 8, size[0] - 8, size[1] - 8), outline=(190, 196, 205), width=2)
    parts = label.split("\n")
    y = size[1] // 2 - (len(parts) - 1) * 12
    for part in parts:
        draw.text((size[0] // 2, y), part, fill=(85, 92, 104), font=font if part == parts[0] else small, anchor="mm")
        y += 26
    return image


def read_image(path: Path | None, label: str) -> tuple[Image.Image, str | None]:
    if path and path.exists():
        return fit_tile(Image.open(path)), path.as_posix()
    return placeholder(label), None


def crop_top_grid_cell(old_grid: Image.Image, sample_id: str, column: str) -> Image.Image | None:
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


def klein_source(old_grid: Image.Image, sample_id: str, variant: str) -> tuple[Image.Image, str | None]:
    number = int(sample_id.split("_")[1])
    if number <= 9:
        crop = crop_top_grid_cell(old_grid, sample_id, variant)
        if crop is None:
            return placeholder(f"missing\n{variant}"), None
        return fit_tile(crop), f"{FULL_15_GRID.as_posix()}#crop:{sample_id}:{variant}"
    variant_dir = "klein9b_lora_28_default" if variant == "klein_28" else "klein9b_lora_28_strong"
    return read_image(EXTRA_KLEIN_ROOT / sample_id / variant_dir / "result.png", f"missing\n{variant}")


def build_grid() -> dict[str, Any]:
    if not FULL_15_GRID.exists():
        raise FileNotFoundError(f"Missing archived full-15 grid: {FULL_15_GRID}")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    old_grid = Image.open(FULL_15_GRID).convert("RGB")
    columns = [
        ("Input", "input"),
        ("SCHP/SAM\n+ Flux Fill/CatVTON", "schp_sam_flux_catvton"),
        ("Klein 9B\n28", "klein_28"),
        ("Klein 9B\n28 strong", "klein_28_strong"),
    ]
    width = ROW_LABEL_W + len(columns) * TILE_SIZE[0]
    height = HEADER_H + len(SAMPLE_IDS) * ROW_H
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    header_font = load_font(15, bold=True)
    label_font = load_font(16, bold=True)
    small_font = load_font(11)

    draw.rectangle((0, 0, width, HEADER_H), fill=(236, 240, 245))
    for index, (title, _) in enumerate(columns):
        x0 = ROW_LABEL_W + index * TILE_SIZE[0]
        draw.rectangle((x0, 0, x0 + TILE_SIZE[0], HEADER_H), outline=(211, 217, 225), fill=(236, 240, 245))
        lines = title.split("\n")
        y = 20 if len(lines) > 1 else HEADER_H // 2
        for line in lines:
            draw.text((x0 + TILE_SIZE[0] // 2, y), line, fill=(25, 29, 36), font=header_font, anchor="mm")
            y += 20

    rows: list[dict[str, Any]] = []
    for row_index, sample_id in enumerate(SAMPLE_IDS):
        y0 = HEADER_H + row_index * ROW_H
        fill = (251, 252, 254) if row_index % 2 == 0 else (255, 255, 255)
        draw.rectangle((0, y0, width, y0 + ROW_H), fill=fill)
        draw.line((0, y0, width, y0), fill=(218, 224, 232), width=1)
        draw.text((10, y0 + 26), sample_id, fill=(24, 28, 35), font=label_font)
        draw.text((10, y0 + 50), "full15 set", fill=(87, 94, 106), font=small_font)

        input_image, input_source = read_image(EVAL_ROOT / sample_id / "person.png", "missing\ninput")
        schp_image, schp_source = read_image(SCHP_ROOT / sample_id / "final_output.png", "missing\nSCHP/SAM")
        klein_image, klein_source_path = klein_source(old_grid, sample_id, "klein_28")
        strong_image, strong_source_path = klein_source(old_grid, sample_id, "klein_28_strong")
        sources = {
            "input": (input_image, input_source),
            "schp_sam_flux_catvton": (schp_image, schp_source),
            "klein_28": (klein_image, klein_source_path),
            "klein_28_strong": (strong_image, strong_source_path),
        }
        row_meta: dict[str, Any] = {"sample_id": sample_id, "sources": {}}
        for col_index, (_title, key) in enumerate(columns):
            x0 = ROW_LABEL_W + col_index * TILE_SIZE[0]
            image, source = sources[key]
            sheet.paste(image, (x0, y0 + 12))
            row_meta["sources"][key] = source
        rows.append(row_meta)

    output_png = OUT_ROOT / "full15_schp_sam_vs_klein_grid.png"
    sheet.save(output_png)
    output_html = OUT_ROOT / "full15_schp_sam_vs_klein_grid.html"
    output_html.write_text(
        f"""<!doctype html>
<html>
<head><meta charset="utf-8"><title>Full 15 SCHP/SAM vs Klein Grid</title>
<style>body{{font-family:Arial,sans-serif;margin:24px;background:#f6f7f9;color:#1f2328}} img{{max-width:100%;border:1px solid #d8dee8;background:white}} code{{background:#edf0f4;padding:2px 5px;border-radius:4px}}</style></head>
<body>
<h1>Full 15 SCHP/SAM vs Klein Grid</h1>
<p>Rows use the archived full-15 test-case contract. Samples 001-009 use cropped archived sources; samples 010-015 use individual extra-eval files.</p>
<img src="{output_png.name}" alt="full 15 comparison grid">
</body></html>
""",
        encoding="utf-8",
    )
    metadata = {
        "output_png": output_png.as_posix(),
        "output_html": output_html.as_posix(),
        "eval_root": EVAL_ROOT.as_posix(),
        "schp_root": SCHP_ROOT.as_posix(),
        "rows": rows,
    }
    (OUT_ROOT / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output_png": output_png.as_posix(), "output_html": output_html.as_posix()}, indent=2), flush=True)
    return metadata


if __name__ == "__main__":
    build_grid()
