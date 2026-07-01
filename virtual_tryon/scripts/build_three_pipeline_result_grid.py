from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VTON_ROOT = PROJECT_ROOT / "virtual_tryon"
OUTPUT_ROOT = VTON_ROOT / "data/outputs/three_pipeline_grid_20260626"
SCHP_ROOT = VTON_ROOT / "data/outputs/schp_sam_mask_tryon_outputs_20260626"
EXTRA_KLEIN_ROOT = VTON_ROOT / "data/outputs/vton_phase2_extra_cases_20260623/klein9b_lora"
FULL_15_GRID = VTON_ROOT / "data/outputs/vton_phase2_full_15_cases_20260625/full_15_test_cases_grid.png"


SAMPLE_IDS = [f"sample_{index:03d}" for index in range(1, 16)]
STRICT_MATCHED_SAMPLE_IDS = ["sample_010", "sample_011", "sample_012", "sample_014", "sample_015"]
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
    font = load_font(17, bold=True)
    small = load_font(12)
    draw.rectangle((8, 8, size[0] - 8, size[1] - 8), outline=(190, 196, 205), width=2)
    parts = label.split("\n")
    y = size[1] // 2 - (len(parts) - 1) * 12
    for part in parts:
        draw.text((size[0] // 2, y), part, fill=(85, 92, 104), font=font if part == parts[0] else small, anchor="mm")
        y += 26
    return image


def crop_old_grid_cell(old_grid: Image.Image, sample_id: str, column: str) -> Image.Image | None:
    number = int(sample_id.split("_")[1])
    if number > 9:
        return None

    # The first 9 samples only exist as cells in the archived full-15 grid.
    # Coordinates match the archived layout:
    # label | person | reference | IDM | IDM expanded | Klein 4 | Klein 28 | Klein 28 strong
    x_ranges = {
        "input": (275, 486),
        "klein_28": (1512, 1731),
        "klein_28_strong": (1762, 1981),
    }
    x0, x1 = x_ranges[column]
    y0 = 75 + (number - 1) * 334
    y1 = y0 + 292
    return old_grid.crop((x0, y0, x1, y1)).convert("RGB")


def read_image(path: Path | None, *, label: str) -> tuple[Image.Image, str | None]:
    if path and path.exists():
        return fit_tile(Image.open(path)), str(path)
    return placeholder(label), None


def first_schp_pass_input(sample_id: str) -> Path | None:
    sample_root = SCHP_ROOT / sample_id
    for pass_root in sorted(sample_root.glob("pass_*")):
        candidate = pass_root / "input_person.png"
        if candidate.exists():
            return candidate
    return None


def source_for_sample(old_grid: Image.Image, sample_id: str) -> dict[str, tuple[Image.Image, str | None]]:
    sample_root = EXTRA_KLEIN_ROOT / sample_id
    schp_path = SCHP_ROOT / sample_id / "final_output.png"
    number = int(sample_id.split("_")[1])

    if number <= 9:
        input_image = fit_tile(crop_old_grid_cell(old_grid, sample_id, "input"))
        input_source = f"{FULL_15_GRID}#crop:{sample_id}:input"
        klein_28_image = fit_tile(crop_old_grid_cell(old_grid, sample_id, "klein_28"))
        klein_28_source = f"{FULL_15_GRID}#crop:{sample_id}:klein_28"
        klein_strong_image = fit_tile(crop_old_grid_cell(old_grid, sample_id, "klein_28_strong"))
        klein_strong_source = f"{FULL_15_GRID}#crop:{sample_id}:klein_28_strong"
    else:
        input_image, input_source = read_image(sample_root / "person_reference.png", label="missing\ninput")
        klein_28_image, klein_28_source = read_image(
            sample_root / "klein9b_lora_28_default/result.png",
            label="missing\nKlein 28",
        )
        klein_strong_image, klein_strong_source = read_image(
            sample_root / "klein9b_lora_28_strong/result.png",
            label="missing\nKlein strong",
        )

    schp_image, schp_source = read_image(schp_path, label="missing\nSCHP/SAM")
    return {
        "input": (input_image, input_source),
        "schp_sam_flux_catvton": (schp_image, schp_source),
        "klein_28": (klein_28_image, klein_28_source),
        "klein_28_strong": (klein_strong_image, klein_strong_source),
    }


def render_grid(
    *,
    sample_ids: list[str],
    output_stem: str,
    title: str,
    note: str,
    source_getter,
) -> dict[str, Any]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    columns = [
        ("Input", "input"),
        ("SCHP/SAM\n+ Flux Fill/CatVTON", "schp_sam_flux_catvton"),
        ("Klein 9B\n28", "klein_28"),
        ("Klein 9B\n28 strong", "klein_28_strong"),
    ]
    width = ROW_LABEL_W + len(columns) * TILE_SIZE[0]
    height = HEADER_H + len(sample_ids) * ROW_H
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
        if len(lines) == 1:
            draw.text((x0 + TILE_SIZE[0] // 2, HEADER_H // 2), title, fill=(25, 29, 36), font=header_font, anchor="mm")
        else:
            draw.text((x0 + TILE_SIZE[0] // 2, 20), lines[0], fill=(25, 29, 36), font=header_font, anchor="mm")
            draw.text((x0 + TILE_SIZE[0] // 2, 40), lines[1], fill=(25, 29, 36), font=header_font, anchor="mm")

    rows: list[dict[str, Any]] = []
    for row_index, sample_id in enumerate(sample_ids):
        y0 = HEADER_H + row_index * ROW_H
        fill = (251, 252, 254) if row_index % 2 == 0 else (255, 255, 255)
        draw.rectangle((0, y0, width, y0 + ROW_H), fill=fill)
        draw.line((0, y0, width, y0), fill=(218, 224, 232), width=1)
        draw.text((10, y0 + 26), sample_id, fill=(24, 28, 35), font=label_font)
        draw.text((10, y0 + 50), "matched files", fill=(87, 94, 106), font=small_font)

        sources = source_getter(sample_id)
        row_meta: dict[str, Any] = {"sample_id": sample_id, "sources": {}}
        for col_index, (_title, key) in enumerate(columns):
            x0 = ROW_LABEL_W + col_index * TILE_SIZE[0]
            image, source = sources[key]
            sheet.paste(image, (x0, y0 + 12))
            row_meta["sources"][key] = source
        rows.append(row_meta)

    output_png = OUTPUT_ROOT / f"{output_stem}.png"
    sheet.save(output_png)

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; background: #f6f7f9; color: #1f2328; }}
    img {{ max-width: 100%; border: 1px solid #d8dee8; background: white; }}
    code {{ background: #edf0f4; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <p>Columns: input, SCHP/SAM mask + Flux Fill/CatVTON result, Klein 9B 28, Klein 9B 28 strong.</p>
  <p>{note}</p>
  <img src="{output_png.name}" alt="three pipeline grid">
</body>
</html>
"""
    output_html = OUTPUT_ROOT / f"{output_stem}.html"
    output_html.write_text(html, encoding="utf-8")

    metadata = {
        "output_png": str(output_png),
        "output_html": str(output_html),
        "title": title,
        "columns": [key for _, key in columns],
        "rows": rows,
        "notes": [note],
    }
    (OUTPUT_ROOT / f"{output_stem}_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata


def matched_source_for_sample(sample_id: str) -> dict[str, tuple[Image.Image, str | None]]:
    sample_root = EXTRA_KLEIN_ROOT / sample_id
    schp_path = SCHP_ROOT / sample_id / "final_output.png"
    input_path = first_schp_pass_input(sample_id) or sample_root / "person_reference.png"

    input_image, input_source = read_image(input_path, label="missing\ninput")
    schp_image, schp_source = read_image(schp_path, label="missing\nSCHP/SAM")
    klein_28_image, klein_28_source = read_image(
        sample_root / "klein9b_lora_28_default/result.png",
        label="missing\nKlein 28",
    )
    klein_strong_image, klein_strong_source = read_image(
        sample_root / "klein9b_lora_28_strong/result.png",
        label="missing\nKlein strong",
    )
    return {
        "input": (input_image, input_source),
        "schp_sam_flux_catvton": (schp_image, schp_source),
        "klein_28": (klein_28_image, klein_28_source),
        "klein_28_strong": (klein_strong_image, klein_strong_source),
    }


def build_matched_grid() -> dict[str, Any]:
    note = (
        "Strict matched grid: samples 001-009 are excluded because the SCHP/SAM run used a different eval set "
        "from the archived Klein full-15 grid. Sample 013 is excluded because the SCHP/SAM output is missing. "
        "This grid only compares rows with individual files for all three result columns."
    )
    metadata = render_grid(
        sample_ids=STRICT_MATCHED_SAMPLE_IDS,
        output_stem="three_pipeline_strict_matched_grid",
        title="Three Pipeline Strict Matched Grid",
        note=note,
        source_getter=matched_source_for_sample,
    )
    audit = {
        "strict_matched_samples": STRICT_MATCHED_SAMPLE_IDS,
        "excluded_samples": {
            "sample_001_to_sample_009": "Different eval set between SCHP/SAM outputs and archived Klein grid crops.",
            "sample_013": "Missing SCHP/SAM final_output.png.",
        },
        "supersedes": "three_pipeline_full15_grid.png was a mixed-source audit artifact and should not be used for visual comparison.",
    }
    (OUTPUT_ROOT / "sample_source_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata


def build_grid() -> dict[str, Any]:
    if not FULL_15_GRID.exists():
        raise FileNotFoundError(f"Missing archived full-15 grid for crop sources: {FULL_15_GRID}")

    old_grid = Image.open(FULL_15_GRID).convert("RGB")
    note = (
        "Audit-only mixed-source grid. Samples 001-009 use crops from an archived full-15 Klein grid, while "
        "the SCHP/SAM outputs come from a different eval set. Do not use this artifact as a strict comparison."
    )
    return render_grid(
        sample_ids=SAMPLE_IDS,
        output_stem="three_pipeline_full15_grid_AUDIT_MIXED_SOURCES",
        title="Three Pipeline Full 15 Grid - Audit Mixed Sources",
        note=note,
        source_getter=lambda sample_id: source_for_sample(old_grid, sample_id),
    )


if __name__ == "__main__":
    result = build_matched_grid()
    print(json.dumps({"output_png": result["output_png"], "output_html": result["output_html"]}, indent=2))
