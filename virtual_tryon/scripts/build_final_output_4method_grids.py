from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from virtual_tryon.final_demo import (
    DEFAULT_FINAL_EVAL_ROOT,
    DEFAULT_FINAL_OUTPUT_ROOT,
    FINAL_METHODS,
    final_method_paths,
    sample_id_for_index,
)

DEFAULT_EVAL_ROOT = DEFAULT_FINAL_EVAL_ROOT
DEFAULT_OUTPUT_ROOT = DEFAULT_FINAL_OUTPUT_ROOT

METHODS = [
    ("Input image", "input"),
    ("Garment / reference", "reference"),
    *[(method.title, method.key) for method in FINAL_METHODS],
]
TILE_SIZE = (270, 380)
TITLE_H = 58
HEADER_H = 74
FOOTER_H = 42


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def fit_tile(path: Path | None, label: str) -> Image.Image:
    if path is None or not path.exists():
        image = Image.new("RGB", TILE_SIZE, (247, 248, 250))
        draw = ImageDraw.Draw(image)
        draw.rectangle((8, 8, TILE_SIZE[0] - 8, TILE_SIZE[1] - 8), outline=(192, 198, 208), width=2)
        draw.text((TILE_SIZE[0] // 2, TILE_SIZE[1] // 2), f"missing\n{label}", fill=(125, 0, 0), font=font(16, True), anchor="mm")
        return image
    image = ImageOps.exif_transpose(Image.open(path).convert("RGB"))
    contained = ImageOps.contain(image, (TILE_SIZE[0] - 16, TILE_SIZE[1] - 16), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", TILE_SIZE, "white")
    canvas.paste(contained, ((TILE_SIZE[0] - contained.width) // 2, (TILE_SIZE[1] - contained.height) // 2))
    return canvas


def draw_multiline_center(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str) -> None:
    lines = text.split("\n")
    text_font = font(15, True)
    line_h = 20
    y = (box[1] + box[3]) // 2 - ((len(lines) - 1) * line_h) // 2
    for line in lines:
        draw.text(((box[0] + box[2]) // 2, y), line, fill=(25, 29, 36), font=text_font, anchor="mm")
        y += line_h


def method_paths(output_root: Path, sample_id: str) -> dict[str, Path]:
    return final_method_paths(output_root, sample_id)


def build_grid(eval_root: Path, output_root: Path, sample_id: str) -> dict[str, Any]:
    sample_dir = eval_root / sample_id
    paths = method_paths(output_root, sample_id)
    paths["input"] = sample_dir / "person.png"
    paths["reference"] = sample_dir / "reference_canvas.png"

    width = TILE_SIZE[0] * len(METHODS)
    height = TITLE_H + HEADER_H + TILE_SIZE[1] + FOOTER_H
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    title_font = font(23, True)
    small_font = font(12)

    draw.rectangle((0, 0, width, TITLE_H), fill=(248, 250, 252))
    test_number = int(sample_id.split("_")[1])
    draw.text((18, TITLE_H // 2), f"Test case {test_number:02d} - 4-method comparison", fill=(20, 24, 32), font=title_font, anchor="lm")

    for index, (label, key) in enumerate(METHODS):
        x0 = index * TILE_SIZE[0]
        draw.rectangle((x0, TITLE_H, x0 + TILE_SIZE[0], TITLE_H + HEADER_H), fill=(236, 240, 245), outline=(211, 217, 225))
        draw_multiline_center(draw, (x0, TITLE_H, x0 + TILE_SIZE[0], TITLE_H + HEADER_H), label)
        sheet.paste(fit_tile(paths.get(key), label), (x0, TITLE_H + HEADER_H))
        draw.rectangle((x0, TITLE_H + HEADER_H, x0 + TILE_SIZE[0] - 1, TITLE_H + HEADER_H + TILE_SIZE[1] - 1), outline=(218, 224, 232))

    footer_y = TITLE_H + HEADER_H + TILE_SIZE[1]
    draw.rectangle((0, footer_y, width, height), fill=(248, 250, 252))
    draw.text(
        (18, footer_y + 20),
        "Method 3 is Klein + Try-On LoRA with default prompt strength; old Klein strong is not used.",
        fill=(75, 83, 96),
        font=small_font,
        anchor="lm",
    )
    out_path = output_root / f"test_case_{test_number:02d}_grid.png"
    sheet.save(out_path)

    return {
        "sample_id": sample_id,
        "grid": out_path.as_posix(),
        "sources": {key: path.as_posix() for key, path in paths.items()},
        "missing": [key for key, path in paths.items() if not path.exists()],
    }


def write_index(output_root: Path, rows: list[dict[str, Any]]) -> None:
    sections = []
    for row in rows:
        name = Path(row["grid"]).name
        missing = row.get("missing") or []
        missing_html = f"<p class='missing'>Missing: {html.escape(', '.join(missing))}</p>" if missing else ""
        sections.append(f"<section><h2>{html.escape(row['sample_id'])}</h2>{missing_html}<a href='{name}'><img src='{name}'></a></section>")
    page = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>FINAL OUTPUT - 4 Methods</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; background: #f6f7f9; color: #1f2328; }}
section {{ margin-bottom: 28px; }}
img {{ max-width: 100%; border: 1px solid #d8dee8; background: white; }}
.missing {{ color: #a40000; font-weight: 700; }}
</style></head><body>
<h1>FINAL OUTPUT - 15 Test Cases / 4 Methods</h1>
<p>Columns: input, garment/reference, SCHP/SAM Flux Fill + CatVTON, Klein 9B, Klein 9B + Try-On LoRA, Klein+LoRA local masked inpaint.</p>
{''.join(sections)}
</body></html>"""
    (output_root / "index.html").write_text(page, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = [build_grid(args.eval_root, args.output_root, sample_id_for_index(index)) for index in range(1, 16)]
    write_index(args.output_root, rows)
    metadata = {
        "output_root": args.output_root.as_posix(),
        "grid_count": len(rows),
        "methods": [label for label, key in METHODS if key not in {"input", "reference"}],
        "rows": rows,
    }
    (args.output_root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    missing = {row["sample_id"]: row["missing"] for row in rows if row.get("missing")}
    print(json.dumps({"output_root": args.output_root.as_posix(), "grid_count": len(rows), "missing": missing}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
