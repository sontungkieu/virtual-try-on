from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps


PROJECT_ROOT = Path("/workspace/Project_Phase2")
KLEIN_ROOT = PROJECT_ROOT / "virtual_tryon/data/outputs/vton_phase2_extra_cases_20260623/klein9b_lora"
MASK_ROOT = PROJECT_ROOT / "virtual_tryon/data/temp/production_vton_smoke_configs"
OUT_ROOT = PROJECT_ROOT / "virtual_tryon/data/outputs/klein_mask_composite_adetailer_style_20260625"


def resize_cover(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    image = image.convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def harden(mask: Image.Image, size: tuple[int, int], dilation: int = 8, blur: int = 5) -> Image.Image:
    mask = mask.convert("L").resize(size, Image.Resampling.NEAREST)
    mask = mask.point(lambda p: 255 if p > 8 else 0)
    if dilation > 0:
        mask = mask.filter(ImageFilter.MaxFilter(dilation * 2 + 1))
    if blur > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(blur))
    return mask


def overlay(image: Image.Image, mask: Image.Image) -> Image.Image:
    base = image.convert("RGBA")
    color = Image.new("RGBA", base.size, (36, 140, 255, 0))
    alpha = mask.convert("L").point(lambda p: int(p * 0.42))
    color.putalpha(alpha)
    return Image.alpha_composite(base, color).convert("RGB")


def union_masks(paths: list[Path], size: tuple[int, int]) -> Image.Image:
    out = Image.new("L", size, 0)
    for path in paths:
        mask = harden(Image.open(path), size)
        out = ImageChops.lighter(out, mask)
    return out


def draw_label(draw: ImageDraw.ImageDraw, text: str, x: int, y: int) -> None:
    draw.rectangle((x, y, x + 250, y + 24), fill=(255, 255, 255))
    draw.text((x + 5, y + 5), text, fill=(0, 0, 0))


def build_sheet(rows: list[dict], sheet_path: Path) -> None:
    cell = (230, 310)
    headers = ["person", "candidate", "mask overlay", "masked composite"]
    sheet = Image.new("RGB", (cell[0] * len(headers), cell[1] * (len(rows) + 1)), "white")
    draw = ImageDraw.Draw(sheet)
    for col, header in enumerate(headers):
        draw.text((col * cell[0] + 8, 8), header, fill=(0, 0, 0))
    for row_idx, row in enumerate(rows, start=1):
        draw.text((8, row_idx * cell[1] + 8), row["name"], fill=(0, 0, 0))
        for col, key in enumerate(["person", "candidate", "overlay", "composite"]):
            thumb = Image.open(row[key]).convert("RGB")
            thumb.thumbnail((cell[0] - 20, cell[1] - 42), Image.Resampling.LANCZOS)
            x = col * cell[0] + (cell[0] - thumb.width) // 2
            y = row_idx * cell[1] + 36
            sheet.paste(thumb, (x, y))
    sheet.save(sheet_path)


def run_sample_015() -> dict:
    sample_dir = KLEIN_ROOT / "sample_015"
    person = Image.open(sample_dir / "person_reference.png").convert("RGB")
    size = person.size
    mask = union_masks(
        [
            MASK_ROOT / "sample_015_dress_mask.png",
            MASK_ROOT / "sample_015_shoes_mask.png",
            MASK_ROOT / "sample_015_hat_mask.png",
        ],
        size,
    )

    out_dir = OUT_ROOT / "sample_015"
    out_dir.mkdir(parents=True, exist_ok=True)
    person.save(out_dir / "person.png")
    mask.save(out_dir / "mask_union_processed.png")
    overlay(person, mask).save(out_dir / "mask_overlay.png")

    rows: list[dict] = []
    variants = [
        ("klein_4step", sample_dir / "klein9b_lora_4step/result.png"),
        ("klein_28", sample_dir / "klein9b_lora_28_default/result.png"),
        ("klein_28_strong", sample_dir / "klein9b_lora_28_strong/result.png"),
    ]
    for name, candidate_path in variants:
        candidate = Image.open(candidate_path).convert("RGB")
        if candidate.size != size:
            candidate = resize_cover(candidate, size)
        composite = Image.composite(candidate, person, mask)
        variant_dir = out_dir / name
        variant_dir.mkdir(parents=True, exist_ok=True)
        candidate.save(variant_dir / "candidate_klein.png")
        composite.save(variant_dir / "masked_composite.png")
        overlay(composite, mask).save(variant_dir / "masked_composite_overlay.png")
        rows.append(
            {
                "name": name,
                "person": out_dir / "person.png",
                "candidate": variant_dir / "candidate_klein.png",
                "overlay": out_dir / "mask_overlay.png",
                "composite": variant_dir / "masked_composite.png",
            }
        )

    build_sheet(rows, out_dir / "klein_mask_composite_sheet.png")
    metadata = {
        "sample_id": "sample_015",
        "method": "klein_candidate_plus_masked_composite",
        "note": "ADetailer-style localization test: Klein generates candidate; only masked target region is pasted back onto original person.",
        "mask_files": [
            str(MASK_ROOT / "sample_015_dress_mask.png"),
            str(MASK_ROOT / "sample_015_shoes_mask.png"),
            str(MASK_ROOT / "sample_015_hat_mask.png"),
        ],
        "variants": [row["name"] for row in rows],
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata | {"output_dir": str(out_dir), "sheet": str(out_dir / "klein_mask_composite_sheet.png")}


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(json.dumps(run_sample_015(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
