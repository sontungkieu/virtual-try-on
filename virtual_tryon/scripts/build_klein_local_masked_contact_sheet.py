from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


def load_thumb(path: Path, size: tuple[int, int]) -> Image.Image:
    if not path.exists():
        image = Image.new("RGB", size, "white")
        draw = ImageDraw.Draw(image)
        draw.text((10, size[1] // 2 - 8), f"missing\n{path.name}", fill=(180, 0, 0))
        return image
    image = Image.open(path).convert("RGB")
    return ImageOps.contain(image, size, Image.Resampling.LANCZOS)


def pass_dirs(output_root: Path) -> list[Path]:
    return sorted(output_root.glob("sample_*/pass_*_*"))


def build_contact_sheet(output_root: Path, output_path: Path) -> None:
    rows = pass_dirs(output_root)
    headers = ["pass", "person", "garment", "mask overlay", "crop", "generated crop", "output"]
    cell_w, cell_h = 210, 270
    header_h = 38
    label_w = 160
    sheet = Image.new("RGB", (label_w + cell_w * (len(headers) - 1), header_h + cell_h * len(rows)), "white")
    draw = ImageDraw.Draw(sheet)
    draw.rectangle((0, 0, sheet.width, header_h), fill=(235, 238, 243))
    draw.text((10, 12), headers[0], fill=(20, 25, 32))
    for i, header in enumerate(headers[1:]):
        x = label_w + i * cell_w
        draw.text((x + 10, 12), header, fill=(20, 25, 32))
    for r, pass_dir in enumerate(rows):
        y = header_h + r * cell_h
        draw.rectangle((0, y, sheet.width, y + cell_h), outline=(220, 224, 230))
        sample = pass_dir.parent.name
        label = f"{sample}\n{pass_dir.name}"
        metadata = pass_dir / "metadata.json"
        if metadata.exists():
            try:
                meta = json.loads(metadata.read_text(encoding="utf-8"))
                label += f"\nmask={meta.get('mask_area_ratio')}\nseed={meta.get('seed')}"
            except Exception:
                pass
        draw.text((10, y + 12), label, fill=(20, 25, 32))
        files = [
            "input_person.png",
            "input_garment.png",
            "mask_overlay.png",
            "crop_image.png",
            "generated_crop.png",
            "output_base.png",
        ]
        for c, name in enumerate(files):
            x = label_w + c * cell_w
            thumb = load_thumb(pass_dir / name, (cell_w - 16, cell_h - 16))
            sheet.paste(thumb, (x + (cell_w - thumb.width) // 2, y + (cell_h - thumb.height) // 2))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    output_root = Path(args.output_root)
    output_path = Path(args.output) if args.output else output_root / "contact_sheet.png"
    build_contact_sheet(output_root, output_path)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
