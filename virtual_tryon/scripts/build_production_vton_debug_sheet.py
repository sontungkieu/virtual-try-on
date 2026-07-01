from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def find_pass_dirs(run_dir: Path) -> list[Path]:
    if (run_dir / "metadata.json").exists():
        return [run_dir]
    return sorted(path for path in run_dir.glob("pass_*") if (path / "metadata.json").exists())


def flag_failures(pass_dir: Path, metadata: dict[str, Any]) -> list[str]:
    flags = list(metadata.get("warnings") or [])
    ratio = float(metadata.get("mask_area_ratio") or 0)
    if ratio > 0.55:
        flags.append(f"mask too large ({ratio:.3f})")
    if ratio < 0.01:
        flags.append(f"mask too small ({ratio:.3f})")
    if not (pass_dir / "output_base.png").exists():
        flags.append("missing output_base.png")
    if metadata.get("output_refined") and not (pass_dir / str(metadata["output_refined"])).exists():
        flags.append("metadata references missing refined output")
    if not (pass_dir / "input_garment.png").exists():
        flags.append("no garment image provided")
    if not (pass_dir / "mask_processed.png").exists():
        flags.append("no mask image provided")
    garment_source = str(metadata.get("input_garment_source") or pass_dir / "input_garment.png").lower()
    if "canvas" in garment_source or "reference_canvas" in garment_source:
        flags.append("multi-item reference used in single-pass mode")
    return sorted(set(flags))


def paste_thumb(canvas: Image.Image, path: Path, box: tuple[int, int, int, int], label: str) -> None:
    draw = ImageDraw.Draw(canvas)
    x0, y0, x1, y1 = box
    draw.rectangle(box, outline=(220, 224, 230), fill=(255, 255, 255))
    draw.text((x0 + 8, y0 + 8), label, fill=(30, 35, 45))
    if not path.exists():
        draw.text((x0 + 8, y0 + 36), "missing", fill=(170, 0, 0))
        return
    image = Image.open(path).convert("RGB")
    max_w = x1 - x0 - 18
    max_h = y1 - y0 - 44
    image.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
    canvas.paste(image, (x0 + (x1 - x0 - image.width) // 2, y0 + 36 + (max_h - image.height) // 2))


def build_debug_sheet(run_dir: Path, output_path: Path | None = None) -> Path:
    pass_dirs = find_pass_dirs(run_dir)
    if not pass_dirs:
        raise FileNotFoundError(f"No pass metadata found under {run_dir}")

    cell_w = 250
    cell_h = 330
    cols = 6
    row_h = cell_h + 70
    canvas = Image.new("RGB", (cols * cell_w, row_h * len(pass_dirs)), "white")
    draw = ImageDraw.Draw(canvas)

    all_flags: dict[str, list[str]] = {}
    headers = ["person", "garment", "raw mask", "overlay", "base output", "refined/final"]
    for row, pass_dir in enumerate(pass_dirs):
        y = row * row_h
        metadata = read_json(pass_dir / "metadata.json")
        pass_name = f"pass {metadata.get('pass_index')} / {metadata.get('target_region')}"
        flags = flag_failures(pass_dir, metadata)
        all_flags[pass_dir.name] = flags
        draw.text((8, y + 6), pass_name, fill=(0, 0, 0))
        if flags:
            draw.text((160, y + 6), " | ".join(flags[:4]), fill=(180, 40, 40))

        paths = [
            pass_dir / "input_person.png",
            pass_dir / "input_garment.png",
            pass_dir / "mask_raw.png",
            pass_dir / "mask_overlay.png",
            pass_dir / "output_base.png",
            pass_dir / str(metadata.get("output_refined") or metadata.get("output_final") or "output_base.png"),
        ]
        for col, (header, path) in enumerate(zip(headers, paths, strict=True)):
            x0 = col * cell_w
            paste_thumb(canvas, path, (x0, y + 34, x0 + cell_w, y + 34 + cell_h), header)

    output_path = output_path or (run_dir / "production_vton_debug_sheet.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    (output_path.with_suffix(".flags.json")).write_text(json.dumps(all_flags, indent=2), encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="Production run directory or one pass directory.")
    parser.add_argument("--output", help="Output PNG path.")
    args = parser.parse_args()
    output = build_debug_sheet(Path(args.run_dir), Path(args.output) if args.output else None)
    print(output.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
