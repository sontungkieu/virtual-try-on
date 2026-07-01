from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from virtual_tryon.final_demo import (
    CANVAS_SIZE,
    DATA_INPUT_ROOT,
    DEFAULT_FINAL_EVAL_ROOT,
    REGION_LABELS,
    SOURCE_SAMPLE_PLAN,
    category_for_regions,
    sample_id_for_index,
)

DEFAULT_SOURCE_ROOT = DATA_INPUT_ROOT
DEFAULT_OUTPUT_ROOT = DEFAULT_FINAL_EVAL_ROOT


def resolve_source_file(sample_dir: Path, requested_name: str) -> Path:
    candidates = [sample_dir / requested_name]
    if requested_name == "Garment.png":
        candidates.append(sample_dir / "Garmet.png")
    if requested_name == "Garmet.png":
        candidates.append(sample_dir / "Garment.png")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    available = ", ".join(path.name for path in sorted(sample_dir.glob("*")))
    raise FileNotFoundError(f"Missing {requested_name} in {sample_dir}. Available: {available}")


def fit_on_canvas(path: Path, size: tuple[int, int]) -> Image.Image:
    image = ImageOps.exif_transpose(Image.open(path).convert("RGB"))
    fitted = ImageOps.contain(image, (size[0] - 24, size[1] - 24), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(fitted, ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2))
    return canvas


def build_reference_canvas(items: list[dict[str, str]], sample_dir: Path) -> None:
    if len(items) == 1:
        canvas = fit_on_canvas(sample_dir / items[0]["file_name"], CANVAS_SIZE)
        canvas.save(sample_dir / "reference_canvas.png")
        return

    width, height = CANVAS_SIZE
    cell_h = height // len(items)
    canvas = Image.new("RGB", CANVAS_SIZE, "white")
    draw = ImageDraw.Draw(canvas)
    for index, item in enumerate(items):
        y0 = index * cell_h
        y1 = height if index == len(items) - 1 else (index + 1) * cell_h
        tile = fit_on_canvas(sample_dir / item["file_name"], (width, y1 - y0))
        canvas.paste(tile, (0, y0))
        if index > 0:
            draw.line((24, y0, width - 24, y0), fill=(225, 229, 235), width=2)
    canvas.save(sample_dir / "reference_canvas.png")


def category_for(regions: list[str]) -> str:
    return category_for_regions(regions)


def make_sample(source_root: Path, output_root: Path, index: int) -> dict[str, object]:
    source_dir = source_root / f"Test case {index}"
    if not source_dir.exists():
        raise FileNotFoundError(source_dir)

    sample_id = sample_id_for_index(index)
    sample_dir = output_root / sample_id
    if sample_dir.exists():
        shutil.rmtree(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)

    person_source = resolve_source_file(source_dir, "Person.png")
    shutil.copy2(person_source, sample_dir / "person.png")

    items: list[dict[str, str]] = []
    for pass_index, spec in enumerate(SOURCE_SAMPLE_PLAN[index], start=1):
        source_path = resolve_source_file(source_dir, spec.source_name)
        target_path = sample_dir / spec.normalized_name
        shutil.copy2(source_path, target_path)
        items.append(
            {
                "pass_index": str(pass_index),
                "region": spec.region,
                "label": REGION_LABELS.get(spec.region, spec.region),
                "source_name": source_path.name,
                "file_name": spec.normalized_name,
            }
        )

    build_reference_canvas(items, sample_dir)
    regions = [item["region"] for item in items]
    metadata: dict[str, object] = {
        "sample_id": sample_id,
        "source_dir": source_dir.as_posix(),
        "person_source": person_source.as_posix(),
        "category": category_for(regions),
        "items": [item["file_name"] for item in items],
        "passes": items,
        "reference_canvas": "reference_canvas.png",
    }
    (sample_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    source_root = args.source_root
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    samples = [make_sample(source_root, output_root, index) for index in range(1, 16)]
    summary = {
        "source_root": source_root.as_posix(),
        "output_root": output_root.as_posix(),
        "sample_count": len(samples),
        "samples": samples,
    }
    (output_root / "metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output_root": output_root.as_posix(), "sample_count": len(samples)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
