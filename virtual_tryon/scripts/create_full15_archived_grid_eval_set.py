from __future__ import annotations

import json
import shutil
from pathlib import Path

from PIL import Image, ImageChops, ImageOps


PROJECT_ROOT = Path("/workspace/Project_Phase2")
VTON_ROOT = PROJECT_ROOT / "virtual_tryon"
GRID_PATH = VTON_ROOT / "data/outputs/vton_phase2_full_15_cases_20260625/full_15_test_cases_grid.png"
EXTRA_EVAL_ROOT = VTON_ROOT / "data/temp/vton_phase2_extra_eval_set"
OUT_ROOT = VTON_ROOT / "data/temp/full15_archived_grid_eval_set"


TOP_ROW_Y0 = 75
TOP_ROW_H = 334
TOP_CELL_Y_H = 292
TOP_PERSON_X = (275, 498)
TOP_REF_X = (503, 753)

TOP_SAMPLE_META = {
    "sample_001": {"category": "upper_body", "items": ["garment_top.png"], "difficulty": "medium", "expected_focus": ["upper_body", "identity"]},
    "sample_002": {"category": "lower_body", "items": ["garment_bottom.png"], "difficulty": "medium", "expected_focus": ["lower_body", "pose"]},
    "sample_003": {"category": "upper_body", "items": ["garment_top.png"], "difficulty": "medium", "expected_focus": ["upper_body", "texture"]},
    "sample_004": {"category": "lower_body", "items": ["garment_bottom.png"], "difficulty": "medium", "expected_focus": ["lower_body", "silhouette"]},
    "sample_005": {"category": "dress", "items": ["garment_dress.png"], "difficulty": "hard", "expected_focus": ["dress", "full_body"]},
    "sample_006": {"category": "upper_body", "items": ["garment_top.png"], "difficulty": "medium", "expected_focus": ["upper_body", "old_garment_removal"]},
    "sample_007": {"category": "upper_body", "items": ["garment_top.png"], "difficulty": "medium", "expected_focus": ["upper_body", "garment_fidelity"]},
    "sample_008": {"category": "upper_body", "items": ["garment_top.png"], "difficulty": "medium", "expected_focus": ["upper_body", "identity"]},
    "sample_009": {"category": "lower_body", "items": ["garment_bottom.png"], "difficulty": "medium", "expected_focus": ["lower_body", "fit"]},
}

EXTRA_SAMPLE_META = {
    "sample_010": {"category": "lower_body", "items": ["garment_bottom.png"], "difficulty": "medium", "expected_focus": ["lower_body", "fit"]},
    "sample_011": {"category": "full_outfit", "items": ["garment_top.png", "garment_bottom.png"], "difficulty": "hard", "expected_focus": ["multi_item", "upper_body", "lower_body"]},
    "sample_012": {"category": "lower_body_hat", "items": ["garment_bottom.png", "accessory_hat.png"], "difficulty": "hard", "expected_focus": ["lower_body", "hat", "sequential_pass"]},
    "sample_013": {"category": "accessory", "items": ["accessory_hat.png", "accessory_watch.png"], "difficulty": "hard", "expected_focus": ["hat", "watch", "accessory_localization"]},
    "sample_014": {"category": "full_outfit_hat", "items": ["garment_top.png", "garment_bottom.png", "accessory_hat.png"], "difficulty": "hard", "expected_focus": ["multi_item", "hat", "identity"]},
    "sample_015": {"category": "dress_shoes_hat", "items": ["garment_dress.png", "accessory_shoes.png", "accessory_hat.png"], "difficulty": "hard", "expected_focus": ["dress", "shoes", "hat", "sequential_pass"]},
}

ITEM_RENAMES = {
    "sample_001": "garment_top.png",
    "sample_002": "garment_bottom.png",
    "sample_003": "garment_top.png",
    "sample_004": "garment_bottom.png",
    "sample_005": "garment_dress.png",
    "sample_006": "garment_top.png",
    "sample_007": "garment_top.png",
    "sample_008": "garment_top.png",
    "sample_009": "garment_bottom.png",
}


def trim_border(image: Image.Image, tolerance: int = 8) -> Image.Image:
    image = ImageOps.exif_transpose(image.convert("RGB"))
    bg = Image.new("RGB", image.size, image.getpixel((0, 0)))
    diff = ImageChops.difference(image, bg).convert("L")
    diff = diff.point(lambda px: 255 if px > tolerance else 0)
    bbox = diff.getbbox()
    if bbox is None:
        return image
    x0, y0, x1, y1 = bbox
    pad = 8
    return image.crop((max(0, x0 - pad), max(0, y0 - pad), min(image.width, x1 + pad), min(image.height, y1 + pad)))


def upscale_for_model(image: Image.Image, target_h: int = 768) -> Image.Image:
    image = trim_border(image)
    if image.height >= target_h:
        return image
    scale = target_h / max(1, image.height)
    return image.resize((max(1, int(image.width * scale)), target_h), Image.Resampling.LANCZOS)


def crop_top_sample(grid: Image.Image, sample_id: str) -> None:
    number = int(sample_id.split("_")[1])
    sample_root = OUT_ROOT / sample_id
    sample_root.mkdir(parents=True, exist_ok=True)
    y0 = TOP_ROW_Y0 + (number - 1) * TOP_ROW_H
    y1 = y0 + TOP_CELL_Y_H

    person = grid.crop((TOP_PERSON_X[0], y0, TOP_PERSON_X[1], y1))
    reference = grid.crop((TOP_REF_X[0], y0, TOP_REF_X[1], y1))

    upscale_for_model(person).save(sample_root / "person.png")
    upscale_for_model(reference).save(sample_root / ITEM_RENAMES[sample_id])
    (sample_root / "metadata.json").write_text(
        json.dumps({"sample_id": sample_id, **TOP_SAMPLE_META[sample_id], "source": GRID_PATH.as_posix()}, indent=2),
        encoding="utf-8",
    )


def copy_extra_sample(sample_id: str) -> None:
    src = EXTRA_EVAL_ROOT / sample_id
    dst = OUT_ROOT / sample_id
    if not src.exists():
        raise FileNotFoundError(f"Missing extra sample source: {src}")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    meta = {"sample_id": sample_id, **EXTRA_SAMPLE_META[sample_id], "source": src.as_posix()}
    (dst / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main() -> int:
    if not GRID_PATH.exists():
        raise FileNotFoundError(f"Missing archived grid: {GRID_PATH}")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    grid = Image.open(GRID_PATH).convert("RGB")
    for index in range(1, 10):
        crop_top_sample(grid, f"sample_{index:03d}")
    for index in range(10, 16):
        copy_extra_sample(f"sample_{index:03d}")
    index = {
        "output_root": OUT_ROOT.as_posix(),
        "grid_source": GRID_PATH.as_posix(),
        "samples": [f"sample_{i:03d}" for i in range(1, 16)],
        "notes": [
            "Samples 001-009 are cropped from the archived full-15 grid because original individual source files are not available.",
            "Samples 010-015 are copied from vton_phase2_extra_eval_set individual files.",
        ],
    }
    (OUT_ROOT / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(json.dumps(index, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
