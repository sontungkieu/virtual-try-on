from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "virtual_tryon/data/temp/klein_local_masked_smoke_configs"


SAMPLE_ROOT_CANDIDATES = [
    PROJECT_ROOT / "virtual_tryon/data/temp/full15_archived_grid_eval_set",
    PROJECT_ROOT / "virtual_tryon/data/temp/catvton_flux_eval_set",
    PROJECT_ROOT / "virtual_tryon/data/outputs/catvton_flux_redux_exact_20260625",
]


def existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def sample_file(sample_id: str, names: list[str]) -> Path | None:
    candidates = []
    for root in SAMPLE_ROOT_CANDIDATES:
        for name in names:
            candidates.append(root / sample_id / name)
    return existing(candidates)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def single_sample(sample_id: str, region: str = "upper") -> dict:
    person = sample_file(sample_id, ["person.png", "input_person.png", "person_reference.png"])
    garment = sample_file(
        sample_id,
        [
            "garment_top.png",
            "garment_upper.png",
            "garment_dress.png",
            "reference.png",
            "reference_canvas.png",
        ],
    )
    missing = []
    if person is None:
        missing.append(f"{sample_id}: person.png/input_person.png")
    if garment is None:
        missing.append(f"{sample_id}: single garment image")
    if missing:
        raise FileNotFoundError("; ".join(missing))
    return {
        "sample_id": sample_id,
        "person_image": rel(person),
        "passes": [
            {
                "pass_index": 1,
                "target_region": region,
                "garment_type": region,
                "garment_image": rel(garment),
                "item_description": "the single reference garment",
                "positive_prompt": "make the garment match the reference image while preserving the person",
                "negative_prompt": "changed face, changed hands, distorted body, wrong garment, blurry, low quality",
                "seed": 2026062601,
                "steps": 8,
                "guidance": 2.5,
                "denoise": 1.0,
            }
        ],
    }


def sample_015_multipass() -> dict:
    sample_id = "sample_015"
    person = sample_file(sample_id, ["person.png", "input_person.png", "person_reference.png"])
    dress = sample_file(sample_id, ["garment_dress.png", "dress.png"])
    shoes = sample_file(sample_id, ["accessory_shoes.png", "garment_shoes.png", "shoes.png"])
    hat = sample_file(sample_id, ["accessory_hat.png", "garment_hat.png", "hat.png"])
    missing = []
    if person is None:
        missing.append("sample_015 person image")
    if dress is None:
        missing.append("sample_015 dress garment image")
    if shoes is None:
        missing.append("sample_015 shoes garment image")
    if hat is None:
        missing.append("sample_015 hat garment image")
    if missing:
        raise FileNotFoundError(
            "Missing files for strict multi-pass config. "
            "I will not use a merged reference_canvas for sample_015. Missing: "
            + ", ".join(missing)
        )
    return {
        "sample_id": sample_id,
        "person_image": rel(person),
        "passes": [
            {
                "pass_index": 1,
                "target_region": "dress",
                "garment_type": "dress",
                "garment_image": rel(dress),
                "item_description": "the dress reference image",
                "positive_prompt": "fit the dress naturally on the person",
                "negative_prompt": "missing dress, wrong dress, changed face, distorted body, blurry",
                "seed": 2026062615,
                "steps": 8,
                "guidance": 2.5,
                "denoise": 1.0,
            },
            {
                "pass_index": 2,
                "target_region": "shoes",
                "garment_type": "shoes",
                "garment_image": rel(shoes),
                "item_description": "the shoes reference image",
                "positive_prompt": "replace only the shoes and preserve legs and floor",
                "negative_prompt": "bare feet, missing shoes, changed legs, distorted floor, blurry",
                "seed": 2026062616,
                "steps": 8,
                "guidance": 2.5,
                "denoise": 1.0,
            },
            {
                "pass_index": 3,
                "target_region": "hat",
                "garment_type": "hat",
                "garment_image": rel(hat),
                "item_description": "the hat reference image",
                "positive_prompt": "place only the hat/headwear and preserve hair and face",
                "negative_prompt": "missing hat, changed face, distorted hair, blurry",
                "seed": 2026062617,
                "steps": 8,
                "guidance": 2.5,
                "denoise": 1.0,
            },
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--include-sample015", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = [single_sample("sample_001", "upper")]
    if args.include_sample015:
        samples.append(sample_015_multipass())

    config = {
        "run_name": "klein_local_masked_smoke",
        "output_root": "virtual_tryon/data/outputs/klein_local_masked_tryon_smoke_20260626",
        "workflow_json": "virtual_tryon/comfyui_workflows/klein_detailed_pipelines_20260626/04_flux2_klein9b_lora_masked_local_inpaint.workflow.json",
        "defaults": {
            "steps": 8,
            "guidance": 2.5,
            "mask_dilation_px": 2,
            "mask_erosion_px": 0,
            "mask_feather_px": 2,
            "use_sam_refine": True,
            "sam_bbox_padding_px": 8,
            "semantic_envelope_dilation_px": 14,
            "fallback_mode": "target_extent_if_empty_or_small",
            "mask_grow_px": 16,
            "mask_blur_px": 10,
            "mask_threshold": 8,
            "mask_fill_holes": True,
            "padding_ratio": 0.22,
            "min_padding_px": 32,
            "max_padding_px": 160,
            "target_multiple": 16,
            "min_crop_size": 256,
            "canvas_width": 768,
            "canvas_height": 1024,
            "lora_strength": 1.0,
            "use_catviton_lora": True,
            "timeout_seconds": 3600,
        },
        "samples": samples,
    }
    path = out_dir / "klein_local_masked_smoke_config.json"
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
