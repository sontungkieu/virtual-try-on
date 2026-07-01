from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_ROOT = PROJECT_ROOT / "virtual_tryon/data/temp/full15_archived_grid_eval_set"
OUT_DIR = PROJECT_ROOT / "virtual_tryon/data/temp/klein_local_masked_full15_configs"


PASS_PLAN: dict[str, list[tuple[str, str, str, str]]] = {
    "sample_001": [("upper", "garment_top.png", "top garment", "fit the top naturally on the upper body")],
    "sample_002": [("lower", "garment_bottom.png", "bottom garment", "fit the lower garment naturally on the legs")],
    "sample_003": [("upper", "garment_top.png", "top garment", "fit the top naturally on the upper body")],
    "sample_004": [("lower", "garment_bottom.png", "bottom garment", "fit the lower garment naturally on the legs")],
    "sample_005": [("dress", "garment_dress.png", "dress", "fit the dress naturally on the person")],
    "sample_006": [("upper", "garment_top.png", "top garment", "fit the top naturally on the upper body")],
    "sample_007": [("upper", "garment_top.png", "top garment", "fit the top naturally on the upper body")],
    "sample_008": [("upper", "garment_top.png", "top garment", "fit the top naturally on the upper body")],
    "sample_009": [("lower", "garment_bottom.png", "bottom garment", "fit the lower garment naturally on the legs")],
    "sample_010": [("lower", "garment_bottom.png", "bottom garment", "fit the lower garment naturally on the legs")],
    "sample_011": [
        ("upper", "garment_top.png", "top garment", "fit the top naturally on the upper body"),
        ("lower", "garment_bottom.png", "bottom garment", "fit the lower garment naturally on the legs"),
    ],
    "sample_012": [
        ("lower", "garment_bottom.png", "bottom garment", "fit the lower garment naturally on the legs"),
        ("hat", "accessory_hat.png", "hat", "place only the hat/headwear while preserving face and hair"),
    ],
    "sample_013": [
        ("hat", "accessory_hat.png", "hat", "place only the hat/headwear while preserving face and hair"),
        ("accessory", "accessory_watch.png", "watch accessory", "place only the watch/accessory while preserving the arm and hand"),
    ],
    "sample_014": [
        ("upper", "garment_top.png", "top garment", "fit the top naturally on the upper body"),
        ("lower", "garment_bottom.png", "bottom garment", "fit the lower garment naturally on the legs"),
        ("hat", "accessory_hat.png", "hat", "place only the hat/headwear while preserving face and hair"),
    ],
    "sample_015": [
        ("dress", "garment_dress.png", "dress", "fit the dress naturally on the person"),
        ("shoes", "accessory_shoes.png", "shoes", "replace only the shoes and preserve legs and floor"),
        ("hat", "accessory_hat.png", "hat", "place only the hat/headwear while preserving face and hair"),
    ],
}


def rel(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def make_pass(sample_id: str, pass_index: int, region: str, file_name: str, item: str, prompt: str) -> dict:
    garment_path = EVAL_ROOT / sample_id / file_name
    if not garment_path.exists():
        raise FileNotFoundError(f"Missing garment for {sample_id} pass {pass_index}: {garment_path}")
    return {
        "pass_index": pass_index,
        "target_region": region,
        "garment_type": item,
        "garment_image": rel(garment_path),
        "item_description": f"the {item} reference image",
        "positive_prompt": prompt,
        "negative_prompt": (
            "changed face, changed hands, distorted body, altered background, missing item, "
            "wrong garment, blurry, low quality, artifacts"
        ),
        "seed": 2026062600 + int(sample_id.split("_")[1]) * 10 + pass_index,
        "steps": 8,
        "guidance": 2.5,
        "denoise": 1.0,
    }


def make_sample(sample_id: str) -> dict:
    person_path = EVAL_ROOT / sample_id / "person.png"
    if not person_path.exists():
        raise FileNotFoundError(f"Missing person image for {sample_id}: {person_path}")
    passes = [
        make_pass(sample_id, index, region, file_name, item, prompt)
        for index, (region, file_name, item, prompt) in enumerate(PASS_PLAN[sample_id], start=1)
    ]
    return {
        "sample_id": sample_id,
        "person_image": rel(person_path),
        "passes": passes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument(
        "--output-root",
        default="virtual_tryon/data/outputs/klein_local_masked_tryon_full15_20260626",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = [make_sample(f"sample_{index:03d}") for index in range(1, 16)]
    config = {
        "run_name": "klein_local_masked_full15",
        "output_root": args.output_root,
        "workflow_json": (
            "virtual_tryon/comfyui_workflows/klein_detailed_pipelines_20260626/"
            "04_flux2_klein9b_lora_masked_local_inpaint.workflow.json"
        ),
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
    path = out_dir / "klein_local_masked_full15_config.json"
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
