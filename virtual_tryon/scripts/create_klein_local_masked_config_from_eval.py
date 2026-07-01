from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from virtual_tryon.final_demo import (
    DEFAULT_FINAL_EVAL_ROOT,
    DEFAULT_FINAL_OUTPUT_ROOT,
    FINAL_PASS_PLAN,
    local_pass_seed,
    sample_id_for_index,
)

DEFAULT_EVAL_ROOT = DEFAULT_FINAL_EVAL_ROOT
DEFAULT_CONFIG_OUTPUT = PROJECT_ROOT / "virtual_tryon/data/temp/final15_data_input_configs/klein_local_masked_config.json"
DEFAULT_OUTPUT_ROOT = DEFAULT_FINAL_OUTPUT_ROOT / "method_04_klein_lora_local_masked_inpaint"


def rel(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def make_pass(eval_root: Path, sample_id: str, pass_index: int, spec) -> dict:
    garment_path = eval_root / sample_id / spec.normalized_name
    if not garment_path.exists():
        raise FileNotFoundError(f"Missing garment for {sample_id} pass {pass_index}: {garment_path}")
    return {
        "pass_index": pass_index,
        "target_region": spec.region,
        "garment_type": spec.garment_type,
        "garment_image": rel(garment_path),
        "item_description": f"the {spec.garment_type} reference image",
        "positive_prompt": spec.positive_prompt,
        "negative_prompt": (
            "changed face, changed hands, distorted body, altered background, missing item, "
            "wrong garment, blurry, low quality, artifacts"
        ),
        "seed": local_pass_seed(sample_id, pass_index),
        "steps": 8,
        "guidance": 2.5,
        "denoise": 1.0,
    }


def make_sample(eval_root: Path, sample_id: str) -> dict:
    person_path = eval_root / sample_id / "person.png"
    if not person_path.exists():
        raise FileNotFoundError(f"Missing person image for {sample_id}: {person_path}")
    passes = [
        make_pass(eval_root, sample_id, index, spec)
        for index, spec in enumerate(FINAL_PASS_PLAN[sample_id], start=1)
    ]
    return {"sample_id": sample_id, "person_image": rel(person_path), "passes": passes}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--config-output", type=Path, default=DEFAULT_CONFIG_OUTPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    samples = [make_sample(args.eval_root, sample_id_for_index(index)) for index in range(1, 16)]
    config = {
        "run_name": "klein_lora_local_masked_inpaint_final15_data_input",
        "output_root": rel(args.output_root),
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
    args.config_output.parent.mkdir(parents=True, exist_ok=True)
    args.config_output.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(args.config_output.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
