from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from virtual_tryon.scripts.run_upstream_flux_redux_workflow import (  # noqa: E402
    NEGATIVE_PROMPT,
    build_upstream_equivalent_prompt,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "virtual_tryon/comfyui_workflows/pipelines_20260626"
SOURCE_UPSTREAM_UI = PROJECT_ROOT / "tmp/comfyuiworkflows/virtual-tryon-flux-redux-workflow.json"
KLEIN_WORKFLOW_DIR = PROJECT_ROOT / "virtual_tryon/comfyui_workflows"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def model_paths() -> dict[str, str]:
    return {
        "flux_fill": "models/unet/FLUX1/fluxFillFP8_v10.safetensors",
        "flux_redux": "models/style_models/flux1-redux-dev.safetensors",
        "sigclip": "models/clip_vision/sigclip_vision_patch14_384.safetensors",
        "catviton_lora": "models/loras/flux/catvton-flux-lora.safetensors",
        "vae": "models/vae/FLUX1/ae.safetensors",
        "clip_l": "models/clip/clip_l.safetensors",
        "t5xxl": "models/clip/t5xxl_fp8_e4m3fn.safetensors",
    }


def patch_common_tryon_settings(
    prompt: dict[str, Any],
    *,
    positive_prompt: str,
    negative_prompt: str,
    steps: int,
    guidance: float,
    denoise: float,
    redux_strength: float,
    lora_strength: float,
) -> dict[str, Any]:
    patched = json.loads(json.dumps(prompt))
    if "7" in patched:
        patched["7"]["inputs"]["text"] = positive_prompt
    if "8" in patched:
        patched["8"]["inputs"]["guidance"] = guidance
    if "9" in patched:
        # The upstream-equivalent graph uses ConditioningZeroOut for the negative
        # branch. Keeping the text here as metadata makes the API JSON self-documenting.
        patched["9"].setdefault("_note", f"negative_prompt: {negative_prompt}")
    if "13" in patched:
        patched["13"]["inputs"]["strength"] = redux_strength
    if "17" in patched:
        patched["17"]["inputs"]["strength_model"] = lora_strength
    if "18" in patched:
        patched["18"]["inputs"]["steps"] = steps
        patched["18"]["inputs"]["cfg"] = 1
        patched["18"]["inputs"]["denoise"] = denoise
    return patched


def build_video_single_pass_api() -> dict[str, Any]:
    prompt = build_upstream_equivalent_prompt(
        person_name="person.png",
        reference_name="garment.png",
        mask_name="mask.png",
        sample_id="single_garment",
        prompt_mode="tryon_prompt",
        seed=2026062601,
        filename_prefix="vton_workflows/flux_redux_catvton_single_pass",
    )
    return patch_common_tryon_settings(
        prompt,
        positive_prompt=(
            "Virtual try-on photo. Replace only the masked region with the reference garment. "
            "Preserve identity, face, hair, hands, body shape, pose, lighting, and background."
        ),
        negative_prompt=NEGATIVE_PROMPT,
        steps=24,
        guidance=3.5,
        denoise=1.0,
        redux_strength=1.0,
        lora_strength=1.0,
    )


def build_video_item_pass_api(
    *,
    person_name: str,
    garment_name: str,
    mask_name: str,
    target_region: str,
    seed: int,
    filename_prefix: str,
    steps: int = 24,
    guidance: float = 3.5,
    denoise: float = 1.0,
    redux_strength: float = 1.0,
    lora_strength: float = 1.0,
) -> dict[str, Any]:
    prompt = build_upstream_equivalent_prompt(
        person_name=person_name,
        reference_name=garment_name,
        mask_name=mask_name,
        sample_id=target_region,
        prompt_mode="tryon_prompt",
        seed=seed,
        filename_prefix=filename_prefix,
    )
    return patch_common_tryon_settings(
        prompt,
        positive_prompt=(
            f"Virtual try-on photo. Replace only the {target_region} masked region "
            "with the single reference item. Preserve all unmasked body parts, face, "
            "hair, hands, pose, lighting, and background."
        ),
        negative_prompt=NEGATIVE_PROMPT,
        steps=steps,
        guidance=guidance,
        denoise=denoise,
        redux_strength=redux_strength,
        lora_strength=lora_strength,
    )


def build_production_fallback_api(
    *,
    person_name: str,
    garment_name: str,
    mask_name: str,
    target_region: str,
    seed: int,
    steps: int,
    guidance: float,
    denoise: float,
    redux_strength: float,
    lora_strength: float,
    filename_prefix: str,
) -> dict[str, Any]:
    positive_prompt = (
        f"Virtual try-on photo. Inpaint only the {target_region} mask with the reference garment. "
        "Keep identity, face, hands, skin, body shape, pose, background, and lighting unchanged."
    )
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": person_name}},
        "2": {"class_type": "LoadImage", "inputs": {"image": mask_name}},
        "3": {"class_type": "ImageToMask", "inputs": {"image": ["2", 0], "channel": "red"}},
        "4": {"class_type": "LoadImage", "inputs": {"image": garment_name}},
        "5": {
            "class_type": "DualCLIPLoader",
            "inputs": {"clip_name1": "clip_l.safetensors", "clip_name2": "t5xxl_fp8_e4m3fn.safetensors", "type": "flux"},
        },
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["5", 0], "text": positive_prompt}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["5", 0], "text": NEGATIVE_PROMPT}},
        "8": {"class_type": "FluxGuidance", "inputs": {"conditioning": ["6", 0], "guidance": guidance}},
        "9": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": "sigclip_vision_patch14_384.safetensors"}},
        "10": {"class_type": "CLIPVisionEncode", "inputs": {"clip_vision": ["9", 0], "image": ["4", 0], "crop": "none"}},
        "11": {"class_type": "StyleModelLoader", "inputs": {"style_model_name": "flux1-redux-dev.safetensors"}},
        "12": {
            "class_type": "StyleModelApply",
            "inputs": {
                "conditioning": ["8", 0],
                "style_model": ["11", 0],
                "clip_vision_output": ["10", 0],
                "strength": redux_strength,
                "strength_type": "multiply",
            },
        },
        "13": {"class_type": "VAELoader", "inputs": {"vae_name": "FLUX1/ae.safetensors"}},
        "14": {"class_type": "UNETLoader", "inputs": {"unet_name": "FLUX1/fluxFillFP8_v10.safetensors", "weight_dtype": "default"}},
        "15": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": ["14", 0], "lora_name": "flux/catvton-flux-lora.safetensors", "strength_model": lora_strength},
        },
        "16": {
            "class_type": "InpaintModelConditioning",
            "inputs": {
                "positive": ["12", 0],
                "negative": ["7", 0],
                "vae": ["13", 0],
                "pixels": ["1", 0],
                "mask": ["3", 0],
                "noise_mask": True,
            },
        },
        "17": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["15", 0],
                "seed": seed,
                "steps": steps,
                "cfg": 1,
                "sampler_name": "euler",
                "scheduler": "normal",
                "positive": ["16", 0],
                "negative": ["16", 1],
                "latent_image": ["16", 2],
                "denoise": denoise,
            },
        },
        "18": {"class_type": "VAEDecode", "inputs": {"samples": ["17", 0], "vae": ["13", 0]}},
        "19": {"class_type": "SaveImage", "inputs": {"images": ["18", 0], "filename_prefix": filename_prefix}},
    }


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def build_readme() -> str:
    return """# ComfyUI Pipeline Workflows

This folder packages the virtual try-on directions used in Phase 2 into importable ComfyUI workflow JSON files.

## Pipelines

| File | Type | Pipeline | Input contract |
|---|---|---|---|
| `00_video_repo_flux_redux_ui_source.json` | UI workflow | Exact workflow copied from `fahdmirza/comfyuiworkflows` when available | Open in ComfyUI UI, then patch image/prompt widgets |
| `01_flux_redux_catvton_single_pass_api.json` | API workflow | Flux Fill + Redux garment reference + CatVTON LoRA + mask inpaint | `person.png`, `garment.png`, `mask.png` |
| `02_schp_sam_mask_consumer_single_pass_api.json` | API workflow | Same try-on graph, intended to consume SCHP/SAM/manual processed masks | `person.png`, `garment.png`, `mask_processed.png` |
| `03_multipass_sample015_pass01_dress_api.json` | API workflow | Multi-pass item 1, dress only | `sample015_person.png`, `sample015_dress.png`, `sample015_dress_mask.png` |
| `04_multipass_sample015_pass02_shoes_api.json` | API workflow | Multi-pass item 2, shoes only | previous pass output as `sample015_after_dress.png`, shoes image, shoes mask |
| `05_multipass_sample015_pass03_hat_api.json` | API workflow | Multi-pass item 3, hat only | previous pass output as `sample015_after_shoes.png`, hat image, hat mask |
| `06_refine_low_denoise_api.json` | API workflow | Optional border/detail refine pass | `base_output.png`, `garment.png`, `artifact_or_edge_mask.png` |
| `07_production_fallback_flux_fill_redux_catvton_api.json` | API workflow | Production fallback graph without IC-LoRA image packing | `person.png`, `garment.png`, `mask.png` |
| `10_klein_4step_sample015_ui.json` | UI workflow | Local Klein 4-step fast preset | custom Klein nodes |
| `11_klein_28_sample015_ui.json` | UI workflow | Local Klein 28-step preset | custom Klein nodes |
| `12_klein_28_strong_sample015_ui.json` | UI workflow | Local Klein 28-step stronger prompt preset | custom Klein nodes |

## Important Contract

The Flux Redux + CatVTON workflows are single-garment passes. Do not put dress, shoes, and hat into one reference canvas. For a full outfit, run sequentially:

1. dress mask + dress garment
2. shoes mask + shoes garment, using pass 1 output as the new person image
3. hat mask + hat garment, using pass 2 output as the new person image

The SCHP/SAM pipeline generates masks outside ComfyUI and these workflows consume the resulting mask PNGs. Rectangle masks are debug-only.

## Required ComfyUI Models

Paths are relative to `/workspace/ComfyUI`:

```text
models/unet/FLUX1/fluxFillFP8_v10.safetensors
models/style_models/flux1-redux-dev.safetensors
models/clip_vision/sigclip_vision_patch14_384.safetensors
models/loras/flux/catvton-flux-lora.safetensors
models/vae/FLUX1/ae.safetensors
models/clip/clip_l.safetensors
models/clip/t5xxl_fp8_e4m3fn.safetensors
```

## Required Custom Nodes

The video-equivalent Flux workflow uses `AddMaskForICLora`, `GrowMask`, `ImageCrop`, Flux nodes, Redux style model nodes, and standard ComfyUI loaders/samplers.

The Klein workflows use the local Phase 2 custom nodes:

```text
VTON Phase2 - Klein Reference Set
VTON Phase2 - Klein Local Sampler
```
"""


def export_workflows(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []

    if copy_if_exists(SOURCE_UPSTREAM_UI, output_dir / "00_video_repo_flux_redux_ui_source.json"):
        entries.append(
            {
                "file": "00_video_repo_flux_redux_ui_source.json",
                "type": "ui_workflow",
                "pipeline": "exact_source_video_workflow",
                "source": str(SOURCE_UPSTREAM_UI),
                "status": "copied",
            }
        )
    else:
        entries.append(
            {
                "file": "00_video_repo_flux_redux_ui_source.json",
                "type": "ui_workflow",
                "pipeline": "exact_source_video_workflow",
                "status": "missing_source",
            }
        )

    workflow_specs: list[tuple[str, str, dict[str, Any], str]] = [
        (
            "01_flux_redux_catvton_single_pass_api.json",
            "flux_redux_catvton_single_pass",
            build_video_single_pass_api(),
            "Video-equivalent single-garment pass. Uses Flux Fill, Redux garment reference, CatVTON LoRA, mask inpaint, and IC-LoRA image packing.",
        ),
        (
            "02_schp_sam_mask_consumer_single_pass_api.json",
            "schp_sam_mask_consumer_single_pass",
            build_video_item_pass_api(
                person_name="person.png",
                garment_name="garment.png",
                mask_name="mask_processed.png",
                target_region="upper",
                seed=2026062602,
                filename_prefix="vton_workflows/schp_sam_mask_consumer_single_pass",
                redux_strength=1.0,
            ),
            "Same try-on graph, but named for the production mask pipeline. It expects a processed mask generated by SCHP/SAM/manual tools.",
        ),
        (
            "03_multipass_sample015_pass01_dress_api.json",
            "multipass_sample015_dress",
            build_video_item_pass_api(
                person_name="sample015_person.png",
                garment_name="sample015_dress.png",
                mask_name="sample015_dress_mask.png",
                target_region="dress",
                seed=2026062615,
                filename_prefix="vton_workflows/sample015/pass01_dress",
                redux_strength=1.0,
            ),
            "Pass 1 of sample 015 multi-item outfit: dress only.",
        ),
        (
            "04_multipass_sample015_pass02_shoes_api.json",
            "multipass_sample015_shoes",
            build_video_item_pass_api(
                person_name="sample015_after_dress.png",
                garment_name="sample015_shoes.png",
                mask_name="sample015_shoes_mask.png",
                target_region="shoes",
                seed=2026062616,
                filename_prefix="vton_workflows/sample015/pass02_shoes",
                steps=28,
                redux_strength=1.1,
            ),
            "Pass 2 of sample 015 multi-item outfit: shoes only, using pass 1 output as person input.",
        ),
        (
            "05_multipass_sample015_pass03_hat_api.json",
            "multipass_sample015_hat",
            build_video_item_pass_api(
                person_name="sample015_after_shoes.png",
                garment_name="sample015_hat.png",
                mask_name="sample015_hat_mask.png",
                target_region="hat",
                seed=2026062617,
                filename_prefix="vton_workflows/sample015/pass03_hat",
                steps=28,
                redux_strength=1.1,
            ),
            "Pass 3 of sample 015 multi-item outfit: hat only, using pass 2 output as person input.",
        ),
        (
            "06_refine_low_denoise_api.json",
            "optional_low_denoise_refine",
            build_video_item_pass_api(
                person_name="base_output.png",
                garment_name="garment.png",
                mask_name="artifact_or_edge_mask.png",
                target_region="artifact/border",
                seed=2026062699,
                filename_prefix="vton_workflows/refine_low_denoise",
                steps=12,
                guidance=2.5,
                denoise=0.25,
                redux_strength=0.6,
                lora_strength=0.75,
            ),
            "Optional refine pass for seam/artifact masks only. Do not mask the whole outfit here.",
        ),
        (
            "07_production_fallback_flux_fill_redux_catvton_api.json",
            "production_fallback_flux_fill_redux_catvton",
            build_production_fallback_api(
                person_name="person.png",
                garment_name="garment.png",
                mask_name="mask.png",
                target_region="upper",
                seed=2026062607,
                steps=24,
                guidance=3.5,
                denoise=1.0,
                redux_strength=0.75,
                lora_strength=1.0,
                filename_prefix="vton_workflows/production_fallback",
            ),
            "Fallback API graph from the production runner. It uses Redux + LoRA but does not pack garment/person with AddMaskForICLora.",
        ),
    ]

    for filename, pipeline, prompt, description in workflow_specs:
        write_json(output_dir / filename, prompt)
        entries.append(
            {
                "file": filename,
                "type": "api_workflow",
                "pipeline": pipeline,
                "description": description,
                "input_contract": {
                    "person_image": "single person/current pass image in ComfyUI input",
                    "garment_image": "one garment/reference image only",
                    "mask_image": "target-region mask image, white means editable",
                },
            }
        )

    klein_sources = [
        ("vton_phase2_klein_4step_sample015.workflow.json", "10_klein_4step_sample015_ui.json", "klein_4step"),
        ("vton_phase2_klein_28_sample015.workflow.json", "11_klein_28_sample015_ui.json", "klein_28"),
        ("vton_phase2_klein_28_strong_sample015.workflow.json", "12_klein_28_strong_sample015_ui.json", "klein_28_strong"),
    ]
    for src_name, dst_name, pipeline in klein_sources:
        src = KLEIN_WORKFLOW_DIR / src_name
        status = "copied" if copy_if_exists(src, output_dir / dst_name) else "missing_source"
        entries.append(
            {
                "file": dst_name,
                "type": "ui_workflow",
                "pipeline": pipeline,
                "source": str(src),
                "status": status,
                "required_custom_nodes": [
                    "VTON Phase2 - Klein Reference Set",
                    "VTON Phase2 - Klein Local Sampler",
                ],
            }
        )

    index = {
        "generated_by": "virtual_tryon/scripts/export_comfyui_pipeline_workflows.py",
        "output_dir": str(output_dir),
        "model_paths_relative_to_comfyui": model_paths(),
        "workflows": entries,
        "notes": [
            "API workflows can be queued through ComfyUI /prompt.",
            "UI workflows are meant to be loaded in the ComfyUI editor.",
            "All Flux Redux + CatVTON passes are single-garment contracts.",
            "Multi-item outfits must be run sequentially, with each output becoming the next pass person image.",
        ],
    }
    write_json(output_dir / "index.json", index)
    (output_dir / "README.md").write_text(build_readme(), encoding="utf-8")
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description="Export packaged ComfyUI workflow JSONs for Phase 2 VTON pipelines.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    index = export_workflows(args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir), "workflow_count": len(index["workflows"])}, indent=2))


if __name__ == "__main__":
    main()
