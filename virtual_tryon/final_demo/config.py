from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_INPUT_ROOT = PROJECT_ROOT / "Data_input"
DEFAULT_FINAL_EVAL_ROOT = PROJECT_ROOT / "virtual_tryon/data/temp/final15_data_input_eval_set"
DEFAULT_FINAL_OUTPUT_ROOT = PROJECT_ROOT / "virtual_tryon/data/outputs/FINAL OUTPUT"
CANVAS_SIZE = (768, 1024)
SAMPLE_COUNT = 15


@dataclass(frozen=True)
class SourceItemSpec:
    region: str
    source_name: str
    normalized_name: str
    garment_type: str
    positive_prompt: str


@dataclass(frozen=True)
class MethodSpec:
    key: str
    title: str
    output_template: str
    pipeline_summary: str
    quality_summary: str
    strengths: tuple[str, ...]
    weaknesses: tuple[str, ...]

    def output_path(self, output_root: Path, sample_id: str) -> Path:
        return output_root / self.output_template.format(sample_id=sample_id)


REGION_LABELS = {
    "upper": "top",
    "lower": "bottom",
    "dress": "dress",
    "shoes": "shoes",
    "hat": "hat",
    "accessory": "accessory",
}

GARMENT_FILE_BY_REGION = {
    "upper": "garment_top.png",
    "lower": "garment_bottom.png",
    "dress": "garment_dress.png",
    "shoes": "accessory_shoes.png",
    "hat": "accessory_hat.png",
    "accessory": "accessory_watch.png",
}

REGION_PROMPTS = {
    "upper": "Replace only the masked upper garment with the reference top.",
    "lower": "Replace only the masked lower garment with the reference lower garment.",
    "dress": "Replace only the masked body garment with the reference dress.",
    "shoes": "Replace only the masked feet region with the reference shoes.",
    "hat": "Add the reference hat only inside the masked head area.",
    "accessory": "Add only the reference accessory inside the masked area.",
}

LOCAL_PASS_PROMPTS = {
    "upper": "fit the top naturally on the upper body",
    "lower": "fit the lower garment naturally on the legs",
    "dress": "fit the dress naturally on the person",
    "shoes": "replace only the shoes and preserve legs and floor",
    "hat": "place only the hat/headwear while preserving face and hair",
    "accessory": "place only the watch/accessory while preserving the arm and hand",
}


def item(region: str, source_name: str) -> SourceItemSpec:
    garment_type = REGION_LABELS.get(region, region)
    return SourceItemSpec(
        region=region,
        source_name=source_name,
        normalized_name=GARMENT_FILE_BY_REGION[region],
        garment_type=garment_type if region != "accessory" else "watch accessory",
        positive_prompt=LOCAL_PASS_PROMPTS.get(region, f"fit the {garment_type} naturally"),
    )


SOURCE_SAMPLE_PLAN: dict[int, tuple[SourceItemSpec, ...]] = {
    1: (item("upper", "Garment.png"),),
    2: (item("lower", "Garmet.png"),),
    3: (item("upper", "Garment.png"),),
    4: (item("lower", "Garment.png"),),
    5: (item("dress", "Garment.png"),),
    6: (item("upper", "Garment.png"),),
    7: (item("upper", "Garment.png"),),
    8: (item("upper", "Garment.png"),),
    9: (item("lower", "Garment.png"),),
    10: (item("lower", "Garment.png"),),
    11: (item("upper", "Garment1.png"), item("lower", "Garment2.png")),
    12: (item("lower", "Garment1.png"), item("hat", "Garment2.png")),
    13: (item("hat", "Garment1.png"), item("accessory", "Garment2.png")),
    14: (item("upper", "Garment1.png"), item("lower", "Garment2.png"), item("hat", "Garment3.png")),
    15: (item("dress", "Garment1.png"), item("shoes", "Garment2.png"), item("hat", "Garment3.png")),
}

FINAL_PASS_PLAN: dict[str, tuple[SourceItemSpec, ...]] = {
    f"sample_{index:03d}": specs for index, specs in SOURCE_SAMPLE_PLAN.items()
}

FINAL_METHODS: tuple[MethodSpec, ...] = (
    MethodSpec(
        key="method_01",
        title="SCHP/SAM\nFlux Fill + CatVTON",
        output_template="method_01_schp_sam_flux_catvton/{sample_id}/final_output.png",
        pipeline_summary=(
            "SCHP/ATR human parsing and SAM refine the target mask; Flux Fill + CatVTON/Redux "
            "inpaints one garment region at a time; multi-item outfits run sequential passes."
        ),
        quality_summary=(
            "Best at preserving the original background and identity when the mask is correct; "
            "sensitive to mask quality and weak on tiny accessories."
        ),
        strengths=("uses explicit masks", "supports sequential multi-pass outfits", "keeps non-masked regions stable"),
        weaknesses=("mask errors damage fit", "small hats/shoes/watches are unreliable", "style transfer can be weak"),
    ),
    MethodSpec(
        key="method_02",
        title="Klein 9B",
        output_template="method_02_klein9b/{sample_id}/output.png",
        pipeline_summary=(
            "FLUX.2 Klein 9B receives a fitted person canvas and a garment/reference canvas, then "
            "generates a full 768x1024 output with the default try-on prompt."
        ),
        quality_summary=(
            "Can make visually pleasing fashion images, but it is global generation rather than strict "
            "try-on and can change pose, body, background, or ignore reference details."
        ),
        strengths=("simple graph", "often produces clean full-body images", "useful as a generative baseline"),
        weaknesses=("no hard mask", "not identity-stable", "multi-item reference canvases are ambiguous"),
    ),
    MethodSpec(
        key="method_03",
        title="Klein 9B\n+ Try-On LoRA",
        output_template="method_03_klein9b_tryon_lora/{sample_id}/output.png",
        pipeline_summary=(
            "Same Klein 9B global pipeline, with fal/flux-klein-9b-virtual-tryon-lora applied at "
            "LoRA strength 1.0 and default prompt strength."
        ),
        quality_summary=(
            "Usually improves clothing adherence over raw Klein, but still has no strict mask or warp "
            "constraint, so body and background drift remain possible."
        ),
        strengths=("better garment intent than base Klein", "good for broad clothing edits", "easy ablation against base Klein"),
        weaknesses=("can still ignore small items", "can change identity or pose", "not a production-safe local edit"),
    ),
    MethodSpec(
        key="method_04",
        title="Klein+LoRA\nlocal masked inpaint",
        output_template="method_04_klein_lora_local_masked_inpaint/{sample_id}/final_output.png",
        pipeline_summary=(
            "SCHP/SAM builds a target mask, the masked area is cropped, Klein 9B + Try-On LoRA generates "
            "the local crop, and the result is pasted back into the original image."
        ),
        quality_summary=(
            "Most production-oriented because it localizes edits and preserves the original image. Quality "
            "depends heavily on crop context and mask fit; too-small masks can leave the input unchanged."
        ),
        strengths=("local edit contract", "debuggable masks/crops/overlays", "sequential multi-pass support"),
        weaknesses=("mask/crop tuning is critical", "seams can appear", "accessories remain hard"),
    ),
)


def sample_id_for_index(index: int) -> str:
    return f"sample_{index:03d}"


def garment_file_for_region(region: str) -> str:
    try:
        return GARMENT_FILE_BY_REGION[region]
    except KeyError as exc:
        raise KeyError(f"Unsupported target region: {region}") from exc


def category_for_regions(regions: list[str]) -> str:
    if len(regions) > 1:
        return "multi_item"
    region = regions[0]
    return {
        "upper": "upper_body",
        "lower": "lower_body",
        "dress": "dress",
        "shoes": "shoes",
        "hat": "hat",
        "accessory": "accessory",
    }.get(region, region)


def region_prompt(sample_id: str, region: str) -> str:
    task = REGION_PROMPTS.get(region, "Replace only the masked region with the reference item.")
    return (
        f"Virtual try-on photo. {task} Preserve the person's identity, face, hair, hands, "
        f"body shape, pose, unmasked clothing, lighting, and background. Sample {sample_id}."
    )


def sample_region(metadata: dict[str, Any]) -> str:
    passes = metadata.get("passes") or []
    regions = [str(item.get("region")) for item in passes if item.get("region")]
    if len(regions) == 1:
        return regions[0]
    return "full_outfit"


def item_description(metadata: dict[str, Any]) -> str:
    passes = metadata.get("passes") or []
    labels = [str(item.get("label") or item.get("region")) for item in passes]
    if len(labels) == 1:
        return f"the reference {labels[0]} item"
    return "the complete outfit shown in the reference image"


def local_pass_seed(sample_id: str, pass_index: int) -> int:
    return 2026062600 + int(sample_id.split("_")[1]) * 10 + pass_index


def final_method_paths(output_root: Path, sample_id: str) -> dict[str, Path]:
    return {method.key: method.output_path(output_root, sample_id) for method in FINAL_METHODS}
