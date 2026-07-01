from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
COMFY_ROOT = Path("/workspace/ComfyUI")
COMFY_INPUT = COMFY_ROOT / "input"
COMFY_OUTPUT = COMFY_ROOT / "output"
COMFY_URL = "http://127.0.0.1:8188"

DEFAULT_WORKFLOW_JSON = (
    PROJECT_ROOT
    / "virtual_tryon/comfyui_workflows/klein_detailed_pipelines_20260626/"
    / "04_flux2_klein9b_lora_masked_local_inpaint.workflow.json"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "virtual_tryon/data/outputs/klein_local_masked_tryon_20260626"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "virtual_tryon/models/flux2-klein-9b"
DEFAULT_LORA_PATH = Path(
    "/workspace/hf-cache/hub/models--fal--flux-klein-9b-virtual-tryon-lora/"
    "snapshots/8b078b15c6d958ce48892b9ef31b66aa7587d792/flux-klein-tryon.safetensors"
)


def post_json(path: str, payload: dict[str, Any], timeout: int = 30) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{COMFY_URL}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def get_json(path: str, timeout: int = 30) -> dict[str, Any]:
    with urllib.request.urlopen(f"{COMFY_URL}{path}", timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def copy_to_comfy_input(path: Path, name: str) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    COMFY_INPUT.mkdir(parents=True, exist_ok=True)
    target = COMFY_INPUT / name
    shutil.copy2(path, target)
    return name


def queue_and_wait(prompt: dict[str, Any], timeout_seconds: int = 3600) -> dict[str, Any]:
    response = post_json("/prompt", {"prompt": prompt})
    prompt_id = response["prompt_id"]
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        time.sleep(3)
        try:
            history = get_json(f"/history/{prompt_id}")
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "prompt_id": prompt_id,
                        "poll_warning": f"{type(exc).__name__}: {exc}",
                        "action": "retry_history_poll",
                    }
                ),
                flush=True,
            )
            continue
        if prompt_id not in history:
            continue
        item = history[prompt_id]
        status = item.get("status", {})
        if status.get("completed") or status.get("status_str") == "error":
            return item | {"prompt_id": prompt_id}
    raise TimeoutError(f"ComfyUI prompt timed out: {prompt_id}")


def saved_images(history_item: dict[str, Any]) -> list[Path]:
    images: list[Path] = []
    for output in history_item.get("outputs", {}).values():
        for image in output.get("images") or []:
            subfolder = image.get("subfolder") or ""
            images.append(COMFY_OUTPUT / subfolder / image["filename"])
    return images


def copy_saved_by_prefix(saved: list[Path], pass_dir: Path, expected: dict[str, str]) -> dict[str, str]:
    copied: dict[str, str] = {}
    for key, prefix in expected.items():
        match = None
        for path in saved:
            if path.name.startswith(f"{prefix}_") or path.name.startswith(prefix):
                match = path
        if match and match.exists():
            dst = pass_dir / f"{key}.png"
            shutil.copy2(match, dst)
            copied[key] = str(dst)
    return copied


def copy_saved_from_output_folder(prefix: str, pass_dir: Path, expected: dict[str, str], copied: dict[str, str]) -> dict[str, str]:
    output_dir = COMFY_OUTPUT / prefix
    for key, file_prefix in expected.items():
        if key in copied:
            continue
        cached_path = pass_dir / f"{key}.png"
        if cached_path.is_file():
            copied[key] = str(cached_path)
            continue
        matches = sorted(output_dir.glob(f"{file_prefix}_*.png"), key=lambda path: path.stat().st_mtime)
        if not matches:
            continue
        dst = pass_dir / f"{key}.png"
        shutil.copy2(matches[-1], dst)
        copied[key] = str(dst)
    return copied


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def pass_prompt(pass_cfg: dict[str, Any]) -> str:
    positive = str(pass_cfg.get("positive_prompt") or "").strip()
    item = str(pass_cfg.get("item_description") or "the single reference garment").strip()
    target = str(pass_cfg.get("target_region") or "upper")
    if positive:
        return (
            f"TRYON cropped local fashion photo. Replace only the {target} target region with {item}. "
            f"{positive} Preserve identity, body shape, pose, skin, hands, lighting, camera angle, and background. "
            "Do not change unrelated regions. Photorealistic, natural fit, clean garment boundaries."
        )
    return (
        f"TRYON cropped local fashion photo. Replace only the {target} target region with {item}. "
        "Preserve identity, body shape, pose, skin, hands, lighting, camera angle, and background. "
        "Do not change unrelated regions. Photorealistic, natural fit, clean garment boundaries."
    )


def build_prompt_graph(
    *,
    person_name: str,
    garment_name: str,
    pass_cfg: dict[str, Any],
    defaults: dict[str, Any],
    filename_prefix: str,
) -> dict[str, Any]:
    target_region = str(pass_cfg.get("target_region") or defaults.get("target_region") or "upper")
    seed = int(pass_cfg.get("seed", defaults.get("seed", 2026062601)))
    steps = int(pass_cfg.get("steps", defaults.get("steps", 8)))
    guidance = float(pass_cfg.get("guidance", defaults.get("guidance", 2.5)))
    lora_strength = float(pass_cfg.get("lora_strength", defaults.get("lora_strength", 1.0)))
    use_lora = bool(pass_cfg.get("use_catviton_lora", defaults.get("use_catviton_lora", True)))
    model_dir = str(resolve_path(pass_cfg.get("model_dir") or defaults.get("model_dir") or DEFAULT_MODEL_DIR))
    lora_path = str(resolve_path(pass_cfg.get("lora_path") or defaults.get("lora_path") or DEFAULT_LORA_PATH))
    prompt = pass_prompt(pass_cfg)

    raw = f"{filename_prefix}/mask_raw"
    processed = f"{filename_prefix}/mask_processed"
    overlay = f"{filename_prefix}/mask_overlay"
    bbox = f"{filename_prefix}/bbox_overlay"
    crop = f"{filename_prefix}/crop_image"
    crop_mask = f"{filename_prefix}/crop_mask"
    generated = f"{filename_prefix}/generated_crop"
    final = f"{filename_prefix}/output_base"
    debug = f"{filename_prefix}/debug_sheet"

    pipeline_source = ["12", 0]
    nodes: dict[str, Any] = {
        "1": {"class_type": "LoadImage", "inputs": {"image": person_name}},
        "2": {"class_type": "LoadImage", "inputs": {"image": garment_name}},
        "3": {
            "class_type": "VTONPhase2SCHPSAMMask",
            "inputs": {
                "person_image": ["1", 0],
                "target_region": target_region,
                "dilation_px": int(pass_cfg.get("mask_dilation_px", defaults.get("mask_dilation_px", 2))),
                "erosion_px": int(pass_cfg.get("mask_erosion_px", defaults.get("mask_erosion_px", 0))),
                "feather_px": int(pass_cfg.get("mask_feather_px", defaults.get("mask_feather_px", 2))),
                "use_sam_refine": bool(pass_cfg.get("use_sam_refine", defaults.get("use_sam_refine", True))),
                "sam_bbox_padding_px": int(pass_cfg.get("sam_bbox_padding_px", defaults.get("sam_bbox_padding_px", 8))),
                "semantic_envelope_dilation_px": int(
                    pass_cfg.get(
                        "semantic_envelope_dilation_px",
                        defaults.get("semantic_envelope_dilation_px", 14),
                    )
                ),
                "fallback_mode": str(
                    pass_cfg.get("fallback_mode", defaults.get("fallback_mode", "target_extent_if_empty_or_small"))
                ),
            },
        },
        "4": {
            "class_type": "VTONPhase2MaskMorphology",
            "inputs": {
                "mask": ["3", 0],
                "grow_px": int(pass_cfg.get("mask_grow_px", defaults.get("mask_grow_px", 16))),
                "blur_px": int(pass_cfg.get("mask_blur_px", defaults.get("mask_blur_px", 10))),
                "threshold": int(pass_cfg.get("mask_threshold", defaults.get("mask_threshold", 8))),
                "invert": bool(pass_cfg.get("mask_invert", defaults.get("mask_invert", False))),
                "fill_holes": bool(pass_cfg.get("mask_fill_holes", defaults.get("mask_fill_holes", True))),
                "keep_largest_component": bool(
                    pass_cfg.get("mask_keep_largest_component", defaults.get("mask_keep_largest_component", False))
                ),
            },
        },
        "5": {
            "class_type": "VTONPhase2MaskBBoxCrop",
            "inputs": {
                "image": ["1", 0],
                "mask": ["4", 0],
                "padding_ratio": float(pass_cfg.get("padding_ratio", defaults.get("padding_ratio", 0.22))),
                "min_padding_px": int(pass_cfg.get("min_padding_px", defaults.get("min_padding_px", 32))),
                "max_padding_px": int(pass_cfg.get("max_padding_px", defaults.get("max_padding_px", 160))),
                "force_square": bool(pass_cfg.get("force_square", defaults.get("force_square", False))),
                "target_multiple": int(pass_cfg.get("target_multiple", defaults.get("target_multiple", 16))),
                "min_crop_size": int(pass_cfg.get("min_crop_size", defaults.get("min_crop_size", 256))),
                "target_region": target_region,
            },
        },
        "6": {"class_type": "VTONPhase2MaskPreviewImage", "inputs": {"mask": ["3", 1]}},
        "7": {"class_type": "VTONPhase2MaskPreviewImage", "inputs": {"mask": ["4", 0]}},
        "8": {"class_type": "VTONPhase2MaskPreviewImage", "inputs": {"mask": ["5", 1]}},
        "9": {
            "class_type": "VTONPhase2FitCanvasWithMeta",
            "inputs": {
                "image": ["5", 0],
                "width": int(pass_cfg.get("canvas_width", defaults.get("canvas_width", 768))),
                "height": int(pass_cfg.get("canvas_height", defaults.get("canvas_height", 1024))),
                "background": "white",
            },
        },
        "10": {
            "class_type": "VTONPhase2KleinFitCanvas",
            "inputs": {
                "image": ["2", 0],
                "width": int(pass_cfg.get("canvas_width", defaults.get("canvas_width", 768))),
                "height": int(pass_cfg.get("canvas_height", defaults.get("canvas_height", 1024))),
            },
        },
        "12": {"class_type": "VTONPhase2KleinLoadBaseModel", "inputs": {"model_dir": model_dir}},
    }
    if use_lora:
        nodes["13"] = {
            "class_type": "VTONPhase2KleinLoadTryOnLoRA",
            "inputs": {"klein_pipeline": ["12", 0], "lora_path": lora_path, "lora_scale": lora_strength},
        }
        pipeline_source = ["13", 0]

    nodes.update(
        {
            "14": {
                "class_type": "VTONPhase2KleinSamplerDetailed",
                "inputs": {
                    "klein_pipeline": pipeline_source,
                    "person_canvas": ["9", 0],
                    "top_reference": ["10", 0],
                    "bottom_reference": ["9", 0],
                    "prompt": prompt,
                    "seed": seed,
                    "steps": steps,
                    "guidance_scale": guidance,
                    "width": int(pass_cfg.get("canvas_width", defaults.get("canvas_width", 768))),
                    "height": int(pass_cfg.get("canvas_height", defaults.get("canvas_height", 1024))),
                },
            },
            "15": {
                "class_type": "VTONPhase2MaskedPasteBack",
                "inputs": {
                    "original_image": ["1", 0],
                    "generated_crop": ["26", 0],
                    "cropped_mask": ["5", 1],
                    "bbox_json": ["5", 2],
                    "feather_px": int(pass_cfg.get("paste_feather_px", defaults.get("paste_feather_px", 12))),
                    "color_match": bool(pass_cfg.get("color_match", defaults.get("color_match", True))),
                    "preserve_outside_mask": bool(
                        pass_cfg.get("preserve_outside_mask", defaults.get("preserve_outside_mask", True))
                    ),
                    "debug": True,
                },
            },
            "16": {
                "class_type": "VTONPhase2TryOnDebugSheet",
                "inputs": {
                    "person_image": ["1", 0],
                    "garment_image": ["2", 0],
                    "raw_mask": ["3", 1],
                    "processed_mask": ["4", 0],
                    "mask_overlay": ["3", 2],
                    "crop_image": ["5", 0],
                    "crop_mask": ["5", 1],
                    "generated_crop": ["26", 0],
                    "final_image": ["15", 0],
                    "status_text": (
                        f"sample={pass_cfg.get('sample_id', '')} pass={pass_cfg.get('pass_index', 1)} "
                        f"region={target_region} seed={seed} steps={steps} guidance={guidance}"
                    ),
                },
            },
            "26": {
                "class_type": "VTONPhase2ExtractFittedCanvasRegion",
                "inputs": {
                    "generated_canvas": ["14", 0],
                    "fit_meta_json": ["9", 1],
                },
            },
            "17": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": raw}},
            "18": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": processed}},
            "19": {"class_type": "SaveImage", "inputs": {"images": ["3", 2], "filename_prefix": overlay}},
            "20": {"class_type": "SaveImage", "inputs": {"images": ["5", 4], "filename_prefix": bbox}},
            "21": {"class_type": "SaveImage", "inputs": {"images": ["5", 0], "filename_prefix": crop}},
            "22": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": crop_mask}},
            "23": {"class_type": "SaveImage", "inputs": {"images": ["26", 0], "filename_prefix": generated}},
            "24": {"class_type": "SaveImage", "inputs": {"images": ["15", 0], "filename_prefix": final}},
            "25": {"class_type": "SaveImage", "inputs": {"images": ["16", 0], "filename_prefix": debug}},
        }
    )
    return nodes


def image_bbox_metadata(person_path: Path, processed_mask_path: Path, pass_cfg: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    from virtual_tryon.masking import BBoxCropConfig, bbox_crop, mask_area_ratio

    person = Image.open(person_path).convert("RGB")
    mask = Image.open(processed_mask_path).convert("L")
    result = bbox_crop(
        person,
        mask,
        BBoxCropConfig(
            padding_ratio=float(pass_cfg.get("padding_ratio", defaults.get("padding_ratio", 0.22))),
            min_padding_px=int(pass_cfg.get("min_padding_px", defaults.get("min_padding_px", 32))),
            max_padding_px=int(pass_cfg.get("max_padding_px", defaults.get("max_padding_px", 160))),
            force_square=bool(pass_cfg.get("force_square", defaults.get("force_square", False))),
            target_multiple=int(pass_cfg.get("target_multiple", defaults.get("target_multiple", 16))),
            min_crop_size=int(pass_cfg.get("min_crop_size", defaults.get("min_crop_size", 256))),
            target_region=str(pass_cfg.get("target_region", defaults.get("target_region", "upper"))),
        ),
    )
    meta = json.loads(result.bbox_json or "{}")
    meta["processed_mask_area_ratio"] = round(mask_area_ratio(mask), 6)
    return meta


def run_pass(
    *,
    sample_id: str,
    pass_index: int,
    person_path: Path,
    pass_cfg: dict[str, Any],
    defaults: dict[str, Any],
    sample_dir: Path,
) -> Path:
    garment_path = resolve_path(pass_cfg["garment_image"])
    target_region = str(pass_cfg.get("target_region") or defaults.get("target_region") or "upper")
    pass_name = f"pass_{pass_index:02d}_{target_region}"
    pass_dir = sample_dir / pass_name
    pass_dir.mkdir(parents=True, exist_ok=True)

    input_person = pass_dir / "input_person.png"
    input_garment = pass_dir / "input_garment.png"
    shutil.copy2(person_path, input_person)
    shutil.copy2(garment_path, input_garment)

    person_name = copy_to_comfy_input(input_person, f"klmi_{sample_id}_{pass_name}_person.png")
    garment_name = copy_to_comfy_input(input_garment, f"klmi_{sample_id}_{pass_name}_garment.png")
    prefix = f"klein_local_masked/{sample_id}/{pass_name}"
    pass_cfg = pass_cfg | {"sample_id": sample_id, "pass_index": pass_index}
    prompt = build_prompt_graph(
        person_name=person_name,
        garment_name=garment_name,
        pass_cfg=pass_cfg,
        defaults=defaults,
        filename_prefix=prefix,
    )
    (pass_dir / "workflow_used_api.json").write_text(json.dumps(prompt, indent=2), encoding="utf-8")

    started = time.perf_counter()
    history = queue_and_wait(prompt, timeout_seconds=int(defaults.get("timeout_seconds", 3600)))
    (pass_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    status = history.get("status", {})
    if not status.get("completed"):
        raise RuntimeError(f"ComfyUI run failed for {sample_id}/{pass_name}: {json.dumps(status)}")

    expected = {
        "mask_raw": "mask_raw",
        "mask_processed": "mask_processed",
        "mask_overlay": "mask_overlay",
        "bbox_overlay": "bbox_overlay",
        "crop_image": "crop_image",
        "crop_mask": "crop_mask",
        "generated_crop": "generated_crop",
        "output_base": "output_base",
        "debug_sheet": "debug_sheet",
    }
    copied = copy_saved_by_prefix(saved_images(history), pass_dir, expected)
    copied = copy_saved_from_output_folder(prefix, pass_dir, expected, copied)
    for key in expected:
        cached_path = pass_dir / f"{key}.png"
        if key not in copied and cached_path.is_file():
            copied[key] = str(cached_path)
    output_path = Path(copied["output_base"]) if "output_base" in copied else Path()
    if not output_path.is_file():
        raise RuntimeError(f"No output_base saved for {sample_id}/{pass_name}")

    bbox_meta: dict[str, Any] = {}
    processed_path = Path(copied["mask_processed"]) if "mask_processed" in copied else Path()
    if "mask_processed" in copied and processed_path.exists():
        bbox_meta = image_bbox_metadata(input_person, processed_path, pass_cfg, defaults)

    metadata = {
        "sample_id": sample_id,
        "pass_index": pass_index,
        "target_region": target_region,
        "garment_type": pass_cfg.get("garment_type", target_region),
        "person_image": str(input_person),
        "garment_image": str(input_garment),
        "positive_prompt": pass_cfg.get("positive_prompt", ""),
        "negative_prompt": pass_cfg.get("negative_prompt", ""),
        "negative_prompt_usage": "recorded_only; VTONPhase2KleinSamplerDetailed has no negative-conditioning input",
        "seed": int(pass_cfg.get("seed", defaults.get("seed", 2026062601))),
        "steps": int(pass_cfg.get("steps", defaults.get("steps", 8))),
        "guidance": float(pass_cfg.get("guidance", defaults.get("guidance", 2.5))),
        "denoise": pass_cfg.get("denoise", defaults.get("denoise", None)),
        "sampler": "VTONPhase2KleinSamplerDetailed",
        "scheduler": "Klein pipeline internal scheduler",
        "redux_strength": pass_cfg.get("redux_strength", defaults.get("redux_strength", None)),
        "lora_strength": float(pass_cfg.get("lora_strength", defaults.get("lora_strength", 1.0))),
        "use_redux": False,
        "use_catviton_lora": bool(pass_cfg.get("use_catviton_lora", defaults.get("use_catviton_lora", True))),
        "mask_area_ratio": bbox_meta.get("processed_mask_area_ratio"),
        "bbox": bbox_meta,
        "model_file_names": {
            "base_model_dir": str(resolve_path(pass_cfg.get("model_dir") or defaults.get("model_dir") or DEFAULT_MODEL_DIR)),
            "lora_path": str(resolve_path(pass_cfg.get("lora_path") or defaults.get("lora_path") or DEFAULT_LORA_PATH)),
        },
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "artifacts": copied,
    }
    (pass_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"sample_id": sample_id, "pass": pass_name, "output": str(output_path)}), flush=True)
    return output_path


def run_sample(sample_cfg: dict[str, Any], defaults: dict[str, Any], output_root: Path) -> dict[str, Any]:
    sample_id = str(sample_cfg["sample_id"])
    sample_dir = output_root / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    person_path = resolve_path(sample_cfg["person_image"])
    workflow_json = resolve_path(sample_cfg.get("workflow_json", defaults.get("workflow_json", DEFAULT_WORKFLOW_JSON)))
    if workflow_json.exists():
        shutil.copy2(workflow_json, sample_dir / "workflow_used_ui.json")

    current_person = person_path
    pass_outputs = []
    for idx, pass_cfg in enumerate(sample_cfg.get("passes") or [], start=1):
        current_person = run_pass(
            sample_id=sample_id,
            pass_index=idx,
            person_path=current_person,
            pass_cfg=pass_cfg,
            defaults=defaults,
            sample_dir=sample_dir,
        )
        pass_outputs.append(str(current_person))

    final_path = sample_dir / "final_output.png"
    shutil.copy2(current_person, final_path)
    sample_metadata = {
        "sample_id": sample_id,
        "person_image": str(person_path),
        "pass_outputs": pass_outputs,
        "final_output": str(final_path),
    }
    (sample_dir / "metadata.json").write_text(json.dumps(sample_metadata, indent=2), encoding="utf-8")
    return sample_metadata


def check_models(config: dict[str, Any]) -> None:
    defaults = config.get("defaults") or {}
    errors: list[str] = []
    workflow_json = resolve_path(config.get("workflow_json") or defaults.get("workflow_json") or DEFAULT_WORKFLOW_JSON)
    if not workflow_json.exists():
        errors.append(f"Missing workflow JSON: {workflow_json}")

    model_paths: set[Path] = set()
    lora_paths: set[Path] = set()
    model_paths.add(resolve_path(defaults.get("model_dir") or DEFAULT_MODEL_DIR))
    if defaults.get("use_catviton_lora", True):
        lora_paths.add(resolve_path(defaults.get("lora_path") or DEFAULT_LORA_PATH))
    for sample in config.get("samples", []):
        person = resolve_path(sample["person_image"])
        if not person.exists():
            errors.append(f"Missing person image for {sample.get('sample_id')}: {person}")
        for pass_cfg in sample.get("passes") or []:
            garment = resolve_path(pass_cfg["garment_image"])
            if not garment.exists():
                errors.append(
                    f"Missing garment image for {sample.get('sample_id')} pass {pass_cfg.get('pass_index')}: {garment}"
                )
            model_paths.add(resolve_path(pass_cfg.get("model_dir") or defaults.get("model_dir") or DEFAULT_MODEL_DIR))
            if pass_cfg.get("use_catviton_lora", defaults.get("use_catviton_lora", True)):
                lora_paths.add(resolve_path(pass_cfg.get("lora_path") or defaults.get("lora_path") or DEFAULT_LORA_PATH))

    for model_dir in model_paths:
        if not model_dir.exists():
            errors.append(f"Missing FLUX.2 Klein 9B model directory: {model_dir}")
        elif model_dir.is_dir() and not (model_dir / "model_index.json").exists():
            errors.append(f"FLUX.2 Klein 9B directory lacks model_index.json: {model_dir}")
    for lora_path in lora_paths:
        if not lora_path.exists():
            errors.append(f"Missing Klein virtual try-on LoRA file: {lora_path}")

    if errors:
        raise FileNotFoundError("Model/input validation failed:\n- " + "\n- ".join(errors))
    print("check_models=ok", flush=True)


def write_report(rows: list[dict[str, Any]], output_root: Path) -> Path:
    lines = [
        "# Klein local masked try-on smoke report",
        "",
        "Pipeline: SCHP/SAM mask -> mask morphology -> bbox crop -> Klein 9B + LoRA local crop edit -> masked paste-back.",
        "",
        "| sample | passes | final output |",
        "|---|---:|---|",
    ]
    for row in rows:
        lines.append(f"| {row['sample_id']} | {len(row.get('pass_outputs', []))} | `{row['final_output']}` |")
    report = output_root / "ablation_report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="JSON config with samples and passes.")
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()

    config_path = resolve_path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    check_models(config)
    defaults = config.get("defaults") or {}
    output_root = resolve_path(args.output_root or config.get("output_root") or DEFAULT_OUTPUT_ROOT)
    output_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_root / "run_config.json")

    rows = [run_sample(sample_cfg, defaults, output_root) for sample_cfg in config.get("samples", [])]
    report = write_report(rows, output_root)
    print(f"report={report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
