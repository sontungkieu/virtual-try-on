from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from virtual_tryon.masking.human_parse_mask import HumanParseMaskConfig, HumanParseMasker
from virtual_tryon.masking.manual_mask_loader import load_manual_mask
from virtual_tryon.masking.mask_morphology import MaskPostprocessConfig, postprocess_mask
from virtual_tryon.masking.sam_mask import SAMMaskConfig, SAMMasker


TARGET_REGIONS = {"upper", "lower", "dress", "shoes", "hat", "accessory"}


@dataclass(frozen=True)
class ModelFiles:
    flux_fill: str = "unet/FLUX1/fluxFillFP8_v10.safetensors"
    flux_redux: str = "style_models/flux1-redux-dev.safetensors"
    sigclip: str = "clip_vision/sigclip_vision_patch14_384.safetensors"
    catviton_lora: str = "loras/flux/catvton-flux-lora.safetensors"
    vae: str = "vae/FLUX1/ae.safetensors"
    clip_l: str = "clip/clip_l.safetensors"
    t5xxl: str = "clip/t5xxl_fp8_e4m3fn.safetensors"

    @classmethod
    def from_dict(cls, values: dict[str, Any] | None) -> "ModelFiles":
        if not values:
            return cls()
        defaults = cls().__dict__.copy()
        defaults.update(values)
        return cls(**defaults)


@dataclass(frozen=True)
class ComfyConfig:
    url: str = "http://127.0.0.1:8188"
    input_dir: Path = Path("/workspace/ComfyUI/input")
    output_dir: Path = Path("/workspace/ComfyUI/output")
    models_dir: Path = Path("/workspace/ComfyUI/models")

    @classmethod
    def from_dict(cls, values: dict[str, Any] | None) -> "ComfyConfig":
        if not values:
            return cls()
        return cls(
            url=str(values.get("url", cls.url)),
            input_dir=Path(values.get("input_dir", cls.input_dir)),
            output_dir=Path(values.get("output_dir", cls.output_dir)),
            models_dir=Path(values.get("models_dir", cls.models_dir)),
        )


@dataclass(frozen=True)
class RefineConfig:
    enabled: bool = False
    steps: int = 12
    denoise: float = 0.25
    dilation_px: int = 10
    erosion_px: int = 8
    feather_px: int = 5

    @classmethod
    def from_dict(cls, values: dict[str, Any] | None) -> "RefineConfig":
        if not values:
            return cls()
        defaults = cls().__dict__.copy()
        defaults.update(values)
        return cls(**defaults)


@dataclass(frozen=True)
class TryOnPass:
    target_region: str
    garment_image: Path
    mask_image: Path | None
    positive_prompt: str
    negative_prompt: str
    seed: int
    steps: int
    guidance: float
    denoise: float
    pass_index: int
    garment_type: str
    sampler: str = "euler"
    scheduler: str = "normal"
    redux_strength: float = 0.75
    lora_strength: float = 1.0
    use_redux: bool = True
    use_catviton_lora: bool = True
    semantic_map_path: Path | None = None
    sam_box_xyxy: tuple[int, int, int, int] | None = None
    allow_style_canvas: bool = False


def resolve_path(value: str | Path | None, base_dir: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def post_json(comfy: ComfyConfig, path: str, payload: dict[str, Any], timeout: int = 30) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{comfy.url}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def get_json(comfy: ComfyConfig, path: str, timeout: int = 30) -> dict[str, Any]:
    with urllib.request.urlopen(f"{comfy.url}{path}", timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def check_models(
    comfy: ComfyConfig,
    models: ModelFiles,
    *,
    use_redux: bool = True,
    use_catviton_lora: bool = True,
) -> dict[str, str]:
    required = {
        "FLUX Fill diffusion model": models.flux_fill,
        "VAE": models.vae,
        "CLIP-L text encoder": models.clip_l,
        "T5 text encoder": models.t5xxl,
    }
    if use_redux:
        required["FLUX Redux style model"] = models.flux_redux
        required["SigCLIP / CLIPVision model"] = models.sigclip
    if use_catviton_lora:
        required["CatVTON / CatVitOn LoRA"] = models.catviton_lora

    missing: list[str] = []
    resolved: dict[str, str] = {}
    for label, rel_path in required.items():
        path = comfy.models_dir / rel_path
        resolved[label] = path.as_posix()
        if not path.exists():
            missing.append(f"- {label}: expected at {path.as_posix()}")
    if missing:
        raise FileNotFoundError(
            "Required ComfyUI model files are missing or placed in the wrong folder:\n"
            + "\n".join(missing)
            + "\nMove/download the files into the listed ComfyUI models directories before running."
        )
    return resolved


def copy_to_comfy_input(comfy: ComfyConfig, path: Path, name: str) -> str:
    comfy.input_dir.mkdir(parents=True, exist_ok=True)
    target = comfy.input_dir / name
    shutil.copy2(path, target)
    return name


def queue_and_wait(comfy: ComfyConfig, prompt: dict[str, Any], timeout_seconds: int) -> dict[str, Any]:
    response = post_json(comfy, "/prompt", {"prompt": prompt})
    prompt_id = response["prompt_id"]
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        time.sleep(3)
        history = get_json(comfy, f"/history/{prompt_id}")
        if prompt_id not in history:
            continue
        item = history[prompt_id]
        status = item.get("status", {})
        if status.get("completed") or status.get("status_str") == "error":
            return item
    raise TimeoutError(f"ComfyUI prompt timed out: {prompt_id}")


def extract_saved_image(comfy: ComfyConfig, history_item: dict[str, Any]) -> Path:
    outputs = history_item.get("outputs", {})
    for output in outputs.values():
        images = output.get("images")
        if not images:
            continue
        image = images[0]
        subfolder = image.get("subfolder") or ""
        return comfy.output_dir / subfolder / image["filename"]
    raise RuntimeError(f"No saved image in ComfyUI history: {json.dumps(history_item)[:1000]}")


def rectangle_debug_mask(person_image: Image.Image, target_region: str) -> Image.Image:
    width, height = person_image.size
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    if target_region == "upper":
        box = (int(width * 0.20), int(height * 0.18), int(width * 0.80), int(height * 0.58))
    elif target_region == "lower":
        box = (int(width * 0.20), int(height * 0.48), int(width * 0.80), int(height * 0.96))
    elif target_region == "dress":
        box = (int(width * 0.20), int(height * 0.18), int(width * 0.80), int(height * 0.94))
    elif target_region == "shoes":
        box = (int(width * 0.20), int(height * 0.82), int(width * 0.80), height)
    elif target_region == "hat":
        box = (int(width * 0.25), 0, int(width * 0.75), int(height * 0.18))
    else:
        box = (int(width * 0.10), int(height * 0.10), int(width * 0.90), int(height * 0.90))
    draw.rounded_rectangle(box, radius=max(8, width // 40), fill=255)
    return mask


def resolve_raw_mask(
    person_image: Image.Image,
    pass_cfg: TryOnPass,
    *,
    allow_rectangle_fallback: bool,
    sam_config: SAMMaskConfig | None,
) -> tuple[Image.Image, str, list[str]]:
    warnings: list[str] = []
    if pass_cfg.mask_image is not None:
        return load_manual_mask(pass_cfg.mask_image, person_image.size), "manual_mask", warnings
    if pass_cfg.semantic_map_path is not None:
        mask = HumanParseMasker(HumanParseMaskConfig(pass_cfg.semantic_map_path)).create_mask(
            person_image, pass_cfg.target_region, pass_cfg.semantic_map_path
        )
        if mask is not None:
            return mask, "human_parse", warnings
    if pass_cfg.sam_box_xyxy is not None:
        masker = SAMMasker(sam_config)
        mask = masker.create_mask_from_box(person_image, pass_cfg.sam_box_xyxy)
        if mask is not None:
            return mask, "sam", warnings
        warnings.append(masker.unavailable_reason or "SAM mask requested but unavailable.")
    if allow_rectangle_fallback:
        warnings.append("rectangle fallback mask used; this is debug-only and not production-grade.")
        return rectangle_debug_mask(person_image, pass_cfg.target_region), "rectangle_fallback", warnings
    raise ValueError(
        f"Pass {pass_cfg.pass_index} has no usable mask_image, semantic_map_path, or SAM box. "
        "Rectangle masks are disabled by default; pass --allow-rectangle-fallback only for debug."
    )


def make_refine_mask(processed_mask: Image.Image, config: RefineConfig) -> Image.Image:
    hard = processed_mask.convert("L")
    outer = hard.filter(ImageFilter.MaxFilter(config.dilation_px * 2 + 1))
    inner = hard.filter(ImageFilter.MinFilter(config.erosion_px * 2 + 1))
    edge = ImageChops.subtract(outer, inner)
    return edge.filter(ImageFilter.GaussianBlur(config.feather_px))


def is_api_prompt_json(workflow: dict[str, Any]) -> bool:
    return bool(workflow) and all(
        isinstance(value, dict) and "class_type" in value and "inputs" in value
        for value in workflow.values()
    )


def _patch_api_node_input(prompt: dict[str, Any], node_id: str, input_name: str, value: Any) -> None:
    if node_id not in prompt:
        raise KeyError(f"workflow_patch references missing node_id={node_id}")
    prompt[node_id].setdefault("inputs", {})[input_name] = value


def patch_api_workflow(
    workflow: dict[str, Any],
    *,
    person_name: str,
    garment_name: str,
    mask_name: str,
    pass_cfg: TryOnPass,
    filename_prefix: str,
    patch_map: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prompt = copy.deepcopy(workflow)
    if patch_map:
        values = {
            "person_image": person_name,
            "garment_image": garment_name,
            "mask_image": mask_name,
            "positive_prompt": pass_cfg.positive_prompt,
            "negative_prompt": pass_cfg.negative_prompt,
            "seed": pass_cfg.seed,
            "steps": pass_cfg.steps,
            "guidance": pass_cfg.guidance,
            "denoise": pass_cfg.denoise,
            "output_path": filename_prefix,
        }
        for key, value in values.items():
            target = patch_map.get(key)
            if not target:
                continue
            _patch_api_node_input(prompt, str(target["node_id"]), str(target["input"]), value)
        return prompt

    load_nodes = sorted(
        (node_id for node_id, node in prompt.items() if node.get("class_type") == "LoadImage"),
        key=lambda item: int(item) if item.isdigit() else item,
    )
    if len(load_nodes) < 3:
        raise ValueError("Workflow JSON mode needs at least three LoadImage nodes for person, garment, and mask.")
    prompt[load_nodes[0]]["inputs"]["image"] = person_name
    prompt[load_nodes[1]]["inputs"]["image"] = garment_name
    prompt[load_nodes[2]]["inputs"]["image"] = mask_name

    text_nodes = sorted(
        (node_id for node_id, node in prompt.items() if node.get("class_type") == "CLIPTextEncode"),
        key=lambda item: int(item) if item.isdigit() else item,
    )
    if text_nodes:
        prompt[text_nodes[0]]["inputs"]["text"] = pass_cfg.positive_prompt
    if len(text_nodes) > 1:
        prompt[text_nodes[1]]["inputs"]["text"] = pass_cfg.negative_prompt

    for node in prompt.values():
        class_type = node.get("class_type")
        inputs = node.setdefault("inputs", {})
        if class_type == "KSampler":
            inputs["seed"] = pass_cfg.seed
            inputs["steps"] = pass_cfg.steps
            inputs["cfg"] = 1
            inputs["sampler_name"] = pass_cfg.sampler
            inputs["scheduler"] = pass_cfg.scheduler
            inputs["denoise"] = pass_cfg.denoise
        elif class_type == "FluxGuidance":
            inputs["guidance"] = pass_cfg.guidance
        elif class_type == "StyleModelApply":
            inputs["strength"] = pass_cfg.redux_strength
        elif class_type == "LoraLoaderModelOnly":
            inputs["strength_model"] = pass_cfg.lora_strength
        elif class_type == "SaveImage":
            inputs["filename_prefix"] = filename_prefix
    return prompt


def patch_ui_workflow_json(
    workflow: dict[str, Any],
    *,
    person_name: str,
    garment_name: str,
    mask_name: str,
    pass_cfg: TryOnPass,
    filename_prefix: str,
) -> dict[str, Any]:
    patched = copy.deepcopy(workflow)
    nodes = sorted(patched.get("nodes", []), key=lambda node: node.get("id", 0))
    load_nodes = [node for node in nodes if node.get("type") == "LoadImage"]
    for node, value in zip(load_nodes, [person_name, garment_name, mask_name], strict=False):
        widgets = node.setdefault("widgets_values", [])
        if widgets:
            widgets[0] = value
    for node in nodes:
        node_type = node.get("type")
        widgets = node.setdefault("widgets_values", [])
        if node_type == "KSampler" and len(widgets) >= 7:
            widgets[0] = pass_cfg.seed
            widgets[2] = pass_cfg.steps
            widgets[3] = 1
            widgets[4] = pass_cfg.sampler
            widgets[5] = pass_cfg.scheduler
            widgets[6] = pass_cfg.denoise
        elif node_type == "CLIPTextEncode" and widgets:
            text = str(widgets[0]).lower()
            widgets[0] = pass_cfg.negative_prompt if "negative" in text or "bad anatomy" in text else pass_cfg.positive_prompt
        elif node_type == "FluxGuidance" and widgets:
            widgets[0] = pass_cfg.guidance
        elif node_type == "StyleModelApply" and widgets:
            widgets[0] = pass_cfg.redux_strength
        elif node_type == "SaveImage" and widgets:
            widgets[0] = filename_prefix
    return patched


def build_api_graph_fallback(
    *,
    person_name: str,
    garment_name: str,
    mask_name: str,
    pass_cfg: TryOnPass,
    models: ModelFiles,
    filename_prefix: str,
) -> dict[str, Any]:
    prompt: dict[str, Any] = {
        "1": {"class_type": "LoadImage", "inputs": {"image": person_name}},
        "2": {"class_type": "LoadImage", "inputs": {"image": mask_name}},
        "3": {"class_type": "ImageToMask", "inputs": {"image": ["2", 0], "channel": "red"}},
        "4": {"class_type": "LoadImage", "inputs": {"image": garment_name}},
        "5": {
            "class_type": "DualCLIPLoader",
            "inputs": {
                "clip_name1": Path(models.clip_l).name,
                "clip_name2": Path(models.t5xxl).name,
                "type": "flux",
            },
        },
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["5", 0], "text": pass_cfg.positive_prompt}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["5", 0], "text": pass_cfg.negative_prompt}},
        "8": {"class_type": "FluxGuidance", "inputs": {"conditioning": ["6", 0], "guidance": pass_cfg.guidance}},
        "13": {"class_type": "VAELoader", "inputs": {"vae_name": "/".join(Path(models.vae).parts[1:])}},
        "14": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": "/".join(Path(models.flux_fill).parts[1:]), "weight_dtype": "default"},
        },
    }
    positive_node = "8"
    if pass_cfg.use_redux:
        prompt.update(
            {
                "9": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": Path(models.sigclip).name}},
                "10": {
                    "class_type": "CLIPVisionEncode",
                    "inputs": {"clip_vision": ["9", 0], "image": ["4", 0], "crop": "none"},
                },
                "11": {"class_type": "StyleModelLoader", "inputs": {"style_model_name": Path(models.flux_redux).name}},
                "12": {
                    "class_type": "StyleModelApply",
                    "inputs": {
                        "conditioning": ["8", 0],
                        "style_model": ["11", 0],
                        "clip_vision_output": ["10", 0],
                        "strength": pass_cfg.redux_strength,
                        "strength_type": "multiply",
                    },
                },
            }
        )
        positive_node = "12"

    model_node = "14"
    if pass_cfg.use_catviton_lora:
        prompt["15"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["14", 0],
                "lora_name": "/".join(Path(models.catviton_lora).parts[1:]),
                "strength_model": pass_cfg.lora_strength,
            },
        }
        model_node = "15"

    prompt.update(
        {
            "16": {
                "class_type": "InpaintModelConditioning",
                "inputs": {
                    "positive": [positive_node, 0],
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
                    "model": [model_node, 0],
                    "seed": pass_cfg.seed,
                    "steps": pass_cfg.steps,
                    "cfg": 1,
                    "sampler_name": pass_cfg.sampler,
                    "scheduler": pass_cfg.scheduler,
                    "positive": ["16", 0],
                    "negative": ["16", 1],
                    "latent_image": ["16", 2],
                    "denoise": pass_cfg.denoise,
                },
            },
            "18": {"class_type": "VAEDecode", "inputs": {"samples": ["17", 0], "vae": ["13", 0]}},
            "19": {"class_type": "SaveImage", "inputs": {"images": ["18", 0], "filename_prefix": filename_prefix}},
        }
    )
    return prompt


def build_prompt_for_pass(
    *,
    workflow_json: Path | None,
    workflow_patch: dict[str, Any] | None,
    person_name: str,
    garment_name: str,
    mask_name: str,
    pass_cfg: TryOnPass,
    models: ModelFiles,
    filename_prefix: str,
    workflow_used_path: Path,
) -> dict[str, Any]:
    if workflow_json is None:
        prompt = build_api_graph_fallback(
            person_name=person_name,
            garment_name=garment_name,
            mask_name=mask_name,
            pass_cfg=pass_cfg,
            models=models,
            filename_prefix=filename_prefix,
        )
        workflow_used_path.write_text(json.dumps(prompt, indent=2), encoding="utf-8")
        return prompt

    workflow = read_json(workflow_json)
    if is_api_prompt_json(workflow):
        prompt = patch_api_workflow(
            workflow,
            person_name=person_name,
            garment_name=garment_name,
            mask_name=mask_name,
            pass_cfg=pass_cfg,
            filename_prefix=filename_prefix,
            patch_map=workflow_patch,
        )
        workflow_used_path.write_text(json.dumps(prompt, indent=2), encoding="utf-8")
        return prompt

    patched_ui = patch_ui_workflow_json(
        workflow,
        person_name=person_name,
        garment_name=garment_name,
        mask_name=mask_name,
        pass_cfg=pass_cfg,
        filename_prefix=filename_prefix,
    )
    workflow_used_path.write_text(json.dumps(patched_ui, indent=2), encoding="utf-8")
    raise ValueError(
        f"{workflow_json} is a UI workflow JSON. It was patched and saved to {workflow_used_path}, "
        "but ComfyUI /prompt execution requires an API workflow JSON. Export the exact video workflow "
        "with 'Save (API Format)' or provide a workflow_patch for an API graph."
    )


def parse_passes(config: dict[str, Any], config_dir: Path) -> list[TryOnPass]:
    defaults = config.get("defaults", {})
    passes: list[TryOnPass] = []
    for index, raw_pass in enumerate(config.get("passes", []), start=1):
        merged = {**defaults, **raw_pass}
        if "garment_images" in merged or isinstance(merged.get("garment_image"), list):
            raise ValueError(
                f"Pass {index} violates the single-garment contract. "
                "Use exactly one garment_image per pass and represent outfits as sequential passes."
            )
        target_region = str(merged["target_region"])
        if target_region not in TARGET_REGIONS:
            raise ValueError(f"Pass {index} has unsupported target_region='{target_region}'")
        garment_image = resolve_path(merged.get("garment_image"), config_dir)
        if garment_image is None:
            raise ValueError(f"Pass {index} is missing garment_image")
        if not garment_image.exists():
            raise FileNotFoundError(f"Pass {index} garment_image does not exist: {garment_image}")
        positive = merged.get("positive_prompt")
        negative = merged.get("negative_prompt")
        if not positive or not negative:
            raise ValueError(f"Pass {index} must provide positive_prompt and negative_prompt")
        passes.append(
            TryOnPass(
                target_region=target_region,
                garment_image=garment_image,
                mask_image=resolve_path(merged.get("mask_image"), config_dir),
                positive_prompt=str(positive),
                negative_prompt=str(negative),
                seed=int(merged["seed"]),
                steps=int(merged["steps"]),
                guidance=float(merged["guidance"]),
                denoise=float(merged["denoise"]),
                pass_index=index,
                garment_type=str(merged.get("garment_type", target_region)),
                sampler=str(merged.get("sampler", "euler")),
                scheduler=str(merged.get("scheduler", "normal")),
                redux_strength=float(merged.get("redux_strength", 0.75)),
                lora_strength=float(merged.get("lora_strength", 1.0)),
                use_redux=bool(merged.get("use_redux", True)),
                use_catviton_lora=bool(merged.get("use_catviton_lora", True)),
                semantic_map_path=resolve_path(merged.get("semantic_map_path"), config_dir),
                sam_box_xyxy=tuple(merged["sam_box_xyxy"]) if merged.get("sam_box_xyxy") else None,
                allow_style_canvas=bool(merged.get("allow_style_canvas", False)),
            )
        )
    if not passes:
        raise ValueError("Run config must contain at least one pass")
    return passes


def detect_single_pass_reference_issues(pass_cfg: TryOnPass) -> list[str]:
    warnings: list[str] = []
    lower_name = pass_cfg.garment_image.name.lower()
    if not pass_cfg.allow_style_canvas and ("reference_canvas" in lower_name or "canvas" in lower_name):
        warnings.append("multi-item/style canvas-like reference used in single-garment pass")
    return warnings


def run_one_pass(
    *,
    comfy: ComfyConfig,
    models: ModelFiles,
    pass_cfg: TryOnPass,
    person_image_path: Path,
    pass_dir: Path,
    run_name: str,
    timeout_seconds: int,
    allow_rectangle_fallback: bool,
    mask_config: MaskPostprocessConfig,
    refine_config: RefineConfig,
    workflow_json: Path | None,
    workflow_patch: dict[str, Any] | None,
    sam_config: SAMMaskConfig | None,
    dry_run: bool,
) -> Path:
    pass_dir.mkdir(parents=True, exist_ok=True)
    person = Image.open(person_image_path).convert("RGB")
    garment = Image.open(pass_cfg.garment_image).convert("RGB")

    shutil.copy2(person_image_path, pass_dir / "input_person.png")
    shutil.copy2(pass_cfg.garment_image, pass_dir / "input_garment.png")

    raw_mask, mask_source, mask_warnings = resolve_raw_mask(
        person,
        pass_cfg,
        allow_rectangle_fallback=allow_rectangle_fallback,
        sam_config=sam_config,
    )
    artifacts = postprocess_mask(person, raw_mask, pass_cfg.target_region, mask_config)
    artifacts.raw_mask.save(pass_dir / "mask_raw.png")
    artifacts.processed_mask.save(pass_dir / "mask_processed.png")
    artifacts.overlay.save(pass_dir / "mask_overlay.png")

    warnings = [
        *mask_warnings,
        *artifacts.warnings,
        *detect_single_pass_reference_issues(pass_cfg),
    ]

    person_name = copy_to_comfy_input(comfy, pass_dir / "input_person.png", f"{run_name}_p{pass_cfg.pass_index:02d}_person.png")
    garment_name = copy_to_comfy_input(comfy, pass_dir / "input_garment.png", f"{run_name}_p{pass_cfg.pass_index:02d}_garment.png")
    mask_name = copy_to_comfy_input(comfy, pass_dir / "mask_processed.png", f"{run_name}_p{pass_cfg.pass_index:02d}_mask.png")
    prefix = f"{run_name}/pass_{pass_cfg.pass_index:02d}_{pass_cfg.target_region}/base"

    prompt = build_prompt_for_pass(
        workflow_json=workflow_json,
        workflow_patch=workflow_patch,
        person_name=person_name,
        garment_name=garment_name,
        mask_name=mask_name,
        pass_cfg=pass_cfg,
        models=models,
        filename_prefix=prefix,
        workflow_used_path=pass_dir / "workflow_used.json",
    )

    model_file_names = {
        "flux_fill": models.flux_fill,
        "flux_redux": models.flux_redux if pass_cfg.use_redux else None,
        "sigclip": models.sigclip if pass_cfg.use_redux else None,
        "catviton_lora": models.catviton_lora if pass_cfg.use_catviton_lora else None,
        "vae": models.vae,
        "clip_l": models.clip_l,
        "t5xxl": models.t5xxl,
    }
    metadata = {
        "pass_index": pass_cfg.pass_index,
        "target_region": pass_cfg.target_region,
        "garment_type": pass_cfg.garment_type,
        "seed": pass_cfg.seed,
        "steps": pass_cfg.steps,
        "guidance": pass_cfg.guidance,
        "denoise": pass_cfg.denoise,
        "sampler": pass_cfg.sampler,
        "scheduler": pass_cfg.scheduler,
        "redux_strength": pass_cfg.redux_strength,
        "lora_strength": pass_cfg.lora_strength,
        "use_redux": pass_cfg.use_redux,
        "use_catviton_lora": pass_cfg.use_catviton_lora,
        "mask_source": mask_source,
        "mask_area_ratio": artifacts.mask_area_ratio,
        "model_file_names": model_file_names,
        "warnings": warnings,
        "input_person": "input_person.png",
        "input_garment": "input_garment.png",
        "input_person_source": person_image_path.as_posix(),
        "input_garment_source": pass_cfg.garment_image.as_posix(),
        "mask_raw": "mask_raw.png",
        "mask_processed": "mask_processed.png",
        "mask_overlay": "mask_overlay.png",
    }

    if dry_run:
        metadata["status"] = "dry_run"
        (pass_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return pass_dir / "input_person.png"

    started = time.perf_counter()
    history = queue_and_wait(comfy, prompt, timeout_seconds)
    (pass_dir / "history_base.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    if not history.get("status", {}).get("completed"):
        metadata["status"] = "failed"
        metadata["error"] = history.get("status", {})
        (pass_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        raise RuntimeError(f"ComfyUI base pass failed: {json.dumps(history.get('status', {}), ensure_ascii=False)}")
    saved = extract_saved_image(comfy, history)
    output_base = pass_dir / "output_base.png"
    shutil.copy2(saved, output_base)
    output_final = output_base

    if refine_config.enabled:
        refine_mask = make_refine_mask(artifacts.processed_mask, refine_config)
        refine_mask.save(pass_dir / "mask_refine.png")
        refine_person = output_base
        refine_pass = TryOnPass(
            **{
                **pass_cfg.__dict__,
                "steps": refine_config.steps,
                "denoise": refine_config.denoise,
            }
        )
        refine_person_name = copy_to_comfy_input(comfy, refine_person, f"{run_name}_p{pass_cfg.pass_index:02d}_refine_person.png")
        refine_mask_name = copy_to_comfy_input(comfy, pass_dir / "mask_refine.png", f"{run_name}_p{pass_cfg.pass_index:02d}_refine_mask.png")
        refine_prefix = f"{run_name}/pass_{pass_cfg.pass_index:02d}_{pass_cfg.target_region}/refined"
        refine_prompt = build_api_graph_fallback(
            person_name=refine_person_name,
            garment_name=garment_name,
            mask_name=refine_mask_name,
            pass_cfg=refine_pass,
            models=models,
            filename_prefix=refine_prefix,
        )
        (pass_dir / "workflow_refine_used.json").write_text(json.dumps(refine_prompt, indent=2), encoding="utf-8")
        refine_history = queue_and_wait(comfy, refine_prompt, timeout_seconds)
        (pass_dir / "history_refine.json").write_text(json.dumps(refine_history, indent=2), encoding="utf-8")
        if refine_history.get("status", {}).get("completed"):
            output_refined = pass_dir / "output_refined.png"
            shutil.copy2(extract_saved_image(comfy, refine_history), output_refined)
            output_final = output_refined
        else:
            warnings.append("refine pass failed; using output_base.png")

    metadata["status"] = "completed"
    metadata["runtime_seconds"] = round(time.perf_counter() - started, 3)
    metadata["output_base"] = "output_base.png"
    metadata["output_refined"] = "output_refined.png" if (pass_dir / "output_refined.png").exists() else None
    metadata["output_final"] = output_final.name
    metadata["warnings"] = warnings
    (pass_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_final


def run_from_config(config_path: Path, args: argparse.Namespace) -> int:
    config = read_json(config_path)
    config_dir = config_path.parent
    run_name = str(config.get("run_name") or f"production_vton_{int(time.time())}")
    comfy = ComfyConfig.from_dict(config.get("comfy"))
    models = ModelFiles.from_dict(config.get("models"))
    passes = parse_passes(config, config_dir)
    output_root = resolve_path(args.output_root or config.get("output_root") or f"virtual_tryon/data/outputs/{run_name}", Path.cwd())
    assert output_root is not None
    output_root.mkdir(parents=True, exist_ok=True)

    use_redux = any(p.use_redux for p in passes)
    use_lora = any(p.use_catviton_lora for p in passes)
    resolved_models = check_models(comfy, models, use_redux=use_redux, use_catviton_lora=use_lora)
    if args.check_models_only:
        print(json.dumps({"status": "ok", "models": resolved_models}, indent=2))
        return 0

    workflow_json = resolve_path(config.get("workflow_json"), config_dir)
    workflow_patch = config.get("workflow_patch")
    refine = RefineConfig.from_dict(config.get("refine"))
    mask_defaults = config.get("mask_postprocess", {})
    mask_config = MaskPostprocessConfig(
        dilation_px=int(mask_defaults.get("dilation_px", 10)),
        erosion_px=int(mask_defaults.get("erosion_px", 0)),
        feather_px=int(mask_defaults.get("feather_px", 6)),
        threshold=int(mask_defaults.get("threshold", 8)),
        remove_face_hair=bool(mask_defaults.get("remove_face_hair", True)),
        remove_hands=bool(mask_defaults.get("remove_hands", True)),
    )
    sam_values = config.get("sam") or {}
    sam_config = SAMMaskConfig(
        checkpoint_path=resolve_path(sam_values.get("checkpoint_path"), config_dir),
        model_type=str(sam_values.get("model_type", "vit_h")),
        device=str(sam_values.get("device", "cuda")),
    )

    person_image = resolve_path(config.get("person_image"), config_dir)
    if person_image is None or not person_image.exists():
        raise FileNotFoundError(f"person_image does not exist: {person_image}")

    summary: list[dict[str, Any]] = []
    current_person = person_image
    for pass_cfg in passes:
        pass_dir = output_root / f"pass_{pass_cfg.pass_index:02d}_{pass_cfg.target_region}"
        final_output = run_one_pass(
            comfy=comfy,
            models=models,
            pass_cfg=pass_cfg,
            person_image_path=current_person,
            pass_dir=pass_dir,
            run_name=run_name,
            timeout_seconds=int(config.get("timeout_seconds", 2400)),
            allow_rectangle_fallback=bool(args.allow_rectangle_fallback or config.get("allow_rectangle_fallback", False)),
            mask_config=mask_config,
            refine_config=refine,
            workflow_json=workflow_json,
            workflow_patch=workflow_patch,
            sam_config=sam_config,
            dry_run=args.dry_run,
        )
        current_person = final_output
        metadata = read_json(pass_dir / "metadata.json")
        summary.append(metadata | {"pass_dir": pass_dir.as_posix()})
        print(json.dumps(summary[-1]), flush=True)

    shutil.copy2(current_person, output_root / "final_output.png")
    (output_root / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Production VTON run config JSON.")
    parser.add_argument("--output-root", help="Override output_root from config.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs, create masks/workflows, but do not queue ComfyUI.")
    parser.add_argument("--check-models-only", action="store_true")
    parser.add_argument("--allow-rectangle-fallback", action="store_true", help="Debug only; disabled by default.")
    args = parser.parse_args()
    return run_from_config(Path(args.config).resolve(), args)


if __name__ == "__main__":
    raise SystemExit(main())
