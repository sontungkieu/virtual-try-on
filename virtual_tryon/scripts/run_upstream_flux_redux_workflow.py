from __future__ import annotations

import argparse
import json
import shutil
import time
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


PROJECT_ROOT = Path("/workspace/Project_Phase2")
COMFY_INPUT = Path("/workspace/ComfyUI/input")
COMFY_OUTPUT = Path("/workspace/ComfyUI/output")
COMFY_URL = "http://127.0.0.1:8188"
EVAL_ROOT = PROJECT_ROOT / "virtual_tryon/data/temp/catvton_flux_eval_set"
UPSTREAM_WORKFLOW = PROJECT_ROOT / "tmp/comfyuiworkflows/virtual-tryon-flux-redux-workflow.json"
OUTPUT_ROOT = PROJECT_ROOT / "virtual_tryon/data/outputs/upstream_flux_redux_workflow_20260625"


NEGATIVE_PROMPT = (
    "bad anatomy, deformed body, distorted face, extra limbs, damaged hands, wrong garment, "
    "blurry, low quality, changed background, changed identity"
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
    COMFY_INPUT.mkdir(parents=True, exist_ok=True)
    target = COMFY_INPUT / name
    shutil.copy2(path, target)
    return name


def queue_and_wait(prompt: dict[str, Any], timeout_seconds: int = 2400) -> dict[str, Any]:
    response = post_json("/prompt", {"prompt": prompt})
    prompt_id = response["prompt_id"]
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        time.sleep(3)
        history = get_json(f"/history/{prompt_id}")
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


def sample_reference(sample_dir: Path) -> Path:
    canvas = sample_dir / "reference_canvas.png"
    if canvas.exists():
        return canvas
    for name in [
        "garment_top.png",
        "garment_bottom.png",
        "garment_dress.png",
        "accessory_hat.png",
        "accessory_shoes.png",
    ]:
        path = sample_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No reference garment/canvas found in {sample_dir}")


def tryon_prompt(sample_id: str) -> str:
    return (
        "Virtual try-on photo. The person wears the reference garment or outfit. "
        "Preserve identity, face, hair, hands, body shape, pose, lighting, and background. "
        f"Sample {sample_id}."
    )


def build_upstream_equivalent_prompt(
    *,
    person_name: str,
    reference_name: str,
    mask_name: str,
    sample_id: str,
    prompt_mode: str,
    seed: int,
    filename_prefix: str,
) -> dict[str, Any]:
    # Equivalent API graph for fahdmirza/comfyuiworkflows virtual-tryon-flux-redux-workflow.json.
    # Missing local classes are mapped as:
    # UnetLoaderGGUF -> UNETLoader, DualCLIPLoaderGGUF -> DualCLIPLoader,
    # StyleModelApplySimple("medium") -> StyleModelApply(strength=1.0),
    # INPAINT_ExpandMask -> GrowMask(expand=16), ImageCrop+ -> ImageCrop.
    positive_text = "" if prompt_mode == "repo_empty_prompt" else tryon_prompt(sample_id)
    negative_text = "" if prompt_mode == "repo_empty_prompt" else NEGATIVE_PROMPT
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": reference_name}},
        "2": {"class_type": "LoadImage", "inputs": {"image": person_name}},
        "3": {"class_type": "LoadImageMask", "inputs": {"image": mask_name, "channel": "red"}},
        "4": {"class_type": "GrowMask", "inputs": {"mask": ["3", 0], "expand": 16, "tapered_corners": True}},
        "5": {
            "class_type": "AddMaskForICLora",
            "inputs": {
                "first_image": ["1", 0],
                "second_image": ["2", 0],
                "second_mask": ["4", 0],
                "patch_mode": "auto",
                "output_length": 1536,
                "patch_color": "#FF0000",
            },
        },
        "6": {
            "class_type": "DualCLIPLoader",
            "inputs": {
                "clip_name1": "clip_l.safetensors",
                "clip_name2": "t5xxl_fp8_e4m3fn.safetensors",
                "type": "flux",
            },
        },
        "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["6", 0], "text": positive_text}},
        "8": {"class_type": "FluxGuidance", "inputs": {"conditioning": ["7", 0], "guidance": 3.5}},
        "9": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["8", 0]}},
        "10": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": "sigclip_vision_patch14_384.safetensors"}},
        "11": {"class_type": "CLIPVisionEncode", "inputs": {"clip_vision": ["10", 0], "image": ["1", 0], "crop": "center"}},
        "12": {"class_type": "StyleModelLoader", "inputs": {"style_model_name": "flux1-redux-dev.safetensors"}},
        "13": {
            "class_type": "StyleModelApply",
            "inputs": {
                "conditioning": ["8", 0],
                "style_model": ["12", 0],
                "clip_vision_output": ["11", 0],
                "strength": 1.0,
                "strength_type": "multiply",
            },
        },
        "14": {"class_type": "VAELoader", "inputs": {"vae_name": "FLUX1/ae.safetensors"}},
        "15": {
            "class_type": "InpaintModelConditioning",
            "inputs": {
                "positive": ["13", 0],
                "negative": ["9", 0],
                "vae": ["14", 0],
                "pixels": ["5", 0],
                "mask": ["5", 1],
                "noise_mask": True,
            },
        },
        "16": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": "FLUX1/fluxFillFP8_v10.safetensors", "weight_dtype": "default"},
        },
        "17": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["16", 0],
                "lora_name": "flux/catvton-flux-lora.safetensors",
                "strength_model": 1.0,
            },
        },
        "18": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["17", 0],
                "seed": seed,
                "steps": 20,
                "cfg": 8,
                "sampler_name": "euler",
                "scheduler": "normal",
                "positive": ["15", 0],
                "negative": ["15", 1],
                "latent_image": ["15", 2],
                "denoise": 1,
            },
        },
        "19": {"class_type": "VAEDecode", "inputs": {"samples": ["18", 0], "vae": ["14", 0]}},
        "20": {
            "class_type": "ImageCrop",
            "inputs": {
                "image": ["19", 0],
                "x": ["5", 2],
                "y": ["5", 3],
                "width": ["5", 4],
                "height": ["5", 5],
            },
        },
        "21": {"class_type": "SaveImage", "inputs": {"images": ["20", 0], "filename_prefix": filename_prefix}},
    }


def build_sheet(rows: list[dict[str, Any]], output_root: Path) -> None:
    cell_w = 270
    cell_h = 350
    header_h = 44
    headers = ["person", "reference", "mask", "output"]
    sheet = Image.new("RGB", (cell_w * len(headers), header_h + cell_h * len(rows)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, header in enumerate(headers):
        draw.rectangle((i * cell_w, 0, (i + 1) * cell_w, header_h), fill=(235, 238, 242), outline=(210, 215, 220))
        draw.text((i * cell_w + 10, 14), header, fill=(30, 35, 45))
    for r, row in enumerate(rows):
        y0 = header_h + r * cell_h
        draw.text((8, y0 + 8), f"{row['sample_id']} / {row['prompt_mode']}", fill=(0, 0, 0))
        paths = [row["person"], row["reference"], row["mask"], row["output"]]
        for c, path in enumerate(paths):
            x0 = c * cell_w
            draw.rectangle((x0, y0, x0 + cell_w, y0 + cell_h), outline=(220, 220, 220))
            if not Path(path).exists():
                draw.text((x0 + 12, y0 + 70), "missing", fill=(180, 0, 0))
                continue
            image = Image.open(path).convert("RGB")
            image.thumbnail((cell_w - 24, cell_h - 48), Image.Resampling.LANCZOS)
            sheet.paste(image, (x0 + (cell_w - image.width) // 2, y0 + 40))
    sheet.save(output_root / "upstream_flux_redux_workflow_grid.png")


def run_one(sample_id: str, prompt_mode: str, seed: int) -> dict[str, Any]:
    sample_dir = EVAL_ROOT / sample_id
    person_path = sample_dir / "person.png"
    reference_path = sample_reference(sample_dir)
    mask_path = sample_dir / "mask.png"
    if not person_path.exists() or not reference_path.exists() or not mask_path.exists():
        raise FileNotFoundError(f"Missing person/reference/mask for {sample_id}: {sample_dir}")

    out_dir = OUTPUT_ROOT / sample_id / prompt_mode
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(person_path, out_dir / "person.png")
    shutil.copy2(reference_path, out_dir / "reference.png")
    shutil.copy2(mask_path, out_dir / "mask.png")
    if UPSTREAM_WORKFLOW.exists():
        shutil.copy2(UPSTREAM_WORKFLOW, out_dir / "workflow_source_ui.json")

    person_name = copy_to_comfy_input(person_path, f"upstream_{sample_id}_person.png")
    reference_name = copy_to_comfy_input(reference_path, f"upstream_{sample_id}_reference.png")
    mask_name = copy_to_comfy_input(mask_path, f"upstream_{sample_id}_mask.png")
    prefix = f"upstream_flux_redux/{sample_id}_{prompt_mode}"
    prompt = build_upstream_equivalent_prompt(
        person_name=person_name,
        reference_name=reference_name,
        mask_name=mask_name,
        sample_id=sample_id,
        prompt_mode=prompt_mode,
        seed=seed,
        filename_prefix=prefix,
    )
    (out_dir / "workflow_used_api.json").write_text(json.dumps(prompt, indent=2), encoding="utf-8")

    started = time.perf_counter()
    history = queue_and_wait(prompt)
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    status = history.get("status", {})
    if not status.get("completed"):
        raise RuntimeError(f"ComfyUI run failed for {sample_id}/{prompt_mode}: {json.dumps(status)}")
    images = saved_images(history)
    if not images:
        raise RuntimeError(f"No saved image for {sample_id}/{prompt_mode}")
    output_path = out_dir / "output.png"
    shutil.copy2(images[-1], output_path)
    metadata = {
        "sample_id": sample_id,
        "prompt_mode": prompt_mode,
        "seed": seed,
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "source_workflow_repo": "https://github.com/fahdmirza/comfyuiworkflows.git",
        "source_workflow_file": "virtual-tryon-flux-redux-workflow.json",
        "compatibility_patches": [
            "UnetLoaderGGUF -> UNETLoader",
            "DualCLIPLoaderGGUF -> DualCLIPLoader",
            "StyleModelApplySimple -> StyleModelApply",
            "INPAINT_ExpandMask -> GrowMask",
            "ImageCrop+ -> ImageCrop",
            "separate mask PNG loaded through LoadImageMask",
        ],
        "person": str(person_path),
        "reference": str(reference_path),
        "mask": str(mask_path),
        "output": str(output_path),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata), flush=True)
    return metadata | {
        "person": str(out_dir / "person.png"),
        "reference": str(out_dir / "reference.png"),
        "mask": str(out_dir / "mask.png"),
        "output": str(output_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", nargs="+", default=["sample_001", "sample_015"])
    parser.add_argument("--prompt-modes", nargs="+", default=["repo_empty_prompt", "tryon_prompt"])
    parser.add_argument("--seed", type=int, default=2026062507)
    args = parser.parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for sample_index, sample_id in enumerate(args.samples):
        for mode_index, prompt_mode in enumerate(args.prompt_modes):
            rows.append(run_one(sample_id, prompt_mode, args.seed + sample_index * 100 + mode_index))
    build_sheet(rows, OUTPUT_ROOT)
    print(f"grid={OUTPUT_ROOT / 'upstream_flux_redux_workflow_grid.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
