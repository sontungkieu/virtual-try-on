from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw


PROJECT_ROOT = Path("/workspace/Project_Phase2")
EVAL_ROOT = PROJECT_ROOT / "virtual_tryon/data/temp/catvton_flux_eval_set"
CONFIG_ROOT = PROJECT_ROOT / "virtual_tryon/data/temp/production_vton_smoke_configs"
OUTPUT_ROOT = PROJECT_ROOT / "virtual_tryon/data/outputs"


def rel(path: Path) -> str:
    return path.as_posix()


def rounded_rect(draw: ImageDraw.ImageDraw, size: tuple[int, int], box: tuple[float, float, float, float], fill: int = 255) -> None:
    width, height = size
    xy = (
        int(width * box[0]),
        int(height * box[1]),
        int(width * box[2]),
        int(height * box[3]),
    )
    draw.rounded_rectangle(xy, radius=max(12, min(width, height) // 30), fill=fill)


def ellipse(draw: ImageDraw.ImageDraw, size: tuple[int, int], box: tuple[float, float, float, float], fill: int = 255) -> None:
    width, height = size
    xy = (
        int(width * box[0]),
        int(height * box[1]),
        int(width * box[2]),
        int(height * box[3]),
    )
    draw.ellipse(xy, fill=fill)


def polygon(draw: ImageDraw.ImageDraw, size: tuple[int, int], points: list[tuple[float, float]], fill: int = 255) -> None:
    width, height = size
    xy = [(int(width * x), int(height * y)) for x, y in points]
    draw.polygon(xy, fill=fill)


def save_upper_mask(person_path: Path, out_path: Path) -> None:
    person = Image.open(person_path).convert("RGB")
    mask = Image.new("L", person.size, 0)
    draw = ImageDraw.Draw(mask)
    rounded_rect(draw, person.size, (0.27, 0.18, 0.73, 0.57))
    rounded_rect(draw, person.size, (0.18, 0.25, 0.36, 0.50))
    rounded_rect(draw, person.size, (0.64, 0.25, 0.82, 0.50))
    mask.save(out_path)


def save_dress_mask(person_path: Path, out_path: Path) -> None:
    person = Image.open(person_path).convert("RGB")
    mask = Image.new("L", person.size, 0)
    draw = ImageDraw.Draw(mask)
    rounded_rect(draw, person.size, (0.30, 0.18, 0.70, 0.48))
    polygon(
        draw,
        person.size,
        [
            (0.30, 0.42),
            (0.70, 0.42),
            (0.79, 0.79),
            (0.68, 0.91),
            (0.32, 0.91),
            (0.21, 0.79),
        ],
    )
    rounded_rect(draw, person.size, (0.24, 0.28, 0.36, 0.52))
    rounded_rect(draw, person.size, (0.64, 0.28, 0.76, 0.52))
    mask.save(out_path)


def save_shoes_mask(person_path: Path, out_path: Path) -> None:
    person = Image.open(person_path).convert("RGB")
    mask = Image.new("L", person.size, 0)
    draw = ImageDraw.Draw(mask)
    ellipse(draw, person.size, (0.20, 0.84, 0.50, 0.99))
    ellipse(draw, person.size, (0.50, 0.84, 0.80, 0.99))
    rounded_rect(draw, person.size, (0.25, 0.84, 0.75, 1.00))
    mask.save(out_path)


def save_hat_mask(person_path: Path, out_path: Path) -> None:
    person = Image.open(person_path).convert("RGB")
    mask = Image.new("L", person.size, 0)
    draw = ImageDraw.Draw(mask)
    ellipse(draw, person.size, (0.28, 0.00, 0.72, 0.18))
    rounded_rect(draw, person.size, (0.24, 0.10, 0.76, 0.20))
    mask.save(out_path)


def base_config(run_name: str, person_image: Path, output_root: Path) -> dict:
    return {
        "run_name": run_name,
        "output_root": rel(output_root),
        "person_image": rel(person_image),
        "timeout_seconds": 2400,
        "allow_rectangle_fallback": False,
        "comfy": {
            "url": "http://127.0.0.1:8188",
            "input_dir": "/workspace/ComfyUI/input",
            "output_dir": "/workspace/ComfyUI/output",
            "models_dir": "/workspace/ComfyUI/models",
        },
        "models": {
            "flux_fill": "unet/FLUX1/fluxFillFP8_v10.safetensors",
            "flux_redux": "style_models/flux1-redux-dev.safetensors",
            "sigclip": "clip_vision/sigclip_vision_patch14_384.safetensors",
            "catviton_lora": "loras/flux/catvton-flux-lora.safetensors",
            "vae": "vae/FLUX1/ae.safetensors",
            "clip_l": "clip/clip_l.safetensors",
            "t5xxl": "clip/t5xxl_fp8_e4m3fn.safetensors",
        },
        "defaults": {
            "steps": 30,
            "guidance": 3.5,
            "denoise": 0.55,
            "sampler": "euler",
            "scheduler": "normal",
            "redux_strength": 0.80,
            "lora_strength": 1.0,
            "use_redux": True,
            "use_catviton_lora": True,
        },
        "mask_postprocess": {
            "dilation_px": 8,
            "erosion_px": 0,
            "feather_px": 6,
            "threshold": 8,
            "remove_face_hair": True,
            "remove_hands": True,
        },
        "refine": {
            "enabled": False,
            "steps": 12,
            "denoise": 0.25,
            "dilation_px": 10,
            "erosion_px": 8,
            "feather_px": 5,
        },
        "workflow_json": None,
        "workflow_patch": None,
    }


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> int:
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)

    sample_001 = EVAL_ROOT / "sample_001"
    sample_015 = EVAL_ROOT / "sample_015"

    mask_001_upper = CONFIG_ROOT / "sample_001_upper_mask.png"
    mask_015_dress = CONFIG_ROOT / "sample_015_dress_mask.png"
    mask_015_shoes = CONFIG_ROOT / "sample_015_shoes_mask.png"
    mask_015_hat = CONFIG_ROOT / "sample_015_hat_mask.png"

    save_upper_mask(sample_001 / "person.png", mask_001_upper)
    save_dress_mask(sample_015 / "person.png", mask_015_dress)
    save_shoes_mask(sample_015 / "person.png", mask_015_shoes)
    save_hat_mask(sample_015 / "person.png", mask_015_hat)

    config_001 = base_config(
        "production_sample_001_single_top",
        sample_001 / "person.png",
        OUTPUT_ROOT / "production_sample_001_single_top_20260625",
    )
    config_001["passes"] = [
        {
            "target_region": "upper",
            "garment_type": "upper",
            "garment_image": rel(sample_001 / "garment_top.png"),
            "mask_image": rel(mask_001_upper),
            "positive_prompt": (
                "High quality virtual try-on photo. Replace only the upper garment with the reference top. "
                "Preserve the person's identity, face, hair, hands, body shape, pose, lower clothing, lighting, and background."
            ),
            "negative_prompt": (
                "bad anatomy, deformed body, distorted face, extra limbs, damaged hands, wrong garment, "
                "changed pants, missing torso, blurry, low quality"
            ),
            "seed": 202606250101,
        }
    ]

    config_015 = base_config(
        "production_sample_015_multipass",
        sample_015 / "person.png",
        OUTPUT_ROOT / "production_sample_015_multipass_20260625",
    )
    config_015["passes"] = [
        {
            "target_region": "dress",
            "garment_type": "dress",
            "garment_image": rel(sample_015 / "garment_dress.png"),
            "mask_image": rel(mask_015_dress),
            "positive_prompt": (
                "High quality virtual try-on photo. Dress the person in the reference dress. "
                "Preserve identity, face, hair, hands, natural body shape, pose, shoes area, lighting, and background."
            ),
            "negative_prompt": (
                "bad anatomy, distorted face, extra limbs, damaged hands, wrong dress, missing legs, "
                "melted fabric, blurry, low quality"
            ),
            "seed": 202606250151,
            "denoise": 0.55,
        },
        {
            "target_region": "shoes",
            "garment_type": "shoes",
            "garment_image": rel(sample_015 / "accessory_shoes.png"),
            "mask_image": rel(mask_015_shoes),
            "positive_prompt": (
                "High quality virtual try-on photo. Add the reference shoes to the feet only. "
                "Preserve the dress from the previous pass, identity, pose, legs, lighting, and background."
            ),
            "negative_prompt": (
                "deformed feet, extra feet, wrong shoes, changed dress, damaged legs, blurry, low quality"
            ),
            "seed": 202606250152,
            "denoise": 0.38,
            "redux_strength": 0.85,
        },
        {
            "target_region": "hat",
            "garment_type": "hat",
            "garment_image": rel(sample_015 / "accessory_hat.png"),
            "mask_image": rel(mask_015_hat),
            "positive_prompt": (
                "High quality virtual try-on photo. Add the reference hat on the head only. "
                "Preserve the face, hairline, dress, shoes, pose, lighting, and background."
            ),
            "negative_prompt": (
                "distorted face, face damage, wrong hat, changed outfit, changed shoes, blurry, low quality"
            ),
            "seed": 202606250153,
            "denoise": 0.35,
            "redux_strength": 0.85,
        },
    ]

    write_json(CONFIG_ROOT / "sample_001_single_top.json", config_001)
    write_json(CONFIG_ROOT / "sample_015_multipass.json", config_015)

    print(
        json.dumps(
            {
                "config_root": rel(CONFIG_ROOT),
                "configs": [
                    rel(CONFIG_ROOT / "sample_001_single_top.json"),
                    rel(CONFIG_ROOT / "sample_015_multipass.json"),
                ],
                "masks": [
                    rel(mask_001_upper),
                    rel(mask_015_dress),
                    rel(mask_015_shoes),
                    rel(mask_015_hat),
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
