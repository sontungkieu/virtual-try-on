from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageOps


TARGET_SIZE = (768, 1024)
PROJECT_ROOT = Path(os.environ.get("VTON_PROJECT_ROOT", "/workspace/Project_Phase2/virtual_tryon"))
MODEL_DIR = Path(os.environ.get("VTON_KLEIN_MODEL_DIR", PROJECT_ROOT / "models" / "flux2-klein-9b"))
LORA_PATH = Path(
    os.environ.get(
        "VTON_KLEIN_LORA_PATH",
        "/workspace/hf-cache/hub/models--fal--flux-klein-9b-virtual-tryon-lora/"
        "snapshots/8b078b15c6d958ce48892b9ef31b66aa7587d792/"
        "flux-klein-tryon.safetensors",
    )
)
OUTPUT_ROOT = Path(os.environ.get("VTON_COMFY_OUTPUT_ROOT", PROJECT_ROOT / "data" / "outputs" / "comfyui_runs"))
WORKSPACE_ROOT = PROJECT_ROOT.parent
IDM_PARSER_ROOT = PROJECT_ROOT / "third_party" / "IDM-VTON" / "preprocess" / "humanparsing"
REAL_PARSER_CKPT = PROJECT_ROOT / "models" / "idm_vton" / "ckpt" / "humanparsing"
IDM_PARSER_CKPT = PROJECT_ROOT / "third_party" / "IDM-VTON" / "ckpt" / "humanparsing"
SAM_CHECKPOINT = PROJECT_ROOT / "models" / "sam" / "sam_vit_b_01ec64.pth"


_KLEIN_PIPE: Any | None = None


def _ensure_project_imports() -> None:
    backend = PROJECT_ROOT / "backend"
    scripts = PROJECT_ROOT / "scripts"
    for path in [backend, scripts, PROJECT_ROOT, WORKSPACE_ROOT]:
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def _patch_sdpa_enable_gqa() -> None:
    import torch

    func = torch.nn.functional.scaled_dot_product_attention
    if getattr(func, "_vton_gqa_compat", False):
        return

    def wrapped(query, key, value, *args, **kwargs):
        enable_gqa = bool(kwargs.pop("enable_gqa", False))
        if enable_gqa and query.ndim >= 4 and key.ndim >= 4:
            q_heads = query.shape[-3]
            k_heads = key.shape[-3]
            if q_heads != k_heads and q_heads % k_heads == 0:
                repeat = q_heads // k_heads
                key = key.repeat_interleave(repeat, dim=-3)
                value = value.repeat_interleave(repeat, dim=-3)
        return func(query, key, value, *args, **kwargs)

    wrapped._vton_gqa_compat = True  # type: ignore[attr-defined]
    torch.nn.functional.scaled_dot_product_attention = wrapped


def _tensor_to_pil(image: Any) -> Image.Image:
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    array = np.asarray(image)
    if array.ndim == 4:
        array = array[0]
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(array).convert("RGB")


def _pil_to_tensor(image: Image.Image) -> Any:
    import torch

    array = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(array)[None,]


def _pil_mask_to_tensor(mask: Image.Image) -> Any:
    import torch

    array = np.asarray(mask.convert("L")).astype(np.float32) / 255.0
    return torch.from_numpy(array)[None,]


def _tensor_to_mask_pil(mask: Any) -> Image.Image:
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()
    array = np.asarray(mask)
    if array.ndim == 4:
        array = array[0]
    if array.ndim == 3:
        if array.shape[-1] in {1, 3, 4}:
            array = array[..., 0]
        else:
            array = array[0]
    if array.max(initial=0) <= 1.0:
        array = array * 255.0
    array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array).convert("L")


def _fit_canvas(image: Image.Image, size: tuple[int, int] = TARGET_SIZE) -> Image.Image:
    image = ImageOps.exif_transpose(image.convert("RGB"))
    contained = ImageOps.contain(image, size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(contained, ((size[0] - contained.width) // 2, (size[1] - contained.height) // 2))
    return canvas


def _resize_cover(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return ImageOps.fit(ImageOps.exif_transpose(image.convert("RGB")), size, Image.Resampling.LANCZOS)


def _prepare_binary_mask(mask: Image.Image, size: tuple[int, int], dilation_px: int, feather_px: int) -> Image.Image:
    mask = mask.convert("L").resize(size, Image.Resampling.NEAREST)
    mask = mask.point(lambda p: 255 if p > 8 else 0)
    if dilation_px > 0:
        mask = mask.filter(ImageFilter.MaxFilter(int(dilation_px) * 2 + 1))
    if feather_px > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(float(feather_px)))
    return mask


def _mask_overlay(image: Image.Image, mask: Image.Image, alpha: float = 0.42) -> Image.Image:
    base = image.convert("RGBA")
    color = Image.new("RGBA", base.size, (36, 140, 255, 0))
    mask_alpha = mask.convert("L").point(lambda p: int(p * max(0.0, min(1.0, alpha))))
    color.putalpha(mask_alpha)
    return Image.alpha_composite(base, color).convert("RGB")


def _mask_area_ratio(mask: Image.Image) -> float:
    array = np.asarray(mask.convert("L"))
    return float((array > 8).mean())


def _crop_upper_person(image: Image.Image) -> Image.Image:
    width, height = image.size
    box = (int(width * 0.05), int(height * 0.03), int(width * 0.95), int(height * 0.58))
    return image.crop(box).convert("RGB")


def _two_panel(top: Image.Image, bottom: Image.Image) -> Image.Image:
    canvas = Image.new("RGB", TARGET_SIZE, "white")
    canvas.paste(_fit_canvas(top, (768, 512)), (0, 0))
    canvas.paste(_fit_canvas(bottom, (768, 512)), (0, 512))
    return canvas


def _three_panel(first: Image.Image, second: Image.Image, third: Image.Image) -> Image.Image:
    canvas = Image.new("RGB", TARGET_SIZE, "white")
    panel_h = TARGET_SIZE[1] // 3
    canvas.paste(_fit_canvas(first, (768, panel_h)), (0, 0))
    canvas.paste(_fit_canvas(second, (768, panel_h)), (0, panel_h))
    canvas.paste(_fit_canvas(third, (768, TARGET_SIZE[1] - panel_h * 2)), (0, panel_h * 2))
    return canvas


def _load_klein_base_pipe():
    global _KLEIN_PIPE
    if _KLEIN_PIPE is not None:
        return _KLEIN_PIPE

    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Missing Klein model dir: {MODEL_DIR}")

    _patch_sdpa_enable_gqa()
    import torch
    from diffusers import Flux2KleinPipeline

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    pipe = Flux2KleinPipeline.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    if hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
    pipe.enable_model_cpu_offload(gpu_id=0)
    pipe._vton_tryon_lora_loaded = False  # type: ignore[attr-defined]
    _KLEIN_PIPE = pipe
    return pipe


def _attach_klein_lora(pipe: Any, lora_scale: float):
    if not LORA_PATH.exists():
        raise FileNotFoundError(f"Missing Klein LoRA weights: {LORA_PATH}")
    if not getattr(pipe, "_vton_tryon_lora_loaded", False):
        pipe.load_lora_weights(
            LORA_PATH.parent,
            weight_name=LORA_PATH.name,
            adapter_name="tryon",
            local_files_only=True,
        )
        pipe._vton_tryon_lora_loaded = True  # type: ignore[attr-defined]
    if hasattr(pipe, "set_adapters"):
        pipe.set_adapters(["tryon"], adapter_weights=[float(lora_scale)])
    return pipe


def _disable_klein_lora_if_loaded(pipe: Any) -> None:
    if getattr(pipe, "_vton_tryon_lora_loaded", False) and hasattr(pipe, "set_adapters"):
        pipe.set_adapters(["tryon"], adapter_weights=[0.0])


def _load_klein_pipe(lora_scale: float):
    pipe = _load_klein_base_pipe()
    return _attach_klein_lora(pipe, lora_scale)


def _write_run_artifacts(run_dir: Path, payload: dict[str, Any], result: Image.Image | None = None) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "status.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if result is not None:
        result.save(run_dir / "result.png")


def _encode_png(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def _multipart_post(
    url: str,
    fields: dict[str, str],
    files: dict[str, tuple[str, bytes, str]],
    timeout: int,
) -> dict[str, Any]:
    boundary = f"----vtonphase2{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"),
                str(value).encode("utf-8"),
                b"\r\n",
            ]
        )
    for name, (filename, data, content_type) in files.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                (
                    f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                    f"Content-Type: {content_type}\r\n\r\n"
                ).encode("utf-8"),
                data,
                b"\r\n",
            ]
        )
    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    request = urllib.request.Request(
        url,
        data=b"".join(chunks),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _get_json(url: str, timeout: int) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _fetch_backend_image(api_base: str, url: str, timeout: int) -> Image.Image:
    full_url = urllib.parse.urljoin(api_base.rstrip("/") + "/", url.lstrip("/"))
    with urllib.request.urlopen(full_url, timeout=timeout) as response:
        return Image.open(BytesIO(response.read())).convert("RGB")


def _garment_upload_field(category: str) -> str:
    if category in {"upper_body", "women_bra"}:
        return "garment_top"
    if category in {"lower_body", "men_underwear", "women_underwear"}:
        return "garment_bottom"
    return "garment_dress"


def _ensure_parser_checkpoint_links() -> None:
    IDM_PARSER_CKPT.mkdir(parents=True, exist_ok=True)
    for name in ["parsing_atr.onnx", "parsing_lip.onnx"]:
        source = REAL_PARSER_CKPT / name
        target = IDM_PARSER_CKPT / name
        if not source.exists() or source.stat().st_size < 1_000_000:
            raise FileNotFoundError(f"Missing human parsing checkpoint: {source}")
        if target.exists() or target.is_symlink():
            if target.stat().st_size >= 1_000_000:
                continue
            target.unlink()
        target.symlink_to(source)


def _run_idm_atr_parser(person_path: Path, run_dir: Path) -> Path:
    _ensure_parser_checkpoint_links()
    _ensure_project_imports()
    for parser_path in [
        IDM_PARSER_ROOT,
        IDM_PARSER_ROOT / "utils",
        IDM_PARSER_ROOT / "datasets",
        IDM_PARSER_ROOT / "networks",
    ]:
        parser_text = str(parser_path)
        if parser_path.exists() and parser_text not in sys.path:
            sys.path.insert(0, parser_text)
    parser_root_text = str(IDM_PARSER_ROOT)
    for module_name in ["utils", "datasets", "networks", "modules"]:
        module = sys.modules.get(module_name)
        module_file = str(getattr(module, "__file__", "")) if module is not None else ""
        if module is not None and parser_root_text not in module_file:
            del sys.modules[module_name]
    from run_parsing import Parsing  # type: ignore

    semantic_path = run_dir / "semantic_atr.png"
    if semantic_path.exists():
        return semantic_path

    import tempfile
    import shutil

    with tempfile.TemporaryDirectory(prefix="comfy_schp_sam_parse_") as tmp:
        tmp_dir = Path(tmp)
        shutil.copy2(person_path, tmp_dir / person_path.name)
        parser = Parsing(0)
        parsed_image, _ = parser(tmp_dir.as_posix())
        parsed_image.save(semantic_path)
    return semantic_path


def _region_fallback_mask(person: Image.Image, target_region: str) -> Image.Image:
    from PIL import ImageDraw

    width, height = person.size
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    if target_region == "upper":
        draw.rounded_rectangle(
            (int(width * 0.24), int(height * 0.20), int(width * 0.76), int(height * 0.58)),
            radius=max(8, width // 30),
            fill=255,
        )
    elif target_region == "lower":
        draw.rounded_rectangle(
            (int(width * 0.26), int(height * 0.43), int(width * 0.74), int(height * 0.84)),
            radius=max(8, width // 30),
            fill=255,
        )
    elif target_region == "dress":
        draw.polygon(
            [
                (int(width * 0.30), int(height * 0.20)),
                (int(width * 0.70), int(height * 0.20)),
                (int(width * 0.82), int(height * 0.82)),
                (int(width * 0.18), int(height * 0.82)),
            ],
            fill=255,
        )
    elif target_region == "shoes":
        draw.rounded_rectangle(
            (int(width * 0.20), int(height * 0.82), int(width * 0.80), int(height * 0.99)),
            radius=max(4, width // 40),
            fill=255,
        )
    elif target_region == "hat":
        draw.ellipse(
            (int(width * 0.28), int(height * 0.01), int(width * 0.72), int(height * 0.25)),
            fill=255,
        )
    elif target_region == "accessory":
        draw.ellipse((int(width * 0.14), int(height * 0.45), int(width * 0.32), int(height * 0.64)), fill=255)
        draw.ellipse((int(width * 0.68), int(height * 0.45), int(width * 0.86), int(height * 0.64)), fill=255)
    else:
        draw.rectangle((int(width * 0.25), int(height * 0.25), int(width * 0.75), int(height * 0.75)), fill=255)
    return mask


class VTONPhase2KleinReferenceSet:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "person_image": ("IMAGE",),
                "ref1_image": ("IMAGE",),
                "ref2_image": ("IMAGE",),
                "ref3_image": ("IMAGE",),
                "reference_mode": (
                    [
                        "duplicate_ref1",
                        "lower_body_ref1_preserve_upper",
                        "top_ref1_bottom_ref2",
                        "accessory_ref1_ref2",
                        "dress_ref1_hat_ref3_shoes_ref2",
                    ],
                    {"default": "duplicate_ref1"},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("person_canvas", "top_reference", "bottom_reference", "primary_reference", "reference_mode")
    FUNCTION = "build"
    CATEGORY = "VTON Phase2"

    def build(self, person_image, ref1_image, ref2_image, ref3_image, reference_mode: str):
        person = _fit_canvas(_tensor_to_pil(person_image))
        ref1 = _fit_canvas(_tensor_to_pil(ref1_image))
        ref2 = _fit_canvas(_tensor_to_pil(ref2_image))
        ref3 = _fit_canvas(_tensor_to_pil(ref3_image))

        if reference_mode == "lower_body_ref1_preserve_upper":
            top = _fit_canvas(_crop_upper_person(_tensor_to_pil(person_image)))
            bottom = ref1
            primary = ref1
        elif reference_mode in {"top_ref1_bottom_ref2", "accessory_ref1_ref2"}:
            top = ref1
            bottom = ref2
            primary = _two_panel(ref1, ref2)
        elif reference_mode == "dress_ref1_hat_ref3_shoes_ref2":
            dress = ref1
            shoes = ref2
            hat = ref3
            top = _two_panel(hat, dress)
            bottom = _two_panel(dress, shoes)
            primary = _three_panel(hat, dress, shoes)
        else:
            top = ref1
            bottom = ref1
            primary = ref1

        return (
            _pil_to_tensor(person),
            _pil_to_tensor(top),
            _pil_to_tensor(bottom),
            _pil_to_tensor(primary),
            reference_mode,
        )


class VTONPhase2KleinFitCanvas:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {"default": TARGET_SIZE[0], "min": 256, "max": 2048, "step": 8}),
                "height": ("INT", {"default": TARGET_SIZE[1], "min": 256, "max": 2048, "step": 8}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("canvas_image",)
    FUNCTION = "run"
    CATEGORY = "VTON Phase2/Klein detailed"

    def run(self, image, width: int, height: int):
        return (_pil_to_tensor(_fit_canvas(_tensor_to_pil(image), (int(width), int(height)))),)


class VTONPhase2KleinBottomCrop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "person_image": ("IMAGE",),
                "x0_ratio": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 1.0, "step": 0.01}),
                "y0_ratio": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),
                "x1_ratio": ("FLOAT", {"default": 0.92, "min": 0.0, "max": 1.0, "step": 0.01}),
                "y1_ratio": ("FLOAT", {"default": 0.98, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("bottom_reference", "crop_box_json")
    FUNCTION = "run"
    CATEGORY = "VTON Phase2/Klein detailed"

    def run(self, person_image, x0_ratio: float, y0_ratio: float, x1_ratio: float, y1_ratio: float):
        person = _tensor_to_pil(person_image)
        width, height = person.size
        box = [
            int(width * float(x0_ratio)),
            int(height * float(y0_ratio)),
            int(width * float(x1_ratio)),
            int(height * float(y1_ratio)),
        ]
        crop = person.crop(tuple(box)).convert("RGB")
        return (_pil_to_tensor(_fit_canvas(crop)), json.dumps(box))


class VTONPhase2KleinPromptBuilder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_region": (
                    ["upper", "lower", "dress", "full_outfit", "shoes", "hat", "accessory"],
                    {"default": "upper"},
                ),
                "prompt_strength": (["default", "strong", "local_masked_tryon"], {"default": "strong"}),
                "item_description": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "the reference garment",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "run"
    CATEGORY = "VTON Phase2/Klein detailed"

    def run(self, target_region: str, prompt_strength: str, item_description: str):
        region_text = {
            "upper": "upper-body clothing / top",
            "lower": "lower-body clothing / pants/skirt",
            "dress": "full dress / one-piece outfit",
            "full_outfit": "full outfit",
            "shoes": "shoes only",
            "hat": "hat/headwear only",
            "accessory": "the accessory only",
        }.get(target_region, target_region)
        if prompt_strength == "local_masked_tryon":
            prompt = (
                f"TRYON cropped local fashion photo. Replace only the {region_text} in this cropped region "
                f"with {item_description.strip()}. Preserve the person's body shape, pose, skin, hands, lighting, "
                "camera angle, and all non-clothing details. Keep the result photorealistic and naturally fitted. "
                "Do not change unrelated regions. The final image is a realistic local crop that will be composited "
                "back into the original photo."
            )
            return (prompt,)
        base = (
            "TRYON full body adult fashion photo. "
            f"Replace the {target_region} region with {item_description.strip()}. "
            "Preserve the person's face, hair, hands, body shape, pose, legs, feet position, "
            "lighting, and background."
        )
        if prompt_strength == "strong":
            base += (
                " Remove the original item completely inside the target region. "
                "The final person must clearly wear the target reference item. "
                "Do not omit target garments, shoes, hats, logos, straps, hems, or accessories."
            )
        return (base,)


class VTONPhase2KleinLoadBaseModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_dir": (
                    "STRING",
                    {
                        "default": str(MODEL_DIR),
                    },
                ),
            }
        }

    RETURN_TYPES = ("VTON_KLEIN_PIPE", "STRING")
    RETURN_NAMES = ("klein_pipeline", "status")
    FUNCTION = "run"
    CATEGORY = "VTON Phase2/Klein detailed"

    def run(self, model_dir: str):
        global MODEL_DIR
        MODEL_DIR = Path(model_dir)
        pipe = _load_klein_base_pipe()
        payload = {
            "model": "black-forest-labs/FLUX.2-klein-9B",
            "model_dir": str(MODEL_DIR),
            "lora_loaded": bool(getattr(pipe, "_vton_tryon_lora_loaded", False)),
        }
        return ({"pipe": pipe, "uses_tryon_lora": False, "lora_scale": 0.0}, json.dumps(payload, indent=2))


class VTONPhase2KleinLoadTryOnLoRA:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "klein_pipeline": ("VTON_KLEIN_PIPE",),
                "lora_path": (
                    "STRING",
                    {
                        "default": str(LORA_PATH),
                    },
                ),
                "lora_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("VTON_KLEIN_PIPE", "STRING")
    RETURN_NAMES = ("klein_pipeline_with_lora", "status")
    FUNCTION = "run"
    CATEGORY = "VTON Phase2/Klein detailed"

    def run(self, klein_pipeline, lora_path: str, lora_scale: float):
        global LORA_PATH
        LORA_PATH = Path(lora_path)
        pipe = _attach_klein_lora(klein_pipeline["pipe"], float(lora_scale))
        payload = {
            "adapter": "fal/flux-klein-9b-virtual-tryon-lora",
            "lora_path": str(LORA_PATH),
            "lora_scale": float(lora_scale),
        }
        return ({"pipe": pipe, "uses_tryon_lora": True, "lora_scale": float(lora_scale)}, json.dumps(payload, indent=2))


class VTONPhase2KleinSamplerDetailed:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "klein_pipeline": ("VTON_KLEIN_PIPE",),
                "person_canvas": ("IMAGE",),
                "top_reference": ("IMAGE",),
                "bottom_reference": ("IMAGE",),
                "prompt": ("STRING", {"forceInput": True}),
                "seed": ("INT", {"default": 4242, "min": 0, "max": 2147483647}),
                "steps": ("INT", {"default": 28, "min": 1, "max": 64}),
                "guidance_scale": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "width": ("INT", {"default": TARGET_SIZE[0], "min": 256, "max": 2048, "step": 8}),
                "height": ("INT", {"default": TARGET_SIZE[1], "min": 256, "max": 2048, "step": 8}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "status")
    FUNCTION = "run"
    CATEGORY = "VTON Phase2/Klein detailed"

    def run(
        self,
        klein_pipeline,
        person_canvas,
        top_reference,
        bottom_reference,
        prompt: str,
        seed: int,
        steps: int,
        guidance_scale: float,
        width: int,
        height: int,
    ):
        import torch

        pipe = klein_pipeline["pipe"]
        uses_lora = bool(klein_pipeline.get("uses_tryon_lora", False))
        lora_scale = float(klein_pipeline.get("lora_scale", 0.0))
        if uses_lora:
            _attach_klein_lora(pipe, lora_scale)
        else:
            _disable_klein_lora_if_loaded(pipe)

        run_id = f"comfy_klein_detailed_{'lora' if uses_lora else 'base'}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        run_dir = OUTPUT_ROOT / run_id
        started = time.perf_counter()
        payload = {
            "status": "running",
            "backend": "ComfyUI detailed custom nodes",
            "base_model": "black-forest-labs/FLUX.2-klein-9B",
            "uses_tryon_lora": uses_lora,
            "lora_path": str(LORA_PATH) if uses_lora else None,
            "lora_scale": lora_scale if uses_lora else 0.0,
            "steps": int(steps),
            "guidance_scale": float(guidance_scale),
            "seed": int(seed),
            "width": int(width),
            "height": int(height),
            "prompt": prompt,
            "run_dir": str(run_dir),
        }
        _write_run_artifacts(run_dir, payload)

        try:
            generator = torch.Generator(device="cuda").manual_seed(int(seed))
            image = pipe(
                image=[
                    _tensor_to_pil(person_canvas),
                    _tensor_to_pil(top_reference),
                    _tensor_to_pil(bottom_reference),
                ],
                prompt=prompt,
                height=int(height),
                width=int(width),
                num_inference_steps=int(steps),
                guidance_scale=float(guidance_scale),
                generator=generator,
            ).images[0].convert("RGB")
            payload.update(
                {
                    "status": "completed",
                    "runtime_seconds": round(time.perf_counter() - started, 3),
                    "result_path": str(run_dir / "result.png"),
                }
            )
            _write_run_artifacts(run_dir, payload, image)
            return (_pil_to_tensor(image), json.dumps(payload, indent=2))
        except Exception as exc:
            payload.update(
                {
                    "status": "failed",
                    "runtime_seconds": round(time.perf_counter() - started, 3),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            _write_run_artifacts(run_dir, payload)
            raise
        finally:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


class VTONPhase2KleinSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "person_canvas": ("IMAGE",),
                "top_reference": ("IMAGE",),
                "bottom_reference": ("IMAGE",),
                "method": (["klein_4step", "klein_28", "klein_28_strong"], {"default": "klein_28"}),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": (
                            "TRYON full body fashion photo using the reference garment. "
                            "Preserve face, hair, body shape, pose, hands, legs, and background."
                        ),
                    },
                ),
                "seed": ("INT", {"default": 4242, "min": 0, "max": 2147483647}),
                "guidance_scale": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "lora_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "status")
    FUNCTION = "run"
    CATEGORY = "VTON Phase2"

    def run(
        self,
        person_canvas,
        top_reference,
        bottom_reference,
        method: str,
        prompt: str,
        seed: int,
        guidance_scale: float,
        lora_scale: float,
    ):
        import torch

        steps_by_method = {
            "klein_4step": 4,
            "klein_28": 28,
            "klein_28_strong": 28,
        }
        if method == "klein_28_strong":
            prompt = (
                prompt.strip()
                + " The final person must clearly wear all target items. Do not omit target garments or accessories. "
                "Preserve face, hair, body shape, pose, legs, feet, and background."
            )

        steps = steps_by_method[method]
        run_id = f"comfy_klein_{method}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        run_dir = OUTPUT_ROOT / run_id
        started = time.perf_counter()
        payload = {
            "status": "running",
            "backend": "ComfyUI custom node",
            "engine": "klein9b_lora",
            "model": str(MODEL_DIR),
            "lora": str(LORA_PATH),
            "method": method,
            "steps": steps,
            "guidance_scale": float(guidance_scale),
            "lora_scale": float(lora_scale),
            "seed": int(seed),
            "prompt": prompt,
            "run_dir": str(run_dir),
        }
        _write_run_artifacts(run_dir, payload)

        try:
            pipe = _load_klein_pipe(lora_scale)
            generator = torch.Generator(device="cuda").manual_seed(int(seed))
            image = pipe(
                image=[
                    _tensor_to_pil(person_canvas),
                    _tensor_to_pil(top_reference),
                    _tensor_to_pil(bottom_reference),
                ],
                prompt=prompt,
                height=TARGET_SIZE[1],
                width=TARGET_SIZE[0],
                num_inference_steps=steps,
                guidance_scale=float(guidance_scale),
                generator=generator,
            ).images[0].convert("RGB")
            payload.update(
                {
                    "status": "completed",
                    "runtime_seconds": round(time.perf_counter() - started, 3),
                    "result_path": str(run_dir / "result.png"),
                }
            )
            _write_run_artifacts(run_dir, payload, image)
            return (_pil_to_tensor(image), json.dumps(payload, indent=2))
        except Exception as exc:
            payload.update(
                {
                    "status": "failed",
                    "runtime_seconds": round(time.perf_counter() - started, 3),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            _write_run_artifacts(run_dir, payload)
            raise
        finally:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


class AddMaskForICLora:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "first_image": ("IMAGE",),
                "second_image": ("IMAGE",),
                "second_mask": ("MASK",),
                "patch_mode": (["auto", "horizontal", "vertical"], {"default": "auto"}),
                "output_length": ("INT", {"default": 1536, "min": 256, "max": 4096, "step": 8}),
                "patch_color": ("STRING", {"default": "#FF0000"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("image", "mask", "x", "y", "width", "height")
    FUNCTION = "run"
    CATEGORY = "VTON Phase2/Compat"

    def run(self, first_image, second_image, second_mask, patch_mode: str, output_length: int, patch_color: str):
        first = _fit_canvas(_tensor_to_pil(first_image), _tensor_to_pil(second_image).size)
        second = _tensor_to_pil(second_image)
        mask = _tensor_to_mask_pil(second_mask).resize(second.size, Image.Resampling.NEAREST)

        if patch_mode == "vertical":
            canvas = Image.new("RGB", (second.width, first.height + second.height), "white")
            canvas.paste(first, (0, 0))
            canvas.paste(second, (0, first.height))
            packed_mask = Image.new("L", canvas.size, 0)
            packed_mask.paste(mask, (0, first.height))
            crop = (0, first.height, second.width, second.height)
        else:
            canvas = Image.new("RGB", (first.width + second.width, second.height), "white")
            canvas.paste(first, (0, 0))
            canvas.paste(second, (first.width, 0))
            packed_mask = Image.new("L", canvas.size, 0)
            packed_mask.paste(mask, (first.width, 0))
            crop = (first.width, 0, second.width, second.height)

        return (
            _pil_to_tensor(canvas),
            _pil_mask_to_tensor(packed_mask),
            int(crop[0]),
            int(crop[1]),
            int(crop[2]),
            int(crop[3]),
        )


class VTONPhase2MaskComposite:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "person_image": ("IMAGE",),
                "candidate_image": ("IMAGE",),
                "mask": ("MASK",),
                "resize_mode": (["resize_exact", "cover_crop", "contain_pad"], {"default": "resize_exact"}),
                "mask_dilation_px": ("INT", {"default": 4, "min": 0, "max": 64, "step": 1}),
                "mask_feather_px": ("INT", {"default": 6, "min": 0, "max": 64, "step": 1}),
                "overlay_alpha": ("FLOAT", {"default": 0.42, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("composite_image", "mask_overlay", "soft_mask", "status")
    FUNCTION = "run"
    CATEGORY = "VTON Phase2/Klein local repair"

    def run(
        self,
        person_image,
        candidate_image,
        mask,
        resize_mode: str,
        mask_dilation_px: int,
        mask_feather_px: int,
        overlay_alpha: float,
    ):
        person = _tensor_to_pil(person_image)
        candidate = _tensor_to_pil(candidate_image)
        if candidate.size != person.size:
            if resize_mode == "cover_crop":
                candidate = _resize_cover(candidate, person.size)
            elif resize_mode == "contain_pad":
                candidate = _fit_canvas(candidate, person.size)
            else:
                candidate = candidate.resize(person.size, Image.Resampling.LANCZOS)

        soft_mask = _prepare_binary_mask(_tensor_to_mask_pil(mask), person.size, int(mask_dilation_px), int(mask_feather_px))
        composite = Image.composite(candidate, person, soft_mask)
        overlay = _mask_overlay(person, soft_mask, float(overlay_alpha))
        payload = {
            "node": "VTON Phase2 - Masked Candidate Composite",
            "resize_mode": resize_mode,
            "mask_dilation_px": int(mask_dilation_px),
            "mask_feather_px": int(mask_feather_px),
            "mask_area_ratio": round(_mask_area_ratio(soft_mask), 5),
            "contract": "Only the masked target region is taken from the Klein candidate; unmasked pixels come from the original person image.",
        }
        return (
            _pil_to_tensor(composite),
            _pil_to_tensor(overlay),
            _pil_mask_to_tensor(soft_mask),
            json.dumps(payload, indent=2, ensure_ascii=False),
        )


class VTONPhase2LocalSeamRepair:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "edge_dilation_px": ("INT", {"default": 8, "min": 0, "max": 96, "step": 1}),
                "edge_erosion_px": ("INT", {"default": 3, "min": 0, "max": 48, "step": 1}),
                "edge_feather_px": ("INT", {"default": 6, "min": 0, "max": 64, "step": 1}),
                "sharpness": ("FLOAT", {"default": 1.12, "min": 0.1, "max": 3.0, "step": 0.05}),
                "contrast": ("FLOAT", {"default": 1.03, "min": 0.1, "max": 3.0, "step": 0.05}),
                "overlay_alpha": ("FLOAT", {"default": 0.55, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "STRING")
    RETURN_NAMES = ("repaired_image", "edge_mask", "edge_overlay", "status")
    FUNCTION = "run"
    CATEGORY = "VTON Phase2/Klein local repair"

    def run(
        self,
        image,
        mask,
        edge_dilation_px: int,
        edge_erosion_px: int,
        edge_feather_px: int,
        sharpness: float,
        contrast: float,
        overlay_alpha: float,
    ):
        base = _tensor_to_pil(image)
        binary = _tensor_to_mask_pil(mask).resize(base.size, Image.Resampling.NEAREST)
        binary = binary.point(lambda p: 255 if p > 8 else 0)
        dilated = binary.filter(ImageFilter.MaxFilter(int(edge_dilation_px) * 2 + 1)) if edge_dilation_px > 0 else binary
        eroded = binary.filter(ImageFilter.MinFilter(int(edge_erosion_px) * 2 + 1)) if edge_erosion_px > 0 else binary
        edge_mask = ImageChops.subtract(dilated, eroded)
        if edge_feather_px > 0:
            edge_mask = edge_mask.filter(ImageFilter.GaussianBlur(float(edge_feather_px)))

        enhanced = ImageEnhance.Sharpness(base).enhance(float(sharpness))
        enhanced = ImageEnhance.Contrast(enhanced).enhance(float(contrast))
        repaired = Image.composite(enhanced, base, edge_mask)
        overlay = _mask_overlay(base, edge_mask, float(overlay_alpha))
        payload = {
            "node": "VTON Phase2 - Local Seam Repair",
            "edge_dilation_px": int(edge_dilation_px),
            "edge_erosion_px": int(edge_erosion_px),
            "edge_feather_px": int(edge_feather_px),
            "sharpness": float(sharpness),
            "contrast": float(contrast),
            "edge_area_ratio": round(_mask_area_ratio(edge_mask), 5),
            "contract": "This is a non-diffusion local seam/detail repair. It is not ADetailer.",
        }
        return (
            _pil_to_tensor(repaired),
            _pil_mask_to_tensor(edge_mask),
            _pil_to_tensor(overlay),
            json.dumps(payload, indent=2, ensure_ascii=False),
        )


class VTONPhase2MaskPreviewImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("mask_image",)
    FUNCTION = "run"
    CATEGORY = "VTON Phase2/Local masked inpaint"

    def run(self, mask):
        preview = _tensor_to_mask_pil(mask).convert("RGB")
        return (_pil_to_tensor(preview),)


class VTONPhase2MaskMorphology:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "grow_px": ("INT", {"default": 16, "min": -64, "max": 128, "step": 1}),
                "blur_px": ("INT", {"default": 10, "min": 0, "max": 96, "step": 1}),
                "threshold": ("INT", {"default": 8, "min": 0, "max": 255, "step": 1}),
                "invert": ("BOOLEAN", {"default": False}),
                "fill_holes": ("BOOLEAN", {"default": True}),
                "keep_largest_component": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("processed_mask", "status")
    FUNCTION = "run"
    CATEGORY = "VTON Phase2/Local masked inpaint"

    def run(self, mask, grow_px: int, blur_px: int, threshold: int, invert: bool, fill_holes: bool, keep_largest_component: bool):
        from virtual_tryon.masking import MaskMorphologyConfig, mask_morphology

        result = mask_morphology(
            _tensor_to_mask_pil(mask),
            MaskMorphologyConfig(
                grow_px=int(grow_px),
                blur_px=int(blur_px),
                threshold=int(threshold),
                invert=bool(invert),
                fill_holes=bool(fill_holes),
                keep_largest_component=bool(keep_largest_component),
            ),
        )
        return (_pil_mask_to_tensor(result.mask), json.dumps(result.status, indent=2, ensure_ascii=False))


class VTONPhase2MaskBBoxCrop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "padding_ratio": ("FLOAT", {"default": 0.22, "min": 0.0, "max": 1.0, "step": 0.01}),
                "min_padding_px": ("INT", {"default": 32, "min": 0, "max": 512, "step": 1}),
                "max_padding_px": ("INT", {"default": 160, "min": 0, "max": 1024, "step": 1}),
                "force_square": ("BOOLEAN", {"default": False}),
                "target_multiple": ("INT", {"default": 16, "min": 1, "max": 128, "step": 1}),
                "min_crop_size": ("INT", {"default": 256, "min": 32, "max": 2048, "step": 8}),
                "target_region": (
                    ["upper", "lower", "dress", "shoes", "hat", "accessory"],
                    {"default": "upper"},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("cropped_image", "cropped_mask", "bbox_json", "status", "bbox_overlay")
    FUNCTION = "run"
    CATEGORY = "VTON Phase2/Local masked inpaint"

    def run(
        self,
        image,
        mask,
        padding_ratio: float,
        min_padding_px: int,
        max_padding_px: int,
        force_square: bool,
        target_multiple: int,
        min_crop_size: int,
        target_region: str,
    ):
        from virtual_tryon.masking import BBoxCropConfig, bbox_crop

        result = bbox_crop(
            _tensor_to_pil(image),
            _tensor_to_mask_pil(mask),
            BBoxCropConfig(
                padding_ratio=float(padding_ratio),
                min_padding_px=int(min_padding_px),
                max_padding_px=int(max_padding_px),
                force_square=bool(force_square),
                target_multiple=int(target_multiple),
                min_crop_size=int(min_crop_size),
                target_region=target_region,
            ),
        )
        return (
            _pil_to_tensor(result.image),
            _pil_mask_to_tensor(result.mask),
            result.bbox_json,
            json.dumps(result.status, indent=2, ensure_ascii=False),
            _pil_to_tensor(result.overlay),
        )


class VTONPhase2MaskedPasteBack:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "generated_crop": ("IMAGE",),
                "cropped_mask": ("MASK",),
                "bbox_json": ("STRING", {"forceInput": True}),
                "feather_px": ("INT", {"default": 12, "min": 0, "max": 96, "step": 1}),
                "color_match": ("BOOLEAN", {"default": True}),
                "preserve_outside_mask": ("BOOLEAN", {"default": True}),
                "debug": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("final_image", "paste_overlay", "status")
    FUNCTION = "run"
    CATEGORY = "VTON Phase2/Local masked inpaint"

    def run(
        self,
        original_image,
        generated_crop,
        cropped_mask,
        bbox_json: str,
        feather_px: int,
        color_match: bool,
        preserve_outside_mask: bool,
        debug: bool,
    ):
        from virtual_tryon.masking import PasteBackConfig, masked_paste_back

        result = masked_paste_back(
            _tensor_to_pil(original_image),
            _tensor_to_pil(generated_crop),
            _tensor_to_mask_pil(cropped_mask),
            bbox_json,
            PasteBackConfig(
                feather_px=int(feather_px),
                color_match=bool(color_match),
                preserve_outside_mask=bool(preserve_outside_mask),
                debug=bool(debug),
            ),
        )
        return (
            _pil_to_tensor(result.image),
            _pil_to_tensor(result.overlay),
            json.dumps(result.status, indent=2, ensure_ascii=False),
        )


class VTONPhase2FitCanvasWithMeta:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {"default": TARGET_SIZE[0], "min": 256, "max": 2048, "step": 8}),
                "height": ("INT", {"default": TARGET_SIZE[1], "min": 256, "max": 2048, "step": 8}),
                "background": (["white", "black", "gray"], {"default": "white"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("canvas_image", "fit_meta_json", "status")
    FUNCTION = "run"
    CATEGORY = "VTON Phase2/Local masked inpaint"

    def run(self, image, width: int, height: int, background: str):
        from virtual_tryon.masking import FitCanvasConfig, fit_canvas_with_meta

        result = fit_canvas_with_meta(
            _tensor_to_pil(image),
            FitCanvasConfig(width=int(width), height=int(height), background=background),
        )
        return (
            _pil_to_tensor(result.image),
            result.bbox_json or "{}",
            json.dumps(result.status, indent=2, ensure_ascii=False),
        )


class VTONPhase2ExtractFittedCanvasRegion:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generated_canvas": ("IMAGE",),
                "fit_meta_json": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("generated_crop", "status")
    FUNCTION = "run"
    CATEGORY = "VTON Phase2/Local masked inpaint"

    def run(self, generated_canvas, fit_meta_json: str):
        from virtual_tryon.masking import extract_fitted_canvas_region

        result = extract_fitted_canvas_region(_tensor_to_pil(generated_canvas), fit_meta_json)
        return (_pil_to_tensor(result.image), json.dumps(result.status, indent=2, ensure_ascii=False))


class VTONPhase2TryOnDebugSheet:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "person_image": ("IMAGE",),
                "garment_image": ("IMAGE",),
                "raw_mask": ("MASK",),
                "processed_mask": ("MASK",),
                "mask_overlay": ("IMAGE",),
                "crop_image": ("IMAGE",),
                "crop_mask": ("MASK",),
                "generated_crop": ("IMAGE",),
                "final_image": ("IMAGE",),
                "status_text": ("STRING", {"multiline": True, "default": "local masked try-on"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("debug_sheet", "status")
    FUNCTION = "run"
    CATEGORY = "VTON Phase2/Local masked inpaint"

    def run(
        self,
        person_image,
        garment_image,
        raw_mask,
        processed_mask,
        mask_overlay,
        crop_image,
        crop_mask,
        generated_crop,
        final_image,
        status_text: str,
    ):
        from virtual_tryon.masking import make_debug_sheet

        result = make_debug_sheet(
            person=_tensor_to_pil(person_image),
            garment=_tensor_to_pil(garment_image),
            raw_mask=_tensor_to_mask_pil(raw_mask),
            processed_mask=_tensor_to_mask_pil(processed_mask),
            overlay=_tensor_to_pil(mask_overlay),
            crop_image=_tensor_to_pil(crop_image),
            crop_mask=_tensor_to_mask_pil(crop_mask),
            generated_crop=_tensor_to_pil(generated_crop),
            final_image=_tensor_to_pil(final_image),
            status_text=status_text,
        )
        return (_pil_to_tensor(result.image), json.dumps(result.status, indent=2, ensure_ascii=False))


class VTONPhase2SCHPSAMMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "person_image": ("IMAGE",),
                "target_region": (
                    ["upper", "lower", "dress", "shoes", "hat", "accessory"],
                    {"default": "upper"},
                ),
                "dilation_px": ("INT", {"default": 2, "min": 0, "max": 64, "step": 1}),
                "erosion_px": ("INT", {"default": 0, "min": 0, "max": 32, "step": 1}),
                "feather_px": ("INT", {"default": 2, "min": 0, "max": 64, "step": 1}),
                "use_sam_refine": ("BOOLEAN", {"default": True}),
                "sam_bbox_padding_px": ("INT", {"default": 8, "min": 0, "max": 96, "step": 1}),
                "semantic_envelope_dilation_px": ("INT", {"default": 14, "min": 0, "max": 128, "step": 1}),
                "fallback_mode": (
                    ["fail_if_empty", "heuristic_if_empty", "target_extent_if_empty_or_small"],
                    {"default": "target_extent_if_empty_or_small"},
                ),
            }
        }

    RETURN_TYPES = ("MASK", "MASK", "IMAGE", "STRING")
    RETURN_NAMES = ("processed_mask", "raw_mask", "mask_overlay", "status")
    FUNCTION = "run"
    CATEGORY = "VTON Phase2/Masking"

    def run(
        self,
        person_image,
        target_region: str,
        dilation_px: int,
        erosion_px: int,
        feather_px: int,
        use_sam_refine: bool,
        sam_bbox_padding_px: int,
        semantic_envelope_dilation_px: int,
        fallback_mode: str,
    ):
        _ensure_project_imports()
        from virtual_tryon.masking import (
            HybridMaskConfig,
            MaskPostprocessConfig,
            SAMMaskConfig,
            build_hybrid_vton_mask,
            create_target_extent_mask,
            should_use_target_extent_fallback,
        )

        run_id = f"comfy_schp_sam_{target_region}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        run_dir = OUTPUT_ROOT / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        person = _tensor_to_pil(person_image)
        person_path = run_dir / "input_person.png"
        person.save(person_path)

        started = time.perf_counter()
        warnings: list[str] = []
        semantic_path = _run_idm_atr_parser(person_path, run_dir)
        config = HybridMaskConfig.atr(
            postprocess=MaskPostprocessConfig(
                dilation_px=int(dilation_px),
                erosion_px=int(erosion_px),
                feather_px=int(feather_px),
                remove_face_hair=True,
                remove_hands=True,
            ),
            sam=SAMMaskConfig(checkpoint_path=SAM_CHECKPOINT, model_type="vit_b", device="cuda"),
            refine_with_sam=bool(use_sam_refine),
            sam_bbox_padding_px=int(sam_bbox_padding_px),
            intersect_sam_with_semantic_envelope=True,
            semantic_envelope_dilation_px=int(semantic_envelope_dilation_px),
            max_area_ratio=0.50,
        )

        fallback_path: Path | None = None

        def fallback_config() -> HybridMaskConfig:
            return HybridMaskConfig.atr(
                postprocess=config.postprocess,
                sam=config.sam,
                refine_with_sam=config.refine_with_sam,
                sam_bbox_padding_px=config.sam_bbox_padding_px,
                intersect_sam_with_semantic_envelope=config.intersect_sam_with_semantic_envelope,
                semantic_envelope_dilation_px=config.semantic_envelope_dilation_px,
                prefer_semantic=False,
                subtract_semantic_protect=target_region not in {"hat", "accessory", "shoes"},
                max_area_ratio=config.max_area_ratio,
            )

        def build_fallback(reason: str, *, prefer_target_extent: bool) -> Any:
            nonlocal fallback_path
            fallback_warnings: list[str] = []
            fallback_label = "legacy heuristic"
            fallback_path = run_dir / f"{target_region}_heuristic_fallback.png"
            if prefer_target_extent:
                try:
                    extent = create_target_extent_mask(
                        person,
                        target_region,  # type: ignore[arg-type]
                        semantic_map_path=semantic_path,
                    )
                    fallback_path = run_dir / f"{target_region}_target_extent_mask.png"
                    extent.mask.save(fallback_path)
                    fallback_label = "target extent"
                    fallback_warnings.extend(extent.warnings)
                except Exception as extent_exc:
                    fallback_warnings.append(
                        "Target extent fallback failed "
                        f"({type(extent_exc).__name__}: {extent_exc}); used legacy heuristic."
                    )
                    _region_fallback_mask(person, target_region).save(fallback_path)
            else:
                _region_fallback_mask(person, target_region).save(fallback_path)

            result = build_hybrid_vton_mask(
                person,
                target_region,  # type: ignore[arg-type]
                fallback_config(),
                semantic_map_path=semantic_path,
                manual_mask_path=fallback_path,
            )
            warnings.extend(
                [
                    f"{reason}; used {fallback_label} fallback mask before SAM/postprocess.",
                    *fallback_warnings,
                ]
            )
            return result

        try:
            result = build_hybrid_vton_mask(
                person,
                target_region,  # type: ignore[arg-type]
                config,
                semantic_map_path=semantic_path,
            )
            if fallback_mode == "target_extent_if_empty_or_small" and should_use_target_extent_fallback(
                target_region,
                result.mask_area_ratio,
                result.warnings,
            ):
                result = build_fallback(
                    "Semantic SCHP/ATR mask was available but too small for the desired try-on target "
                    f"(area_ratio={result.mask_area_ratio:.4f})",
                    prefer_target_extent=True,
                )
        except ValueError as exc:
            if fallback_mode == "fail_if_empty":
                raise
            result = build_fallback(
                f"Semantic SCHP/ATR target was empty for region '{target_region}' ({exc})",
                prefer_target_extent=fallback_mode == "target_extent_if_empty_or_small",
            )

        raw_path = run_dir / "mask_raw.png"
        processed_path = run_dir / "mask_processed.png"
        overlay_path = run_dir / "mask_overlay.png"
        protect_path = run_dir / "mask_protect.png"
        result.raw_mask.save(raw_path)
        result.processed_mask.save(processed_path)
        result.overlay.save(overlay_path)
        result.protect_mask.save(protect_path)

        payload = {
            "status": "completed",
            "node": "VTON Phase2 - SCHP/SAM Mask",
            "target_region": target_region,
            "source": result.source,
            "semantic_map_path": str(semantic_path),
            "sam_checkpoint": str(SAM_CHECKPOINT),
            "fallback_mask_path": str(fallback_path) if fallback_path else None,
            "raw_mask": str(raw_path),
            "processed_mask": str(processed_path),
            "overlay": str(overlay_path),
            "protect_mask": str(protect_path),
            "mask_area_ratio": round(result.mask_area_ratio, 5),
            "bbox_xyxy": result.bbox_xyxy,
            "warnings": warnings + result.warnings,
            "runtime_seconds": round(time.perf_counter() - started, 3),
            "run_dir": str(run_dir),
        }
        (run_dir / "status.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return (
            _pil_mask_to_tensor(result.processed_mask),
            _pil_mask_to_tensor(result.raw_mask),
            _pil_to_tensor(result.overlay),
            json.dumps(payload, indent=2, ensure_ascii=False),
        )


class VTONPhase2BackendTryOnAPI:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "person_image": ("IMAGE",),
                "garment_image": ("IMAGE",),
                "category": (
                    [
                        "upper_body",
                        "lower_body",
                        "dress",
                        "full_outfit",
                        "men_underwear",
                        "women_underwear",
                        "women_bra",
                    ],
                    {"default": "women_underwear"},
                ),
                "engine_mode": (
                    ["idm_vton", "idm_mask_expanded", "idm_vton_flux", "idm_mask_expanded_flux", "klein_lora", "catvton"],
                    {"default": "idm_vton"},
                ),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": (
                            "Adult non-sexual virtual try-on. Replace only the target garment region with the "
                            "reference garment. Preserve identity, pose, skin, lighting, and background."
                        ),
                    },
                ),
                "seed": ("INT", {"default": 2026070201, "min": 0, "max": 2147483647}),
                "api_base": ("STRING", {"default": "http://127.0.0.1:8000"}),
                "output_width": ("INT", {"default": 512, "min": 256, "max": 1536, "step": 8}),
                "output_height": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 8}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 64, "step": 1}),
                "deterministic": ("BOOLEAN", {"default": True}),
                "timeout_s": ("INT", {"default": 900, "min": 30, "max": 7200, "step": 30}),
                "poll_interval_s": ("FLOAT", {"default": 1.0, "min": 0.2, "max": 10.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("result_image", "mask_preview", "status")
    FUNCTION = "run"
    CATEGORY = "VTON Phase2/API"

    def run(
        self,
        person_image,
        garment_image,
        category: str,
        engine_mode: str,
        prompt: str,
        seed: int,
        api_base: str,
        output_width: int,
        output_height: int,
        steps: int,
        deterministic: bool,
        timeout_s: int,
        poll_interval_s: float,
    ):
        api_base = api_base.rstrip("/")
        garment_field = _garment_upload_field(category)
        started = time.perf_counter()
        fields = {
            "category": category,
            "prompt": prompt,
            "use_refiner": str(engine_mode in {"idm_vton_flux", "idm_mask_expanded_flux"}).lower(),
            "repair_mode": "false",
            "run_mode": "async",
            "engine_mode": engine_mode,
            "seed": str(int(seed)),
            "deterministic": str(bool(deterministic)).lower(),
            "output_width": str(int(output_width)),
            "output_height": str(int(output_height)),
            "steps": str(int(steps)),
            "save_intermediates": "true",
        }
        files = {
            "person_image": ("person.png", _encode_png(_tensor_to_pil(person_image)), "image/png"),
            garment_field: ("garment.png", _encode_png(_tensor_to_pil(garment_image)), "image/png"),
        }

        try:
            created = _multipart_post(f"{api_base}/tryon", fields, files, timeout=30)
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Backend API is unreachable at {api_base}: {exc}") from exc

        job_id = created.get("job_id")
        if not job_id:
            raise RuntimeError(f"Backend did not return a job_id: {created}")

        deadline = time.time() + int(timeout_s)
        status = created
        while time.time() < deadline:
            status = _get_json(f"{api_base}/tryon/{job_id}", timeout=30)
            if status.get("status") in {"completed", "failed", "cancelled"}:
                break
            time.sleep(float(poll_interval_s))
        else:
            raise TimeoutError(f"Backend job {job_id} did not finish in {timeout_s}s.")

        if status.get("status") != "completed":
            raise RuntimeError(f"Backend job {job_id} ended as {status.get('status')}: {status.get('error')}")

        result_url = status.get("result_url")
        if not result_url:
            raise RuntimeError(f"Backend job {job_id} completed without result_url.")
        result = _fetch_backend_image(api_base, result_url, timeout=60)

        debug = status.get("debug") or {}
        mask_url = debug.get("mask_url")
        mask_urls = debug.get("mask_urls") or []
        if not mask_url and mask_urls:
            mask_url = mask_urls[0]
        if not mask_url:
            mask_url = debug.get("refine_mask_url")
        if mask_url:
            mask_preview = _fetch_backend_image(api_base, mask_url, timeout=60)
        else:
            mask_preview = Image.new("RGB", result.size, "black")

        payload = {
            "node": "VTON Phase2 - Backend Try-On API",
            "job_id": job_id,
            "status": status.get("status"),
            "api_base": api_base,
            "category": category,
            "engine_mode": engine_mode,
            "output_width": int(output_width),
            "output_height": int(output_height),
            "steps": int(steps),
            "seed": int(seed),
            "deterministic": bool(deterministic),
            "result_url": result_url,
            "mask_url": mask_url,
            "runtime_seconds": round(time.perf_counter() - started, 3),
            "backend_runtime_seconds": status.get("runtime_seconds"),
            "stages": status.get("stages", []),
            "engine_status": status.get("engine_status", {}),
        }
        return (_pil_to_tensor(result), _pil_to_tensor(mask_preview), json.dumps(payload, indent=2, ensure_ascii=False))


class VTONPhase2IDMRun:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "person_image": ("IMAGE",),
                "garment_image": ("IMAGE",),
                "category": (
                    [
                        "upper_body",
                        "lower_body",
                        "dress",
                        "full_outfit",
                        "men_underwear",
                        "women_underwear",
                        "women_bra",
                    ],
                    {"default": "upper_body"},
                ),
                "mask_expanded": ("BOOLEAN", {"default": False}),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": (
                            "Replace only the target clothing region with the reference garment. "
                            "Preserve face, hair, body shape, pose, hands, legs, and background."
                        ),
                    },
                ),
                "seed": ("INT", {"default": 4242, "min": 0, "max": 2147483647}),
            },
            "optional": {
                "engine_mode": (
                    ["auto", "idm_vton", "idm_mask_expanded", "idm_vton_flux", "idm_mask_expanded_flux"],
                    {"default": "auto"},
                ),
                "output_width": ("INT", {"default": 512, "min": 256, "max": 1536, "step": 8}),
                "output_height": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 8}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 64, "step": 1}),
                "deterministic": ("BOOLEAN", {"default": True}),
                "save_intermediates": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "status")
    FUNCTION = "run"
    CATEGORY = "VTON Phase2"

    def run(
        self,
        person_image,
        garment_image,
        category: str,
        mask_expanded: bool,
        prompt: str,
        seed: int,
        engine_mode: str = "auto",
        output_width: int = 512,
        output_height: int = 768,
        steps: int = 8,
        deterministic: bool = True,
        save_intermediates: bool = True,
    ):
        _ensure_project_imports()
        from app.core.config import load_settings
        from app.preprocessing.image_loader import load_image_from_path
        from app.services.storage_service import StorageService
        from app.services.tryon_pipeline import PipelineRequest, TryOnPipeline

        run_id = f"comfy_idm_{'expanded' if mask_expanded else 'base'}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        run_dir = OUTPUT_ROOT / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        person_path = run_dir / "input_person.png"
        garment_path = run_dir / "input_garment.png"
        _tensor_to_pil(person_image).save(person_path)
        _tensor_to_pil(garment_image).save(garment_path)

        settings = load_settings().model_copy(deep=True)
        settings.storage.outputs_dir = OUTPUT_ROOT
        settings.pipeline.engine = "idm_vton"
        settings.pipeline.save_intermediates = bool(save_intermediates)
        settings.mask_experiments.upper_body_expand_hem.enabled = bool(mask_expanded)

        garment = load_image_from_path(garment_path, max_side=settings.image.max_side)
        resolved_engine_mode = engine_mode
        if resolved_engine_mode == "auto":
            resolved_engine_mode = "idm_mask_expanded" if mask_expanded else "idm_vton"
        uses_refiner = resolved_engine_mode in {"idm_vton_flux", "idm_mask_expanded_flux"}
        request = PipelineRequest(
            job_id=run_id,
            person_image=load_image_from_path(person_path, max_side=settings.image.max_side),
            garment_top=garment if category in {"upper_body", "women_bra"} else None,
            garment_bottom=garment if category in {"lower_body", "men_underwear", "women_underwear"} else None,
            garment_dress=garment if category in {"dress", "full_outfit"} else None,
            category=category,  # type: ignore[arg-type]
            prompt=prompt,
            use_refiner=uses_refiner,
            repair_mode=False,
            seed=int(seed),
            deterministic=bool(deterministic),
            engine_mode=resolved_engine_mode,
            output_width=int(output_width),
            output_height=int(output_height),
            steps=int(steps),
            save_intermediates=bool(save_intermediates),
        )
        started = time.perf_counter()
        response = TryOnPipeline(settings, StorageService(settings.storage)).run(request)
        result_path = OUTPUT_ROOT / run_id / "result.png"
        if not result_path.exists():
            raise FileNotFoundError(f"IDM completed without result image: {result_path}")
        image = Image.open(result_path).convert("RGB")
        payload = {
            "status": response.status,
            "engine": "idm_vton",
            "mask_expanded": bool(mask_expanded),
            "category": category,
            "engine_mode": resolved_engine_mode,
            "use_refiner": uses_refiner,
            "output_width": int(output_width),
            "output_height": int(output_height),
            "steps": int(steps),
            "seed": int(seed),
            "deterministic": bool(deterministic),
            "runtime_seconds": round(time.perf_counter() - started, 3),
            "result_path": str(result_path),
            "run_dir": str(OUTPUT_ROOT / run_id),
        }
        (OUTPUT_ROOT / run_id / "comfy_status.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return (_pil_to_tensor(image), json.dumps(payload, indent=2))


NODE_CLASS_MAPPINGS = {
    "VTONPhase2KleinFitCanvas": VTONPhase2KleinFitCanvas,
    "VTONPhase2KleinBottomCrop": VTONPhase2KleinBottomCrop,
    "VTONPhase2KleinPromptBuilder": VTONPhase2KleinPromptBuilder,
    "VTONPhase2KleinLoadBaseModel": VTONPhase2KleinLoadBaseModel,
    "VTONPhase2KleinLoadTryOnLoRA": VTONPhase2KleinLoadTryOnLoRA,
    "VTONPhase2KleinSamplerDetailed": VTONPhase2KleinSamplerDetailed,
    "VTONPhase2KleinReferenceSet": VTONPhase2KleinReferenceSet,
    "VTONPhase2KleinSampler": VTONPhase2KleinSampler,
    "AddMaskForICLora": AddMaskForICLora,
    "VTONPhase2MaskComposite": VTONPhase2MaskComposite,
    "VTONPhase2LocalSeamRepair": VTONPhase2LocalSeamRepair,
    "VTONPhase2MaskPreviewImage": VTONPhase2MaskPreviewImage,
    "VTONPhase2MaskMorphology": VTONPhase2MaskMorphology,
    "VTONPhase2MaskBBoxCrop": VTONPhase2MaskBBoxCrop,
    "VTONPhase2MaskedPasteBack": VTONPhase2MaskedPasteBack,
    "VTONPhase2FitCanvasWithMeta": VTONPhase2FitCanvasWithMeta,
    "VTONPhase2ExtractFittedCanvasRegion": VTONPhase2ExtractFittedCanvasRegion,
    "VTONPhase2TryOnDebugSheet": VTONPhase2TryOnDebugSheet,
    "VTONPhase2SCHPSAMMask": VTONPhase2SCHPSAMMask,
    "VTONPhase2BackendTryOnAPI": VTONPhase2BackendTryOnAPI,
    "VTONPhase2IDMRun": VTONPhase2IDMRun,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VTONPhase2KleinFitCanvas": "VTON Phase2 - Klein Fit Canvas",
    "VTONPhase2KleinBottomCrop": "VTON Phase2 - Klein Bottom Preserve Crop",
    "VTONPhase2KleinPromptBuilder": "VTON Phase2 - Klein Prompt Builder",
    "VTONPhase2KleinLoadBaseModel": "VTON Phase2 - Load FLUX.2 Klein 9B",
    "VTONPhase2KleinLoadTryOnLoRA": "VTON Phase2 - Load Klein Try-On LoRA",
    "VTONPhase2KleinSamplerDetailed": "VTON Phase2 - Klein Detailed Sampler",
    "VTONPhase2KleinReferenceSet": "VTON Phase2 - Klein Reference Set",
    "VTONPhase2KleinSampler": "VTON Phase2 - Klein Local Sampler",
    "AddMaskForICLora": "Add Mask For IC Lora",
    "VTONPhase2MaskComposite": "VTON Phase2 - Masked Candidate Composite",
    "VTONPhase2LocalSeamRepair": "VTON Phase2 - Local Seam Repair",
    "VTONPhase2MaskPreviewImage": "VTON Phase2 - Mask Preview Image",
    "VTONPhase2MaskMorphology": "VTON Phase2 - Mask Morphology",
    "VTONPhase2MaskBBoxCrop": "VTON Phase2 - Mask BBox Crop",
    "VTONPhase2MaskedPasteBack": "VTON Phase2 - Masked Paste Back",
    "VTONPhase2FitCanvasWithMeta": "VTON Phase2 - Fit Canvas With Meta",
    "VTONPhase2ExtractFittedCanvasRegion": "VTON Phase2 - Extract Fitted Canvas Region",
    "VTONPhase2TryOnDebugSheet": "VTON Phase2 - Try-On Debug Sheet",
    "VTONPhase2SCHPSAMMask": "VTON Phase2 - SCHP/SAM Mask",
    "VTONPhase2BackendTryOnAPI": "VTON Phase2 - Backend Try-On API",
    "VTONPhase2IDMRun": "VTON Phase2 - IDM / IDM Expanded",
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VTONPhase2KleinFitCanvas": "VTON Phase2 - Klein Fit Canvas",
    "VTONPhase2KleinBottomCrop": "VTON Phase2 - Klein Bottom Preserve Crop",
    "VTONPhase2KleinPromptBuilder": "VTON Phase2 - Klein Prompt Builder",
    "VTONPhase2KleinLoadBaseModel": "VTON Phase2 - Load FLUX.2 Klein 9B",
    "VTONPhase2KleinLoadTryOnLoRA": "VTON Phase2 - Load Klein Try-On LoRA",
    "VTONPhase2KleinSamplerDetailed": "VTON Phase2 - Klein Detailed Sampler",
    "VTONPhase2KleinReferenceSet": "VTON Phase2 - Klein Reference Set",
    "VTONPhase2KleinSampler": "VTON Phase2 - Klein Local Sampler",
    "VTONPhase2MaskComposite": "VTON Phase2 - Masked Candidate Composite",
    "VTONPhase2LocalSeamRepair": "VTON Phase2 - Local Seam Repair (not ADetailer)",
    "VTONPhase2MaskPreviewImage": "VTON Phase2 - Mask Preview Image",
    "VTONPhase2MaskMorphology": "VTON Phase2 - Mask Morphology",
    "VTONPhase2MaskBBoxCrop": "VTON Phase2 - Mask BBox Crop",
    "VTONPhase2MaskedPasteBack": "VTON Phase2 - Masked Paste Back",
    "VTONPhase2FitCanvasWithMeta": "VTON Phase2 - Fit Canvas With Meta",
    "VTONPhase2ExtractFittedCanvasRegion": "VTON Phase2 - Extract Fitted Canvas Region",
    "VTONPhase2TryOnDebugSheet": "VTON Phase2 - Try-On Debug Sheet",
    "VTONPhase2SCHPSAMMask": "VTON Phase2 - SCHP/SAM Mask",
    "VTONPhase2IDMRun": "VTON Phase2 - IDM / IDM Expanded",
}
