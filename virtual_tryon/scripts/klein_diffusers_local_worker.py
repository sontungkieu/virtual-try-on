from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


DEFAULT_QUANTIZE_COMPONENTS = ["transformer", "text_encoder"]


def patch_sdpa_enable_gqa() -> None:
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


def fit_canvas(path: Path, size: tuple[int, int]) -> Image.Image:
    image = ImageOps.exif_transpose(Image.open(path).convert("RGB"))
    contained = ImageOps.contain(image, size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(contained, ((size[0] - contained.width) // 2, (size[1] - contained.height) // 2))
    return canvas


def validate_runtime(model_dir: Path | None, lora_path: Path | None) -> dict[str, Any]:
    import importlib.metadata as metadata
    import importlib.util

    import peft  # noqa: F401
    import torch
    from diffusers import Flux2KleinPipeline  # noqa: F401

    payload = {
        "torch": metadata.version("torch"),
        "diffusers": metadata.version("diffusers"),
        "transformers": metadata.version("transformers"),
        "accelerate": metadata.version("accelerate"),
        "peft": metadata.version("peft"),
        "cuda_available": torch.cuda.is_available(),
        "model_index_exists": bool(model_dir and (model_dir / "model_index.json").is_file()),
        "lora_exists": bool(lora_path and lora_path.is_file()),
        "torchao_available": importlib.util.find_spec("torchao") is not None,
        "bitsandbytes_available": importlib.util.find_spec("bitsandbytes") is not None,
    }
    return payload


def normalize_device_map(value: str | None) -> str:
    mode = (value or "cpu_offload").strip().lower().replace("-", "_")
    aliases = {
        "offload": "cpu_offload",
        "model_cpu_offload": "cpu_offload",
        "gpu": "cuda",
        "all_gpu": "cuda",
        "all_cuda": "cuda",
        "full_cuda": "cuda",
        "sequential_offload": "sequential_cpu_offload",
    }
    mode = aliases.get(mode, mode)
    allowed = {"cpu_offload", "sequential_cpu_offload", "cuda", "balanced", "auto"}
    if mode not in allowed:
        raise ValueError(f"Unsupported Klein device_map '{value}'. Expected one of: {', '.join(sorted(allowed))}.")
    return mode


def normalize_quantization(value: str | None) -> str:
    mode = (value or "none").strip().lower().replace("-", "_")
    aliases = {
        "": "none",
        "false": "none",
        "off": "none",
        "no": "none",
        "int8": "torchao_int8",
        "torchao_int8wo": "torchao_int8",
        "torchao_int8_weight_only": "torchao_int8",
        "int4": "torchao_int4",
        "torchao_int4wo": "torchao_int4",
        "torchao_int4_weight_only": "torchao_int4",
        "fp8": "torchao_fp8",
        "float8": "torchao_fp8",
        "bnb4": "bnb_4bit",
        "bitsandbytes_4bit": "bnb_4bit",
        "bnb8": "bnb_8bit",
        "bitsandbytes_8bit": "bnb_8bit",
    }
    mode = aliases.get(mode, mode)
    allowed = {"none", "torchao_int8", "torchao_int4", "torchao_fp8", "bnb_4bit", "bnb_8bit"}
    if mode not in allowed:
        raise ValueError(f"Unsupported Klein quantization '{value}'. Expected one of: {', '.join(sorted(allowed))}.")
    return mode


def normalize_components(value: Any) -> list[str]:
    if value is None:
        return list(DEFAULT_QUANTIZE_COMPONENTS)
    if isinstance(value, str):
        parts = [item.strip() for item in value.split(",")]
    else:
        parts = [str(item).strip() for item in value]
    components = [item for item in parts if item]
    return components or list(DEFAULT_QUANTIZE_COMPONENTS)


def build_quantization_config(mode: str, components: list[str]):
    if mode == "none":
        return None

    from diffusers import PipelineQuantizationConfig

    if mode.startswith("torchao_"):
        from diffusers import TorchAoConfig

        try:
            import torchao.quantization as aq
        except ImportError as exc:
            raise RuntimeError(f"Klein quantization '{mode}' requires the torchao package.") from exc

        if mode == "torchao_int8":
            factory = getattr(aq, "Int8WeightOnlyConfig", None)
        elif mode == "torchao_int4":
            factory = getattr(aq, "Int4WeightOnlyConfig", None)
        elif mode == "torchao_fp8":
            factory = getattr(aq, "Float8WeightOnlyConfig", None) or getattr(
                aq,
                "Float8DynamicActivationFloat8WeightConfig",
                None,
            )
        else:
            factory = None
        if factory is None:
            raise RuntimeError(f"Installed torchao does not expose a config class for '{mode}'.")
        return PipelineQuantizationConfig(
            quant_mapping={component: TorchAoConfig(factory()) for component in components}
        )

    if mode.startswith("bnb_"):
        from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
        from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
        import torch

        try:
            import bitsandbytes  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(f"Klein quantization '{mode}' requires the bitsandbytes package.") from exc

        def make_config(component: str):
            cls = TransformersBitsAndBytesConfig if component == "text_encoder" else DiffusersBitsAndBytesConfig
            if mode == "bnb_4bit":
                return cls(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            return cls(load_in_8bit=True)

        return PipelineQuantizationConfig(quant_mapping={component: make_config(component) for component in components})

    raise ValueError(f"Unsupported Klein quantization '{mode}'.")


def load_pipe(
    model_dir: Path,
    lora_path: Path,
    lora_scale: float,
    *,
    device_map: str,
    quantization: str,
    quantize_components: list[str],
):
    import torch
    from diffusers import Flux2KleinPipeline

    device_map = normalize_device_map(device_map)
    quantization = normalize_quantization(quantization)
    quantize_components = normalize_components(quantize_components)
    quantization_config = build_quantization_config(quantization, quantize_components)

    patch_sdpa_enable_gqa()
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    from_pretrained_kwargs = {
        "torch_dtype": torch.bfloat16,
        "local_files_only": True,
        "low_cpu_mem_usage": True,
    }
    if quantization_config is not None:
        from_pretrained_kwargs["quantization_config"] = quantization_config
    if device_map in {"balanced", "auto"}:
        from_pretrained_kwargs["device_map"] = device_map
    try:
        pipe = Flux2KleinPipeline.from_pretrained(model_dir, **from_pretrained_kwargs)
    except TypeError:
        from_pretrained_kwargs.pop("low_cpu_mem_usage", None)
        pipe = Flux2KleinPipeline.from_pretrained(model_dir, **from_pretrained_kwargs)
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
    if device_map == "cpu_offload" and torch.cuda.is_available() and hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload(gpu_id=0)
    elif device_map == "sequential_cpu_offload" and torch.cuda.is_available() and hasattr(pipe, "enable_sequential_cpu_offload"):
        pipe.enable_sequential_cpu_offload(gpu_id=0)
    elif device_map == "cuda" and torch.cuda.is_available():
        pipe.to("cuda")
    elif device_map == "cuda":
        raise RuntimeError("Klein device_map='cuda' requires CUDA.")
    pipe.load_lora_weights(
        lora_path.parent,
        weight_name=lora_path.name,
        adapter_name="tryon",
        local_files_only=True,
    )
    if hasattr(pipe, "set_adapters"):
        pipe.set_adapters(["tryon"], adapter_weights=[float(lora_scale)])
    return pipe


def run_request(request_path: Path, output_dir: Path) -> dict[str, Any]:
    import torch

    request = json.loads(request_path.read_text(encoding="utf-8"))
    model_dir = Path(request["model_dir"])
    lora_path = Path(request["lora_path"])
    width = int(request["width"])
    height = int(request["height"])
    seed = int(request.get("seed") or 0)
    device_map = normalize_device_map(request.get("device_map"))
    quantization = normalize_quantization(request.get("quantization"))
    quantize_components = normalize_components(request.get("quantize_components"))
    references = [fit_canvas(Path(path), (width, height)) for path in request["image_paths"]]

    started = time.perf_counter()
    pipe = load_pipe(
        model_dir,
        lora_path,
        float(request.get("lora_scale", 1.0)),
        device_map=device_map,
        quantization=quantization,
        quantize_components=quantize_components,
    )
    generator_device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(seed)
    output = pipe(
        image=references,
        prompt=request["prompt"],
        height=height,
        width=width,
        num_inference_steps=int(request["steps"]),
        guidance_scale=float(request["guidance_scale"]),
        generator=generator,
    )
    image = output.images[0].convert("RGB")
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "klein_lora_result.png"
    image.save(result_path)
    payload = {
        "status": "completed",
        "result_path": str(result_path),
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "width": width,
        "height": height,
        "steps": int(request["steps"]),
        "guidance_scale": float(request["guidance_scale"]),
        "seed": seed,
        "model_dir": str(model_dir),
        "lora_path": str(lora_path),
        "lora_scale": float(request.get("lora_scale", 1.0)),
        "device_map": device_map,
        "quantization": quantization,
        "quantize_components": quantize_components,
        "cuda_max_memory_allocated_mb": (
            round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2)
            if torch.cuda.is_available()
            else None
        ),
        "cuda_max_memory_reserved_mb": (
            round(torch.cuda.max_memory_reserved() / 1024 / 1024, 2)
            if torch.cuda.is_available()
            else None
        ),
    }
    (output_dir / "worker_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local FLUX.2 Klein Try-On LoRA generation.")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--lora-path", type=Path, default=None)
    parser.add_argument("--request", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.check:
            print(json.dumps(validate_runtime(args.model_dir, args.lora_path), indent=2), flush=True)
            return 0
        if not args.request or not args.output_dir:
            raise ValueError("--request and --output-dir are required unless --check is used.")
        print(json.dumps(run_request(args.request, args.output_dir), indent=2), flush=True)
        return 0
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
