from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


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
    }
    return payload


def load_pipe(model_dir: Path, lora_path: Path, lora_scale: float):
    import torch
    from diffusers import Flux2KleinPipeline

    patch_sdpa_enable_gqa()
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    try:
        pipe = Flux2KleinPipeline.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
    except TypeError:
        pipe = Flux2KleinPipeline.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
    if torch.cuda.is_available() and hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload(gpu_id=0)
    elif torch.cuda.is_available():
        pipe.to("cuda")
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
    references = [fit_canvas(Path(path), (width, height)) for path in request["image_paths"]]

    started = time.perf_counter()
    pipe = load_pipe(model_dir, lora_path, float(request.get("lora_scale", 1.0)))
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
