from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUANTIZE_COMPONENTS = ["transformer", "text_encoder"]
DEFAULT_TENSORRT_CACHE_DIR = PROJECT_ROOT / "data" / "temp" / "klein_tensorrt_cache"
RESIDENT_PROTOCOL_PREFIX = "__KLEIN_TRYON_WORKER__ "
_PIPE_CACHE: dict[str, Any] = {}
TENSORRT_PROFILE_ALIASES = {
    "": "none",
    "off": "none",
    "false": "none",
    "no": "none",
    "disabled": "none",
    "default": "none",
    "safe": "vae_decode",
    "stable": "vae_decode",
    "vae": "vae_decode",
    "vae.decode": "vae_decode",
    "vae_decoder": "vae_decode",
    "transformer": "transformer_debug",
    "full": "full_debug",
    "all": "full_debug",
}
TENSORRT_COMPONENT_ALIASES = {
    "vae": "vae_decode",
    "vae.decode": "vae_decode",
    "vae_decoder": "vae_decode",
    "diffusion_transformer": "transformer",
}
TENSORRT_PROFILE_COMPONENTS = {
    "none": [],
    "vae_decode": ["vae_decode"],
    "transformer_debug": ["transformer"],
    "full_debug": ["transformer", "vae_decode"],
}


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


def configure_torch_determinism(seed: int, deterministic: bool) -> None:
    import torch

    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(bool(deterministic), warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(bool(deterministic))
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = not bool(deterministic)
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = not bool(deterministic)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
        "torchao_available": importlib.util.find_spec("torchao") is not None,
        "bitsandbytes_available": importlib.util.find_spec("bitsandbytes") is not None,
        "torch_tensorrt_available": importlib.util.find_spec("torch_tensorrt") is not None,
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


def normalize_tensorrt_profile(value: str | None) -> str:
    mode = (value or "none").strip().lower().replace("-", "_")
    mode = TENSORRT_PROFILE_ALIASES.get(mode, mode)
    if mode not in TENSORRT_PROFILE_COMPONENTS:
        valid = sorted(set(TENSORRT_PROFILE_COMPONENTS) | set(TENSORRT_PROFILE_ALIASES))
        raise ValueError(f"Unsupported Klein TensorRT profile '{value}'. Expected one of: {', '.join(valid)}.")
    return mode


def normalize_tensorrt_components(profile: str, value: Any) -> list[str]:
    profile = normalize_tensorrt_profile(profile)
    if value is None:
        return list(TENSORRT_PROFILE_COMPONENTS[profile])
    if isinstance(value, str):
        parts = [item.strip() for item in value.split(",")]
    else:
        parts = [str(item).strip() for item in value]
    components = [TENSORRT_COMPONENT_ALIASES.get(item, item) for item in parts if item]
    if not components:
        return list(TENSORRT_PROFILE_COMPONENTS[profile])
    allowed = {"vae_decode", "transformer"}
    invalid = sorted(set(components) - allowed)
    if invalid:
        raise ValueError(
            "Unsupported Klein TensorRT component(s): "
            + ", ".join(invalid)
            + f". Expected one of: {', '.join(sorted(allowed))}."
        )
    return components


def resolve_tensorrt_cache_dir(value: str | Path | None) -> Path:
    cache_dir = Path(value) if value else DEFAULT_TENSORRT_CACHE_DIR
    if not cache_dir.is_absolute():
        cache_dir = PROJECT_ROOT / cache_dir
    return cache_dir.resolve()


def validate_tensorrt_request(
    profile: str,
    components: list[str],
    *,
    quantization: str,
    quantize_components: list[str],
) -> None:
    if profile == "none" or not components:
        return
    if "transformer" in components and quantization != "none" and "transformer" in quantize_components:
        raise RuntimeError(
            "Klein TensorRT transformer/full profiles are incompatible with quantized transformer weights. "
            "Torch-TensorRT fails on the bitsandbytes UInt8 transformer graph. Use "
            "TRYON_KLEIN_TRT_PROFILE=vae_decode with bnb_4bit, or disable transformer quantization only for "
            "debug builds that fit in VRAM."
        )


def _safe_tensorrt_options(options: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in options.items():
        if key == "enabled_precisions":
            safe[key] = sorted(str(item) for item in value)
        else:
            safe[key] = value
    return safe


def _tensorrt_options(cache_dir: Path, min_block_size: int) -> dict[str, Any]:
    import torch

    cache_dir.mkdir(parents=True, exist_ok=True)
    return {
        "enabled_precisions": {torch.float16},
        "truncate_double": True,
        "require_full_compilation": False,
        "min_block_size": min_block_size,
        "workspace_size": 0,
        "use_fast_partitioner": True,
        "pass_through_build_failures": False,
        "cache_built_engines": True,
        "reuse_cached_engines": True,
        "engine_cache_dir": str(cache_dir),
        "timing_cache_path": str(cache_dir / "timing_cache.bin"),
        "runtime_cache_path": str(cache_dir / "runtime_cache.bin"),
    }


def _compile_with_tensorrt(module: Any, name: str, options: dict[str, Any]):
    import torch
    import torch_tensorrt  # noqa: F401

    print(
        f"Compiling Klein {name} with Torch-TensorRT options={_safe_tensorrt_options(options)}",
        file=sys.stderr,
        flush=True,
    )
    return torch.compile(
        module,
        backend="tensorrt",
        fullgraph=False,
        dynamic=False,
        options=options,
    )


def apply_klein_tensorrt_optimization(
    pipe: Any,
    *,
    profile: str,
    components: list[str] | None = None,
    quantization: str,
    quantize_components: list[str],
    engine_cache_dir: str | Path | None = None,
    min_block_size: int | None = None,
) -> dict[str, Any]:
    import torch

    profile = normalize_tensorrt_profile(profile)
    components = normalize_tensorrt_components(profile, components)
    validate_tensorrt_request(
        profile,
        components,
        quantization=normalize_quantization(quantization),
        quantize_components=normalize_components(quantize_components),
    )
    if profile == "none" or not components:
        metadata = {
            "tensorrt_profile": "none",
            "tensorrt_components": [],
            "tensorrt_compile_setup_seconds": 0.0,
            "tensorrt_engine_cache_dir": None,
            "tensorrt_min_block_size": None,
        }
        setattr(pipe, "_vton_tensorrt_metadata", metadata)
        return metadata
    if not torch.cuda.is_available():
        raise RuntimeError("Klein TensorRT requires CUDA.")
    if importlib.util.find_spec("torch_tensorrt") is None:
        raise RuntimeError("Klein TensorRT requires torch_tensorrt in TRYON_KLEIN_PYTHON.")

    cache_dir = resolve_tensorrt_cache_dir(engine_cache_dir)
    min_block_size = int(min_block_size or 5)
    options = _tensorrt_options(cache_dir, min_block_size)
    started = time.perf_counter()
    compiled_components: list[str] = []

    if "transformer" in components:
        if not hasattr(pipe, "transformer"):
            raise RuntimeError("Klein pipeline does not expose a transformer module for TensorRT.")
        pipe.transformer = _compile_with_tensorrt(pipe.transformer, "transformer", options)
        compiled_components.append("transformer")
    if "vae_decode" in components:
        if not hasattr(pipe, "vae") or not hasattr(pipe.vae, "decode"):
            raise RuntimeError("Klein pipeline does not expose vae.decode for TensorRT.")
        pipe.vae.decode = _compile_with_tensorrt(pipe.vae.decode, "vae.decode", options)
        compiled_components.append("vae_decode")

    metadata = {
        "tensorrt_profile": profile,
        "tensorrt_components": compiled_components,
        "tensorrt_compile_setup_seconds": round(time.perf_counter() - started, 3),
        "tensorrt_engine_cache_dir": str(cache_dir),
        "tensorrt_min_block_size": min_block_size,
        "tensorrt_options": _safe_tensorrt_options(options),
    }
    setattr(pipe, "_vton_tensorrt_metadata", metadata)
    return metadata


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
    tensorrt_profile: str = "none",
    tensorrt_components: list[str] | None = None,
    tensorrt_engine_cache_dir: str | Path | None = None,
    tensorrt_min_block_size: int | None = None,
):
    import torch
    from diffusers import Flux2KleinPipeline

    device_map = normalize_device_map(device_map)
    quantization = normalize_quantization(quantization)
    quantize_components = normalize_components(quantize_components)
    tensorrt_profile = normalize_tensorrt_profile(tensorrt_profile)
    tensorrt_components = normalize_tensorrt_components(tensorrt_profile, tensorrt_components)
    validate_tensorrt_request(
        tensorrt_profile,
        tensorrt_components,
        quantization=quantization,
        quantize_components=quantize_components,
    )
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
    apply_klein_tensorrt_optimization(
        pipe,
        profile=tensorrt_profile,
        components=tensorrt_components,
        quantization=quantization,
        quantize_components=quantize_components,
        engine_cache_dir=tensorrt_engine_cache_dir,
        min_block_size=tensorrt_min_block_size,
    )
    return pipe


def _pipe_cache_key(request: dict[str, Any]) -> str:
    return json.dumps(
        {
            "model_dir": request["model_dir"],
            "lora_path": request["lora_path"],
            "lora_scale": float(request.get("lora_scale", 1.0)),
            "device_map": normalize_device_map(request.get("device_map")),
            "quantization": normalize_quantization(request.get("quantization")),
            "quantize_components": normalize_components(request.get("quantize_components")),
            "tensorrt_profile": normalize_tensorrt_profile(request.get("tensorrt_profile")),
            "tensorrt_components": normalize_tensorrt_components(
                normalize_tensorrt_profile(request.get("tensorrt_profile")),
                request.get("tensorrt_components"),
            ),
            "tensorrt_engine_cache_dir": request.get("tensorrt_engine_cache_dir"),
            "tensorrt_min_block_size": request.get("tensorrt_min_block_size"),
        },
        sort_keys=True,
    )


def _emit_progress(progress: Any, stage: str, status: str, **payload: Any) -> None:
    if progress is None:
        return
    progress({"stage": stage, "status": status, **payload})


def load_cached_pipe(request: dict[str, Any], progress: Any = None):
    key = _pipe_cache_key(request)
    cached = _PIPE_CACHE.get(key)
    if cached is not None:
        _emit_progress(progress, "loading_model", "skipped", cached=True)
        return cached, True, 0.0

    model_dir = Path(request["model_dir"])
    lora_path = Path(request["lora_path"])
    quantization = normalize_quantization(request.get("quantization"))
    quantize_components = normalize_components(request.get("quantize_components"))
    tensorrt_profile = normalize_tensorrt_profile(request.get("tensorrt_profile"))
    tensorrt_components = normalize_tensorrt_components(tensorrt_profile, request.get("tensorrt_components"))
    validate_tensorrt_request(
        tensorrt_profile,
        tensorrt_components,
        quantization=quantization,
        quantize_components=quantize_components,
    )
    _emit_progress(progress, "loading_model", "running", cached=False)
    started = time.perf_counter()
    pipe = load_pipe(
        model_dir,
        lora_path,
        float(request.get("lora_scale", 1.0)),
        device_map=normalize_device_map(request.get("device_map")),
        quantization=quantization,
        quantize_components=quantize_components,
        tensorrt_profile=tensorrt_profile,
        tensorrt_components=tensorrt_components,
        tensorrt_engine_cache_dir=request.get("tensorrt_engine_cache_dir"),
        tensorrt_min_block_size=request.get("tensorrt_min_block_size"),
    )
    load_seconds = time.perf_counter() - started
    _PIPE_CACHE.clear()
    _PIPE_CACHE[key] = pipe
    _emit_progress(progress, "loading_model", "completed", cached=False, runtime_seconds=round(load_seconds, 3))
    return pipe, False, load_seconds


def prepare_payload(request: dict[str, Any], progress: Any = None) -> dict[str, Any]:
    pipe, cached, load_seconds = load_cached_pipe(request, progress=progress)
    return {
        "status": "ready",
        "cached": cached,
        "load_model_seconds": round(load_seconds, 3),
        "model_dir": request["model_dir"],
        "lora_path": request["lora_path"],
        "device_map": normalize_device_map(request.get("device_map")),
        "quantization": normalize_quantization(request.get("quantization")),
        "quantize_components": normalize_components(request.get("quantize_components")),
        **getattr(pipe, "_vton_tensorrt_metadata", {}),
    }


def run_payload(request: dict[str, Any], output_dir: Path, progress: Any = None) -> dict[str, Any]:
    import torch

    width = int(request["width"])
    height = int(request["height"])
    seed = int(request.get("seed") or 0)
    deterministic = bool(request.get("deterministic", False))
    references = [fit_canvas(Path(path), (width, height)) for path in request["image_paths"]]
    configure_torch_determinism(seed, deterministic)

    started = time.perf_counter()
    pipe, cached, load_seconds = load_cached_pipe(request, progress=progress)
    generator_device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(seed)
    _emit_progress(progress, "generating", "running")
    generation_started = time.perf_counter()
    output = pipe(
        image=references,
        prompt=request["prompt"],
        height=height,
        width=width,
        num_inference_steps=int(request["steps"]),
        guidance_scale=float(request["guidance_scale"]),
        generator=generator,
    )
    generation_seconds = time.perf_counter() - generation_started
    _emit_progress(progress, "generating", "completed", runtime_seconds=round(generation_seconds, 3))
    image = output.images[0].convert("RGB")
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "klein_lora_result.png"
    image.save(result_path)
    payload = {
        "status": "completed",
        "result_path": str(result_path),
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "load_model_seconds": round(load_seconds, 3),
        "generation_seconds": round(generation_seconds, 3),
        "pipe_cached": cached,
        "width": width,
        "height": height,
        "steps": int(request["steps"]),
        "guidance_scale": float(request["guidance_scale"]),
        "seed": seed,
        "deterministic": deterministic,
        "model_dir": str(request["model_dir"]),
        "lora_path": str(request["lora_path"]),
        "lora_scale": float(request.get("lora_scale", 1.0)),
        "device_map": normalize_device_map(request.get("device_map")),
        "quantization": normalize_quantization(request.get("quantization")),
        "quantize_components": normalize_components(request.get("quantize_components")),
        **getattr(pipe, "_vton_tensorrt_metadata", {}),
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


def run_request(request_path: Path, output_dir: Path) -> dict[str, Any]:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    return run_payload(request, output_dir)


def _protocol_message(payload: dict[str, Any]) -> None:
    print(RESIDENT_PROTOCOL_PREFIX + json.dumps(payload, separators=(",", ":"), default=str), flush=True)


def resident_loop() -> int:
    _protocol_message({"type": "ready", "ok": True})
    for line in sys.stdin:
        if not line.strip():
            continue
        request_id = None
        try:
            message = json.loads(line)
            request_id = message.get("request_id")
            message_type = message.get("type")
            if message_type == "shutdown":
                _protocol_message({"type": "shutdown", "ok": True, "request_id": request_id})
                return 0
            payload = message.get("payload") or {}

            def progress(event: dict[str, Any]) -> None:
                _protocol_message({"type": "progress", "ok": True, "request_id": request_id, **event})

            if message_type == "prepare":
                result = prepare_payload(payload, progress=progress)
            elif message_type == "run":
                output_dir = Path(message["output_dir"])
                result = run_payload(payload, output_dir, progress=progress)
            else:
                raise ValueError(f"unsupported resident request type: {message_type}")
            _protocol_message({"type": "result", "ok": True, "request_id": request_id, "result": result})
        except Exception as exc:
            _protocol_message({
                "type": "result",
                "ok": False,
                "request_id": request_id,
                "error": f"{type(exc).__name__}: {exc}",
            })
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local FLUX.2 Klein Try-On LoRA generation.")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--resident", action="store_true")
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--lora-path", type=Path, default=None)
    parser.add_argument("--request", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.resident:
            return resident_loop()
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
