#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any


PROTOCOL_PREFIX = "__IDM_VTON_WORKER__ "
NEGATIVE_PROMPT = "monochrome, lowres, bad anatomy, worst quality, low quality"
DEFAULT_TENSORRT_MODULES = ("vae_decode",)
FULL_TENSORRT_MODULES = ("unet_blocks", "unet_encoder_blocks", "vae_decode")
VALID_TENSORRT_MODULES = (
    "unet",
    "unet_encoder",
    "unet_blocks",
    "unet_encoder_blocks",
    "vae_decode",
)
UNSAFE_TENSORRT_UNET_MODULES = {"unet", "unet_encoder", "unet_blocks", "unet_encoder_blocks"}
TENSORRT_MODULE_ALIASES = {
    "vae": "vae_decode",
    "vae.decode": "vae_decode",
    "stable": "vae_decode",
    "safe": "vae_decode",
    "safe_unet": "unet_blocks",
    "safe_unet_encoder": "unet_encoder_blocks",
}
TENSORRT_OP_PRESETS = {
    "none": (),
    "conv": (
        "aten.convolution.default",
    ),
    "attention": (
        "aten._scaled_dot_product_flash_attention.default",
        "aten._scaled_dot_product_efficient_attention.default",
        "aten.scaled_dot_product_attention.default",
    ),
    "shape": (
        "aten._reshape_copy.default",
        "aten.reshape.default",
        "aten.view.default",
        "aten.permute.default",
        "aten.transpose.int",
    ),
    "matmul": (
        "aten.addmm.default",
        "aten.mm.default",
        "aten.bmm.default",
    ),
    "norm": (
        "aten.native_layer_norm.default",
        "aten.native_group_norm.default",
    ),
    "safe_unet": (
        "aten.convolution.default",
        "aten._scaled_dot_product_flash_attention.default",
        "aten._scaled_dot_product_efficient_attention.default",
        "aten.scaled_dot_product_attention.default",
        "aten._reshape_copy.default",
        "aten.reshape.default",
        "aten.view.default",
        "aten.permute.default",
        "aten.transpose.int",
        "aten.addmm.default",
        "aten.mm.default",
        "aten.bmm.default",
        "aten.native_layer_norm.default",
        "aten.native_group_norm.default",
    ),
}


def _emit(payload: dict[str, Any], output) -> None:
    print(PROTOCOL_PREFIX + json.dumps(payload, separators=(",", ":"), default=str), file=output, flush=True)


@contextlib.contextmanager
def _redirect_noisy_stdout(protocol_output):
    original_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = original_stdout


class ResidentIDMVTonPipeline:
    def __init__(
        self,
        repo_path: Path,
        model_name: str,
        device: str,
        optimization_mode: str = "eager",
        torch_compile_backend: str = "inductor",
        torch_compile_mode: str = "reduce-overhead",
    ) -> None:
        self.repo_path = repo_path.resolve()
        self.model_name = model_name
        self.optimization_mode = optimization_mode
        self.torch_compile_backend = torch_compile_backend
        self.torch_compile_mode = torch_compile_mode
        self._requested_tensorrt_modules = self._tensorrt_modules() if optimization_mode == "tensorrt" else set()
        sys.path.insert(0, str(self.repo_path))
        os.chdir(self.repo_path)

        import torch
        import torch.utils.data
        import torchvision
        from diffusers import AutoencoderKL, DDPMScheduler
        from inference import VitonHDTestDataset, pil_to_tensor
        from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
        from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModelRef
        from src.unet_hacked_tryon import UNet2DConditionModel
        from transformers import (
            AutoTokenizer,
            CLIPImageProcessor,
            CLIPTextModel,
            CLIPTextModelWithProjection,
            CLIPVisionModelWithProjection,
        )

        self.torch = torch
        self.torchvision = torchvision
        self.VitonHDTestDataset = VitonHDTestDataset
        self.pil_to_tensor = pil_to_tensor
        self.weight_dtype = torch.float16
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)

        noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")
        vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae", torch_dtype=self.weight_dtype)
        unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet", torch_dtype=self.weight_dtype)
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            model_name,
            subfolder="image_encoder",
            torch_dtype=self.weight_dtype,
        )
        unet_encoder = UNet2DConditionModelRef.from_pretrained(
            model_name,
            subfolder="unet_encoder",
            torch_dtype=self.weight_dtype,
        )
        text_encoder_one = CLIPTextModel.from_pretrained(
            model_name,
            subfolder="text_encoder",
            torch_dtype=self.weight_dtype,
        )
        text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
            model_name,
            subfolder="text_encoder_2",
            torch_dtype=self.weight_dtype,
        )
        tokenizer_one = AutoTokenizer.from_pretrained(
            model_name,
            subfolder="tokenizer",
            revision=None,
            use_fast=False,
        )
        tokenizer_two = AutoTokenizer.from_pretrained(
            model_name,
            subfolder="tokenizer_2",
            revision=None,
            use_fast=False,
        )

        for module in [unet, vae, image_encoder, unet_encoder, text_encoder_one, text_encoder_two]:
            module.requires_grad_(False)
        unet.eval()
        unet_encoder.eval()
        unet_encoder.to(self.device, self.weight_dtype)

        self.pipe = TryonPipeline.from_pretrained(
            model_name,
            unet=unet,
            vae=vae,
            feature_extractor=CLIPImageProcessor(),
            text_encoder=text_encoder_one,
            text_encoder_2=text_encoder_two,
            tokenizer=tokenizer_one,
            tokenizer_2=tokenizer_two,
            scheduler=noise_scheduler,
            image_encoder=image_encoder,
            unet_encoder=unet_encoder,
            torch_dtype=self.weight_dtype,
        ).to(self.device)
        self._apply_optimization()

    def _apply_optimization(self) -> None:
        if self.optimization_mode == "eager":
            return
        if self.optimization_mode == "torch_compile":
            if not hasattr(self.torch, "compile"):
                raise RuntimeError("torch.compile is not available in this PyTorch runtime.")
            self.pipe.unet = self.torch.compile(
                self.pipe.unet,
                backend=self.torch_compile_backend,
                mode=self.torch_compile_mode,
                fullgraph=False,
            )
            if hasattr(self.pipe, "unet_encoder"):
                self.pipe.unet_encoder = self.torch.compile(
                    self.pipe.unet_encoder,
                    backend=self.torch_compile_backend,
                    mode=self.torch_compile_mode,
                    fullgraph=False,
                )
            return
        if self.optimization_mode == "tensorrt":
            try:
                import torch_tensorrt  # noqa: F401
            except Exception as exc:
                raise RuntimeError("torch_tensorrt is not installed in this runtime.") from exc
            self._apply_tensorrt_optimization()
            return
        raise RuntimeError(f"Unsupported optimization mode: {self.optimization_mode}")

    def _tensorrt_options(self) -> dict[str, Any]:
        cache_dir = Path(
            os.environ.get(
                "TRYON_TRT_ENGINE_CACHE_DIR",
                "/tmp/torch_tensorrt_engine_cache/idm_vton",
            )
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        min_block_size = int(os.environ.get("TRYON_TRT_MIN_BLOCK_SIZE", "5"))
        workspace_size = int(os.environ.get("TRYON_TRT_WORKSPACE_SIZE", "0"))
        optimization_level_text = os.environ.get("TRYON_TRT_OPTIMIZATION_LEVEL")
        optimization_level = int(optimization_level_text) if optimization_level_text else None
        cpu_memory_budget_text = os.environ.get("TRYON_TRT_CPU_MEMORY_BUDGET")
        cpu_memory_budget = int(cpu_memory_budget_text) if cpu_memory_budget_text else None
        options: dict[str, Any] = {
            "enabled_precisions": {self.torch.float16},
            "truncate_double": True,
            "require_full_compilation": False,
            "min_block_size": min_block_size,
            "workspace_size": workspace_size,
            "use_fast_partitioner": self._bool_env("TRYON_TRT_USE_FAST_PARTITIONER", True),
            "pass_through_build_failures": self._bool_env("TRYON_TRT_PASS_THROUGH_BUILD_FAILURES", False),
            "cache_built_engines": True,
            "reuse_cached_engines": True,
            "engine_cache_dir": str(cache_dir),
            "timing_cache_path": str(cache_dir / "timing_cache.bin"),
            "runtime_cache_path": str(cache_dir / "runtime_cache.bin"),
            "torch_executed_ops": self._tensorrt_torch_executed_ops(),
        }
        if optimization_level is not None:
            options["optimization_level"] = optimization_level
        if self._bool_env("TRYON_TRT_ENABLE_RESOURCE_PARTITIONING", False):
            options["enable_resource_partitioning"] = True
        if self._bool_env("TRYON_TRT_LAZY_ENGINE_INIT", False):
            options["lazy_engine_init"] = True
        if cpu_memory_budget is not None:
            options["cpu_memory_budget"] = cpu_memory_budget
        return options

    def _compile_module_with_tensorrt(self, module: Any, name: str, options: dict[str, Any]) -> Any:
        print(f"Compiling {name} with Torch-TensorRT backend options={self._safe_tensorrt_options(options)}", file=sys.stderr)
        return self.torch.compile(
            module,
            backend="tensorrt",
            fullgraph=False,
            dynamic=False,
            options=options,
        )

    @staticmethod
    def _safe_tensorrt_options(options: dict[str, Any]) -> dict[str, Any]:
        safe: dict[str, Any] = {}
        for key, value in options.items():
            if key == "enabled_precisions":
                safe[key] = sorted(str(item) for item in value)
            elif key == "torch_executed_ops":
                safe[key] = sorted(str(item) for item in value)
            else:
                safe[key] = value
        return safe

    @staticmethod
    def _bool_env(name: str, default: bool = False) -> bool:
        value = os.environ.get(name)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "on"}

    def _tensorrt_op_target(self, name: str) -> Any:
        if name.startswith("aten."):
            path = name.removeprefix("aten.").split(".")
            if len(path) != 2:
                raise RuntimeError(f"Invalid TensorRT torch-executed op: {name}")
            packet_name, overload = path
            packet = getattr(self.torch.ops.aten, packet_name, None)
            if packet is None or not hasattr(packet, overload):
                raise RuntimeError(f"Unknown TensorRT torch-executed op: {name}")
            return getattr(packet, overload)
        raise RuntimeError(f"Invalid TensorRT torch-executed op: {name}. Use aten.<op>.<overload>.")

    def _tensorrt_torch_executed_ops(self) -> set[Any]:
        preset_text = os.environ.get("TRYON_TRT_PARTITION_PRESET", "attention")
        op_text = os.environ.get("TRYON_TRT_TORCH_EXECUTED_OPS")
        names: list[str] = []
        if preset_text:
            for preset in [item.strip().lower() for item in preset_text.split(",") if item.strip()]:
                if preset not in TENSORRT_OP_PRESETS:
                    raise RuntimeError(
                        "Invalid TRYON_TRT_PARTITION_PRESET value: "
                        + preset
                        + f". Valid values: {', '.join(sorted(TENSORRT_OP_PRESETS))}."
                    )
                names.extend(TENSORRT_OP_PRESETS[preset])
        if op_text:
            names.extend(item.strip() for item in op_text.split(",") if item.strip())
        return {self._tensorrt_op_target(name) for name in names}

    @staticmethod
    def _tensorrt_modules() -> set[str]:
        raw = os.environ.get("TRYON_TRT_MODULES", ",".join(DEFAULT_TENSORRT_MODULES))
        requested = {item.strip().lower() for item in raw.split(",") if item.strip()}
        if not requested or requested == {"none"}:
            return set()
        if "all" in requested:
            requested = set(FULL_TENSORRT_MODULES)
        requested = {TENSORRT_MODULE_ALIASES.get(item, item) for item in requested}
        valid = set(VALID_TENSORRT_MODULES)
        invalid = sorted(requested - valid)
        if invalid:
            raise RuntimeError(
                "Invalid TRYON_TRT_MODULES value(s): "
                + ", ".join(invalid)
                + f". Valid values: {', '.join(VALID_TENSORRT_MODULES)}, all, none."
            )
        unsafe = sorted(requested & UNSAFE_TENSORRT_UNET_MODULES)
        allow_unsafe_unet = os.environ.get("TRYON_TRT_ALLOW_UNSAFE_UNET", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if unsafe and not allow_unsafe_unet:
            raise RuntimeError(
                "TensorRT UNet modules are disabled by default because Torch-TensorRT/TensorRT can segfault "
                "inside the native builder on IDM-VTON UNet graphs. Requested unsafe module(s): "
                + ", ".join(unsafe)
                + ". Use TRYON_TRT_MODULES=vae_decode for the stable TensorRT path, use optimization=torch_compile "
                "for UNet speedups, or set TRYON_TRT_ALLOW_UNSAFE_UNET=true only for isolated benchmark experiments."
            )
        return requested

    def _compile_attr_with_tensorrt(self, parent: Any, attr: str, name: str, options: dict[str, Any]) -> None:
        if not hasattr(parent, attr):
            return
        child = getattr(parent, attr)
        if isinstance(child, self.torch.nn.Module):
            setattr(parent, attr, self._compile_module_with_tensorrt(child, name, options))

    def _compile_module_list_with_tensorrt(self, parent: Any, attr: str, name: str, options: dict[str, Any]) -> None:
        module_list = getattr(parent, attr, None)
        if module_list is None:
            return
        for index, child in enumerate(module_list):
            if isinstance(child, self.torch.nn.Module):
                module_list[index] = self._compile_module_with_tensorrt(child, f"{name}.{index}", options)

    def _compile_unet_blocks_with_tensorrt(self, module: Any, name: str, options: dict[str, Any]) -> None:
        self._compile_module_list_with_tensorrt(module, "down_blocks", f"{name}.down_blocks", options)
        self._compile_attr_with_tensorrt(module, "mid_block", f"{name}.mid_block", options)
        self._compile_module_list_with_tensorrt(module, "up_blocks", f"{name}.up_blocks", options)

    def _apply_tensorrt_optimization(self) -> None:
        if self.device.type != "cuda":
            raise RuntimeError("TensorRT optimization requires a CUDA device.")
        options = self._tensorrt_options()
        modules = self._requested_tensorrt_modules
        print(f"TensorRT compile modules={sorted(modules) or ['none']}", file=sys.stderr)
        if "unet_blocks" in modules:
            self._compile_unet_blocks_with_tensorrt(self.pipe.unet, "unet", options)
        if "unet" in modules:
            self.pipe.unet = self._compile_module_with_tensorrt(self.pipe.unet, "unet", options)
        if "unet_encoder_blocks" in modules and hasattr(self.pipe, "unet_encoder"):
            self._compile_unet_blocks_with_tensorrt(self.pipe.unet_encoder, "unet_encoder", options)
        if "unet_encoder" in modules and hasattr(self.pipe, "unet_encoder"):
            self.pipe.unet_encoder = self._compile_module_with_tensorrt(
                self.pipe.unet_encoder,
                "unet_encoder",
                options,
            )
        if "vae_decode" in modules and hasattr(self.pipe, "vae") and hasattr(self.pipe.vae, "decode"):
            self.pipe.vae.decode = self._compile_module_with_tensorrt(
                self.pipe.vae.decode,
                "vae.decode",
                options,
            )

    def _autocast(self):
        if self.device.type == "cuda":
            return self.torch.cuda.amp.autocast()
        return contextlib.nullcontext()

    def run(self, request: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        output_dir = Path(request["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        width = int(request["width"])
        height = int(request["height"])
        steps = int(request["num_inference_steps"])
        guidance_scale = float(request["guidance_scale"])
        seed = int(request["seed"])
        order = "unpaired" if request.get("unpaired", True) else "paired"

        dataset = self.VitonHDTestDataset(
            dataroot_path=str(request["data_dir"]),
            phase="test",
            order=order,
            size=(height, width),
        )
        dataloader = self.torch.utils.data.DataLoader(
            dataset,
            shuffle=False,
            batch_size=int(request.get("test_batch_size", 1)),
            num_workers=0,
        )

        with self.torch.no_grad():
            with self._autocast():
                for sample in dataloader:
                    img_emb_list = [sample["cloth"][i] for i in range(sample["cloth"].shape[0])]
                    prompt = sample["caption"]
                    num_prompts = sample["cloth"].shape[0]
                    negative_prompt: str | list[str] = NEGATIVE_PROMPT

                    if not isinstance(prompt, list):
                        prompt = [prompt] * num_prompts
                    if not isinstance(negative_prompt, list):
                        negative_prompt = [negative_prompt] * num_prompts

                    image_embeds = self.torch.cat(img_emb_list, dim=0)

                    with self.torch.inference_mode():
                        (
                            prompt_embeds,
                            negative_prompt_embeds,
                            pooled_prompt_embeds,
                            negative_pooled_prompt_embeds,
                        ) = self.pipe.encode_prompt(
                            prompt,
                            num_images_per_prompt=1,
                            do_classifier_free_guidance=True,
                            negative_prompt=negative_prompt,
                        )

                        cloth_prompt = sample["caption_cloth"]
                        cloth_negative_prompt: str | list[str] = NEGATIVE_PROMPT
                        if not isinstance(cloth_prompt, list):
                            cloth_prompt = [cloth_prompt] * num_prompts
                        if not isinstance(cloth_negative_prompt, list):
                            cloth_negative_prompt = [cloth_negative_prompt] * num_prompts

                        (
                            prompt_embeds_c,
                            _,
                            _,
                            _,
                        ) = self.pipe.encode_prompt(
                            cloth_prompt,
                            num_images_per_prompt=1,
                            do_classifier_free_guidance=False,
                            negative_prompt=cloth_negative_prompt,
                        )

                        generator = self.torch.Generator(self.pipe.device).manual_seed(seed)
                        images = self.pipe(
                            prompt_embeds=prompt_embeds,
                            negative_prompt_embeds=negative_prompt_embeds,
                            pooled_prompt_embeds=pooled_prompt_embeds,
                            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                            num_inference_steps=steps,
                            generator=generator,
                            strength=1.0,
                            pose_img=sample["pose_img"],
                            text_embeds_cloth=prompt_embeds_c,
                            cloth=sample["cloth_pure"].to(self.device),
                            mask_image=sample["inpaint_mask"],
                            image=(sample["image"] + 1.0) / 2.0,
                            height=height,
                            width=width,
                            guidance_scale=guidance_scale,
                            ip_adapter_image=image_embeds,
                        )[0]

                    for i, image in enumerate(images):
                        x_sample = self.pil_to_tensor(image)
                        output_path = output_dir / sample["im_name"][i]
                        self.torchvision.utils.save_image(x_sample, output_path)

        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)
        return {
            "ok": True,
            "optimization_mode": self.optimization_mode,
            "tensorrt_modules": sorted(self._requested_tensorrt_modules)
            if self.optimization_mode == "tensorrt"
            else [],
            "runtime_seconds": time.perf_counter() - started,
            "output_dir": str(output_dir),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent JSONL worker for IDM-VTON inference.")
    parser.add_argument("--repo-path", required=True)
    parser.add_argument("--model-name", default="yisol/IDM-VTON")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--optimization-mode", choices=["eager", "torch_compile", "tensorrt"], default="eager")
    parser.add_argument("--torch-compile-backend", default="inductor")
    parser.add_argument("--torch-compile-mode", default="reduce-overhead")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol_output = sys.stdout
    load_started = time.perf_counter()
    try:
        with _redirect_noisy_stdout(protocol_output):
            worker = ResidentIDMVTonPipeline(
                Path(args.repo_path),
                args.model_name,
                args.device,
                optimization_mode=args.optimization_mode,
                torch_compile_backend=args.torch_compile_backend,
                torch_compile_mode=args.torch_compile_mode,
            )
        _emit(
            {
                "type": "ready",
                "ok": True,
                "model_name": args.model_name,
                "device": str(worker.device),
                "optimization_mode": worker.optimization_mode,
                "torch_compile_backend": worker.torch_compile_backend,
                "torch_compile_mode": worker.torch_compile_mode,
                "load_runtime_seconds": time.perf_counter() - load_started,
            },
            protocol_output,
        )
    except Exception as exc:  # pragma: no cover - exercised on remote failures.
        traceback.print_exc(file=sys.stderr)
        _emit({"type": "ready", "ok": False, "error": str(exc)}, protocol_output)
        return 1

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            request_id = request.get("request_id")
            if request.get("type") == "shutdown":
                _emit({"type": "shutdown", "ok": True, "request_id": request_id}, protocol_output)
                return 0
            if request.get("type") != "run":
                _emit(
                    {"type": "result", "ok": False, "request_id": request_id, "error": "Unsupported request type."},
                    protocol_output,
                )
                continue
            with _redirect_noisy_stdout(protocol_output):
                result = worker.run(request)
            result.update({"type": "result", "request_id": request_id})
            _emit(result, protocol_output)
        except Exception as exc:  # pragma: no cover - exercised on remote failures.
            traceback.print_exc(file=sys.stderr)
            _emit(
                {
                    "type": "result",
                    "ok": False,
                    "request_id": locals().get("request_id"),
                    "error": str(exc),
                },
                protocol_output,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
