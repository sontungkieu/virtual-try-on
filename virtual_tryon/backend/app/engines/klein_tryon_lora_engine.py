from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from urllib.parse import urlsplit, urlunsplit
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image
from PIL import ImageOps

from app.core.config import EngineConfig
from app.engines.base import TryOnInputs, TryOnResult
from app.engines.klein_prompt_builder import build_klein_tryon_prompt
from app.utils.errors import EngineExecutionError, ModelUnavailableError
from app.utils.image_io import save_image


SENSITIVE_KEY_PARTS = {"token", "key", "authorization", "secret", "credential"}
SUPPORTED_BACKENDS = {"fal_api", "diffusers_local", "disabled"}
_LOCAL_PIPE_CACHE: dict[str, Any] = {}
_LOCAL_PIPE_LOCK = threading.RLock()


@dataclass(frozen=True)
class EngineAvailability:
    available: bool
    status: str
    missing: list[str] = field(default_factory=list)
    error_code: str | None = None

    def __bool__(self) -> bool:
        return self.available


@dataclass(frozen=True)
class KleinReferences:
    person_image: Image.Image
    top_image: Image.Image
    bottom_image: Image.Image | None
    person_path: Path
    top_path: Path
    bottom_path: Path | None
    bottom_source: str
    warnings: list[str]


def _sanitize_payload(value: Any) -> Any:
    if isinstance(value, dict):
        clean: dict[str, Any] = {}
        for key, item in value.items():
            if any(part in key.lower() for part in SENSITIVE_KEY_PARTS):
                clean[key] = "[redacted]"
            else:
                clean[key] = _sanitize_payload(item)
        return clean
    if isinstance(value, list):
        return [_sanitize_payload(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_payload(item) for item in value]
    if isinstance(value, str):
        clean = value.replace("FAL_KEY", "fal credential")
        clean = clean.replace("Authorization", "redacted authorization")
        clean = clean.replace("Bearer ", "redacted bearer ")
        if clean.startswith(("http://", "https://")):
            parsed = urlsplit(clean)
            if parsed.query or parsed.fragment:
                return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "[redacted-query]", ""))
        return clean
    return value


def _patch_sdpa_enable_gqa() -> None:
    try:
        import torch
    except ImportError:
        return

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


def _fit_canvas(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    image = ImageOps.exif_transpose(image.convert("RGB"))
    contained = ImageOps.contain(image, size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(contained, ((size[0] - contained.width) // 2, (size[1] - contained.height) // 2))
    return canvas


class KleinTryOnLoraEngine:
    name = "klein_tryon_lora"

    def __init__(self, config: EngineConfig) -> None:
        self.config = config

    @property
    def steps(self) -> int:
        return int(self.config.num_inference_steps or self.config.steps)

    @property
    def lora_path_for_api(self) -> str:
        repo = self.config.lora_repo or "fal/flux-klein-9b-virtual-tryon-lora"
        weight = self.config.lora_weight_api or "flux-klein-tryon.safetensors"
        return f"{repo.rstrip('/')}/{weight}"

    @property
    def local_model_dir(self) -> Path | None:
        if self.config.model_path:
            return self.config.model_path
        return self.config.checkpoint_dir

    @property
    def output_size(self) -> tuple[int, int]:
        width = int(self.config.default_width or self.config.resolution or 768)
        height = int(self.config.default_height or self.config.resolution or 1024)
        return width, height

    @staticmethod
    def _diffusers_local_import_error() -> str | None:
        try:
            import torch  # noqa: F401
            import peft  # noqa: F401
            from diffusers import Flux2KleinPipeline  # noqa: F401
        except ImportError as exc:
            return f"{type(exc).__name__}: {exc}"
        return None

    def _local_cache_key(self) -> str:
        return json.dumps(
            {
                "model_dir": str(self.local_model_dir),
                "lora_path": str(self.config.lora_path),
                "device_map": self.config.device_map,
                "quantization": self.config.quantization,
                "quantize_components": list(self.config.quantize_components),
            },
            sort_keys=True,
        )

    @property
    def local_worker_python(self) -> str:
        return os.getenv("TRYON_KLEIN_PYTHON") or sys.executable

    def _local_worker_check_error(self) -> str | None:
        if not self.config.entrypoint:
            return self._diffusers_local_import_error()
        if not self.config.entrypoint.exists():
            return f"worker entrypoint not found: {self.config.entrypoint}"
        command = [
            self.local_worker_python,
            str(self.config.entrypoint),
            "--check",
            "--model-dir",
            str(self.local_model_dir),
            "--lora-path",
            str(self.config.lora_path),
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=self.config.entrypoint.parent,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return f"worker runtime check failed: {type(exc).__name__}: {exc}"
        if completed.returncode != 0:
            stderr = completed.stderr.strip()[-1000:]
            return f"worker runtime check failed: {stderr or completed.stdout.strip()[-1000:]}"
        return None

    def _load_diffusers_local_pipe(self):
        model_dir = self.local_model_dir
        lora_path = self.config.lora_path
        if model_dir is None:
            raise ModelUnavailableError("Klein Try-On LoRA local model path is not configured.")
        if lora_path is None:
            raise ModelUnavailableError("Klein Try-On LoRA local LoRA path is not configured.")

        cache_key = self._local_cache_key()
        with _LOCAL_PIPE_LOCK:
            cached = _LOCAL_PIPE_CACHE.get(cache_key)
            if cached is not None:
                if hasattr(cached, "set_adapters"):
                    cached.set_adapters(["tryon"], adapter_weights=[float(self.config.lora_scale)])
                return cached

            _patch_sdpa_enable_gqa()
            import torch
            from diffusers import Flux2KleinPipeline

            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.set_float32_matmul_precision("high")

            try:
                quantization_config = None
                if self.config.quantization != "none":
                    from scripts.klein_diffusers_local_worker import (
                        build_quantization_config,
                        normalize_components,
                        normalize_quantization,
                    )

                    quantization_config = build_quantization_config(
                        normalize_quantization(self.config.quantization),
                        normalize_components(self.config.quantize_components),
                    )
                from_pretrained_kwargs = {
                    "torch_dtype": torch.bfloat16,
                    "local_files_only": True,
                    "low_cpu_mem_usage": True,
                }
                if quantization_config is not None:
                    from_pretrained_kwargs["quantization_config"] = quantization_config
                if self.config.device_map in {"balanced", "auto"}:
                    from_pretrained_kwargs["device_map"] = self.config.device_map
                pipe = Flux2KleinPipeline.from_pretrained(
                    model_dir,
                    **from_pretrained_kwargs,
                )
            except TypeError:
                from_pretrained_kwargs.pop("low_cpu_mem_usage", None)
                pipe = Flux2KleinPipeline.from_pretrained(
                    model_dir,
                    **from_pretrained_kwargs,
                )
            if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
                pipe.vae.enable_tiling()
            device_map = self.config.device_map.replace("-", "_")
            if device_map in {"offload", "model_cpu_offload"}:
                device_map = "cpu_offload"
            elif device_map in {"gpu", "all_gpu", "all_cuda", "full_cuda"}:
                device_map = "cuda"
            if device_map == "cpu_offload" and torch.cuda.is_available() and hasattr(pipe, "enable_model_cpu_offload"):
                pipe.enable_model_cpu_offload(gpu_id=0)
            elif device_map == "sequential_cpu_offload" and torch.cuda.is_available() and hasattr(pipe, "enable_sequential_cpu_offload"):
                pipe.enable_sequential_cpu_offload(gpu_id=0)
            elif device_map == "cuda" and torch.cuda.is_available():
                pipe.to("cuda")

            pipe.load_lora_weights(
                lora_path.parent,
                weight_name=lora_path.name,
                adapter_name="tryon",
                local_files_only=True,
            )
            if hasattr(pipe, "set_adapters"):
                pipe.set_adapters(["tryon"], adapter_weights=[float(self.config.lora_scale)])

            _LOCAL_PIPE_CACHE[cache_key] = pipe
            return pipe

    def is_available(self) -> EngineAvailability:
        missing: list[str] = []
        error_code: str | None = None
        backend = self.config.backend or "disabled"

        if not self.config.enabled:
            missing.append("klein_tryon_lora.enabled is false")
            error_code = "DISABLED"
        if backend not in SUPPORTED_BACKENDS:
            missing.append(f"unsupported backend: {backend}")
            error_code = error_code or "INVALID_BACKEND"
        if backend == "disabled":
            missing.append("klein_tryon_lora.backend is disabled")
            error_code = error_code or "DISABLED"
        if not self.config.base_model:
            missing.append("base_model is not configured")
            error_code = error_code or "CONFIG_MISSING"

        if backend == "fal_api":
            if not os.getenv("FAL_KEY"):
                missing.append("FAL_KEY is not set")
                error_code = error_code or "MISSING_FAL_KEY"
            if not self.config.fal_endpoint:
                missing.append("fal_endpoint is not configured")
                error_code = error_code or "CONFIG_MISSING"
            if not (self.config.lora_repo and self.config.lora_weight_api):
                missing.append("lora_repo/lora_weight_api are not configured")
                error_code = error_code or "CONFIG_MISSING"
            try:
                import fal_client  # noqa: F401
            except ImportError:
                missing.append("fal_client package is not installed")
                error_code = error_code or "DEPENDENCY_MISSING"

        if backend == "diffusers_local":
            model_dir = self.local_model_dir
            if not model_dir or not model_dir.exists():
                missing.append(f"local FLUX.2 Klein base model directory not found: {model_dir}")
                error_code = error_code or "MODEL_MISSING"
            elif model_dir.is_dir() and not (model_dir / "model_index.json").exists():
                missing.append(f"local FLUX.2 Klein model directory lacks model_index.json: {model_dir}")
                error_code = error_code or "MODEL_MISSING"
            if not self.config.lora_path or not self.config.lora_path.exists():
                missing.append(f"LoRA weights not found: {self.config.lora_path}")
                error_code = error_code or "LORA_MISSING"
            if error_code not in {"MODEL_MISSING", "LORA_MISSING"}:
                import_error = self._local_worker_check_error()
                if import_error:
                    missing.append(f"diffusers_local dependencies unavailable: {import_error}")
                    error_code = error_code or "DEPENDENCY_MISSING"

        if missing:
            return EngineAvailability(
                available=False,
                status="unavailable: " + "; ".join(missing),
                missing=missing,
                error_code=error_code,
            )
        return EngineAvailability(available=True, status="available")

    def missing_requirements(self) -> list[str]:
        return self.is_available().missing

    def status(self) -> str:
        return self.is_available().status

    def prepare(self) -> None:
        availability = self.is_available()
        if not availability:
            raise ModelUnavailableError("Klein Try-On LoRA is not available. " + availability.status)

    def _write_status(self, output_dir: Path, payload: dict[str, Any]) -> None:
        payload = _sanitize_payload(payload)
        (output_dir / "klein_lora_status.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (output_dir / "status.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _save_prompt(self, output_dir: Path, prompt: str) -> None:
        (output_dir / "klein_lora_prompt.txt").write_text(prompt, encoding="utf-8")
        (output_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

    def _save_json_aliases(self, output_dir: Path, stem: str, payload: dict[str, Any]) -> None:
        payload = _sanitize_payload(payload)
        for name in [f"klein_lora_{stem}.json", f"{stem}_sanitized.json"]:
            (output_dir / name).write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    def _crop_bottom_from_person(self, person: Image.Image, output_dir: Path) -> tuple[Image.Image, Path, dict[str, Any]]:
        crop_config = self.config.bottom_crop or {}
        y_start_ratio = float(crop_config.get("y_start_ratio", 0.50))
        y_end_ratio = float(crop_config.get("y_end_ratio", 0.98))
        x_margin_ratio = float(crop_config.get("x_margin_ratio", 0.08))
        width, height = person.size
        left = max(0, int(width * x_margin_ratio))
        right = min(width, int(width * (1.0 - x_margin_ratio)))
        top = max(0, int(height * y_start_ratio))
        bottom = min(height, int(height * y_end_ratio))
        if right - left < 16 or bottom - top < 16:
            raise ValueError("bottom crop box is too small")
        cropped = person.crop((left, top, right, bottom)).convert("RGB")
        path = save_image(cropped, output_dir / "auto_bottom_reference.png")
        return cropped, path, {
            "strategy": "crop_from_person",
            "box": [left, top, right, bottom],
            "source": "person_image",
        }

    def _blank_bottom_placeholder(self, person: Image.Image, output_dir: Path) -> tuple[Image.Image, Path, dict[str, Any]]:
        width, height = person.size
        placeholder = Image.new("RGB", (max(64, width // 2), max(64, height // 2)), (235, 235, 235))
        path = save_image(placeholder, output_dir / "auto_bottom_reference.png")
        return placeholder, path, {
            "strategy": "blank_placeholder",
            "warning": "Neutral placeholder used because no bottom garment was provided.",
        }

    def _prepare_references(self, inputs: TryOnInputs, output_dir: Path) -> KleinReferences:
        warnings: list[str] = []
        person = inputs.person_image.convert("RGB")
        top_image = (
            inputs.extra.get("garment_top_image")
            or inputs.extra.get("top_image")
            or inputs.garment_image
        ).convert("RGB")
        bottom_image = inputs.extra.get("garment_bottom_image") or inputs.extra.get("bottom_image")

        person_path = save_image(person, output_dir / "person_reference.png")
        top_path = save_image(top_image, output_dir / "top_reference.png")

        bottom_path: Path | None = None
        bottom_source = "provided"
        if bottom_image is not None:
            bottom_image = bottom_image.convert("RGB")
            bottom_path = save_image(bottom_image, output_dir / "bottom_reference.png")
        elif inputs.category == "upper_body" and self.config.bottom_strategy == "crop_from_person":
            try:
                bottom_image, bottom_path, metadata = self._crop_bottom_from_person(person, output_dir)
                bottom_source = "auto_cropped_from_person"
                warnings.append(json.dumps(metadata, separators=(",", ":")))
            except Exception as exc:
                warnings.append(f"bottom crop failed: {exc}")
                if self.config.bottom_strategy == "blank_placeholder":
                    bottom_image, bottom_path, _ = self._blank_bottom_placeholder(person, output_dir)
                    bottom_source = "blank_placeholder"
        elif self.config.bottom_strategy == "blank_placeholder":
            bottom_image, bottom_path, _ = self._blank_bottom_placeholder(person, output_dir)
            bottom_source = "blank_placeholder"
        else:
            bottom_source = "missing"

        if self.config.require_three_images and bottom_image is None:
            warnings.append("required bottom reference image is missing")

        return KleinReferences(
            person_image=person,
            top_image=top_image,
            bottom_image=bottom_image,
            person_path=person_path,
            top_path=top_path,
            bottom_path=bottom_path,
            bottom_source=bottom_source,
            warnings=warnings,
        )

    def _request_payload(self, prompt: str, references: KleinReferences) -> dict[str, Any]:
        image_paths = [references.person_path.as_posix(), references.top_path.as_posix()]
        if references.bottom_path:
            image_paths.append(references.bottom_path.as_posix())
        return {
            "backend": self.config.backend,
            "endpoint": self.config.fal_endpoint,
            "prompt": prompt,
            "image_paths": image_paths,
            "loras": [{"path": self.lora_path_for_api, "scale": self.config.lora_scale}],
            "num_inference_steps": self.steps,
            "guidance_scale": self.config.guidance_scale,
            "resolution": self.config.resolution,
            "require_three_images": self.config.require_three_images,
            "bottom_source": references.bottom_source,
            "device_map": self.config.device_map,
            "quantization": self.config.quantization,
            "quantize_components": list(self.config.quantize_components),
        }

    @staticmethod
    def _extract_image_url(response: Any) -> str | None:
        if isinstance(response, dict):
            for key in ["image", "output", "result"]:
                value = response.get(key)
                if isinstance(value, str) and value.startswith(("http://", "https://")):
                    return value
                if isinstance(value, dict):
                    url = value.get("url") or value.get("image_url")
                    if isinstance(url, str):
                        return url
            images = response.get("images")
            if isinstance(images, list) and images:
                first = images[0]
                if isinstance(first, str):
                    return first
                if isinstance(first, dict):
                    url = first.get("url") or first.get("image_url")
                    if isinstance(url, str):
                        return url
        return None

    @staticmethod
    def _download_image(url: str, output_path: Path, timeout_seconds: int) -> Image.Image:
        request = urllib.request.Request(url, headers={"User-Agent": "virtual-tryon-klein-lora/0.1"})
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            output_path.write_bytes(response.read())
        return Image.open(output_path).convert("RGB")

    def _run_fal_api(self, prompt: str, references: KleinReferences, output_dir: Path) -> TryOnResult:
        import fal_client

        image_urls = [
            fal_client.upload_file(str(references.person_path)),
            fal_client.upload_file(str(references.top_path)),
        ]
        if references.bottom_path:
            image_urls.append(fal_client.upload_file(str(references.bottom_path)))
        if self.config.require_three_images and len(image_urls) < 3:
            raise ModelUnavailableError("Klein Try-On LoRA requires three image references for fal_api.")

        arguments = {
            "prompt": prompt,
            "image_urls": image_urls,
            "loras": [{"path": self.lora_path_for_api, "scale": self.config.lora_scale}],
            "num_inference_steps": self.steps,
            "guidance_scale": self.config.guidance_scale,
        }
        self._save_json_aliases(output_dir, "request", {
            **self._request_payload(prompt, references),
            "image_urls": image_urls,
        })

        started = time.perf_counter()
        try:
            response = fal_client.subscribe(
                self.config.fal_endpoint,
                arguments=arguments,
                with_logs=True,
            )
        except TypeError:
            response = fal_client.subscribe(self.config.fal_endpoint, arguments=arguments)
        runtime_seconds = time.perf_counter() - started
        response_payload = _sanitize_payload(response if isinstance(response, dict) else {"response": str(response)})
        self._save_json_aliases(output_dir, "response", response_payload)

        result_url = self._extract_image_url(response)
        if not result_url:
            raise EngineExecutionError("fal.ai response did not include a result image URL.")

        image = self._download_image(result_url, output_dir / "klein_lora_result.png", self.config.timeout_seconds)
        save_image(image, output_dir / "result.png")
        metadata = {
            "engine": self.name,
            "backend": "fal_api",
            "runtime_seconds": round(runtime_seconds, 3),
            "endpoint": self.config.fal_endpoint,
            "bottom_source": references.bottom_source,
            "warnings": references.warnings,
        }
        return TryOnResult(image=image, metadata=metadata)

    def _release_idm_resident_worker(self) -> None:
        try:
            from app.engines.idm_vton_engine import stop_resident_clients

            stop_resident_clients()
        except Exception:
            pass

    def _run_diffusers_local(
        self,
        prompt: str,
        references: KleinReferences,
        output_dir: Path,
        seed: int | None,
    ) -> TryOnResult:
        if self.config.entrypoint:
            return self._run_diffusers_local_subprocess(prompt, references, output_dir, seed)
        return self._run_diffusers_local_in_process(prompt, references, output_dir, seed)

    def _run_diffusers_local_subprocess(
        self,
        prompt: str,
        references: KleinReferences,
        output_dir: Path,
        seed: int | None,
    ) -> TryOnResult:
        if not self.config.entrypoint:
            raise ModelUnavailableError("Klein Try-On LoRA local worker entrypoint is not configured.")
        if self.config.require_three_images and references.bottom_image is None:
            raise ModelUnavailableError("Klein Try-On LoRA requires three image references for diffusers_local.")
        self._release_idm_resident_worker()

        width, height = self.output_size
        image_paths = [references.person_path.as_posix(), references.top_path.as_posix()]
        if references.bottom_path:
            image_paths.append(references.bottom_path.as_posix())
        request_payload = {
            **self._request_payload(prompt, references),
            "model_dir": str(self.local_model_dir),
            "lora_path": str(self.config.lora_path),
            "width": width,
            "height": height,
            "steps": self.steps,
            "guidance_scale": self.config.guidance_scale,
            "lora_scale": self.config.lora_scale,
            "seed": seed,
            "image_paths": image_paths,
            "device_map": self.config.device_map,
            "quantization": self.config.quantization,
            "quantize_components": list(self.config.quantize_components),
            "local_files_only": True,
            "worker_python": self.local_worker_python,
            "worker_entrypoint": str(self.config.entrypoint),
        }
        request_path = output_dir / "local_worker_request.json"
        request_path.write_text(json.dumps(_sanitize_payload(request_payload), indent=2), encoding="utf-8")
        self._save_json_aliases(output_dir, "local_generation", request_payload)

        command = [
            self.local_worker_python,
            str(self.config.entrypoint),
            "--request",
            str(request_path),
            "--output-dir",
            str(output_dir),
        ]
        started = time.perf_counter()
        completed = subprocess.run(
            command,
            cwd=self.config.entrypoint.parent,
            capture_output=True,
            text=True,
            timeout=self.config.timeout_seconds,
            check=False,
        )
        runtime_seconds = time.perf_counter() - started
        (output_dir / "local_worker_stdout.txt").write_text(completed.stdout[-8000:], encoding="utf-8")
        (output_dir / "local_worker_stderr.txt").write_text(completed.stderr[-8000:], encoding="utf-8")
        if completed.returncode != 0:
            message = (completed.stderr.strip() or completed.stdout.strip() or "local Klein worker failed")[-2000:]
            raise EngineExecutionError(f"local Klein worker failed rc={completed.returncode}: {message}")
        worker_result_path = output_dir / "worker_result.json"
        if worker_result_path.exists():
            worker_payload = json.loads(worker_result_path.read_text(encoding="utf-8"))
        else:
            worker_payload = json.loads(completed.stdout)
        result_path = Path(worker_payload.get("result_path", output_dir / "klein_lora_result.png"))
        image = Image.open(result_path).convert("RGB")
        save_image(image, output_dir / "result.png")
        metadata = {
            "engine": self.name,
            "backend": "diffusers_local",
            "runtime_seconds": round(runtime_seconds, 3),
            "worker": worker_payload,
            "model_dir": str(self.local_model_dir),
            "lora_path": str(self.config.lora_path),
            "lora_scale": self.config.lora_scale,
            "steps": self.steps,
            "guidance_scale": self.config.guidance_scale,
            "device_map": self.config.device_map,
            "quantization": self.config.quantization,
            "quantize_components": list(self.config.quantize_components),
            "width": width,
            "height": height,
            "seed": seed,
            "bottom_source": references.bottom_source,
            "warnings": references.warnings,
        }
        return TryOnResult(image=image, metadata=metadata)

    def _run_diffusers_local_in_process(
        self,
        prompt: str,
        references: KleinReferences,
        output_dir: Path,
        seed: int | None,
    ) -> TryOnResult:
        if self.config.require_three_images and references.bottom_image is None:
            raise ModelUnavailableError("Klein Try-On LoRA requires three image references for diffusers_local.")

        self._release_idm_resident_worker()

        import torch

        width, height = self.output_size
        image_references = [
            _fit_canvas(references.person_image, (width, height)),
            _fit_canvas(references.top_image, (width, height)),
        ]
        if references.bottom_image is not None:
            image_references.append(_fit_canvas(references.bottom_image, (width, height)))

        self._save_json_aliases(
            output_dir,
            "local_generation",
            {
                **self._request_payload(prompt, references),
                "model_dir": str(self.local_model_dir),
                "lora_path": str(self.config.lora_path),
                "width": width,
                "height": height,
                "seed": seed,
                "device_map": self.config.device_map,
                "quantization": self.config.quantization,
                "quantize_components": list(self.config.quantize_components),
                "local_files_only": True,
            },
        )

        pipe = self._load_diffusers_local_pipe()
        generator_device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(int(seed or 0))
        started = time.perf_counter()
        try:
            output = pipe(
                image=image_references,
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=self.steps,
                guidance_scale=float(self.config.guidance_scale),
                generator=generator,
            )
        except TypeError:
            output = pipe(
                image=image_references,
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=self.steps,
                guidance_scale=float(self.config.guidance_scale),
            )
        runtime_seconds = time.perf_counter() - started
        image = output.images[0].convert("RGB")
        save_image(image, output_dir / "klein_lora_result.png")
        save_image(image, output_dir / "result.png")
        metadata = {
            "engine": self.name,
            "backend": "diffusers_local",
            "runtime_seconds": round(runtime_seconds, 3),
            "model_dir": str(self.local_model_dir),
            "lora_path": str(self.config.lora_path),
            "lora_scale": self.config.lora_scale,
            "steps": self.steps,
            "guidance_scale": self.config.guidance_scale,
            "device_map": self.config.device_map,
            "quantization": self.config.quantization,
            "quantize_components": list(self.config.quantize_components),
            "width": width,
            "height": height,
            "seed": seed,
            "bottom_source": references.bottom_source,
            "warnings": references.warnings,
        }
        return TryOnResult(image=image, metadata=metadata)

    def run(self, inputs: TryOnInputs) -> TryOnResult:
        output_dir = Path(inputs.output_dir or ".")
        output_dir.mkdir(parents=True, exist_ok=True)

        prompt = build_klein_tryon_prompt(
            inputs.extra.get("person_description"),
            inputs.extra.get("top_description"),
            inputs.extra.get("bottom_description"),
            inputs.category,
            preserve_original_bottom=inputs.category == "upper_body",
            extra_instruction=inputs.prompt,
        )
        self._save_prompt(output_dir, prompt)
        references = self._prepare_references(inputs, output_dir)
        request_payload = self._request_payload(prompt, references)
        self._save_json_aliases(output_dir, "request", request_payload)

        availability = self.is_available()
        if self.config.require_three_images and references.bottom_image is None:
            availability = EngineAvailability(
                available=False,
                status="unavailable: required bottom reference image is missing",
                missing=[*availability.missing, "required bottom reference image is missing"],
                error_code=availability.error_code or "BOTTOM_REFERENCE_MISSING",
            )
        if not availability:
            status_payload = {
                "status": "unavailable",
                "engine": self.name,
                "backend": self.config.backend,
                "error_code": availability.error_code or "ENGINE_UNAVAILABLE",
                "message": availability.status,
                "bottom_source": references.bottom_source,
                "warnings": references.warnings,
            }
            self._write_status(output_dir, status_payload)
            raise ModelUnavailableError("Klein Try-On LoRA is not available. " + availability.status)

        status_payload = {
            "status": "running",
            "engine": self.name,
            "backend": self.config.backend,
            "bottom_source": references.bottom_source,
            "warnings": references.warnings,
        }
        self._write_status(output_dir, status_payload)
        try:
            if self.config.backend == "fal_api":
                result = self._run_fal_api(prompt, references, output_dir)
            elif self.config.backend == "diffusers_local":
                result = self._run_diffusers_local(prompt, references, output_dir, inputs.seed)
            else:
                raise ModelUnavailableError("Klein Try-On LoRA backend is disabled.")
        except Exception as exc:
            if isinstance(exc, (ModelUnavailableError, EngineExecutionError)):
                message = str(exc)
                error_code = "ENGINE_UNAVAILABLE" if isinstance(exc, ModelUnavailableError) else "ENGINE_EXECUTION_FAILED"
            else:
                message = f"{type(exc).__name__}: {exc}"
                error_code = "ENGINE_EXECUTION_FAILED"
            self._write_status(
                output_dir,
                {
                    "status": "failed",
                    "engine": self.name,
                    "backend": self.config.backend,
                    "error_code": error_code,
                    "message": message,
                    "bottom_source": references.bottom_source,
                    "warnings": references.warnings,
                },
            )
            if isinstance(exc, (ModelUnavailableError, EngineExecutionError)):
                raise
            raise EngineExecutionError(message) from exc

        self._write_status(
            output_dir,
            {
                "status": "completed",
                "engine": self.name,
                "backend": self.config.backend,
                "bottom_source": references.bottom_source,
                "warnings": references.warnings,
                "metadata": result.metadata,
            },
        )
        return result
