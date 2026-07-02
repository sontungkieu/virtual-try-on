from __future__ import annotations

import json
import os
import atexit
import queue
import subprocess
import sys
import threading
import time
import urllib.request
from urllib.parse import urlsplit, urlunsplit
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from PIL import Image
from PIL import ImageOps

from app.core.config import EngineConfig
from app.engines.base import TryOnInputs, TryOnResult
from app.engines.klein_prompt_builder import build_klein_tryon_prompt
from app.utils.errors import EngineExecutionError, ModelUnavailableError
from app.utils.image_io import save_image


SENSITIVE_KEY_PARTS = {"token", "key", "authorization", "secret", "credential"}
SUPPORTED_BACKENDS = {"fal_api", "diffusers_local", "disabled"}
RESIDENT_PROTOCOL_PREFIX = "__KLEIN_TRYON_WORKER__ "
_LOCAL_PIPE_CACHE: dict[str, Any] = {}
_LOCAL_PIPE_LOCK = threading.RLock()


class KleinResidentClient:
    def __init__(self, python_executable: str, entrypoint: Path, timeout_seconds: int) -> None:
        self.python_executable = python_executable
        self.entrypoint = entrypoint
        self.timeout_seconds = timeout_seconds
        self._process: subprocess.Popen | None = None
        self._lock = threading.RLock()
        self._stdout_queue: queue.Queue[str] = queue.Queue()
        self._stdout_tail: deque[str] = deque(maxlen=200)
        self._stderr_tail: deque[str] = deque(maxlen=400)

    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def status(self) -> str:
        if self.is_running():
            assert self._process is not None
            return f"running pid={self._process.pid}"
        if self._process is None:
            return "not_started"
        return f"exited rc={self._process.returncode}"

    def stdout_tail(self) -> str:
        return "".join(self._stdout_tail)

    def stderr_tail(self) -> str:
        return "".join(self._stderr_tail)

    def start(self) -> None:
        if self.is_running():
            return
        if not self.entrypoint.exists():
            raise ModelUnavailableError(f"Klein resident worker entrypoint not found: {self.entrypoint}")
        self._stdout_queue = queue.Queue()
        self._stdout_tail.clear()
        self._stderr_tail.clear()
        command = [self.python_executable, str(self.entrypoint), "--resident"]
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"}
        self._process = subprocess.Popen(
            command,
            cwd=self.entrypoint.parent,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
        self._start_reader_threads()
        ready = self._read_protocol_message(
            expected_type="ready",
            request_id=None,
            timeout_seconds=min(max(self.timeout_seconds, 60), 600),
        )
        if not ready.get("ok"):
            error = ready.get("error") or "unknown startup error"
            stderr_tail = self.stderr_tail()
            self.stop()
            raise EngineExecutionError(f"Klein resident worker failed to start: {error}\n{stderr_tail[-2000:]}")

    def _start_reader_threads(self) -> None:
        assert self._process is not None
        if self._process.stdout is not None:
            threading.Thread(target=self._read_stdout, args=(self._process.stdout,), daemon=True).start()
        if self._process.stderr is not None:
            threading.Thread(target=self._read_stderr, args=(self._process.stderr,), daemon=True).start()

    def _read_stdout(self, stream) -> None:
        for line in stream:
            self._stdout_tail.append(line)
            self._stdout_queue.put(line)

    def _read_stderr(self, stream) -> None:
        for line in stream:
            self._stderr_tail.append(line)

    def _read_protocol_message(
        self,
        *,
        expected_type: str | None,
        request_id: str | None,
        timeout_seconds: int,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.stop()
                raise EngineExecutionError(
                    f"Timed out waiting for Klein resident worker message after {timeout_seconds}s. "
                    f"stderr_tail={self.stderr_tail()[-2000:]}"
                )
            try:
                line = self._stdout_queue.get(timeout=min(0.5, remaining))
            except queue.Empty:
                if self._process is not None and self._process.poll() is not None:
                    raise EngineExecutionError(
                        f"Klein resident worker exited early. returncode={self._process.returncode} "
                        f"stderr_tail={self.stderr_tail()[-2000:]}"
                    )
                continue
            if not line.startswith(RESIDENT_PROTOCOL_PREFIX):
                continue
            try:
                payload = json.loads(line[len(RESIDENT_PROTOCOL_PREFIX) :])
            except json.JSONDecodeError:
                continue
            if expected_type is not None and payload.get("type") != expected_type:
                continue
            if request_id is not None and payload.get("request_id") != request_id:
                continue
            return payload

    def request(
        self,
        message_type: str,
        payload: dict[str, Any],
        *,
        output_dir: Path | None = None,
        progress_callback: Any = None,
    ) -> dict[str, Any]:
        with self._lock:
            self.start()
            if self._process is None or self._process.stdin is None:
                raise EngineExecutionError("Klein resident worker stdin is unavailable.")
            request_id = uuid4().hex
            message = {"type": message_type, "request_id": request_id, "payload": payload}
            if output_dir is not None:
                message["output_dir"] = str(output_dir)
            self._process.stdin.write(json.dumps(message, separators=(",", ":"), default=str) + "\n")
            self._process.stdin.flush()
            while True:
                response = self._read_protocol_message(
                    expected_type=None,
                    request_id=request_id,
                    timeout_seconds=self.timeout_seconds,
                )
                response_type = response.get("type")
                if response_type == "progress":
                    if progress_callback is not None:
                        progress_callback(response["stage"], response["status"])
                    continue
                if response_type != "result":
                    continue
                if not response.get("ok"):
                    raise EngineExecutionError(
                        "Klein resident worker request failed. "
                        f"error={response.get('error')} stderr_tail={self.stderr_tail()[-2000:]}"
                    )
                result = response.get("result")
                if not isinstance(result, dict):
                    raise EngineExecutionError("Klein resident worker returned an invalid result payload.")
                return result

    def stop(self) -> None:
        process = self._process
        if process is None or process.poll() is not None:
            return
        try:
            if process.stdin is not None:
                process.stdin.write(json.dumps({"type": "shutdown", "request_id": uuid4().hex}) + "\n")
                process.stdin.flush()
            process.terminate()
            process.wait(timeout=5)
        except Exception:
            process.kill()
            process.wait(timeout=5)


_RESIDENT_CLIENTS: dict[str, KleinResidentClient] = {}
_RESIDENT_CLIENTS_LOCK = threading.RLock()
_RESIDENT_ATEXIT_REGISTERED = False


def _resident_client_key(python_executable: str, entrypoint: Path) -> str:
    return json.dumps({"python": python_executable, "entrypoint": str(entrypoint)}, sort_keys=True)


def _stop_resident_clients() -> None:
    with _RESIDENT_CLIENTS_LOCK:
        for client in _RESIDENT_CLIENTS.values():
            client.stop()


def stop_klein_resident_clients() -> None:
    _stop_resident_clients()


def _get_resident_client(config: EngineConfig, python_executable: str) -> KleinResidentClient:
    global _RESIDENT_ATEXIT_REGISTERED
    if config.entrypoint is None:
        raise ModelUnavailableError("Klein resident worker entrypoint is not configured.")
    key = _resident_client_key(python_executable, config.entrypoint)
    with _RESIDENT_CLIENTS_LOCK:
        if not _RESIDENT_ATEXIT_REGISTERED:
            atexit.register(_stop_resident_clients)
            _RESIDENT_ATEXIT_REGISTERED = True
        client = _RESIDENT_CLIENTS.get(key)
        if client is None:
            client = KleinResidentClient(python_executable, config.entrypoint, config.timeout_seconds)
            _RESIDENT_CLIENTS[key] = client
        return client


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
                "tensorrt_profile": self.config.tensorrt_profile,
                "tensorrt_components": list(self.config.tensorrt_components),
                "tensorrt_engine_cache_dir": (
                    str(self.config.tensorrt_engine_cache_dir)
                    if self.config.tensorrt_engine_cache_dir
                    else None
                ),
                "tensorrt_min_block_size": self.config.tensorrt_min_block_size,
            },
            sort_keys=True,
        )

    def _tensorrt_payload(self) -> dict[str, Any]:
        return {
            "tensorrt_profile": self.config.tensorrt_profile,
            "tensorrt_components": list(self.config.tensorrt_components),
            "tensorrt_engine_cache_dir": (
                str(self.config.tensorrt_engine_cache_dir)
                if self.config.tensorrt_engine_cache_dir
                else None
            ),
            "tensorrt_min_block_size": self.config.tensorrt_min_block_size,
        }

    @property
    def local_worker_python(self) -> str:
        return os.getenv("TRYON_KLEIN_PYTHON") or sys.executable

    def _model_payload(self) -> dict[str, Any]:
        return {
            "model_dir": str(self.local_model_dir),
            "lora_path": str(self.config.lora_path),
            "lora_scale": self.config.lora_scale,
            "device_map": self.config.device_map,
            "quantization": self.config.quantization,
            "quantize_components": list(self.config.quantize_components),
            **self._tensorrt_payload(),
            "local_files_only": True,
            "worker_python": self.local_worker_python,
            "worker_entrypoint": str(self.config.entrypoint) if self.config.entrypoint else None,
        }

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
                        apply_klein_tensorrt_optimization,
                        build_quantization_config,
                        normalize_components,
                        normalize_quantization,
                    )

                    quantization_config = build_quantization_config(
                        normalize_quantization(self.config.quantization),
                        normalize_components(self.config.quantize_components),
                    )
                else:
                    from scripts.klein_diffusers_local_worker import (
                        apply_klein_tensorrt_optimization,
                        normalize_components,
                        normalize_quantization,
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

            apply_klein_tensorrt_optimization(
                pipe,
                profile=self.config.tensorrt_profile,
                components=list(self.config.tensorrt_components),
                quantization=normalize_quantization(self.config.quantization),
                quantize_components=normalize_components(self.config.quantize_components),
                engine_cache_dir=self.config.tensorrt_engine_cache_dir,
                min_block_size=self.config.tensorrt_min_block_size,
            )
            _LOCAL_PIPE_CACHE[cache_key] = pipe
            return pipe

    def is_available(self) -> EngineAvailability:
        missing: list[str] = []
        error_code: str | None = None
        backend = self.config.backend or "disabled"

        if not self.config.enabled:
            missing.append("klein_tryon_lora.enabled is false")
            error_code = "DISABLED"
            return EngineAvailability(
                available=False,
                status="unavailable: " + "; ".join(missing),
                missing=missing,
                error_code=error_code,
            )
        if backend not in SUPPORTED_BACKENDS:
            missing.append(f"unsupported backend: {backend}")
            error_code = error_code or "INVALID_BACKEND"
        if backend == "disabled":
            missing.append("klein_tryon_lora.backend is disabled")
            error_code = error_code or "DISABLED"
            return EngineAvailability(
                available=False,
                status="unavailable: " + "; ".join(missing),
                missing=missing,
                error_code=error_code,
            )
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

    def preload(self, *, release_idm: bool = True) -> dict[str, Any]:
        started = time.perf_counter()
        self.prepare()
        if self.config.backend != "diffusers_local":
            return {
                "status": "ready",
                "engine": self.name,
                "backend": self.config.backend,
                "runtime_seconds": round(time.perf_counter() - started, 3),
            }
        if release_idm:
            self._release_idm_resident_worker()
        if self.config.entrypoint:
            client = _get_resident_client(self.config, self.local_worker_python)
            result = client.request("prepare", self._model_payload())
            return {
                "status": "ready",
                "engine": self.name,
                "backend": "diffusers_local",
                "resident_worker": client.status(),
                "runtime_seconds": round(time.perf_counter() - started, 3),
                "worker": result,
            }
        pipe = self._load_diffusers_local_pipe()
        return {
            "status": "ready",
            "engine": self.name,
            "backend": "diffusers_local",
            "resident_worker": "in_process",
            "runtime_seconds": round(time.perf_counter() - started, 3),
            "tensorrt": getattr(pipe, "_vton_tensorrt_metadata", {}),
        }

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
        explicit_top = inputs.extra.get("garment_top_image") or inputs.extra.get("top_image")
        if explicit_top is not None:
            top_image = explicit_top.convert("RGB")
        elif inputs.category in {"men_underwear", "women_underwear"} and inputs.agnostic_image is not None:
            top_image = inputs.agnostic_image.convert("RGB")
            warnings.append("top reference uses agnostic person image for underwear-bottom preservation")
        elif inputs.category in {"men_underwear", "women_underwear"}:
            top_image = person.copy()
            warnings.append("top reference copied from person image because agnostic image is unavailable")
        else:
            top_image = inputs.garment_image.convert("RGB")
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
            **self._tensorrt_payload(),
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
        deterministic: bool,
        progress_callback: Any = None,
    ) -> TryOnResult:
        if self.config.entrypoint:
            return self._run_diffusers_local_subprocess(
                prompt,
                references,
                output_dir,
                seed,
                deterministic,
                progress_callback=progress_callback,
            )
        return self._run_diffusers_local_in_process(
            prompt,
            references,
            output_dir,
            seed,
            deterministic,
            progress_callback=progress_callback,
        )

    def _run_diffusers_local_subprocess(
        self,
        prompt: str,
        references: KleinReferences,
        output_dir: Path,
        seed: int | None,
        deterministic: bool,
        progress_callback: Any = None,
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
            **self._model_payload(),
            "width": width,
            "height": height,
            "steps": self.steps,
            "guidance_scale": self.config.guidance_scale,
            "seed": seed,
            "deterministic": deterministic,
            "image_paths": image_paths,
        }
        request_path = output_dir / "local_worker_request.json"
        request_path.write_text(json.dumps(_sanitize_payload(request_payload), indent=2), encoding="utf-8")
        self._save_json_aliases(output_dir, "local_generation", request_payload)

        started = time.perf_counter()
        client = _get_resident_client(self.config, self.local_worker_python)
        try:
            worker_payload = client.request(
                "run",
                request_payload,
                output_dir=output_dir,
                progress_callback=progress_callback,
            )
        finally:
            (output_dir / "local_worker_stdout.txt").write_text(client.stdout_tail()[-8000:], encoding="utf-8")
            (output_dir / "local_worker_stderr.txt").write_text(client.stderr_tail()[-8000:], encoding="utf-8")
        runtime_seconds = time.perf_counter() - started
        result_path = Path(worker_payload.get("result_path", output_dir / "klein_lora_result.png"))
        image = Image.open(result_path).convert("RGB")
        save_image(image, output_dir / "result.png")
        metadata = {
            "engine": self.name,
            "backend": "diffusers_local",
            "runtime_seconds": round(runtime_seconds, 3),
            "worker": worker_payload,
            "resident_worker": client.status(),
            "model_dir": str(self.local_model_dir),
            "lora_path": str(self.config.lora_path),
            "lora_scale": self.config.lora_scale,
            "steps": self.steps,
            "guidance_scale": self.config.guidance_scale,
            "device_map": self.config.device_map,
            "quantization": self.config.quantization,
            "quantize_components": list(self.config.quantize_components),
            **self._tensorrt_payload(),
            "width": width,
            "height": height,
            "seed": seed,
            "deterministic": deterministic,
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
        deterministic: bool,
        progress_callback: Any = None,
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
                "deterministic": deterministic,
                "device_map": self.config.device_map,
                "quantization": self.config.quantization,
                "quantize_components": list(self.config.quantize_components),
                **self._tensorrt_payload(),
                "local_files_only": True,
            },
        )

        if progress_callback is not None:
            progress_callback("loading_model", "running")
        load_started = time.perf_counter()
        pipe = self._load_diffusers_local_pipe()
        load_seconds = time.perf_counter() - load_started
        if progress_callback is not None:
            progress_callback("loading_model", "completed")
        generator_device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(int(seed or 0))
        if progress_callback is not None:
            progress_callback("generating", "running")
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
        if progress_callback is not None:
            progress_callback("generating", "completed")
        image = output.images[0].convert("RGB")
        save_image(image, output_dir / "klein_lora_result.png")
        save_image(image, output_dir / "result.png")
        metadata = {
            "engine": self.name,
            "backend": "diffusers_local",
            "runtime_seconds": round(runtime_seconds, 3),
            "load_model_seconds": round(load_seconds, 3),
            "model_dir": str(self.local_model_dir),
            "lora_path": str(self.config.lora_path),
            "lora_scale": self.config.lora_scale,
            "steps": self.steps,
            "guidance_scale": self.config.guidance_scale,
            "device_map": self.config.device_map,
            "quantization": self.config.quantization,
            "quantize_components": list(self.config.quantize_components),
            **self._tensorrt_payload(),
            "tensorrt": getattr(pipe, "_vton_tensorrt_metadata", {}),
            "width": width,
            "height": height,
            "seed": seed,
            "deterministic": deterministic,
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
                result = self._run_diffusers_local(
                    prompt,
                    references,
                    output_dir,
                    inputs.seed,
                    bool(inputs.extra.get("deterministic", False)),
                    progress_callback=inputs.extra.get("progress_callback"),
                )
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
