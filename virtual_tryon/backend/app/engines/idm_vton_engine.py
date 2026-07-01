from __future__ import annotations

import atexit
import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.core.config import EngineConfig
from app.core.paths import PROJECT_ROOT
from app.engines.base import TryOnInputs, TryOnResult
from app.preprocessing.image_loader import fit_to_canvas
from app.utils.errors import EngineExecutionError, ModelUnavailableError
from app.utils.image_io import open_rgb, save_image


logger = logging.getLogger(__name__)


REQUIRED_CHECKPOINTS = (
    "densepose/model_final_162be9.pkl",
    "humanparsing/parsing_atr.onnx",
    "humanparsing/parsing_lip.onnx",
    "openpose/ckpts/body_pose_model.pth",
)
RESIDENT_PROTOCOL_PREFIX = "__IDM_VTON_WORKER__ "
TENSORRT_RESIDENT_ENV_KEYS = (
    "TRYON_TRT_PROFILE",
    "TRYON_TRT_MODULES",
    "TRYON_TRT_PARTITION_PRESET",
    "TRYON_TRT_TORCH_EXECUTED_OPS",
    "TRYON_TRT_MIN_BLOCK_SIZE",
    "TRYON_TRT_WORKSPACE_SIZE",
    "TRYON_TRT_OPTIMIZATION_LEVEL",
    "TRYON_TRT_ENGINE_CACHE_DIR",
    "TRYON_TRT_ENABLE_RESOURCE_PARTITIONING",
    "TRYON_TRT_CPU_MEMORY_BUDGET",
    "TRYON_TRT_LAZY_ENGINE_INIT",
    "TRYON_TRT_PASS_THROUGH_BUILD_FAILURES",
    "TRYON_TRT_USE_FAST_PARTITIONER",
    "TRYON_TRT_ALLOW_UNSAFE_UNET",
)


def _worker_script_for_config(config: EngineConfig) -> Path:
    return config.resident_worker_entrypoint or PROJECT_ROOT / "scripts" / "idm_vton_resident_worker.py"


class IDMVTonResidentClient:
    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.worker_script = _worker_script_for_config(config)
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
        if not self.worker_script.exists():
            raise ModelUnavailableError(f"IDM-VTON resident worker script not found: {self.worker_script}")
        if not self.config.repo_path:
            raise ModelUnavailableError("IDM-VTON repo_path is not configured.")

        self._stdout_queue = queue.Queue()
        self._stdout_tail.clear()
        self._stderr_tail.clear()
        command = [
            sys.executable,
            str(self.worker_script),
            "--repo-path",
            str(self.config.repo_path),
            "--model-name",
            self.config.model_name or "yisol/IDM-VTON",
            "--device",
            "cuda:0",
            "--optimization-mode",
            self.config.resident_worker_optimization,
            "--torch-compile-backend",
            self.config.resident_worker_torch_compile_backend,
            "--torch-compile-mode",
            self.config.resident_worker_torch_compile_mode,
        ]
        env = {
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
        self._process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
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
            timeout_seconds=self.config.resident_worker_startup_timeout_seconds,
        )
        if not ready.get("ok"):
            error = ready.get("error") or "unknown startup error"
            stderr_tail = self.stderr_tail()
            self.stop()
            raise EngineExecutionError(f"IDM-VTON resident worker failed to start: {error}\n{stderr_tail[-2000:]}")

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
        expected_type: str,
        request_id: str | None,
        timeout_seconds: int,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.stop()
                raise EngineExecutionError(
                    f"Timed out waiting for IDM-VTON resident worker {expected_type} message after "
                    f"{timeout_seconds}s. stderr_tail={self.stderr_tail()[-2000:]}"
                )
            try:
                line = self._stdout_queue.get(timeout=min(0.5, remaining))
            except queue.Empty:
                if self._process is not None and self._process.poll() is not None:
                    raise EngineExecutionError(
                        f"IDM-VTON resident worker exited before {expected_type}. "
                        f"returncode={self._process.returncode} stderr_tail={self.stderr_tail()[-2000:]}"
                    )
                continue

            if not line.startswith(RESIDENT_PROTOCOL_PREFIX):
                continue
            try:
                payload = json.loads(line[len(RESIDENT_PROTOCOL_PREFIX) :])
            except json.JSONDecodeError:
                continue
            if payload.get("type") != expected_type:
                continue
            if request_id is not None and payload.get("request_id") != request_id:
                continue
            return payload

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self.start()
            if self._process is None or self._process.stdin is None:
                raise EngineExecutionError("IDM-VTON resident worker stdin is unavailable.")
            request_id = uuid4().hex
            request = {"type": "run", "request_id": request_id, **payload}
            self._process.stdin.write(json.dumps(request, separators=(",", ":"), default=str) + "\n")
            self._process.stdin.flush()
            response = self._read_protocol_message(
                expected_type="result",
                request_id=request_id,
                timeout_seconds=self.config.resident_worker_request_timeout_seconds,
            )
            if not response.get("ok"):
                raise EngineExecutionError(
                    "IDM-VTON resident worker request failed. "
                    f"error={response.get('error')} stderr_tail={self.stderr_tail()[-2000:]}"
                )
            return response

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


_RESIDENT_CLIENTS: dict[str, IDMVTonResidentClient] = {}
_RESIDENT_CLIENTS_LOCK = threading.RLock()
_RESIDENT_ATEXIT_REGISTERED = False


def _resident_client_key(config: EngineConfig) -> str:
    tensorrt_env = (
        {key: os.environ.get(key) for key in TENSORRT_RESIDENT_ENV_KEYS}
        if config.resident_worker_optimization == "tensorrt"
        else {}
    )
    return json.dumps(
        {
            "python": sys.executable,
            "repo_path": str(config.repo_path),
            "model_name": config.model_name or "yisol/IDM-VTON",
            "worker_script": str(_worker_script_for_config(config)),
            "optimization": config.resident_worker_optimization,
            "torch_compile_backend": config.resident_worker_torch_compile_backend,
            "torch_compile_mode": config.resident_worker_torch_compile_mode,
            "tensorrt_env": tensorrt_env,
        },
        sort_keys=True,
    )


def _stop_resident_clients() -> None:
    with _RESIDENT_CLIENTS_LOCK:
        for client in _RESIDENT_CLIENTS.values():
            client.stop()


def stop_resident_clients() -> None:
    _stop_resident_clients()


def _get_resident_client(config: EngineConfig) -> IDMVTonResidentClient:
    global _RESIDENT_ATEXIT_REGISTERED
    key = _resident_client_key(config)
    with _RESIDENT_CLIENTS_LOCK:
        if not _RESIDENT_ATEXIT_REGISTERED:
            atexit.register(_stop_resident_clients)
            _RESIDENT_ATEXIT_REGISTERED = True
        client = _RESIDENT_CLIENTS.get(key)
        if client is None:
            client = IDMVTonResidentClient(config)
            _RESIDENT_CLIENTS[key] = client
        return client


def _resident_status(config: EngineConfig) -> str:
    key = _resident_client_key(config)
    with _RESIDENT_CLIENTS_LOCK:
        client = _RESIDENT_CLIENTS.get(key)
        return client.status() if client is not None else "not_started"


@dataclass(frozen=True)
class IDMVTonRunContext:
    data_dir: Path
    output_dir: Path
    person_name: str
    garment_name: str
    expected_output: Path
    command: list[str]


class IDMVTonEngine:
    name = "idm_vton"

    def __init__(self, config: EngineConfig) -> None:
        self.config = config

    def missing_requirements(self) -> list[str]:
        missing: list[str] = []
        if not self.config.enabled:
            missing.append("idm_vton.enabled is false")
        if not self.config.repo_path or not self.config.repo_path.exists():
            missing.append(f"repo_path not found: {self.config.repo_path}")
        if not self.config.entrypoint or not self.config.entrypoint.exists():
            missing.append(f"entrypoint not found: {self.config.entrypoint}")
        if self.config.resident_worker:
            worker_script = _worker_script_for_config(self.config)
            if not worker_script.exists():
                missing.append(f"resident worker script not found: {worker_script}")
        if not self.config.checkpoint_dir:
            missing.append("checkpoint_dir is not configured")
        else:
            if not self.config.checkpoint_dir.exists():
                missing.append(f"IDM-VTON checkpoint not found at {self.config.checkpoint_dir}")
            for rel_path in REQUIRED_CHECKPOINTS:
                checkpoint = self.config.checkpoint_dir / rel_path
                if not checkpoint.exists():
                    missing.append(f"missing checkpoint: {rel_path}")
                elif checkpoint.stat().st_size < 1024:
                    missing.append(f"checkpoint looks incomplete: {rel_path}")
        try:
            import accelerate  # noqa: F401
        except Exception:
            missing.append("python package missing: accelerate")
        return missing

    def status(self) -> str:
        missing = self.missing_requirements()
        if missing:
            return "missing: " + "; ".join(missing)
        if self.config.resident_worker:
            return (
                f"available; resident_worker={_resident_status(self.config)}; "
                f"optimization={self.config.resident_worker_optimization}"
            )
        return "available"

    def is_available(self) -> bool:
        return not self.missing_requirements()

    def prepare(self) -> None:
        missing = self.missing_requirements()
        if missing:
            raise ModelUnavailableError(
                "IDM-VTON is not available. " + "; ".join(missing)
            )

    def build_dataset(self, inputs: TryOnInputs) -> IDMVTonRunContext:
        if inputs.output_dir is None:
            raise EngineExecutionError("IDM-VTON requires inputs.output_dir for dataset staging.")
        job_dir = Path(inputs.output_dir)
        data_dir = job_dir / "idm_vton_dataset"
        output_dir = job_dir / "idm_vton_result"
        test_dir = data_dir / "test"

        for folder in ["image", "cloth", "agnostic-mask", "image-densepose"]:
            (test_dir / folder).mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        person_name = "person_0001.jpg"
        garment_name = "garment_0001.jpg"
        width = self.config.default_width
        height = self.config.default_height

        person = fit_to_canvas(inputs.person_image, width, height)
        garment = fit_to_canvas(inputs.garment_image, width, height)
        save_image(person, test_dir / "image" / person_name)
        save_image(garment, test_dir / "cloth" / garment_name)

        mask_name = person_name.replace(".jpg", "_mask.png")
        mask = inputs.agnostic_mask.convert("L").resize((width, height))
        save_image(mask, test_dir / "agnostic-mask" / mask_name)

        if inputs.densepose_image is not None:
            densepose = fit_to_canvas(inputs.densepose_image, width, height)
        else:
            logger.warning("No densepose image provided; using person image as densepose placeholder.")
            densepose = person
        save_image(densepose, test_dir / "image-densepose" / person_name)
        save_image(densepose, job_dir / "densepose.png")

        tags = self._tags_for_category(inputs.category, inputs.prompt)
        manifest = {"data": [{"file_name": garment_name, "category_name": "TOPS", "tag_info": tags}]}
        (test_dir / "vitonhd_test_tagged.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (data_dir / "test_pairs.txt").write_text(f"{person_name} {garment_name}\n", encoding="utf-8")

        command = self.build_command(data_dir, output_dir, inputs.seed)
        return IDMVTonRunContext(
            data_dir=data_dir,
            output_dir=output_dir,
            person_name=person_name,
            garment_name=garment_name,
            expected_output=output_dir / person_name,
            command=command,
        )

    def build_command(self, data_dir: Path, output_dir: Path, seed: int | None) -> list[str]:
        entrypoint = self.config.entrypoint
        if entrypoint is None:
            raise ModelUnavailableError("IDM-VTON entrypoint is not configured.")
        command = [
            sys.executable,
            "-m",
            "accelerate.commands.launch",
            str(entrypoint),
            "--pretrained_model_name_or_path",
            self.config.model_name or "yisol/IDM-VTON",
            "--width",
            str(self.config.default_width),
            "--height",
            str(self.config.default_height),
            "--num_inference_steps",
            str(self.config.steps),
            "--output_dir",
            str(output_dir),
            "--unpaired",
            "--data_dir",
            str(data_dir),
            "--seed",
            str(seed if seed is not None else 42),
            "--test_batch_size",
            "1",
            "--guidance_scale",
            str(self.config.guidance_scale),
        ]
        return command

    def build_resident_request(
        self,
        context: IDMVTonRunContext,
        seed: int | None,
        deterministic: bool = False,
    ) -> dict[str, Any]:
        return {
            "data_dir": str(context.data_dir),
            "output_dir": str(context.output_dir),
            "width": self.config.default_width,
            "height": self.config.default_height,
            "num_inference_steps": self.config.steps,
            "seed": seed if seed is not None else 42,
            "test_batch_size": 1,
            "guidance_scale": self.config.guidance_scale,
            "unpaired": True,
            "optimization_mode": self.config.resident_worker_optimization,
            "deterministic": deterministic,
        }

    @staticmethod
    def _tags_for_category(category: str, prompt: str | None) -> list[dict[str, str | None]]:
        item = {
            "upper_body": "shirts",
            "lower_body": "pants",
            "dress": "dress",
            "full_outfit": "outfit",
            "men_underwear": "pants",
            "women_underwear": "pants",
            "women_bra": "shirts",
        }.get(category, "shirts")
        default_details = {
            "men_underwear": "adult men's brief underwear",
            "women_underwear": "adult women's brief underwear",
            "women_bra": "adult women's bra or upper innerwear",
        }
        prompt_value = prompt or default_details.get(category, item)
        return [
            {"tag_name": "item", "tag_category": item},
            {"tag_name": "sleeveLength", "tag_category": "regular"},
            {"tag_name": "neckLine", "tag_category": "regular"},
            {"tag_name": "details", "tag_category": prompt_value[:80]},
            {"tag_name": "colors", "tag_category": None},
            {"tag_name": "textures", "tag_category": None},
        ]

    def _run_subprocess(self, context: IDMVTonRunContext, job_dir: Path) -> str:
        command_text = " ".join(str(part) for part in context.command)
        (job_dir / "idm_vton_command.txt").write_text(command_text, encoding="utf-8")
        logger.info("Running IDM-VTON command: %s", command_text)
        completed = subprocess.run(
            context.command,
            cwd=self.config.repo_path,
            capture_output=True,
            text=True,
            check=False,
        )
        (job_dir / "idm_vton_stdout.txt").write_text(completed.stdout or "", encoding="utf-8")
        (job_dir / "idm_vton_stderr.txt").write_text(completed.stderr or "", encoding="utf-8")
        if completed.returncode != 0:
            raise EngineExecutionError(
                "IDM-VTON execution failed. "
                f"stdout={completed.stdout[-1000:]} stderr={completed.stderr[-1000:]}"
            )
        return command_text

    def _run_resident(
        self,
        context: IDMVTonRunContext,
        job_dir: Path,
        seed: int | None,
        deterministic: bool = False,
    ) -> str:
        request = self.build_resident_request(context, seed, deterministic)
        command_text = "resident-idm-vton-worker " + json.dumps(request, sort_keys=True, default=str)
        (job_dir / "idm_vton_command.txt").write_text(command_text, encoding="utf-8")
        (job_dir / "idm_vton_worker_request.json").write_text(json.dumps(request, indent=2, default=str), encoding="utf-8")
        logger.info("Running IDM-VTON resident worker request: %s", command_text)
        client = _get_resident_client(self.config)
        response = client.run(request)
        (job_dir / "idm_vton_worker_response.json").write_text(
            json.dumps(response, indent=2, default=str),
            encoding="utf-8",
        )
        (job_dir / "idm_vton_stdout.txt").write_text(client.stdout_tail(), encoding="utf-8")
        (job_dir / "idm_vton_stderr.txt").write_text(client.stderr_tail(), encoding="utf-8")
        return command_text

    def run(self, inputs: TryOnInputs) -> TryOnResult:
        start = time.perf_counter()
        self.prepare()
        context = self.build_dataset(inputs)
        job_dir = inputs.output_dir or Path.cwd()
        runtime_backend = "subprocess"
        if self.config.resident_worker:
            try:
                command_text = self._run_resident(
                    context,
                    job_dir,
                    inputs.seed,
                    bool(inputs.extra.get("deterministic", False)),
                )
                runtime_backend = "resident_worker"
            except Exception as exc:
                (job_dir / "idm_vton_resident_error.txt").write_text(str(exc), encoding="utf-8")
                if not self.config.resident_worker_fallback:
                    raise
                logger.warning("IDM-VTON resident worker failed; falling back to subprocess: %s", exc)
                command_text = self._run_subprocess(context, job_dir)
                runtime_backend = "subprocess_fallback"
        else:
            command_text = self._run_subprocess(context, job_dir)

        if not context.expected_output.exists():
            raise EngineExecutionError(f"IDM-VTON finished but did not create output at {context.expected_output}")

        core_output = Path(job_dir) / "core_output.png"
        image = open_rgb(context.expected_output)
        save_image(image, core_output)
        elapsed = time.perf_counter() - start
        logger.info("IDM-VTON completed in %.2fs", elapsed)
        return TryOnResult(
            image,
            {
                "engine": self.name,
                "runtime_seconds": elapsed,
                "runtime_backend": runtime_backend,
                "resident_worker_enabled": self.config.resident_worker,
                "command": command_text,
                "data_dir": str(context.data_dir),
                "output_dir": str(context.output_dir),
            },
        )
