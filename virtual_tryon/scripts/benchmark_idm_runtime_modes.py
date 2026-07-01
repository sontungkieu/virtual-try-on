#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.util
import json
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import load_settings  # noqa: E402
from app.engines.base import TryOnInputs  # noqa: E402
from app.engines.idm_vton_engine import IDMVTonEngine, IDMVTonResidentClient  # noqa: E402
from app.preprocessing.agnostic_mask import create_agnostic_mask  # noqa: E402
from app.preprocessing.image_loader import load_image_from_path  # noqa: E402
from app.utils.image_io import open_rgb, save_image  # noqa: E402


DEFAULT_PRESETS = "512x768,640x896,768x1024"
DEFAULT_PROMPT = "replace the shirt with the reference garment, preserve face, pose, and background"
CSV_COLUMNS = [
    "mode",
    "preset",
    "width",
    "height",
    "steps",
    "phase",
    "iteration",
    "status",
    "wall_seconds",
    "worker_runtime_seconds",
    "load_runtime_seconds",
    "vram_before_mb",
    "vram_after_mb",
    "output_path",
    "error",
]
TENSORRT_ENV_KEYS = (
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


def _parse_presets(value: str) -> list[tuple[int, int]]:
    presets: list[tuple[int, int]] = []
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        try:
            width_text, height_text = item.split("x", 1)
            width = int(width_text)
            height = int(height_text)
        except ValueError as exc:
            raise SystemExit(f"Invalid preset '{item}'. Use WIDTHxHEIGHT.") from exc
        presets.append((width, height))
    if not presets:
        raise SystemExit("No presets were provided.")
    return presets


def _parse_modes(value: str) -> list[str]:
    supported = {"eager", "torch_compile", "tensorrt"}
    modes = [item.strip() for item in value.split(",") if item.strip()]
    invalid = sorted(set(modes) - supported)
    if invalid:
        raise SystemExit(f"Unsupported mode(s): {', '.join(invalid)}")
    return modes or ["eager", "torch_compile", "tensorrt"]


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _vram_mb(pid: int | None = None) -> int | None:
    import subprocess

    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    total = 0
    seen = False
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            row_pid = int(parts[0])
            memory = int(parts[1])
        except ValueError:
            continue
        if pid is None or row_pid == pid:
            total += memory
            seen = True
    return total if seen else None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in CSV_COLUMNS})


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("phase") != "measure" or row.get("status") != "completed":
            continue
        grouped.setdefault((row["mode"], row["preset"]), []).append(row)

    summary: list[dict[str, Any]] = []
    for (mode, preset), group in sorted(grouped.items()):
        wall = [float(row["wall_seconds"]) for row in group if row.get("wall_seconds") is not None]
        worker = [
            float(row["worker_runtime_seconds"])
            for row in group
            if row.get("worker_runtime_seconds") is not None
        ]
        summary.append(
            {
                "mode": mode,
                "preset": preset,
                "runs": len(group),
                "wall_seconds_median": round(statistics.median(wall), 4) if wall else None,
                "wall_seconds_mean": round(statistics.fmean(wall), 4) if wall else None,
                "worker_runtime_seconds_median": round(statistics.median(worker), 4) if worker else None,
                "worker_runtime_seconds_mean": round(statistics.fmean(worker), 4) if worker else None,
            }
        )
    return summary


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "mode",
        "preset",
        "runs",
        "wall_seconds_median",
        "wall_seconds_mean",
        "worker_runtime_seconds_median",
        "worker_runtime_seconds_mean",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _fallback_image_path(kind: str) -> Path | None:
    candidates: list[Path] = []
    if kind == "person":
        candidates.extend(
            [
                PROJECT_ROOT / "data/eval_set/sample_001/person.jpg",
                PROJECT_ROOT / "data/eval_set/sample_001/person.png",
            ]
        )
        candidates.extend(sorted(PROJECT_ROOT.glob("data/outputs/*/person.png")))
    else:
        candidates.extend(
            [
                PROJECT_ROOT / "data/eval_set/sample_001/garment_top.jpg",
                PROJECT_ROOT / "data/eval_set/sample_001/garment_top.png",
            ]
        )
        candidates.extend(sorted(PROJECT_ROOT.glob("data/outputs/*/garment_top.png")))
    return next((path for path in candidates if path.exists()), None)


def _resolve_input_path(path: Path, kind: str) -> Path:
    if path.exists():
        return path
    fallback = _fallback_image_path(kind)
    if fallback is not None:
        return fallback
    raise SystemExit(f"Input image not found: {path}. Pass --{kind.replace('_', '-')} explicitly.")


def _load_inputs(args: argparse.Namespace):
    args.person = _resolve_input_path(args.person, "person")
    args.garment_top = _resolve_input_path(args.garment_top, "garment_top")
    person = load_image_from_path(args.person, max_side=args.max_side)
    garment = load_image_from_path(args.garment_top, max_side=args.max_side)
    return person, garment


def _make_tryon_inputs(
    *,
    person: Image.Image,
    garment: Image.Image,
    category: str,
    prompt: str,
    seed: int,
    output_dir: Path,
    settings,
) -> TryOnInputs:
    mask_result = create_agnostic_mask(person, category, settings.preprocessing)
    return TryOnInputs(
        person_image=person,
        garment_image=garment,
        category=category,
        agnostic_mask=mask_result.soft_mask,
        agnostic_image=mask_result.agnostic_image,
        prompt=prompt,
        seed=seed,
        output_dir=output_dir,
    )


def _skip_rows_for_mode(
    *,
    mode: str,
    presets: list[tuple[int, int]],
    steps: int,
    reason: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for width, height in presets:
        rows.append(
            {
                "mode": mode,
                "preset": f"{width}x{height}",
                "width": width,
                "height": height,
                "steps": steps,
                "phase": "measure",
                "iteration": 0,
                "status": "skipped",
                "error": reason,
            }
        )
    return rows


def _tensorrt_skip_reason() -> str | None:
    if importlib.util.find_spec("torch_tensorrt") is None and importlib.util.find_spec("tensorrt") is None:
        return "Neither torch_tensorrt nor tensorrt is installed."
    return None


@contextlib.contextmanager
def _temporary_env(overrides: dict[str, str | None]):
    previous = {key: os.environ.get(key) for key in overrides}
    try:
        for key, value in overrides.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _tensorrt_env_overrides(args: argparse.Namespace, output_dir: Path) -> dict[str, str]:
    overrides: dict[str, str] = {}
    if args.tensorrt_profile:
        overrides["TRYON_TRT_PROFILE"] = args.tensorrt_profile
    if args.tensorrt_modules:
        overrides["TRYON_TRT_MODULES"] = args.tensorrt_modules
    if args.tensorrt_partition_preset:
        overrides["TRYON_TRT_PARTITION_PRESET"] = args.tensorrt_partition_preset
    if args.tensorrt_torch_executed_ops:
        overrides["TRYON_TRT_TORCH_EXECUTED_OPS"] = args.tensorrt_torch_executed_ops
    if args.tensorrt_min_block_size is not None:
        overrides["TRYON_TRT_MIN_BLOCK_SIZE"] = str(args.tensorrt_min_block_size)
    if args.tensorrt_workspace_size is not None:
        overrides["TRYON_TRT_WORKSPACE_SIZE"] = str(args.tensorrt_workspace_size)
    if args.tensorrt_optimization_level is not None:
        overrides["TRYON_TRT_OPTIMIZATION_LEVEL"] = str(args.tensorrt_optimization_level)
    if args.tensorrt_enable_resource_partitioning:
        overrides["TRYON_TRT_ENABLE_RESOURCE_PARTITIONING"] = "true"
    if args.tensorrt_cpu_memory_budget is not None:
        overrides["TRYON_TRT_CPU_MEMORY_BUDGET"] = str(args.tensorrt_cpu_memory_budget)
    if args.tensorrt_lazy_engine_init:
        overrides["TRYON_TRT_LAZY_ENGINE_INIT"] = "true"
    if args.tensorrt_pass_through_build_failures:
        overrides["TRYON_TRT_PASS_THROUGH_BUILD_FAILURES"] = "true"
    if args.tensorrt_disable_fast_partitioner:
        overrides["TRYON_TRT_USE_FAST_PARTITIONER"] = "false"
    if args.allow_unsafe_tensorrt_unet:
        overrides["TRYON_TRT_ALLOW_UNSAFE_UNET"] = "true"
    cache_dir = args.tensorrt_engine_cache_dir or (output_dir / "tensorrt_engine_cache")
    if not cache_dir.is_absolute():
        cache_dir = PROJECT_ROOT / cache_dir
    cache_dir = cache_dir.resolve()
    overrides["TRYON_TRT_ENGINE_CACHE_DIR"] = str(cache_dir)
    return overrides


def _run_mode(
    *,
    mode: str,
    presets: list[tuple[int, int]],
    args: argparse.Namespace,
    output_dir: Path,
    settings,
    person: Image.Image,
    garment: Image.Image,
) -> list[dict[str, Any]]:
    mode_dir = output_dir / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    if mode == "tensorrt":
        reason = _tensorrt_skip_reason()
        if reason:
            return _skip_rows_for_mode(mode=mode, presets=presets, steps=args.steps, reason=reason)
        tensorrt_overrides = _tensorrt_env_overrides(args, output_dir)
        (mode_dir / "tensorrt_env.json").write_text(json.dumps(tensorrt_overrides, indent=2), encoding="utf-8")
    else:
        tensorrt_overrides = {}

    config = settings.idm_vton.model_copy(deep=True)
    config.resident_worker = True
    config.resident_worker_fallback = False
    config.resident_worker_optimization = mode
    config.resident_worker_startup_timeout_seconds = args.startup_timeout
    config.resident_worker_request_timeout_seconds = args.request_timeout

    client = IDMVTonResidentClient(config)
    with _temporary_env(tensorrt_overrides):
        ready_started = time.perf_counter()
        try:
            client.start()
            load_runtime = time.perf_counter() - ready_started
            worker_pid = client._process.pid if client._process is not None else None
            (mode_dir / "worker_stdout.txt").write_text(client.stdout_tail(), encoding="utf-8")
            (mode_dir / "worker_stderr.txt").write_text(client.stderr_tail(), encoding="utf-8")
        except Exception as exc:
            (mode_dir / "startup_error.txt").write_text(str(exc), encoding="utf-8")
            client.stop()
            return _skip_rows_for_mode(mode=mode, presets=presets, steps=args.steps, reason=f"startup failed: {exc}")

        try:
            for width, height in presets:
                preset_name = f"{width}x{height}"
                config.default_width = width
                config.default_height = height
                config.steps = args.steps
                engine = IDMVTonEngine(config)
                phases = [("warmup", idx) for idx in range(args.warmup)] + [
                    ("measure", idx) for idx in range(args.repeat)
                ]
                skip_measure_reason: str | None = None
                for phase, iteration in phases:
                    run_dir = mode_dir / preset_name / f"{phase}_{iteration + 1:02d}"
                    run_dir.mkdir(parents=True, exist_ok=True)
                    seed = args.seed + iteration + (0 if phase == "measure" else 10_000)
                    if phase == "measure" and skip_measure_reason:
                        row = {
                            "mode": mode,
                            "preset": preset_name,
                            "width": width,
                            "height": height,
                            "steps": args.steps,
                            "phase": phase,
                            "iteration": iteration + 1,
                            "status": "skipped",
                            "load_runtime_seconds": round(load_runtime, 4),
                            "vram_before_mb": _vram_mb(worker_pid),
                            "vram_after_mb": _vram_mb(worker_pid),
                            "error": f"warmup failed: {skip_measure_reason}",
                        }
                        rows.append(row)
                        with (output_dir / "runs.jsonl").open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(row, default=str) + "\n")
                        continue
                    inputs = _make_tryon_inputs(
                        person=person,
                        garment=garment,
                        category=args.category,
                        prompt=args.prompt,
                        seed=seed,
                        output_dir=run_dir,
                        settings=settings,
                    )
                    context = engine.build_dataset(inputs)
                    request = engine.build_resident_request(context, seed)
                    request_path = run_dir / "idm_vton_worker_request.json"
                    request_path.write_text(json.dumps(request, indent=2, default=str), encoding="utf-8")
                    vram_before = _vram_mb(worker_pid)
                    started = time.perf_counter()
                    row = {
                        "mode": mode,
                        "preset": preset_name,
                        "width": width,
                        "height": height,
                        "steps": args.steps,
                        "phase": phase,
                        "iteration": iteration + 1,
                        "load_runtime_seconds": round(load_runtime, 4),
                        "vram_before_mb": vram_before,
                    }
                    if tensorrt_overrides:
                        row["tensorrt_env"] = dict(tensorrt_overrides)
                    try:
                        response = client.run(request)
                        wall_seconds = time.perf_counter() - started
                        response_path = run_dir / "idm_vton_worker_response.json"
                        response_path.write_text(json.dumps(response, indent=2, default=str), encoding="utf-8")
                        output_image = open_rgb(context.expected_output)
                        output_path = save_image(output_image, run_dir / "core_output.png")
                        row.update(
                            {
                                "status": "completed",
                                "wall_seconds": round(wall_seconds, 4),
                                "worker_runtime_seconds": round(float(response.get("runtime_seconds", 0.0)), 4),
                                "vram_after_mb": _vram_mb(worker_pid),
                                "output_path": _relative(output_path, output_dir),
                                "error": None,
                            }
                        )
                    except Exception as exc:
                        if phase == "warmup":
                            skip_measure_reason = str(exc)
                        row.update(
                            {
                                "status": "failed",
                                "wall_seconds": round(time.perf_counter() - started, 4),
                                "vram_after_mb": _vram_mb(worker_pid),
                                "error": str(exc),
                            }
                        )
                    rows.append(row)
                    with (output_dir / "runs.jsonl").open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(row, default=str) + "\n")
        finally:
            (mode_dir / "worker_stdout.txt").write_text(client.stdout_tail(), encoding="utf-8")
            (mode_dir / "worker_stderr.txt").write_text(client.stderr_tail(), encoding="utf-8")
            client.stop()
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark IDM-VTON resident runtime optimization modes.")
    parser.add_argument("--person", type=Path, default=PROJECT_ROOT / "data/inputs/smoke_api/person.png")
    parser.add_argument("--garment-top", type=Path, default=PROJECT_ROOT / "data/inputs/smoke_api/garment_top.png")
    parser.add_argument("--category", default="upper_body")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--modes", default="eager,torch_compile,tensorrt")
    parser.add_argument("--presets", default=DEFAULT_PRESETS)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-side", type=int, default=1536)
    parser.add_argument("--startup-timeout", type=int, default=900)
    parser.add_argument("--request-timeout", type=int, default=900)
    parser.add_argument("--tensorrt-modules", default=None)
    parser.add_argument(
        "--tensorrt-profile",
        default=None,
        help=(
            "TensorRT worker profile. Use stable for VAE decode only, full_safe for blockwise "
            "UNet+UNet-encoder+VAE with conservative partitioning, or whole_unet_debug for isolated debugging."
        ),
    )
    parser.add_argument("--tensorrt-partition-preset", default=None)
    parser.add_argument("--tensorrt-torch-executed-ops", default=None)
    parser.add_argument("--tensorrt-min-block-size", type=int, default=None)
    parser.add_argument("--tensorrt-workspace-size", type=int, default=None)
    parser.add_argument("--tensorrt-optimization-level", type=int, default=None)
    parser.add_argument("--tensorrt-engine-cache-dir", type=Path, default=None)
    parser.add_argument("--tensorrt-enable-resource-partitioning", action="store_true")
    parser.add_argument("--tensorrt-cpu-memory-budget", type=int, default=None)
    parser.add_argument("--tensorrt-lazy-engine-init", action="store_true")
    parser.add_argument("--tensorrt-pass-through-build-failures", action="store_true")
    parser.add_argument("--tensorrt-disable-fast-partitioner", action="store_true")
    parser.add_argument("--allow-unsafe-tensorrt-unet", action="store_true")
    parser.add_argument("--force-tensorrt", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = load_settings()
    presets = _parse_presets(args.presets)
    modes = _parse_modes(args.modes)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output or settings.storage.outputs_dir / f"idm_runtime_benchmark_{timestamp}"
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "runs.jsonl").write_text("", encoding="utf-8")

    person, garment = _load_inputs(args)
    config_payload = {
        "created_at": datetime.now().isoformat(),
        "modes": modes,
        "presets": [f"{width}x{height}" for width, height in presets],
        "steps": args.steps,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "person": str(args.person),
        "garment_top": str(args.garment_top),
        "tensorrt": {
            "profile": args.tensorrt_profile,
            "modules": args.tensorrt_modules,
            "partition_preset": args.tensorrt_partition_preset,
            "torch_executed_ops": args.tensorrt_torch_executed_ops,
            "min_block_size": args.tensorrt_min_block_size,
            "workspace_size": args.tensorrt_workspace_size,
            "optimization_level": args.tensorrt_optimization_level,
            "engine_cache_dir": str(args.tensorrt_engine_cache_dir) if args.tensorrt_engine_cache_dir else None,
            "enable_resource_partitioning": args.tensorrt_enable_resource_partitioning,
            "cpu_memory_budget": args.tensorrt_cpu_memory_budget,
            "lazy_engine_init": args.tensorrt_lazy_engine_init,
            "pass_through_build_failures": args.tensorrt_pass_through_build_failures,
            "disable_fast_partitioner": args.tensorrt_disable_fast_partitioner,
            "allow_unsafe_unet": args.allow_unsafe_tensorrt_unet,
        },
    }
    (output_dir / "benchmark_config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    rows: list[dict[str, Any]] = []
    for mode in modes:
        mode_rows = _run_mode(
            mode=mode,
            presets=presets,
            args=args,
            output_dir=output_dir,
            settings=settings,
            person=person,
            garment=garment,
        )
        rows.extend(mode_rows)
        _write_csv(output_dir / "runs.csv", rows)
        summary_rows = _summarize(rows)
        _write_summary_csv(output_dir / "summary.csv", summary_rows)
        (output_dir / "summary.json").write_text(
            json.dumps({"config": config_payload, "summary": summary_rows, "runs": rows}, indent=2, default=str),
            encoding="utf-8",
        )

    print(json.dumps({"output_dir": str(output_dir), "summary": _summarize(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
