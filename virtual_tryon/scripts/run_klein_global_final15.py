from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from virtual_tryon.final_demo import DEFAULT_FINAL_EVAL_ROOT, item_description, sample_region

COMFY_ROOT = Path("/workspace/ComfyUI")
COMFY_INPUT = COMFY_ROOT / "input"
COMFY_OUTPUT = COMFY_ROOT / "output"
COMFY_URL = "http://127.0.0.1:8188"
DEFAULT_EVAL_ROOT = DEFAULT_FINAL_EVAL_ROOT
DEFAULT_MODEL_DIR = PROJECT_ROOT / "virtual_tryon/models/flux2-klein-9b"
DEFAULT_LORA_PATH = Path(
    "/workspace/hf-cache/hub/models--fal--flux-klein-9b-virtual-tryon-lora/"
    "snapshots/8b078b15c6d958ce48892b9ef31b66aa7587d792/flux-klein-tryon.safetensors"
)


def post_json(path: str, payload: dict[str, Any], timeout: int = 30) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{COMFY_URL}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def get_json(path: str, timeout: int = 30) -> dict[str, Any]:
    with urllib.request.urlopen(f"{COMFY_URL}{path}", timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def queue_and_wait(prompt: dict[str, Any], timeout_seconds: int = 3600) -> dict[str, Any]:
    response = post_json("/prompt", {"prompt": prompt})
    prompt_id = response["prompt_id"]
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        time.sleep(3)
        history = get_json(f"/history/{prompt_id}")
        if prompt_id not in history:
            continue
        item = history[prompt_id]
        status = item.get("status", {})
        if status.get("completed") or status.get("status_str") == "error":
            return item | {"prompt_id": prompt_id}
    raise TimeoutError(f"ComfyUI prompt timed out: {prompt_id}")


def saved_images(history_item: dict[str, Any]) -> list[Path]:
    images: list[Path] = []
    for output in history_item.get("outputs", {}).values():
        for image in output.get("images") or []:
            subfolder = image.get("subfolder") or ""
            images.append(COMFY_OUTPUT / subfolder / image["filename"])
    return images


def copy_to_comfy_input(path: Path, name: str) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    COMFY_INPUT.mkdir(parents=True, exist_ok=True)
    target = COMFY_INPUT / name
    shutil.copy2(path, target)
    return name


def build_graph(
    *,
    method: str,
    person_name: str,
    reference_name: str,
    target_region: str,
    item_text: str,
    seed: int,
    steps: int,
    guidance: float,
    filename_prefix: str,
    model_dir: Path,
    lora_path: Path,
    lora_strength: float,
) -> dict[str, Any]:
    nodes: dict[str, Any] = {
        "1": {"class_type": "LoadImage", "inputs": {"image": person_name}},
        "2": {"class_type": "LoadImage", "inputs": {"image": reference_name}},
        "3": {"class_type": "VTONPhase2KleinFitCanvas", "inputs": {"image": ["1", 0], "width": 768, "height": 1024}},
        "4": {"class_type": "VTONPhase2KleinFitCanvas", "inputs": {"image": ["2", 0], "width": 768, "height": 1024}},
        "5": {
            "class_type": "VTONPhase2KleinBottomCrop",
            "inputs": {"person_image": ["3", 0], "x0_ratio": 0.08, "y0_ratio": 0.50, "x1_ratio": 0.92, "y1_ratio": 0.98},
        },
        "6": {
            "class_type": "VTONPhase2KleinPromptBuilder",
            "inputs": {
                "target_region": target_region,
                "prompt_strength": "default",
                "item_description": item_text,
            },
        },
        "7": {"class_type": "VTONPhase2KleinLoadBaseModel", "inputs": {"model_dir": str(model_dir)}},
    }
    pipeline_source: list[Any] = ["7", 0]
    sampler_id = "8"
    save_id = "9"
    if method == "lora":
        nodes["8"] = {
            "class_type": "VTONPhase2KleinLoadTryOnLoRA",
            "inputs": {"klein_pipeline": ["7", 0], "lora_path": str(lora_path), "lora_scale": lora_strength},
        }
        pipeline_source = ["8", 0]
        sampler_id = "9"
        save_id = "10"
    nodes[sampler_id] = {
        "class_type": "VTONPhase2KleinSamplerDetailed",
        "inputs": {
            "klein_pipeline": pipeline_source,
            "person_canvas": ["3", 0],
            "top_reference": ["4", 0],
            "bottom_reference": ["5", 0],
            "prompt": ["6", 0],
            "seed": seed,
            "steps": steps,
            "guidance_scale": guidance,
            "width": 768,
            "height": 1024,
        },
    }
    nodes[save_id] = {
        "class_type": "SaveImage",
        "inputs": {"images": [sampler_id, 0], "filename_prefix": filename_prefix},
    }
    return nodes


def check_models(method: str, model_dir: Path, lora_path: Path) -> None:
    errors: list[str] = []
    if not model_dir.exists():
        errors.append(f"Missing FLUX.2 Klein model directory: {model_dir}")
    elif not (model_dir / "model_index.json").exists():
        errors.append(f"FLUX.2 Klein model directory lacks model_index.json: {model_dir}")
    if method == "lora" and not lora_path.exists():
        errors.append(f"Missing Try-On LoRA: {lora_path}")
    if errors:
        raise FileNotFoundError("Model validation failed:\n- " + "\n- ".join(errors))


def run_sample(
    *,
    sample_dir: Path,
    output_root: Path,
    method: str,
    steps: int,
    guidance: float,
    model_dir: Path,
    lora_path: Path,
    lora_strength: float,
) -> dict[str, Any]:
    sample_id = sample_dir.name
    metadata_path = sample_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    person_path = sample_dir / "person.png"
    reference_path = sample_dir / "reference_canvas.png"
    if not reference_path.exists():
        reference_path = sample_dir / str(metadata["items"][0])

    sample_out = output_root / sample_id
    sample_out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(person_path, sample_out / "input_person.png")
    shutil.copy2(reference_path, sample_out / "input_reference.png")

    person_name = copy_to_comfy_input(person_path, f"{method}_{sample_id}_person.png")
    reference_name = copy_to_comfy_input(reference_path, f"{method}_{sample_id}_reference.png")
    seed = 2026063000 + int(sample_id.split("_")[1]) + (1000 if method == "lora" else 0)
    filename_prefix = f"final_output/{method}/{sample_id}/result"
    graph = build_graph(
        method=method,
        person_name=person_name,
        reference_name=reference_name,
        target_region=sample_region(metadata),
        item_text=item_description(metadata),
        seed=seed,
        steps=steps,
        guidance=guidance,
        filename_prefix=filename_prefix,
        model_dir=model_dir,
        lora_path=lora_path,
        lora_strength=lora_strength,
    )
    (sample_out / "workflow_used_api.json").write_text(json.dumps(graph, indent=2), encoding="utf-8")

    started = time.perf_counter()
    history = queue_and_wait(graph, timeout_seconds=3600)
    (sample_out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    status = history.get("status", {})
    if not status.get("completed"):
        raise RuntimeError(f"ComfyUI failed for {method}/{sample_id}: {json.dumps(status)}")

    images = saved_images(history)
    if not images:
        raise RuntimeError(f"No saved image for {method}/{sample_id}")
    output_path = sample_out / "output.png"
    shutil.copy2(images[-1], output_path)
    with Image.open(output_path) as image:
        output_size = list(image.size)

    run_meta = {
        "sample_id": sample_id,
        "method": "FLUX.2 Klein 9B" if method == "base" else "FLUX.2 Klein 9B + Try-On LoRA",
        "person_image": str(person_path),
        "reference_image": str(reference_path),
        "target_region": sample_region(metadata),
        "seed": seed,
        "steps": steps,
        "guidance": guidance,
        "prompt_strength": "default",
        "lora_strength": lora_strength if method == "lora" else 0.0,
        "output": str(output_path),
        "output_size": output_size,
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "model_file_names": {"base_model_dir": str(model_dir), "lora_path": str(lora_path) if method == "lora" else None},
    }
    (sample_out / "metadata.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    print(json.dumps({"sample_id": sample_id, "method": method, "output": str(output_path)}), flush=True)
    return run_meta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--method", choices=["base", "lora"], required=True)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance", type=float, default=2.5)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--lora-path", type=Path, default=DEFAULT_LORA_PATH)
    parser.add_argument("--lora-strength", type=float, default=1.0)
    args = parser.parse_args()

    check_models(args.method, args.model_dir, args.lora_path)
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = [
        run_sample(
            sample_dir=sample_dir,
            output_root=args.output_root,
            method=args.method,
            steps=args.steps,
            guidance=args.guidance,
            model_dir=args.model_dir,
            lora_path=args.lora_path,
            lora_strength=args.lora_strength,
        )
        for sample_dir in sorted(args.eval_root.glob("sample_*"))
        if (sample_dir / "person.png").exists()
    ]
    summary = {"method": args.method, "sample_count": len(rows), "rows": rows}
    (args.output_root / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output_root": args.output_root.as_posix(), "sample_count": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
