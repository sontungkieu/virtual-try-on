from __future__ import annotations

import argparse
import json
import shutil
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps


COMFY_URL = "http://127.0.0.1:8188"
PROJECT_ROOT = Path("/workspace/Project_Phase2/virtual_tryon")
COMFY_INPUT = Path("/workspace/ComfyUI/input")
COMFY_OUTPUT = Path("/workspace/ComfyUI/output")


BASE_PROMPT = (
    "High quality virtual try-on photo. The person wears all target garments and accessories "
    "from the reference image. Preserve the person's identity, face, pose, body shape, hands, "
    "skin, lighting, and background. Keep the full body visible from head to feet. "
    "Only edit the masked garment or accessory regions. Realistic fabric fit, natural occlusion, clean edges."
)
NEGATIVE_PROMPT = (
    "bad anatomy, deformed body, extra limbs, missing limbs, duplicated person, distorted face, "
    "warped hands, wrong garment, missing garment, missing shoes, missing hat, blurry, low quality"
)


GRID_MAP = {
    "sample_001": {"person": (0, 0), "garment_top": (0, 1), "category": "upper_body"},
    "sample_002": {"person": (0, 2), "garment_bottom": (0, 3), "category": "lower_body"},
    "sample_003": {"person": (0, 4), "garment_top": (0, 5), "category": "upper_body"},
    "sample_004": {"person": (1, 0), "garment_bottom": (1, 1), "category": "lower_body"},
    "sample_005": {"person": (1, 2), "garment_dress": (1, 3), "category": "dress"},
    "sample_006": {"person": (1, 4), "garment_top": (1, 5), "category": "upper_body"},
    "sample_007": {"person": (2, 0), "garment_top": (2, 1), "category": "upper_body"},
    "sample_008": {"person": (2, 2), "garment_top": (2, 3), "category": "upper_body"},
    "sample_009": {"person": (2, 4), "garment_bottom": (2, 5), "category": "lower_body"},
}


@dataclass
class SampleSpec:
    sample_id: str
    sample_dir: Path
    category: str
    item_paths: list[Path]
    person_path: Path
    mask_path: Path
    reference_path: Path


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


def crop_grid_cell(grid: Image.Image, row: int, col: int, cols: int = 6, rows: int = 3) -> Image.Image:
    cell_w = grid.width // cols
    cell_h = grid.height // rows
    label_h = max(28, int(cell_h * 0.11))
    left = col * cell_w
    top = row * cell_h + label_h
    right = (col + 1) * cell_w
    bottom = (row + 1) * cell_h
    return grid.crop((left, top, right, bottom)).convert("RGB")


def copy_image(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(src).convert("RGB")
    image.save(dst)
    return dst


def make_reference_canvas(item_paths: list[Path], output_path: Path) -> Path:
    images = [Image.open(path).convert("RGB") for path in item_paths if path.exists()]
    if not images:
        raise FileNotFoundError("No item paths were available for reference canvas")

    max_w = 768
    thumb_h = 512 if len(images) <= 2 else 360
    thumbs: list[Image.Image] = []
    for image in images:
        image.thumbnail((max_w, thumb_h), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (max_w, thumb_h), "white")
        canvas.paste(image, ((max_w - image.width) // 2, (thumb_h - image.height) // 2))
        thumbs.append(canvas)

    out = Image.new("RGB", (max_w, thumb_h * len(thumbs)), "white")
    y = 0
    for thumb in thumbs:
        out.paste(thumb, (0, y))
        y += thumb_h
    out.save(output_path)
    return output_path


def make_mask(person_path: Path, category: str, item_names: list[str], output_path: Path) -> Path:
    person = Image.open(person_path).convert("RGB")
    w, h = person.size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    def rect(x0: float, y0: float, x1: float, y1: float) -> None:
        draw.rectangle((int(w * x0), int(h * y0), int(w * x1), int(h * y1)), fill=255)

    has_hat = any("hat" in name for name in item_names)
    has_shoes = any("shoes" in name for name in item_names)
    has_watch = any("watch" in name for name in item_names)

    if category == "upper_body":
        rect(0.23, 0.17, 0.77, 0.58)
        rect(0.16, 0.23, 0.84, 0.44)
    elif category == "lower_body":
        rect(0.25, 0.43, 0.75, 0.82)
        rect(0.20, 0.62, 0.80, 0.96)
    elif category in {"dress", "full_outfit"}:
        rect(0.27, 0.18, 0.73, 0.54)
        rect(0.22, 0.45, 0.78, 0.78)
        rect(0.28, 0.72, 0.72, 0.92)
    elif category == "accessory":
        rect(0.10, 0.02, 0.90, 0.42)
        if has_watch:
            rect(0.02, 0.30, 0.98, 0.72)
    else:
        rect(0.12, 0.13, 0.88, 0.98)

    if has_hat:
        rect(0.26, 0.00, 0.74, 0.18)
    if has_shoes:
        rect(0.24, 0.84, 0.76, 1.00)
    if has_watch and category != "accessory":
        rect(0.02, 0.33, 0.98, 0.70)

    mask = ImageOps.expand(mask, border=max(2, min(w, h) // 80), fill=0).resize(
        (w, h), Image.Resampling.BILINEAR
    )
    mask.save(output_path)
    return output_path


def prepare_eval_set() -> list[SampleSpec]:
    eval_root = PROJECT_ROOT / "data" / "temp" / "catvton_flux_eval_set"
    eval_root.mkdir(parents=True, exist_ok=True)
    specs: list[SampleSpec] = []

    raw001 = PROJECT_ROOT / "data" / "eval_set" / "sample_001"
    if raw001.exists():
        sample_dir = eval_root / "sample_001"
        sample_dir.mkdir(parents=True, exist_ok=True)
        person_path = copy_image(raw001 / "person.jpg", sample_dir / "person.png")
        item = copy_image(raw001 / "garment_top.jpg", sample_dir / "garment_top.png")
        category = "upper_body"
        reference = make_reference_canvas([item], sample_dir / "reference_canvas.png")
        mask = make_mask(person_path, category, [item.name], sample_dir / "mask.png")
        specs.append(SampleSpec("sample_001", sample_dir, category, [item], person_path, mask, reference))

    source_grid_path = PROJECT_ROOT / "data" / "temp" / "vton_phase2_source_pairs_grid.png"
    if source_grid_path.exists():
        grid = Image.open(source_grid_path).convert("RGB")
        for sample_id, mapping in GRID_MAP.items():
            if sample_id == "sample_001" and raw001.exists():
                continue
            sample_dir = eval_root / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)
            category = str(mapping["category"])
            person_path = sample_dir / "person.png"
            crop_grid_cell(grid, *mapping["person"]).save(person_path)
            item_paths: list[Path] = []
            for name, cell in mapping.items():
                if name in {"person", "category"}:
                    continue
                item_path = sample_dir / f"{name}.png"
                crop_grid_cell(grid, *cell).save(item_path)
                item_paths.append(item_path)
            reference = make_reference_canvas(item_paths, sample_dir / "reference_canvas.png")
            mask = make_mask(person_path, category, [path.name for path in item_paths], sample_dir / "mask.png")
            metadata = {
                "sample_id": sample_id,
                "category": category,
                "source": "vton_phase2_source_pairs_grid.png",
                "items": [path.name for path in item_paths],
            }
            (sample_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            specs.append(SampleSpec(sample_id, sample_dir, category, item_paths, person_path, mask, reference))

    extra_root = PROJECT_ROOT / "data" / "temp" / "vton_phase2_extra_eval_set"
    for sample_dir in sorted(extra_root.glob("sample_*")):
        sample_id = sample_dir.name
        out_dir = eval_root / sample_id
        out_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = sample_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        category = str(metadata.get("category") or "full_outfit")
        person_src = sample_dir / "person.png"
        if not person_src.exists():
            continue
        person_path = copy_image(person_src, out_dir / "person.png")
        item_paths: list[Path] = []
        for item_src in sorted(sample_dir.glob("*.png")):
            if item_src.name in {"person.png", "mask.png", "reference_canvas.png"}:
                continue
            item_paths.append(copy_image(item_src, out_dir / item_src.name))
        if not item_paths:
            continue
        reference = make_reference_canvas(item_paths, out_dir / "reference_canvas.png")
        mask = make_mask(person_path, category, [path.name for path in item_paths], out_dir / "mask.png")
        (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        specs.append(SampleSpec(sample_id, out_dir, category, item_paths, person_path, mask, reference))

    specs_by_id = {spec.sample_id: spec for spec in specs}
    return [specs_by_id[key] for key in sorted(specs_by_id)]


def copy_to_comfy_input(path: Path, name: str) -> str:
    COMFY_INPUT.mkdir(parents=True, exist_ok=True)
    target = COMFY_INPUT / name
    shutil.copy2(path, target)
    return name


def item_prompt(item_paths: list[Path], category: str) -> str:
    labels: list[str] = []
    for path in item_paths:
        name = path.stem.lower()
        if "garment_top" in name:
            labels.append("top")
        elif "garment_bottom" in name:
            labels.append("bottom")
        elif "garment_dress" in name:
            labels.append("dress")
        elif "hat" in name:
            labels.append("hat")
        elif "shoes" in name:
            labels.append("shoes")
        elif "watch" in name:
            labels.append("watch")
    if not labels:
        labels.append(category.replace("_", " "))
    if len(labels) == 1:
        item_text = labels[0]
    else:
        item_text = ", ".join(labels[:-1]) + ", and " + labels[-1]
    return f"{BASE_PROMPT} Target items: {item_text}."


def build_prompt(
    spec: SampleSpec,
    steps: int,
    seed: int,
    prefix: str,
    style_strength: float,
    flux_guidance: float,
    denoise: float,
) -> dict[str, Any]:
    person_name = copy_to_comfy_input(spec.person_path, f"redux_{spec.sample_id}_person.png")
    ref_name = copy_to_comfy_input(spec.reference_path, f"redux_{spec.sample_id}_reference.png")
    mask_name = copy_to_comfy_input(spec.mask_path, f"redux_{spec.sample_id}_mask.png")
    positive_prompt = item_prompt(spec.item_paths, spec.category)

    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": person_name}},
        "2": {"class_type": "LoadImage", "inputs": {"image": mask_name}},
        "3": {"class_type": "ImageToMask", "inputs": {"image": ["2", 0], "channel": "red"}},
        "4": {"class_type": "LoadImage", "inputs": {"image": ref_name}},
        "5": {
            "class_type": "DualCLIPLoader",
            "inputs": {
                "clip_name1": "clip_l.safetensors",
                "clip_name2": "t5xxl_fp8_e4m3fn.safetensors",
                "type": "flux",
            },
        },
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["5", 0], "text": positive_prompt}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["5", 0], "text": NEGATIVE_PROMPT}},
        "8": {"class_type": "FluxGuidance", "inputs": {"conditioning": ["6", 0], "guidance": flux_guidance}},
        "9": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": "sigclip_vision_patch14_384.safetensors"}},
        "10": {"class_type": "CLIPVisionEncode", "inputs": {"clip_vision": ["9", 0], "image": ["4", 0], "crop": "none"}},
        "11": {"class_type": "StyleModelLoader", "inputs": {"style_model_name": "flux1-redux-dev.safetensors"}},
        "12": {
            "class_type": "StyleModelApply",
            "inputs": {
                "conditioning": ["8", 0],
                "style_model": ["11", 0],
                "clip_vision_output": ["10", 0],
                "strength": style_strength,
                "strength_type": "multiply",
            },
        },
        "13": {"class_type": "VAELoader", "inputs": {"vae_name": "FLUX1/ae.safetensors"}},
        "14": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": "FLUX1/fluxFillFP8_v10.safetensors", "weight_dtype": "default"},
        },
        "15": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": ["14", 0], "lora_name": "flux/catvton-flux-lora.safetensors", "strength_model": 1.0},
        },
        "16": {
            "class_type": "InpaintModelConditioning",
            "inputs": {
                "positive": ["12", 0],
                "negative": ["7", 0],
                "vae": ["13", 0],
                "pixels": ["1", 0],
                "mask": ["3", 0],
                "noise_mask": True,
            },
        },
        "17": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["15", 0],
                "seed": seed,
                "steps": steps,
                "cfg": 1,
                "sampler_name": "euler",
                "scheduler": "normal",
                "positive": ["16", 0],
                "negative": ["16", 1],
                "latent_image": ["16", 2],
                "denoise": denoise,
            },
        },
        "18": {"class_type": "VAEDecode", "inputs": {"samples": ["17", 0], "vae": ["13", 0]}},
        "19": {"class_type": "SaveImage", "inputs": {"images": ["18", 0], "filename_prefix": prefix}},
    }


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
        if status.get("completed"):
            return item
        if status.get("status_str") == "error":
            return item
    raise TimeoutError(f"ComfyUI prompt timed out: {prompt_id}")


def extract_saved_image(history_item: dict[str, Any]) -> Path:
    outputs = history_item.get("outputs", {})
    for output in outputs.values():
        images = output.get("images")
        if not images:
            continue
        image = images[0]
        subfolder = image.get("subfolder") or ""
        filename = image["filename"]
        return COMFY_OUTPUT / subfolder / filename
    raise RuntimeError(f"No saved image in history: {json.dumps(history_item)[:1000]}")


def build_report(rows: list[dict[str, Any]], output_root: Path) -> None:
    cell_w = 280
    cell_h = 360
    header_h = 46
    cols = 4
    grid = Image.new("RGB", (cols * cell_w, header_h + len(rows) * cell_h), "white")
    draw = ImageDraw.Draw(grid)
    headers = ["Input person", "Reference via SigCLIP/Redux", "Mask", "Output"]
    for i, header in enumerate(headers):
        draw.rectangle((i * cell_w, 0, (i + 1) * cell_w, header_h), fill=(235, 238, 242), outline=(210, 215, 220))
        draw.text((i * cell_w + 10, 15), header, fill=(30, 35, 45))
    for r, row in enumerate(rows):
        sample_dir = Path(row["output_dir"])
        paths = [
            sample_dir / "input_person.png",
            sample_dir / "reference_canvas.png",
            sample_dir / "mask.png",
            sample_dir / "result.png",
        ]
        y0 = header_h + r * cell_h
        draw.text((8, y0 + 8), f"{row['sample_id']} ({row['category']})", fill=(0, 0, 0))
        for c, path in enumerate(paths):
            x0 = c * cell_w
            draw.rectangle((x0, y0, x0 + cell_w, y0 + cell_h), outline=(220, 220, 220))
            if not path.exists():
                draw.text((x0 + 12, y0 + 44), "missing", fill=(150, 0, 0))
                continue
            image = Image.open(path).convert("RGB")
            image.thumbnail((cell_w - 24, cell_h - 62), Image.Resampling.LANCZOS)
            grid.paste(image, (x0 + (cell_w - image.width) // 2, y0 + 44 + (cell_h - 62 - image.height) // 2))
    grid.save(output_root / "catvton_flux_redux_all_cases_grid.png")

    html_rows = []
    for row in rows:
        sample_id = row["sample_id"]
        html_rows.append(
            "<tr>"
            f"<td>{sample_id}</td>"
            f"<td>{row['category']}</td>"
            f"<td>{row['status']}</td>"
            f"<td>{row['runtime_seconds']}</td>"
            f"<td><img src='{sample_id}/input_person.png'></td>"
            f"<td><img src='{sample_id}/reference_canvas.png'></td>"
            f"<td><img src='{sample_id}/mask.png'></td>"
            f"<td><img src='{sample_id}/result.png'></td>"
            "</tr>"
        )
    html = """<!doctype html>
<html><head><meta charset="utf-8"><title>CatVTON Flux Redux Report</title>
<style>
body{font-family:Arial,sans-serif;margin:24px;color:#20242a} table{border-collapse:collapse;width:100%}
th,td{border:1px solid #d8dde5;padding:8px;vertical-align:top} th{background:#edf1f6}
img{max-width:230px;max-height:310px;object-fit:contain;background:#f8f8f8}
code{background:#f2f4f8;padding:2px 5px;border-radius:4px}
</style></head><body>
<h1>CatVTON Flux Redux - Exact Video Pipeline</h1>
<p><code>person image + mask -> InpaintModelConditioning</code>, <code>reference garment image -> SigCLIP/CLIPVision -> Flux Redux/StyleModelApply</code>, then <code>FLUX Fill FP8 + CatVTON LoRA -> KSampler -> VAE Decode</code>.</p>
<table><thead><tr><th>Sample</th><th>Category</th><th>Status</th><th>Runtime(s)</th><th>Input person</th><th>Redux reference</th><th>Mask</th><th>Output</th></tr></thead><tbody>
"""
    html += "\n".join(html_rows)
    html += "</tbody></table></body></html>\n"
    (output_root / "catvton_flux_redux_report.html").write_text(html, encoding="utf-8")


def run_suite(args: argparse.Namespace) -> int:
    specs = prepare_eval_set()
    selected = set(args.samples) if args.samples else None
    if selected:
        specs = [spec for spec in specs if spec.sample_id in selected]

    output_root = PROJECT_ROOT / "data" / "outputs" / args.output_name
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for index, spec in enumerate(specs):
        sample_out = output_root / spec.sample_id
        sample_out.mkdir(parents=True, exist_ok=True)
        result_path = sample_out / "result.png"
        if args.skip_existing and result_path.exists():
            status = "completed_existing"
            runtime = 0.0
        else:
            prefix = f"{args.output_name}/{spec.sample_id}/result"
            seed = args.seed_base + index * 100
            prompt = build_prompt(
                spec,
                steps=args.steps,
                seed=seed,
                prefix=prefix,
                style_strength=args.style_strength,
                flux_guidance=args.flux_guidance,
                denoise=args.denoise,
            )
            (sample_out / "prompt_api.json").write_text(json.dumps(prompt, indent=2), encoding="utf-8")
            started = time.perf_counter()
            try:
                history = queue_and_wait(prompt, timeout_seconds=args.timeout_seconds)
                (sample_out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
                status_payload = history.get("status", {})
                if not status_payload.get("completed"):
                    raise RuntimeError(json.dumps(status_payload, ensure_ascii=False))
                saved = extract_saved_image(history)
                shutil.copy2(saved, result_path)
                status = "completed"
            except Exception as exc:  # noqa: BLE001 - preserve batch progress
                status = "failed"
                (sample_out / "error.txt").write_text(f"{type(exc).__name__}: {exc}", encoding="utf-8")
            runtime = time.perf_counter() - started

        shutil.copy2(spec.person_path, sample_out / "input_person.png")
        shutil.copy2(spec.reference_path, sample_out / "reference_canvas.png")
        shutil.copy2(spec.mask_path, sample_out / "mask.png")
        row = {
            "sample_id": spec.sample_id,
            "category": spec.category,
            "status": status,
            "runtime_seconds": round(runtime, 3),
            "output_dir": sample_out.as_posix(),
            "item_paths": [path.as_posix() for path in spec.item_paths],
            "style_strength": args.style_strength,
            "flux_guidance": args.flux_guidance,
            "denoise": args.denoise,
            "steps": args.steps,
        }
        rows.append(row)
        (sample_out / "status.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        print(json.dumps(row), flush=True)

    (output_root / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    build_report(rows, output_root)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-name", default="catvton_flux_redux_20260625")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed-base", type=int, default=20260625)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--style-strength", type=float, default=0.75)
    parser.add_argument("--flux-guidance", type=float, default=3.5)
    parser.add_argument("--denoise", type=float, default=0.55)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--samples", nargs="*")
    args = parser.parse_args()
    return run_suite(args)


if __name__ == "__main__":
    raise SystemExit(main())
