from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from virtual_tryon.scripts.run_upstream_flux_redux_workflow import (
    COMFY_INPUT,
    COMFY_OUTPUT,
    EVAL_ROOT,
    NEGATIVE_PROMPT,
    OUTPUT_ROOT,
    build_upstream_equivalent_prompt,
    copy_to_comfy_input,
    queue_and_wait,
    saved_images,
)


PROJECT_ROOT = Path("/workspace/Project_Phase2")
MASK_ROOT = PROJECT_ROOT / "virtual_tryon/data/temp/precise_target_masks"
RUN_ROOT = PROJECT_ROOT / "virtual_tryon/data/outputs/precise_mask_tryon_outputs_20260625"


def run_pass(
    *,
    sample_id: str,
    pass_index: int,
    target_region: str,
    person_path: Path,
    garment_path: Path,
    mask_path: Path,
    seed: int,
    prompt: str,
) -> Path:
    if not person_path.exists():
        raise FileNotFoundError(f"Missing person image: {person_path}")
    if not garment_path.exists():
        raise FileNotFoundError(f"Missing garment image: {garment_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Missing mask image: {mask_path}")

    pass_dir = RUN_ROOT / sample_id / f"pass_{pass_index:02d}_{target_region}"
    pass_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(person_path, pass_dir / "input_person.png")
    shutil.copy2(garment_path, pass_dir / "input_garment.png")
    shutil.copy2(mask_path, pass_dir / "mask_processed.png")

    person_name = copy_to_comfy_input(person_path, f"precise_{sample_id}_p{pass_index:02d}_person.png")
    garment_name = copy_to_comfy_input(garment_path, f"precise_{sample_id}_p{pass_index:02d}_garment.png")
    mask_name = copy_to_comfy_input(mask_path, f"precise_{sample_id}_p{pass_index:02d}_mask.png")
    filename_prefix = f"precise_mask_tryon/{sample_id}/pass_{pass_index:02d}_{target_region}"
    graph = build_upstream_equivalent_prompt(
        person_name=person_name,
        reference_name=garment_name,
        mask_name=mask_name,
        sample_id=f"{sample_id}_{target_region}",
        prompt_mode="tryon_prompt",
        seed=seed,
        filename_prefix=filename_prefix,
    )
    # The upstream helper generates a generic prompt. Patch it to be item-specific.
    graph["7"]["inputs"]["text"] = prompt
    graph["18"]["inputs"]["steps"] = 24
    graph["18"]["inputs"]["cfg"] = 7
    graph["18"]["inputs"]["denoise"] = 1.0
    (pass_dir / "workflow_used_api.json").write_text(json.dumps(graph, indent=2), encoding="utf-8")

    started = time.perf_counter()
    history = queue_and_wait(graph, timeout_seconds=2400)
    (pass_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    status = history.get("status", {})
    if not status.get("completed"):
        raise RuntimeError(f"ComfyUI failed at {sample_id} pass {pass_index}: {json.dumps(status)}")

    images = saved_images(history)
    if not images:
        raise RuntimeError(f"No ComfyUI image for {sample_id} pass {pass_index}")
    output = pass_dir / "output_base.png"
    shutil.copy2(images[-1], output)
    metadata = {
        "sample_id": sample_id,
        "pass_index": pass_index,
        "target_region": target_region,
        "seed": seed,
        "steps": 24,
        "cfg": 7,
        "flux_guidance": 3.5,
        "denoise": 1.0,
        "sampler": "euler",
        "scheduler": "normal",
        "person": person_path.as_posix(),
        "garment": garment_path.as_posix(),
        "mask": mask_path.as_posix(),
        "output_base": output.as_posix(),
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "prompt": prompt,
        "negative_prompt": NEGATIVE_PROMPT,
    }
    (pass_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata), flush=True)
    return output


def build_sheet(rows: list[dict[str, Any]], output_path: Path) -> None:
    headers = ["person/input", "garment", "mask", "output"]
    cell_w, cell_h, header_h = 260, 330, 42
    sheet = Image.new("RGB", (cell_w * len(headers), header_h + cell_h * len(rows)), "white")
    draw = ImageDraw.Draw(sheet)
    for c, header in enumerate(headers):
        draw.rectangle((c * cell_w, 0, (c + 1) * cell_w, header_h), fill=(235, 238, 242), outline=(214, 219, 226))
        draw.text((c * cell_w + 10, 13), header, fill=(24, 29, 38))

    for r, row in enumerate(rows):
        y0 = header_h + r * cell_h
        draw.text((8, y0 + 8), row["label"], fill=(0, 0, 0))
        for c, key in enumerate(["person", "garment", "mask", "output"]):
            path = Path(row[key])
            x0 = c * cell_w
            draw.rectangle((x0, y0, x0 + cell_w, y0 + cell_h), outline=(220, 224, 230))
            if not path.exists():
                draw.text((x0 + 10, y0 + 52), "missing", fill=(180, 0, 0))
                continue
            image = Image.open(path).convert("RGB")
            image.thumbnail((cell_w - 24, cell_h - 48), Image.Resampling.LANCZOS)
            sheet.paste(image, (x0 + (cell_w - image.width) // 2, y0 + 38 + (cell_h - 48 - image.height) // 2))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def main() -> int:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    COMFY_INPUT.mkdir(parents=True, exist_ok=True)
    COMFY_OUTPUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    sample001_person = EVAL_ROOT / "sample_001/person.png"
    sample001_garment = EVAL_ROOT / "sample_001/garment_top.png"
    sample001_mask = MASK_ROOT / "sample_001/sample_001_upper_mask.png"
    sample001_output = run_pass(
        sample_id="sample_001",
        pass_index=1,
        target_region="upper",
        person_path=sample001_person,
        garment_path=sample001_garment,
        mask_path=sample001_mask,
        seed=2026062511,
        prompt=(
            "Virtual try-on photo. Replace only the masked upper garment with the reference blue velvet short-sleeve "
            "wrap top. Preserve the person's face, hair, hands, pants, body shape, pose, lighting, and background."
        ),
    )
    rows.append(
        {
            "label": "sample_001 / upper",
            "person": (RUN_ROOT / "sample_001/pass_01_upper/input_person.png").as_posix(),
            "garment": (RUN_ROOT / "sample_001/pass_01_upper/input_garment.png").as_posix(),
            "mask": (RUN_ROOT / "sample_001/pass_01_upper/mask_processed.png").as_posix(),
            "output": sample001_output.as_posix(),
        }
    )

    sample015_person = EVAL_ROOT / "sample_015/person.png"
    pass_specs = [
        (
            "dress",
            EVAL_ROOT / "sample_015/garment_dress.png",
            MASK_ROOT / "sample_015/sample_015_dress_mask.png",
            "Virtual try-on photo. Replace only the masked body garment with the reference yellow sleeveless dress. Preserve face, hair, hands, legs, feet, pose, lighting, and background.",
        ),
        (
            "shoes",
            EVAL_ROOT / "sample_015/accessory_shoes.png",
            MASK_ROOT / "sample_015/sample_015_shoes_mask.png",
            "Virtual try-on photo. Replace only the masked feet region with the reference white high heels. Preserve dress, body, legs, face, pose, lighting, and background.",
        ),
        (
            "hat",
            EVAL_ROOT / "sample_015/accessory_hat.png",
            MASK_ROOT / "sample_015/sample_015_hat_strict_mask.png",
            "Virtual try-on photo. Add the reference pink bucket hat only inside the masked head area. Preserve face, eyes, mouth, body, outfit, pose, lighting, and background.",
        ),
    ]
    current_person = sample015_person
    for index, (region, garment, mask, prompt) in enumerate(pass_specs, start=1):
        out = run_pass(
            sample_id="sample_015",
            pass_index=index,
            target_region=region,
            person_path=current_person,
            garment_path=garment,
            mask_path=mask,
            seed=2026062520 + index,
            prompt=prompt,
        )
        pass_dir = RUN_ROOT / "sample_015" / f"pass_{index:02d}_{region}"
        rows.append(
            {
                "label": f"sample_015 / {region}",
                "person": (pass_dir / "input_person.png").as_posix(),
                "garment": (pass_dir / "input_garment.png").as_posix(),
                "mask": (pass_dir / "mask_processed.png").as_posix(),
                "output": out.as_posix(),
            }
        )
        current_person = out
    shutil.copy2(current_person, RUN_ROOT / "sample_015_final_output.png")

    build_sheet(rows, RUN_ROOT / "precise_mask_tryon_output_sheet.png")
    print(f"sheet={(RUN_ROOT / 'precise_mask_tryon_output_sheet.png').as_posix()}")
    print(f"sample_001_output={sample001_output.as_posix()}")
    print(f"sample_015_final={current_person.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
