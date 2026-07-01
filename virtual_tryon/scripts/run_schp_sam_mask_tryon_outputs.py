from __future__ import annotations

import json
import shutil
import sys
import time
import argparse
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from virtual_tryon.final_demo import garment_file_for_region, region_prompt
from virtual_tryon.scripts.run_upstream_flux_redux_workflow import (
    COMFY_INPUT,
    COMFY_OUTPUT,
    EVAL_ROOT,
    NEGATIVE_PROMPT,
    build_upstream_equivalent_prompt,
    copy_to_comfy_input,
    queue_and_wait,
    saved_images,
)


PROJECT_ROOT = Path("/workspace/Project_Phase2")
MASK_META_PATH = PROJECT_ROOT / "virtual_tryon/data/outputs/schp_sam_masks_20260626/metadata.json"
RUN_ROOT = PROJECT_ROOT / "virtual_tryon/data/outputs/schp_sam_mask_tryon_outputs_20260626"

def garment_path_for(sample_id: str, region: str) -> Path | None:
    sample_dir = EVAL_ROOT / sample_id
    try:
        name = garment_file_for_region(region)
    except KeyError:
        return None
    path = sample_dir / name
    return path if path.exists() else None


def copy_debug_inputs(pass_dir: Path, person_path: Path, garment_path: Path, mask_path: Path) -> None:
    pass_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(person_path, pass_dir / "input_person.png")
    shutil.copy2(garment_path, pass_dir / "input_garment.png")
    shutil.copy2(mask_path, pass_dir / "mask_processed.png")


def run_pass(
    *,
    sample_id: str,
    pass_index: int,
    target_region: str,
    person_path: Path,
    garment_path: Path,
    mask_path: Path,
    seed: int,
) -> Path:
    if not person_path.exists():
        raise FileNotFoundError(f"Missing person image: {person_path}")
    if not garment_path.exists():
        raise FileNotFoundError(f"Missing garment image: {garment_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Missing mask image: {mask_path}")

    pass_dir = RUN_ROOT / sample_id / f"pass_{pass_index:02d}_{target_region}"
    copy_debug_inputs(pass_dir, person_path, garment_path, mask_path)

    person_name = copy_to_comfy_input(person_path, f"schp_sam_{sample_id}_p{pass_index:02d}_person.png")
    garment_name = copy_to_comfy_input(garment_path, f"schp_sam_{sample_id}_p{pass_index:02d}_garment.png")
    mask_name = copy_to_comfy_input(mask_path, f"schp_sam_{sample_id}_p{pass_index:02d}_mask.png")
    filename_prefix = f"schp_sam_mask_tryon/{sample_id}/pass_{pass_index:02d}_{target_region}"

    graph = build_upstream_equivalent_prompt(
        person_name=person_name,
        reference_name=garment_name,
        mask_name=mask_name,
        sample_id=f"{sample_id}_{target_region}",
        prompt_mode="tryon_prompt",
        seed=seed,
        filename_prefix=filename_prefix,
    )
    graph["7"]["inputs"]["text"] = region_prompt(sample_id, target_region)
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
        "prompt": region_prompt(sample_id, target_region),
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


def write_html(rows: list[dict[str, Any]], skipped: list[dict[str, Any]]) -> Path:
    def rel(path: str | Path) -> str:
        return Path(path).relative_to(RUN_ROOT).as_posix()

    sections: list[str] = []
    for row in rows:
        sections.append(
            "<section>"
            f"<h2>{row['label']}</h2>"
            "<div class='grid'>"
            f"<figure><img src='{rel(row['person'])}'><figcaption>person</figcaption></figure>"
            f"<figure><img src='{rel(row['garment'])}'><figcaption>garment</figcaption></figure>"
            f"<figure><img src='{rel(row['mask'])}'><figcaption>mask</figcaption></figure>"
            f"<figure><img src='{rel(row['output'])}'><figcaption>output</figcaption></figure>"
            "</div></section>"
        )
    skipped_items = "".join(
        f"<li>{item['sample_id']} / {item['region']}: {item['reason']}</li>" for item in skipped
    )
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>SCHP/SAM Mask Try-On Outputs</title>"
        "<style>body{font-family:Arial,sans-serif;margin:24px;background:#f7f8fa;color:#17202a}"
        "section{background:#fff;border:1px solid #d8dee8;border-radius:8px;padding:16px;margin:0 0 18px}"
        ".grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px} img{max-width:100%;border:1px solid #d8dee8}"
        "figcaption{font-size:13px;color:#566573} li{margin:4px 0}</style></head><body>"
        "<h1>SCHP/SAM Mask Try-On Outputs</h1>"
        f"<h2>Skipped</h2><ul>{skipped_items or '<li>None</li>'}</ul>"
        f"{''.join(sections)}</body></html>"
    )
    path = RUN_ROOT / "schp_sam_mask_tryon_report.html"
    path.write_text(html, encoding="utf-8")
    return path


def main() -> int:
    global MASK_META_PATH, RUN_ROOT, EVAL_ROOT

    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-meta", type=Path, default=MASK_META_PATH)
    parser.add_argument("--eval-root", type=Path, default=EVAL_ROOT)
    parser.add_argument("--output-root", type=Path, default=RUN_ROOT)
    args = parser.parse_args()

    MASK_META_PATH = args.mask_meta
    EVAL_ROOT = args.eval_root
    RUN_ROOT = args.output_root

    if not MASK_META_PATH.exists():
        raise FileNotFoundError(f"Missing mask metadata: {MASK_META_PATH}")

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    COMFY_INPUT.mkdir(parents=True, exist_ok=True)
    COMFY_OUTPUT.mkdir(parents=True, exist_ok=True)

    mask_meta = json.loads(MASK_META_PATH.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"passes": [], "skipped": skipped}

    for sample_index, sample_id in enumerate(sorted(k for k in mask_meta if k.startswith("sample_"))):
        sample_meta = mask_meta[sample_id]
        masks = sample_meta.get("masks", {})
        current_person = Path(sample_meta["person_path"])
        pass_index = 0
        for region, info in masks.items():
            if not info.get("source") or not info.get("mask_path"):
                skipped.append(
                    {
                        "sample_id": sample_id,
                        "region": region,
                        "reason": info.get("error") or "no usable mask",
                    }
                )
                continue
            garment_path = garment_path_for(sample_id, region)
            if garment_path is None:
                skipped.append({"sample_id": sample_id, "region": region, "reason": "missing garment image"})
                continue
            mask_path = Path(info["mask_path"])
            pass_index += 1
            seed = 2026062600 + sample_index * 100 + pass_index
            output = run_pass(
                sample_id=sample_id,
                pass_index=pass_index,
                target_region=region,
                person_path=current_person,
                garment_path=garment_path,
                mask_path=mask_path,
                seed=seed,
            )
            pass_dir = RUN_ROOT / sample_id / f"pass_{pass_index:02d}_{region}"
            row = {
                "label": f"{sample_id} / {region}",
                "person": (pass_dir / "input_person.png").as_posix(),
                "garment": (pass_dir / "input_garment.png").as_posix(),
                "mask": (pass_dir / "mask_processed.png").as_posix(),
                "output": output.as_posix(),
            }
            rows.append(row)
            summary["passes"].append(row)
            current_person = output
        if pass_index > 0:
            final_path = RUN_ROOT / sample_id / "final_output.png"
            shutil.copy2(current_person, final_path)
            summary.setdefault("final_outputs", {})[sample_id] = final_path.as_posix()

    build_sheet(rows, RUN_ROOT / "schp_sam_mask_tryon_output_sheet.png")
    report_path = write_html(rows, skipped)
    summary["sheet"] = (RUN_ROOT / "schp_sam_mask_tryon_output_sheet.png").as_posix()
    summary["report"] = report_path.as_posix()
    (RUN_ROOT / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
