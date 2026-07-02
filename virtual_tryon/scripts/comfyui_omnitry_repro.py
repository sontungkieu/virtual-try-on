from __future__ import annotations

import argparse
import json
import shutil
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


COMFY = "http://127.0.0.1:8188"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OMNITRY_ROOT = PROJECT_ROOT / "data" / "inputs" / "omnitry" / "output_omnitry"
COMFY_ROOT = Path("/workspace/ComfyUI")
COMFY_INPUT = COMFY_ROOT / "input"
WORKFLOW_DIR = PROJECT_ROOT / "comfyui_workflows" / "omnitry_innerwear_repro"
DEFAULT_OUTPUT_WIDTH = 512
DEFAULT_OUTPUT_HEIGHT = 768
DEFAULT_STEPS = 8
DEFAULT_SEED = 2026070201


@dataclass(frozen=True)
class ReproCase:
    case_id: str
    category: str
    target_region: str
    person_path: Path
    garment_path: Path
    prompt: str
    seed: int


def _short_stem(path: Path) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in path.stem.lower()).strip("_")[:48]


def discover_cases(limit: int | None = None) -> list[ReproCase]:
    models = OMNITRY_ROOT / "input_models"
    female_person = models / "female_model.jpg"
    male_people = [models / "male_model_1.jpg", models / "male_model_2.jpg"]
    female_garments = sorted((OMNITRY_ROOT / "input_undergarment" / "female").glob("*"))
    male_garments = sorted((OMNITRY_ROOT / "input_undergarment" / "male").glob("*"))
    bra_dirs = [
        OMNITRY_ROOT / "input_undergarment" / "bra",
        OMNITRY_ROOT / "input_bra",
    ]

    cases: list[ReproCase] = []
    for index, garment in enumerate(female_garments, start=1):
        cases.append(
            ReproCase(
                case_id=f"female_underwear_{index:02d}_{_short_stem(garment)}",
                category="women_underwear",
                target_region="lower",
                person_path=female_person,
                garment_path=garment,
                prompt=(
                    "Adult non-sexual product virtual try-on. Replace only the lower innerwear region "
                    "with the referenced women's underwear. Remove the original lower garment inside "
                    "the mask completely. Preserve face, hair, pose, skin, upper clothing, legs outside "
                    "the mask, lighting, and background."
                ),
                seed=DEFAULT_SEED + index,
            )
        )

    case_index = len(cases) + 1
    for person in male_people:
        for garment in male_garments:
            cases.append(
                ReproCase(
                    case_id=f"{_short_stem(person)}_men_underwear_{case_index:02d}_{_short_stem(garment)}",
                    category="men_underwear",
                    target_region="lower",
                    person_path=person,
                    garment_path=garment,
                    prompt=(
                        "Adult non-sexual product virtual try-on. Replace only the lower innerwear region "
                        "with the referenced men's underwear. Remove the original shorts, pants, briefs, "
                        "and old fabric inside the mask completely. Preserve face, hair, pose, skin, "
                        "upper clothing, legs outside the mask, lighting, and background."
                    ),
                    seed=DEFAULT_SEED + case_index,
                )
            )
            case_index += 1

    for bra_dir in bra_dirs:
        if not bra_dir.exists():
            continue
        for garment in sorted(bra_dir.glob("*")):
            cases.append(
                ReproCase(
                    case_id=f"female_bra_{case_index:02d}_{_short_stem(garment)}",
                    category="women_bra",
                    target_region="upper",
                    person_path=female_person,
                    garment_path=garment,
                    prompt=(
                        "Adult non-sexual product virtual try-on. Replace only the bra or upper innerwear "
                        "region with the referenced bra. Preserve face, hair, pose, skin, lower clothing, "
                        "abdomen outside the mask, lighting, and background."
                    ),
                    seed=DEFAULT_SEED + case_index,
                )
            )
            case_index += 1

    if limit is not None:
        return cases[:limit]
    return cases


def copy_case_inputs(cases: list[ReproCase]) -> list[dict[str, str]]:
    input_dir = COMFY_INPUT / "vton_omnitry_repro"
    input_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for case in cases:
        if not case.person_path.exists():
            raise FileNotFoundError(case.person_path)
        if not case.garment_path.exists():
            raise FileNotFoundError(case.garment_path)
        person_name = f"{case.case_id}_person{case.person_path.suffix.lower()}"
        garment_name = f"{case.case_id}_garment{case.garment_path.suffix.lower()}"
        shutil.copy2(case.person_path, input_dir / person_name)
        shutil.copy2(case.garment_path, input_dir / garment_name)
        rows.append(
            {
                "case_id": case.case_id,
                "category": case.category,
                "target_region": case.target_region,
                "person_image": f"vton_omnitry_repro/{person_name}",
                "garment_image": f"vton_omnitry_repro/{garment_name}",
                "person_source": case.person_path.as_posix(),
                "garment_source": case.garment_path.as_posix(),
                "seed": str(case.seed),
            }
        )
    return rows


def build_api_prompt(
    *,
    row: dict[str, str],
    prompt: str,
    width: int,
    height: int,
    steps: int,
    engine_mode: str,
) -> dict[str, Any]:
    filename_prefix = f"vton_omnitry_repro/{row['case_id']}/{engine_mode}"
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": row["person_image"]}},
        "2": {"class_type": "LoadImage", "inputs": {"image": row["garment_image"]}},
        "3": {
            "class_type": "VTONPhase2BackendTryOnAPI",
            "inputs": {
                "person_image": ["1", 0],
                "garment_image": ["2", 0],
                "category": row["category"],
                "engine_mode": engine_mode,
                "prompt": prompt,
                "seed": int(row["seed"]),
                "api_base": COMFY.replace(":8188", ":8000"),
                "output_width": width,
                "output_height": height,
                "steps": steps,
                "deterministic": True,
                "timeout_s": 900,
                "poll_interval_s": 1.0,
            },
        },
        "4": {
            "class_type": "SaveImage",
            "inputs": {"images": ["3", 0], "filename_prefix": f"{filename_prefix}/result"},
        },
        "5": {
            "class_type": "SaveImage",
            "inputs": {"images": ["3", 1], "filename_prefix": f"{filename_prefix}/mask_preview"},
        },
    }


def _ui_node(
    node_id: int,
    node_type: str,
    pos: tuple[int, int],
    widgets_values: list[Any],
    inputs: list[dict[str, Any]] | None = None,
    outputs: list[dict[str, Any]] | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    node = {
        "id": node_id,
        "type": node_type,
        "pos": list(pos),
        "size": [360, 120],
        "flags": {},
        "order": node_id,
        "mode": 0,
        "inputs": inputs or [],
        "outputs": outputs or [],
        "properties": {},
        "widgets_values": widgets_values,
    }
    if title:
        node["title"] = title
    return node


def build_ui_workflow(
    *,
    row: dict[str, str],
    prompt: str,
    width: int,
    height: int,
    steps: int,
    engine_mode: str,
) -> dict[str, Any]:
    links = [
        [1, 1, 0, 3, 0, "IMAGE"],
        [2, 2, 0, 3, 1, "IMAGE"],
        [3, 3, 0, 4, 0, "IMAGE"],
        [4, 3, 1, 5, 0, "IMAGE"],
    ]
    return {
        "last_node_id": 5,
        "last_link_id": len(links),
        "nodes": [
            _ui_node(1, "LoadImage", (0, 0), [row["person_image"], "image"], outputs=[{"name": "IMAGE", "type": "IMAGE", "links": [1]}]),
            _ui_node(2, "LoadImage", (0, 220), [row["garment_image"], "image"], outputs=[{"name": "IMAGE", "type": "IMAGE", "links": [2]}]),
            _ui_node(
                3,
                "VTONPhase2BackendTryOnAPI",
                (430, 0),
                [
                    row["category"],
                    engine_mode,
                    prompt,
                    int(row["seed"]),
                    COMFY.replace(":8188", ":8000"),
                    width,
                    height,
                    steps,
                    True,
                    900,
                    1.0,
                ],
                inputs=[
                    {"name": "person_image", "type": "IMAGE", "link": 1},
                    {"name": "garment_image", "type": "IMAGE", "link": 2},
                ],
                outputs=[
                    {"name": "result_image", "type": "IMAGE", "links": [3]},
                    {"name": "mask_preview", "type": "IMAGE", "links": [4]},
                    {"name": "status", "type": "STRING", "links": None},
                ],
                title="Backend try-on API",
            ),
            _ui_node(4, "SaveImage", (900, 0), [f"vton_omnitry_repro/{row['case_id']}/{engine_mode}/result"], inputs=[{"name": "images", "type": "IMAGE", "link": 3}]),
            _ui_node(5, "SaveImage", (900, 180), [f"vton_omnitry_repro/{row['case_id']}/{engine_mode}/mask_preview"], inputs=[{"name": "images", "type": "IMAGE", "link": 4}]),
        ],
        "links": links,
        "groups": [],
        "config": {},
        "extra": {"ds": {"scale": 0.75, "offset": [100, 50]}},
        "version": 0.4,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def post_json(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(f"{COMFY}{path}", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_json(path: str) -> dict[str, Any]:
    with urllib.request.urlopen(f"{COMFY}{path}", timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def queue_prompt(prompt: dict[str, Any], timeout_s: int) -> dict[str, Any]:
    response = post_json("/prompt", {"prompt": prompt})
    prompt_id = response["prompt_id"]
    started = time.time()
    while time.time() - started < timeout_s:
        time.sleep(2)
        history = get_json(f"/history/{prompt_id}")
        if prompt_id in history:
            return {"prompt_id": prompt_id, **history[prompt_id]}
    raise TimeoutError(f"ComfyUI prompt {prompt_id} did not finish in {timeout_s}s")


def build_readme(rows: list[dict[str, str]], engine_mode: str, width: int, height: int, steps: int) -> str:
    lines = [
        "# Omnitry Innerwear Reproduction Workflows",
        "",
        "These ComfyUI workflows reproduce the adult, non-sexual innerwear try-on cases from `data/inputs/omnitry/output_omnitry`.",
        "",
        f"Default engine: `{engine_mode}`",
        f"Default resolution: `{width}x{height}`",
        f"Default steps: `{steps}`",
        "",
        "## Files",
        "",
        "| File | Type | Notes |",
        "|---|---|---|",
        "| `manifest.json` | metadata | Copied ComfyUI input names and source paths. |",
        "| `*_api.json` | API prompt | Queue with ComfyUI `/prompt`. |",
        "| `*_ui.workflow.json` | UI workflow | Load in the ComfyUI editor. |",
        "",
        "## Inputs",
        "",
        "The script copies inputs into `/workspace/ComfyUI/input/vton_omnitry_repro/`.",
        "",
        "## Cases",
        "",
        "| case | category | target | person | garment | seed |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['case_id']}` | `{row['category']}` | `{row['target_region']}` | "
            f"`{row['person_image']}` | `{row['garment_image']}` | `{row['seed']}` |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build reproducible ComfyUI workflows for Omnitry innerwear cases.")
    parser.add_argument("--limit", type=int, default=None, help="Only export the first N discovered cases.")
    parser.add_argument("--case", dest="case_id", default=None, help="Only export one discovered case id.")
    parser.add_argument(
        "--engine-mode",
        default="idm_vton",
        choices=[
            "idm_vton",
            "idm_mask_expanded",
            "idm_vton_flux",
            "idm_mask_expanded_flux",
            "klein_lora",
            "idm_klein_hybrid",
            "flux_redux_catvton",
        ],
    )
    parser.add_argument("--width", type=int, default=DEFAULT_OUTPUT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_OUTPUT_HEIGHT)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--queue-first", action="store_true", help="Queue the first exported API workflow through ComfyUI.")
    parser.add_argument("--timeout-s", type=int, default=900)
    args = parser.parse_args()

    cases = discover_cases(limit=args.limit)
    if args.case_id:
        cases = [case for case in cases if case.case_id == args.case_id]
    if not cases:
        raise RuntimeError("No Omnitry cases were discovered for export.")

    rows = copy_case_inputs(cases)
    prompt_by_id = {case.case_id: case.prompt for case in cases}
    WORKFLOW_DIR.mkdir(parents=True, exist_ok=True)

    exported: list[dict[str, str]] = []
    first_prompt: dict[str, Any] | None = None
    for row in rows:
        prompt = prompt_by_id[row["case_id"]]
        api_prompt = build_api_prompt(
            row=row,
            prompt=prompt,
            width=args.width,
            height=args.height,
            steps=args.steps,
            engine_mode=args.engine_mode,
        )
        ui_workflow = build_ui_workflow(
            row=row,
            prompt=prompt,
            width=args.width,
            height=args.height,
            steps=args.steps,
            engine_mode=args.engine_mode,
        )
        api_path = WORKFLOW_DIR / f"{row['case_id']}_{args.engine_mode}_api.json"
        ui_path = WORKFLOW_DIR / f"{row['case_id']}_{args.engine_mode}_ui.workflow.json"
        write_json(api_path, api_prompt)
        write_json(ui_path, ui_workflow)
        exported.append({"case_id": row["case_id"], "api": api_path.as_posix(), "ui": ui_path.as_posix()})
        if first_prompt is None:
            first_prompt = api_prompt

    manifest = {
        "generated_by": "scripts/comfyui_omnitry_repro.py",
        "workflow_dir": WORKFLOW_DIR.as_posix(),
        "comfy_input_dir": (COMFY_INPUT / "vton_omnitry_repro").as_posix(),
        "engine_mode": args.engine_mode,
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "case_count": len(rows),
        "rows": rows,
        "workflows": exported,
    }
    write_json(WORKFLOW_DIR / "manifest.json", manifest)
    (WORKFLOW_DIR / "README.md").write_text(
        build_readme(rows, args.engine_mode, args.width, args.height, args.steps),
        encoding="utf-8",
    )

    queue_result = None
    if args.queue_first:
        if first_prompt is None:
            raise RuntimeError("No API prompt available to queue.")
        queue_result = queue_prompt(first_prompt, args.timeout_s)

    print(
        json.dumps(
            {
                "workflow_dir": WORKFLOW_DIR.as_posix(),
                "case_count": len(rows),
                "engine_mode": args.engine_mode,
                "queued": queue_result is not None,
                "queue_status": queue_result.get("status") if queue_result else None,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
