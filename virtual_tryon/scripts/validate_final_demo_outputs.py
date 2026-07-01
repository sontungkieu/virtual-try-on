from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from virtual_tryon.final_demo import DEFAULT_FINAL_OUTPUT_ROOT, FINAL_METHODS, final_method_paths, sample_id_for_index

VTON_ROOT = PROJECT_ROOT / "virtual_tryon"
COMFY_URL = "http://127.0.0.1:8188"

SAMPLE_IDS = [sample_id_for_index(index) for index in range(1, 16)]

WORKFLOWS = {
    "method_01_schp_sam_flux_catvton": VTON_ROOT
    / "comfyui_workflows/pipelines_20260626/13_schp_sam_mask_consumer_single_pass_ui.json",
    "method_02_klein9b": VTON_ROOT / "comfyui_workflows/pipelines_20260626/11_klein_28_sample015_ui.json",
    "method_03_klein9b_tryon_lora": VTON_ROOT
    / "comfyui_workflows/klein_detailed_pipelines_20260626/02_flux2_klein9b_lora_strong_detailed.workflow.json",
    "method_04_klein_lora_local_masked_inpaint": VTON_ROOT
    / "comfyui_workflows/klein_detailed_pipelines_20260626/04_flux2_klein9b_lora_masked_local_inpaint.workflow.json",
}

FINAL_ROOT = DEFAULT_FINAL_OUTPUT_ROOT


def get_json(url: str, timeout: int = 20) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def image_ok(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"ok": False, "reason": "missing", "path": path.as_posix()}
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
        return {"ok": width > 0 and height > 0, "path": path.as_posix(), "width": width, "height": height}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": repr(exc), "path": path.as_posix()}


def workflow_node_types(path: Path) -> set[str]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    types: set[str] = set()
    for node in data.get("nodes", []):
        node_type = node.get("type")
        if node_type:
            types.add(str(node_type))
    if not types and isinstance(data, dict):
        for node in data.values():
            if isinstance(node, dict) and node.get("class_type"):
                types.add(str(node["class_type"]))
    return types


def validate_workflows() -> dict[str, Any]:
    result: dict[str, Any] = {}
    try:
        object_info = get_json(f"{COMFY_URL}/object_info")
        available = set(object_info)
        comfy_ok = True
    except Exception as exc:  # noqa: BLE001
        available = set()
        comfy_ok = False
        result["comfyui_error"] = repr(exc)

    for name, path in WORKFLOWS.items():
        entry: dict[str, Any] = {"path": path.as_posix(), "exists": path.exists()}
        if path.exists():
            node_types = workflow_node_types(path)
            entry["node_count"] = len(node_types)
            entry["missing_node_types"] = sorted(node_types - available) if comfy_ok else []
            entry["ok"] = comfy_ok and not entry["missing_node_types"]
        else:
            entry["ok"] = False
            entry["missing_node_types"] = []
        result[name] = entry
    result["comfyui_reachable"] = comfy_ok
    return result


def method_paths(sample_id: str) -> dict[str, dict[str, Any]]:
    number = int(sample_id.split("_")[1])
    paths: dict[str, dict[str, Any]] = {}
    for method in FINAL_METHODS:
        paths[method.key] = {
            "source_type": "image_file",
            "path": final_method_paths(FINAL_ROOT, sample_id)[method.key],
            "title": method.title,
        }
    paths.update(
        {
            "final_grid": {
                "source_type": "image_file",
                "path": FINAL_ROOT / f"test_case_{number:02d}_grid.png",
            },
        }
    )
    return paths


def validate_outputs() -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for sample_id in SAMPLE_IDS:
        row: dict[str, Any] = {}
        for method, info in method_paths(sample_id).items():
            check = image_ok(info["path"])
            check["source_type"] = info["source_type"]
            if "crop" in info:
                check["crop"] = info["crop"]
            row[method] = check
        rows[sample_id] = row
    return rows


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    workflow_failures = [
        name for name, entry in report["workflows"].items() if isinstance(entry, dict) and not entry.get("ok")
    ]
    output_failures = []
    for sample_id, row in report["outputs"].items():
        for method, entry in row.items():
            if not entry.get("ok"):
                output_failures.append({"sample_id": sample_id, "method": method, "path": entry.get("path")})
    return {
        "workflows_ok": not workflow_failures,
        "outputs_ok": not output_failures,
        "workflow_failures": workflow_failures,
        "output_failures": output_failures,
        "sample_count": len(SAMPLE_IDS),
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    summary = report["summary"]
    lines = [
        "# Final Demo Validation",
        "",
        f"- ComfyUI reachable: `{report['workflows'].get('comfyui_reachable')}`",
        f"- Workflows OK: `{summary['workflows_ok']}`",
        f"- Outputs OK: `{summary['outputs_ok']}`",
        f"- Sample count: `{summary['sample_count']}`",
        "",
        "## Workflows",
        "",
        "| method | ok | file | missing node types |",
        "|---|---:|---|---|",
    ]
    for name, entry in report["workflows"].items():
        if name == "comfyui_reachable":
            continue
        missing = ", ".join(entry.get("missing_node_types") or [])
        lines.append(f"| {name} | {entry.get('ok')} | `{entry.get('path')}` | {missing} |")
    lines += ["", "## Outputs", "", "| sample | all outputs ok | notes |", "|---|---:|---|"]
    for sample_id, row in report["outputs"].items():
        ok = all(entry.get("ok") for entry in row.values())
        notes = []
        for method, entry in row.items():
            if not entry.get("ok"):
                notes.append(f"{method}: {entry.get('reason', 'not ok')}")
        lines.append(f"| {sample_id} | {ok} | {'; '.join(notes)} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    global FINAL_ROOT

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(FINAL_ROOT))
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    FINAL_ROOT = output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "workflows": validate_workflows(),
        "outputs": validate_outputs(),
    }
    report["summary"] = summarize(report)
    json_path = output_dir / "demo_validation_report.json"
    md_path = output_dir / "demo_validation_report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(report, md_path)
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    return 0 if report["summary"]["outputs_ok"] and report["summary"]["workflows_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
