from __future__ import annotations

import json
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_ROOT = PROJECT_ROOT / "virtual_tryon/data/temp/full15_archived_grid_eval_set"
COMFY_INPUT = Path("/workspace/ComfyUI/input")
COMFY_SUBDIR = COMFY_INPUT / "vton_final15"

ITEMS_ORDER = [
    "person.png",
    "garment_top.png",
    "garment_bottom.png",
    "garment_dress.png",
    "accessory_shoes.png",
    "accessory_hat.png",
    "accessory_watch.png",
]


def copy_inputs() -> list[dict[str, str]]:
    COMFY_SUBDIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for index in range(1, 16):
        sample_id = f"sample_{index:03d}"
        sample_dir = EVAL_ROOT / sample_id
        for name in ITEMS_ORDER:
            source = sample_dir / name
            if not source.exists():
                continue
            if name == "person.png":
                output_name = f"vton15_{sample_id}_person.png"
                role = "person"
            else:
                role = name.removesuffix(".png")
                output_name = f"vton15_{sample_id}_{role}.png"
            target = COMFY_SUBDIR / output_name
            shutil.copy2(source, target)
            rows.append(
                {
                    "sample_id": sample_id,
                    "role": role,
                    "comfy_load_image_value": f"vton_final15/{output_name}",
                    "source": source.as_posix(),
                    "target": target.as_posix(),
                }
            )
    return rows


def copy_default_placeholders() -> None:
    default_person = EVAL_ROOT / "sample_001/person.png"
    default_garment = EVAL_ROOT / "sample_001/garment_top.png"
    if default_person.exists():
        shutil.copy2(default_person, COMFY_INPUT / "sample_person.png")
    if default_garment.exists():
        shutil.copy2(default_garment, COMFY_INPUT / "sample_garment.png")


def write_manifest(rows: list[dict[str, str]]) -> tuple[Path, Path]:
    manifest_json = PROJECT_ROOT / "virtual_tryon/data/temp/comfyui_final15_inputs_manifest.json"
    manifest_md = PROJECT_ROOT / "virtual_tryon/data/temp/comfyui_final15_inputs_manifest.md"
    manifest_json.parent.mkdir(parents=True, exist_ok=True)
    manifest_json.write_text(
        json.dumps({"input_dir": COMFY_SUBDIR.as_posix(), "count": len(rows), "rows": rows}, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# ComfyUI Final 15 Inputs",
        "",
        f"ComfyUI input folder: `{COMFY_SUBDIR.as_posix()}`",
        "",
        "| sample | role | Load Image value |",
        "|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['sample_id']} | {row['role']} | `{row['comfy_load_image_value']}` |"
        )
    manifest_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest_json, manifest_md


def main() -> int:
    rows = copy_inputs()
    copy_default_placeholders()
    manifest_json, manifest_md = write_manifest(rows)
    print(
        json.dumps(
            {
                "copied": len(rows),
                "input_dir": COMFY_SUBDIR.as_posix(),
                "manifest_json": manifest_json.as_posix(),
                "manifest_md": manifest_md.as_posix(),
                "default_person": "sample_person.png",
                "default_garment": "sample_garment.png",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
