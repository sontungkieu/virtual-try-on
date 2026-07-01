from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from virtual_tryon.scripts.run_upstream_flux_redux_workflow import (
    EVAL_ROOT,
    NEGATIVE_PROMPT,
    build_upstream_equivalent_prompt,
    copy_to_comfy_input,
    queue_and_wait,
    saved_images,
)


PROJECT_ROOT = Path("/workspace/Project_Phase2")
RUN_ROOT = PROJECT_ROOT / "virtual_tryon/data/outputs/precise_mask_tryon_outputs_20260625"
MASK_ROOT = PROJECT_ROOT / "virtual_tryon/data/temp/precise_target_masks"


def main() -> int:
    source_person = RUN_ROOT / "sample_015/pass_02_shoes/output_base.png"
    garment = EVAL_ROOT / "sample_015/accessory_hat.png"
    mask = MASK_ROOT / "sample_015/sample_015_hat_strict_mask.png"
    out_dir = RUN_ROOT / "sample_015/pass_03_hat_fix"
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_person, out_dir / "input_person.png")
    shutil.copy2(garment, out_dir / "input_garment.png")
    shutil.copy2(mask, out_dir / "mask_processed.png")

    person_name = copy_to_comfy_input(source_person, "precise_sample_015_p03_hat_fix_person.png")
    garment_name = copy_to_comfy_input(garment, "precise_sample_015_p03_hat_fix_garment.png")
    mask_name = copy_to_comfy_input(mask, "precise_sample_015_p03_hat_fix_mask.png")
    graph = build_upstream_equivalent_prompt(
        person_name=person_name,
        reference_name=garment_name,
        mask_name=mask_name,
        sample_id="sample_015_hat_fix",
        prompt_mode="tryon_prompt",
        seed=2026062533,
        filename_prefix="precise_mask_tryon/sample_015/pass_03_hat_fix",
    )
    graph["4"]["inputs"]["expand"] = 0
    graph["7"]["inputs"]["text"] = (
        "Virtual try-on photo. Add a small pink bucket hat above the head only. "
        "Do not cover the face. Preserve the woman's face, eyes, nose, mouth, hairline, dress, shoes, body, pose, lighting, and background."
    )
    graph["13"]["inputs"]["strength"] = 0.65
    graph["17"]["inputs"]["strength_model"] = 0.75
    graph["18"]["inputs"]["steps"] = 24
    graph["18"]["inputs"]["cfg"] = 4
    graph["18"]["inputs"]["denoise"] = 0.68
    (out_dir / "workflow_used_api.json").write_text(json.dumps(graph, indent=2), encoding="utf-8")

    started = time.perf_counter()
    history = queue_and_wait(graph, timeout_seconds=2400)
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    if not history.get("status", {}).get("completed"):
        raise RuntimeError(json.dumps(history.get("status", {}), ensure_ascii=False))
    images = saved_images(history)
    if not images:
        raise RuntimeError("No image saved")
    output = out_dir / "output_base.png"
    shutil.copy2(images[-1], output)
    shutil.copy2(output, RUN_ROOT / "sample_015_final_output_hat_fix.png")
    metadata = {
        "sample_id": "sample_015",
        "pass_index": 3,
        "target_region": "hat_fix",
        "seed": 2026062533,
        "grow_mask": 0,
        "steps": 24,
        "cfg": 4,
        "denoise": 0.68,
        "redux_strength": 0.65,
        "lora_strength": 0.75,
        "person": source_person.as_posix(),
        "garment": garment.as_posix(),
        "mask": mask.as_posix(),
        "output_base": output.as_posix(),
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "negative_prompt": NEGATIVE_PROMPT,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata), flush=True)
    print(f"output={output.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
