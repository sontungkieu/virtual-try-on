from __future__ import annotations

import json
import shutil
import time
import urllib.request
from pathlib import Path


COMFY = "http://127.0.0.1:8188"
PROJECT_ROOT = Path("/workspace/Project_Phase2/virtual_tryon")
SAMPLE_DIR = PROJECT_ROOT / "data" / "temp" / "vton_phase2_extra_eval_set" / "sample_015"
COMFY_INPUT = Path("/workspace/ComfyUI/input")


def post_json(path: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{COMFY}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_json(path: str) -> dict:
    with urllib.request.urlopen(f"{COMFY}{path}", timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    COMFY_INPUT.mkdir(parents=True, exist_ok=True)
    files = {
        "sample_015_person.png": SAMPLE_DIR / "person.png",
        "sample_015_dress.png": SAMPLE_DIR / "garment_dress.png",
        "sample_015_shoes.png": SAMPLE_DIR / "accessory_shoes.png",
        "sample_015_hat.png": SAMPLE_DIR / "accessory_hat.png",
    }
    for name, path in files.items():
        if not path.exists():
            raise FileNotFoundError(path)
        shutil.copy2(path, COMFY_INPUT / name)

    prompt = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "sample_015_person.png"}},
        "2": {"class_type": "LoadImage", "inputs": {"image": "sample_015_dress.png"}},
        "3": {"class_type": "LoadImage", "inputs": {"image": "sample_015_shoes.png"}},
        "4": {"class_type": "LoadImage", "inputs": {"image": "sample_015_hat.png"}},
        "5": {
            "class_type": "VTONPhase2KleinReferenceSet",
            "inputs": {
                "person_image": ["1", 0],
                "ref1_image": ["2", 0],
                "ref2_image": ["3", 0],
                "ref3_image": ["4", 0],
                "reference_mode": "dress_ref1_hat_ref3_shoes_ref2",
            },
        },
        "6": {
            "class_type": "VTONPhase2KleinSampler",
            "inputs": {
                "person_canvas": ["5", 0],
                "top_reference": ["5", 1],
                "bottom_reference": ["5", 2],
                "method": "klein_4step",
                "prompt": (
                    "TRYON full body adult fashion photo. Replace the outfit with the yellow sleeveless dress "
                    "reference. Add the pink bucket hat on the head and white pointed high heel shoes on both feet. "
                    "Preserve face, hair, body shape, pose, legs, feet position, and studio background."
                ),
                "seed": 94015,
                "guidance_scale": 2.5,
                "lora_scale": 1.0,
            },
        },
        "7": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["6", 0],
                "filename_prefix": "vton_phase2_comfy_smoke/sample_015_klein4",
            },
        },
    }

    response = post_json("/prompt", {"prompt": prompt})
    prompt_id = response["prompt_id"]
    print(f"prompt_id={prompt_id}", flush=True)
    for _ in range(240):
        time.sleep(2)
        history = get_json(f"/history/{prompt_id}")
        if prompt_id not in history:
            continue
        item = history[prompt_id]
        print("status=" + json.dumps(item.get("status", {}), ensure_ascii=False), flush=True)
        print("outputs=" + json.dumps(item.get("outputs", {}), ensure_ascii=False)[:4000], flush=True)
        return 0
    raise TimeoutError("ComfyUI prompt did not finish")


if __name__ == "__main__":
    raise SystemExit(main())
