from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from virtual_tryon.masking.local_inpaint_ops import (  # noqa: E402
    BBoxCropConfig,
    FitCanvasConfig,
    PasteBackConfig,
    bbox_crop,
    extract_fitted_canvas_region,
    fit_canvas_with_meta,
    masked_paste_back,
)


WORKFLOW = (
    PROJECT_ROOT
    / "virtual_tryon/comfyui_workflows/klein_detailed_pipelines_20260626/"
    / "04_flux2_klein9b_lora_masked_local_inpaint.workflow.json"
)
NODE_FILE = PROJECT_ROOT / "virtual_tryon/comfyui_nodes/vton_phase2_nodes/__init__.py"


def test_workflow_json_uses_registered_custom_nodes() -> None:
    workflow = json.loads(WORKFLOW.read_text(encoding="utf-8"))
    node_text = NODE_FILE.read_text(encoding="utf-8")
    registered = set(re.findall(r'"(VTONPhase2[^"]+)":\s*VTONPhase2', node_text))
    custom_types = {node["type"] for node in workflow["nodes"] if node["type"].startswith("VTONPhase2")}
    assert custom_types
    assert custom_types <= registered


def test_bbox_crop_empty_mask_uses_target_fallback() -> None:
    person = Image.new("RGB", (320, 480), "white")
    mask = Image.new("L", person.size, 0)
    result = bbox_crop(person, mask, BBoxCropConfig(target_region="hat", min_crop_size=128))
    meta = json.loads(result.bbox_json or "{}")
    assert result.image is not None
    assert result.mask is not None
    assert meta["used_fallback"] is True
    assert meta["bbox_width"] >= 128
    assert result.mask.getbbox() is not None


def test_masked_paste_back_preserves_unmasked_area() -> None:
    person = Image.new("RGB", (200, 160), (40, 70, 90))
    crop_box = (50, 40, 150, 120)
    mask = Image.new("L", (100, 80), 0)
    mask.paste(255, (20, 20, 80, 60))
    generated = Image.new("RGB", (100, 80), (220, 40, 80))
    bbox_json = json.dumps({"bbox_xyxy": list(crop_box)})

    result = masked_paste_back(
        person,
        generated,
        mask,
        bbox_json,
        PasteBackConfig(feather_px=0, color_match=False, preserve_outside_mask=True),
    )
    assert result.image is not None
    full_mask = Image.new("L", person.size, 0)
    full_mask.paste(mask, crop_box[:2])
    outside = ImageChops.invert(full_mask)
    diff = ImageChops.difference(person, result.image)
    outside_diff = np.asarray(diff.convert("L"))[np.asarray(outside) > 0]
    assert outside_diff.max() == 0


def test_masked_paste_back_resizes_generated_crop() -> None:
    person = Image.new("RGB", (200, 160), (10, 20, 30))
    mask = Image.new("L", (100, 80), 255)
    generated = Image.new("RGB", (768, 1024), (200, 100, 20))
    result = masked_paste_back(
        person,
        generated,
        mask,
        json.dumps({"bbox_xyxy": [50, 40, 150, 120]}),
        PasteBackConfig(feather_px=2, color_match=False, preserve_outside_mask=True),
    )
    assert result.image is not None
    assert result.image.size == person.size
    assert result.overlay is not None


def test_fit_canvas_extract_round_trip_size() -> None:
    crop = Image.new("RGB", (180, 120), (80, 120, 160))
    fitted = fit_canvas_with_meta(crop, FitCanvasConfig(width=256, height=256))
    assert fitted.image is not None
    assert fitted.bbox_json is not None
    extracted = extract_fitted_canvas_region(fitted.image, fitted.bbox_json)
    assert extracted.image is not None
    assert extracted.image.size == crop.size
