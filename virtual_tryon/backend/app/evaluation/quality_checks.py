from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

from app.core.config import QualityConfig
from app.preprocessing.mask_utils import invert, to_l_mask
from app.schemas.tryon import QualityScores


def _mean_abs_diff(a: Image.Image, b: Image.Image, mask: Image.Image | None = None) -> float:
    arr_a = np.array(a.convert("RGB"), dtype=np.float32)
    arr_b = np.array(b.convert("RGB").resize(a.size), dtype=np.float32)
    diff = np.abs(arr_a - arr_b).mean(axis=2) / 255.0
    if mask is not None:
        mask_arr = np.array(to_l_mask(mask).resize(a.size), dtype=np.float32) / 255.0
        denom = float(mask_arr.sum())
        if denom <= 1e-6:
            return 0.0
        return float((diff * mask_arr).sum() / denom)
    return float(diff.mean())


def _artifact_heuristic(image: Image.Image, config: QualityConfig) -> tuple[float, list[str]]:
    notes: list[str] = []
    resolution_ok = image.width >= config.min_output_width and image.height >= config.min_output_height
    if not resolution_ok:
        notes.append("Output resolution is below configured minimum.")

    arr = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    mean = float(arr.mean())
    std = float(arr.std())
    if mean < 0.03:
        notes.append("Output image is too dark or blank.")
    if std < 0.015:
        notes.append("Output image has very low color variation.")
    score = max(0.0, min(1.0, 1.0 - (len(notes) * 0.35)))
    return score, notes


def _basic_image_checks(image: Image.Image, config: QualityConfig) -> dict:
    arr = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    mean = float(arr.mean())
    std = float(arr.std())
    return {
        "blank_or_corrupt_check": "fail" if mean < 0.03 else "pass",
        "color_collapse_check": "fail" if std < 0.015 else "pass",
        "output_resolution_check": (
            "pass" if image.width >= config.min_output_width and image.height >= config.min_output_height else "fail"
        ),
    }


def evaluate_output_quality(
    person_image: Image.Image,
    output_image: Image.Image,
    garment_mask: Image.Image,
    config: QualityConfig,
) -> dict:
    notes: list[str] = []
    background_mask = invert(garment_mask)
    background_change = _mean_abs_diff(person_image, output_image, background_mask)
    background_preservation = max(0.0, 1.0 - background_change)
    garment_change = _mean_abs_diff(person_image, output_image, garment_mask)
    over_edit_score = background_change
    artifact_score, artifact_notes = _artifact_heuristic(output_image, config)
    notes.extend(artifact_notes)

    if background_change > config.background_change_threshold:
        notes.append("Background changed more than expected outside the refine mask.")
    if garment_change < config.garment_change_threshold:
        notes.append("Garment region changed too little.")
    notes.append("Face preservation score is unavailable because face parser/bbox is not wired.")

    needs_refine = bool(
        background_change > config.background_change_threshold
        or garment_change < config.garment_change_threshold
        or artifact_score < (1.0 - config.artifact_threshold)
    )

    return {
        "background_preservation_score": background_preservation,
        "face_preservation_score": None,
        "garment_change_score": garment_change,
        "outside_mask_delta": background_change,
        "garment_region_delta": garment_change,
        "over_edit_score": over_edit_score,
        "artifact_heuristic_score": artifact_score,
        "needs_refine": needs_refine,
        "notes": notes,
        **_basic_image_checks(output_image, config),
    }


def refined_is_accepted(refined_quality: dict, config: QualityConfig) -> bool:
    if refined_quality["artifact_heuristic_score"] is not None and refined_quality["artifact_heuristic_score"] < (
        1.0 - config.artifact_threshold
    ):
        return False
    if refined_quality["over_edit_score"] is not None and refined_quality["over_edit_score"] > config.background_change_threshold:
        return False
    return True


def build_quality_report(
    person_image: Image.Image,
    core_image: Image.Image,
    refined_image: Image.Image | None,
    garment_mask: Image.Image,
    config: QualityConfig,
    *,
    refine_notes: list[str] | None = None,
    baselines: dict | None = None,
    engine_status: dict | None = None,
) -> dict:
    core = evaluate_output_quality(person_image, core_image, garment_mask, config)
    refined = None
    final_choice = "core"
    final_choice_reason = "core output is the only accepted image"
    if refined_image is not None:
        refined = evaluate_output_quality(person_image, refined_image, garment_mask, config)
        refined["accepted"] = refined_is_accepted(refined, config)
        if refine_notes:
            refined["notes"].extend(refine_notes)
        final_choice = "refined" if refined["accepted"] else "core"
        final_choice_reason = (
            "refined output passed quality gate"
            if refined["accepted"]
            else "refined output failed quality gate; using core output"
        )
    else:
        refined = {
            "background_preservation_score": None,
            "face_preservation_score": None,
            "garment_change_score": None,
            "outside_mask_delta": None,
            "garment_region_delta": None,
            "over_edit_score": None,
            "artifact_heuristic_score": None,
            "blank_or_corrupt_check": None,
            "color_collapse_check": None,
            "output_resolution_check": None,
            "accepted": False,
            "notes": refine_notes or [],
        }
        if refine_notes:
            final_choice_reason = "refiner unavailable or skipped; using core output"
    return {
        "core": core,
        "refined": refined,
        "baselines": baselines or {"catvton": None, "klein_lora": None},
        "final_choice": final_choice,
        "final_choice_reason": final_choice_reason,
        "engine_status": engine_status
        or {
            "idm_vton": "success",
            "flux_refiner": "skipped",
            "catvton": "skipped",
            "klein_lora": "skipped",
        },
    }


def run_quality_checks(
    person_image: Image.Image,
    output_image: Image.Image,
    garment_image: Image.Image | None,
    garment_mask: Image.Image,
    config: QualityConfig,
) -> QualityScores:
    notes: list[str] = []
    needs_refine = False

    if output_image.width < config.min_output_width or output_image.height < config.min_output_height:
        notes.append("Output resolution is below configured minimum.")
        needs_refine = True

    background_mask = invert(garment_mask)
    background_change = _mean_abs_diff(person_image, output_image, background_mask)
    background_preservation = max(0.0, 1.0 - background_change)
    if background_change > config.background_change_threshold:
        notes.append("Background changed more than expected.")
        needs_refine = True

    garment_change = _mean_abs_diff(person_image, output_image, garment_mask)
    if garment_change < config.garment_change_threshold:
        notes.append("Garment region changed too little; core try-on may have failed.")
        needs_refine = True

    boundary = garment_mask.convert("L").filter(ImageFilter.FIND_EDGES)
    artifact_score = min(1.0, _mean_abs_diff(person_image, output_image, boundary) * 2.0)
    if artifact_score > config.artifact_threshold:
        notes.append("Visible mask boundary artifacts detected.")
        needs_refine = True

    garment_similarity = None
    if garment_image is not None:
        garment_similarity = max(0.0, 1.0 - _mean_abs_diff(garment_image, output_image, garment_mask))

    return QualityScores(
        identity_score=None,
        garment_similarity_score=garment_similarity,
        background_preservation_score=background_preservation,
        artifact_score=artifact_score,
        needs_refine=needs_refine,
        notes=notes,
    )
