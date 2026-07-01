from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(BACKEND_ROOT))
sys.path.insert(0, str(SCRIPTS_ROOT))

from app.core.config import Settings, load_settings  # noqa: E402
from app.engines.factory import create_refiner  # noqa: E402
from app.evaluation.quality_checks import build_quality_report  # noqa: E402
from app.preprocessing.image_loader import load_image_from_path  # noqa: E402
from app.prompts.prompt_builder import build_prompt  # noqa: E402
from app.prompts.prompt_types import EngineMode, PromptVariant  # noqa: E402
from app.prompts.testcase_prompt_library import get_testcase  # noqa: E402
from app.services.storage_service import StorageService  # noqa: E402
from app.services.tryon_pipeline import PipelineRequest, TryOnPipeline  # noqa: E402
from app.utils.errors import ModelUnavailableError  # noqa: E402
from app.utils.image_io import save_image  # noqa: E402
from validate_eval_set import EvalSample, validate_sample  # noqa: E402


SUPPORTED_VARIANTS = {
    "idm_original",
    "idm_mask_expanded",
    "idm_mask_expanded_flux_local",
}
DEFAULT_VARIANTS = [
    "idm_original",
    "idm_mask_expanded",
    "idm_mask_expanded_flux_local",
]
REFINER_PROMPT = (
    "Remove any remaining pink shirt visible under the blue velvet wrap top. "
    "Keep only the blue velvet wrap top from the reference garment. Preserve the person's "
    "face, hair, hands, body shape, black pants, and background."
)
SUMMARY_COLUMNS = [
    "sample_id",
    "variant",
    "seed",
    "mask_config",
    "runtime_seconds",
    "status",
    "final_choice",
    "prompt_variant",
    "prompt_hash",
    "prompt_path",
    "notes",
]
MANUAL_RATING_COLUMNS = [
    "sample_id",
    "variant",
    "identity_1_5",
    "garment_fidelity_1_5",
    "old_garment_removed_1_5",
    "realism_1_5",
    "pose_preservation_1_5",
    "overedit_1_5",
    "winner",
    "notes",
]


def _prompt_artifact_paths(variant_dir: Path, output_dir: Path) -> dict[str, str | None]:
    prompt_path = variant_dir / "prompt_core.txt"
    legacy_prompt_path = variant_dir / "prompt.txt"
    metadata_path = variant_dir / "prompt_metadata.json"
    prompt_hash = None
    if metadata_path.exists():
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            prompt_hash = (payload.get("metadata") or {}).get("prompt_hash")
        except json.JSONDecodeError:
            prompt_hash = None
    return {
        "prompt_path": (
            prompt_path.relative_to(output_dir).as_posix()
            if prompt_path.exists()
            else legacy_prompt_path.relative_to(output_dir).as_posix()
            if legacy_prompt_path.exists()
            else None
        ),
        "prompt_hash": prompt_hash,
    }


def _engine_mode_for_variant(variant: str) -> str:
    if variant == "idm_mask_expanded_flux_local":
        return "idm_mask_expanded_flux"
    if variant == "idm_mask_expanded":
        return "idm_mask_expanded"
    return "idm_vton"


def _build_flux_prompt(testcase_id: str | None, prompt_variant: str):
    if not testcase_id:
        return None
    testcase = get_testcase(testcase_id)
    return build_prompt(
        testcase.build_request(
            EngineMode.IDM_MASK_EXPANDED_FLUX,
            PromptVariant(prompt_variant),
        )
    )


def parse_variants(value: str) -> list[str]:
    variants = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [item for item in variants if item not in SUPPORTED_VARIANTS]
    if invalid:
        raise ValueError(f"Unsupported variant(s): {', '.join(invalid)}")
    return variants or list(DEFAULT_VARIANTS)


def _variant_settings(
    base_settings: Settings,
    variant: str,
    output_dir: Path,
    *,
    mock: bool,
) -> Settings:
    settings = base_settings.model_copy(deep=True)
    settings.storage.outputs_dir = output_dir
    settings.pipeline.engine = "mock" if mock else "idm_vton"
    settings.mask_experiments.upper_body_expand_hem.enabled = variant != "idm_original"
    return settings


def _request_for_sample(
    sample: EvalSample,
    *,
    job_id: str,
    seed: int,
    settings: Settings,
    prompt: str | None,
    prompt_source: str,
    prompt_variant: str,
    testcase_id: str | None,
) -> PipelineRequest:
    return PipelineRequest(
        job_id=job_id,
        person_image=load_image_from_path(sample.person_path, max_side=settings.image.max_side),
        garment_top=(
            load_image_from_path(sample.garment_top, max_side=settings.image.max_side)
            if sample.garment_top
            else None
        ),
        garment_bottom=(
            load_image_from_path(sample.garment_bottom, max_side=settings.image.max_side)
            if sample.garment_bottom
            else None
        ),
        garment_dress=(
            load_image_from_path(sample.garment_dress, max_side=settings.image.max_side)
            if sample.garment_dress
            else None
        ),
        category=sample.category,
        prompt=prompt,
        use_refiner=False,
        repair_mode=False,
        seed=seed,
        engine_mode=_engine_mode_for_variant(job_id),
        testcase_id=testcase_id,
        prompt_variant=prompt_variant,
        auto_prompt=prompt_source == "auto" and testcase_id is not None,
    )


def _quality_notes(report: dict[str, Any] | None) -> str:
    if not report:
        return ""
    final_choice = report.get("final_choice") or "core"
    selected = report.get(final_choice) or report.get("core") or {}
    notes = [str(note) for note in selected.get("notes") or []]
    reason = report.get("final_choice_reason")
    if reason:
        notes.insert(0, str(reason))
    return "; ".join(notes)


def _run_core_variant(
    sample: EvalSample,
    variant: str,
    output_dir: Path,
    base_settings: Settings,
    *,
    seed: int,
    mock: bool,
    prompt_source: str = "manual",
    prompt_variant: str = "default",
    testcase_id: str | None = None,
) -> tuple[dict[str, Any], Path]:
    started = time.perf_counter()
    settings = _variant_settings(base_settings, variant, output_dir, mock=mock)
    storage = StorageService(settings.storage)
    variant_dir = output_dir / variant
    request = _request_for_sample(
        sample,
        job_id=variant,
        seed=seed,
        settings=settings,
        prompt=(
            None
            if prompt_source == "auto" and testcase_id
            else "replace the shirt with the reference garment, preserve face, pose, and body shape"
        ),
        prompt_source=prompt_source,
        prompt_variant=prompt_variant,
        testcase_id=testcase_id,
    )
    try:
        response = TryOnPipeline(settings, storage).run(request)
        report_path = variant_dir / "quality_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        row = {
            "sample_id": sample.sample_id,
            "variant": variant,
            "seed": seed,
            "mask_config": (
                "upper_body_expand_hem"
                if settings.mask_experiments.upper_body_expand_hem.enabled
                else "default"
            ),
            "runtime_seconds": round(time.perf_counter() - started, 3),
            "status": response.status,
            "final_choice": report.get("final_choice", "core"),
            "prompt_variant": prompt_variant if prompt_source == "auto" and testcase_id else "manual",
            **_prompt_artifact_paths(variant_dir, output_dir),
            "notes": _quality_notes(report),
            "output_path": f"{variant}/result.png",
            "metadata_path": f"{variant}/metadata.json",
            "quality_report_path": f"{variant}/quality_report.json",
        }
        return row, variant_dir
    except Exception as exc:
        variant_dir.mkdir(parents=True, exist_ok=True)
        (variant_dir / "ablation_error.txt").write_text(str(exc), encoding="utf-8")
        return (
            {
                "sample_id": sample.sample_id,
                "variant": variant,
                "seed": seed,
                "mask_config": (
                    "upper_body_expand_hem"
                    if settings.mask_experiments.upper_body_expand_hem.enabled
                    else "default"
                ),
                "runtime_seconds": round(time.perf_counter() - started, 3),
                "status": "failed",
                "final_choice": None,
                "prompt_variant": prompt_variant if prompt_source == "auto" and testcase_id else "manual",
                "prompt_hash": None,
                "prompt_path": None,
                "notes": str(exc),
                "output_path": None,
                "metadata_path": None,
                "quality_report_path": None,
            },
            variant_dir,
        )


def run_flux_local_variant(
    sample: EvalSample,
    output_dir: Path,
    base_settings: Settings,
    expanded_dir: Path,
    *,
    seed: int,
    refiner_factory=create_refiner,
    prompt_source: str = "manual",
    prompt_variant: str = "default",
    testcase_id: str | None = None,
) -> dict[str, Any]:
    variant = "idm_mask_expanded_flux_local"
    variant_dir = output_dir / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    settings = _variant_settings(base_settings, variant, output_dir, mock=False)

    core_path = expanded_dir / "core_output.png"
    person_path = expanded_dir / "person.png"
    garment_path = expanded_dir / "garment_normalized.png"
    safe_mask_path = expanded_dir / "safe_refine_mask.png"
    boundary_mask_path = expanded_dir / "boundary_refine_mask.png"
    selected_mask_path = safe_mask_path if safe_mask_path.exists() else boundary_mask_path
    status_payload: dict[str, Any] = {
        "status": "not_started",
        "backend": settings.flux_refiner.backend,
        "mask_source": selected_mask_path.name,
        "seed": seed,
    }
    prompt_result = (
        _build_flux_prompt(testcase_id, prompt_variant)
        if prompt_source == "auto" and testcase_id
        else None
    )
    refiner_prompt = prompt_result.refine_prompt if prompt_result and prompt_result.refine_prompt else REFINER_PROMPT
    (variant_dir / "refiner_prompt.txt").write_text(refiner_prompt, encoding="utf-8")
    if prompt_result:
        (variant_dir / "prompt_core.txt").write_text(
            prompt_result.core_prompt or prompt_result.positive_prompt,
            encoding="utf-8",
        )
        (variant_dir / "prompt_refine.txt").write_text(refiner_prompt, encoding="utf-8")
        if prompt_result.negative_prompt:
            (variant_dir / "negative_prompt.txt").write_text(prompt_result.negative_prompt, encoding="utf-8")
        (variant_dir / "prompt_metadata.json").write_text(
            json.dumps(prompt_result.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    if not core_path.exists() or not selected_mask_path.exists():
        status_payload.update(
            status="skipped",
            reason="Expanded-mask core output or local refinement mask is missing.",
        )
        (variant_dir / "refiner_status.json").write_text(
            json.dumps(status_payload, indent=2),
            encoding="utf-8",
        )
        return {
            "sample_id": sample.sample_id,
            "variant": variant,
            "seed": seed,
            "mask_config": "upper_body_expand_hem+safe_refine_mask",
            "runtime_seconds": round(time.perf_counter() - started, 3),
            "status": "skipped",
            "final_choice": "core",
            "prompt_variant": prompt_variant if prompt_result else "manual",
            **_prompt_artifact_paths(variant_dir, output_dir),
            "notes": status_payload["reason"],
            "output_path": None,
            "metadata_path": None,
            "quality_report_path": None,
        }

    shutil.copy2(core_path, variant_dir / "core_output.png")
    shutil.copy2(selected_mask_path, variant_dir / "refiner_mask.png")
    core_image = Image.open(core_path).convert("RGB")
    person_image = Image.open(person_path).convert("RGB")
    garment_image = Image.open(garment_path).convert("RGB")
    mask_image = Image.open(selected_mask_path).convert("L")

    local_model_path = settings.flux_refiner.model_path
    local_checkpoint_dir = settings.flux_refiner.checkpoint_dir
    has_local_model = bool(
        (local_model_path and local_model_path.exists())
        or (
            local_checkpoint_dir
            and local_checkpoint_dir.exists()
            and any(local_checkpoint_dir.iterdir())
        )
    )
    if settings.flux_refiner.backend in {"flux2_dev", "flux2_klein"} and not has_local_model:
        reason = (
            "FLUX local refine unavailable: no local model_path or populated checkpoint_dir; "
            "remote model download is disabled for this ablation."
        )
        status_payload.update(status="skipped", reason=reason)
        (variant_dir / "refiner_status.json").write_text(
            json.dumps(status_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return {
            "sample_id": sample.sample_id,
            "variant": variant,
            "seed": seed,
            "mask_config": "upper_body_expand_hem+safe_refine_mask",
            "runtime_seconds": round(time.perf_counter() - started, 3),
            "status": "skipped",
            "final_choice": "core",
            "prompt_variant": prompt_variant if prompt_result else "manual",
            **_prompt_artifact_paths(variant_dir, output_dir),
            "notes": reason,
            "output_path": None,
            "metadata_path": None,
            "quality_report_path": None,
        }

    refiner = refiner_factory(settings)

    try:
        if not refiner.is_available():
            raise ModelUnavailableError(refiner.status())
        refined = refiner.refine(
            core_image,
            mask_image,
            refiner_prompt,
            references={"person": person_image, "garment": garment_image},
            seed=seed,
        )
        refined_path = save_image(refined.image, variant_dir / "refined_output.png")
        quality_report = build_quality_report(
            person_image,
            core_image,
            refined.image,
            mask_image,
            settings.quality,
            engine_status={
                "idm_vton": "success",
                "flux_refiner": "success",
                "catvton": "skipped",
                "klein_lora": "skipped",
            },
        )
        (variant_dir / "quality_report.json").write_text(
            json.dumps(quality_report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        status_payload.update(
            status="success",
            runtime_seconds=refined.metadata.get("runtime_seconds"),
            metadata=refined.metadata,
        )
        status = "completed"
        final_choice = quality_report["final_choice"]
        notes = _quality_notes(quality_report)
        output_path = refined_path.relative_to(output_dir).as_posix()
        quality_report_path = f"{variant}/quality_report.json"
    except (ModelUnavailableError, RuntimeError) as exc:
        status_payload.update(status="skipped", reason=str(exc))
        status = "skipped"
        final_choice = "core"
        notes = f"FLUX local refine skipped: {exc}"
        output_path = None
        quality_report_path = None
    except Exception as exc:
        status_payload.update(status="skipped", reason=f"{type(exc).__name__}: {exc}")
        status = "skipped"
        final_choice = "core"
        notes = f"FLUX local refine skipped: {type(exc).__name__}: {exc}"
        output_path = None
        quality_report_path = None

    (variant_dir / "refiner_status.json").write_text(
        json.dumps(status_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {
        "sample_id": sample.sample_id,
        "variant": variant,
        "seed": seed,
        "mask_config": "upper_body_expand_hem+safe_refine_mask",
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "status": status,
        "final_choice": final_choice,
        "prompt_variant": prompt_variant if prompt_result else "manual",
        **_prompt_artifact_paths(variant_dir, output_dir),
        "notes": notes,
        "output_path": output_path,
        "metadata_path": None,
        "quality_report_path": quality_report_path,
    }


def _load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _write_comparison_grid(
    sample: EvalSample,
    output_dir: Path,
    rows: list[dict[str, Any]],
) -> Path:
    row_by_variant = {row["variant"]: row for row in rows}
    columns: list[tuple[str, Path | None]] = [
        ("Person input", sample.person_path),
        ("Garment reference", sample.primary_garment()),
    ]
    for label, variant in [
        ("IDM original", "idm_original"),
        ("IDM expanded mask", "idm_mask_expanded"),
        ("IDM expanded + FLUX", "idm_mask_expanded_flux_local"),
    ]:
        row = row_by_variant.get(variant)
        path = output_dir / row["output_path"] if row and row.get("output_path") else None
        columns.append((label, path))

    cell_w, cell_h, header_h = 360, 480, 58
    grid = Image.new("RGB", (cell_w * len(columns), cell_h + header_h), "white")
    draw = ImageDraw.Draw(grid)
    font = _load_font(20)
    for index, (label, path) in enumerate(columns):
        x0 = index * cell_w
        bbox = draw.textbbox((0, 0), label, font=font)
        draw.text(
            (x0 + (cell_w - (bbox[2] - bbox[0])) // 2, 16),
            label,
            fill=(20, 20, 20),
            font=font,
        )
        if path and path.exists():
            image = Image.open(path).convert("RGB")
            image.thumbnail((cell_w, cell_h), Image.Resampling.LANCZOS)
            grid.paste(
                image,
                (
                    x0 + (cell_w - image.width) // 2,
                    header_h + (cell_h - image.height) // 2,
                ),
            )
        else:
            draw.rectangle(
                (x0 + 20, header_h + 20, x0 + cell_w - 20, header_h + cell_h - 20),
                outline=(180, 180, 180),
                width=2,
            )
            draw.text((x0 + 120, header_h + cell_h // 2), "Unavailable", fill=(110, 110, 110))
        if index:
            draw.line((x0, 0, x0, cell_h + header_h), fill=(210, 210, 210), width=2)
    path = output_dir / "comparison_grid.png"
    grid.save(path)
    return path


def _write_comparison_html(
    sample: EvalSample,
    output_dir: Path,
    rows: list[dict[str, Any]],
) -> Path:
    row_by_variant = {row["variant"]: row for row in rows}
    cells = [
        ("Person input", sample.person_path),
        ("Garment reference", sample.primary_garment()),
    ]
    for label, variant in [
        ("IDM original", "idm_original"),
        ("IDM expanded mask", "idm_mask_expanded"),
        ("IDM expanded mask + FLUX local refine", "idm_mask_expanded_flux_local"),
    ]:
        row = row_by_variant.get(variant)
        cells.append((label, output_dir / row["output_path"] if row and row.get("output_path") else None))

    html_cells: list[str] = []
    for index, (label, path) in enumerate(cells):
        if index == 0:
            asset = output_dir / "input_person.png"
        elif index == 1:
            asset = output_dir / "input_garment.png"
        else:
            asset = path
        image_html = (
            f'<img src="{html.escape(asset.relative_to(output_dir).as_posix())}" alt="{html.escape(label)}">'
            if asset and asset.exists()
            else '<div class="missing">Unavailable or skipped</div>'
        )
        html_cells.append(f"<section><h2>{html.escape(label)}</h2>{image_html}</section>")
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Upper-body mask ablation</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; color: #161616; background: #f5f5f5; }}
    header {{ padding: 20px 24px; background: #fff; border-bottom: 1px solid #ddd; }}
    main {{ display: grid; grid-template-columns: repeat(5, minmax(220px, 1fr)); gap: 1px; background: #ddd; }}
    section {{ min-width: 0; padding: 16px; background: #fff; }}
    h1 {{ margin: 0; font-size: 24px; }} h2 {{ min-height: 44px; margin: 0 0 12px; font-size: 16px; }}
    img {{ display: block; width: 100%; height: auto; object-fit: contain; }}
    .missing {{ display: grid; min-height: 320px; place-items: center; color: #666; border: 1px dashed #aaa; }}
    @media (max-width: 900px) {{ main {{ grid-template-columns: 1fr; }} h2 {{ min-height: 0; }} }}
  </style>
</head>
<body>
  <header><h1>Upper-body mask ablation: {html.escape(sample.sample_id)}</h1></header>
  <main>{''.join(html_cells)}</main>
</body>
</html>
"""
    path = output_dir / "comparison_index.html"
    path.write_text(document, encoding="utf-8")
    return path


def _write_manual_template(
    sample_id: str,
    variants: list[str],
    output_dir: Path,
) -> Path:
    path = output_dir / "manual_ratings_mask_ablation.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANUAL_RATING_COLUMNS)
        writer.writeheader()
        for variant in variants:
            writer.writerow({"sample_id": sample_id, "variant": variant})
    return path


def run_ablation(
    *,
    sample_dir: Path,
    seed: int,
    variants: list[str],
    output_dir: Path,
    mock: bool = False,
    refiner_factory=create_refiner,
    prompt_source: str = "manual",
    prompt_variant: str = "default",
    save_prompts: bool = False,
    testcase_id: str | None = None,
) -> dict[str, Any]:
    sample, issues = validate_sample(sample_dir)
    if sample is None:
        raise ValueError("Invalid eval sample: " + "; ".join(issues))
    output_dir.mkdir(parents=True, exist_ok=True)
    base_settings = load_settings()

    person = load_image_from_path(sample.person_path, max_side=base_settings.image.max_side)
    garment_path = sample.primary_garment()
    if garment_path is None:
        raise ValueError(f"No primary garment found for {sample.sample_id}")
    garment = load_image_from_path(garment_path, max_side=base_settings.image.max_side)
    save_image(person, output_dir / "input_person.png")
    save_image(garment, output_dir / "input_garment.png")

    rows: list[dict[str, Any]] = []
    expanded_dir = output_dir / "idm_mask_expanded"
    core_rows: dict[str, dict[str, Any]] = {}
    for variant in ["idm_original", "idm_mask_expanded"]:
        if variant not in variants and not (
            variant == "idm_mask_expanded"
            and "idm_mask_expanded_flux_local" in variants
        ):
            continue
        row, variant_dir = _run_core_variant(
            sample,
            variant,
            output_dir,
            base_settings,
            seed=seed,
            mock=mock,
            prompt_source=prompt_source,
            prompt_variant=prompt_variant,
            testcase_id=testcase_id,
        )
        core_rows[variant] = row
        if variant in variants:
            rows.append(row)
        if variant == "idm_mask_expanded":
            expanded_dir = variant_dir

    if "idm_mask_expanded_flux_local" in variants:
        if mock:
            class UnavailableRefiner:
                def is_available(self) -> bool:
                    return False

                def status(self) -> str:
                    return "unavailable: mock ablation does not require FLUX"

            flux_factory = lambda _settings: UnavailableRefiner()
        else:
            flux_factory = refiner_factory
        rows.append(
            run_flux_local_variant(
                sample,
                output_dir,
                base_settings,
                expanded_dir,
                seed=seed,
                refiner_factory=flux_factory,
                prompt_source=prompt_source,
                prompt_variant=prompt_variant,
                testcase_id=testcase_id,
            )
        )

    summary = {
        "sample_id": sample.sample_id,
        "seed": seed,
        "variants": variants,
        "rows": rows,
        "production_default_changed": False,
        "prompt_source": prompt_source,
        "prompt_variant": prompt_variant,
        "save_prompts": save_prompts,
    }
    summary_json = output_dir / "summary.json"
    summary_csv = output_dir / "summary.csv"
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    _write_comparison_grid(sample, output_dir, rows)
    _write_comparison_html(sample, output_dir, rows)
    _write_manual_template(sample.sample_id, variants, output_dir)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an upper-body garment-mask ablation.")
    parser.add_argument("--sample", default="data/eval_set/sample_001")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--output", default=None)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--prompt-source", default="manual", choices=["manual", "auto"])
    parser.add_argument(
        "--prompt-variant",
        default="default",
        choices=[variant.value for variant in PromptVariant],
    )
    parser.add_argument("--save-prompts", action="store_true")
    parser.add_argument("--testcase-id", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    variants = parse_variants(args.variants)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output)
        if args.output
        else PROJECT_ROOT / "data" / "outputs" / f"ablation_upper_body_mask_{timestamp}"
    )
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    summary = run_ablation(
        sample_dir=Path(args.sample),
        seed=args.seed,
        variants=variants,
        output_dir=output_dir,
        mock=args.mock,
        prompt_source=args.prompt_source,
        prompt_variant=args.prompt_variant,
        save_prompts=args.save_prompts,
        testcase_id=args.testcase_id,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"comparison_grid={output_dir / 'comparison_grid.png'}")
    print(f"comparison_index={output_dir / 'comparison_index.html'}")
    print(f"summary_csv={output_dir / 'summary.csv'}")
    print(f"summary_json={output_dir / 'summary.json'}")
    print(f"manual_ratings={output_dir / 'manual_ratings_mask_ablation.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
