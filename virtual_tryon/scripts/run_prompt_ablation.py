from __future__ import annotations

import argparse
import csv
import html
import json
import os
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(BACKEND_ROOT))
sys.path.insert(0, str(SCRIPTS_ROOT))

from app.core.config import load_settings  # noqa: E402
from app.engines.factory import create_tryon_engine  # noqa: E402
from app.prompts.prompt_builder import build_prompt  # noqa: E402
from app.prompts.prompt_types import EngineMode, PromptVariant  # noqa: E402
from app.prompts.testcase_prompt_library import get_testcase  # noqa: E402
from app.services.storage_service import StorageService  # noqa: E402
from app.services.tryon_pipeline import PipelineRequest, TryOnPipeline  # noqa: E402
from app.preprocessing.image_loader import load_image_from_path  # noqa: E402
from validate_eval_set import validate_sample  # noqa: E402


ENGINE_ALIASES = {
    "idm": EngineMode.IDM,
    "idm_mask_expanded": EngineMode.IDM_MASK_EXPANDED,
    "idm_mask_expanded_flux": EngineMode.IDM_MASK_EXPANDED_FLUX,
    "klein_lora": EngineMode.KLEIN_LORA,
    "catvton": EngineMode.CATVTON,
    "adetailer_repair": EngineMode.ADETAILER_REPAIR,
}
MANUAL_COLUMNS = [
    "testcase_id",
    "engine",
    "prompt_variant",
    "identity_1_5",
    "garment_fidelity_1_5",
    "old_garment_removed_1_5",
    "realism_1_5",
    "pose_preservation_1_5",
    "overedit_1_5",
    "winner",
    "notes",
]


def _save_prompt_result(output_dir: Path, result) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "prompt_core.txt").write_text(
        result.core_prompt or result.positive_prompt,
        encoding="utf-8",
    )
    if result.refine_prompt:
        (output_dir / "prompt_refine.txt").write_text(result.refine_prompt, encoding="utf-8")
    if result.negative_prompt:
        (output_dir / "negative_prompt.txt").write_text(result.negative_prompt, encoding="utf-8")
    (output_dir / "prompt_metadata.json").write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _safe_note(value: str) -> str:
    return (
        value.replace("FAL_KEY", "fal credential")
        .replace("Authorization", "redacted authorization")
        .replace("Bearer ", "redacted bearer ")
        .replace("token=", "redacted_token=")
        .replace("key=", "redacted_key=")
    )


def _placeholder(label: str) -> Image.Image:
    image = Image.new("RGB", (256, 342), (245, 246, 248))
    draw = ImageDraw.Draw(image)
    draw.rectangle((12, 12, 244, 330), outline=(170, 176, 184), width=2)
    draw.text((20, 154), label[:32], fill=(70, 76, 84))
    return image


def _write_grid(rows: list[dict], output_dir: Path) -> None:
    if not rows:
        return
    cell_w, cell_h, header_h = 256, 342, 40
    canvas = Image.new("RGB", (len(rows) * cell_w, cell_h + header_h), "white")
    draw = ImageDraw.Draw(canvas)
    for index, row in enumerate(rows):
        result_path = row.get("result_path")
        path = output_dir / result_path if result_path else None
        image = Image.open(path).convert("RGB") if path and path.exists() else _placeholder(row["status"])
        image.thumbnail((cell_w, cell_h), Image.Resampling.LANCZOS)
        x = index * cell_w + (cell_w - image.width) // 2
        y = header_h + (cell_h - image.height) // 2
        canvas.paste(image, (x, y))
        draw.text(
            (index * cell_w + 8, 12),
            f"{row['engine']} / {row['prompt_variant']}"[:38],
            fill=(20, 20, 20),
        )
    canvas.save(output_dir / "comparison_grid.png")


def _write_html(rows: list[dict], output_dir: Path) -> None:
    cards = []
    for row in rows:
        image = (
            f'<img src="{html.escape(row["result_path"])}" alt="result">'
            if row.get("result_path")
            else f'<div class="missing">{html.escape(row["status"])}</div>'
        )
        cards.append(
            f"<section><h2>{html.escape(row['engine'])} / "
            f"{html.escape(row['prompt_variant'])}</h2>{image}"
            f"<p>{html.escape(row.get('notes') or '')}</p></section>"
        )
    (output_dir / "comparison_index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Prompt ablation</title>"
        "<style>body{font-family:Arial;margin:0}main{display:grid;grid-template-columns:"
        "repeat(auto-fit,minmax(260px,1fr));gap:1px;background:#ddd}section{background:#fff;"
        "padding:16px}img{width:100%;height:auto}.missing{min-height:340px;display:grid;"
        "place-items:center;border:1px dashed #999}</style><main>"
        + "".join(cards)
        + "</main>",
        encoding="utf-8",
    )


def run_prompt_ablation(
    *,
    sample_dir: Path,
    testcase_id: str,
    engines: list[str],
    variants: list[PromptVariant],
    output_dir: Path,
    mock: bool = False,
) -> dict:
    sample, issues = validate_sample(sample_dir)
    if sample is None:
        raise ValueError("Invalid eval sample: " + "; ".join(issues))
    testcase = get_testcase(testcase_id)
    settings = load_settings()
    settings.storage.outputs_dir = output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for engine_name in engines:
        if engine_name not in ENGINE_ALIASES:
            raise ValueError(f"Unsupported engine: {engine_name}")
        engine_mode = ENGINE_ALIASES[engine_name]
        for variant in variants:
            variant_dir = output_dir / engine_name / variant.value
            prompt_result = build_prompt(testcase.build_request(engine_mode, variant))
            _save_prompt_result(variant_dir, prompt_result)
            row = {
                "testcase_id": testcase_id,
                "engine": engine_name,
                "prompt_variant": variant.value,
                "prompt_hash": prompt_result.metadata["prompt_hash"],
                "prompt_path": (variant_dir / "prompt_core.txt").relative_to(output_dir).as_posix(),
                "status": "unavailable",
                "runtime_seconds": 0.0,
                "result_path": None,
                "notes": "",
            }
            if engine_mode == EngineMode.ADETAILER_REPAIR:
                row["status"] = "skipped"
                row["notes"] = "Repair mode requires an existing generated output and artifact mask."
                rows.append(row)
                continue
            mode_settings = settings.model_copy(deep=True)
            if engine_mode in {
                EngineMode.IDM,
                EngineMode.IDM_MASK_EXPANDED,
                EngineMode.IDM_MASK_EXPANDED_FLUX,
            }:
                mode_settings.pipeline.engine = "mock" if mock else "idm_vton"
                mode_settings.mask_experiments.upper_body_expand_hem.enabled = engine_mode in {
                    EngineMode.IDM_MASK_EXPANDED,
                    EngineMode.IDM_MASK_EXPANDED_FLUX,
                }
            elif engine_mode == EngineMode.KLEIN_LORA:
                mode_settings.pipeline.engine = "klein_tryon_lora"
                mode_settings.klein_tryon_lora.enabled = True
            elif engine_mode == EngineMode.CATVTON:
                mode_settings.pipeline.engine = "catvton"
            engine = create_tryon_engine(mode_settings)
            if not engine.is_available():
                row["notes"] = _safe_note(
                    engine.status() if hasattr(engine, "status") else "Engine unavailable"
                )
                rows.append(row)
                continue
            started = time.perf_counter()
            request = PipelineRequest(
                job_id=f"{engine_name}/{variant.value}",
                person_image=load_image_from_path(sample.person_path, max_side=settings.image.max_side),
                garment_top=load_image_from_path(sample.garment_top, max_side=settings.image.max_side) if sample.garment_top else None,
                garment_bottom=load_image_from_path(sample.garment_bottom, max_side=settings.image.max_side) if sample.garment_bottom else None,
                garment_dress=load_image_from_path(sample.garment_dress, max_side=settings.image.max_side) if sample.garment_dress else None,
                category=sample.category,
                prompt=prompt_result.core_prompt or prompt_result.positive_prompt,
                use_refiner=engine_mode == EngineMode.IDM_MASK_EXPANDED_FLUX,
                repair_mode=False,
                seed=0,
            )
            try:
                response = TryOnPipeline(mode_settings, StorageService(mode_settings.storage)).run(request)
                row["status"] = response.status
                row["runtime_seconds"] = round(time.perf_counter() - started, 3)
                if response.result_url:
                    result_path = StorageService(mode_settings.storage).file_path_from_public_url(response.result_url)
                    row["result_path"] = result_path.relative_to(output_dir).as_posix()
            except Exception as exc:
                row["status"] = "skipped"
                row["notes"] = _safe_note(str(exc))
                row["runtime_seconds"] = round(time.perf_counter() - started, 3)
            rows.append(row)
    summary = {"testcase_id": testcase_id, "rows": rows}
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    with (output_dir / "manual_ratings_prompt_ablation.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=MANUAL_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "testcase_id": testcase_id,
                    "engine": row["engine"],
                    "prompt_variant": row["prompt_variant"],
                }
            )
    _write_grid(rows, output_dir)
    _write_html(rows, output_dir)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare prompt variants for VTON engines.")
    parser.add_argument("--sample", required=True)
    parser.add_argument("--testcase-id", required=True)
    parser.add_argument("--engine", default="idm,klein_lora")
    parser.add_argument(
        "--variants",
        default="default,strong_remove_old_garment,identity_strict",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--mock", action="store_true", default=os.getenv("TRYON_ENGINE") == "mock")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_prompt_ablation(
        sample_dir=Path(args.sample),
        testcase_id=args.testcase_id,
        engines=[item.strip() for item in args.engine.split(",") if item.strip()],
        variants=[PromptVariant(item.strip()) for item in args.variants.split(",") if item.strip()],
        output_dir=Path(args.output),
        mock=args.mock,
    )
    print(f"rows={len(summary['rows'])}")
    print(f"grid={Path(args.output) / 'comparison_grid.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
