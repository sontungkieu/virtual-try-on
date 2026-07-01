from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import load_settings  # noqa: E402
from app.preprocessing.image_loader import load_image_from_path  # noqa: E402
from app.services.storage_service import StorageService  # noqa: E402
from app.services.tryon_pipeline import PipelineRequest, TryOnPipeline  # noqa: E402
from app.utils.errors import TryOnError  # noqa: E402
from app.utils.image_io import save_image  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
INNERWEAR_BOTTOM_CATEGORIES = {"men_underwear", "women_underwear"}
INNERWEAR_TOP_CATEGORIES = {"women_bra"}
VALID_CATEGORIES = {
    "upper_body",
    "lower_body",
    "dress",
    "full_outfit",
    *INNERWEAR_BOTTOM_CATEGORIES,
    *INNERWEAR_TOP_CATEGORIES,
}


def _discover_samples(examples_dir: Path, max_samples: int) -> list[tuple[str, Path, Path]]:
    persons = sorted(
        path
        for path in examples_dir.glob("person_*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    tops = sorted(
        path
        for path in examples_dir.glob("top_*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    samples: list[tuple[str, Path, Path]] = []
    for idx, (person, garment) in enumerate(zip(persons, tops), start=1):
        samples.append((f"sample_{idx:03d}", person, garment))
        if len(samples) >= max_samples:
            break
    return samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a fixed IDM-VTON baseline suite on local example images.")
    parser.add_argument("--examples-dir", default=str(PROJECT_ROOT / "data" / "examples"))
    parser.add_argument("--max-samples", type=int, default=5)
    parser.add_argument("--category", default="upper_body", choices=sorted(VALID_CATEGORIES))
    parser.add_argument("--prompt", default="replace the shirt with the reference garment, preserve face, pose, and body shape")
    parser.add_argument("--mock", action="store_true", help="Use mock engine for local script validation.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    examples_dir = Path(args.examples_dir)
    settings = load_settings()
    if args.mock:
        settings.pipeline.engine = "mock"
        settings.pipeline.allow_mock_engine = True

    samples = _discover_samples(examples_dir, args.max_samples)
    suite_dir = settings.storage.outputs_dir / "baseline_suite"
    suite_dir.mkdir(parents=True, exist_ok=True)

    if not samples:
        message = (
            f"No paired person_*/top_* images found in {examples_dir}. "
            "Add files like person_001.jpg and top_001.jpg, then rerun this script."
        )
        print(message)
        (suite_dir / "baseline_summary.json").write_text(
            json.dumps({"samples": [], "notes": [message]}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return 0

    if len(samples) < 3:
        print(
            f"Found {len(samples)} paired sample(s). The suite can run with one real sample, "
            "but add more person_*/top_* pairs for a stronger baseline."
        )

    storage = StorageService(settings.storage)
    pipeline = TryOnPipeline(settings, storage)
    summary: list[dict] = []

    for sample_id, person_path, garment_path in samples:
        job_id = f"baseline_suite/{sample_id}"
        job_dir = storage.job_dir(job_id)
        started = time.perf_counter()
        notes: list[str] = []
        try:
            person = load_image_from_path(person_path, max_side=settings.image.max_side)
            garment = load_image_from_path(garment_path, max_side=settings.image.max_side)
            save_image(person, job_dir / "input_person.png")
            save_image(garment, job_dir / "input_garment.png")

            request = PipelineRequest(
                job_id=job_id,
                person_image=person,
                garment_top=garment if args.category in {"upper_body", "full_outfit", *INNERWEAR_TOP_CATEGORIES} else None,
                garment_bottom=garment if args.category in {"lower_body", *INNERWEAR_BOTTOM_CATEGORIES} else None,
                garment_dress=garment if args.category == "dress" else None,
                category=args.category,
                prompt=args.prompt,
                use_refiner=False,
                repair_mode=False,
                seed=0,
            )
            response = pipeline.run(request)
            status = response.status
            output_path = storage.file_path_from_public_url(response.result_url) if response.result_url else None
            if (job_dir / "metadata.json").exists():
                shutil.copyfile(job_dir / "metadata.json", job_dir / "run_metadata.json")
            if (job_dir / "mask_preview.png").exists():
                shutil.copyfile(job_dir / "mask_preview.png", job_dir / "mask_preview_baseline.png")
        except TryOnError as exc:
            status = "failed"
            output_path = None
            notes.append(str(exc))
            (job_dir / "baseline_error.txt").write_text(str(exc), encoding="utf-8")
        except Exception as exc:
            status = "failed"
            output_path = None
            notes.append(f"Unexpected baseline failure: {exc}")
            (job_dir / "baseline_error.txt").write_text(f"Unexpected baseline failure: {exc}", encoding="utf-8")

        summary.append(
            {
                "sample_id": sample_id,
                "runtime_seconds": round(time.perf_counter() - started, 3),
                "status": status,
                "success": status == "completed",
                "output_path": str(output_path) if output_path else None,
                "notes": notes,
            }
        )

    summary_path = suite_dir / "baseline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    successes = sum(1 for row in summary if row["success"])
    print(f"baseline_summary={summary_path}")
    print(f"successes={successes}/{len(summary)}")
    return 0 if successes > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
