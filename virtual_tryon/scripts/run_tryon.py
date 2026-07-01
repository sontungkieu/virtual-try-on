from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import load_settings  # noqa: E402
from app.preprocessing.image_loader import load_image_from_path  # noqa: E402
from app.services.storage_service import StorageService  # noqa: E402
from app.services.tryon_pipeline import PipelineRequest, TryOnPipeline  # noqa: E402
from app.utils.errors import TryOnError  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single Virtual Try-On job.")
    parser.add_argument("--person", required=True, help="Path to person image.")
    parser.add_argument("--garment", required=True, help="Path to garment image.")
    parser.add_argument(
        "--category",
        default="upper_body",
        choices=[
            "upper_body",
            "lower_body",
            "dress",
            "full_outfit",
            "men_underwear",
            "women_underwear",
            "women_bra",
        ],
    )
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--use-refiner", action="store_true")
    parser.add_argument("--repair-mode", action="store_true", help="Run local repair after an accepted refiner output.")
    parser.add_argument("--no-repair", action="store_true", help="Deprecated compatibility flag; repair is off by default.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", required=True, help="Output image path.")
    parser.add_argument("--mock", action="store_true", help="Use mock engine for end-to-end validation without checkpoints.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = load_settings()
    if args.mock:
        settings.pipeline.engine = "mock"
        settings.pipeline.allow_mock_engine = True

    storage = StorageService(settings.storage)
    pipeline = TryOnPipeline(settings, storage)
    person = load_image_from_path(args.person, max_side=settings.image.max_side)
    garment = load_image_from_path(args.garment, max_side=settings.image.max_side)
    job_id = f"cli_{Path(args.output).stem}"

    request = PipelineRequest(
        job_id=job_id,
        person_image=person,
        garment_top=garment if args.category in {"upper_body", "women_bra", "full_outfit"} else None,
        garment_bottom=garment if args.category in {"lower_body", "men_underwear", "women_underwear"} else None,
        garment_dress=garment if args.category == "dress" else None,
        category=args.category,
        prompt=args.prompt,
        use_refiner=args.use_refiner,
        repair_mode=args.repair_mode and not args.no_repair,
        seed=args.seed,
    )
    try:
        response = pipeline.run(request)
    except TryOnError as exc:
        raise SystemExit(f"Try-on failed: {exc}") from exc
    if not response.result_url:
        raise SystemExit("Pipeline completed without result_url.")

    result_path = storage.file_path_from_public_url(response.result_url)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(result_path, output_path)
    print(f"job_id={response.job_id}")
    print(f"result={output_path}")
    print(f"seed={response.seed}")
    if response.quality:
        print(response.quality.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
