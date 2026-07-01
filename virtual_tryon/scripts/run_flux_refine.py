from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.core.config import load_settings  # noqa: E402
from app.engines.flux_refiner_engine import FluxRefinerEngine  # noqa: E402
from app.preprocessing.image_loader import load_image_from_path  # noqa: E402
from app.utils.image_io import save_image  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run configured FLUX refiner adapter on one image.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask", default=None)
    parser.add_argument("--prompt", default="Refine garment boundaries while preserving identity and background.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    settings = load_settings()
    image = load_image_from_path(args.image, max_side=settings.image.max_side)
    mask = load_image_from_path(args.mask, max_side=settings.image.max_side).convert("L") if args.mask else None
    result = FluxRefinerEngine(settings.flux_refiner).refine(image, mask, args.prompt, seed=args.seed)
    save_image(result.image, args.output)
    print(f"result={args.output}")
    print(result.metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
