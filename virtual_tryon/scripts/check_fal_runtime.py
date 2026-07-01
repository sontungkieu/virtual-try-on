from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import load_settings  # noqa: E402
from app.engines.klein_tryon_lora_engine import KleinTryOnLoraEngine  # noqa: E402


def fal_client_available() -> bool:
    try:
        import fal_client  # noqa: F401
    except ImportError:
        return False
    return True


def build_status() -> dict:
    settings = load_settings()
    klein_config = settings.klein_tryon_lora.model_copy(deep=True)
    klein_config.enabled = True
    klein_config.backend = klein_config.backend or "fal_api"
    engine = KleinTryOnLoraEngine(klein_config)
    availability = engine.is_available()
    fal_key_set = bool(os.getenv("FAL_KEY"))
    client_available = fal_client_available()
    messages: list[str] = []
    if not fal_key_set:
        messages.append("FAL_KEY is not set")
    if not client_available:
        messages.append("fal_client package is not installed")
    if availability.missing:
        messages.extend(item for item in availability.missing if item not in messages)
    return {
        "fal_key_set": fal_key_set,
        "fal_client_available": client_available,
        "klein_lora_available": availability.available,
        "engine_status": availability.status,
        "error_code": availability.error_code,
        "messages": messages,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check fal.ai runtime readiness without printing credentials.")
    parser.add_argument("--strict", action="store_true", help="Exit with code 1 if fal runtime is unavailable.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    status = build_status()
    print(json.dumps(status, indent=2, ensure_ascii=False))
    if args.strict and not status["klein_lora_available"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
