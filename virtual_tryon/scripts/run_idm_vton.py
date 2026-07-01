from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.core.config import load_settings  # noqa: E402
from app.engines.idm_vton_engine import IDMVTonEngine  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Check IDM-VTON adapter readiness.")
    parser.add_argument("--check", action="store_true", help="Only check checkpoint/entrypoint availability.")
    args = parser.parse_args()

    settings = load_settings()
    engine = IDMVTonEngine(settings.idm_vton)
    print(f"checkpoint_dir={settings.idm_vton.checkpoint_dir}")
    print(f"repo_path={settings.idm_vton.repo_path}")
    print(f"entrypoint={settings.idm_vton.entrypoint}")
    print(f"available={engine.is_available()}")
    if args.check:
        return 0 if engine.is_available() else 2
    print("Use the backend pipeline or scripts/run_tryon.py for real execution.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
