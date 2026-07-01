from __future__ import annotations

import os
from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(os.getenv("VIRTUAL_TRYON_ROOT", Path(__file__).resolve().parents[3]))
REPO_ROOT = PROJECT_ROOT.parent
CONFIG_DIR = PROJECT_ROOT / "configs"


def resolve_project_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def ensure_directory(path: str | Path) -> Path:
    resolved = resolve_project_path(path) if not Path(path).is_absolute() else Path(path)
    assert resolved is not None
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved
