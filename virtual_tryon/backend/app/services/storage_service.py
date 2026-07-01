from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from app.core.config import StorageConfig
from app.services.artifact_service import build_artifact_url, is_allowed_artifact
from app.utils.image_io import save_image


class StorageService:
    def __init__(self, config: StorageConfig) -> None:
        self.config = config
        self.inputs_dir = Path(config.inputs_dir)
        self.outputs_dir = Path(config.outputs_dir)
        self.temp_dir = Path(config.temp_dir)
        for directory in [self.inputs_dir, self.outputs_dir, self.temp_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def job_dir(self, job_id: str) -> Path:
        path = self.outputs_dir / job_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_job_image(self, job_id: str, name: str, image: Image.Image) -> Path:
        return save_image(image, self.job_dir(job_id) / name)

    def save_json(self, job_id: str, name: str, payload: dict[str, Any]) -> Path:
        path = self.job_dir(job_id) / name
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def public_url(self, path: str | Path | None) -> str | None:
        if path is None:
            return None
        path = Path(path)
        if not path.exists():
            return None
        try:
            relative = path.relative_to(self.outputs_dir)
        except ValueError:
            return None
        if len(relative.parts) < 2 or not is_allowed_artifact(relative):
            return None
        return build_artifact_url(relative.parts[0], Path(*relative.parts[1:]), self.config.public_outputs_prefix)

    def file_path_from_public_url(self, url: str) -> Path:
        prefix = self.config.public_outputs_prefix.rstrip("/") + "/"
        if not url.startswith(prefix):
            raise ValueError(f"URL is not under public output prefix: {url}")
        return self.outputs_dir / url[len(prefix) :]
