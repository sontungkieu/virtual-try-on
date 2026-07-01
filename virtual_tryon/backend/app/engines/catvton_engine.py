from __future__ import annotations

import logging
from pathlib import Path

from app.core.config import EngineConfig
from app.engines.base import TryOnInputs, TryOnResult
from app.utils.errors import ModelUnavailableError


logger = logging.getLogger(__name__)


class CatVTonEngine:
    name = "catvton"

    def __init__(self, config: EngineConfig) -> None:
        self.config = config

    def missing_requirements(self) -> list[str]:
        missing: list[str] = []
        if not self.config.enabled:
            missing.append("catvton.enabled is false")
        if not self.config.repo_path or not self.config.repo_path.exists():
            missing.append(f"repo_path not found: {self.config.repo_path}")
        if not self.config.entrypoint or not self.config.entrypoint.exists():
            missing.append(f"entrypoint not found: {self.config.entrypoint}")
        checkpoint_dir = self.config.checkpoint_dir
        if not checkpoint_dir or not checkpoint_dir.exists() or not any(checkpoint_dir.iterdir()):
            missing.append(f"checkpoint not found at {checkpoint_dir}")
        return missing

    def status(self) -> str:
        missing = self.missing_requirements()
        if missing:
            return "unavailable: " + "; ".join(missing)
        return "available"

    def is_available(self) -> bool:
        return not self.missing_requirements()

    def prepare(self) -> None:
        missing = self.missing_requirements()
        if missing:
            raise ModelUnavailableError("CatVTON is not available. " + "; ".join(missing))

    def run(self, inputs: TryOnInputs) -> TryOnResult:
        self.prepare()
        if inputs.output_dir:
            (Path(inputs.output_dir) / "catvton_command.txt").write_text(
                "CatVTON adapter is configured but execution command is not wired yet.",
                encoding="utf-8",
            )
        raise ModelUnavailableError(
            "CatVTON adapter is reserved as a baseline. Configure catvton.entrypoint before running this engine."
        )
