from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DensePoseResult:
    densepose_path: Path | None
    available: bool
    warning: str | None = None


class DensePoseEstimator:
    def __init__(self, checkpoint_dir: Path | None = None) -> None:
        self.checkpoint_dir = checkpoint_dir

    def is_available(self) -> bool:
        return bool(self.checkpoint_dir and self.checkpoint_dir.exists() and any(self.checkpoint_dir.iterdir()))

    def estimate(self, image: Image.Image, output_dir: Path) -> DensePoseResult:
        if not self.is_available():
            warning = "DensePose dependency/checkpoint is not configured; continuing without densepose conditioning."
            logger.warning(warning)
            return DensePoseResult(densepose_path=None, available=False, warning=warning)
        warning = "DensePose adapter is configured but not wired to a concrete estimator yet."
        logger.warning(warning)
        return DensePoseResult(densepose_path=None, available=False, warning=warning)
