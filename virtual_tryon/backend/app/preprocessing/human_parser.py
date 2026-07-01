from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HumanParsingResult:
    semantic_map_path: Path | None
    available: bool
    warning: str | None = None


class HumanParser:
    def __init__(self, checkpoint_dir: Path | None = None) -> None:
        self.checkpoint_dir = checkpoint_dir

    def is_available(self) -> bool:
        return bool(self.checkpoint_dir and self.checkpoint_dir.exists() and any(self.checkpoint_dir.iterdir()))

    def parse(self, image: Image.Image, output_dir: Path) -> HumanParsingResult:
        if not self.is_available():
            warning = "Human parsing model is not configured; using heuristic agnostic masks."
            logger.warning(warning)
            return HumanParsingResult(semantic_map_path=None, available=False, warning=warning)
        warning = "Human parsing adapter is configured but not wired to a concrete parser yet."
        logger.warning(warning)
        return HumanParsingResult(semantic_map_path=None, available=False, warning=warning)
