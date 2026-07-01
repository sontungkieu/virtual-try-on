from __future__ import annotations

import logging
import time

from PIL import Image, ImageFilter

from app.core.config import RepairConfig
from app.engines.base import RefineResult
from app.preprocessing.mask_utils import blur, dilate


logger = logging.getLogger(__name__)


class ADetailerRepairEngine:
    name = "adetailer_repair"

    def __init__(self, config: RepairConfig) -> None:
        self.config = config

    def is_available(self) -> bool:
        return self.config.enabled

    def prepare(self) -> None:
        return None

    def refine(
        self,
        image: Image.Image,
        mask: Image.Image | None,
        prompt: str,
        references: dict | None = None,
        seed: int | None = None,
    ) -> RefineResult:
        start = time.perf_counter()
        base = image.convert("RGB")
        if mask is None:
            elapsed = time.perf_counter() - start
            return RefineResult(base, {"engine": self.name, "runtime_seconds": elapsed, "skipped": "no_mask"})

        repair_mask = blur(dilate(mask, self.config.mask_dilation_px), self.config.mask_blur_radius)
        repaired = base.filter(ImageFilter.UnsharpMask(radius=0.8, percent=110, threshold=4))
        out = base.copy()
        out.paste(repaired, mask=repair_mask)
        elapsed = time.perf_counter() - start
        logger.info("ADetailer-like repair completed in %.2fs", elapsed)
        return RefineResult(out, {"engine": self.name, "runtime_seconds": elapsed, "prompt": prompt, "seed": seed})
