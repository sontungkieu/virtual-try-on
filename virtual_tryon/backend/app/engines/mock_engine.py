from __future__ import annotations

import time

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from app.engines.base import RefineResult, TryOnInputs, TryOnResult


class MockTryOnEngine:
    name = "mock"

    def is_available(self) -> bool:
        return True

    def prepare(self) -> None:
        return None

    def run(self, inputs: TryOnInputs) -> TryOnResult:
        start = time.perf_counter()
        person = inputs.person_image.convert("RGB")
        garment = inputs.garment_image.convert("RGB").resize(person.size, Image.Resampling.LANCZOS)
        mask = inputs.agnostic_mask.convert("L").filter(ImageFilter.GaussianBlur(radius=4))

        person_arr = np.array(person, dtype=np.float32)
        garment_arr = np.array(garment, dtype=np.float32)
        alpha = (np.array(mask, dtype=np.float32) / 255.0)[..., None] * 0.82
        out = person_arr * (1.0 - alpha) + garment_arr * alpha
        image = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")
        elapsed = time.perf_counter() - start
        return TryOnResult(image, {"engine": self.name, "runtime_seconds": elapsed, "mock": True})


class MockRefinerEngine:
    name = "mock_refiner"

    def is_available(self) -> bool:
        return True

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
        refined = ImageEnhance.Contrast(base.filter(ImageFilter.UnsharpMask(radius=0.7))).enhance(1.03)
        out = base.copy()
        if mask:
            out.paste(refined, mask=mask.convert("L"))
        else:
            out = refined
        elapsed = time.perf_counter() - start
        return RefineResult(out, {"engine": self.name, "runtime_seconds": elapsed, "mock": True, "seed": seed})
