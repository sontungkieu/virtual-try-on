from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

from PIL import Image, ImageChops, ImageFilter

from app.core.config import Settings
from app.engines.base import TryOnInputs, TryOnResult
from app.engines.idm_vton_engine import IDMVTonEngine
from app.engines.klein_tryon_lora_engine import KleinTryOnLoraEngine
from app.preprocessing.mask_utils import composite_masked
from app.utils.image_io import save_image


def _constrain_mask(mask: Image.Image, constraint: Image.Image) -> Image.Image:
    mask_l = mask.convert("L")
    constraint_l = constraint.convert("L")
    if constraint_l.size != mask_l.size:
        constraint_l = constraint_l.resize(mask_l.size, Image.Resampling.LANCZOS)
    return ImageChops.multiply(mask_l, constraint_l)


def _idm_delta_mask(person: Image.Image, idm_image: Image.Image, constraint_mask: Image.Image) -> Image.Image:
    person_rgb = person.convert("RGB")
    idm_rgb = idm_image.convert("RGB")
    if idm_rgb.size != person_rgb.size:
        idm_rgb = idm_rgb.resize(person_rgb.size, Image.Resampling.LANCZOS)
    diff = ImageChops.difference(person_rgb, idm_rgb).convert("L")
    delta = diff.point(lambda value: 255 if value > 10 else 0)
    delta = delta.filter(ImageFilter.MaxFilter(7)).filter(ImageFilter.GaussianBlur(3))
    return _constrain_mask(delta, constraint_mask)


class IDMKleinHybridEngine:
    name = "idm_klein_hybrid"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.idm_engine = IDMVTonEngine(settings.idm_vton)
        self.klein_engine = KleinTryOnLoraEngine(settings.klein_tryon_lora)

    def status(self) -> str:
        idm_status = self.idm_engine.status()
        klein_status = self.klein_engine.status()
        if idm_status.startswith("available") and klein_status.startswith("available"):
            return f"available; idm=({idm_status}); klein=({klein_status})"
        return f"unavailable: idm=({idm_status}); klein=({klein_status})"

    def is_available(self) -> bool:
        return self.status().startswith("available")

    def prepare(self) -> None:
        self.idm_engine.prepare()

    def run(self, inputs: TryOnInputs) -> TryOnResult:
        started = time.perf_counter()
        output_dir = Path(inputs.output_dir or ".")
        output_dir.mkdir(parents=True, exist_ok=True)

        idm_dir = output_dir / "hybrid_idm_run"
        klein_dir = output_dir / "hybrid_klein_run"
        idm_inputs = replace(inputs, output_dir=idm_dir)
        klein_inputs = replace(inputs, output_dir=klein_dir)

        idm_result = self.idm_engine.run(idm_inputs)
        idm_base = composite_masked(inputs.person_image, idm_result.image, inputs.agnostic_mask)
        save_image(idm_result.image, output_dir / "hybrid_idm_raw.png")
        save_image(idm_base, output_dir / "hybrid_idm_base.png")

        klein_result = self.klein_engine.run(klein_inputs)
        klein_detail = composite_masked(inputs.person_image, klein_result.image, inputs.agnostic_mask)
        save_image(klein_result.image, output_dir / "hybrid_klein_raw.png")
        save_image(klein_detail, output_dir / "hybrid_klein_detail.png")

        delta_mask = _idm_delta_mask(inputs.person_image, idm_base, inputs.agnostic_mask)
        save_image(delta_mask, output_dir / "hybrid_idm_delta_mask.png")

        hybrid = Image.composite(klein_detail, idm_base, delta_mask)
        save_image(hybrid, output_dir / "hybrid_result.png")

        return TryOnResult(
            hybrid,
            {
                "engine": self.name,
                "runtime_seconds": round(time.perf_counter() - started, 3),
                "fusion": "idm_delta_masked_klein_detail",
                "idm_metadata": idm_result.metadata,
                "klein_metadata": klein_result.metadata,
                "artifacts": {
                    "idm_raw": str(output_dir / "hybrid_idm_raw.png"),
                    "idm_base": str(output_dir / "hybrid_idm_base.png"),
                    "klein_raw": str(output_dir / "hybrid_klein_raw.png"),
                    "klein_detail": str(output_dir / "hybrid_klein_detail.png"),
                    "delta_mask": str(output_dir / "hybrid_idm_delta_mask.png"),
                    "hybrid_result": str(output_dir / "hybrid_result.png"),
                },
            },
        )
