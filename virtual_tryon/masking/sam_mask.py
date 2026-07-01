from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class SAMMaskConfig:
    checkpoint_path: Path | None = None
    model_type: str = "vit_h"
    device: str = "cuda"


class SAMMasker:
    """Optional Segment Anything adapter.

    The dependency and checkpoint are intentionally optional. Production runs can
    enable this by installing segment-anything and passing a checkpoint. When it is
    unavailable, callers should fail clearly or use manual masks.
    """

    def __init__(self, config: SAMMaskConfig | None = None) -> None:
        self.config = config or SAMMaskConfig()
        self._predictor = None
        self._unavailable_reason: str | None = None

    @property
    def unavailable_reason(self) -> str | None:
        return self._unavailable_reason

    def is_available(self) -> bool:
        if self._predictor is not None:
            return True
        if not self.config.checkpoint_path or not self.config.checkpoint_path.exists():
            self._unavailable_reason = "SAM checkpoint is not configured or does not exist."
            return False
        try:
            from segment_anything import SamPredictor, sam_model_registry  # type: ignore
        except Exception as exc:  # noqa: BLE001 - dependency adapter
            self._unavailable_reason = f"segment_anything is not installed: {exc}"
            return False
        try:
            sam = sam_model_registry[self.config.model_type](checkpoint=str(self.config.checkpoint_path))
            sam.to(device=self.config.device)
            self._predictor = SamPredictor(sam)
            return True
        except Exception as exc:  # noqa: BLE001 - checkpoint/model adapter
            self._unavailable_reason = f"SAM initialization failed: {exc}"
            return False

    def create_mask_from_box(
        self,
        person_image: Image.Image,
        box_xyxy: tuple[int, int, int, int],
    ) -> Image.Image | None:
        if not self.is_available() or self._predictor is None:
            return None
        image = np.array(person_image.convert("RGB"))
        self._predictor.set_image(image)
        masks, scores, _ = self._predictor.predict(
            box=np.array(box_xyxy, dtype=np.float32),
            multimask_output=True,
        )
        best = int(np.argmax(scores))
        mask = (masks[best].astype(np.uint8) * 255)
        out = Image.fromarray(mask, mode="L")
        if out.getbbox() is None:
            raise ValueError(f"SAM produced an empty mask for box={box_xyxy}")
        return out
