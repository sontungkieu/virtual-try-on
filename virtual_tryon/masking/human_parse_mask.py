from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_LABELS = {
    # Common CIHP/LIP-style labels. Projects can override this map in config.
    "hat": [1],
    "hair": [2],
    "upper": [5, 6, 7],
    "dress": [5, 6, 7, 9, 12, 13],
    "lower": [9, 12, 13],
    "shoes": [18, 19],
    "accessory": [1, 16, 17],
}


@dataclass(frozen=True)
class HumanParseMaskConfig:
    semantic_map_path: Path | None = None
    label_map: dict[str, list[int]] = field(default_factory=lambda: dict(DEFAULT_LABELS))


class HumanParseMasker:
    """Build target masks from an existing human-parsing semantic label map.

    This class is intentionally an adapter, not a fake parser. If a real parser has
    not produced a semantic map, it returns None so callers can fail clearly or use
    an explicit debug fallback.
    """

    def __init__(self, config: HumanParseMaskConfig | None = None) -> None:
        self.config = config or HumanParseMaskConfig()

    def is_available(self) -> bool:
        return bool(self.config.semantic_map_path and self.config.semantic_map_path.exists())

    def create_mask(
        self,
        person_image: Image.Image,
        target_region: str,
        semantic_map_path: str | Path | None = None,
    ) -> Image.Image | None:
        path = Path(semantic_map_path) if semantic_map_path else self.config.semantic_map_path
        if path is None or not path.exists():
            return None
        labels = self.config.label_map.get(target_region)
        if not labels:
            raise ValueError(f"No human-parse labels configured for target_region='{target_region}'")

        semantic = Image.open(path)
        if semantic.mode not in {"L", "I", "I;16"}:
            semantic = semantic.convert("L")
        if semantic.size != person_image.size:
            semantic = semantic.resize(person_image.size, Image.Resampling.NEAREST)
        arr = np.array(semantic)
        selected = np.isin(arr, np.array(labels, dtype=arr.dtype))
        mask = Image.fromarray((selected.astype(np.uint8) * 255), mode="L")
        if mask.getbbox() is None:
            raise ValueError(f"Human parse produced an empty mask for target_region='{target_region}' from {path}")
        return mask
