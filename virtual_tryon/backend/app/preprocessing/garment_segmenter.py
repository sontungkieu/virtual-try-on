from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageChops

from app.preprocessing.mask_utils import bbox_from_mask, blur, threshold


@dataclass(frozen=True)
class GarmentSegmentationResult:
    garment_image: Image.Image
    cloth_mask: Image.Image
    normalized_crop: Image.Image


class GarmentSegmenter:
    def segment(self, garment_image: Image.Image, target_size: tuple[int, int]) -> GarmentSegmentationResult:
        image = garment_image.convert("RGBA")
        if "A" in image.getbands():
            alpha = image.getchannel("A")
            mask = threshold(alpha, 8)
        else:
            rgb = image.convert("RGB")
            bg = Image.new("RGB", rgb.size, rgb.getpixel((0, 0)))
            diff = ImageChops.difference(rgb, bg).convert("L")
            mask = threshold(diff, 12)
            if np.array(mask).sum() == 0:
                mask = Image.new("L", rgb.size, 255)

        box = bbox_from_mask(mask)
        crop = image.convert("RGB").crop(box) if box else image.convert("RGB")
        crop.thumbnail(target_size, Image.Resampling.LANCZOS)
        normalized = Image.new("RGB", target_size, (255, 255, 255))
        x = (target_size[0] - crop.width) // 2
        y = (target_size[1] - crop.height) // 2
        normalized.paste(crop, (x, y))

        return GarmentSegmentationResult(
            garment_image=image.convert("RGB"),
            cloth_mask=blur(mask, 2),
            normalized_crop=normalized,
        )
