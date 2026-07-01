from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter


PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", "/workspace/Project_Phase2"))
EVAL_ROOT = PROJECT_ROOT / "virtual_tryon/data/temp/catvton_flux_eval_set"
OUT_ROOT = PROJECT_ROOT / "virtual_tryon/data/outputs/precise_target_masks_20260625"
TEMP_ROOT = PROJECT_ROOT / "virtual_tryon/data/temp/precise_target_masks"


def point(size: tuple[int, int], x: float, y: float) -> tuple[int, int]:
    width, height = size
    return int(round(width * x)), int(round(height * y))


def polygon(draw: ImageDraw.ImageDraw, size: tuple[int, int], pts: list[tuple[float, float]], fill: int = 255) -> None:
    draw.polygon([point(size, x, y) for x, y in pts], fill=fill)


def ellipse(draw: ImageDraw.ImageDraw, size: tuple[int, int], box: tuple[float, float, float, float], fill: int = 255) -> None:
    width, height = size
    xy = (
        int(round(width * box[0])),
        int(round(height * box[1])),
        int(round(width * box[2])),
        int(round(height * box[3])),
    )
    draw.ellipse(xy, fill=fill)


def rect(draw: ImageDraw.ImageDraw, size: tuple[int, int], box: tuple[float, float, float, float], fill: int = 255) -> None:
    width, height = size
    xy = (
        int(round(width * box[0])),
        int(round(height * box[1])),
        int(round(width * box[2])),
        int(round(height * box[3])),
    )
    draw.rounded_rectangle(xy, radius=max(4, min(width, height) // 32), fill=fill)


def mask_area(mask: Image.Image) -> float:
    arr = np.array(mask.convert("L"), dtype=np.uint8)
    return float((arr > 8).sum()) / float(arr.size)


def clean_mask(mask: Image.Image, *, blur: int = 0) -> Image.Image:
    out = mask.convert("L").point(lambda p: 255 if p > 8 else 0)
    if blur:
        out = out.filter(ImageFilter.GaussianBlur(blur))
    return out


def overlay(image: Image.Image, mask: Image.Image, color: tuple[int, int, int]) -> Image.Image:
    base = image.convert("RGBA")
    layer = Image.new("RGBA", base.size, (*color, 0))
    alpha = mask.convert("L").point(lambda p: int(p * 0.45))
    layer.putalpha(alpha)
    return Image.alpha_composite(base, layer).convert("RGB")


def union_masks(masks: list[Image.Image]) -> Image.Image:
    out = Image.new("L", masks[0].size, 0)
    for mask in masks:
        out = ImageChops.lighter(out, mask.convert("L"))
    return out


def create_sample015_masks(person: Image.Image) -> dict[str, Image.Image]:
    size = person.size

    dress = Image.new("L", size, 0)
    d = ImageDraw.Draw(dress)
    # Narrow straps and bodice. Keep face/hair and most arms out of the mask.
    polygon(d, size, [(0.405, 0.205), (0.455, 0.205), (0.470, 0.455), (0.415, 0.455)])
    polygon(d, size, [(0.545, 0.205), (0.595, 0.205), (0.585, 0.455), (0.530, 0.455)])
    polygon(
        d,
        size,
        [
            (0.395, 0.225),
            (0.605, 0.225),
            (0.640, 0.500),
            (0.585, 0.610),
            (0.415, 0.610),
            (0.360, 0.500),
        ],
    )
    # Skirt volume: enough room for the reference dress flare, but not a full-body column.
    polygon(
        d,
        size,
        [
            (0.355, 0.505),
            (0.645, 0.505),
            (0.705, 0.745),
            (0.635, 0.815),
            (0.365, 0.815),
            (0.295, 0.745),
        ],
    )
    # Preserve bare arms/hands. The target dress is sleeveless, so the mask should
    # not ask the model to repaint skin on either side of the torso.
    polygon(
        d,
        size,
        [
            (0.280, 0.220),
            (0.370, 0.230),
            (0.365, 0.585),
            (0.315, 0.725),
            (0.250, 0.705),
            (0.235, 0.405),
        ],
        fill=0,
    )
    polygon(
        d,
        size,
        [
            (0.630, 0.230),
            (0.720, 0.220),
            (0.765, 0.405),
            (0.750, 0.705),
            (0.685, 0.725),
            (0.635, 0.585),
        ],
        fill=0,
    )
    dress = clean_mask(dress)

    shoes = Image.new("L", size, 0)
    s = ImageDraw.Draw(shoes)
    # Keep this tight around feet/ankles. Wider masks tend to redraw legs.
    ellipse(s, size, (0.385, 0.858, 0.505, 0.995))
    ellipse(s, size, (0.495, 0.858, 0.615, 0.995))
    rect(s, size, (0.390, 0.875, 0.610, 0.992))
    shoes = clean_mask(shoes)

    hat = Image.new("L", size, 0)
    h = ImageDraw.Draw(hat)
    # Hat target only: crown and brim above the eyes. Avoid face region.
    ellipse(h, size, (0.380, 0.000, 0.620, 0.135))
    rect(h, size, (0.330, 0.082, 0.670, 0.158))
    # Keep the eyes, nose, and mouth out of the mask. Hat generation can touch
    # hairline/forehead, but should not repaint the face.
    rect(h, size, (0.340, 0.155, 0.660, 0.310), fill=0)
    hat = clean_mask(hat)

    hat_strict = Image.new("L", size, 0)
    hs = ImageDraw.Draw(hat_strict)
    # Stricter hat mask for face-preserving passes: it gives the sampler just
    # enough space above the head for a bucket hat and avoids eyes/nose/mouth.
    ellipse(hs, size, (0.400, 0.000, 0.600, 0.118))
    rect(hs, size, (0.360, 0.080, 0.640, 0.142))
    rect(hs, size, (0.350, 0.142, 0.650, 0.330), fill=0)
    hat_strict = clean_mask(hat_strict)

    union = union_masks([dress, shoes, hat_strict])
    return {"dress": dress, "shoes": shoes, "hat": hat, "hat_strict": hat_strict, "union": union}


def create_sample001_masks(person: Image.Image) -> dict[str, Image.Image]:
    size = person.size
    upper = Image.new("L", size, 0)
    u = ImageDraw.Draw(upper)
    # Full upper garment for a short-sleeve top. This covers the old blouse
    # down to the hem while keeping face, hair, hands, and pants untouched.
    polygon(
        u,
        size,
        [
            (0.345, 0.245),
            (0.655, 0.245),
            (0.705, 0.615),
            (0.645, 0.750),
            (0.365, 0.750),
            (0.295, 0.615),
        ],
    )
    polygon(u, size, [(0.245, 0.285), (0.365, 0.250), (0.355, 0.455), (0.225, 0.420)])
    polygon(u, size, [(0.635, 0.250), (0.760, 0.285), (0.785, 0.420), (0.645, 0.455)])
    # Exclude head and a small V-neck/neck area. The model can paint collar
    # detail around the neckline without being asked to redraw the face.
    rect(u, size, (0.315, 0.000, 0.685, 0.232), fill=0)
    polygon(u, size, [(0.445, 0.230), (0.555, 0.230), (0.505, 0.325)], fill=0)
    # Preserve the visible hands alongside the body.
    ellipse(u, size, (0.225, 0.530, 0.320, 0.710), fill=0)
    ellipse(u, size, (0.680, 0.530, 0.775, 0.710), fill=0)
    upper = clean_mask(upper)
    return {"upper": upper, "union": upper}


def save_masks(sample_id: str, masks: dict[str, Image.Image], person: Image.Image) -> dict[str, dict[str, str | float]]:
    out_dir = OUT_ROOT / sample_id
    temp_dir = TEMP_ROOT / sample_id
    out_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    person.save(out_dir / "person.png")

    colors = {
        "upper": (36, 140, 255),
        "dress": (36, 140, 255),
        "shoes": (255, 154, 0),
        "hat": (180, 100, 255),
        "hat_strict": (220, 80, 255),
        "union": (24, 170, 110),
    }
    metadata: dict[str, dict[str, str | float]] = {}
    for name, mask in masks.items():
        mask_path = temp_dir / f"{sample_id}_{name}_mask.png"
        preview_mask_path = out_dir / f"{name}_mask.png"
        overlay_path = out_dir / f"{name}_overlay.png"
        mask.save(mask_path)
        mask.save(preview_mask_path)
        overlay(person, mask, colors.get(name, (36, 140, 255))).save(overlay_path)
        metadata[name] = {
            "mask_path": mask_path.as_posix(),
            "preview_mask_path": preview_mask_path.as_posix(),
            "overlay_path": overlay_path.as_posix(),
            "area_ratio": round(mask_area(mask), 4),
        }
    return metadata


def build_grid(sample_id: str, person: Image.Image, masks: dict[str, Image.Image]) -> Path:
    out_dir = OUT_ROOT / sample_id
    names = [name for name in ["upper", "dress", "shoes", "hat", "hat_strict", "union"] if name in masks]
    cell_w, cell_h = 240, 300
    grid = Image.new("RGB", (cell_w * (len(names) + 1), cell_h), "white")
    draw = ImageDraw.Draw(grid)
    thumbs = [("person", person)] + [(name, overlay(person, masks[name], (36, 140, 255))) for name in names]
    for i, (name, image) in enumerate(thumbs):
        x0 = i * cell_w
        draw.rectangle((x0, 0, x0 + cell_w, cell_h), outline=(220, 220, 220))
        draw.text((x0 + 8, 8), name, fill=(0, 0, 0))
        thumb = image.copy()
        thumb.thumbnail((cell_w - 20, cell_h - 34), Image.Resampling.LANCZOS)
        grid.paste(thumb, (x0 + (cell_w - thumb.width) // 2, 28))
    path = out_dir / "mask_grid.png"
    grid.save(path)
    return path


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)

    all_meta = {}
    for sample_id, creator in [
        ("sample_001", create_sample001_masks),
        ("sample_015", create_sample015_masks),
    ]:
        person_path = EVAL_ROOT / sample_id / "person.png"
        person = Image.open(person_path).convert("RGB")
        masks = creator(person)
        metadata = save_masks(sample_id, masks, person)
        grid_path = build_grid(sample_id, person, masks)
        all_meta[sample_id] = {
            "person_path": person_path.as_posix(),
            "grid_path": grid_path.as_posix(),
            "masks": metadata,
        }

    (OUT_ROOT / "metadata.json").write_text(json.dumps(all_meta, indent=2), encoding="utf-8")
    print(json.dumps(all_meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
