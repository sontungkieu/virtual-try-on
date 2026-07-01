from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from PIL import Image

from app.schemas.tryon import TryOnCategory


@dataclass
class TryOnInputs:
    person_image: Image.Image
    garment_image: Image.Image
    category: TryOnCategory
    agnostic_mask: Image.Image
    agnostic_image: Image.Image | None = None
    densepose_image: Image.Image | None = None
    prompt: str | None = None
    seed: int | None = None
    output_dir: Path | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class TryOnResult:
    image: Image.Image
    metadata: dict = field(default_factory=dict)


@dataclass
class RefineResult:
    image: Image.Image
    metadata: dict = field(default_factory=dict)


class TryOnEngine(Protocol):
    name: str

    def is_available(self) -> bool:
        ...

    def prepare(self) -> None:
        ...

    def run(self, inputs: TryOnInputs) -> TryOnResult:
        ...


class RefinerEngine(Protocol):
    name: str

    def is_available(self) -> bool:
        ...

    def prepare(self) -> None:
        ...

    def refine(
        self,
        image: Image.Image,
        mask: Image.Image | None,
        prompt: str,
        references: dict | None = None,
        seed: int | None = None,
    ) -> RefineResult:
        ...
