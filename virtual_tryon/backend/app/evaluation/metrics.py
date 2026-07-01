from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any


def to_jsonable_metrics(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value
