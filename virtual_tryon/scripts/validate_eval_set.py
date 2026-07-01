from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
INNERWEAR_BOTTOM_CATEGORIES = {"men_underwear", "women_underwear"}
INNERWEAR_TOP_CATEGORIES = {"women_bra"}
VALID_CATEGORIES = {
    "upper_body",
    "lower_body",
    "dress",
    "full_outfit",
    *INNERWEAR_BOTTOM_CATEGORIES,
    *INNERWEAR_TOP_CATEGORIES,
}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}


@dataclass(frozen=True)
class EvalSample:
    sample_id: str
    root: Path
    person_path: Path
    category: str
    difficulty: str
    metadata: dict[str, Any]
    garment_top: Path | None = None
    garment_bottom: Path | None = None
    garment_dress: Path | None = None

    def primary_garment(self) -> Path | None:
        if self.category in {"upper_body", *INNERWEAR_TOP_CATEGORIES}:
            return self.garment_top
        if self.category in {"lower_body", *INNERWEAR_BOTTOM_CATEGORIES}:
            return self.garment_bottom
        if self.category == "dress":
            return self.garment_dress
        return self.garment_dress or self.garment_top or self.garment_bottom


def _find_image(sample_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTENSIONS:
        candidate = sample_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def _load_metadata(sample_dir: Path) -> tuple[dict[str, Any] | None, list[str]]:
    path = sample_dir / "metadata.json"
    if not path.exists():
        return None, ["metadata.json is missing"]
    try:
        return json.loads(path.read_text(encoding="utf-8")), []
    except json.JSONDecodeError as exc:
        return None, [f"metadata.json is invalid JSON: {exc}"]


def validate_sample(sample_dir: Path) -> tuple[EvalSample | None, list[str]]:
    errors: list[str] = []
    metadata, metadata_errors = _load_metadata(sample_dir)
    errors.extend(metadata_errors)
    if metadata is None:
        return None, errors

    sample_id = str(metadata.get("sample_id") or sample_dir.name)
    category = metadata.get("category")
    difficulty = metadata.get("difficulty")
    expected_focus = metadata.get("expected_focus")
    if sample_id != sample_dir.name:
        errors.append(f"sample_id '{sample_id}' does not match folder name '{sample_dir.name}'")
    if category not in VALID_CATEGORIES:
        errors.append(f"category must be one of {sorted(VALID_CATEGORIES)}")
    if difficulty not in VALID_DIFFICULTIES:
        errors.append(f"difficulty must be one of {sorted(VALID_DIFFICULTIES)}")
    if not isinstance(expected_focus, list) or not all(isinstance(item, str) for item in expected_focus):
        errors.append("expected_focus must be a list of strings")

    person_path = _find_image(sample_dir, "person")
    if person_path is None:
        errors.append("person image is missing")
    garment_top = _find_image(sample_dir, "garment_top")
    garment_bottom = _find_image(sample_dir, "garment_bottom")
    garment_dress = _find_image(sample_dir, "garment_dress")

    if category in {"upper_body", *INNERWEAR_TOP_CATEGORIES} and garment_top is None:
        errors.append(f"{category} sample requires garment_top image")
    if category in {"lower_body", *INNERWEAR_BOTTOM_CATEGORIES} and garment_bottom is None:
        errors.append(f"{category} sample requires garment_bottom image")
    if category == "dress" and garment_dress is None:
        errors.append("dress sample requires garment_dress image")
    if category == "full_outfit" and garment_dress is None and (garment_top is None or garment_bottom is None):
        errors.append("full_outfit sample requires garment_dress, or both garment_top and garment_bottom")

    if errors or person_path is None or category is None or difficulty is None:
        return None, errors
    return (
        EvalSample(
            sample_id=sample_id,
            root=sample_dir,
            person_path=person_path,
            garment_top=garment_top,
            garment_bottom=garment_bottom,
            garment_dress=garment_dress,
            category=category,
            difficulty=difficulty,
            metadata=metadata,
        ),
        [],
    )


def discover_eval_samples(eval_set: Path) -> tuple[list[EvalSample], list[dict[str, Any]]]:
    eval_set = Path(eval_set)
    if not eval_set.exists():
        return [], [{"sample_id": None, "errors": [f"eval set folder not found: {eval_set}"]}]
    sample_dirs = sorted(path for path in eval_set.iterdir() if path.is_dir())
    if not sample_dirs:
        return [], [{"sample_id": None, "errors": [f"eval set is empty: {eval_set}"]}]

    samples: list[EvalSample] = []
    issues: list[dict[str, Any]] = []
    for sample_dir in sample_dirs:
        sample, errors = validate_sample(sample_dir)
        if errors:
            issues.append({"sample_id": sample_dir.name, "errors": errors})
        if sample is not None:
            samples.append(sample)
    return samples, issues


def print_summary(samples: list[EvalSample], issues: list[dict[str, Any]]) -> None:
    if issues:
        for issue in issues:
            sample_id = issue["sample_id"] or "eval_set"
            for error in issue["errors"]:
                print(f"warning: {sample_id}: {error}")
    category_counts = Counter(sample.category for sample in samples)
    difficulty_counts = Counter(sample.difficulty for sample in samples)
    print(f"valid_samples={len(samples)}")
    print("by_category=" + json.dumps(dict(sorted(category_counts.items())), ensure_ascii=False))
    print("by_difficulty=" + json.dumps(dict(sorted(difficulty_counts.items())), ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a Virtual Try-On golden evaluation set.")
    parser.add_argument("--eval-set", default=str(PROJECT_ROOT / "data" / "eval_set"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    samples, issues = discover_eval_samples(Path(args.eval_set))
    print_summary(samples, issues)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
