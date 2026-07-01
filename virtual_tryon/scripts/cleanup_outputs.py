from __future__ import annotations

import argparse
import shutil
import time
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CleanupCandidate:
    path: Path
    reason: str


def collect_cleanup_candidates(outputs_dir: Path, older_than_hours: float | None, keep_latest: int) -> list[CleanupCandidate]:
    root = outputs_dir.resolve()
    if not root.exists():
        return []
    children = [path for path in root.iterdir() if path.is_dir()]
    children.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    keep = {path.resolve() for path in children[: max(0, keep_latest)]}
    cutoff = None if older_than_hours is None else time.time() - older_than_hours * 3600
    candidates: list[CleanupCandidate] = []
    for path in children:
        resolved = path.resolve()
        if resolved == root or root not in resolved.parents:
            continue
        if resolved in keep:
            continue
        if cutoff is not None and path.stat().st_mtime >= cutoff:
            continue
        candidates.append(CleanupCandidate(path=path, reason="older-than-threshold" if cutoff is not None else "not-kept"))
    return candidates


def cleanup_outputs(outputs_dir: Path, older_than_hours: float | None, keep_latest: int, dry_run: bool) -> list[CleanupCandidate]:
    candidates = collect_cleanup_candidates(outputs_dir, older_than_hours, keep_latest)
    if dry_run:
        return candidates
    root = outputs_dir.resolve()
    for candidate in candidates:
        resolved = candidate.path.resolve()
        if root not in resolved.parents:
            raise RuntimeError(f"Refusing to delete path outside outputs_dir: {candidate.path}")
        shutil.rmtree(resolved)
    return candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean old generated output folders safely.")
    parser.add_argument("--outputs-dir", default=str(PROJECT_ROOT / "data" / "outputs"))
    parser.add_argument("--older-than-hours", type=float, default=24.0)
    parser.add_argument("--keep-latest", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidates = cleanup_outputs(
        Path(args.outputs_dir),
        older_than_hours=args.older_than_hours,
        keep_latest=args.keep_latest,
        dry_run=args.dry_run,
    )
    action = "would_remove" if args.dry_run else "removed"
    for candidate in candidates:
        print(f"{action}: {candidate.path} ({candidate.reason})")
    print(f"{action}_count={len(candidates)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
