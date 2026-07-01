from __future__ import annotations

import os
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from cleanup_outputs import cleanup_outputs  # noqa: E402


def _touch_dir(path: Path, age_hours: float) -> None:
    path.mkdir(parents=True)
    timestamp = time.time() - age_hours * 3600
    os.utime(path, (timestamp, timestamp))


def test_cleanup_outputs_dry_run_does_not_delete(tmp_path):
    outputs = tmp_path / "outputs"
    old_job = outputs / "old_job"
    _touch_dir(old_job, age_hours=48)

    candidates = cleanup_outputs(outputs, older_than_hours=24, keep_latest=0, dry_run=True)

    assert len(candidates) == 1
    assert old_job.exists()


def test_cleanup_outputs_keep_latest(tmp_path):
    outputs = tmp_path / "outputs"
    old_job = outputs / "old_job"
    newer_job = outputs / "newer_job"
    _touch_dir(old_job, age_hours=48)
    _touch_dir(newer_job, age_hours=47)

    candidates = cleanup_outputs(outputs, older_than_hours=None, keep_latest=1, dry_run=False)

    assert {candidate.path.name for candidate in candidates} == {"old_job"}
    assert not old_job.exists()
    assert newer_job.exists()
