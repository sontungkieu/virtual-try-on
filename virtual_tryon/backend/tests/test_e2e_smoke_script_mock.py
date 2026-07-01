from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from e2e_smoke_test import run_smoke  # noqa: E402


class FakeResponse:
    def __init__(self, payload=None, content=b"ok") -> None:
        self._payload = payload or {}
        self.content = content

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


class FakeClient:
    def __init__(self) -> None:
        self.polls = 0

    def get(self, url: str):
        if url.endswith("/health"):
            return FakeResponse({"status": "ok"})
        if "/tryon/job123" in url:
            self.polls += 1
            return FakeResponse(
                {
                    "job_id": "job123",
                    "status": "completed",
                    "result_url": "/artifacts/job123/result.png",
                    "debug": {"quality_report_url": "/artifacts/job123/quality_report.json"},
                }
            )
        return FakeResponse(content=b"artifact")

    def post(self, url: str, data, files):
        return FakeResponse({"job_id": "job123", "status": "queued"})


def _sample(tmp_path: Path) -> Path:
    sample = tmp_path / "sample_001"
    sample.mkdir()
    Image.new("RGB", (64, 96), (180, 180, 180)).save(sample / "person.jpg")
    Image.new("RGB", (64, 96), (20, 80, 210)).save(sample / "garment_top.jpg")
    (sample / "metadata.json").write_text(
        json.dumps({"sample_id": "sample_001", "category": "upper_body"}),
        encoding="utf-8",
    )
    return sample


def test_e2e_smoke_script_mock_flow(tmp_path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    report = run_smoke(
        api_base="http://testserver",
        sample_dir=_sample(tmp_path),
        use_refiner=False,
        timeout=30,
        client=FakeClient(),
    )

    assert report["status"] == "passed"
    assert report["job_id"] == "job123"
