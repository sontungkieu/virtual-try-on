from __future__ import annotations

import json

import pytest

from app.services.artifact_service import (
    build_artifact_manifest,
    build_artifact_url,
    is_allowed_artifact,
    write_artifact_manifest,
)


def test_manifest_created_with_valid_urls(tmp_path):
    job_dir = tmp_path / "job123"
    job_dir.mkdir()
    (job_dir / "result.png").write_bytes(b"image")
    (job_dir / "quality_report.json").write_text("{}", encoding="utf-8")
    (job_dir / "model.onnx").write_bytes(b"private")

    path, manifest = write_artifact_manifest("job123", job_dir)

    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8")) == manifest
    assert {item["name"] for item in manifest["files"]} == {"quality_report.json", "result.png"}
    assert all(item["url"].startswith("/artifacts/job123/") for item in manifest["files"])


def test_forbidden_extensions_are_blocked(tmp_path):
    assert is_allowed_artifact("result.png")
    assert not is_allowed_artifact("checkpoint.pkl")
    assert not is_allowed_artifact("weights.safetensors")
    assert build_artifact_manifest("job", tmp_path)["files"] == []


def test_artifact_url_rejects_traversal():
    with pytest.raises(ValueError):
        build_artifact_url("job", "../secret.txt")
