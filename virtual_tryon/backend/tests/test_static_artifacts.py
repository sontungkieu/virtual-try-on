from __future__ import annotations

from fastapi.testclient import TestClient

from app.core.config import get_settings


def test_artifact_route_serves_output_file(client, tmp_path):
    settings = get_settings()
    settings.storage.outputs_dir = tmp_path / "outputs"
    settings.storage.public_outputs_prefix = "/artifacts"
    artifact = settings.storage.outputs_dir / "job123" / "result.png"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"fake-image")

    api = TestClient(client)
    response = api.get("/artifacts/job123/result.png")

    assert response.status_code == 200
    assert response.content == b"fake-image"


def test_artifact_route_blocks_path_traversal(client, tmp_path):
    settings = get_settings()
    settings.storage.outputs_dir = tmp_path / "outputs"
    settings.storage.outputs_dir.mkdir(parents=True)
    secret = tmp_path / "secret.txt"
    secret.write_text("secret", encoding="utf-8")

    api = TestClient(client)
    response = api.get("/artifacts/%2e%2e/secret.txt")

    assert response.status_code == 404


def test_artifact_route_missing_file_returns_404(client, tmp_path):
    settings = get_settings()
    settings.storage.outputs_dir = tmp_path / "outputs"
    settings.storage.outputs_dir.mkdir(parents=True)

    api = TestClient(client)
    response = api.get("/artifacts/nope/result.png")

    assert response.status_code == 404
