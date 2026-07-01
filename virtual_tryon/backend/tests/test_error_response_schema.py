from __future__ import annotations

from fastapi.testclient import TestClient


def _assert_error_schema(response) -> None:
    payload = response.json()
    assert set(payload) == {"error"}
    assert set(payload["error"]) == {"code", "message", "details"}
    assert isinstance(payload["error"]["code"], str)
    assert isinstance(payload["error"]["message"], str)
    assert isinstance(payload["error"]["details"], dict)


def test_validation_errors_use_unified_schema(client, png_file):
    response = TestClient(client).post(
        "/tryon",
        data={"category": "upper_body"},
        files={"garment_top": png_file("top.png")},
    )
    assert response.status_code == 422
    _assert_error_schema(response)


def test_not_found_errors_use_unified_schema(client):
    response = TestClient(client).get("/tryon/missing-job")
    assert response.status_code == 404
    _assert_error_schema(response)
    assert response.json()["error"]["code"] == "JOB_NOT_FOUND"
