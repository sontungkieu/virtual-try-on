from __future__ import annotations

from io import BytesIO

from fastapi.testclient import TestClient

from app.core.config import get_settings


def _post(api: TestClient, person_file):
    return api.post(
        "/tryon",
        data={"category": "upper_body", "use_refiner": "false"},
        files={
            "person_image": person_file,
            "garment_top": ("top.png", BytesIO(b"not-read"), "image/png"),
        },
    )


def test_rejects_non_image_mime(client):
    api = TestClient(client)
    response = _post(api, ("person.txt", BytesIO(b"plain text"), "text/plain"))
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "INVALID_IMAGE"


def test_rejects_oversized_image(client, png_file):
    get_settings().api.max_upload_mb = 0
    api = TestClient(client)
    response = api.post(
        "/tryon",
        data={"category": "upper_body"},
        files={
            "person_image": png_file("person.png"),
            "garment_top": png_file("top.png"),
        },
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "FILE_TOO_LARGE"


def test_rejects_corrupt_image(client, png_file):
    api = TestClient(client)
    response = api.post(
        "/tryon",
        data={"category": "upper_body"},
        files={
            "person_image": ("person.png", BytesIO(b"corrupt png"), "image/png"),
            "garment_top": png_file("top.png"),
        },
    )
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "INVALID_IMAGE"
