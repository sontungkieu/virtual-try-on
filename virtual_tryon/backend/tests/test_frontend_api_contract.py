from __future__ import annotations

from fastapi.testclient import TestClient


def test_tryon_response_frontend_contract(client, png_file):
    api = TestClient(client)
    response = api.post(
        "/tryon",
        data={"category": "upper_body", "use_refiner": "false", "repair_mode": "false", "run_mode": "sync"},
        files={
            "person_image": png_file("person.png", (170, 170, 170)),
            "garment_top": png_file("top.png", (20, 80, 210)),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["result_url"].startswith("/artifacts/")
    assert payload["debug"]["core_output_url"].startswith("/artifacts/")
    assert payload["debug"]["quality_report_url"].startswith("/artifacts/")
    assert payload["debug"]["mask_urls"]
