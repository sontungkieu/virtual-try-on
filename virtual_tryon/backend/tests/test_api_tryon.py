from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient


def test_health_returns_ok(client):
    api = TestClient(client)
    response = api.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "idm_vton" in payload["models"]


def test_prepare_model_endpoint_uses_selected_engine(client):
    api = TestClient(client)
    response = api.post("/tryon/model/prepare", json={"engine_mode": "idm_vton"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["engine_mode"] == "idm_vton"
    assert payload["engine"] == "mock"
    assert payload["metadata"]["engine"] == "mock"


def test_prepare_model_rejects_invalid_engine_mode(client):
    api = TestClient(client)
    response = api.post("/tryon/model/prepare", json={"engine_mode": "not_real"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_REQUEST"


def test_tryon_reject_missing_person(client, png_file):
    api = TestClient(client)
    response = api.post(
        "/tryon",
        data={"category": "upper_body"},
        files={"garment_top": png_file("top.png")},
    )
    assert response.status_code in {400, 422}


def test_tryon_reject_no_garment(client, png_file):
    api = TestClient(client)
    response = api.post(
        "/tryon",
        data={"category": "upper_body"},
        files={"person_image": png_file("person.png")},
    )
    assert response.status_code == 400
    assert "garment" in response.json()["error"]["message"].lower()


def test_tryon_accepts_upper_body_request(client, png_file):
    api = TestClient(client)
    response = api.post(
        "/tryon",
        data={"category": "upper_body", "use_refiner": "true", "repair_mode": "true"},
        files={
            "person_image": png_file("person.png", (170, 170, 170)),
            "garment_top": png_file("top.png", (20, 80, 210)),
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["result_url"]
    stages = {stage["key"]: stage for stage in payload["stages"]}
    assert payload["current_stage"] == "completed"
    assert stages["generating"]["status"] == "completed"
    assert stages["generating"]["runtime_seconds"] is not None
    assert payload["debug"]["mask_url"]


@pytest.mark.parametrize(
    ("category", "garment_field"),
    [
        ("men_underwear", "garment_bottom"),
        ("women_underwear", "garment_bottom"),
        ("women_bra", "garment_top"),
    ],
)
def test_tryon_accepts_adult_innerwear_categories(client, png_file, category, garment_field):
    api = TestClient(client)
    response = api.post(
        "/tryon",
        data={"category": category, "use_refiner": "false", "repair_mode": "false"},
        files={
            "person_image": png_file("person.png", (170, 170, 170)),
            garment_field: png_file("innerwear.png", (220, 80, 120)),
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["result_url"]
    stages = {stage["key"]: stage for stage in payload["stages"]}
    assert stages["refining"]["status"] == "skipped"


def test_history_gender_filter_applies_before_limit(client, png_file):
    api = TestClient(client)
    women_response = api.post(
        "/tryon",
        data={"category": "women_underwear", "use_refiner": "false", "repair_mode": "false"},
        files={
            "person_image": png_file("woman.png", (170, 170, 170)),
            "garment_bottom": png_file("brief.png", (220, 80, 120)),
        },
    )
    assert women_response.status_code == 200
    man_response = api.post(
        "/tryon",
        data={"category": "men_underwear", "use_refiner": "false", "repair_mode": "false"},
        files={
            "person_image": png_file("man.png", (170, 170, 170)),
            "garment_bottom": png_file("brief.png", (80, 80, 220)),
        },
    )
    assert man_response.status_code == 200

    history = api.get("/tryon/history?limit=1&gender=woman")
    assert history.status_code == 200
    items = history.json()["items"]
    assert len(items) == 1
    assert items[0]["config"]["category"] == "women_underwear"

    success_history = api.get("/tryon/history?limit=10&gender=man&success_only=true")
    assert success_history.status_code == 200
    assert success_history.json()["items"]
    assert all(item["status"] == "completed" for item in success_history.json()["items"])


def test_engine_mode_default_remains_idm(client, png_file):
    api = TestClient(client)
    response = api.post(
        "/tryon",
        data={"category": "upper_body", "use_refiner": "false", "repair_mode": "false"},
        files={
            "person_image": png_file("person.png", (170, 170, 170)),
            "garment_top": png_file("top.png", (20, 80, 210)),
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["result_url"]


def test_tryon_accepts_generation_overrides(client, png_file):
    api = TestClient(client)
    response = api.post(
        "/tryon",
        data={
            "category": "upper_body",
            "use_refiner": "false",
            "repair_mode": "false",
            "output_width": "512",
            "output_height": "768",
            "steps": "12",
            "seed": "98765",
            "deterministic": "true",
        },
        files={
            "person_image": png_file("person.png", (170, 170, 170)),
            "garment_top": png_file("top.png", (20, 80, 210)),
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["seed"] == 98765
    assert payload["deterministic"] is True

    history = api.get("/tryon/history?limit=5")
    assert history.status_code == 200
    items = history.json()["items"]
    assert items
    newest = items[0]
    assert newest["job_id"]
    assert newest["result_url"]
    assert newest["inputs"]["person_url"]
    assert newest["config"]["output_width"] == 512
    assert newest["config"]["output_height"] == 768
    assert newest["config"]["steps"] == 12
    assert newest["config"]["seed"] == 98765
    assert newest["config"]["deterministic"] is True
    assert newest["finished_at"]
    assert newest["current_stage"] == "completed"
    assert {stage["key"] for stage in newest["stages"]} >= {
        "queued",
        "running",
        "loading_model",
        "generating",
        "refining",
        "completed",
    }


def test_tryon_rejects_invalid_generation_overrides(client, png_file):
    api = TestClient(client)
    response = api.post(
        "/tryon",
        data={
            "category": "upper_body",
            "use_refiner": "false",
            "repair_mode": "false",
            "output_width": "510",
            "output_height": "768",
            "steps": "2",
        },
        files={
            "person_image": png_file("person.png", (170, 170, 170)),
            "garment_top": png_file("top.png", (20, 80, 210)),
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_REQUEST"


def test_engine_mode_klein_lora_unavailable_clean_error(client, png_file, monkeypatch):
    monkeypatch.delenv("FAL_KEY", raising=False)
    api = TestClient(client)
    response = api.post(
        "/tryon",
        data={
            "category": "upper_body",
            "engine_mode": "klein_lora",
            "use_refiner": "false",
            "repair_mode": "false",
        },
        files={
            "person_image": png_file("person.png", (170, 170, 170)),
            "garment_top": png_file("top.png", (20, 80, 210)),
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "failed"
    assert payload["error_code"] == "ENGINE_UNAVAILABLE"
    assert "Klein Try-On LoRA" in payload["error"]
    assert "Traceback" not in payload["error"]


def test_tryon_api_error_shape(monkeypatch, png_file):
    monkeypatch.setenv("TRYON_ENGINE", "idm_vton")
    from app.core.config import clear_settings_cache
    from app.engines.idm_vton_engine import IDMVTonEngine
    from app.services.container import clear_container_cache
    from app.utils.errors import ModelUnavailableError
    import app.main as main_module

    def fail_run(self, inputs):
        raise ModelUnavailableError("IDM-VTON is not available. missing checkpoint: densepose/model_final_162be9.pkl")

    monkeypatch.setattr(IDMVTonEngine, "run", fail_run)
    clear_settings_cache()
    clear_container_cache()
    reloaded = importlib.reload(main_module)
    api = TestClient(reloaded.app)
    response = api.post(
        "/tryon",
        data={"category": "upper_body", "use_refiner": "false", "repair_mode": "false"},
        files={
            "person_image": png_file("person.png", (170, 170, 170)),
            "garment_top": png_file("top.png", (20, 80, 210)),
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "failed"
    assert "IDM-VTON" in payload["error"]
    assert "Traceback" not in payload["error"]

    monkeypatch.setenv("TRYON_ENGINE", "mock")
    clear_settings_cache()
    clear_container_cache()
    importlib.reload(main_module)
