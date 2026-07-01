from __future__ import annotations

import importlib.util

from app.core.config import EngineConfig
from app.engines.flux_refiner_engine import FluxRefinerEngine


def test_flux_refiner_disabled_backend_is_unavailable():
    engine = FluxRefinerEngine(EngineConfig(enabled=True, backend="disabled"))
    assert engine.status() == "unavailable: disabled"
    assert engine.is_available() is False


def test_flux_refiner_local_backend_reports_missing_model(tmp_path, monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: object())
    engine = FluxRefinerEngine(
        EngineConfig(enabled=True, backend="flux2_dev", model_name=None, checkpoint_dir=tmp_path / "missing")
    )
    assert "missing model" in engine.status()


def test_flux_refiner_api_backend_reads_env_without_logging_secret(monkeypatch):
    engine = FluxRefinerEngine(
        EngineConfig(
            enabled=True,
            backend="flux2_api",
            api_url_env="TEST_FLUX_API_URL",
            api_key_env="TEST_FLUX_API_KEY",
        )
    )
    assert "TEST_FLUX_API_URL" in engine.status() or "missing dependency requests" in engine.status()

    monkeypatch.setenv("TEST_FLUX_API_URL", "https://example.invalid/refine")
    monkeypatch.setenv("TEST_FLUX_API_KEY", "secret-value")
    status = engine.status()
    if importlib.util.find_spec("requests") is None:
        assert status == "unavailable: missing dependency requests"
    else:
        assert status == "available"
    assert "secret-value" not in status
