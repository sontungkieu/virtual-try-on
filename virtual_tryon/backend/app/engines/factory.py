from __future__ import annotations

from app.core.config import Settings
from app.engines.adetailer_repair_engine import ADetailerRepairEngine
from app.engines.catvton_engine import CatVTonEngine
from app.engines.flux_refiner_engine import FluxRefinerEngine
from app.engines.idm_vton_engine import IDMVTonEngine
from app.engines.klein_tryon_lora_engine import KleinTryOnLoraEngine
from app.engines.mock_engine import MockRefinerEngine, MockTryOnEngine


def create_tryon_engine(settings: Settings):
    engine_name = settings.pipeline.engine
    if engine_name == "mock":
        return MockTryOnEngine()
    if engine_name == "idm_vton":
        return IDMVTonEngine(settings.idm_vton)
    if engine_name == "catvton":
        return CatVTonEngine(settings.catvton)
    if engine_name == "klein_tryon_lora":
        return KleinTryOnLoraEngine(settings.klein_tryon_lora)
    raise ValueError(f"Unknown try-on engine: {engine_name}")


def create_refiner(settings: Settings):
    if settings.pipeline.engine == "mock":
        return MockRefinerEngine()
    return FluxRefinerEngine(settings.flux_refiner)


def create_repair_engine(settings: Settings):
    return ADetailerRepairEngine(settings.repair)


def model_statuses(settings: Settings) -> dict[str, str]:
    idm_engine = IDMVTonEngine(settings.idm_vton)
    engines = {
        "flux_refiner": FluxRefinerEngine(settings.flux_refiner),
        "catvton": CatVTonEngine(settings.catvton),
        "klein_tryon_lora": KleinTryOnLoraEngine(settings.klein_tryon_lora),
        "repair": ADetailerRepairEngine(settings.repair),
    }
    statuses = {"idm_vton": idm_engine.status()}
    statuses.update(
        {
            name: engine.status() if hasattr(engine, "status") else ("available" if engine.is_available() else "missing")
            for name, engine in engines.items()
        }
    )
    return statuses
