from __future__ import annotations

from app.core.config import Settings
from app.engines.adetailer_repair_engine import ADetailerRepairEngine
from app.engines.catvton_engine import CatVTonEngine
from app.engines.comfyui_flux_redux_engine import ComfyUIFluxReduxEngine
from app.engines.flux_refiner_engine import FluxRefinerEngine
from app.engines.idm_klein_hybrid_engine import IDMKleinHybridEngine
from app.engines.idm_vton_engine import IDMVTonEngine, active_idm_resident_model
from app.engines.klein_tryon_lora_engine import KleinTryOnLoraEngine, active_klein_resident_model
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
    if engine_name == "idm_klein_hybrid":
        return IDMKleinHybridEngine(settings)
    if engine_name == "comfyui_flux_redux":
        return ComfyUIFluxReduxEngine(settings)
    raise ValueError(f"Unknown try-on engine: {engine_name}")


def create_refiner(settings: Settings):
    if settings.pipeline.engine == "mock":
        return MockRefinerEngine()
    return FluxRefinerEngine(settings.flux_refiner)


def create_repair_engine(settings: Settings):
    return ADetailerRepairEngine(settings.repair)


def _selectable_engine_status_settings(settings: Settings) -> Settings:
    status_settings = settings.model_copy(deep=True)
    status_settings.idm_vton.enabled = True
    status_settings.klein_tryon_lora.enabled = True
    return status_settings


def _klein_bnb_4bit_status_settings(settings: Settings) -> Settings:
    status_settings = _selectable_engine_status_settings(settings)
    status_settings.klein_tryon_lora.device_map = "cuda"
    status_settings.klein_tryon_lora.quantization = "bnb_4bit"
    status_settings.klein_tryon_lora.quantize_components = ["transformer", "text_encoder"]
    status_settings.klein_tryon_lora.tensorrt_profile = "none"
    status_settings.klein_tryon_lora.tensorrt_components = []
    return status_settings


def _hybrid_pro_status_settings(settings: Settings) -> Settings:
    status_settings = _klein_bnb_4bit_status_settings(settings)
    status_settings.idm_vton.resident_worker = True
    status_settings.idm_vton.resident_worker_optimization = "torch_compile"
    status_settings.idm_vton.resident_worker_fallback = True
    return status_settings


def _preset_status(base_status: str, preset: str) -> str:
    if base_status.startswith("available"):
        return f"{base_status}; preset={preset}"
    return base_status


def _hybrid_status(idm_status: str, klein_status: str) -> str:
    if idm_status.startswith("available") and klein_status.startswith("available"):
        return f"available; idm=({idm_status}); klein=({klein_status})"
    return f"unavailable: idm=({idm_status}); klein=({klein_status})"


def model_statuses(settings: Settings) -> dict[str, str]:
    status_settings = _selectable_engine_status_settings(settings)
    hybrid_pro_status_settings = _hybrid_pro_status_settings(settings)
    idm_status = IDMVTonEngine(status_settings.idm_vton).status()
    idm_compile_status = IDMVTonEngine(hybrid_pro_status_settings.idm_vton).status()
    klein_status = KleinTryOnLoraEngine(status_settings.klein_tryon_lora).status()
    klein_bnb_status = _preset_status(klein_status, "device_map=cuda; quantization=bnb_4bit")
    engines = {
        "flux_refiner": FluxRefinerEngine(settings.flux_refiner),
        "catvton": CatVTonEngine(settings.catvton),
        "repair": ADetailerRepairEngine(settings.repair),
    }
    statuses = {
        "idm_vton": idm_status,
        "klein_tryon_lora": klein_status,
        "klein_bnb_4bit": klein_bnb_status,
        "idm_klein_hybrid": _hybrid_status(idm_status, klein_status),
        "idm_klein_hybrid_pro": _hybrid_status(idm_compile_status, klein_bnb_status),
    }
    statuses.update(
        {
            name: engine.status() if hasattr(engine, "status") else ("available" if engine.is_available() else "missing")
            for name, engine in engines.items()
        }
    )
    return statuses


def loaded_engine_state(settings: Settings) -> dict:
    klein_state = active_klein_resident_model()
    if klein_state is not None:
        return {
            "active_engine": klein_state["engine"],
            "active_engine_mode": klein_state["engine_mode"],
            "loaded_engine": klein_state["engine"],
            "loaded_engine_mode": klein_state["engine_mode"],
            "loaded_model": klein_state,
            "default_engine_mode": "klein_bnb_4bit",
        }

    idm_state = active_idm_resident_model()
    if idm_state is not None:
        return {
            "active_engine": idm_state["engine"],
            "active_engine_mode": idm_state["engine_mode"],
            "loaded_engine": idm_state["engine"],
            "loaded_engine_mode": idm_state["engine_mode"],
            "loaded_model": idm_state,
            "default_engine_mode": "klein_bnb_4bit",
        }

    return {
        "active_engine": None,
        "active_engine_mode": None,
        "loaded_engine": None,
        "loaded_engine_mode": None,
        "loaded_model": None,
        "default_engine_mode": "klein_bnb_4bit",
    }
