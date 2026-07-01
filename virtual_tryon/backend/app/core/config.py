from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from app.core.paths import CONFIG_DIR, PROJECT_ROOT, resolve_project_path


class StorageConfig(BaseModel):
    inputs_dir: Path
    outputs_dir: Path
    temp_dir: Path
    public_outputs_prefix: str = "/artifacts"


class ImageConfig(BaseModel):
    max_side: int = 1536
    output_width: int = 768
    output_height: int = 1024


class PreprocessingConfig(BaseModel):
    dilation_px: int = 18
    blur_radius: int = 8
    preserve_face: bool = True
    preserve_hands: bool = True
    preserve_hair: bool = True
    innerwear_dilation_px: int = 12
    innerwear_blur_radius: int = 5
    innerwear_use_silhouette_clip: bool = True
    innerwear_silhouette_clip_dilation_px: int = 8
    innerwear_reference_crop_enabled: bool = False
    mask_cache_enabled: bool = True


class MaskExperimentConfig(BaseModel):
    enabled: bool = False
    torso_down_extension_ratio: float = 0.12
    waist_extra_dilation_px: int = 24
    preserve_face: bool = True
    preserve_hair: bool = True
    preserve_hands: bool = True
    save_debug_overlays: bool = True


class MaskExperimentsConfig(BaseModel):
    upper_body_expand_hem: MaskExperimentConfig = Field(default_factory=MaskExperimentConfig)


class PipelineConfig(BaseModel):
    engine: str = "idm_vton"
    allow_mock_engine: bool = False
    save_intermediates: bool = True
    fail_on_missing_core_model: bool = True


class ApiConfig(BaseModel):
    run_mode: str = "sync"
    job_poll_interval_seconds: int = 2
    max_job_runtime_seconds: int = 900
    max_retries: int = 0
    max_concurrent_jobs: int = 1
    queue_policy: str = "queue"
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:5173", "http://127.0.0.1:5173"]
    )
    max_upload_mb: int = 20
    allowed_image_mime_types: list[str] = Field(
        default_factory=lambda: ["image/jpeg", "image/png", "image/webp"]
    )
    allow_public_artifacts: bool = True
    artifact_ttl_hours: int = 24


class ModelRuntimeConfig(BaseModel):
    device: str = "cuda"
    precision: str = "bf16"


class EngineConfig(BaseModel):
    enabled: bool = True
    backend: str = "disabled"
    fallback_to_core: bool = True
    fallback_on_error: bool = True
    repo_path: Path | None = None
    checkpoint_dir: Path | None = None
    entrypoint: Path | None = None
    model_name: str | None = None
    model_path: Path | None = None
    base_model: str | None = None
    lora_repo: str | None = None
    lora_weight_api: str | None = None
    lora_weight_comfy: str | None = None
    lora_path: Path | None = None
    fal_endpoint: str | None = None
    quantized: bool = False
    quantization: str = "none"
    quantize_components: list[str] = Field(default_factory=lambda: ["transformer", "text_encoder"])
    device_map: str = "cpu_offload"
    tensorrt_profile: str = "none"
    tensorrt_components: list[str] = Field(default_factory=list)
    tensorrt_engine_cache_dir: Path | None = None
    tensorrt_min_block_size: int | None = None
    remote_text_encoder: bool = False
    max_retries: int = 1
    api_url_env: str | None = None
    api_key_env: str | None = None
    default_width: int = 768
    default_height: int = 1024
    steps: int = 30
    num_inference_steps: int | None = None
    guidance_scale: float = 2.0
    default_strength: float = 0.35
    lora_scale: float = 1.0
    resolution: int = 1024
    require_three_images: bool = False
    bottom_strategy: str = "skip"
    bottom_crop: dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: int = 900
    resident_worker: bool = False
    resident_worker_fallback: bool = True
    resident_worker_entrypoint: Path | None = None
    resident_worker_startup_timeout_seconds: int = 900
    resident_worker_request_timeout_seconds: int = 900
    resident_worker_optimization: str = "eager"
    resident_worker_torch_compile_backend: str = "inductor"
    resident_worker_torch_compile_mode: str = "reduce-overhead"


class RepairConfig(BaseModel):
    enabled: bool = True
    mask_dilation_px: int = 16
    mask_blur_radius: int = 8


class QualityConfig(BaseModel):
    min_output_width: int = 256
    min_output_height: int = 256
    background_change_threshold: float = 0.18
    garment_change_threshold: float = 0.04
    artifact_threshold: float = 0.35


class RefinementConfig(BaseModel):
    enabled: bool = True
    mask_mode: str = "safe"
    boundary_dilation_px: int = 18
    boundary_erosion_px: int = 6
    soft_blur_radius: int = 8
    refine_only_masked_region: bool = True
    preserve_face: bool = True
    preserve_hands: bool = True
    preserve_hair: bool = True
    preserve_background: bool = True
    default_prompt: str = "Refine garment boundaries while preserving identity."


class AppConfig(BaseModel):
    name: str = "Virtual Try-On"
    environment: str = "development"
    debug: bool = True


class Settings(BaseModel):
    app: AppConfig
    storage: StorageConfig
    image: ImageConfig
    preprocessing: PreprocessingConfig
    pipeline: PipelineConfig
    api: ApiConfig
    runtime: ModelRuntimeConfig
    idm_vton: EngineConfig
    flux_refiner: EngineConfig
    catvton: EngineConfig
    klein_tryon_lora: EngineConfig
    repair: RepairConfig
    quality: QualityConfig
    refinement: RefinementConfig
    mask_experiments: MaskExperimentsConfig = Field(default_factory=MaskExperimentsConfig)
    repair_regions: list[str] = Field(default_factory=list)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_paths(config: dict[str, Any]) -> dict[str, Any]:
    for key in ["inputs_dir", "outputs_dir", "temp_dir"]:
        if key in config.get("storage", {}):
            config["storage"][key] = resolve_project_path(config["storage"][key])

    for section in ["idm_vton", "flux_refiner", "catvton", "klein_tryon_lora"]:
        section_config = config.get(section, {})
        for key in [
            "repo_path",
            "checkpoint_dir",
            "lora_path",
            "entrypoint",
            "model_path",
            "resident_worker_entrypoint",
            "tensorrt_engine_cache_dir",
        ]:
            if section_config.get(key):
                section_config[key] = resolve_project_path(section_config[key])
    return config


def _apply_env_overrides(config: dict[str, Any]) -> dict[str, Any]:
    engine = os.getenv("TRYON_ENGINE")
    if engine:
        config.setdefault("pipeline", {})["engine"] = engine

    allow_mock = os.getenv("TRYON_ALLOW_MOCK")
    if allow_mock is not None:
        config.setdefault("pipeline", {})["allow_mock_engine"] = allow_mock.lower() in {"1", "true", "yes", "on"}

    api_run_mode = os.getenv("TRYON_API_RUN_MODE")
    if api_run_mode:
        config.setdefault("api", {})["run_mode"] = api_run_mode

    resident_worker = os.getenv("TRYON_IDM_RESIDENT_WORKER")
    if resident_worker is not None:
        config.setdefault("idm_vton", {})["resident_worker"] = resident_worker.lower() in {"1", "true", "yes", "on"}

    resident_worker_optimization = os.getenv("TRYON_IDM_WORKER_OPTIMIZATION")
    if resident_worker_optimization:
        config.setdefault("idm_vton", {})["resident_worker_optimization"] = resident_worker_optimization

    klein_backend = os.getenv("TRYON_KLEIN_BACKEND")
    if klein_backend:
        config.setdefault("klein_tryon_lora", {})["backend"] = klein_backend

    klein_model_path = os.getenv("TRYON_KLEIN_MODEL_PATH")
    if klein_model_path:
        config.setdefault("klein_tryon_lora", {})["model_path"] = klein_model_path

    klein_lora_path = os.getenv("TRYON_KLEIN_LORA_PATH")
    if klein_lora_path:
        config.setdefault("klein_tryon_lora", {})["lora_path"] = klein_lora_path

    klein_entrypoint = os.getenv("TRYON_KLEIN_ENTRYPOINT")
    if klein_entrypoint:
        config.setdefault("klein_tryon_lora", {})["entrypoint"] = klein_entrypoint

    klein_device_map = os.getenv("TRYON_KLEIN_DEVICE_MAP")
    if klein_device_map:
        config.setdefault("klein_tryon_lora", {})["device_map"] = klein_device_map

    klein_quantization = os.getenv("TRYON_KLEIN_QUANTIZATION")
    if klein_quantization:
        config.setdefault("klein_tryon_lora", {})["quantization"] = klein_quantization

    klein_quantize_components = os.getenv("TRYON_KLEIN_QUANTIZE_COMPONENTS")
    if klein_quantize_components:
        config.setdefault("klein_tryon_lora", {})["quantize_components"] = [
            item.strip()
            for item in klein_quantize_components.split(",")
            if item.strip()
        ]

    klein_trt_profile = os.getenv("TRYON_KLEIN_TRT_PROFILE")
    if klein_trt_profile:
        config.setdefault("klein_tryon_lora", {})["tensorrt_profile"] = klein_trt_profile

    klein_trt_components = os.getenv("TRYON_KLEIN_TRT_COMPONENTS")
    if klein_trt_components:
        config.setdefault("klein_tryon_lora", {})["tensorrt_components"] = [
            item.strip()
            for item in klein_trt_components.split(",")
            if item.strip()
        ]

    klein_trt_engine_cache_dir = os.getenv("TRYON_KLEIN_TRT_ENGINE_CACHE_DIR")
    if klein_trt_engine_cache_dir:
        config.setdefault("klein_tryon_lora", {})["tensorrt_engine_cache_dir"] = klein_trt_engine_cache_dir

    klein_trt_min_block_size = os.getenv("TRYON_KLEIN_TRT_MIN_BLOCK_SIZE")
    if klein_trt_min_block_size:
        config.setdefault("klein_tryon_lora", {})["tensorrt_min_block_size"] = int(klein_trt_min_block_size)

    device = os.getenv("TRYON_DEVICE")
    if device:
        config["device"] = device
    return config


def load_settings() -> Settings:
    config: dict[str, Any] = {}
    for filename in ["default.yaml", "models.yaml", "pipeline.yaml"]:
        config = _deep_merge(config, _read_yaml(CONFIG_DIR / filename))

    config = _apply_env_overrides(config)
    config = _resolve_paths(config)

    return Settings(
        app=AppConfig(**config.get("app", {})),
        storage=StorageConfig(**config.get("storage", {})),
        image=ImageConfig(**config.get("image", {})),
        preprocessing=PreprocessingConfig(**config.get("preprocessing", {})),
        pipeline=PipelineConfig(**config.get("pipeline", {})),
        api=ApiConfig(**config.get("api", {})),
        runtime=ModelRuntimeConfig(device=config.get("device", "cuda"), precision=config.get("precision", "bf16")),
        idm_vton=EngineConfig(**config.get("idm_vton", {})),
        flux_refiner=EngineConfig(**config.get("flux_refiner", {})),
        catvton=EngineConfig(**config.get("catvton", {})),
        klein_tryon_lora=EngineConfig(**config.get("klein_tryon_lora", {})),
        repair=RepairConfig(**config.get("repair", {})),
        quality=QualityConfig(**config.get("quality", {})),
        refinement=RefinementConfig(**config.get("refinement", {})),
        mask_experiments=MaskExperimentsConfig(**config.get("mask_experiments", {})),
        repair_regions=config.get("repair_regions", []),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()


def clear_settings_cache() -> None:
    get_settings.cache_clear()
