from __future__ import annotations

from app.core.config import load_settings


def test_models_yaml_loads():
    settings = load_settings()
    assert settings.idm_vton.default_width == 768
    assert settings.flux_refiner.steps == 20
    assert settings.klein_tryon_lora.guidance_scale == 2.5


def test_default_values_valid():
    settings = load_settings()
    assert settings.image.output_width > 0
    assert settings.image.output_height > 0
    assert settings.preprocessing.dilation_px >= 0
    assert settings.storage.outputs_dir.name == "outputs"
    assert settings.mask_experiments.upper_body_expand_hem.enabled is False


def test_config_model_paths():
    settings = load_settings()
    assert settings.idm_vton.checkpoint_dir.name == "ckpt"
    assert settings.idm_vton.entrypoint is not None
    assert settings.idm_vton.entrypoint.name == "inference.py"
    assert settings.idm_vton.repo_path is not None
    assert settings.idm_vton.repo_path.name == "IDM-VTON"
