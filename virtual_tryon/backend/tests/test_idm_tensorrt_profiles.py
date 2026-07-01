from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from idm_vton_resident_worker import (  # noqa: E402
    ResidentIDMVTonPipeline,
    PROJECT_ROOT as WORKER_PROJECT_ROOT,
    _normalize_tensorrt_profile,
    _tensorrt_profile_defaults,
    resolve_tensorrt_cache_dir,
)


def test_tensorrt_full_profile_aliases_to_safe_blockwise_defaults(monkeypatch):
    monkeypatch.setenv("TRYON_TRT_PROFILE", "full")
    monkeypatch.delenv("TRYON_TRT_MODULES", raising=False)
    monkeypatch.delenv("TRYON_TRT_ALLOW_UNSAFE_UNET", raising=False)

    defaults = _tensorrt_profile_defaults()

    assert _normalize_tensorrt_profile("safe-full") == "full_safe"
    assert defaults["modules"] == "all"
    assert defaults["partition_preset"] == "safe_unet"
    assert defaults["min_block_size"] == 20
    assert defaults["allow_unsafe_unet"] is True
    assert ResidentIDMVTonPipeline._tensorrt_modules() == {
        "unet_blocks",
        "unet_encoder_blocks",
        "vae_decode",
    }


def test_tensorrt_explicit_modules_override_profile(monkeypatch):
    monkeypatch.setenv("TRYON_TRT_PROFILE", "full_safe")
    monkeypatch.setenv("TRYON_TRT_MODULES", "vae_decode")

    assert ResidentIDMVTonPipeline._tensorrt_modules() == {"vae_decode"}


def test_tensorrt_relative_cache_dir_resolves_under_project_root():
    resolved = resolve_tensorrt_cache_dir(Path("data/temp/trt_cache"))

    assert resolved == (WORKER_PROJECT_ROOT / "data/temp/trt_cache").resolve()
