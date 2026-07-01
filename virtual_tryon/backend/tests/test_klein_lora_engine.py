from __future__ import annotations

import csv
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest
from PIL import Image

from app.core.config import EngineConfig
from app.engines.base import TryOnInputs
from app.engines.klein_tryon_lora_engine import KleinTryOnLoraEngine
from app.utils.errors import ModelUnavailableError


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _config(**overrides) -> EngineConfig:
    payload = {
        "enabled": True,
        "backend": "fal_api",
        "base_model": "black-forest-labs/FLUX.2-klein-9B",
        "lora_repo": "fal/flux-klein-9b-virtual-tryon-lora",
        "lora_weight_api": "flux-klein-tryon.safetensors",
        "fal_endpoint": "fal-ai/flux-2/klein/9b/base/edit/lora",
        "num_inference_steps": 28,
        "guidance_scale": 2.5,
        "lora_scale": 1.0,
        "require_three_images": True,
        "bottom_strategy": "crop_from_person",
        "bottom_crop": {
            "y_start_ratio": 0.50,
            "y_end_ratio": 0.98,
            "x_margin_ratio": 0.08,
            "save_debug": True,
        },
    }
    payload.update(overrides)
    return EngineConfig(**payload)


def _inputs(output_dir: Path) -> TryOnInputs:
    person = Image.new("RGB", (256, 384), (180, 160, 140))
    top = Image.new("RGB", (256, 384), (20, 80, 210))
    return TryOnInputs(
        person_image=person,
        garment_image=top,
        category="upper_body",
        agnostic_mask=Image.new("L", person.size, 255),
        prompt=None,
        seed=42,
        output_dir=output_dir,
        extra={"garment_top_image": top},
    )


def _write_eval_sample(root: Path) -> Path:
    eval_set = root / "eval_set" / "sample_001"
    eval_set.mkdir(parents=True)
    Image.new("RGB", (128, 192), (180, 180, 180)).save(eval_set / "person.jpg")
    Image.new("RGB", (128, 192), (20, 80, 210)).save(eval_set / "garment_top.jpg")
    (eval_set / "metadata.json").write_text(
        json.dumps(
            {
                "sample_id": "sample_001",
                "category": "upper_body",
                "difficulty": "easy",
                "expected_focus": ["identity"],
            }
        ),
        encoding="utf-8",
    )
    return eval_set


def _run_ablation(tmp_path: Path, *extra_args: str) -> Path:
    eval_set = _write_eval_sample(tmp_path)
    output_dir = tmp_path / "ablation"
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_klein_lora_ablation.py"),
            "--sample",
            str(eval_set),
            "--output",
            str(output_dir),
            "--mock",
            *extra_args,
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return output_dir


def test_fal_runtime_check_missing_key(monkeypatch):
    monkeypatch.delenv("FAL_KEY", raising=False)
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "check_fal_runtime.py"),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    payload = json.loads(completed.stdout)
    assert payload["fal_key_set"] is False
    assert payload["klein_lora_available"] is False
    assert "FAL_KEY is not set" in payload["messages"]
    assert "hf_" not in completed.stdout


def test_klein_lora_availability_missing_token(monkeypatch):
    monkeypatch.delenv("FAL_KEY", raising=False)
    engine = KleinTryOnLoraEngine(_config())
    availability = engine.is_available()
    assert not availability
    assert availability.error_code in {"MISSING_FAL_KEY", "DEPENDENCY_MISSING"}
    assert "FAL_KEY" in availability.status


def test_klein_lora_disabled_does_not_run_local_worker_check(monkeypatch, tmp_path):
    def fail_check(self):
        raise AssertionError("disabled Klein should not run the local worker runtime check")

    monkeypatch.setattr(KleinTryOnLoraEngine, "_local_worker_check_error", fail_check)
    model_dir = tmp_path / "flux2-klein-9b"
    model_dir.mkdir()
    (model_dir / "model_index.json").write_text("{}", encoding="utf-8")
    lora_path = tmp_path / "flux-klein-tryon.safetensors"
    lora_path.write_bytes(b"fake")

    engine = KleinTryOnLoraEngine(
        _config(
            enabled=False,
            backend="diffusers_local",
            model_path=model_dir,
            lora_path=lora_path,
        )
    )

    availability = engine.is_available()
    assert not availability
    assert availability.error_code == "DISABLED"
    assert "enabled is false" in availability.status


def test_klein_lora_local_availability_uses_model_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(KleinTryOnLoraEngine, "_diffusers_local_import_error", staticmethod(lambda: None))
    model_dir = tmp_path / "flux2-klein-9b"
    model_dir.mkdir()
    (model_dir / "model_index.json").write_text("{}", encoding="utf-8")
    lora_path = tmp_path / "flux-klein-tryon.safetensors"
    lora_path.write_bytes(b"fake")

    engine = KleinTryOnLoraEngine(
        _config(
            backend="diffusers_local",
            model_path=model_dir,
            lora_path=lora_path,
        )
    )

    availability = engine.is_available()
    assert availability.available
    assert "FAL_KEY" not in availability.status


def test_klein_lora_local_run_uses_diffusers_pipeline(monkeypatch, tmp_path):
    model_dir = tmp_path / "flux2-klein-9b"
    model_dir.mkdir()
    (model_dir / "model_index.json").write_text("{}", encoding="utf-8")
    lora_path = tmp_path / "flux-klein-tryon.safetensors"
    lora_path.write_bytes(b"fake")
    captured: dict = {}
    released = {"idm": False}

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class FakeGenerator:
        def __init__(self, device: str):
            captured["generator_device"] = device

        def manual_seed(self, seed: int):
            captured["seed"] = seed
            return self

    fake_torch = types.SimpleNamespace(cuda=FakeCuda(), Generator=FakeGenerator)

    class FakePipe:
        def __call__(self, **kwargs):
            captured.update(kwargs)
            return types.SimpleNamespace(images=[Image.new("RGB", (80, 120), (10, 90, 40))])

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(KleinTryOnLoraEngine, "_diffusers_local_import_error", staticmethod(lambda: None))
    monkeypatch.setattr(KleinTryOnLoraEngine, "_load_diffusers_local_pipe", lambda self: FakePipe())
    monkeypatch.setattr(
        KleinTryOnLoraEngine,
        "_release_idm_resident_worker",
        lambda self: released.__setitem__("idm", True),
    )

    engine = KleinTryOnLoraEngine(
        _config(
            backend="diffusers_local",
            model_path=model_dir,
            lora_path=lora_path,
            default_width=128,
            default_height=192,
            num_inference_steps=4,
            device_map="cuda",
            quantization="torchao_int8",
            quantize_components=["transformer"],
            tensorrt_profile="vae_decode",
            tensorrt_components=["vae_decode"],
            tensorrt_engine_cache_dir=tmp_path / "trt-cache",
            tensorrt_min_block_size=7,
        )
    )
    result = engine.run(_inputs(tmp_path))

    assert result.image.size == (80, 120)
    assert released["idm"] is True
    assert captured["seed"] == 42
    assert captured["height"] == 192
    assert captured["width"] == 128
    assert captured["num_inference_steps"] == 4
    assert len(captured["image"]) == 3
    assert (tmp_path / "klein_lora_result.png").exists()
    local_payload = json.loads((tmp_path / "local_generation_sanitized.json").read_text(encoding="utf-8"))
    assert local_payload["backend"] == "diffusers_local"
    assert local_payload["local_files_only"] is True
    assert local_payload["lora_path"] == str(lora_path)
    assert local_payload["device_map"] == "cuda"
    assert local_payload["quantization"] == "torchao_int8"
    assert local_payload["quantize_components"] == ["transformer"]
    assert local_payload["tensorrt_profile"] == "vae_decode"
    assert local_payload["tensorrt_components"] == ["vae_decode"]
    assert local_payload["tensorrt_engine_cache_dir"] == str(tmp_path / "trt-cache")
    assert local_payload["tensorrt_min_block_size"] == 7


def test_klein_lora_cache_key_includes_placement_and_quantization(tmp_path):
    model_dir = tmp_path / "flux2-klein-9b"
    model_dir.mkdir()
    lora_path = tmp_path / "flux-klein-tryon.safetensors"
    lora_path.write_bytes(b"fake")

    base = KleinTryOnLoraEngine(
        _config(
            backend="diffusers_local",
            model_path=model_dir,
            lora_path=lora_path,
            device_map="cpu_offload",
            quantization="none",
        )
    )
    quantized = KleinTryOnLoraEngine(
        _config(
            backend="diffusers_local",
            model_path=model_dir,
            lora_path=lora_path,
            device_map="cuda",
            quantization="torchao_int8",
            quantize_components=["transformer"],
            tensorrt_profile="vae_decode",
            tensorrt_components=["vae_decode"],
        )
    )

    assert base._local_cache_key() != quantized._local_cache_key()


def test_klein_lora_cache_key_includes_tensorrt(tmp_path):
    model_dir = tmp_path / "flux2-klein-9b"
    model_dir.mkdir()
    lora_path = tmp_path / "flux-klein-tryon.safetensors"
    lora_path.write_bytes(b"fake")

    base = KleinTryOnLoraEngine(
        _config(
            backend="diffusers_local",
            model_path=model_dir,
            lora_path=lora_path,
            tensorrt_profile="none",
        )
    )
    trt = KleinTryOnLoraEngine(
        _config(
            backend="diffusers_local",
            model_path=model_dir,
            lora_path=lora_path,
            tensorrt_profile="vae_decode",
            tensorrt_components=["vae_decode"],
        )
    )

    assert base._local_cache_key() != trt._local_cache_key()


def test_klein_tensorrt_guard_rejects_quantized_transformer():
    from scripts.klein_diffusers_local_worker import (
        normalize_tensorrt_components,
        validate_tensorrt_request,
    )

    components = normalize_tensorrt_components("full_debug", [])
    with pytest.raises(RuntimeError, match="quantized transformer"):
        validate_tensorrt_request(
            "full_debug",
            components,
            quantization="bnb_4bit",
            quantize_components=["transformer", "text_encoder"],
        )


def test_klein_tensorrt_profile_aliases():
    from scripts.klein_diffusers_local_worker import (
        normalize_tensorrt_components,
        normalize_tensorrt_profile,
    )

    assert normalize_tensorrt_profile("stable") == "vae_decode"
    assert normalize_tensorrt_components("vae_decode", []) == ["vae_decode"]
    assert normalize_tensorrt_components("full_debug", None) == ["transformer", "vae_decode"]


def test_klein_lora_bottom_crop_from_person(monkeypatch, tmp_path):
    monkeypatch.delenv("FAL_KEY", raising=False)
    engine = KleinTryOnLoraEngine(_config())
    try:
        engine.run(_inputs(tmp_path))
    except ModelUnavailableError:
        pass
    crop_path = tmp_path / "auto_bottom_reference.png"
    assert crop_path.exists()
    crop = Image.open(crop_path)
    assert crop.width < 256
    assert crop.height > 100
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["bottom_source"] == "auto_cropped_from_person"


def test_klein_lora_sanitizes_request_response(monkeypatch, tmp_path):
    monkeypatch.delenv("FAL_KEY", raising=False)
    engine = KleinTryOnLoraEngine(_config(api_key_env="FAL_KEY"))
    try:
        engine.run(_inputs(tmp_path))
    except ModelUnavailableError:
        pass
    request_payload = json.loads((tmp_path / "request_sanitized.json").read_text(encoding="utf-8"))
    serialized = json.dumps(request_payload)
    assert "hf_" not in serialized
    assert "FAL_KEY" not in serialized
    assert request_payload["loras"][0]["path"].endswith("flux-klein-tryon.safetensors")


def test_fal_upload_file_called_for_local_inputs(monkeypatch, tmp_path):
    calls: list[str] = []
    captured: dict = {}

    def fake_upload_file(path: str) -> str:
        calls.append(Path(path).name)
        return f"https://fal.invalid/upload/{Path(path).name}?token=private"

    def fake_subscribe(endpoint: str, *, arguments: dict, with_logs: bool = False) -> dict:
        captured["endpoint"] = endpoint
        captured["arguments"] = arguments
        captured["with_logs"] = with_logs
        return {
            "images": [{"url": "https://fal.invalid/result.png?token=private"}],
            "api_key": "should-not-be-saved",
        }

    def fake_download_image(url: str, output_path: Path, timeout_seconds: int) -> Image.Image:
        image = Image.new("RGB", (64, 96), (10, 20, 30))
        image.save(output_path)
        return image

    fake_module = types.SimpleNamespace(upload_file=fake_upload_file, subscribe=fake_subscribe)
    monkeypatch.setitem(sys.modules, "fal_client", fake_module)
    monkeypatch.setenv("FAL_KEY", "test-fal-key")
    monkeypatch.setattr(KleinTryOnLoraEngine, "_download_image", staticmethod(fake_download_image))

    result = KleinTryOnLoraEngine(_config()).run(_inputs(tmp_path))

    assert result.image.size == (64, 96)
    assert calls == ["person_reference.png", "top_reference.png", "auto_bottom_reference.png"]
    assert captured["endpoint"] == "fal-ai/flux-2/klein/9b/base/edit/lora"
    assert len(captured["arguments"]["image_urls"]) == 3
    request_text = (tmp_path / "request_sanitized.json").read_text(encoding="utf-8")
    response_text = (tmp_path / "response_sanitized.json").read_text(encoding="utf-8")
    assert "private" not in request_text
    assert "private" not in response_text
    assert "should-not-be-saved" not in response_text


def test_klein_lora_prompt_variants_saved(tmp_path):
    output_dir = _run_ablation(tmp_path)
    default_prompt = output_dir / "sample_001" / "klein_lora_default" / "prompt.txt"
    strong_prompt = output_dir / "sample_001" / "klein_lora_strong_remove_old_shirt" / "prompt.txt"
    assert default_prompt.exists()
    assert strong_prompt.exists()
    assert default_prompt.read_text(encoding="utf-8").startswith("TRYON")
    assert strong_prompt.read_text(encoding="utf-8").startswith("TRYON")
    assert "The final image is a full body shot." in default_prompt.read_text(encoding="utf-8")
    assert "Keep only the blue velvet wrap top" in strong_prompt.read_text(encoding="utf-8")


def test_klein_lora_ablation_summary_schema(tmp_path):
    output_dir = _run_ablation(tmp_path)
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    rows = summary["rows"]
    by_variant = {row["variant"]: row for row in rows}
    assert {"idm_original", "klein_lora_default", "klein_lora_strong_remove_old_shirt"}.issubset(by_variant)
    for row in rows:
        for column in [
            "sample_id",
            "variant",
            "prompt_variant",
            "status",
            "runtime_seconds",
            "output_path",
            "prompt_path",
            "engine_status",
            "error_code",
            "notes",
        ]:
            assert column in row
    assert by_variant["klein_lora_default"]["prompt_path"] == "sample_001/klein_lora_default/prompt.txt"
    assert (output_dir / "comparison_grid.png").exists()
    assert (output_dir / "comparison_index.html").exists()


def test_klein_lora_manual_rating_template(tmp_path):
    output_dir = _run_ablation(tmp_path)
    template = output_dir / "manual_ratings_klein_lora.csv"
    assert template.exists()
    rows = list(csv.DictReader(template.open(encoding="utf-8")))
    variants = {row["variant"] for row in rows}
    assert {"klein_lora_default", "klein_lora_strong_remove_old_shirt"}.issubset(variants)
    assert all(row["identity_1_5"] == "" for row in rows)


def test_klein_lora_manual_ratings_no_subjective_autofill(tmp_path):
    output_dir = _run_ablation(tmp_path)
    rows = list(csv.DictReader((output_dir / "manual_ratings_klein_lora.csv").open(encoding="utf-8")))
    subjective_columns = [
        "identity_1_5",
        "garment_fidelity_1_5",
        "old_garment_removed_1_5",
        "realism_1_5",
        "pose_preservation_1_5",
        "body_shape_preservation_1_5",
        "background_preservation_1_5",
        "overedit_1_5",
        "winner",
        "notes",
    ]
    for row in rows:
        assert row["sample_id"] == "sample_001"
        assert "variant" in row
        for column in subjective_columns:
            assert row[column] == ""


def test_quality_report_no_subjective_notes(tmp_path):
    output_dir = _run_ablation(tmp_path)
    report_path = output_dir / "sample_001" / "idm_original" / "quality_report.json"
    assert report_path.exists()
    serialized = report_path.read_text(encoding="utf-8").lower()
    assert "pink shirt remains visible" not in serialized
    assert "old_garment_removed_1_5" not in serialized
    assert "identity_1_5" not in serialized
