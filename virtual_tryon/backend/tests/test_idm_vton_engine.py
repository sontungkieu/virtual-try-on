from __future__ import annotations

import subprocess
from pathlib import Path

from PIL import Image

from app.core.config import load_settings
from app.engines.base import TryOnInputs
from app.engines.idm_vton_engine import IDMVTonEngine, REQUIRED_CHECKPOINTS
from app.preprocessing.agnostic_mask import create_agnostic_mask


def _write_fake_file(path: Path, size: int = 2048) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)


def _settings_for_idm(tmp_path):
    settings = load_settings()
    settings.pipeline.engine = "idm_vton"
    settings.storage.inputs_dir = tmp_path / "inputs"
    settings.storage.outputs_dir = tmp_path / "outputs"
    settings.storage.temp_dir = tmp_path / "temp"
    settings.idm_vton.repo_path = tmp_path / "IDM-VTON"
    settings.idm_vton.repo_path.mkdir()
    settings.idm_vton.entrypoint = settings.idm_vton.repo_path / "inference.py"
    settings.idm_vton.entrypoint.write_text("print('fake')\n", encoding="utf-8")
    settings.idm_vton.checkpoint_dir = tmp_path / "models" / "idm_vton" / "ckpt"
    settings.idm_vton.resident_worker = False
    return settings


def _make_inputs(output_dir: Path) -> TryOnInputs:
    person = Image.new("RGB", (256, 384), (180, 180, 180))
    garment = Image.new("RGB", (256, 384), (20, 80, 220))
    mask_result = create_agnostic_mask(person, "upper_body", load_settings().preprocessing)
    return TryOnInputs(
        person_image=person,
        garment_image=garment,
        category="upper_body",
        agnostic_mask=mask_result.soft_mask,
        agnostic_image=mask_result.agnostic_image,
        prompt="blue shirt",
        seed=123,
        output_dir=output_dir,
    )


def test_idm_vton_availability_missing_checkpoint(tmp_path):
    settings = _settings_for_idm(tmp_path)
    engine = IDMVTonEngine(settings.idm_vton)
    status = engine.status()
    assert status.startswith("missing:")
    assert "densepose/model_final_162be9.pkl" in status
    assert "humanparsing/parsing_atr.onnx" in status
    assert "openpose/ckpts/body_pose_model.pth" in status


def test_idm_vton_command_building(tmp_path):
    settings = _settings_for_idm(tmp_path)
    engine = IDMVTonEngine(settings.idm_vton)
    context = engine.build_dataset(_make_inputs(tmp_path / "job"))
    command = context.command
    assert "accelerate.commands.launch" in command
    assert str(settings.idm_vton.entrypoint) in command
    assert "--data_dir" in command
    assert str(context.data_dir) in command
    assert "--output_dir" in command
    assert str(context.output_dir) in command
    assert (context.data_dir / "test" / "vitonhd_test_tagged.json").exists()
    assert (context.data_dir / "test_pairs.txt").read_text(encoding="utf-8").strip() == "person_0001.jpg garment_0001.jpg"


def test_idm_vton_resident_request_building(tmp_path):
    settings = _settings_for_idm(tmp_path)
    settings.idm_vton.default_width = 512
    settings.idm_vton.default_height = 768
    settings.idm_vton.steps = 12
    engine = IDMVTonEngine(settings.idm_vton)
    context = engine.build_dataset(_make_inputs(tmp_path / "job"))
    request = engine.build_resident_request(context, seed=321, deterministic=True)
    assert request["data_dir"] == str(context.data_dir)
    assert request["output_dir"] == str(context.output_dir)
    assert request["width"] == 512
    assert request["height"] == 768
    assert request["num_inference_steps"] == 12
    assert request["seed"] == 321
    assert request["deterministic"] is True


def test_idm_vton_innerwear_tags_preserve_model_compatible_item_class():
    men_tags = IDMVTonEngine._tags_for_category("men_underwear", None)
    women_tags = IDMVTonEngine._tags_for_category("women_underwear", None)
    bra_tags = IDMVTonEngine._tags_for_category("women_bra", None)

    assert men_tags[0]["tag_category"] == "pants"
    assert women_tags[0]["tag_category"] == "pants"
    assert bra_tags[0]["tag_category"] == "shirts"
    assert "brief underwear" in men_tags[3]["tag_category"]
    assert "brief underwear" in women_tags[3]["tag_category"]
    assert "bra" in bra_tags[3]["tag_category"]


def test_idm_vton_run_with_monkeypatched_subprocess(tmp_path, monkeypatch):
    settings = _settings_for_idm(tmp_path)
    for rel_path in REQUIRED_CHECKPOINTS:
        _write_fake_file(settings.idm_vton.checkpoint_dir / rel_path)

    def fake_run(command, cwd, capture_output, text, check):
        output_dir = Path(command[command.index("--output_dir") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (768, 1024), (30, 120, 220)).save(output_dir / "person_0001.jpg")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    engine = IDMVTonEngine(settings.idm_vton)
    monkeypatch.setattr(engine, "missing_requirements", lambda: [])
    response = engine.run(_make_inputs(tmp_path / "job"))
    assert response.image.size == (768, 1024)
    assert (tmp_path / "job" / "core_output.png").exists()
    assert (tmp_path / "job" / "idm_vton_command.txt").exists()


def test_idm_vton_run_with_resident_worker(tmp_path, monkeypatch):
    settings = _settings_for_idm(tmp_path)
    settings.idm_vton.resident_worker = True
    for rel_path in REQUIRED_CHECKPOINTS:
        _write_fake_file(settings.idm_vton.checkpoint_dir / rel_path)

    class FakeResidentClient:
        def run(self, request):
            output_dir = Path(request["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (768, 1024), (50, 130, 210)).save(output_dir / "person_0001.jpg")
            return {"ok": True, "runtime_seconds": 0.1}

        def stdout_tail(self):
            return "worker stdout"

        def stderr_tail(self):
            return "worker stderr"

    def fail_subprocess(*args, **kwargs):
        raise AssertionError("subprocess should not run when resident worker succeeds")

    monkeypatch.setattr(subprocess, "run", fail_subprocess)
    monkeypatch.setattr("app.engines.idm_vton_engine._get_resident_client", lambda config: FakeResidentClient())
    engine = IDMVTonEngine(settings.idm_vton)
    monkeypatch.setattr(engine, "missing_requirements", lambda: [])
    response = engine.run(_make_inputs(tmp_path / "job"))
    assert response.image.size == (768, 1024)
    assert response.metadata["runtime_backend"] == "resident_worker"
    assert (tmp_path / "job" / "idm_vton_worker_request.json").exists()
    assert "resident-idm-vton-worker" in (tmp_path / "job" / "idm_vton_command.txt").read_text(encoding="utf-8")


def test_idm_vton_resident_worker_falls_back_to_subprocess(tmp_path, monkeypatch):
    settings = _settings_for_idm(tmp_path)
    settings.idm_vton.resident_worker = True
    settings.idm_vton.resident_worker_fallback = True
    for rel_path in REQUIRED_CHECKPOINTS:
        _write_fake_file(settings.idm_vton.checkpoint_dir / rel_path)

    class FailingResidentClient:
        def run(self, request):
            raise RuntimeError("resident failed")

    def fake_run(command, cwd, capture_output, text, check):
        output_dir = Path(command[command.index("--output_dir") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (768, 1024), (30, 120, 220)).save(output_dir / "person_0001.jpg")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr("app.engines.idm_vton_engine._get_resident_client", lambda config: FailingResidentClient())
    engine = IDMVTonEngine(settings.idm_vton)
    monkeypatch.setattr(engine, "missing_requirements", lambda: [])
    response = engine.run(_make_inputs(tmp_path / "job"))
    assert response.metadata["runtime_backend"] == "subprocess_fallback"
    assert (tmp_path / "job" / "idm_vton_resident_error.txt").read_text(encoding="utf-8") == "resident failed"
