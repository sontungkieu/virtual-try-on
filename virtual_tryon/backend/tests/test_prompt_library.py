from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from app.prompts.engine_prompt_templates import CATVTON_TEMPLATE
from app.prompts.prompt_builder import build_prompt
from app.prompts.prompt_types import EngineMode, PromptVariant
from app.prompts.testcase_prompt_library import get_testcase


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _request(testcase_id: str, engine: EngineMode, variant: PromptVariant = PromptVariant.DEFAULT):
    return get_testcase(testcase_id).build_request(engine, variant)


def _write_eval_sample(root: Path) -> Path:
    sample_dir = root / "sample_001"
    sample_dir.mkdir(parents=True)
    Image.new("RGB", (128, 192), (180, 180, 180)).save(sample_dir / "person.jpg")
    Image.new("RGB", (128, 192), (20, 80, 210)).save(sample_dir / "garment_top.jpg")
    (sample_dir / "metadata.json").write_text(
        json.dumps(
            {
                "sample_id": "sample_001",
                "testcase_id": "tc10",
                "category": "upper_body",
                "difficulty": "easy",
                "expected_focus": ["identity", "prompt"],
            }
        ),
        encoding="utf-8",
    )
    return root


def test_prompt_builder_idm_contains_person_garment_task():
    result = build_prompt(_request("tc10", EngineMode.IDM, PromptVariant.STRONG_REMOVE_OLD_GARMENT))
    prompt = result.positive_prompt
    assert "Person:" in prompt
    assert "Garment:" in prompt
    assert "Task:" in prompt
    assert "Superman-themed" in prompt
    assert "Remove the original garment completely" in prompt


def test_prompt_builder_klein_starts_with_tryon():
    result = build_prompt(_request("tc1", EngineMode.KLEIN_LORA))
    assert result.positive_prompt.startswith("TRYON")


def test_prompt_builder_klein_contains_full_body_shot():
    result = build_prompt(_request("tc1", EngineMode.KLEIN_LORA))
    assert "The final image is a full body shot." in result.positive_prompt


def test_prompt_builder_flux_contains_only_masked_region():
    result = build_prompt(_request("tc10", EngineMode.IDM_MASK_EXPANDED_FLUX))
    assert result.core_prompt
    assert result.refine_prompt
    assert "Refine only the masked clothing region" in result.refine_prompt
    assert "all unmasked clothing exactly" in result.refine_prompt


def test_prompt_builder_catvton_minimal():
    result = build_prompt(_request("tc1", EngineMode.CATVTON))
    assert result.positive_prompt == CATVTON_TEMPLATE


def test_prompt_builder_adetailer_preserve_unmasked():
    result = build_prompt(_request("tc1", EngineMode.ADETAILER_REPAIR))
    assert "Preserve all unmasked pixels exactly." in result.positive_prompt


def test_prompt_safety_child_no_underwear_terms():
    result = build_prompt(_request("tc11", EngineMode.KLEIN_LORA))
    text = result.positive_prompt.lower()
    for forbidden in ["underwear", "brief", "bikini", "lingerie", "bra", "sexy", "seductive"]:
        assert forbidden not in text
    assert "young girl" in text


def test_prompt_safety_adult_underwear_neutral_terms():
    result = build_prompt(_request("tc10", EngineMode.IDM))
    text = result.positive_prompt.lower()
    assert "adult" in text
    assert "brief underwear" in text
    for forbidden in ["sexy", "seductive", "provocative"]:
        assert forbidden not in text


def test_prompt_hash_stable():
    request = _request("tc10", EngineMode.IDM_MASK_EXPANDED, PromptVariant.IDENTITY_STRICT)
    first = build_prompt(request).metadata["prompt_hash"]
    second = build_prompt(request).metadata["prompt_hash"]
    assert first == second


def test_generate_prompts_cli_all_engines(tmp_path):
    output_dir = tmp_path / "prompts"
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "generate_prompts.py"),
            "--testcase",
            "tc10",
            "--engine",
            "all",
            "--variant",
            "strong_remove_old_garment",
            "--output",
            str(output_dir),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    summary = json.loads((output_dir / "prompts_summary.json").read_text(encoding="utf-8"))
    assert summary["count"] == len(EngineMode)
    assert (output_dir / "tc10_idm_prompt.txt").exists()
    assert (output_dir / "tc10_klein_lora_prompt.txt").exists()
    assert (output_dir / "tc10_flux_refine_prompt.txt").exists()


def test_benchmark_saves_prompt_artifacts(tmp_path):
    eval_set = _write_eval_sample(tmp_path / "eval_set")
    output_dir = tmp_path / "benchmark"
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "benchmark_pipeline.py"),
            "--eval-set",
            str(eval_set),
            "--modes",
            "idm",
            "--limit",
            "1",
            "--output",
            str(output_dir),
            "--mock",
            "--prompt-source",
            "auto",
            "--prompt-variant",
            "strong_remove_old_garment",
            "--testcase-id",
            "tc10",
            "--save-prompts",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    mode_dir = output_dir / "sample_001" / "idm"
    assert (mode_dir / "prompt_core.txt").exists()
    assert (mode_dir / "negative_prompt.txt").exists()
    metadata = json.loads((mode_dir / "prompt_metadata.json").read_text(encoding="utf-8"))
    assert metadata["metadata"]["prompt_hash"]
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    row = summary["rows"][0]
    assert row["prompt_variant"] == "strong_remove_old_garment"
    assert row["prompt_hash"] == metadata["metadata"]["prompt_hash"]


def test_api_auto_prompt_preserves_existing_behavior(client, png_file):
    api = TestClient(client)
    response = api.post(
        "/tryon",
        data={
            "category": "upper_body",
            "engine_mode": "idm_vton",
            "auto_prompt": "true",
            "testcase_id": "tc1",
            "prompt_variant": "default",
            "use_refiner": "false",
            "repair_mode": "false",
            "run_mode": "sync",
        },
        files={
            "person_image": png_file("person.png", (170, 170, 170)),
            "garment_top": png_file("top.png", (20, 80, 210)),
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["debug"]["prompt_core_url"].endswith("/prompt_core.txt")
    assert payload["debug"]["prompt_metadata_url"].endswith("/prompt_metadata.json")


def test_quality_report_does_not_store_subjective_prompt_notes(tmp_path):
    eval_set = _write_eval_sample(tmp_path / "eval_set")
    output_dir = tmp_path / "benchmark"
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "benchmark_pipeline.py"),
            "--eval-set",
            str(eval_set),
            "--modes",
            "idm",
            "--limit",
            "1",
            "--output",
            str(output_dir),
            "--mock",
            "--prompt-source",
            "auto",
            "--testcase-id",
            "tc10",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    serialized = (output_dir / "sample_001" / "idm" / "quality_report.json").read_text(encoding="utf-8").lower()
    assert "old_garment_removed_1_5" not in serialized
    assert "identity_1_5" not in serialized
    assert "subjective" not in serialized
