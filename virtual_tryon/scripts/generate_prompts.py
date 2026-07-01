from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from app.prompts.prompt_builder import build_prompt  # noqa: E402
from app.prompts.prompt_types import EngineMode, PromptVariant  # noqa: E402
from app.prompts.testcase_prompt_library import get_testcase, list_testcases  # noqa: E402


def _parse_testcases(value: str):
    return list_testcases() if value == "all" else [get_testcase(value)]


def _parse_engines(value: str) -> list[EngineMode]:
    if value == "all":
        return list(EngineMode)
    try:
        return [EngineMode(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def _write_text(path: Path, result) -> None:
    sections = [result.positive_prompt]
    if result.refine_prompt and result.refine_prompt != result.positive_prompt:
        sections.extend(["", "Refine prompt:", result.refine_prompt])
    if result.negative_prompt:
        sections.extend(["", "Negative prompt:", result.negative_prompt])
    path.write_text("\n".join(sections) + "\n", encoding="utf-8")


def generate_prompts(
    *,
    testcase_value: str,
    engine_value: str,
    variant: PromptVariant,
    output_dir: Path,
    output_format: str = "txt",
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    markdown = ["# Generated Prompts", ""]
    for testcase in _parse_testcases(testcase_value):
        for engine in _parse_engines(engine_value):
            result = build_prompt(testcase.build_request(engine, variant))
            stem = f"{testcase.testcase_id}_{engine.value}_prompt"
            if engine == EngineMode.IDM_MASK_EXPANDED_FLUX:
                stem = f"{testcase.testcase_id}_flux_refine_prompt"
            extension = "json" if output_format == "json" else "txt"
            prompt_path = output_dir / f"{stem}.{extension}"
            if output_format == "json":
                prompt_path.write_text(
                    json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            else:
                _write_text(prompt_path, result)
            row = {
                "testcase_id": testcase.testcase_id,
                "engine_mode": engine.value,
                "prompt_variant": variant.value,
                "prompt_hash": result.metadata["prompt_hash"],
                "prompt_path": prompt_path.name,
                "warnings": result.warnings,
                "positive_prompt": result.positive_prompt,
                "negative_prompt": result.negative_prompt,
                "core_prompt": result.core_prompt,
                "refine_prompt": result.refine_prompt,
            }
            rows.append(row)
            markdown.extend(
                [
                    f"## {testcase.testcase_id} / {engine.value}",
                    "",
                    f"- Variant: `{variant.value}`",
                    f"- Hash: `{result.metadata['prompt_hash']}`",
                    "",
                    result.positive_prompt,
                    "",
                ]
            )
            if result.refine_prompt:
                markdown.extend(["**Refine prompt**", "", result.refine_prompt, ""])
            if result.warnings:
                markdown.extend(["**Warnings:** " + "; ".join(result.warnings), ""])
    summary = {
        "testcase": testcase_value,
        "engine": engine_value,
        "variant": variant.value,
        "count": len(rows),
        "rows": rows,
    }
    (output_dir / "prompts_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "prompts_summary.md").write_text("\n".join(markdown), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate deterministic engine-specific VTON prompts.")
    parser.add_argument("--testcase", default="all")
    parser.add_argument("--engine", default="all")
    parser.add_argument(
        "--variant",
        default=PromptVariant.DEFAULT.value,
        choices=[item.value for item in PromptVariant],
    )
    parser.add_argument("--format", default="txt", choices=["txt", "json", "markdown"])
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = generate_prompts(
        testcase_value=args.testcase,
        engine_value=args.engine,
        variant=PromptVariant(args.variant),
        output_dir=Path(args.output),
        output_format="txt" if args.format == "markdown" else args.format,
    )
    print(f"generated={summary['count']}")
    print(f"summary_json={Path(args.output) / 'prompts_summary.json'}")
    print(f"summary_markdown={Path(args.output) / 'prompts_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
