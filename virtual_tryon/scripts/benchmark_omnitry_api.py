#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from PIL import Image, ImageDraw, ImageFont


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
IMAGE_CONTENT_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}
DEFAULT_FLOWS = "idm_vton:10,idm_mask_expanded:10,klein_lora:4"
TERMINAL_STATUSES = {"completed", "failed", "cancelled", "canceled"}
CSV_COLUMNS = [
    "row_id",
    "round",
    "case_id",
    "gender",
    "category",
    "flow",
    "engine_mode",
    "steps",
    "seed",
    "width",
    "height",
    "status",
    "job_id",
    "runtime_seconds",
    "score",
    "improved",
    "background_preservation_score",
    "face_preservation_score",
    "garment_change_score",
    "garment_similarity_score",
    "artifact_score",
    "over_edit_score",
    "needs_refine",
    "result_path",
    "quality_report_path",
    "error",
]


@dataclass(frozen=True)
class OmniTryCase:
    case_id: str
    gender: str
    category: str
    person_path: Path
    garment_path: Path
    garment_field: str


@dataclass(frozen=True)
class Flow:
    label: str
    engine_mode: str
    steps: int
    use_refiner: bool
    repair_mode: bool


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return safe.strip("._") or "item"


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in CSV_COLUMNS})


def discover_cases(dataset_root: Path) -> list[OmniTryCase]:
    input_models = dataset_root / "input_models"
    female_model = input_models / "female_model.jpg"
    male_model_1 = input_models / "male_model_1.jpg"
    male_model_2 = input_models / "male_model_2.jpg"
    required = [female_model, male_model_1, male_model_2]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"Missing OmniTry model image(s): {', '.join(path.as_posix() for path in missing)}")

    cases: list[OmniTryCase] = []
    female_dir = dataset_root / "female_undergarmentt"
    for garment_path in sorted(female_dir.glob("*")):
        if garment_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        case_id = f"female_{_safe_name(garment_path.stem)}"
        cases.append(
            OmniTryCase(
                case_id=case_id,
                gender="female",
                category="women_underwear",
                person_path=female_model,
                garment_path=garment_path,
                garment_field="garment_bottom",
            )
        )

    male_dir = dataset_root / "male_undergarment"
    male_model_1_numbers = {3, 4, 8, 9}
    for garment_path in sorted(male_dir.glob("*")):
        if garment_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        match = re.search(r"(\d+)$", garment_path.stem)
        number = int(match.group(1)) if match else None
        person_path = male_model_1 if number in male_model_1_numbers else male_model_2
        model_id = "m1" if person_path == male_model_1 else "m2"
        case_id = f"male_{model_id}_{_safe_name(garment_path.stem)}"
        cases.append(
            OmniTryCase(
                case_id=case_id,
                gender="male",
                category="men_underwear",
                person_path=person_path,
                garment_path=garment_path,
                garment_field="garment_bottom",
            )
        )

    if not cases:
        raise SystemExit(f"No OmniTry garment images found under {dataset_root}")
    return cases


def filter_cases(
    cases: list[OmniTryCase],
    *,
    case_ids: list[str] | None,
    gender: str | None,
    case_regex: str | None,
) -> list[OmniTryCase]:
    selected = list(cases)
    if case_ids:
        wanted = set(case_ids)
        selected = [case for case in selected if case.case_id in wanted]
        missing = sorted(wanted - {case.case_id for case in selected})
        if missing:
            raise SystemExit(f"Unknown case id(s): {', '.join(missing)}")
    if gender:
        selected = [case for case in selected if case.gender == gender]
    if case_regex:
        pattern = re.compile(case_regex)
        selected = [case for case in selected if pattern.search(case.case_id)]
    if not selected:
        raise SystemExit("Case filters matched no OmniTry cases.")
    return selected


def parse_flows(value: str) -> list[Flow]:
    flows: list[Flow] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "=" in token:
            label, spec = token.split("=", 1)
            label = _safe_name(label)
        else:
            label = ""
            spec = token
        parts = [part.strip() for part in spec.split(":") if part.strip()]
        engine_mode = parts[0]
        steps = int(parts[1]) if len(parts) >= 2 else 10
        flags = set(parts[2:])
        use_refiner = "refine" in flags or engine_mode.endswith("_flux")
        repair_mode = "repair" in flags
        label = label or f"{engine_mode}_s{steps}"
        if use_refiner:
            label += "_refine" if not label.endswith("_refine") else ""
        if repair_mode:
            label += "_repair" if not label.endswith("_repair") else ""
        flows.append(
            Flow(
                label=label,
                engine_mode=engine_mode,
                steps=steps,
                use_refiner=use_refiner,
                repair_mode=repair_mode,
            )
        )
    if not flows:
        raise SystemExit("No flows were provided.")
    return flows


def _load_state(output_dir: Path, resume: bool) -> dict[str, Any]:
    state_path = output_dir / "state.json"
    if resume:
        state = _read_json(state_path, {})
    else:
        state = {}
    state.setdefault("created_at", _now())
    state.setdefault("updated_at", _now())
    state.setdefault("rows", [])
    state.setdefault("completed_keys", [])
    state.setdefault("best_by_case", {})
    state.setdefault("completed_rounds", [])
    state.setdefault("no_improvement_rounds", 0)
    return state


def _save_state(output_dir: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = _now()
    _write_json(output_dir / "state.json", state)
    rows = list(state.get("rows", []))
    _write_json(output_dir / "summary.json", rows)
    _write_csv(output_dir / "summary.csv", rows)


def _round_key(round_index: int, case: OmniTryCase, flow: Flow) -> str:
    return f"r{round_index}:{case.case_id}:{flow.label}"


def _next_tasks(
    cases: list[OmniTryCase],
    flows: list[Flow],
    state: dict[str, Any],
    limit: int,
) -> list[tuple[int, OmniTryCase, Flow]]:
    completed = set(state.get("completed_keys", []))
    tasks: list[tuple[int, OmniTryCase, Flow]] = []
    round_index = 1
    while len(tasks) < limit:
        for flow in flows:
            for case in cases:
                key = _round_key(round_index, case, flow)
                if key in completed:
                    continue
                tasks.append((round_index, case, flow))
                if len(tasks) >= limit:
                    return tasks
        round_index += 1
    return tasks


def _encode_multipart(fields: dict[str, Any], files: dict[str, Path]) -> tuple[bytes, str]:
    boundary = f"----omnitry-{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for name, value in fields.items():
        if value is None:
            continue
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"),
                f"{value}\r\n".encode("utf-8"),
            ]
        )
    for name, path in files.items():
        content_type = IMAGE_CONTENT_TYPES.get(path.suffix.lower()) or mimetypes.guess_type(path.name)[0]
        content_type = content_type or "application/octet-stream"
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                (
                    f'Content-Disposition: form-data; name="{name}"; '
                    f'filename="{path.name}"\r\n'
                ).encode("utf-8"),
                f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"),
                path.read_bytes(),
                b"\r\n",
            ]
        )
    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def _json_request(url: str, timeout: float) -> dict[str, Any]:
    try:
        with urlopen(url, timeout=timeout) as response:
            payload = response.read()
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body[:1000]}") from exc
    return json.loads(payload.decode("utf-8"))


def _submit_job(
    api_base: str,
    case: OmniTryCase,
    flow: Flow,
    category: str,
    seed: int,
    width: int,
    height: int,
    auto_prompt: bool,
    prompt_variant: str,
    prompt: str | None,
    timeout: float,
) -> dict[str, Any]:
    fields = {
        "category": category,
        "run_mode": "async",
        "engine_mode": flow.engine_mode,
        "use_refiner": str(flow.use_refiner).lower(),
        "repair_mode": str(flow.repair_mode).lower(),
        "deterministic": "true",
        "seed": seed,
        "output_width": width,
        "output_height": height,
        "steps": flow.steps,
        "auto_prompt": str(auto_prompt).lower(),
        "prompt_variant": prompt_variant,
    }
    if prompt:
        fields["prompt"] = prompt
    files = {
        "person_image": case.person_path,
        case.garment_field: case.garment_path,
    }
    body, content_type = _encode_multipart(fields, files)
    request = Request(
        f"{api_base.rstrip('/')}/tryon",
        data=body,
        headers={"Content-Type": content_type, "Accept": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = response.read()
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body[:1000]}") from exc
    return json.loads(payload.decode("utf-8"))


def _poll_job(
    api_base: str,
    job_id: str,
    poll_interval: float,
    job_timeout: float,
    request_timeout: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + job_timeout
    last_payload: dict[str, Any] = {}
    while time.monotonic() < deadline:
        payload = _json_request(f"{api_base.rstrip('/')}/tryon/{job_id}", timeout=request_timeout)
        last_payload = payload
        status = str(payload.get("status", "")).lower()
        if status in TERMINAL_STATUSES:
            return payload
        time.sleep(poll_interval)
    last_payload["status"] = last_payload.get("status") or "timeout"
    last_payload["error"] = f"Timed out after {job_timeout:.1f}s"
    return last_payload


def _iter_artifact_urls(job: dict[str, Any]) -> list[tuple[str, str]]:
    urls: list[tuple[str, str]] = []
    scalar_keys = [
        "result_url",
        "output_url",
        "mask_url",
        "quality_report_url",
        "artifact_manifest_url",
    ]
    for key in scalar_keys:
        value = job.get(key)
        if isinstance(value, str) and value:
            urls.append((key, value))

    for group_key in ("artifacts", "debug_artifacts", "debug_urls", "outputs"):
        group = job.get(group_key)
        if isinstance(group, dict):
            for key, value in group.items():
                if isinstance(value, str) and value:
                    urls.append((f"{group_key}_{key}", value))
        elif isinstance(group, list):
            for index, item in enumerate(group):
                if isinstance(item, str):
                    urls.append((f"{group_key}_{index}", item))
                elif isinstance(item, dict):
                    url = item.get("url") or item.get("path")
                    name = item.get("name") or item.get("key") or str(index)
                    if isinstance(url, str) and url:
                        urls.append((f"{group_key}_{name}", url))
    return urls


def _absolute_url(api_base: str, url: str) -> str:
    if url.startswith("http://") or url.startswith("https://"):
        return url
    if not url.startswith("/"):
        url = f"/{url}"
    return f"{api_base.rstrip('/')}{url}"


def _download_url(url: str, path: Path, timeout: float) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urlopen(url, timeout=timeout) as response:
            data = response.read()
    except (HTTPError, URLError, TimeoutError, OSError):
        return False
    path.write_bytes(data)
    return True


def _download_artifacts(
    api_base: str,
    job: dict[str, Any],
    job_dir: Path,
    request_timeout: float,
) -> dict[str, str]:
    downloaded: dict[str, str] = {}
    seen: set[str] = set()
    for name, url in _iter_artifact_urls(job):
        if url in seen:
            continue
        seen.add(url)
        parsed = urlparse(url)
        suffix = Path(parsed.path).suffix or ".bin"
        if "quality" in name and suffix == ".bin":
            suffix = ".json"
        file_name = f"{_safe_name(name)}{suffix}"
        output_path = job_dir / file_name
        if _download_url(_absolute_url(api_base, url), output_path, timeout=request_timeout):
            downloaded[name] = output_path.as_posix()

    return downloaded


def _first_existing(downloaded: dict[str, str], names: list[str]) -> str | None:
    for name in names:
        path = downloaded.get(name)
        if path and Path(path).exists():
            return path
    for key, path in downloaded.items():
        if any(name in key for name in names) and Path(path).exists():
            return path
    return None


def _load_quality_report(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.exists() or candidate.suffix.lower() != ".json":
        return None
    try:
        return _read_json(candidate, None)
    except json.JSONDecodeError:
        return None


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _quality_metrics(job: dict[str, Any], quality_report: dict[str, Any] | None) -> dict[str, Any]:
    metrics: dict[str, Any] = {}

    direct = job.get("quality") or job.get("metrics")
    if isinstance(direct, dict):
        metrics.update(direct)

    if quality_report:
        final_choice = quality_report.get("final_choice") or "core"
        chosen = quality_report.get(final_choice)
        if not isinstance(chosen, dict):
            chosen = quality_report.get("core")
        if isinstance(chosen, dict):
            metrics.update(chosen)
        for key in (
            "background_preservation_score",
            "face_preservation_score",
            "garment_change_score",
            "garment_similarity_score",
            "artifact_score",
            "over_edit_score",
            "needs_refine",
        ):
            if key in quality_report and key not in metrics:
                metrics[key] = quality_report[key]

    return {
        "background_preservation_score": _float_or_none(metrics.get("background_preservation_score")),
        "face_preservation_score": _float_or_none(metrics.get("face_preservation_score")),
        "garment_change_score": _float_or_none(metrics.get("garment_change_score")),
        "garment_similarity_score": _float_or_none(metrics.get("garment_similarity_score")),
        "artifact_score": _float_or_none(metrics.get("artifact_score")),
        "over_edit_score": _float_or_none(metrics.get("over_edit_score")),
        "needs_refine": bool(metrics.get("needs_refine")) if metrics.get("needs_refine") is not None else None,
    }


def _score(metrics: dict[str, Any]) -> float | None:
    values = {
        key: _float_or_none(metrics.get(key))
        for key in (
            "background_preservation_score",
            "face_preservation_score",
            "garment_change_score",
            "garment_similarity_score",
            "artifact_score",
            "over_edit_score",
        )
    }
    if not any(value is not None for value in values.values()):
        return None
    score = 0.0
    score += 0.35 * (values["garment_change_score"] if values["garment_change_score"] is not None else 0.5)
    score += 0.25 * (values["garment_similarity_score"] if values["garment_similarity_score"] is not None else 0.5)
    score += 0.2 * (
        values["background_preservation_score"] if values["background_preservation_score"] is not None else 0.5
    )
    score += 0.1 * (values["face_preservation_score"] if values["face_preservation_score"] is not None else 0.5)
    score -= 0.2 * (values["artifact_score"] if values["artifact_score"] is not None else 0.0)
    score -= 0.25 * (values["over_edit_score"] if values["over_edit_score"] is not None else 0.0)
    if metrics.get("needs_refine"):
        score -= 0.05
    return round(score, 6)


def _mark_improvement(
    state: dict[str, Any],
    case: OmniTryCase,
    row: dict[str, Any],
    epsilon: float,
) -> bool:
    score = _float_or_none(row.get("score"))
    if score is None or row.get("status") != "completed":
        return False
    best_by_case = state.setdefault("best_by_case", {})
    previous = best_by_case.get(case.case_id)
    previous_score = _float_or_none(previous.get("score")) if isinstance(previous, dict) else None
    improved = previous_score is None or score > previous_score + epsilon
    if improved:
        best_by_case[case.case_id] = {
            "score": score,
            "row_id": row["row_id"],
            "flow": row["flow"],
            "engine_mode": row["engine_mode"],
            "result_path": row.get("result_path"),
            "updated_at": _now(),
        }
    return improved


def _refresh_round_stop_state(
    state: dict[str, Any],
    cases: list[OmniTryCase],
    flows: list[Flow],
) -> None:
    completed_rounds = set(state.get("completed_rounds", []))
    completed_keys = set(state.get("completed_keys", []))
    rows = list(state.get("rows", []))
    max_round = 0
    for key in completed_keys:
        match = re.match(r"r(\d+):", key)
        if match:
            max_round = max(max_round, int(match.group(1)))

    for round_index in range(1, max_round + 1):
        if round_index in completed_rounds:
            continue
        expected = {_round_key(round_index, case, flow) for flow in flows for case in cases}
        if not expected.issubset(completed_keys):
            continue
        improved = any(
            int(row.get("round", 0)) == round_index and str(row.get("improved", "")).lower() == "true"
            for row in rows
        )
        if improved:
            state["no_improvement_rounds"] = 0
        else:
            state["no_improvement_rounds"] = int(state.get("no_improvement_rounds", 0)) + 1
        completed_rounds.add(round_index)
    state["completed_rounds"] = sorted(completed_rounds)


def _font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError:
        return ImageFont.load_default()


def _title_font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", 17)
    except OSError:
        return ImageFont.load_default()


def _draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill: tuple[int, int, int], font: ImageFont.ImageFont) -> None:
    draw.text(xy, text, fill=fill, font=font)


def _thumb(path: Path, size: tuple[int, int]) -> Image.Image:
    try:
        image = Image.open(path).convert("RGB")
    except OSError:
        image = Image.new("RGB", size, (235, 235, 235))
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (248, 248, 248))
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def _sheet_cell(
    image_path: Path,
    title: str,
    subtitle: str,
    cell_size: tuple[int, int],
    image_size: tuple[int, int],
) -> Image.Image:
    cell_w, cell_h = cell_size
    image_h = image_size[1]
    cell = Image.new("RGB", cell_size, (255, 255, 255))
    draw = ImageDraw.Draw(cell)
    title_font = _title_font()
    normal_font = _font()
    _draw_text(draw, (10, 8), title[:34], (20, 20, 20), title_font)
    _draw_text(draw, (10, 31), subtitle[:42], (70, 70, 70), normal_font)
    preview = _thumb(image_path, image_size)
    cell.paste(preview, ((cell_w - image_size[0]) // 2, cell_h - image_h - 10))
    draw.rectangle((0, 0, cell_w - 1, cell_h - 1), outline=(220, 220, 220))
    return cell


def write_case_sheets(
    output_dir: Path,
    cases: list[OmniTryCase],
    rows: list[dict[str, Any]],
) -> None:
    sheet_dir = output_dir / "sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    rows_by_case: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_case.setdefault(str(row.get("case_id")), []).append(row)

    cell_size = (280, 410)
    image_size = (250, 330)
    cols = 4
    for case in cases:
        cells = [
            _sheet_cell(case.person_path, "person", case.person_path.name, cell_size, image_size),
            _sheet_cell(case.garment_path, "garment", case.garment_path.name, cell_size, image_size),
        ]
        case_rows = sorted(
            rows_by_case.get(case.case_id, []),
            key=lambda row: (int(row.get("round", 0)), str(row.get("flow", "")), str(row.get("row_id", ""))),
        )
        for row in case_rows:
            result_path = row.get("result_path")
            if not result_path:
                continue
            path = Path(result_path)
            if not path.exists():
                path = output_dir / result_path
            score = row.get("score")
            score_text = f"score={score:.3f}" if isinstance(score, (int, float)) else "score=N/A"
            status = str(row.get("status", ""))
            title = f"r{row.get('round')} {row.get('flow')}"
            subtitle = f"{status} {score_text}"
            cells.append(_sheet_cell(path, title, subtitle, cell_size, image_size))

        rows_count = (len(cells) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * cell_size[0], rows_count * cell_size[1]), (242, 242, 242))
        for index, cell in enumerate(cells):
            x = (index % cols) * cell_size[0]
            y = (index // cols) * cell_size[1]
            sheet.paste(cell, (x, y))
        sheet.save(sheet_dir / f"{case.case_id}.jpg", quality=92)


def write_latest_grid(output_dir: Path, rows: list[dict[str, Any]], limit: int = 24) -> None:
    completed = [row for row in rows if row.get("status") == "completed" and row.get("result_path")]
    completed = completed[-limit:]
    if not completed:
        return
    cell_size = (280, 410)
    image_size = (250, 330)
    cols = min(4, len(completed))
    rows_count = (len(completed) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell_size[0], rows_count * cell_size[1]), (242, 242, 242))
    for index, row in enumerate(completed):
        result_path = Path(str(row["result_path"]))
        if not result_path.exists():
            result_path = output_dir / str(row["result_path"])
        score = row.get("score")
        score_text = f"score={score:.3f}" if isinstance(score, (int, float)) else "score=N/A"
        title = f"{row.get('case_id')} | {row.get('flow')}"
        subtitle = f"r{row.get('round')} {score_text}"
        cell = _sheet_cell(result_path, title, subtitle, cell_size, image_size)
        x = (index % cols) * cell_size[0]
        y = (index // cols) * cell_size[1]
        sheet.paste(cell, (x, y))
    sheet.save(output_dir / "grid_latest.jpg", quality=92)


def _run_one(
    args: argparse.Namespace,
    output_dir: Path,
    state: dict[str, Any],
    cases: list[OmniTryCase],
    flows: list[Flow],
    round_index: int,
    case: OmniTryCase,
    flow: Flow,
    case_index: int,
    flow_index: int,
) -> dict[str, Any]:
    row_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    seed = int(args.seed) + (round_index - 1) * 100_000 + flow_index * 1_000 + case_index * int(args.seed_stride)
    key = _round_key(round_index, case, flow)
    job_dir = output_dir / "jobs" / case.case_id / flow.label / f"round_{round_index}" / row_id
    job_dir.mkdir(parents=True, exist_ok=True)
    category = args.category_override or case.category

    row: dict[str, Any] = {
        "row_id": row_id,
        "round": round_index,
        "case_id": case.case_id,
        "gender": case.gender,
        "category": category,
        "flow": flow.label,
        "engine_mode": flow.engine_mode,
        "steps": flow.steps,
        "seed": seed,
        "width": args.width,
        "height": args.height,
        "status": "submitted",
        "job_id": "",
        "runtime_seconds": None,
        "score": None,
        "improved": False,
        "result_path": "",
        "quality_report_path": "",
        "error": "",
    }
    start = time.monotonic()
    try:
        submit_timeout = args.submit_timeout if args.submit_timeout is not None else args.job_timeout + 30.0
        submit_payload = _submit_job(
            args.api_base,
            case,
            flow,
            category=category,
            seed=seed,
            width=args.width,
            height=args.height,
            auto_prompt=args.auto_prompt,
            prompt_variant=args.prompt_variant,
            prompt=args.prompt,
            timeout=submit_timeout,
        )
        job_id = str(submit_payload.get("job_id") or submit_payload.get("id") or "")
        if not job_id:
            raise RuntimeError(f"Submit did not return a job id: {submit_payload}")
        row["job_id"] = job_id
        submit_status = str(submit_payload.get("status", "")).lower()
        print(
            f"submitted {case.case_id} {flow.label} r{round_index} "
            f"job={job_id} seed={seed} status={submit_status or 'unknown'}",
            flush=True,
        )
        if submit_status in TERMINAL_STATUSES:
            job_payload = submit_payload
        else:
            job_payload = _poll_job(
                args.api_base,
                job_id,
                poll_interval=args.poll_interval,
                job_timeout=args.job_timeout,
                request_timeout=args.request_timeout,
            )
        row["status"] = str(job_payload.get("status") or "unknown").lower()
        row["runtime_seconds"] = round(time.monotonic() - start, 3)
        _write_json(job_dir / "job_status.json", job_payload)
        downloaded = _download_artifacts(args.api_base, job_payload, job_dir, request_timeout=args.request_timeout)
        result_path = _first_existing(
            downloaded,
            ["result_url", "output_url", "artifacts_final", "outputs_final", "final", "result"],
        )
        if not result_path:
            result_path = _first_existing(downloaded, ["core_output", "core", "raw"])
        quality_report_path = _first_existing(downloaded, ["quality_report_url", "quality"])
        row["result_path"] = result_path or ""
        row["quality_report_path"] = quality_report_path or ""
        quality_report = _load_quality_report(quality_report_path)
        metrics = _quality_metrics(job_payload, quality_report)
        row.update(metrics)
        row["score"] = _score(metrics)
        row["improved"] = _mark_improvement(state, case, row, epsilon=args.improvement_epsilon)
        if row["status"] != "completed":
            row["error"] = str(job_payload.get("error") or job_payload.get("message") or "")
    except Exception as exc:
        row["status"] = "failed"
        row["runtime_seconds"] = round(time.monotonic() - start, 3)
        row["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        state.setdefault("rows", []).append(row)
        state.setdefault("completed_keys", []).append(key)
        _refresh_round_stop_state(state, cases, flows)
        _save_state(output_dir, state)
        write_case_sheets(output_dir, cases, list(state.get("rows", [])))
        write_latest_grid(output_dir, list(state.get("rows", [])))
    score_text = row["score"] if row["score"] is not None else "N/A"
    print(
        f"finished {case.case_id} {flow.label} r{round_index} "
        f"status={row['status']} score={score_text} improved={row['improved']} "
        f"runtime={row['runtime_seconds']}s",
        flush=True,
    )
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an OmniTry API sweep across try-on engines and flows.")
    parser.add_argument("--dataset-root", required=True, type=Path, help="Folder containing input_models and garment dirs.")
    parser.add_argument("--api-base", default="http://127.0.0.1:8000", help="Try-on backend base URL.")
    parser.add_argument("--output", default=Path("data/outputs/omnitry_engine_sweep"), type=Path)
    parser.add_argument("--flows", default=DEFAULT_FLOWS, help="Comma list like idm_vton:10,idm_mask_expanded:10,klein_lora:4.")
    parser.add_argument("--case-id", action="append", default=[], help="Run only the selected case id. Can be repeated.")
    parser.add_argument("--case-regex", default=None, help="Run only cases whose id matches this regex.")
    parser.add_argument("--gender", choices=["male", "female"], default=None)
    parser.add_argument("--min-jobs", default=15, type=int, help="Minimum jobs to schedule in this invocation.")
    parser.add_argument("--max-jobs", default=15, type=int, help="Maximum jobs to schedule in this invocation.")
    parser.add_argument("--stop-after-no-improvement-rounds", default=4, type=int)
    parser.add_argument("--width", default=512, type=int)
    parser.add_argument("--height", default=768, type=int)
    parser.add_argument("--seed", default=30012005, type=int)
    parser.add_argument("--seed-stride", default=97, type=int)
    parser.add_argument("--prompt-variant", default="default")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--category-override", default=None)
    parser.add_argument("--auto-prompt", action="store_true")
    parser.add_argument("--improvement-epsilon", default=0.01, type=float)
    parser.add_argument("--poll-interval", default=3.0, type=float)
    parser.add_argument("--job-timeout", default=900.0, type=float)
    parser.add_argument("--request-timeout", default=60.0, type=float)
    parser.add_argument("--submit-timeout", default=None, type=float)
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing state.json.")
    parser.add_argument("--dry-run", action="store_true", help="Print the next schedule without submitting jobs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = filter_cases(
        discover_cases(args.dataset_root),
        case_ids=args.case_id,
        gender=args.gender,
        case_regex=args.case_regex,
    )
    flows = parse_flows(args.flows)
    state = _load_state(output_dir, resume=not args.no_resume)

    if int(state.get("no_improvement_rounds", 0)) >= args.stop_after_no_improvement_rounds:
        print(
            "stop condition already reached: "
            f"{state.get('no_improvement_rounds')} full rounds without improvement",
            flush=True,
        )
        return

    limit = max(args.min_jobs, args.max_jobs)
    tasks = _next_tasks(cases, flows, state, limit=limit)
    if args.max_jobs:
        tasks = tasks[: args.max_jobs]
    if not tasks:
        print("No pending tasks found.", flush=True)
        return

    print(
        f"OmniTry sweep: cases={len(cases)} flows={len(flows)} scheduled={len(tasks)} "
        f"output={output_dir.as_posix()}",
        flush=True,
    )
    for index, (round_index, case, flow) in enumerate(tasks, start=1):
        print(f"schedule[{index}] r{round_index} {case.case_id} {flow.label}", flush=True)

    if args.dry_run:
        return

    case_index_by_id = {case.case_id: index for index, case in enumerate(cases)}
    flow_index_by_label = {flow.label: index for index, flow in enumerate(flows)}
    for task_index, (round_index, case, flow) in enumerate(tasks, start=1):
        if int(state.get("no_improvement_rounds", 0)) >= args.stop_after_no_improvement_rounds:
            print(
                "stop condition reached: "
                f"{state.get('no_improvement_rounds')} full rounds without improvement",
                flush=True,
            )
            break
        print(f"run[{task_index}/{len(tasks)}] r{round_index} {case.case_id} {flow.label}", flush=True)
        _run_one(
            args,
            output_dir,
            state,
            cases,
            flows,
            round_index,
            case,
            flow,
            case_index_by_id[case.case_id],
            flow_index_by_label[flow.label],
        )

    _save_state(output_dir, state)
    write_case_sheets(output_dir, cases, list(state.get("rows", [])))
    write_latest_grid(output_dir, list(state.get("rows", [])))
    completed = [row for row in state.get("rows", []) if row.get("status") == "completed"]
    failed = [row for row in state.get("rows", []) if row.get("status") != "completed"]
    print(
        f"done: total_rows={len(state.get('rows', []))} completed={len(completed)} "
        f"non_completed={len(failed)} no_improvement_rounds={state.get('no_improvement_rounds', 0)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
