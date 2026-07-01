from __future__ import annotations

import json
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, File, Form, UploadFile

from app.core.config import get_settings
from app.engines.factory import create_refiner
from app.preprocessing.image_loader import load_image_from_bytes, validate_mime
from app.schemas.tryon import (
    GenerationConfigSummary,
    HistoryInputs,
    RefineResponse,
    TryOnCategory,
    TryOnHistoryItem,
    TryOnHistoryResponse,
    TryOnResponse,
    TryOnStatusResponse,
)
from app.services.artifact_service import build_artifact_url
from app.services.container import get_job_service, get_storage_service
from app.services.tryon_pipeline import PipelineRequest
from app.utils.errors import ApiError, InputValidationError, ModelUnavailableError
from app.utils.image_io import save_image
from app.utils.seed import normalize_seed, set_seed


router = APIRouter(tags=["tryon"])


MIN_OUTPUT_SIDE = 384
MAX_OUTPUT_SIDE = 1536
MAX_OUTPUT_PIXELS = 1024 * 1536
MIN_STEPS = 4
MAX_STEPS = 50


def _validate_generation_overrides(
    output_width: int | None,
    output_height: int | None,
    steps: int | None,
) -> None:
    if (output_width is None) ^ (output_height is None):
        raise ApiError(
            "INVALID_REQUEST",
            "output_width and output_height must be provided together.",
            status_code=400,
        )
    if output_width is not None and output_height is not None:
        if output_width < MIN_OUTPUT_SIDE or output_height < MIN_OUTPUT_SIDE:
            raise ApiError(
                "INVALID_REQUEST",
                f"Output resolution must be at least {MIN_OUTPUT_SIDE}px on each side.",
                status_code=400,
            )
        if output_width > MAX_OUTPUT_SIDE or output_height > MAX_OUTPUT_SIDE:
            raise ApiError(
                "INVALID_REQUEST",
                f"Output resolution cannot exceed {MAX_OUTPUT_SIDE}px on either side.",
                status_code=400,
            )
        if output_width * output_height > MAX_OUTPUT_PIXELS:
            raise ApiError(
                "INVALID_REQUEST",
                f"Output resolution cannot exceed {MAX_OUTPUT_PIXELS} total pixels.",
                status_code=400,
            )
        if output_width % 8 or output_height % 8:
            raise ApiError(
                "INVALID_REQUEST",
                "output_width and output_height must be multiples of 8.",
                status_code=400,
            )
    if steps is not None and not (MIN_STEPS <= steps <= MAX_STEPS):
        raise ApiError(
            "INVALID_REQUEST",
            f"steps must be between {MIN_STEPS} and {MAX_STEPS}.",
            status_code=400,
        )


def _read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _runtime_seconds(started_at: str | None, finished_at: str | None) -> float | None:
    started = _parse_time(started_at)
    finished = _parse_time(finished_at)
    if not started or not finished:
        return None
    return max(0.0, round((finished - started).total_seconds(), 3))


def _artifact_if_exists(job_id: str, job_dir, name: str) -> str | None:
    path = job_dir / name
    if not path.exists():
        return None
    return build_artifact_url(job_id, name, get_settings().storage.public_outputs_prefix)


def _history_item_from_dir(job_dir) -> TryOnHistoryItem | None:
    job_path = job_dir / "job.json"
    if not job_path.exists():
        return None
    job_id = job_dir.name
    job_payload = _read_json(job_path)
    metadata = _read_json(job_dir / "metadata.json")
    generation_config = metadata.get("generation_config") or {}
    request_config = metadata.get("request_config") or {}
    engine_config = metadata.get("engine_config") or {}
    engine_settings = engine_config.get("engine") or {}
    return TryOnHistoryItem(
        job_id=job_id,
        status=job_payload.get("status", "failed"),
        created_at=job_payload.get("created_at"),
        started_at=job_payload.get("started_at"),
        finished_at=job_payload.get("finished_at"),
        runtime_seconds=_runtime_seconds(job_payload.get("started_at"), job_payload.get("finished_at")),
        current_stage=job_payload.get("current_stage"),
        stages=job_payload.get("stages") or [],
        result_url=job_payload.get("result_url") or _artifact_if_exists(job_id, job_dir, "result.png"),
        inputs=HistoryInputs(
            person_url=_artifact_if_exists(job_id, job_dir, "person.png"),
            garment_url=_artifact_if_exists(job_id, job_dir, "garment.png"),
            garment_top_url=_artifact_if_exists(job_id, job_dir, "garment_top.png"),
            garment_bottom_url=_artifact_if_exists(job_id, job_dir, "garment_bottom.png"),
            garment_dress_url=_artifact_if_exists(job_id, job_dir, "garment_dress.png"),
        ),
        config=GenerationConfigSummary(
            output_width=generation_config.get("output_width") or engine_settings.get("default_width"),
            output_height=generation_config.get("output_height") or engine_settings.get("default_height"),
            steps=generation_config.get("steps") or engine_settings.get("steps"),
            seed=metadata.get("seed") or job_payload.get("seed"),
            deterministic=request_config.get("deterministic")
            if "deterministic" in request_config
            else job_payload.get("deterministic"),
            engine=metadata.get("engine"),
            category=request_config.get("category") or metadata.get("category"),
            prompt=metadata.get("prompt"),
            use_refiner=request_config.get("use_refiner")
            if "use_refiner" in request_config
            else ((metadata.get("refiner_status") != "skipped") if metadata else None),
            repair_mode=request_config.get("repair_mode")
            if "repair_mode" in request_config
            else (bool((metadata.get("quality_report") or {}).get("repair")) if metadata else None),
        ),
        engine_status=job_payload.get("engine_status") or (metadata.get("engine_status") or {}),
        quality=job_payload.get("quality"),
        error=job_payload.get("error"),
    )


async def _read_upload(file: UploadFile | None):
    if file is None:
        return None
    settings = get_settings()
    max_bytes = settings.api.max_upload_mb * 1024 * 1024
    try:
        validate_mime(
            file.content_type,
            file.filename,
            allowed_mime_types=set(settings.api.allowed_image_mime_types),
        )
    except InputValidationError as exc:
        raise ApiError("INVALID_IMAGE", str(exc), status_code=415) from exc
    data = await file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ApiError(
            "FILE_TOO_LARGE",
            f"Image exceeds the {settings.api.max_upload_mb} MB upload limit.",
            status_code=413,
            details={"max_upload_mb": settings.api.max_upload_mb},
        )
    try:
        return load_image_from_bytes(data, max_side=settings.image.max_side)
    except InputValidationError as exc:
        raise ApiError("INVALID_IMAGE", "Uploaded file is not a valid image.", status_code=415) from exc


@router.post("/tryon", response_model=TryOnStatusResponse)
async def create_tryon(
    person_image: Annotated[UploadFile, File(...)],
    background_tasks: BackgroundTasks,
    garment_top: Annotated[UploadFile | None, File()] = None,
    garment_bottom: Annotated[UploadFile | None, File()] = None,
    garment_dress: Annotated[UploadFile | None, File()] = None,
    category: Annotated[TryOnCategory, Form()] = "upper_body",
    prompt: Annotated[str | None, Form()] = None,
    use_refiner: Annotated[bool, Form()] = True,
    repair_mode: Annotated[bool, Form()] = True,
    run_mode: Annotated[str | None, Form()] = None,
    engine_mode: Annotated[str | None, Form()] = None,
    seed: Annotated[int | None, Form()] = None,
    deterministic: Annotated[bool, Form()] = False,
    testcase_id: Annotated[str | None, Form()] = None,
    prompt_variant: Annotated[str, Form()] = "default",
    auto_prompt: Annotated[bool, Form()] = False,
    output_width: Annotated[int | None, Form()] = None,
    output_height: Annotated[int | None, Form()] = None,
    steps: Annotated[int | None, Form()] = None,
    save_intermediates: Annotated[bool | None, Form()] = None,
) -> TryOnStatusResponse:
    if not any([garment_top, garment_bottom, garment_dress]):
        raise ApiError("INVALID_REQUEST", "At least one garment image is required.", status_code=400)
    _validate_generation_overrides(output_width, output_height, steps)
    valid_engine_modes = {
        "idm_vton",
        "idm_mask_expanded",
        "idm_vton_flux",
        "idm_mask_expanded_flux",
        "klein_lora",
        "catvton",
    }
    if engine_mode and engine_mode not in valid_engine_modes:
        raise ApiError(
            "INVALID_REQUEST",
            "engine_mode must be one of: " + ", ".join(sorted(valid_engine_modes)) + ".",
            status_code=400,
        )
    valid_prompt_variants = {
        "default",
        "strong_remove_old_garment",
        "identity_strict",
        "accessory_stress",
        "flux_local_refine",
        "catvton_minimal",
        "adetailer_repair",
    }
    if prompt_variant not in valid_prompt_variants:
        raise ApiError("INVALID_REQUEST", "Unsupported prompt_variant.", status_code=400)
    if auto_prompt and not testcase_id:
        raise ApiError(
            "INVALID_REQUEST",
            "testcase_id is required when auto_prompt=true.",
            status_code=400,
        )

    person = await _read_upload(person_image)
    top = await _read_upload(garment_top)
    bottom = await _read_upload(garment_bottom)
    dress = await _read_upload(garment_dress)

    job_service = get_job_service()
    job_id = job_service.new_job_id()
    normalized_seed = normalize_seed(seed)
    request = PipelineRequest(
        job_id=job_id,
        person_image=person,
        garment_top=top,
        garment_bottom=bottom,
        garment_dress=dress,
        category=category,
        prompt=prompt,
        use_refiner=use_refiner,
        repair_mode=repair_mode,
        seed=normalized_seed,
        deterministic=deterministic,
        engine_mode=engine_mode,
        testcase_id=testcase_id,
        prompt_variant=prompt_variant,
        auto_prompt=auto_prompt,
        output_width=output_width,
        output_height=output_height,
        steps=steps,
        save_intermediates=save_intermediates,
    )
    settings = get_settings()
    selected_run_mode = (run_mode or settings.api.run_mode).lower()
    if selected_run_mode not in {"sync", "async"}:
        raise ApiError("INVALID_REQUEST", "run_mode must be 'sync' or 'async'.", status_code=400)
    if selected_run_mode == "async":
        queued = job_service.queue_tryon_job(request)
        background_tasks.add_task(job_service.run_queued_job, request)
        return queued
    return job_service.create_tryon_job(request)


@router.get("/tryon/history", response_model=TryOnHistoryResponse)
def list_tryon_history(limit: int = 20) -> TryOnHistoryResponse:
    settings = get_settings()
    bounded_limit = max(1, min(limit, 100))
    output_root = settings.storage.outputs_dir
    if not output_root.exists():
        return TryOnHistoryResponse(items=[])
    candidates = sorted(
        (path for path in output_root.iterdir() if path.is_dir() and (path / "job.json").exists()),
        key=lambda path: (path / "job.json").stat().st_mtime,
        reverse=True,
    )
    items: list[TryOnHistoryItem] = []
    for job_dir in candidates:
        item = _history_item_from_dir(job_dir)
        if item is not None:
            items.append(item)
        if len(items) >= bounded_limit:
            break
    return TryOnHistoryResponse(items=items)


@router.get("/tryon/{job_id}", response_model=TryOnStatusResponse)
def get_tryon(job_id: str) -> TryOnStatusResponse:
    job = get_job_service().get_job(job_id)
    if job is None:
        raise ApiError("JOB_NOT_FOUND", f"Job not found: {job_id}", status_code=404)
    return job


@router.delete("/tryon/{job_id}", response_model=TryOnStatusResponse)
def cancel_tryon(job_id: str) -> TryOnStatusResponse:
    job = get_job_service().cancel_job(job_id)
    if job is None:
        raise ApiError("JOB_NOT_FOUND", f"Job not found: {job_id}", status_code=404)
    return job


@router.post("/tryon/refine", response_model=RefineResponse)
async def refine_image(
    image: Annotated[UploadFile, File(...)],
    mask: Annotated[UploadFile | None, File()] = None,
    prompt: Annotated[str, Form()] = "Refine garment boundary while preserving identity, pose, face, and background.",
    seed: Annotated[int | None, Form()] = None,
) -> RefineResponse:
    settings = get_settings()
    storage = get_storage_service()
    job_id = get_job_service().new_job_id()
    job_dir = storage.job_dir(job_id)
    normalized_seed = normalize_seed(seed)
    set_seed(normalized_seed)

    base_image = await _read_upload(image)
    mask_image = await _read_upload(mask)

    save_image(base_image, job_dir / "refine_input.png")
    if mask_image:
        save_image(mask_image.convert("L"), job_dir / "refine_mask.png")

    refiner = create_refiner(settings)
    try:
        result = refiner.refine(base_image, mask_image, prompt, seed=normalized_seed)
    except ModelUnavailableError as exc:
        return RefineResponse(job_id=job_id, status="failed", error=str(exc), seed=normalized_seed)

    result_path = save_image(result.image, job_dir / "refined_output.png")
    storage.save_json(job_id, "metadata.json", {"prompt": prompt, "seed": normalized_seed, "metadata": result.metadata})
    return RefineResponse(
        job_id=job_id,
        status="completed",
        result_url=storage.public_url(result_path),
        seed=normalized_seed,
    )
