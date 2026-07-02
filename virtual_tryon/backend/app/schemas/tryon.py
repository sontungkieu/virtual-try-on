from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


TryOnCategory = Literal[
    "upper_body",
    "lower_body",
    "dress",
    "full_outfit",
    "men_underwear",
    "women_underwear",
    "women_bra",
]
INNERWEAR_BOTTOM_CATEGORIES = {"men_underwear", "women_underwear"}
INNERWEAR_TOP_CATEGORIES = {"women_bra"}
JobStatus = Literal["queued", "running", "completed", "failed", "cancelled", "cancel_requested"]
StageStatus = Literal["pending", "running", "completed", "skipped", "failed", "cancelled"]


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: dict = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: ErrorDetail


class DebugUrls(BaseModel):
    mask_url: str | None = None
    mask_urls: list[str] = Field(default_factory=list)
    agnostic_url: str | None = None
    core_output_url: str | None = None
    refined_output_url: str | None = None
    quality_report_url: str | None = None
    refine_mask_url: str | None = None
    mask_metadata_url: str | None = None
    prompt_core_url: str | None = None
    prompt_refine_url: str | None = None
    prompt_metadata_url: str | None = None


class QualityScores(BaseModel):
    identity_score: float | None = None
    garment_similarity_score: float | None = None
    background_preservation_score: float | None = None
    artifact_score: float | None = None
    needs_refine: bool = False
    notes: list[str] = Field(default_factory=list)


class PipelineStage(BaseModel):
    key: str
    label: str
    status: StageStatus = "pending"
    started_at: str | None = None
    finished_at: str | None = None
    runtime_seconds: float | None = None


class TryOnResponse(BaseModel):
    job_id: str
    status: JobStatus
    result_url: str | None = None
    debug: DebugUrls = Field(default_factory=DebugUrls)
    quality: QualityScores | None = None
    error: str | None = None
    seed: int | None = None
    deterministic: bool | None = None


class TryOnStatusResponse(TryOnResponse):
    created_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    updated_at: str | None = None
    current_stage: str | None = None
    stages: list[PipelineStage] = Field(default_factory=list)
    cancel_requested: bool = False
    engine_status: dict[str, str] = Field(default_factory=dict)
    artifact_manifest: dict | None = None
    error_code: str | None = None
    retry_count: int = 0


class HistoryInputs(BaseModel):
    person_url: str | None = None
    garment_url: str | None = None
    garment_top_url: str | None = None
    garment_bottom_url: str | None = None
    garment_dress_url: str | None = None


class GenerationConfigSummary(BaseModel):
    output_width: int | None = None
    output_height: int | None = None
    steps: int | None = None
    seed: int | None = None
    deterministic: bool | None = None
    engine: str | None = None
    category: str | None = None
    prompt: str | None = None
    use_refiner: bool | None = None
    repair_mode: bool | None = None


class TryOnHistoryItem(BaseModel):
    job_id: str
    status: JobStatus
    created_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    runtime_seconds: float | None = None
    current_stage: str | None = None
    stages: list[PipelineStage] = Field(default_factory=list)
    result_url: str | None = None
    inputs: HistoryInputs = Field(default_factory=HistoryInputs)
    config: GenerationConfigSummary = Field(default_factory=GenerationConfigSummary)
    engine_status: dict[str, str] = Field(default_factory=dict)
    quality: QualityScores | None = None
    error: str | None = None


class TryOnHistoryResponse(BaseModel):
    items: list[TryOnHistoryItem] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    device: str
    models: dict[str, str]
    active_engine: str | None = None
    active_engine_mode: str | None = None
    loaded_engine: str | None = None
    loaded_engine_mode: str | None = None
    loaded_model: dict | None = None
    default_engine_mode: str = "klein_bnb_4bit"


class ModelPrepareRequest(BaseModel):
    engine_mode: str | None = None


class ModelPrepareResponse(BaseModel):
    status: str
    engine: str
    engine_mode: str | None = None
    runtime_seconds: float | None = None
    metadata: dict = Field(default_factory=dict)
    message: str | None = None


class RefineResponse(BaseModel):
    job_id: str
    status: JobStatus
    result_url: str | None = None
    error: str | None = None
    seed: int | None = None
