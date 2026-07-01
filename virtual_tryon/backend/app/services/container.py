from __future__ import annotations

from functools import lru_cache

from app.core.config import get_settings
from app.services.job_service import JobService
from app.services.storage_service import StorageService
from app.services.tryon_pipeline import TryOnPipeline


@lru_cache(maxsize=1)
def get_storage_service() -> StorageService:
    return StorageService(get_settings().storage)


@lru_cache(maxsize=1)
def get_tryon_pipeline() -> TryOnPipeline:
    return TryOnPipeline(get_settings(), get_storage_service())


@lru_cache(maxsize=1)
def get_job_service() -> JobService:
    return JobService(get_tryon_pipeline(), get_storage_service())


def clear_container_cache() -> None:
    get_storage_service.cache_clear()
    get_tryon_pipeline.cache_clear()
    get_job_service.cache_clear()
