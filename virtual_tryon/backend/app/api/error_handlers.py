from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.utils.errors import ApiError


logger = logging.getLogger(__name__)


def _payload(code: str, message: str, details: dict | None = None) -> dict:
    return {"error": {"code": code, "message": message, "details": details or {}}}


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def api_error_handler(_request: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=_payload(exc.code, exc.message, exc.details))

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=_payload("INVALID_REQUEST", "Request validation failed.", {"errors": exc.errors()}),
        )

    @app.exception_handler(HTTPException)
    async def http_error_handler(_request: Request, exc: HTTPException) -> JSONResponse:
        code = {
            404: "JOB_NOT_FOUND",
            413: "FILE_TOO_LARGE",
            415: "INVALID_IMAGE",
        }.get(exc.status_code, "INVALID_REQUEST")
        message = exc.detail if isinstance(exc.detail, str) else "Request failed."
        details = {} if isinstance(exc.detail, str) else {"detail": exc.detail}
        return JSONResponse(status_code=exc.status_code, content=_payload(code, message, details))

    @app.exception_handler(Exception)
    async def internal_error_handler(_request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled API error")
        return JSONResponse(
            status_code=500,
            content=_payload("INTERNAL_ERROR", "An internal error occurred.", {}),
        )
