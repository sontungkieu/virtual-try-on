from __future__ import annotations


class TryOnError(Exception):
    """Base exception for recoverable try-on failures."""


class InputValidationError(TryOnError):
    """Raised when user input is missing or incompatible."""


class ModelUnavailableError(TryOnError):
    """Raised when a required model checkpoint or dependency is missing."""


class EngineExecutionError(TryOnError):
    """Raised when a configured engine fails during execution."""


class ApiError(TryOnError):
    def __init__(self, code: str, message: str, *, status_code: int = 400, details: dict | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}


class QueueFullError(ApiError):
    def __init__(self, message: str = "The inference queue is full.") -> None:
        super().__init__("QUEUE_FULL", message, status_code=429)


class JobTimeoutError(TryOnError):
    """Raised when a completed attempt exceeded the configured runtime limit."""
