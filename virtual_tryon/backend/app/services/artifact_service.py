from __future__ import annotations

import json
from pathlib import Path, PurePosixPath


ALLOWED_ARTIFACT_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".json",
    ".csv",
    ".html",
    ".txt",
}
FORBIDDEN_ARTIFACT_EXTENSIONS = {
    ".env",
    ".py",
    ".pt",
    ".pth",
    ".pkl",
    ".safetensors",
    ".onnx",
}


def is_allowed_artifact(filename: str | Path) -> bool:
    suffix = Path(filename).suffix.lower()
    return suffix in ALLOWED_ARTIFACT_EXTENSIONS and suffix not in FORBIDDEN_ARTIFACT_EXTENSIONS


def build_artifact_url(job_id: str, filename: str | Path, prefix: str = "/artifacts") -> str:
    job_path = PurePosixPath(job_id)
    relative = PurePosixPath(str(filename).replace("\\", "/"))
    if (
        job_path.is_absolute()
        or relative.is_absolute()
        or ".." in job_path.parts
        or ".." in relative.parts
        or not job_path.parts
    ):
        raise ValueError("Invalid artifact path.")
    return f"{prefix.rstrip('/')}/{job_path.as_posix()}/{relative.as_posix()}"


def _artifact_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        return "image"
    if suffix == ".json":
        return "json"
    if suffix == ".csv":
        return "csv"
    if suffix == ".html":
        return "html"
    return "text"


def build_artifact_manifest(job_id: str, job_dir: Path, prefix: str = "/artifacts") -> dict:
    files: list[dict] = []
    if job_dir.exists():
        for path in sorted(candidate for candidate in job_dir.rglob("*") if candidate.is_file()):
            relative = path.relative_to(job_dir)
            if relative.as_posix() == "artifact_manifest.json" or not is_allowed_artifact(relative):
                continue
            files.append(
                {
                    "name": relative.as_posix(),
                    "url": build_artifact_url(job_id, relative, prefix),
                    "type": _artifact_type(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    return {"job_id": job_id, "files": files}


def write_artifact_manifest(job_id: str, job_dir: Path, prefix: str = "/artifacts") -> tuple[Path, dict]:
    manifest = build_artifact_manifest(job_id, job_dir, prefix)
    path = job_dir / "artifact_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path, manifest
