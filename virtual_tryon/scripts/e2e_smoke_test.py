from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _bool_text(value: str | bool) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return "true" if value.lower() in {"1", "true", "yes", "on"} else "false"


def _load_sample(sample_dir: Path) -> tuple[dict[str, Any], dict[str, tuple[str, bytes, str]]]:
    metadata = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))
    files: dict[str, tuple[str, bytes, str]] = {
        "person_image": ("person.jpg", (sample_dir / "person.jpg").read_bytes(), "image/jpeg")
    }
    for field, name in [
        ("garment_top", "garment_top.jpg"),
        ("garment_bottom", "garment_bottom.jpg"),
        ("garment_dress", "garment_dress.jpg"),
    ]:
        path = sample_dir / name
        if path.exists():
            files[field] = (name, path.read_bytes(), "image/jpeg")
    return metadata, files


def _absolute_url(api_base: str, url: str) -> str:
    if url.startswith("http"):
        return url
    return api_base.rstrip("/") + url


def run_smoke(
    *,
    api_base: str,
    sample_dir: Path,
    use_refiner: bool,
    timeout: int,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    owns_client = client is None
    client = client or httpx.Client(timeout=60)
    started = time.time()
    try:
        health = client.get(f"{api_base.rstrip('/')}/health")
        health.raise_for_status()
        metadata, files = _load_sample(sample_dir)
        data = {
            "category": metadata["category"],
            "prompt": "replace the shirt with the reference garment, preserve face, pose, and body shape",
            "use_refiner": _bool_text(use_refiner),
            "repair_mode": "false",
            "run_mode": "async",
        }
        response = client.post(f"{api_base.rstrip('/')}/tryon", data=data, files=files)
        response.raise_for_status()
        payload = response.json()
        job_id = payload["job_id"]
        while payload["status"] in {"queued", "running"}:
            if time.time() - started > timeout:
                raise TimeoutError(f"Timed out waiting for job {job_id}")
            time.sleep(2)
            poll = client.get(f"{api_base.rstrip('/')}/tryon/{job_id}")
            poll.raise_for_status()
            payload = poll.json()
        if payload["status"] != "completed":
            raise RuntimeError(f"Job {job_id} ended with status={payload['status']} error={payload.get('error')}")
        if not payload.get("result_url"):
            raise RuntimeError("Completed job did not return result_url")
        result_response = client.get(_absolute_url(api_base, payload["result_url"]))
        result_response.raise_for_status()
        quality_url = (payload.get("debug") or {}).get("quality_report_url")
        if not quality_url:
            raise RuntimeError("Completed job did not return debug.quality_report_url")
        quality_response = client.get(_absolute_url(api_base, quality_url))
        quality_response.raise_for_status()
        return {
            "status": "passed",
            "job_id": job_id,
            "result_url": payload["result_url"],
            "quality_report_url": quality_url,
            "runtime_seconds": round(time.time() - started, 3),
        }
    finally:
        if owns_client:
            client.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an end-to-end upload/generate/poll/artifact smoke test.")
    parser.add_argument("--api-base", default="http://127.0.0.1:8000")
    parser.add_argument("--sample", default=str(PROJECT_ROOT / "data" / "eval_set" / "sample_001"))
    parser.add_argument("--use-refiner", default="false")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--report-path", default=str(PROJECT_ROOT / "data" / "outputs" / "e2e_smoke_report.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = run_smoke(
            api_base=args.api_base,
            sample_dir=Path(args.sample),
            use_refiner=_bool_text(args.use_refiner) == "true",
            timeout=args.timeout,
        )
    except Exception as exc:
        report = {"status": "failed", "error": str(exc)}
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 1

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
