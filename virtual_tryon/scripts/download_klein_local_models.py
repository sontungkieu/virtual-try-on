from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ID = "black-forest-labs/FLUX.2-klein-9B"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "flux2-klein-9b"
DEFAULT_LORA_REPO = "fal/flux-klein-9b-virtual-tryon-lora"
DEFAULT_LORA_FILENAME = "flux-klein-tryon.safetensors"
DEFAULT_LORA_PATH = PROJECT_ROOT / "models" / "loras" / DEFAULT_LORA_FILENAME


def _hub_token_available() -> bool:
    if os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN"):
        return True
    try:
        from huggingface_hub import HfFolder

        return bool(HfFolder.get_token())
    except Exception:
        return False


def _snapshot_download(repo_id: str, local_dir: Path) -> Path:
    from huggingface_hub import snapshot_download

    local_dir.mkdir(parents=True, exist_ok=True)
    kwargs = {
        "repo_id": repo_id,
        "local_dir": local_dir,
        "resume_download": True,
    }
    try:
        return Path(snapshot_download(**kwargs, local_dir_use_symlinks=False))
    except TypeError:
        return Path(snapshot_download(**kwargs))


def _download_lora(repo_id: str, filename: str, target_path: Path) -> Path:
    from huggingface_hub import hf_hub_download

    target_path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {
        "repo_id": repo_id,
        "filename": filename,
        "local_dir": target_path.parent,
        "resume_download": True,
    }
    try:
        downloaded = Path(hf_hub_download(**kwargs, local_dir_use_symlinks=False))
    except TypeError:
        downloaded = Path(hf_hub_download(**kwargs))
    if downloaded != target_path:
        shutil.copy2(downloaded, target_path)
    return target_path


def _validate(model_dir: Path, lora_path: Path) -> dict:
    return {
        "model_dir": str(model_dir),
        "model_index": str(model_dir / "model_index.json"),
        "model_index_exists": (model_dir / "model_index.json").is_file(),
        "lora_path": str(lora_path),
        "lora_exists": lora_path.is_file(),
        "lora_size_mb": round(lora_path.stat().st_size / 1024 / 1024, 2) if lora_path.is_file() else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download local FLUX.2 Klein + Try-On LoRA assets.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--lora-repo", default=DEFAULT_LORA_REPO)
    parser.add_argument("--lora-filename", default=DEFAULT_LORA_FILENAME)
    parser.add_argument("--lora-path", type=Path, default=DEFAULT_LORA_PATH)
    parser.add_argument("--skip-model", action="store_true", help="Only validate/download the LoRA file.")
    parser.add_argument("--skip-lora", action="store_true", help="Only validate/download the base model.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    needs_model_download = not args.skip_model and not (args.model_dir / "model_index.json").is_file()
    needs_lora_download = not args.skip_lora and not args.lora_path.is_file()
    if (needs_model_download or needs_lora_download) and not _hub_token_available():
        print(
            "HF token is not configured. Accept the model license on Hugging Face, then login with "
            "`huggingface-cli login` or set HF_TOKEN/HUGGINGFACE_HUB_TOKEN.",
            file=sys.stderr,
        )
        return 2

    if needs_model_download:
        _snapshot_download(args.model_id, args.model_dir)
    if needs_lora_download:
        _download_lora(args.lora_repo, args.lora_filename, args.lora_path)

    payload = _validate(args.model_dir, args.lora_path)
    print(json.dumps(payload, indent=2))
    if not payload["model_index_exists"] and not args.skip_model:
        return 3
    if not payload["lora_exists"] and not args.skip_lora:
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
