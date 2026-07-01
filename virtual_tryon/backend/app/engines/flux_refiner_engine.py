from __future__ import annotations

import base64
import importlib.util
import io
import logging
import os
import time
from typing import Any

from PIL import Image

from app.core.config import EngineConfig
from app.engines.base import RefineResult
from app.utils.errors import ModelUnavailableError


logger = logging.getLogger(__name__)

LOCAL_BACKENDS = {"flux2_dev", "flux2_klein"}
API_BACKENDS = {"flux2_api", "fal_tryon_lora"}
VALID_BACKENDS = {"disabled", *LOCAL_BACKENDS, *API_BACKENDS}


class FluxRefinerEngine:
    name = "flux_refiner"

    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self._pipe = None
        self.last_status = "not_checked"

    def status(self) -> str:
        backend = self.config.backend or "disabled"
        if not self.config.enabled or backend == "disabled":
            return "unavailable: disabled"
        if backend not in VALID_BACKENDS:
            return f"unavailable: unknown backend '{backend}'"
        if backend in API_BACKENDS:
            return self._api_status()
        return self._local_status()

    def is_available(self) -> bool:
        self.last_status = self.status()
        return self.last_status == "available"

    def _api_status(self) -> str:
        if importlib.util.find_spec("requests") is None:
            return "unavailable: missing dependency requests"
        url_env = self.config.api_url_env or "FLUX_REFINER_API_URL"
        key_env = self.config.api_key_env or "FLUX_REFINER_API_KEY"
        if not os.getenv(url_env):
            return f"unavailable: missing API endpoint env {url_env}"
        if not os.getenv(key_env):
            return f"unavailable: missing API key env {key_env}"
        return "available"

    def _local_status(self) -> str:
        missing_deps = [
            dep
            for dep in ["torch", "diffusers", "transformers", "accelerate"]
            if importlib.util.find_spec(dep) is None
        ]
        if missing_deps:
            return "unavailable: missing dependency " + ", ".join(missing_deps)
        if not self._model_id():
            return "unavailable: missing model"
        return "available"

    def _model_id(self) -> str | None:
        if self.config.model_path and self.config.model_path.exists():
            return str(self.config.model_path)
        if self.config.checkpoint_dir and self.config.checkpoint_dir.exists() and any(self.config.checkpoint_dir.iterdir()):
            return str(self.config.checkpoint_dir)
        if self.config.backend == "flux2_klein" and self.config.base_model:
            return self.config.base_model
        return self.config.model_name

    def prepare(self) -> None:
        status = self.status()
        self.last_status = status
        if status != "available":
            raise ModelUnavailableError(f"FLUX refiner {status}")
        if self.config.backend in API_BACKENDS:
            return
        if self._pipe is not None:
            return

        try:
            import torch
            import diffusers
        except Exception as exc:
            raise ModelUnavailableError(f"FLUX refiner dependencies are not importable: {exc}") from exc

        model_id = self._model_id()
        if not model_id:
            raise ModelUnavailableError("FLUX refiner model_name/model_path/checkpoint_dir is not configured.")

        dtype = torch.bfloat16 if torch.cuda.is_available() and not self.config.quantized else torch.float32
        pipeline_cls = getattr(diffusers, "FluxFillPipeline", None) or getattr(diffusers, "FluxInpaintPipeline", None)
        if pipeline_cls is None:
            pipeline_cls = getattr(diffusers, "AutoPipelineForInpainting", None)
        if pipeline_cls is None:
            raise ModelUnavailableError(
                "FLUX refiner unavailable: missing dependency pipeline class in diffusers. "
                "Install a FLUX-compatible diffusers build or set flux_refiner.backend=disabled."
            )

        try:
            self._pipe = pipeline_cls.from_pretrained(model_id, torch_dtype=dtype)
            if torch.cuda.is_available():
                self._pipe = self._pipe.to("cuda")
            if hasattr(self._pipe, "enable_attention_slicing"):
                self._pipe.enable_attention_slicing()
        except Exception as exc:
            self._pipe = None
            message = self._classify_load_error(exc)
            raise ModelUnavailableError(message) from exc

    def _classify_load_error(self, exc: Exception) -> str:
        text = str(exc)
        lowered = text.lower()
        if any(marker in lowered for marker in ["gated", "401", "403", "private", "not a valid model identifier", "token"]):
            return (
                "FLUX refiner unavailable: license/access not accepted or model is private. "
                "Accept the model license and provide credentials through environment variables only."
            )
        if "out of memory" in lowered or "cuda" in lowered and "memory" in lowered:
            return "FLUX refiner unavailable: not enough VRAM to load the configured backend."
        return f"FLUX refiner failed to load model '{self._model_id()}': {text}"

    def refine(
        self,
        image: Image.Image,
        mask: Image.Image | None,
        prompt: str,
        references: dict | None = None,
        seed: int | None = None,
    ) -> RefineResult:
        if self.config.backend in API_BACKENDS:
            return self._refine_with_api(image, mask, prompt, references=references, seed=seed)
        return self._refine_with_diffusers(image, mask, prompt, references=references, seed=seed)

    def _refine_with_diffusers(
        self,
        image: Image.Image,
        mask: Image.Image | None,
        prompt: str,
        references: dict | None = None,
        seed: int | None = None,
    ) -> RefineResult:
        start = time.perf_counter()
        self.prepare()
        base = image.convert("RGB")
        mask_image = mask.convert("L") if mask is not None else Image.new("L", base.size, 255)

        try:
            import torch

            generator = None
            if seed is not None and torch.cuda.is_available():
                generator = torch.Generator("cuda").manual_seed(seed)
            elif seed is not None:
                generator = torch.Generator().manual_seed(seed)

            kwargs: dict[str, Any] = {
                "prompt": prompt,
                "image": base,
                "mask_image": mask_image,
                "num_inference_steps": self.config.steps,
                "guidance_scale": self.config.guidance_scale,
                "generator": generator,
            }
            if self.config.default_strength is not None:
                kwargs["strength"] = self.config.default_strength
            if references and references.get("garment") is not None:
                kwargs["reference_image"] = references["garment"]

            try:
                output = self._pipe(**kwargs)
            except TypeError:
                kwargs.pop("reference_image", None)
                output = self._pipe(**kwargs)
        except Exception as exc:
            raise ModelUnavailableError(f"FLUX refiner execution failed: {exc}") from exc

        refined_region = output.images[0].convert("RGB")
        result = self._paste_masked(base, refined_region, mask_image)
        elapsed = time.perf_counter() - start
        logger.info("FLUX refiner completed in %.2fs", elapsed)
        return RefineResult(
            result,
            {
                "engine": self.name,
                "backend": self.config.backend,
                "runtime_seconds": elapsed,
                "prompt": prompt,
                "seed": seed,
                "model": self._model_id(),
                "status": "success",
            },
        )

    def _refine_with_api(
        self,
        image: Image.Image,
        mask: Image.Image | None,
        prompt: str,
        references: dict | None = None,
        seed: int | None = None,
    ) -> RefineResult:
        start = time.perf_counter()
        self.prepare()
        import requests

        url = os.environ[self.config.api_url_env or "FLUX_REFINER_API_URL"]
        key = os.environ[self.config.api_key_env or "FLUX_REFINER_API_KEY"]
        base = image.convert("RGB")
        mask_image = mask.convert("L") if mask is not None else Image.new("L", base.size, 255)
        headers = {"Authorization": f"Bearer {key}"}
        files = {
            "image": ("core_output.png", self._image_bytes(base), "image/png"),
            "mask": ("safe_refine_mask.png", self._image_bytes(mask_image), "image/png"),
        }
        if references and references.get("person") is not None:
            files["person"] = ("person.png", self._image_bytes(references["person"].convert("RGB")), "image/png")
        if references and references.get("garment") is not None:
            files["garment"] = ("garment.png", self._image_bytes(references["garment"].convert("RGB")), "image/png")
        data = {
            "prompt": prompt,
            "seed": "" if seed is None else str(seed),
            "steps": str(self.config.steps),
            "guidance_scale": str(self.config.guidance_scale),
        }

        try:
            response = requests.post(url, headers=headers, data=data, files=files, timeout=600)
            response.raise_for_status()
        except Exception as exc:
            raise ModelUnavailableError(f"FLUX API refiner failed without exposing credentials: {exc}") from exc

        refined_region = self._image_from_api_response(response)
        result = self._paste_masked(base, refined_region, mask_image)
        elapsed = time.perf_counter() - start
        return RefineResult(
            result,
            {
                "engine": self.name,
                "backend": self.config.backend,
                "runtime_seconds": elapsed,
                "prompt": prompt,
                "seed": seed,
                "model": "api",
                "status": "success",
            },
        )

    @staticmethod
    def _image_bytes(image: Image.Image) -> bytes:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    @staticmethod
    def _image_from_api_response(response) -> Image.Image:
        content_type = response.headers.get("content-type", "")
        if content_type.startswith("image/"):
            return Image.open(io.BytesIO(response.content)).convert("RGB")
        payload = response.json()
        if payload.get("image_base64"):
            return Image.open(io.BytesIO(base64.b64decode(payload["image_base64"]))).convert("RGB")
        raise ModelUnavailableError("FLUX API refiner response did not contain an image.")

    @staticmethod
    def _paste_masked(base: Image.Image, refined_region: Image.Image, mask_image: Image.Image) -> Image.Image:
        if refined_region.size != base.size:
            refined_region = refined_region.resize(base.size, Image.Resampling.LANCZOS)
        out = base.copy()
        out.paste(refined_region, mask=mask_image)
        return out
