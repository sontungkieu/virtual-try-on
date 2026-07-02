from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from PIL import Image

from app.core.config import Settings
from app.core.paths import REPO_ROOT
from app.engines.base import TryOnInputs
from app.engines.factory import create_refiner, create_repair_engine, create_tryon_engine
from app.evaluation.quality_checks import build_quality_report, run_quality_checks
from app.preprocessing.agnostic_mask import AgnosticMaskResult, create_agnostic_mask
from app.preprocessing.densepose import DensePoseEstimator
from app.preprocessing.garment_segmenter import GarmentSegmenter
from app.preprocessing.human_parser import HumanParser
from app.preprocessing.image_loader import fit_to_canvas
from app.preprocessing.mask_utils import composite_masked, mask_area
from app.preprocessing.refine_mask import build_refine_masks, select_refine_mask
from app.prompts.prompt_builder import build_prompt
from app.prompts.prompt_types import EngineMode, PromptBuildResult, PromptVariant
from app.prompts.testcase_prompt_library import get_testcase
from app.schemas.tryon import (
    DebugUrls,
    INNERWEAR_BOTTOM_CATEGORIES,
    INNERWEAR_TOP_CATEGORIES,
    QualityScores,
    TryOnCategory,
    TryOnResponse,
)
from app.services.artifact_service import write_artifact_manifest
from app.services.storage_service import StorageService
from app.utils.errors import InputValidationError, ModelUnavailableError
from app.utils.image_io import save_image
from app.utils.seed import normalize_seed, set_seed


logger = logging.getLogger(__name__)


MASK_CACHE_VERSION = "mask-v6-innerwear-hand-protection"
MASK_CACHE_IMAGE_FILES = {
    "raw_mask": ("raw_mask.png", "L"),
    "dilated_mask": ("agnostic_mask.png", "L"),
    "soft_mask": ("soft_mask.png", "L"),
    "preview": ("mask_preview.png", "RGB"),
    "agnostic_image": ("agnostic.png", "RGB"),
    "body_silhouette_mask": ("mask_body_silhouette.png", "L"),
    "innerwear_shape_mask": ("mask_innerwear_shape.png", "L"),
    "original_upper_body_mask": ("mask_original_upper_body.png", "L"),
    "expanded_upper_body_mask": ("mask_expanded_upper_body.png", "L"),
    "diff_upper_body_mask": ("mask_diff_upper_body.png", "L"),
    "original_upper_body_overlay": ("mask_original_upper_body_overlay.png", "RGB"),
    "expanded_upper_body_overlay": ("mask_expanded_upper_body_overlay.png", "RGB"),
    "diff_upper_body_overlay": ("mask_diff_upper_body_overlay.png", "RGB"),
}


INNERWEAR_DEFAULT_PROMPTS = {
    "men_underwear": (
        "replace the masked lower-body clothing with the adult men's underwear from the reference garment; "
        "remove original shorts, pants, briefs, and old fabric patterns inside the target mask completely; "
        "preserve face, pose, body shape, upper-body clothing, legs outside the target region, and background"
    ),
    "women_underwear": (
        "replace the masked lower-body clothing with the adult women's underwear from the reference garment; "
        "remove original shorts, pants, briefs, and old fabric patterns inside the target mask completely; "
        "preserve face, pose, body shape, upper-body clothing, legs outside the target region, and background"
    ),
    "women_bra": (
        "replace only the adult women's bra or upper innerwear region with the reference bra garment; "
        "preserve face, hair, hands, body shape, lower-body clothing, abdomen outside the target region, and background"
    ),
}


@dataclass
class PipelineRequest:
    job_id: str
    person_image: Image.Image
    garment_top: Image.Image | None
    garment_bottom: Image.Image | None
    garment_dress: Image.Image | None
    category: TryOnCategory
    prompt: str | None
    use_refiner: bool
    repair_mode: bool
    seed: int | None = None
    deterministic: bool = False
    engine_mode: str | None = None
    testcase_id: str | None = None
    prompt_variant: str = "default"
    auto_prompt: bool = False
    output_width: int | None = None
    output_height: int | None = None
    steps: int | None = None
    save_intermediates: bool | None = None
    progress_callback: Callable[[str, str], None] | None = field(default=None, repr=False, compare=False)


class TryOnPipeline:
    def __init__(self, settings: Settings, storage: StorageService) -> None:
        self.settings = settings
        self.storage = storage
        self.segmenter = GarmentSegmenter()
        self.human_parser = HumanParser()
        self.densepose = DensePoseEstimator()

    @staticmethod
    def _emit_progress(request: PipelineRequest, stage: str, status: str) -> None:
        if request.progress_callback is None:
            return
        try:
            request.progress_callback(stage, status)
        except Exception:
            logger.warning("Progress callback failed for job %s stage %s=%s", request.job_id, stage, status)

    def _select_garment(self, request: PipelineRequest) -> Image.Image:
        if request.category in {"upper_body", *INNERWEAR_TOP_CATEGORIES} and request.garment_top:
            return request.garment_top
        if request.category in {"lower_body", *INNERWEAR_BOTTOM_CATEGORIES} and request.garment_bottom:
            return request.garment_bottom
        if request.category == "dress" and request.garment_dress:
            return request.garment_dress
        if request.category == "full_outfit":
            return request.garment_dress or request.garment_top or request.garment_bottom  # type: ignore[return-value]
        raise InputValidationError(f"No garment image provided for category '{request.category}'.")

    def _settings_for_request(self, request: PipelineRequest) -> Settings:
        settings = self.settings
        has_generation_overrides = any([request.output_width, request.output_height, request.steps])
        if not request.engine_mode and not has_generation_overrides:
            return settings
        mode = request.engine_mode
        mode_settings = settings.model_copy(deep=True)
        if request.output_width:
            mode_settings.image.output_width = request.output_width
            mode_settings.idm_vton.default_width = request.output_width
            mode_settings.klein_tryon_lora.default_width = request.output_width
        if request.output_height:
            mode_settings.image.output_height = request.output_height
            mode_settings.idm_vton.default_height = request.output_height
            mode_settings.klein_tryon_lora.default_height = request.output_height
        if request.steps:
            mode_settings.idm_vton.steps = request.steps
            mode_settings.klein_tryon_lora.num_inference_steps = request.steps
            mode_settings.klein_tryon_lora.steps = request.steps
        if not mode:
            return mode_settings
        if mode == "idm_vton":
            mode_settings.pipeline.engine = "mock" if settings.pipeline.engine == "mock" else "idm_vton"
        elif mode == "idm_mask_expanded":
            mode_settings.pipeline.engine = "mock" if settings.pipeline.engine == "mock" else "idm_vton"
            mode_settings.mask_experiments.upper_body_expand_hem.enabled = True
        elif mode == "idm_vton_flux":
            mode_settings.pipeline.engine = "mock" if settings.pipeline.engine == "mock" else "idm_vton"
            mode_settings.flux_refiner.enabled = True
            mode_settings.refinement.enabled = True
        elif mode == "idm_mask_expanded_flux":
            mode_settings.pipeline.engine = "mock" if settings.pipeline.engine == "mock" else "idm_vton"
            mode_settings.mask_experiments.upper_body_expand_hem.enabled = True
            mode_settings.flux_refiner.enabled = True
            mode_settings.refinement.enabled = True
        elif mode == "flux_redux_catvton":
            mode_settings.pipeline.engine = "mock" if settings.pipeline.engine == "mock" else "comfyui_flux_redux"
        elif mode == "klein_lora":
            mode_settings.pipeline.engine = "klein_tryon_lora"
            mode_settings.klein_tryon_lora.enabled = True
        elif mode == "catvton":
            mode_settings.pipeline.engine = "catvton"
        else:
            raise InputValidationError(f"Unsupported engine_mode '{mode}'.")
        return mode_settings

    @staticmethod
    def _prompt_engine_mode(request: PipelineRequest, settings: Settings) -> EngineMode:
        mapping = {
            "idm_vton": EngineMode.IDM,
            "idm_mask_expanded": EngineMode.IDM_MASK_EXPANDED,
            "idm_vton_flux": EngineMode.IDM_MASK_EXPANDED_FLUX,
            "idm_mask_expanded_flux": EngineMode.IDM_MASK_EXPANDED_FLUX,
            "flux_redux_catvton": EngineMode.IDM_MASK_EXPANDED_FLUX,
            "klein_lora": EngineMode.KLEIN_LORA,
            "catvton": EngineMode.CATVTON,
        }
        if request.engine_mode:
            return mapping[request.engine_mode]
        if settings.pipeline.engine == "klein_tryon_lora":
            return EngineMode.KLEIN_LORA
        if settings.pipeline.engine == "catvton":
            return EngineMode.CATVTON
        return EngineMode.IDM

    def _resolve_prompts(
        self,
        request: PipelineRequest,
        settings: Settings,
    ) -> tuple[str | None, str | None, PromptBuildResult | None]:
        engine_mode = self._prompt_engine_mode(request, settings)
        if request.auto_prompt:
            if not request.testcase_id:
                raise InputValidationError("testcase_id is required when auto_prompt=true.")
            try:
                testcase = get_testcase(request.testcase_id)
                variant = PromptVariant(request.prompt_variant)
            except (KeyError, ValueError) as exc:
                raise InputValidationError(str(exc)) from exc
            prompt_request = testcase.build_request(engine_mode, variant)
            if request.prompt:
                prompt_request.extra_user_instruction = request.prompt
            result = build_prompt(prompt_request)
            return result.core_prompt or result.positive_prompt, result.refine_prompt, result

        prompt = request.prompt
        if engine_mode == EngineMode.KLEIN_LORA and prompt:
            from app.engines.klein_prompt_builder import build_klein_tryon_prompt

            prompt = build_klein_tryon_prompt(
                None,
                None,
                None,
                request.category,
                extra_instruction=prompt,
            )
        if prompt is None and settings.pipeline.engine != "klein_tryon_lora":
            prompt = INNERWEAR_DEFAULT_PROMPTS.get(request.category, settings.refinement.default_prompt)
        return prompt, None, None

    def _save_prompt_artifacts(
        self,
        job_dir: Path,
        core_prompt: str | None,
        refine_prompt: str | None,
        prompt_result: PromptBuildResult | None,
    ) -> dict[str, Path | None]:
        paths: dict[str, Path | None] = {
            "core": None,
            "refine": None,
            "metadata": None,
        }
        if core_prompt:
            paths["core"] = job_dir / "prompt_core.txt"
            paths["core"].write_text(core_prompt, encoding="utf-8")
            (job_dir / "prompt.txt").write_text(core_prompt, encoding="utf-8")
        if refine_prompt:
            paths["refine"] = job_dir / "prompt_refine.txt"
            paths["refine"].write_text(refine_prompt, encoding="utf-8")
        if prompt_result:
            if prompt_result.negative_prompt:
                (job_dir / "negative_prompt.txt").write_text(
                    prompt_result.negative_prompt,
                    encoding="utf-8",
                )
            paths["metadata"] = job_dir / "prompt_metadata.json"
            paths["metadata"].write_text(
                json.dumps(prompt_result.model_dump(mode="json"), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        return paths

    def validate_inputs(self, request: PipelineRequest) -> None:
        if request.person_image is None:
            raise InputValidationError("person_image is required.")
        if not any([request.garment_top, request.garment_bottom, request.garment_dress]):
            raise InputValidationError("At least one garment image is required.")
        self._select_garment(request)

    @staticmethod
    def _config_hash(payload: dict) -> str:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _image_hash(image: Image.Image) -> str:
        normalized = image.convert("RGB")
        digest = hashlib.sha256()
        digest.update(f"{normalized.width}x{normalized.height}:RGB:".encode("utf-8"))
        digest.update(normalized.tobytes())
        return digest.hexdigest()

    @staticmethod
    def _open_cached_image(path: Path, mode: str) -> Image.Image | None:
        if not path.exists():
            return None
        with Image.open(path) as image:
            return image.convert(mode).copy()

    def _mask_cache_payload(
        self,
        *,
        original_person_hash: str,
        category: TryOnCategory,
        width: int,
        height: int,
        mask_config: dict,
    ) -> dict:
        return {
            "version": MASK_CACHE_VERSION,
            "original_person_hash": original_person_hash,
            "category": category,
            "output_width": width,
            "output_height": height,
            "mask_config": mask_config,
        }

    def _load_mask_cache(self, cache_dir: Path) -> tuple[AgnosticMaskResult, dict] | None:
        metadata_path = cache_dir / "metadata.json"
        if not metadata_path.exists():
            return None
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            raw_mask = self._open_cached_image(cache_dir / MASK_CACHE_IMAGE_FILES["raw_mask"][0], "L")
            dilated_mask = self._open_cached_image(cache_dir / MASK_CACHE_IMAGE_FILES["dilated_mask"][0], "L")
            soft_mask = self._open_cached_image(cache_dir / MASK_CACHE_IMAGE_FILES["soft_mask"][0], "L")
            preview = self._open_cached_image(cache_dir / MASK_CACHE_IMAGE_FILES["preview"][0], "RGB")
            agnostic_image = self._open_cached_image(cache_dir / MASK_CACHE_IMAGE_FILES["agnostic_image"][0], "RGB")
            if not all([raw_mask, dilated_mask, soft_mask, preview, agnostic_image]):
                return None
            optionals = {
                name: self._open_cached_image(cache_dir / filename, mode)
                for name, (filename, mode) in MASK_CACHE_IMAGE_FILES.items()
                if name not in {"raw_mask", "dilated_mask", "soft_mask", "preview", "agnostic_image"}
            }
            mask_metadata = metadata.get("mask_metadata", {})
            result = AgnosticMaskResult(
                raw_mask=raw_mask,  # type: ignore[arg-type]
                dilated_mask=dilated_mask,  # type: ignore[arg-type]
                soft_mask=soft_mask,  # type: ignore[arg-type]
                preview=preview,  # type: ignore[arg-type]
                agnostic_image=agnostic_image,  # type: ignore[arg-type]
                mask_source=mask_metadata.get("source", "cache"),
                mask_warnings=tuple(mask_metadata.get("warnings", [])),
                body_bbox_xyxy=tuple(mask_metadata["body_bbox_xyxy"]) if mask_metadata.get("body_bbox_xyxy") else None,
                body_silhouette_mask=optionals["body_silhouette_mask"],
                innerwear_shape_mask=optionals["innerwear_shape_mask"],
                original_upper_body_mask=optionals["original_upper_body_mask"],
                expanded_upper_body_mask=optionals["expanded_upper_body_mask"],
                diff_upper_body_mask=optionals["diff_upper_body_mask"],
                original_upper_body_overlay=optionals["original_upper_body_overlay"],
                expanded_upper_body_overlay=optionals["expanded_upper_body_overlay"],
                diff_upper_body_overlay=optionals["diff_upper_body_overlay"],
            )
            return result, metadata
        except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning("Ignoring corrupt mask cache at %s: %s", cache_dir, exc)
            return None

    @staticmethod
    def _mask_metadata(category: TryOnCategory, mask_result: AgnosticMaskResult) -> dict:
        return {
            "category": category,
            "source": mask_result.mask_source,
            "warnings": list(mask_result.mask_warnings),
            "body_bbox_xyxy": list(mask_result.body_bbox_xyxy) if mask_result.body_bbox_xyxy else None,
            "raw_mask_bbox_xyxy": list(mask_result.raw_mask.getbbox()) if mask_result.raw_mask.getbbox() else None,
            "raw_mask_area_px": mask_area(mask_result.raw_mask),
            "dilated_mask_area_px": mask_area(mask_result.dilated_mask),
            "soft_mask_area_px": mask_area(mask_result.soft_mask),
        }

    def _save_mask_cache(
        self,
        cache_dir: Path,
        mask_result: AgnosticMaskResult,
        *,
        cache_key: str,
        payload: dict,
        mask_metadata: dict,
    ) -> None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        images = {
            "raw_mask": mask_result.raw_mask,
            "dilated_mask": mask_result.dilated_mask,
            "soft_mask": mask_result.soft_mask,
            "preview": mask_result.preview,
            "agnostic_image": mask_result.agnostic_image,
            "body_silhouette_mask": mask_result.body_silhouette_mask,
            "innerwear_shape_mask": mask_result.innerwear_shape_mask,
            "original_upper_body_mask": mask_result.original_upper_body_mask,
            "expanded_upper_body_mask": mask_result.expanded_upper_body_mask,
            "diff_upper_body_mask": mask_result.diff_upper_body_mask,
            "original_upper_body_overlay": mask_result.original_upper_body_overlay,
            "expanded_upper_body_overlay": mask_result.expanded_upper_body_overlay,
            "diff_upper_body_overlay": mask_result.diff_upper_body_overlay,
        }
        for name, image in images.items():
            if image is None:
                continue
            filename, _ = MASK_CACHE_IMAGE_FILES[name]
            save_image(image, cache_dir / filename)
        (cache_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "cache_key": cache_key,
                    "payload": payload,
                    "mask_metadata": mask_metadata,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def _get_or_create_mask(
        self,
        *,
        person: Image.Image,
        original_person_hash: str,
        category: TryOnCategory,
        settings: Settings,
        mask_config: dict,
        mask_experiment,
    ) -> tuple[AgnosticMaskResult, dict, dict]:
        payload = self._mask_cache_payload(
            original_person_hash=original_person_hash,
            category=category,
            width=person.width,
            height=person.height,
            mask_config=mask_config,
        )
        cache_key = self._config_hash(payload)
        cache_dir = Path(settings.storage.temp_dir) / "mask_cache" / cache_key
        cache_info = {
            "enabled": settings.preprocessing.mask_cache_enabled,
            "hit": False,
            "key": cache_key,
            "path": str(cache_dir),
            "version": MASK_CACHE_VERSION,
        }
        if settings.preprocessing.mask_cache_enabled:
            cached = self._load_mask_cache(cache_dir)
            if cached is not None:
                result, cache_metadata = cached
                mask_metadata = dict(cache_metadata.get("mask_metadata", self._mask_metadata(category, result)))
                mask_metadata["cache"] = {**cache_info, "hit": True}
                return result, mask_metadata, {**cache_info, "hit": True}

        result = create_agnostic_mask(
            person,
            category,
            settings.preprocessing,
            mask_experiment,
        )
        mask_metadata = self._mask_metadata(category, result)
        if settings.preprocessing.mask_cache_enabled:
            self._save_mask_cache(
                cache_dir,
                result,
                cache_key=cache_key,
                payload=payload,
                mask_metadata=mask_metadata,
            )
        mask_metadata["cache"] = cache_info
        return result, mask_metadata, cache_info

    @staticmethod
    def _pad_bbox(
        box: tuple[int, int, int, int],
        size: tuple[int, int],
        *,
        x_ratio: float,
        y_ratio: float,
    ) -> tuple[int, int, int, int]:
        left, top, right, bottom = box
        width = right - left
        height = bottom - top
        pad_x = max(8, int(width * x_ratio))
        pad_y = max(8, int(height * y_ratio))
        image_w, image_h = size
        return (
            max(0, left - pad_x),
            max(0, top - pad_y),
            min(image_w, right + pad_x),
            min(image_h, bottom + pad_y),
        )

    def _prepare_garment_reference(
        self,
        garment: Image.Image,
        category: TryOnCategory,
        settings: Settings,
        mask_experiment,
        job_dir: Path,
    ) -> Image.Image:
        if category not in {*INNERWEAR_BOTTOM_CATEGORIES, *INNERWEAR_TOP_CATEGORIES}:
            return garment
        if not settings.preprocessing.innerwear_reference_crop_enabled:
            return garment

        try:
            reference_mask = create_agnostic_mask(
                garment,
                category,
                settings.preprocessing,
                mask_experiment,
            )
        except Exception as exc:
            logger.warning("Could not extract innerwear garment reference crop: %s", exc)
            return garment

        mask = reference_mask.soft_mask
        box = mask.getbbox()
        if box is None:
            return garment

        padded = self._pad_bbox(box, garment.size, x_ratio=0.18, y_ratio=0.16)
        crop = garment.crop(padded).convert("RGB")
        crop_mask = mask.crop(padded).convert("L")
        white = Image.new("RGB", crop.size, (255, 255, 255))
        extracted = Image.composite(crop, white, crop_mask)
        save_image(mask, job_dir / "garment_reference_mask.png")
        save_image(extracted, job_dir / "garment_reference_region.png")
        return extracted

    @staticmethod
    def _commit_sha() -> str:
        try:
            return subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return "unknown"

    def run(self, request: PipelineRequest) -> TryOnResponse:
        self.validate_inputs(request)
        settings = self._settings_for_request(request)
        seed = normalize_seed(request.seed)
        set_seed(seed, deterministic=request.deterministic)
        self._emit_progress(request, "running", "running")

        job_dir = self.storage.job_dir(request.job_id)
        width = settings.image.output_width
        height = settings.image.output_height
        save_intermediates = settings.pipeline.save_intermediates
        if request.save_intermediates is not None:
            save_intermediates = settings.pipeline.save_intermediates and request.save_intermediates
        original_person_hash = self._image_hash(request.person_image)
        prompt, refine_prompt, prompt_result = self._resolve_prompts(request, settings)
        prompt_paths = self._save_prompt_artifacts(job_dir, prompt, refine_prompt, prompt_result)

        person = fit_to_canvas(request.person_image, width, height)
        garment = fit_to_canvas(self._select_garment(request), width, height)
        garment_top = fit_to_canvas(request.garment_top, width, height) if request.garment_top else None
        garment_bottom = fit_to_canvas(request.garment_bottom, width, height) if request.garment_bottom else None
        garment_dress = fit_to_canvas(request.garment_dress, width, height) if request.garment_dress else None
        save_image(person, job_dir / "person.png")
        save_image(garment, job_dir / "garment.png")
        if garment_top is not None:
            save_image(garment_top, job_dir / "garment_top.png")
        if garment_bottom is not None:
            save_image(garment_bottom, job_dir / "garment_bottom.png")
        if garment_dress is not None:
            save_image(garment_dress, job_dir / "garment_dress.png")

        human_parse = self.human_parser.parse(person, job_dir)
        densepose = self.densepose.estimate(person, job_dir)
        mask_experiment = settings.mask_experiments.upper_body_expand_hem
        mask_config = {
            "preprocessing": settings.preprocessing.model_dump(mode="json"),
            "upper_body_expand_hem": mask_experiment.model_dump(mode="json"),
        }
        mask_result, mask_metadata, mask_cache_info = self._get_or_create_mask(
            person=person,
            original_person_hash=original_person_hash,
            category=request.category,
            settings=settings,
            mask_config=mask_config,
            mask_experiment=mask_experiment,
        )
        engine_garment = self._prepare_garment_reference(
            garment,
            request.category,
            settings,
            mask_experiment,
            job_dir,
        )
        if engine_garment is not garment:
            if save_intermediates:
                save_image(engine_garment, job_dir / "garment_engine_input.png")
        garment_seg = self.segmenter.segment(engine_garment, (width, height))

        if save_intermediates:
            save_image(mask_result.raw_mask, job_dir / "raw_mask.png")
        save_image(mask_result.dilated_mask, job_dir / "agnostic_mask.png")
        if save_intermediates:
            save_image(mask_result.soft_mask, job_dir / "soft_mask.png")
            save_image(mask_result.preview, job_dir / "mask_preview.png")
            save_image(mask_result.agnostic_image, job_dir / "agnostic.png")
            experiment_debug_images = {
                "mask_original_upper_body.png": mask_result.original_upper_body_mask,
                "mask_expanded_upper_body.png": mask_result.expanded_upper_body_mask,
                "mask_diff_upper_body.png": mask_result.diff_upper_body_mask,
                "mask_original_upper_body_overlay.png": mask_result.original_upper_body_overlay,
                "mask_expanded_upper_body_overlay.png": mask_result.expanded_upper_body_overlay,
                "mask_diff_upper_body_overlay.png": mask_result.diff_upper_body_overlay,
            }
            for filename, image in experiment_debug_images.items():
                if image is not None:
                    save_image(image, job_dir / filename)
            if mask_result.innerwear_shape_mask is not None:
                save_image(mask_result.innerwear_shape_mask, job_dir / "mask_innerwear_shape.png")
            if mask_result.body_silhouette_mask is not None:
                save_image(mask_result.body_silhouette_mask, job_dir / "mask_body_silhouette.png")
            save_image(garment_seg.cloth_mask, job_dir / "cloth_mask.png")
            save_image(garment_seg.normalized_crop, job_dir / "garment_normalized.png")
        if save_intermediates and densepose.densepose_path is None:
            save_image(person, job_dir / "densepose_placeholder.png")
        self._emit_progress(request, "running", "completed")

        engine = create_tryon_engine(settings)
        inputs = TryOnInputs(
            person_image=person,
            garment_image=garment_seg.normalized_crop,
            category=request.category,
            agnostic_mask=mask_result.soft_mask,
            agnostic_image=mask_result.agnostic_image,
            prompt=prompt,
            seed=seed,
            output_dir=job_dir,
            extra={
                "person_path": job_dir / "person.png",
                "garment_path": job_dir / "garment.png",
                "garment_top_image": garment_top,
                "garment_bottom_image": garment_bottom,
                "garment_dress_image": garment_dress,
                "garment_top_path": job_dir / "garment_top.png" if garment_top is not None else None,
                "garment_bottom_path": job_dir / "garment_bottom.png" if garment_bottom is not None else None,
                "garment_dress_path": job_dir / "garment_dress.png" if garment_dress is not None else None,
                "garment_engine_image": engine_garment,
                "job_id": request.job_id,
                "mask_path": job_dir / "agnostic_mask.png",
                "human_parse": human_parse.warning,
                "densepose": densepose.warning,
                "deterministic": request.deterministic,
            },
        )

        self._emit_progress(request, "generating", "running")
        try:
            core = engine.run(inputs)
        except Exception:
            self._emit_progress(request, "generating", "failed")
            raise
        self._emit_progress(request, "generating", "completed")

        raw_core_image = core.image.convert("RGB")
        raw_core_path = save_image(raw_core_image, job_dir / "core_output_raw.png")
        core_image = composite_masked(person, raw_core_image, mask_result.soft_mask)
        core_path = save_image(core_image, job_dir / "core_output.png")
        current_image = core_image

        refine_masks = build_refine_masks(person, mask_result.soft_mask, settings.refinement)
        if save_intermediates:
            save_image(refine_masks.garment_refine_mask, job_dir / "garment_refine_mask.png")
            save_image(refine_masks.boundary_refine_mask, job_dir / "boundary_refine_mask.png")
            save_image(refine_masks.safe_refine_mask, job_dir / "safe_refine_mask.png")
            save_image(refine_masks.garment_overlay, job_dir / "garment_refine_mask_overlay.png")
            save_image(refine_masks.boundary_overlay, job_dir / "boundary_refine_mask_overlay.png")
            save_image(refine_masks.safe_overlay, job_dir / "safe_refine_mask_overlay.png")
        active_refine_mask = select_refine_mask(refine_masks, settings.refinement.mask_mode)

        quality: QualityScores = run_quality_checks(
            person,
            core_image,
            garment_seg.normalized_crop,
            active_refine_mask,
            settings.quality,
        )

        refined_path: Path | None = None
        refined_image: Image.Image | None = None
        refine_notes = list(refine_masks.notes)
        core_engine_name = getattr(engine, "name", "unknown")
        engine_status = {
            "idm_vton": "success" if core_engine_name in {"idm_vton", "mock"} else "skipped",
            "flux_refiner": "skipped",
            "comfyui_flux_redux": "success" if core_engine_name == "comfyui_flux_redux" else "skipped",
            "catvton": "success" if core_engine_name == "catvton" else "skipped",
            "klein_lora": "success" if core_engine_name == "klein_tryon_lora" else "skipped",
        }
        refiner_status = "skipped"
        use_refiner = False if request.engine_mode == "flux_redux_catvton" else request.use_refiner or request.engine_mode in {
            "idm_vton_flux",
            "idm_mask_expanded_flux",
        }
        if use_refiner and settings.refinement.enabled and settings.flux_refiner.enabled:
            self._emit_progress(request, "refining", "running")
            try:
                refiner = create_refiner(settings)
                refined = refiner.refine(
                    core_image,
                    active_refine_mask,
                    refine_prompt or prompt,
                    references={"person": person, "garment": garment_seg.normalized_crop},
                    seed=seed,
                )
                refined_image = refined.image
                refined_path = save_image(refined_image, job_dir / "refined_output.png")
                refiner_status = "success"
                engine_status["flux_refiner"] = "success"
                self._emit_progress(request, "refining", "completed")
            except ModelUnavailableError as exc:
                message = f"Refiner unavailable; returning core output. {exc}"
                quality.notes.append(message)
                refine_notes.append(message)
                (job_dir / "flux_refiner_error.txt").write_text(message, encoding="utf-8")
                refiner_status = "skipped"
                engine_status["flux_refiner"] = "skipped"
                self._emit_progress(request, "refining", "skipped")
                logger.warning("Skipping refiner: %s", exc)
            except Exception as exc:
                message = f"Refiner failed; returning core output. {exc}"
                quality.notes.append(message)
                refine_notes.append(message)
                (job_dir / "flux_refiner_error.txt").write_text(message, encoding="utf-8")
                refiner_status = "failed"
                engine_status["flux_refiner"] = "failed"
                self._emit_progress(request, "refining", "failed")
                logger.exception("Refiner failed; falling back to core output.")
        else:
            self._emit_progress(request, "refining", "skipped")

        quality_report = build_quality_report(
            person,
            core_image,
            refined_image,
            active_refine_mask,
            settings.quality,
            refine_notes=refine_notes,
            engine_status=engine_status,
        )
        if quality_report["final_choice"] == "refined" and refined_image is not None:
            current_image = refined_image
        else:
            current_image = core_image

        if request.repair_mode and settings.repair.enabled and refined_image is not None and quality_report["final_choice"] == "refined":
            repair_engine = create_repair_engine(settings)
            repaired = repair_engine.refine(current_image, active_refine_mask, prompt, seed=seed)
            current_image = repaired.image
            refined_path = save_image(current_image, job_dir / "refined_output.png")
            quality_report["repair"] = repaired.metadata

        self._emit_progress(request, "completed", "running")
        result_path = save_image(current_image, job_dir / "result.png")
        quality_report_path = self.storage.save_json(request.job_id, "quality_report.json", quality_report)
        mask_metadata_path = self.storage.save_json(request.job_id, "mask_metadata.json", mask_metadata)
        active_engine_settings = (
            settings.idm_vton
            if settings.pipeline.engine in {"idm_vton", "mock", "comfyui_flux_redux"}
            else getattr(settings, settings.pipeline.engine)
        )
        engine_config = {
            "pipeline_engine": settings.pipeline.engine,
            "runtime": settings.runtime.model_dump(mode="json"),
            "engine": active_engine_settings.model_dump(mode="json"),
        }
        metadata = {
            "job_id": request.job_id,
            "seed": seed,
            "generation_config": {
                "output_width": width,
                "output_height": height,
                "steps": active_engine_settings.num_inference_steps or active_engine_settings.steps,
                "requested_output_width": request.output_width,
                "requested_output_height": request.output_height,
                "requested_steps": request.steps,
            },
            "request_config": {
                "category": request.category,
                "engine_mode": request.engine_mode,
                "use_refiner": request.use_refiner,
                "repair_mode": request.repair_mode,
                "deterministic": request.deterministic,
                "save_intermediates": save_intermediates,
                "auto_prompt": request.auto_prompt,
                "prompt_variant": request.prompt_variant,
                "testcase_id": request.testcase_id,
            },
            "mask_config_hash": self._config_hash(mask_config),
            "engine_config_hash": self._config_hash(engine_config),
            "mask_config": mask_config,
            "mask_metadata": mask_metadata,
            "mask_cache": mask_cache_info,
            "engine_config": engine_config,
            "commit_sha": self._commit_sha(),
            "category": request.category,
            "engine": getattr(engine, "name", "unknown"),
            "prompt": prompt,
            "prompt_variant": request.prompt_variant,
            "testcase_id": request.testcase_id,
            "auto_prompt": request.auto_prompt,
            "prompt_hash": prompt_result.metadata.get("prompt_hash") if prompt_result else None,
            "quality": quality.model_dump(),
            "quality_report": quality_report,
            "refiner_status": refiner_status,
            "engine_status": engine_status,
            "core_metadata": {
                **core.metadata,
                "masked_composite": True,
                "raw_core_output": str(raw_core_path),
            },
        }
        self.storage.save_json(request.job_id, "metadata.json", metadata)
        write_artifact_manifest(
            request.job_id,
            job_dir,
            settings.storage.public_outputs_prefix,
        )

        return TryOnResponse(
            job_id=request.job_id,
            status="completed",
            result_url=self.storage.public_url(result_path),
            debug=DebugUrls(
                mask_url=self.storage.public_url(job_dir / "mask_preview.png"),
                mask_urls=[
                    url
                    for url in [
                        self.storage.public_url(job_dir / "mask_preview.png"),
                        self.storage.public_url(job_dir / "garment_refine_mask_overlay.png"),
                        self.storage.public_url(job_dir / "boundary_refine_mask_overlay.png"),
                        self.storage.public_url(job_dir / "safe_refine_mask_overlay.png"),
                        self.storage.public_url(job_dir / "mask_innerwear_shape.png"),
                        self.storage.public_url(job_dir / "mask_body_silhouette.png"),
                    ]
                    if url
                ],
                agnostic_url=self.storage.public_url(job_dir / "agnostic.png"),
                core_output_url=self.storage.public_url(core_path),
                refined_output_url=self.storage.public_url(refined_path),
                quality_report_url=self.storage.public_url(quality_report_path),
                refine_mask_url=self.storage.public_url(job_dir / "safe_refine_mask_overlay.png"),
                mask_metadata_url=self.storage.public_url(mask_metadata_path),
                prompt_core_url=self.storage.public_url(prompt_paths["core"]),
                prompt_refine_url=self.storage.public_url(prompt_paths["refine"]),
                prompt_metadata_url=self.storage.public_url(prompt_paths["metadata"]),
            ),
            quality=quality,
            seed=seed,
            deterministic=request.deterministic,
        )
