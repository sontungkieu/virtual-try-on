from __future__ import annotations

import json
import os
import shutil
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from PIL import Image

from app.core.config import Settings
from app.engines.base import TryOnInputs, TryOnResult
from app.utils.errors import ModelUnavailableError


@dataclass(frozen=True)
class _ComfyPaths:
    url: str = "http://127.0.0.1:8188"
    input_dir: Path = Path("/workspace/ComfyUI/input")
    output_dir: Path = Path("/workspace/ComfyUI/output")
    models_dir: Path = Path("/workspace/ComfyUI/models")

    @classmethod
    def from_env(cls) -> "_ComfyPaths":
        return cls(
            url=os.getenv("TRYON_COMFYUI_URL", cls.url).rstrip("/"),
            input_dir=Path(os.getenv("TRYON_COMFYUI_INPUT_DIR", str(cls.input_dir))),
            output_dir=Path(os.getenv("TRYON_COMFYUI_OUTPUT_DIR", str(cls.output_dir))),
            models_dir=Path(os.getenv("TRYON_COMFYUI_MODELS_DIR", str(cls.models_dir))),
        )


class ComfyUIFluxReduxEngine:
    name = "comfyui_flux_redux"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.paths = _ComfyPaths.from_env()
        self.timeout_seconds = int(os.getenv("TRYON_COMFYUI_TIMEOUT_SECONDS", "900"))
        self.redux_strength = float(os.getenv("TRYON_FLUX_REDUX_STRENGTH", "1.0"))
        self.lora_strength = float(os.getenv("TRYON_FLUX_CATVTON_LORA_STRENGTH", "1.0"))
        self.denoise = float(os.getenv("TRYON_FLUX_REDUX_DENOISE", "1.0"))
        self.mask_grow_px = int(os.getenv("TRYON_FLUX_REDUX_MASK_GROW", "4"))
        self.graph_mode = os.getenv("TRYON_FLUX_REDUX_GRAPH", "fallback").lower()

    def status(self) -> str:
        missing = self._missing_model_files()
        if missing:
            return "unavailable: missing " + ", ".join(missing)
        try:
            self._get_json("/system_stats", timeout=5)
        except Exception as exc:
            return f"unavailable: ComfyUI is not reachable at {self.paths.url}: {exc}"
        return "available"

    def is_available(self) -> bool:
        return self.status() == "available"

    def prepare(self) -> None:
        status = self.status()
        if status != "available":
            raise ModelUnavailableError(f"ComfyUI Flux Redux {status}")

    def run(self, inputs: TryOnInputs) -> TryOnResult:
        self.prepare()
        output_dir = inputs.output_dir or Path("data/outputs/comfyui_flux_redux") / uuid4().hex
        output_dir.mkdir(parents=True, exist_ok=True)
        run_id = f"{inputs.extra.get('job_id') or output_dir.name}_{uuid4().hex[:8]}"

        person_path = output_dir / "comfyui_flux_person.png"
        garment_path = output_dir / "comfyui_flux_garment.png"
        mask_path = output_dir / "comfyui_flux_mask.png"
        inputs.person_image.convert("RGB").save(person_path)
        garment = inputs.extra.get("garment_engine_image") or inputs.garment_image
        garment.convert("RGB").save(garment_path)
        inputs.agnostic_mask.convert("L").save(mask_path)

        person_name = self._copy_to_input(person_path, f"{run_id}_person.png")
        garment_name = self._copy_to_input(garment_path, f"{run_id}_garment.png")
        mask_name = self._copy_to_input(mask_path, f"{run_id}_mask.png")
        filename_prefix = f"vton_flux_redux_ui/{run_id}/result"

        prompt = self._build_prompt(
            person_name=person_name,
            garment_name=garment_name,
            mask_name=mask_name,
            positive_prompt=self._positive_prompt(inputs.prompt, inputs.category),
            negative_prompt=self._negative_prompt(),
            seed=int(inputs.seed or 0),
            steps=int(self.settings.idm_vton.steps or 8),
            guidance=float(os.getenv("TRYON_FLUX_REDUX_GUIDANCE", "3.0")),
            filename_prefix=filename_prefix,
        )
        (output_dir / "comfyui_flux_redux_workflow.json").write_text(json.dumps(prompt, indent=2), encoding="utf-8")

        started = time.perf_counter()
        history = self._queue_and_wait(prompt)
        elapsed = time.perf_counter() - started
        (output_dir / "comfyui_flux_redux_history.json").write_text(
            json.dumps(history, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if not history.get("status", {}).get("completed"):
            raise ModelUnavailableError(f"ComfyUI Flux Redux failed: {json.dumps(history.get('status', {}))[:1000]}")

        saved = self._extract_saved_image(history)
        local_output = output_dir / "comfyui_flux_redux_output.png"
        shutil.copy2(saved, local_output)
        with Image.open(local_output) as image:
            result = image.convert("RGB").copy()
        return TryOnResult(
            result,
            {
                "engine": self.name,
                "backend": "comfyui",
                "runtime_seconds": elapsed,
                "comfyui_url": self.paths.url,
                "workflow": str(output_dir / "comfyui_flux_redux_workflow.json"),
                "history": str(output_dir / "comfyui_flux_redux_history.json"),
                "raw_output": str(local_output),
                "redux_strength": self.redux_strength,
                "lora_strength": self.lora_strength,
                "denoise": self.denoise,
                "mask_grow_px": self.mask_grow_px,
                "graph_mode": self.graph_mode,
            },
        )

    def _missing_model_files(self) -> list[str]:
        required = {
            "FLUX Fill": "unet/FLUX1/fluxFillFP8_v10.safetensors",
            "Redux": "style_models/flux1-redux-dev.safetensors",
            "SigCLIP": "clip_vision/sigclip_vision_patch14_384.safetensors",
            "CatVTON LoRA": "loras/flux/catvton-flux-lora.safetensors",
            "VAE": "vae/FLUX1/ae.safetensors",
            "CLIP-L": "clip/clip_l.safetensors",
            "T5": "clip/t5xxl_fp8_e4m3fn.safetensors",
        }
        return [label for label, rel_path in required.items() if not (self.paths.models_dir / rel_path).exists()]

    def _copy_to_input(self, source: Path, name: str) -> str:
        self.paths.input_dir.mkdir(parents=True, exist_ok=True)
        target = self.paths.input_dir / name
        shutil.copy2(source, target)
        return name

    def _post_json(self, path: str, payload: dict[str, Any], timeout: int = 30) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.paths.url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def _get_json(self, path: str, timeout: int = 30) -> dict[str, Any]:
        with urllib.request.urlopen(f"{self.paths.url}{path}", timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def _queue_and_wait(self, prompt: dict[str, Any]) -> dict[str, Any]:
        prompt_id = self._post_json("/prompt", {"prompt": prompt})["prompt_id"]
        deadline = time.time() + self.timeout_seconds
        while time.time() < deadline:
            time.sleep(2)
            history = self._get_json(f"/history/{prompt_id}", timeout=30)
            if prompt_id not in history:
                continue
            item = history[prompt_id]
            status = item.get("status", {})
            if status.get("completed") or status.get("status_str") == "error":
                return item
        raise TimeoutError(f"ComfyUI prompt timed out: {prompt_id}")

    def _extract_saved_image(self, history_item: dict[str, Any]) -> Path:
        outputs = history_item.get("outputs", {})
        for output in outputs.values():
            images = output.get("images")
            if not images:
                continue
            image = images[0]
            subfolder = image.get("subfolder") or ""
            return self.paths.output_dir / subfolder / image["filename"]
        raise ModelUnavailableError(f"No saved image in ComfyUI history: {json.dumps(history_item)[:1000]}")

    def _build_prompt(
        self,
        *,
        person_name: str,
        garment_name: str,
        mask_name: str,
        positive_prompt: str,
        negative_prompt: str,
        seed: int,
        steps: int,
        guidance: float,
        filename_prefix: str,
    ) -> dict[str, Any]:
        if self.graph_mode != "iclora":
            return {
                "1": {"class_type": "LoadImage", "inputs": {"image": person_name}},
                "2": {"class_type": "LoadImage", "inputs": {"image": mask_name}},
                "3": {"class_type": "ImageToMask", "inputs": {"image": ["2", 0], "channel": "red"}},
                "4": {"class_type": "LoadImage", "inputs": {"image": garment_name}},
                "5": {
                    "class_type": "DualCLIPLoader",
                    "inputs": {"clip_name1": "clip_l.safetensors", "clip_name2": "t5xxl_fp8_e4m3fn.safetensors", "type": "flux"},
                },
                "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["5", 0], "text": positive_prompt}},
                "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["5", 0], "text": negative_prompt}},
                "8": {"class_type": "FluxGuidance", "inputs": {"conditioning": ["6", 0], "guidance": guidance}},
                "9": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": "sigclip_vision_patch14_384.safetensors"}},
                "10": {"class_type": "CLIPVisionEncode", "inputs": {"clip_vision": ["9", 0], "image": ["4", 0], "crop": "none"}},
                "11": {"class_type": "StyleModelLoader", "inputs": {"style_model_name": "flux1-redux-dev.safetensors"}},
                "12": {
                    "class_type": "StyleModelApply",
                    "inputs": {
                        "conditioning": ["8", 0],
                        "style_model": ["11", 0],
                        "clip_vision_output": ["10", 0],
                        "strength": self.redux_strength,
                        "strength_type": "multiply",
                    },
                },
                "13": {"class_type": "VAELoader", "inputs": {"vae_name": "FLUX1/ae.safetensors"}},
                "14": {
                    "class_type": "UNETLoader",
                    "inputs": {"unet_name": "FLUX1/fluxFillFP8_v10.safetensors", "weight_dtype": "default"},
                },
                "15": {
                    "class_type": "LoraLoaderModelOnly",
                    "inputs": {
                        "model": ["14", 0],
                        "lora_name": "flux/catvton-flux-lora.safetensors",
                        "strength_model": self.lora_strength,
                    },
                },
                "16": {
                    "class_type": "InpaintModelConditioning",
                    "inputs": {
                        "positive": ["12", 0],
                        "negative": ["7", 0],
                        "vae": ["13", 0],
                        "pixels": ["1", 0],
                        "mask": ["3", 0],
                        "noise_mask": True,
                    },
                },
                "17": {
                    "class_type": "KSampler",
                    "inputs": {
                        "model": ["15", 0],
                        "seed": seed,
                        "steps": steps,
                        "cfg": 1,
                        "sampler_name": "euler",
                        "scheduler": "normal",
                        "positive": ["16", 0],
                        "negative": ["16", 1],
                        "latent_image": ["16", 2],
                        "denoise": self.denoise,
                    },
                },
                "18": {"class_type": "VAEDecode", "inputs": {"samples": ["17", 0], "vae": ["13", 0]}},
                "19": {"class_type": "SaveImage", "inputs": {"images": ["18", 0], "filename_prefix": filename_prefix}},
            }
        return {
            "1": {"class_type": "LoadImage", "inputs": {"image": garment_name}},
            "2": {"class_type": "LoadImage", "inputs": {"image": person_name}},
            "3": {"class_type": "LoadImageMask", "inputs": {"image": mask_name, "channel": "red"}},
            "4": {"class_type": "GrowMask", "inputs": {"mask": ["3", 0], "expand": self.mask_grow_px, "tapered_corners": True}},
            "5": {
                "class_type": "AddMaskForICLora",
                "inputs": {
                    "first_image": ["1", 0],
                    "second_image": ["2", 0],
                    "second_mask": ["4", 0],
                    "patch_mode": "auto",
                    "output_length": 1536,
                    "patch_color": "#FF0000",
                },
            },
            "6": {
                "class_type": "DualCLIPLoader",
                "inputs": {"clip_name1": "clip_l.safetensors", "clip_name2": "t5xxl_fp8_e4m3fn.safetensors", "type": "flux"},
            },
            "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["6", 0], "text": positive_prompt}},
            "8": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["6", 0], "text": negative_prompt}},
            "9": {"class_type": "FluxGuidance", "inputs": {"conditioning": ["7", 0], "guidance": guidance}},
            "10": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": "sigclip_vision_patch14_384.safetensors"}},
            "11": {"class_type": "CLIPVisionEncode", "inputs": {"clip_vision": ["10", 0], "image": ["1", 0], "crop": "none"}},
            "12": {"class_type": "StyleModelLoader", "inputs": {"style_model_name": "flux1-redux-dev.safetensors"}},
            "13": {
                "class_type": "StyleModelApply",
                "inputs": {
                    "conditioning": ["9", 0],
                    "style_model": ["12", 0],
                    "clip_vision_output": ["11", 0],
                    "strength": self.redux_strength,
                    "strength_type": "multiply",
                },
            },
            "14": {"class_type": "VAELoader", "inputs": {"vae_name": "FLUX1/ae.safetensors"}},
            "15": {
                "class_type": "UNETLoader",
                "inputs": {"unet_name": "FLUX1/fluxFillFP8_v10.safetensors", "weight_dtype": "default"},
            },
            "16": {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {"model": ["15", 0], "lora_name": "flux/catvton-flux-lora.safetensors", "strength_model": self.lora_strength},
            },
            "17": {
                "class_type": "InpaintModelConditioning",
                "inputs": {
                    "positive": ["13", 0],
                    "negative": ["8", 0],
                    "vae": ["14", 0],
                    "pixels": ["5", 0],
                    "mask": ["5", 1],
                    "noise_mask": True,
                },
            },
            "18": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["16", 0],
                    "seed": seed,
                    "steps": steps,
                    "cfg": 1,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "positive": ["17", 0],
                    "negative": ["17", 1],
                    "latent_image": ["17", 2],
                    "denoise": self.denoise,
                },
            },
            "19": {"class_type": "VAEDecode", "inputs": {"samples": ["18", 0], "vae": ["14", 0]}},
            "20": {"class_type": "ImageCrop", "inputs": {"image": ["19", 0], "x": ["5", 2], "y": ["5", 3], "width": ["5", 4], "height": ["5", 5]}},
            "21": {"class_type": "SaveImage", "inputs": {"images": ["20", 0], "filename_prefix": filename_prefix}},
        }

    @staticmethod
    def _default_prompt(category: str) -> str:
        return (
            f"Adult non-sexual virtual try-on for {category}. Replace only the masked target garment region "
            "with the reference garment. Preserve face, identity, pose, skin outside the mask, lighting, and background."
        )

    @staticmethod
    def _positive_prompt(prompt: str | None, category: str) -> str:
        base = prompt or ComfyUIFluxReduxEngine._default_prompt(category)
        if category in {"men_underwear", "women_underwear", "women_bra"}:
            return (
                f"{base}. Render the reference innerwear as a real sharp garment with crisp fabric edges, "
                "clear printed details, and no censor mosaic, no privacy blur, no pixelated patch. "
                "Keep this adult non-sexual product try-on fully clothed in the target garment."
            )
        return base

    @staticmethod
    def _negative_prompt() -> str:
        return (
            "low quality, blurry, distorted anatomy, extra limbs, changed face, changed background, text artifacts, "
            "watermark, uncovered body outside garment, edits outside the mask, censor blur, mosaic, pixelated patch, "
            "smudged garment, privacy blur"
        )
