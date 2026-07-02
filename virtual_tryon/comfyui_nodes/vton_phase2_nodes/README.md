# VTON Phase2 ComfyUI Nodes

These nodes expose the Phase 2 Virtual Try-On experiment methods inside ComfyUI.

## Nodes

| Node | Purpose |
|---|---|
| `VTON Phase2 - IDM / IDM Expanded` | Runs IDM-VTON with standard or expanded agnostic mask. |
| `VTON Phase2 - Klein Reference Set` | Builds the three local Klein conditioning images: person, top reference, bottom reference. |
| `VTON Phase2 - Klein Local Sampler` | Runs local `FLUX.2-klein-9B` plus Try-On LoRA with 4 or 28 steps. |
| `VTON Phase2 - Klein Fit Canvas` | Resizes/pads one image to the Klein target canvas. |
| `VTON Phase2 - Klein Bottom Preserve Crop` | Crops the lower/body-preservation reference from the person image. |
| `VTON Phase2 - Klein Prompt Builder` | Builds default or strong Klein prompts from a target region and item description. |
| `VTON Phase2 - Load FLUX.2 Klein 9B` | Loads the local `black-forest-labs/FLUX.2-klein-9B` Diffusers pipeline. |
| `VTON Phase2 - Load Klein Try-On LoRA` | Applies `fal/flux-klein-9b-virtual-tryon-lora` to the loaded Klein pipeline. |
| `VTON Phase2 - Klein Detailed Sampler` | Runs sampling from explicit person/top/bottom references and prompt. |
| `VTON Phase2 - SCHP/SAM Mask` | Runs the IDM-VTON ATR/SCHP-style human parser inside ComfyUI, optionally refines the target mask with SAM, then outputs raw mask, processed mask, and overlay. |
| `Add Mask For IC Lora` | Compatibility node for Flux Fill + Redux + CatVTON LoRA in-context workflows; packs garment and person images with the target mask on the person panel. |

## Method Mapping

| Report method | ComfyUI node setup |
|---|---|
| `IDM` | `VTON Phase2 - IDM / IDM Expanded`, `mask_expanded=false` |
| `IDM expanded` | `VTON Phase2 - IDM / IDM Expanded`, `mask_expanded=true` |
| `Klein 4-step` | `VTON Phase2 - Klein Reference Set` -> `VTON Phase2 - Klein Local Sampler`, `method=klein_4step` |
| `Klein 28` | `VTON Phase2 - Klein Reference Set` -> `VTON Phase2 - Klein Local Sampler`, `method=klein_28` |
| `Klein 28 strong` | `VTON Phase2 - Klein Reference Set` -> `VTON Phase2 - Klein Local Sampler`, `method=klein_28_strong` |
| `FLUX.2 Klein 9B strong` | `Fit Canvas` + `Bottom Preserve Crop` + `Prompt Builder` + `Load FLUX.2 Klein 9B` + `Detailed Sampler` |
| `FLUX.2 Klein 9B + Try-On LoRA strong` | Same graph plus `Load Klein Try-On LoRA` before `Detailed Sampler` |
| `SCHP/SAM inside graph` | `VTON Phase2 - SCHP/SAM Mask` -> `AddMaskForICLora` -> Flux Fill + Redux + CatVTON LoRA inpaint graph |

## Runtime

Set these environment variables before starting ComfyUI if the defaults do not match the RunPod layout:

```bash
export VTON_PROJECT_ROOT=/workspace/Project_Phase2/virtual_tryon
export VTON_KLEIN_MODEL_DIR=/workspace/Project_Phase2/virtual_tryon/models/flux2-klein-9b
export VTON_KLEIN_LORA_PATH=/workspace/hf-cache/hub/models--fal--flux-klein-9b-virtual-tryon-lora/snapshots/8b078b15c6d958ce48892b9ef31b66aa7587d792/flux-klein-tryon.safetensors
export VTON_COMFY_OUTPUT_ROOT=/workspace/Project_Phase2/virtual_tryon/data/outputs/comfyui_runs
```

The Klein node runs locally with `local_files_only=True`. It does not call FAL, Nano Banana, or any external inference API.

The SCHP/SAM node expects these local files on RunPod:

```bash
/workspace/Project_Phase2/virtual_tryon/models/idm_vton/ckpt/humanparsing/parsing_atr.onnx
/workspace/Project_Phase2/virtual_tryon/models/idm_vton/ckpt/humanparsing/parsing_lip.onnx
/workspace/Project_Phase2/virtual_tryon/models/sam/sam_vit_b_01ec64.pth
```

The Flux Fill + Redux in-context workflow additionally expects these local ComfyUI models:

```bash
/workspace/ComfyUI/models/unet/FLUX1/fluxFillFP8_v10.safetensors
/workspace/ComfyUI/models/style_models/flux1-redux-dev.safetensors
/workspace/ComfyUI/models/clip/clip_l.safetensors
/workspace/ComfyUI/models/clip/t5xxl_fp8_e4m3fn.safetensors
/workspace/ComfyUI/models/clip_vision/sigclip_vision_patch14_384.safetensors
/workspace/ComfyUI/models/vae/FLUX1/ae.safetensors
/workspace/ComfyUI/models/loras/flux/catvton-flux-lora.safetensors
```
