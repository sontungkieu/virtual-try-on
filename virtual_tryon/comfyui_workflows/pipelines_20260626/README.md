# ComfyUI Pipeline Workflows

This folder packages the virtual try-on directions used in Phase 2 into importable ComfyUI workflow JSON files.

## Pipelines

| File | Type | Pipeline | Input contract |
|---|---|---|---|
| `00_video_repo_flux_redux_ui_source.json` | UI workflow | Exact workflow copied from `fahdmirza/comfyuiworkflows` when available | Open in ComfyUI UI, then patch image/prompt widgets |
| `01_flux_redux_catvton_single_pass_api.json` | API workflow | Flux Fill + Redux garment reference + CatVTON LoRA + mask inpaint | `person.png`, `garment.png`, `mask.png` |
| `02_schp_sam_mask_consumer_single_pass_api.json` | API workflow | Same try-on graph, intended to consume SCHP/SAM/manual processed masks | `person.png`, `garment.png`, `mask_processed.png` |
| `13_schp_sam_mask_consumer_single_pass_ui.json` | UI workflow | Visual version of the SCHP/SAM mask consumer graph | `person.png`, `garment.png`, `mask_processed.png` |
| `14_schp_sam_inside_graph_single_pass_ui.json` | UI workflow | Runs SCHP/SAM mask generation inside ComfyUI before Flux Fill + Redux + CatVTON LoRA | `person.png`, `garment.png`, target region widget |
| `15_schp_sam_inside_graph_clean_ui.json` | UI workflow | Same as file 14, but with a cleaner left-to-right layout for inspection/demo | `person.png`, `garment.png`, target region widget |
| `03_multipass_sample015_pass01_dress_api.json` | API workflow | Multi-pass item 1, dress only | `sample015_person.png`, `sample015_dress.png`, `sample015_dress_mask.png` |
| `04_multipass_sample015_pass02_shoes_api.json` | API workflow | Multi-pass item 2, shoes only | previous pass output as `sample015_after_dress.png`, shoes image, shoes mask |
| `05_multipass_sample015_pass03_hat_api.json` | API workflow | Multi-pass item 3, hat only | previous pass output as `sample015_after_shoes.png`, hat image, hat mask |
| `06_refine_low_denoise_api.json` | API workflow | Optional border/detail refine pass | `base_output.png`, `garment.png`, `artifact_or_edge_mask.png` |
| `07_production_fallback_flux_fill_redux_catvton_api.json` | API workflow | Production fallback graph without IC-LoRA image packing | `person.png`, `garment.png`, `mask.png` |
| `10_klein_4step_sample015_ui.json` | UI workflow | Local Klein 4-step fast preset | custom Klein nodes |
| `11_klein_28_sample015_ui.json` | UI workflow | Local Klein 28-step preset | custom Klein nodes |
| `12_klein_28_strong_sample015_ui.json` | UI workflow | Local Klein 28-step stronger prompt preset | custom Klein nodes |

## Important Contract

The Flux Redux + CatVTON workflows are single-garment passes. Do not put dress, shoes, and hat into one reference canvas. For a full outfit, run sequentially:

1. dress mask + dress garment
2. shoes mask + shoes garment, using pass 1 output as the new person image
3. hat mask + hat garment, using pass 2 output as the new person image

The SCHP/SAM pipeline generates masks outside ComfyUI and these workflows consume the resulting mask PNGs. Rectangle masks are debug-only.

## Required ComfyUI Models

Paths are relative to `/workspace/ComfyUI`:

```text
models/unet/FLUX1/fluxFillFP8_v10.safetensors
models/style_models/flux1-redux-dev.safetensors
models/clip_vision/sigclip_vision_patch14_384.safetensors
models/loras/flux/catvton-flux-lora.safetensors
models/vae/FLUX1/ae.safetensors
models/clip/clip_l.safetensors
models/clip/t5xxl_fp8_e4m3fn.safetensors
```

## Required Custom Nodes

The video-equivalent Flux workflow uses `AddMaskForICLora`, `GrowMask`, `ImageCrop`, Flux nodes, Redux style model nodes, and standard ComfyUI loaders/samplers.

The Klein workflows use the local Phase 2 custom nodes:

```text
VTON Phase2 - Klein Reference Set
VTON Phase2 - Klein Local Sampler
```
