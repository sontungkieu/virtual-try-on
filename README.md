# Project_Phase2 - Virtual Try-On

This repository contains a ComfyUI-based virtual try-on experiment suite for 15 test cases. The current final demo compares four methods side by side:

1. SCHP/SAM + Flux Fill + CatVTON
2. FLUX.2 Klein 9B
3. FLUX.2 Klein 9B + Try-On LoRA
4. FLUX.2 Klein 9B + Try-On LoRA local masked inpaint

The project is designed to run model inference locally on a GPU machine such as RunPod. It does not require calling a hosted image API for the final pipelines.

Python dependencies are managed from `virtual_tryon/pyproject.toml` with `uv`; `virtual_tryon/uv.lock` is the reproducible lockfile. Use `uv add` for new Python dependencies. Real IDM-VTON runtime packages are in the optional `idm` extra, and TensorRT runtime packages are in the optional `tensorrt` extra.

## Current Status

The final output folder is:

```text
virtual_tryon/data/outputs/FINAL OUTPUT/
```

It contains 15 comparison grids:

```text
test_case_01_grid.png
...
test_case_15_grid.png
index.html
metadata.json
```

Older comparison grids are preserved in:

```text
virtual_tryon/data/outputs/old_final/
```

The extracted Klein + Try-On LoRA results are in:

```text
virtual_tryon/data/outputs/result of klein + lora/
```

## Refactored Layout

The final-demo configuration is centralized here:

```text
virtual_tryon/final_demo/config.py
```

This file defines:

- the 15 test-case garment plans;
- the normalized garment filenames;
- the four final methods;
- output paths for each method;
- shared prompt fragments;
- quality notes for each method.

The main final-demo scripts are:

```text
virtual_tryon/scripts/create_data_input_eval_set.py
virtual_tryon/scripts/generate_schp_sam_masks.py
virtual_tryon/scripts/run_schp_sam_mask_tryon_outputs.py
virtual_tryon/scripts/run_klein_global_final15.py
virtual_tryon/scripts/create_klein_local_masked_config_from_eval.py
virtual_tryon/scripts/run_klein_local_masked_tryon.py
virtual_tryon/scripts/build_final_output_4method_grids.py
virtual_tryon/scripts/validate_final_demo_outputs.py
```

Legacy/ablation scripts are kept for reproducibility but are not the main final path.

## Input Contract

The current input folder is:

```text
Data_input/
```

Expected structure:

```text
Data_input/
  Test case 1/
    Person.png
    Garment.png
  Test case 2/
    Person.png
    Garmet.png
  ...
  Test case 15/
    Person.png
    Garment1.png
    Garment2.png
    Garment3.png
```

The normalizer converts those folders into:

```text
virtual_tryon/data/temp/final15_data_input_eval_set/
  sample_001/
    person.png
    garment_top.png
    reference_canvas.png
    metadata.json
  ...
```

The runtime API supports these try-on categories:

- `upper_body`: top garment via `garment_top`.
- `lower_body`: outer lower garment via `garment_bottom`.
- `dress`: dress garment via `garment_dress`.
- `full_outfit`: dress, top, and/or bottom references.
- `men_underwear`: adult men's underwear bottom via `garment_bottom`.
- `women_underwear`: adult women's underwear bottom via `garment_bottom`.
- `women_bra`: adult women's bra or upper innerwear via `garment_top`.

The frontend and `/tryon` API can override output resolution and IDM-VTON inference steps per job, so fast preview runs can use lower resolution or fewer steps without changing YAML config. IDM-VTON can also run through the resident worker, which loads the model once and keeps it in GPU memory between jobs. Runtime optimization modes can be measured with `virtual_tryon/scripts/benchmark_idm_runtime_modes.py`; TensorRT defaults to the stable VAE-decode path, while UNet TensorRT experiments require the benchmark's explicit unsafe flag.

Try-on masks are anchored to a dynamic body estimate from the uploaded person image, then fall back to a conservative body box only when foreground estimation is unreliable. Adult innerwear categories use a dedicated anatomy-shaped mask path rather than the broad outerwear rectangles. `men_underwear` and `women_underwear` create a pelvis/brief-shaped mask with smaller dilation and blur. `women_bra` creates a cup/band/strap-shaped mask. Each job writes `mask_metadata.json`, `mask_innerwear_shape.png`, and, when available, `mask_body_silhouette.png` for debugging. Repeated jobs with the same person image, category, resolution, and mask config reuse cached masks from `data/temp/mask_cache`.

Single-item cases run as one pass. Multi-item cases are sequential:

- sample 11: top then bottom
- sample 12: bottom then hat
- sample 13: hat then accessory
- sample 14: top then bottom then hat
- sample 15: dress then shoes then hat

## Method 1 - SCHP/SAM + Flux Fill + CatVTON

**Goal:** mask-guided inpainting with a garment reference.

**Inputs per pass:**

- person image;
- one garment image;
- target region: upper, lower, dress, shoes, hat, accessory;
- SCHP/SAM processed mask;
- positive prompt for the target region;
- seed, steps, CFG, denoise.

**Pipeline:**

```text
person image
  -> SCHP/ATR human parsing
  -> SAM boundary refinement
  -> processed target mask
  -> Flux Fill inpaint conditioning
garment image
  -> Redux / reference conditioning
Flux Fill + CatVTON
  -> inpainted output
multi-item case
  -> output of pass N becomes person image for pass N+1
```

**Output path:**

```text
FINAL OUTPUT/method_01_schp_sam_flux_catvton/{sample_id}/final_output.png
```

**Quality:**

This is the most explicit inpainting baseline. It preserves identity and background well when the mask is correct. It fails when the mask is too large, too small, or semantically wrong. Hats, shoes, watches, and small accessories are the weakest targets.

## Method 2 - FLUX.2 Klein 9B

**Goal:** global Klein baseline.

**Inputs:**

- person canvas;
- reference garment/outfit canvas;
- target region inferred from metadata;
- default try-on prompt;
- seed, steps, guidance.

**Pipeline:**

```text
person image
  -> fit to 768x1024 canvas
garment/reference image
  -> fit to 768x1024 canvas
person lower crop
  -> bottom reference helper
Klein prompt builder
  -> default try-on prompt
FLUX.2 Klein 9B sampler
  -> full generated image
```

**Output path:**

```text
FINAL OUTPUT/method_02_klein9b/{sample_id}/output.png
```

**Quality:**

Klein 9B can generate visually clean fashion images, but it is not a strict try-on method. It has no hard mask, no garment warping constraint, and no identity lock. It can change pose, body shape, background, pants, shoes, or ignore the reference garment.

## Method 3 - FLUX.2 Klein 9B + Try-On LoRA

**Goal:** global Klein baseline with the fal Try-On LoRA.

**Inputs:**

- same as Method 2;
- Try-On LoRA checkpoint;
- LoRA strength, currently 1.0;
- default prompt strength.

**Pipeline:**

```text
Load FLUX.2 Klein 9B
  -> apply fal/flux-klein-9b-virtual-tryon-lora
person canvas + reference canvas + prompt
  -> Klein sampler
  -> full generated image
```

**Output path:**

```text
FINAL OUTPUT/method_03_klein9b_tryon_lora/{sample_id}/output.png
```

**Quality:**

The Try-On LoRA usually improves clothing intent compared with raw Klein, especially for broad top/dress edits. It still remains global generation. Without a mask, it can alter unrelated parts of the person or miss small items. This method replaced the older "Klein strong" column in the current final output.

## Method 4 - Klein + LoRA Local Masked Inpaint

**Goal:** production-oriented local edit using mask, crop, generate, paste-back.

**Inputs per pass:**

- person image;
- one garment image;
- target region;
- SCHP/SAM mask generated inside the custom ComfyUI node;
- local positive prompt;
- seed, steps, guidance, denoise;
- mask morphology settings;
- crop/paste-back settings.

**Pipeline:**

```text
person image
  -> SCHP/SAM mask node
  -> mask raw / processed / overlay
  -> bounding box crop around target region
garment image
  -> fitted garment reference
Klein 9B + Try-On LoRA
  -> generate local crop
generated crop + processed mask
  -> paste back into original image
  -> final output
```

**Output path:**

```text
FINAL OUTPUT/method_04_klein_lora_local_masked_inpaint/{sample_id}/final_output.png
```

**Quality:**

This is the best structured pipeline because it localizes the edit and keeps the original image outside the mask. It is also the easiest to debug because it saves masks, crop images, overlays, and paste-back outputs. Its main weakness is sensitivity to mask and crop quality. If the mask is too small, the output may remain almost unchanged. If the mask is too large, skin/body/background may be damaged.

This is not a true ADetailer implementation. It is an ADetailer-style local masked repair/inpaint pipeline.

## Running the Final 15-Case Experiment

These commands are intended for the RunPod layout:

```text
/workspace/Project_Phase2
/workspace/ComfyUI
/workspace/venvs/project_phase2
```

Use the shared RunPod venv through UV:

```bash
export VIRTUAL_ENV=/workspace/venvs/project_phase2
export PATH=/root/.local/bin:$VIRTUAL_ENV/bin:$PATH
```

Start ComfyUI:

```bash
cd /workspace/ComfyUI
nohup uv run --active --no-sync --no-project python main.py \
  --listen 127.0.0.1 \
  --port 8188 \
  >/tmp/comfyui_final_output.log 2>&1 &
```

Normalize `Data_input`:

```bash
cd /workspace/Project_Phase2
uv run --project virtual_tryon --active --no-sync python \
  virtual_tryon/scripts/create_data_input_eval_set.py \
  --source-root /workspace/Project_Phase2/Data_input \
  --output-root /workspace/Project_Phase2/virtual_tryon/data/temp/final15_data_input_eval_set
```

Generate SCHP/SAM masks:

```bash
uv run --project virtual_tryon --active --no-sync python \
  virtual_tryon/scripts/generate_schp_sam_masks.py \
  --eval-root /workspace/Project_Phase2/virtual_tryon/data/temp/final15_data_input_eval_set \
  --output-root "/workspace/Project_Phase2/virtual_tryon/data/outputs/FINAL OUTPUT/masks_schp_sam" \
  --temp-root /workspace/Project_Phase2/virtual_tryon/data/temp/final15_data_input_masks
```

Run Method 1:

```bash
uv run --project virtual_tryon --active --no-sync python \
  virtual_tryon/scripts/run_schp_sam_mask_tryon_outputs.py \
  --mask-meta "/workspace/Project_Phase2/virtual_tryon/data/outputs/FINAL OUTPUT/masks_schp_sam/metadata.json" \
  --eval-root /workspace/Project_Phase2/virtual_tryon/data/temp/final15_data_input_eval_set \
  --output-root "/workspace/Project_Phase2/virtual_tryon/data/outputs/FINAL OUTPUT/method_01_schp_sam_flux_catvton"
```

Run Method 2:

```bash
uv run --project virtual_tryon --active --no-sync python \
  virtual_tryon/scripts/run_klein_global_final15.py \
  --eval-root /workspace/Project_Phase2/virtual_tryon/data/temp/final15_data_input_eval_set \
  --output-root "/workspace/Project_Phase2/virtual_tryon/data/outputs/FINAL OUTPUT/method_02_klein9b" \
  --method base \
  --steps 28 \
  --guidance 2.5
```

Run Method 3:

```bash
uv run --project virtual_tryon --active --no-sync python \
  virtual_tryon/scripts/run_klein_global_final15.py \
  --eval-root /workspace/Project_Phase2/virtual_tryon/data/temp/final15_data_input_eval_set \
  --output-root "/workspace/Project_Phase2/virtual_tryon/data/outputs/FINAL OUTPUT/method_03_klein9b_tryon_lora" \
  --method lora \
  --steps 28 \
  --guidance 2.5 \
  --lora-strength 1.0
```

Create Method 4 config:

```bash
uv run --project virtual_tryon --active --no-sync python \
  virtual_tryon/scripts/create_klein_local_masked_config_from_eval.py \
  --eval-root /workspace/Project_Phase2/virtual_tryon/data/temp/final15_data_input_eval_set \
  --output-root "/workspace/Project_Phase2/virtual_tryon/data/outputs/FINAL OUTPUT/method_04_klein_lora_local_masked_inpaint" \
  --config-output /workspace/Project_Phase2/virtual_tryon/data/temp/final15_data_input_configs/klein_local_masked_config.json
```

Run Method 4:

```bash
uv run --project virtual_tryon --active --no-sync python \
  virtual_tryon/scripts/run_klein_local_masked_tryon.py \
  --config /workspace/Project_Phase2/virtual_tryon/data/temp/final15_data_input_configs/klein_local_masked_config.json
```

Build final grids:

```bash
uv run --project virtual_tryon --active --no-sync python \
  virtual_tryon/scripts/build_final_output_4method_grids.py \
  --eval-root /workspace/Project_Phase2/virtual_tryon/data/temp/final15_data_input_eval_set \
  --output-root "/workspace/Project_Phase2/virtual_tryon/data/outputs/FINAL OUTPUT"
```

Validate outputs:

```bash
uv run --project virtual_tryon --active --no-sync python \
  virtual_tryon/scripts/validate_final_demo_outputs.py \
  --output-dir "/workspace/Project_Phase2/virtual_tryon/data/outputs/FINAL OUTPUT"
```

## ComfyUI Workflows

Workflow JSONs are stored under:

```text
virtual_tryon/comfyui_workflows/
```

Important files:

```text
virtual_tryon/comfyui_workflows/pipelines_20260626/13_schp_sam_mask_consumer_single_pass_ui.json
virtual_tryon/comfyui_workflows/pipelines_20260626/15_schp_sam_inside_graph_clean_ui.json
virtual_tryon/comfyui_workflows/pipelines_20260626/11_klein_28_sample015_ui.json
virtual_tryon/comfyui_workflows/klein_detailed_pipelines_20260626/02_flux2_klein9b_lora_strong_detailed.workflow.json
virtual_tryon/comfyui_workflows/klein_detailed_pipelines_20260626/04_flux2_klein9b_lora_masked_local_inpaint.workflow.json
```

The JSON files are primarily visual/demo graphs. The batch runners patch inputs, references, masks, seeds, prompts, and output paths programmatically.

## Model Requirements

Required for the final demo:

```text
FLUX Fill / Flux inpaint model
FLUX Redux / reference conditioning model
CatVTON / CatVitOn LoRA or equivalent try-on adapter
FLUX.2 Klein 9B model directory with model_index.json
fal/flux-klein-9b-virtual-tryon-lora checkpoint
VAE and text encoders required by ComfyUI Flux nodes
SCHP/ATR parser checkpoints:
  virtual_tryon/models/idm_vton/ckpt/humanparsing/parsing_atr.onnx
  virtual_tryon/models/idm_vton/ckpt/humanparsing/parsing_lip.onnx
SAM checkpoint:
  virtual_tryon/models/sam/sam_vit_b_01ec64.pth
```

The final scripts validate the most critical model paths where possible and fail with explicit missing-file errors.

## Quality Notes

The current 4-method grid is useful as a comparison and ablation, not as a claim that all 15 outputs are production-perfect.

Observed quality pattern:

- Method 1 is usually best when masks are accurate.
- Method 2 is a useful global-generation baseline but is not reliable for strict try-on.
- Method 3 improves garment intent compared with Method 2 but can still drift.
- Method 4 has the strongest production structure but requires better masks/crops to avoid unchanged outputs or seams.

Common failure modes:

- target mask too small, causing no visible edit;
- target mask too large, damaging body or background;
- multi-item reference canvas confusing the global Klein methods;
- hats, shoes, watches, and tiny accessories being ignored;
- full-body generation changing pose, identity, or background;
- local paste-back seams at garment boundaries.

For a polished demo, use the grids to explain method behavior and select the best examples per method. For a product-quality pipeline, prioritize Method 4 and improve mask/crop/refine logic.

## Backend and Legacy API

The repository still includes the earlier FastAPI backend and IDM-VTON scaffold under:

```text
virtual_tryon/backend/
```

Basic backend test command:

```bash
cd virtual_tryon/backend
TRYON_ENGINE=mock pytest
```

That backend is separate from the current ComfyUI final-demo batch pipeline.

## Safety and Reproducibility

Do not commit:

- Hugging Face tokens;
- API keys;
- model checkpoints;
- large generated output folders;
- RunPod SSH keys.

Every final batch run should save:

- input person;
- garment/reference;
- masks and overlays where applicable;
- workflow JSON/API graph;
- metadata JSON;
- final output image.
