# Klein LoRA Real Eval

Klein Try-On LoRA is an experimental baseline. IDM-VTON remains the default core engine until manual evaluation across several samples proves the LoRA path is safer.

## Runtime Check

Install the local Klein dependency set in a worker venv and validate model assets:

```bash
/root/.local/bin/uv venv /workspace/venvs/project_phase2_klein --python 3.11
/root/.local/bin/uv pip install --python /workspace/venvs/project_phase2_klein/bin/python \
  "diffusers @ git+https://github.com/huggingface/diffusers.git" \
  "transformers>=4.56" "accelerate>=1.0" "peft>=0.17" "safetensors>=0.4" pillow numpy
export TRYON_KLEIN_PYTHON=/workspace/venvs/project_phase2_klein/bin/python
$TRYON_KLEIN_PYTHON scripts/download_klein_local_models.py
```

The downloader uses a Hugging Face token from `huggingface-cli login`,
`HF_TOKEN`, or `HUGGINGFACE_HUB_TOKEN` without printing it. Accept the
`black-forest-labs/FLUX.2-klein-9B` license before downloading. Do not commit
tokens, `.env`, model weights, or generated output images.

Expected validation JSON includes:

```json
{
  "model_index_exists": true,
  "lora_exists": true
}
```

For the optional fal.ai backend only, set `TRYON_KLEIN_BACKEND=fal_api`,
`FAL_KEY`, and run `uv run python scripts/check_fal_runtime.py --strict`.

## Real Ablation Command

Run:

```bash
uv run python scripts/run_klein_lora_ablation.py \
  --sample data/eval_set/sample_001 \
  --seed 42 \
  --bottom-strategy crop_from_person \
  --output data/outputs/klein_lora_ablation_real
```

Expected structure:

```text
data/outputs/klein_lora_ablation_real/
  comparison_grid.png
  comparison_index.html
  summary.csv
  summary.json
  manual_ratings_klein_lora.csv
  sample_001/
    idm_original/
      result.png
    klein_lora_default/
      result.png
      prompt.txt
      auto_bottom_reference.png
      status.json
      request_sanitized.json
      local_generation_sanitized.json
    klein_lora_strong_remove_old_shirt/
      result.png
      prompt.txt
      auto_bottom_reference.png
      status.json
      request_sanitized.json
      local_generation_sanitized.json
```

If local model files or dependencies are missing, the script still writes the summary, grid, index, prompt files, auto bottom references, and manual rating template. Klein rows are marked `unavailable`.

## Secret Scan

Before opening or sharing generated artifacts, scan the output folder:

```bash
uv run python scripts/scan_outputs_for_secrets.py \
  --path data/outputs/klein_lora_ablation_real \
  --patterns FAL_KEY HF_TOKEN HUGGINGFACE_HUB_TOKEN Authorization Bearer token= key=
```

The scanner reports only file names, line numbers, and matched pattern names. It does not print matched secret values. If a finding appears, delete or redact the generated output and fix the sanitizer before continuing review.

## Reading The Grid

Open `comparison_grid.png` or `comparison_index.html`.

Compare:

- Person input.
- Top garment reference.
- Auto bottom reference cropped from the person image.
- IDM original.
- Klein LoRA default prompt.
- Klein LoRA strong remove-old-shirt prompt.

The grid is for visual review only. Subjective observations belong in `manual_ratings_klein_lora.csv`, not in `quality_report.json`.

## Manual Ratings

Fill only after looking at the images:

```text
sample_id,variant,output_path,prompt_path,identity_1_5,garment_fidelity_1_5,old_garment_removed_1_5,realism_1_5,pose_preservation_1_5,body_shape_preservation_1_5,background_preservation_1_5,overedit_1_5,winner,notes
```

The script auto-fills only `sample_id`, `variant`, `output_path`, and `prompt_path`. It does not auto-fill subjective scores or notes.

## Merge Gate

Klein LoRA can be proposed for a frontend advanced mode only if:

- `old_garment_removed_1_5` is better than IDM original.
- `identity_1_5 >= 4`.
- `pose_preservation_1_5 >= 4`.
- `body_shape_preservation_1_5 >= 4`.
- `background_preservation_1_5 >= 4`.
- `overedit_1_5 <= 2`.
- At least 3-5 eval samples win or tie IDM-VTON.

If it wins only one sample, keep it as an experimental benchmark/docs mode. Do not change the IDM-VTON default.
