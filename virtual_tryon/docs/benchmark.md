# Benchmark And Review

## Golden Eval Set

Eval samples live under:

```text
data/eval_set/{sample_id}/
  person.jpg
  garment_top.jpg
  garment_bottom.jpg
  garment_dress.jpg
  metadata.json
```

`metadata.json`:

```json
{
  "sample_id": "sample_001",
  "category": "upper_body",
  "difficulty": "easy",
  "expected_focus": ["identity", "garment_texture"],
  "notes": ""
}
```

Validate:

```bash
cd virtual_tryon
uv run python scripts/validate_eval_set.py --eval-set data/eval_set
```

The validator reports warnings instead of crashing when the folder is empty or a sample is incomplete.

## Model Comparison Benchmark

Run:

```bash
cd virtual_tryon
uv run python scripts/benchmark_pipeline.py \
  --eval-set data/eval_set \
  --modes idm,idm_flux,idm_mask_expanded,idm_mask_expanded_flux,klein_lora \
  --prompt-source auto \
  --prompt-variant strong_remove_old_garment \
  --testcase-id tc10 \
  --save-prompts \
  --limit 1 \
  --output data/outputs/benchmark_phase6_test
```

Modes:

- `idm`: IDM-VTON core only.
- `idm_flux`: IDM-VTON plus optional FLUX refiner.
- `idm_mask_expanded`: IDM-VTON with the experimental upper-body hem expansion enabled for that run only.
- `idm_mask_expanded_flux`: expanded-mask IDM-VTON plus optional FLUX refiner.
- `repair`: IDM-VTON plus accepted FLUX output plus repair.
- `catvton`: CatVTON baseline.
- `klein_lora`: local FLUX.2 Klein 9B + Try-On LoRA experimental baseline.

If a baseline engine is unavailable, its row is marked `unavailable` and the benchmark continues.

## IDM Runtime Optimization Benchmark

Use this benchmark when comparing resident IDM runtime implementations, not visual quality variants. It starts an isolated resident worker for each mode, runs the same input across fixed resolution buckets, and writes timing plus VRAM data.

```bash
cd virtual_tryon
uv run python scripts/benchmark_idm_runtime_modes.py \
  --modes eager,torch_compile,tensorrt \
  --presets 512x768,640x896,768x1024 \
  --steps 4 \
  --warmup 1 \
  --repeat 1 \
  --output data/outputs/idm_runtime_benchmark_test
```

Modes:

- `eager`: resident IDM worker without compiler changes.
- `torch_compile`: resident worker with `torch.compile` on IDM UNet modules. The first warmup per shape includes compile overhead.
- `tensorrt`: experimental Torch-TensorRT backend for resident IDM. It requires `uv sync --extra idm --extra tensorrt`; the default module set is the stable `vae_decode` path. IDM UNet TensorRT is disabled by default because Torch-TensorRT/TensorRT can crash inside the native builder on these graphs.

TensorRT tuning flags:

- `--tensorrt-modules vae_decode`: stable smoke bench and production-safe TensorRT path.
- `--tensorrt-modules all --allow-unsafe-tensorrt-unet`: compile IDM UNet block-wise (`unet_blocks`, `unet_encoder_blocks`) plus `vae_decode` for isolated experiments. This avoids wrapping the entire UNet forward pass in one graph, but it can still fail or crash in the native builder.
- `--tensorrt-modules unet,unet_encoder,vae_decode --allow-unsafe-tensorrt-unet`: aggressive whole-UNet debugging mode. Do not use this for production jobs.
- `--tensorrt-partition-preset attention|none|shape|matmul|norm|conv|safe_unet`: choose ATen ops that must stay in PyTorch to split unsafe TensorRT partitions. `safe_unet` keeps convolution, attention, reshape, matmul, and norm ops in PyTorch to avoid the most fragile TensorRT UNet partitions, so it is mainly useful for debugging rather than speed.
- `--tensorrt-torch-executed-ops`: comma-separated explicit ATen ops such as `aten._reshape_copy.default`.
- `--tensorrt-min-block-size`, `--tensorrt-workspace-size`, and `--tensorrt-optimization-level`: pass through to the Torch-TensorRT Dynamo backend.
- `--tensorrt-enable-resource-partitioning`, `--tensorrt-cpu-memory-budget`, and `--tensorrt-lazy-engine-init`: extra safety knobs for native TensorRT builder memory pressure.
- `--tensorrt-engine-cache-dir`: set a persistent engine/timing cache path.

Output:

```text
data/outputs/idm_runtime_benchmark_{timestamp}/
  benchmark_config.json
  runs.csv
  runs.jsonl
  summary.csv
  summary.json
  eager/
  torch_compile/
  tensorrt/
```

`summary.csv` reports measured-run median and mean wall time plus worker runtime per mode and preset. Warmup rows remain in `runs.csv` but are excluded from aggregate timings.

Prompt options:

- `--prompt-source manual|auto`: manual uses `--prompt`; auto builds prompts from `backend/app/prompts/testcase_prompt_library.py`.
- `--prompt-variant default|strong_remove_old_garment|identity_strict`: selects a deterministic prompt variant.
- `--testcase-id tc1..tc15`: testcase metadata used by auto prompt mode.
- `--save-prompts`: records prompt artifacts in each mode folder.

When auto prompts are enabled, mode folders include `prompt_core.txt`, optional `prompt_refine.txt`, `negative_prompt.txt`, and `prompt_metadata.json`. `summary.csv` includes `prompt_variant`, `prompt_hash`, and `prompt_path`.

Output:

```text
data/outputs/benchmark_{timestamp}/
  summary.csv
  summary.json
  grid.png
  index.html
  manual_ratings.csv
  sample_001/
    input_person.png
    input_garment_top.png
    quality_report.json
    run_metadata.json
    idm/
    idm_flux/
    idm_mask_expanded/
    idm_mask_expanded_flux/
    catvton/
    klein_lora/
```

`klein_lora` rows include `prompt_path`, `engine_status`, and `error_code`. When local Klein model files, `klein-local` dependencies, or optional fal.ai credentials are missing for the selected backend, the row is skipped with `ENGINE_UNAVAILABLE`; this is expected on CI and local CPU-only environments.

## Review Gallery

Generate or rebuild:

```bash
uv run python scripts/build_review_gallery.py \
  --benchmark-dir data/outputs/benchmark_phase6_test
```

Open `index.html` offline. Skipped modes are shown as placeholders. Use `manual_ratings.csv` for human scoring with columns for identity, garment fidelity, realism, pose preservation, artifact score, winner, and notes.

Generated benchmark folders are regular outputs and can be cleaned with:

```bash
uv run python scripts/cleanup_outputs.py --older-than-hours 24 --keep-latest 5 --dry-run
```

## Upper-Body Mask Ablation

The upper-body hem expansion is an experiment and is disabled in the production configuration. Run all controlled variants with the same sample and seed:

```bash
uv run python scripts/run_mask_ablation.py \
  --sample data/eval_set/sample_001 \
  --seed 0 \
  --variants idm_original,idm_mask_expanded,idm_mask_expanded_flux_local \
  --prompt-source auto \
  --prompt-variant strong_remove_old_garment \
  --testcase-id tc10 \
  --save-prompts \
  --output data/outputs/ablation_upper_body_mask_test
```

Use `--mock` for CI or local schema verification without IDM-VTON or FLUX:

```bash
uv run python scripts/run_mask_ablation.py \
  --sample data/eval_set/sample_001 \
  --seed 0 \
  --variants idm_original,idm_mask_expanded,idm_mask_expanded_flux_local \
  --output data/outputs/ablation_upper_body_mask_mock \
  --mock
```

The runner writes:

```text
data/outputs/ablation_upper_body_mask_test/
  comparison_grid.png
  comparison_index.html
  summary.csv
  summary.json
  manual_ratings_mask_ablation.csv
  idm_original/
  idm_mask_expanded/
  idm_mask_expanded_flux_local/
```

The expanded variant also saves original, expanded, and difference masks plus overlays. The FLUX variant uses `safe_refine_mask.png`, falling back to `boundary_refine_mask.png`, and never refines the whole image. If the local FLUX backend is unavailable or runs out of memory, the row is marked skipped and the core comparison remains usable.

Do not enable the expanded mask by default based on automated metrics alone. Merge it only after manual review confirms stronger old-garment removal without weaker identity, pose, background, or garment shape.

## Klein LoRA Ablation

Run the dedicated LoRA comparison:

```bash
uv run python scripts/run_klein_lora_ablation.py \
  --sample data/eval_set/sample_001 \
  --seed 42 \
  --prompt-source auto \
  --testcase-id tc10 \
  --save-prompts \
  --bottom-strategy crop_from_person \
  --output data/outputs/klein_lora_ablation_test
```

Without the local Klein base model, LoRA file, or `klein-local` dependencies, the script still creates `summary.csv`, `summary.json`, `comparison_grid.png`, `comparison_index.html`, and `manual_ratings_klein_lora.csv`; Klein rows are marked unavailable and include sanitized status artifacts. With `models/flux2-klein-9b`, `models/loras/flux-klein-tryon.safetensors`, and a Diffusers build that exports `Flux2KleinPipeline`, the same command attempts the local endpoint.

Before a local run, download/validate assets:

```bash
/root/.local/bin/uv venv /workspace/venvs/project_phase2_klein --python 3.11
/root/.local/bin/uv pip install --python /workspace/venvs/project_phase2_klein/bin/python \
  "diffusers @ git+https://github.com/huggingface/diffusers.git" \
  "transformers>=4.56" "accelerate>=1.0" "peft>=0.17" "safetensors>=0.4" pillow numpy
export TRYON_KLEIN_PYTHON=/workspace/venvs/project_phase2_klein/bin/python
$TRYON_KLEIN_PYTHON scripts/download_klein_local_models.py
```

`scripts/check_fal_runtime.py --strict` is still available only for the optional `TRYON_KLEIN_BACKEND=fal_api` path.

The comparison grid contains:

- Person input.
- Top garment reference.
- Auto bottom reference.
- IDM original.
- `klein_lora_default`.
- `klein_lora_strong_remove_old_shirt`.

Use `manual_ratings_klein_lora.csv` for subjective scoring. Do not promote Klein LoRA to the default engine unless it wins or ties IDM-VTON on multiple eval samples while preserving identity, pose, body shape, and background.
