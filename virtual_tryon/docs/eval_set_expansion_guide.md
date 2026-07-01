# Evaluation Set Expansion Guide

## Goal

Expand `data/eval_set/` from the single smoke sample to 5-10 consented, license-compatible samples. Keep source images private when their redistribution terms are unclear; the validator and benchmark can operate on local untracked samples.

Include a deliberate mix:

- Simple upper-body transfer with a clear frontal pose and unobstructed torso.
- Top with logo, typography, stripe, plaid, or other fine-pattern garment.
- Hair occlusion crossing the collar or upper garment.
- Hand occlusion, with one or both hands overlapping the garment.
- Side pose or non-frontal three-quarter pose.
- Lower-body, dress, or adult innerwear samples only where the selected engine and inputs support them.

## Folder Layout

```text
data/eval_set/sample_002/
  person.jpg
  garment_top.jpg
  metadata.json
```

Use `garment_bottom.jpg` for `lower_body`, `men_underwear`, and `women_underwear`; use `garment_top.jpg` for `upper_body` and `women_bra`; use `garment_dress.jpg` for `dress`; and use either a dress image or both top and bottom images for `full_outfit`. Accepted image extensions are `.jpg`, `.jpeg`, `.png`, and `.webp`.

## Metadata Schema

```json
{
  "sample_id": "sample_002",
  "category": "upper_body",
  "difficulty": "hard",
  "expected_focus": ["logo_fidelity", "hand_occlusion"],
  "notes": "Right hand overlaps the shirt graphic."
}
```

Rules:

- `sample_id` must match the folder name.
- `category` must be `upper_body`, `lower_body`, `dress`, `full_outfit`, `men_underwear`, `women_underwear`, or `women_bra`.
- `difficulty` must be `easy`, `medium`, or `hard`.
- `expected_focus` must be a JSON list of strings.
- `notes` is optional but should record the visual challenge and review target.

## Validate

From `virtual_tryon/` run:

```bash
uv run python scripts/validate_eval_set.py --eval-set data/eval_set
```

Resolve every warning before benchmarking. Also inspect that each person/garment pair has the intended category, readable orientation, and no accidental duplicate.

## Benchmark

With local model dependencies available, run:

```bash
uv run python scripts/benchmark_pipeline.py \
  --eval-set data/eval_set \
  --modes idm,idm_mask_expanded,klein_lora \
  --limit 5
```

Review `summary.json`, `summary.csv`, the generated grid/gallery, and `manual_ratings.csv`. Compare identity, garment fidelity, realism, pose preservation, and artifacts rather than selecting a winner from one automated score alone.

For Klein LoRA prompt-specific review, run:

```bash
uv run python scripts/run_klein_lora_ablation.py \
  --sample data/eval_set/sample_001 \
  --seed 42 \
  --bottom-strategy crop_from_person \
  --output data/outputs/klein_lora_ablation_sample_001
```

Repeat for each local sample. Do not commit private or license-unclear images. Do not expose Klein LoRA as a default or advanced UI engine until at least 3-5 samples win or tie IDM-VTON under the manual criteria in `docs/klein_lora_real_eval.md`.
