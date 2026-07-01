# Evaluation

The first quality checker is intentionally lightweight and deterministic. The API still exposes a compact `quality` block for compatibility:

```json
{
  "identity_score": null,
  "garment_similarity_score": 0.0,
  "background_preservation_score": 0.0,
  "artifact_score": 0.0,
  "needs_refine": true,
  "notes": []
}
```

Implemented checks:

- Output exists through pipeline storage.
- Output resolution is above configured minimum.
- Background preservation via difference outside the garment mask.
- Garment region change via difference inside the garment mask.
- Over-edit score outside the active refine mask.
- `outside_mask_delta` and `garment_region_delta` for benchmark summaries.
- Blank/corrupt, color collapse, and output resolution checks.
- Artifact heuristic for low-resolution, blank, or low-variation outputs.
- Rough garment similarity when a garment reference is available.

Every completed core job also writes `quality_report.json`:

```json
{
  "core": {
    "background_preservation_score": 0.92,
    "face_preservation_score": null,
    "garment_change_score": 0.31,
    "over_edit_score": 0.08,
    "artifact_heuristic_score": 1.0,
    "needs_refine": false,
    "notes": ["Face preservation score is unavailable because face parser/bbox is not wired."]
  },
  "refined": {
    "background_preservation_score": null,
    "face_preservation_score": null,
    "garment_change_score": null,
    "over_edit_score": null,
    "artifact_heuristic_score": null,
    "accepted": false,
    "notes": []
  },
  "baselines": {
    "catvton": null,
    "klein_lora": null
  },
  "final_choice": "core",
  "final_choice_reason": "refiner unavailable or skipped; using core output",
  "engine_status": {
    "idm_vton": "success",
    "flux_refiner": "skipped",
    "catvton": "skipped",
    "klein_lora": "skipped"
  }
}
```

If `use_refiner=true` but FLUX is unavailable or fails, the job remains `completed`, `final_choice` stays `core`, and the failure reason is written to `flux_refiner_error.txt` plus the refined report notes.

## Automated Metrics Versus Manual Review

`quality_report.json` contains only signals that the code can compute or verify from the available images and masks. It must not contain subjective claims such as "identity looks good", "the old shirt is visible", or human 1-5 ratings unless a detector or metric is implemented for that claim.

Subjective observations belong in the benchmark's `manual_ratings.csv`, the mask ablation's `manual_ratings_mask_ablation.csv`, or the Klein LoRA ablation's `manual_ratings_klein_lora.csv`. Human reviewers should score identity, garment fidelity, old-garment removal, realism, pose preservation, over-editing, and the winning variant there.

Prompt variant comparisons use `manual_ratings_prompt_ablation.csv` from `scripts/run_prompt_ablation.py`. Prompt success/failure observations, such as old garment remnants, logo fidelity, or over-editing caused by a strong prompt, must be recorded in that manual ratings file rather than `quality_report.json`.

Klein LoRA manual ratings use this schema:

```text
sample_id,variant,output_path,prompt_path,identity_1_5,garment_fidelity_1_5,old_garment_removed_1_5,realism_1_5,pose_preservation_1_5,body_shape_preservation_1_5,background_preservation_1_5,overedit_1_5,winner,notes
```

The template auto-fills only sample, variant, `output_path`, and `prompt_path` fields. Human scores and notes stay blank until a reviewer fills them in.

## Benchmark

Run:

```bash
cd virtual_tryon
uv run python scripts/benchmark_pipeline.py \
  --person data/examples/person_001.jpg \
  --garment data/examples/top_001.jpg \
  --category upper_body
```

The benchmark writes:

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
    catvton/
    klein_lora/
```

Rows include `sample_id`, `mode`, `runtime_seconds`, `output_path`, `background_preservation_score`, `face_preservation_score`, `garment_change_score`, `over_edit_score`, `final_choice`, and `notes`. The benchmark keeps running when the refiner fails, so it is safe to use while FLUX weights or compatible diffusers builds are still being prepared.

Prompt-aware benchmark rows also include `prompt_variant`, `prompt_hash`, and `prompt_path` when `--prompt-source auto` or saved prompt artifacts are enabled.

For golden-set benchmarking, prefer:

```bash
uv run python scripts/benchmark_pipeline.py \
  --eval-set data/eval_set \
  --modes idm,idm_flux,idm_mask_expanded,idm_mask_expanded_flux,klein_lora \
  --limit 1 \
  --output data/outputs/benchmark_phase6_test
```

Future upgrades:

- Face embedding similarity.
- Human parsing based region preservation.
- Garment CLIP/DINO similarity.
- Per-region artifact detection for collar, sleeve, hands, and hemline.
- Detector or segmentation comparison for old-garment remnants near the hemline.
