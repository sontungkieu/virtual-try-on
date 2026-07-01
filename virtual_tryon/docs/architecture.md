# Architecture

## Why IDM-VTON Is The Core

IDM-VTON-style methods are designed for garment-conditioned person image synthesis. They are the right core because the task is not general image editing; it must preserve pose, body shape, face, hair, and background while replacing only the garment region with a reference garment.

## Why FLUX Is A Refiner

FLUX.2 dev/base is used as a post-process image-edit/inpaint refiner. Its job is to repair boundary quality, fabric folds, texture, lighting, hand overlap, collar, sleeves, and hemline. It should not replace the whole try-on engine because unrestricted editing can drift identity, pose, or garment geometry.

## Why ADetailer Is Not Core

ADetailer-like logic is a repair module. It can detect or mask local problem regions and inpaint them, but it does not solve garment transfer. It is useful after core try-on and optional FLUX refinement.

## Pipeline Diagram

```text
person image + garment image + category + prompt
  -> validate inputs
  -> normalize person and garment
  -> human parser stub or model
  -> densepose stub or model
  -> agnostic mask and agnostic person image
     -> innerwear categories use anatomy-shaped masks plus optional foreground silhouette clipping
  -> garment segmentation
  -> core try-on engine, default IDM-VTON
     -> stages a one-sample VITON-HD-style dataset
     -> runs IDM-VTON inference.py through accelerate
     -> copies generated image to core_output.png
  -> build garment, boundary, and safe refinement masks
  -> quality checks on core_output.png
  -> optional FLUX refiner on the selected mask region
     -> if FLUX is missing, OOMs, or fails to load, keep core_output.png
  -> quality checks on refined_output.png when present
  -> choose final result by quality gate
  -> optional ADetailer-like local repair after an accepted refiner output
  -> save result, quality_report.json, mask_metadata.json, masks, overlays, and debug logs
```

Benchmark orchestration runs the same pipeline in separate job folders for `idm`, `idm_flux`, and `repair`, and treats `catvton` and `klein_lora` as optional baseline engines. Unavailable baselines are recorded as skipped rows; they do not affect the default IDM-VTON API path.

## API Runtime

The backend serves generated files through `/artifacts/{job_id}/...`, mapped only to `data/outputs`. Path traversal and files outside outputs return 404.

`POST /tryon` supports sync and async modes. Sync runs the job before responding. Async writes `job.json`, returns `queued`, runs the pipeline in a FastAPI background task, and serializes GPU work with a local lock so concurrent requests do not start competing IDM-VTON runs.

## Engine Contract

All core engines implement:

```python
is_available() -> bool
prepare() -> None
run(inputs: TryOnInputs) -> TryOnResult
```

## IDM-VTON Adapter

The IDM-VTON adapter keeps the official repository isolated in `third_party/IDM-VTON`. The backend does not copy official source files into `backend/app`. For each job it writes:

```text
data/outputs/{job_id}/idm_vton_dataset/test/image/person_0001.jpg
data/outputs/{job_id}/idm_vton_dataset/test/cloth/garment_0001.jpg
data/outputs/{job_id}/idm_vton_dataset/test/agnostic-mask/person_0001_mask.png
data/outputs/{job_id}/idm_vton_dataset/test/image-densepose/person_0001.jpg
data/outputs/{job_id}/idm_vton_dataset/test/vitonhd_test_tagged.json
data/outputs/{job_id}/idm_vton_dataset/test_pairs.txt
```

When `idm_vton.resident_worker=true`, the adapter starts `scripts/idm_vton_resident_worker.py` once per backend process, keeps the IDM-VTON pipeline loaded in that worker, and sends each staged dataset through a JSONL request. The worker is cached across jobs by Python executable, IDM repo path, worker script, and model name. If the worker fails and `resident_worker_fallback=true`, the adapter records `idm_vton_resident_error.txt` and falls back to the original subprocess path.

If the resident worker is disabled, or when fallback is used, it runs:

```text
python -m accelerate.commands.launch third_party/IDM-VTON/inference.py ...
```

If checkpoint, path, or dependency checks fail, the API returns a failed job with a clear error string instead of a raw stack trace.

All refiners implement:

```python
is_available() -> bool
prepare() -> None
refine(image, mask, prompt, references=None, seed=None) -> RefineResult
```

## Dynamic Masks

Every try-on mask is anchored to a body estimate for the current person image. The estimator tries global foreground separation, adaptive row/column border-background separation, and center-saliency separation before falling back to a conservative geometric body box. `mask_metadata.json` records the selected source, bbox, area, warnings, and cache state.

`men_underwear`, `women_underwear`, and `women_bra` do not reuse the broad `lower_body` or `upper_body` rectangles. The backend builds a tighter anatomy-shaped raw mask first:

- men and women underwear: pelvis/brief-shaped mask with leg openings;
- women bra: cup, bridge, band, and strap mask;
- foreground body silhouette clipping when the person can be separated from the image background;
- conservative geometric body fallback when foreground estimation is unreliable.

Innerwear masks use `preprocessing.innerwear_dilation_px` and `preprocessing.innerwear_blur_radius`, which are smaller than the outerwear mask settings. Every innerwear job writes `mask_innerwear_shape.png`, optional `mask_body_silhouette.png`, and `mask_metadata.json`. When `preprocessing.mask_cache_enabled=true`, repeated jobs with the same original person image, category, resolution, and mask config reuse `data/temp/mask_cache/{cache_key}`.

## Refinement Masks

The pipeline writes three refiner masks for every completed core job:

```text
garment_refine_mask.png
boundary_refine_mask.png
safe_refine_mask.png
garment_refine_mask_overlay.png
boundary_refine_mask_overlay.png
safe_refine_mask_overlay.png
```

`garment_refine_mask.png` is the soft clothing region. `boundary_refine_mask.png` is `dilate(mask) - erode(mask)` for edge repair. `safe_refine_mask.png` currently falls back to the garment mask and records a warning because face, hair, and hand parser exclusion is not wired yet.

## Quality Gate

Each job writes `quality_report.json`:

```json
{
  "core": {
    "background_preservation_score": 0.0,
    "face_preservation_score": null,
    "garment_change_score": 0.0,
    "over_edit_score": 0.0,
    "artifact_heuristic_score": 1.0,
    "needs_refine": true,
    "notes": []
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
  "final_choice": "core"
}
```

The gate rejects a refined image when it over-edits outside the active mask or fails artifact heuristics. Missing parser signals produce `null` scores and notes, not crashes.

`quality_report.json` also includes `engine_status` and `final_choice_reason` so benchmark rows can compare completed, skipped, and failed engines without parsing raw logs.
