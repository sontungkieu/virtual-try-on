# Demo Script

## Problem

Virtual Try-On generates an image of a person wearing a reference garment while preserving identity, pose, body shape, and as much scene context as possible.

## Demo Input And Output

Use a clear person photo and one garment image matching the selected category. The output panel shows the final image, intermediate masks, core output, quality report, and job artifacts.

## Pipeline

`person + garment -> IDM-VTON core -> optional FLUX refiner -> quality gate -> artifacts`

IDM-VTON is the core because it is purpose-built for garment-conditioned try-on and provides the main clothing transfer. FLUX is only a guarded refiner: it may improve boundaries or local texture, but the quality gate can keep the IDM-VTON output when refinement changes too much.

## Run On RunPod

```bash
cd /workspace/Project_Phase2/virtual_tryon
PYTHONPATH=. uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
cd frontend && npm run dev
```

Open the proxied frontend port, upload the `sample_001` person and top garment, choose `Top`, disable refiner for the fastest deterministic demo, then select **Generate**.

## Read The Quality Report

- `final_choice`: whether the core or refined image was selected.
- `final_choice_reason`: why the quality gate selected that image.
- `outside_mask_delta`: change outside the intended edit region; lower is safer.
- `garment_region_delta`: change inside the garment region.
- `engine_status`: success, skipped, or failed state for each configured engine.

## Current Limitations

Fine logos and text may deform. Difficult poses, occlusion, hands, and long hair can produce boundary errors. The optional refiner can over-edit, so it is constrained by masks and quality checks.

## Next Development

- Expand the golden evaluation set.
- Connect and validate a real FLUX backend.
- Add real CatVTON and Klein baselines.
- Collect manual reviewer ratings.
- Replace the in-process queue with Redis and Celery for multi-worker production.
