# Phase 2 ComfyUI Workflow Presets

These workflow files are ComfyUI presets for reproducible Phase 2 try-on runs.

| Workflow file | Direction | Key preset |
|---|---|---|
| `vton_phase2_klein_4step_sample015.workflow.json` | Klein 4-step | `method=klein_4step`, fast preview |
| `vton_phase2_klein_28_sample015.workflow.json` | Klein 28 | `method=klein_28`, default 28-step quality |
| `vton_phase2_klein_28_strong_sample015.workflow.json` | Klein 28 strong | `method=klein_28_strong`, stronger accessory/garment prompt |

Each workflow uses these custom nodes:

```text
VTON Phase2 - Klein Reference Set
VTON Phase2 - Klein Local Sampler
```

The preset image inputs are for `sample_015`:

```text
sample_015_person.png
sample_015_dress.png
sample_015_shoes.png
sample_015_hat.png
```

On RunPod they are copied into:

```text
/workspace/ComfyUI/input/
```

After opening a workflow in ComfyUI, replace the four `Load Image` nodes with the target testcase images when needed.

## Omnitry Innerwear Reproduction

Run this script on RunPod to generate reproducible ComfyUI API prompts and UI workflows for the Omnitry adult innerwear cases:

```bash
cd /workspace/Project_Phase2/virtual_tryon
export VIRTUAL_ENV=/workspace/venvs/project_phase2
export PATH=/root/.local/bin:$VIRTUAL_ENV/bin:$PATH
uv run --active --no-sync python scripts/comfyui_omnitry_repro.py
```

Outputs are written to:

```text
comfyui_workflows/omnitry_innerwear_repro/
/workspace/ComfyUI/input/vton_omnitry_repro/
```

Each case gets:

```text
*_api.json
*_ui.workflow.json
```

The API workflow calls the local FastAPI backend from ComfyUI, then saves the generated try-on result and backend mask preview with fixed seed, resolution, steps, category, and prompt. The UI workflow can be opened in ComfyUI for manual reproduction and inspection without loading the IDM model inside the ComfyUI process.
