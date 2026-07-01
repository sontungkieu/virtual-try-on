# Phase 2 ComfyUI Workflow Presets

These workflow files are separate ComfyUI presets for the three Klein directions used in the comparison report.

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
