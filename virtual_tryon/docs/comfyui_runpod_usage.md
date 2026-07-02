# RunPod ComfyUI Usage For Phase 2 VTON

## Start ComfyUI On RunPod

```bash
cd /workspace/Project_Phase2/virtual_tryon
bash scripts/setup_runpod_comfyui_phase2.sh
/workspace/ComfyUI/run_vton_phase2_comfyui.sh
```

By default the setup script creates or refreshes a separate ComfyUI venv at `/workspace/venvs/project_phase2_comfyui`. It does not install ComfyUI dependencies into the main backend/IDM venv at `/workspace/venvs/project_phase2`.

To skip dependency refresh after the ComfyUI venv already exists:

```bash
INSTALL_COMFY_DEPS=0 bash scripts/setup_runpod_comfyui_phase2.sh
```

The setup script installs ComfyUI under `/workspace/ComfyUI` and links:

```text
/workspace/ComfyUI/custom_nodes/vton_phase2_nodes
-> /workspace/Project_Phase2/virtual_tryon/comfyui_nodes/vton_phase2_nodes
```

It also copies importable UI workflows into:

```text
/workspace/ComfyUI/user/default/workflows/
```

The launch script starts ComfyUI on port `8188`.

## Access From Local Machine

If RunPod does not expose port `8188` in the HTTP services list, use SSH tunneling:

```powershell
ssh -N -L 8188:127.0.0.1:8188 root@64.119.209.250 -p 13590 -i ~/.ssh/id_ed25519
```

Then open:

```text
http://127.0.0.1:8188
```

## Custom Nodes

Search for these nodes in ComfyUI:

```text
VTON Phase2 - IDM / IDM Expanded
VTON Phase2 - SCHP/SAM Mask
VTON Phase2 - Klein Reference Set
VTON Phase2 - Klein Local Sampler
```

## Three Klein Direction Workflows

The three Klein directions are available as separate ComfyUI workflow files, not just as a dropdown preset:

```text
/workspace/ComfyUI/user/default/workflows/vton_phase2_klein_4step_sample015.workflow.json
/workspace/ComfyUI/user/default/workflows/vton_phase2_klein_28_sample015.workflow.json
/workspace/ComfyUI/user/default/workflows/vton_phase2_klein_28_strong_sample015.workflow.json
```

They are also tracked in the project under:

```text
virtual_tryon/comfyui_workflows/
```

Each workflow is preset with `sample_015` input images for quick testing. Replace the four `Load Image` nodes to run another testcase.

## Method Recipes

### IDM

```text
LoadImage(person)
LoadImage(garment)
-> VTON Phase2 - IDM / IDM Expanded
   category=<upper_body|lower_body|dress|full_outfit|men_underwear|women_underwear|women_bra>
   mask_expanded=false
-> SaveImage
```

### IDM Expanded

```text
LoadImage(person)
LoadImage(garment)
-> VTON Phase2 - IDM / IDM Expanded
   category=<upper_body|lower_body|dress|full_outfit|men_underwear|women_underwear|women_bra>
   mask_expanded=true
-> SaveImage
```

### Klein 4-Step

```text
LoadImage(person)
LoadImage(ref1)
LoadImage(ref2)
LoadImage(ref3)
-> VTON Phase2 - Klein Reference Set
-> VTON Phase2 - Klein Local Sampler
   method=klein_4step
   seed=<integer>
   guidance_scale=2.5
   lora_scale=1.0
-> SaveImage
```

### Klein 28

Same graph as Klein 4-Step, but:

```text
method=klein_28
```

### Klein 28 Strong

Same graph as Klein 4-Step, but:

```text
method=klein_28_strong
```

This appends stronger target-preservation instructions for hard multi-item cases.

## Klein Reference Modes

| Mode | Use case |
|---|---|
| `duplicate_ref1` | Dress or single garment reference. |
| `lower_body_ref1_preserve_upper` | Lower-body try-on while preserving upper body. |
| `top_ref1_bottom_ref2` | Full outfit with separate top and bottom refs. |
| `accessory_ref1_ref2` | Accessory stress tests such as hat/watch. |
| `dress_ref1_hat_ref3_shoes_ref2` | Dress + shoes + hat, used for sample 015. |

## Smoke Test

With ComfyUI running:

```bash
cd /workspace/Project_Phase2/virtual_tryon
export VIRTUAL_ENV=/workspace/venvs/project_phase2
export PATH=/root/.local/bin:$VIRTUAL_ENV/bin:$PATH
uv run --active --no-sync python scripts/comfyui_smoke_sample15.py
```

Expected output:

```text
status={"status_str": "success", "completed": true, ...}
outputs={"7": {"images": [{"filename": "sample_015_klein4_00001_.png", ...}]}}
```

The smoke image is saved under:

```text
/workspace/ComfyUI/output/vton_phase2_comfy_smoke/
```

## Omnitry Innerwear Reproduction Flow

Generate workflow files and copy the adult Omnitry innerwear inputs into ComfyUI:

```bash
cd /workspace/Project_Phase2/virtual_tryon
export VIRTUAL_ENV=/workspace/venvs/project_phase2
export PATH=/root/.local/bin:$VIRTUAL_ENV/bin:$PATH
uv run --active --no-sync python scripts/comfyui_omnitry_repro.py
```

The generated files are:

```text
/workspace/Project_Phase2/virtual_tryon/comfyui_workflows/omnitry_innerwear_repro/
/workspace/ComfyUI/input/vton_omnitry_repro/
```

Each case has an API prompt and a UI workflow:

```text
<case>_idm_vton_api.json
<case>_idm_vton_ui.workflow.json
```

To queue the first generated case through ComfyUI:

```bash
uv run --active --no-sync python scripts/comfyui_omnitry_repro.py --limit 1 --queue-first
```

The Omnitry flow uses `VTON Phase2 - Backend Try-On API` to call the local FastAPI backend from ComfyUI. The backend still builds the dynamic mask and returns the mask preview artifact, but ComfyUI does not load the IDM model directly. The API node supports these production categories:

```text
upper_body
lower_body
dress
full_outfit
men_underwear
women_underwear
women_bra
```
