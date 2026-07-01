# Klein Detailed ComfyUI Workflows

Folder này chứa 2 workflow UI JSON cho hai hướng Klein đã chạy trước đó, nhưng được tách thành các node nhỏ để nhìn rõ pipeline trong ComfyUI.

## Files

| File | Hướng | Model | LoRA | Prompt |
|---|---|---|---|---|
| `01_flux2_klein9b_strong_detailed.workflow.json` | Flux strong | `black-forest-labs/FLUX.2-klein-9B` | Không | Strong |
| `02_flux2_klein9b_lora_strong_detailed.workflow.json` | Flux + LoRA strong | `black-forest-labs/FLUX.2-klein-9B` | `fal/flux-klein-9b-virtual-tryon-lora` | Strong |

## Graph Contract

Các workflow này không dùng node cũ `VTON Phase2 - Klein Local Sampler` kiểu một cục. Pipeline được tách ra:

```text
Load person image
Load garment/reference image
Fit person to Klein canvas
Fit garment/reference to Klein canvas
Crop lower-body preserve reference from person
Build strong prompt
Load FLUX.2 Klein 9B base model
[optional] Load virtual try-on LoRA
Klein detailed sampler
Save image
```

## Required Custom Nodes

Các node này nằm trong `virtual_tryon/comfyui_nodes/vton_phase2_nodes`:

```text
VTON Phase2 - Klein Fit Canvas
VTON Phase2 - Klein Bottom Preserve Crop
VTON Phase2 - Klein Prompt Builder
VTON Phase2 - Load FLUX.2 Klein 9B
VTON Phase2 - Load Klein Try-On LoRA
VTON Phase2 - Klein Detailed Sampler
```

Sau khi cập nhật node source trên RunPod, cần restart ComfyUI để nó nhận node mới.

## Default Inputs

Các workflow đang trỏ tới tên input mẫu:

```text
sample_015_person.png
sample_015_dress.png
```

Bạn có thể đổi trực tiếp trong các node `Load Image`.
