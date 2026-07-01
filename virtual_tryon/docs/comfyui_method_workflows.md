# ComfyUI Method Workflows For Phase 2 Virtual Try-On

This document describes the ComfyUI-style workflow blueprint for each method used in the Phase 2 comparison grid. These are implementation blueprints, not generic marketing descriptions. Each workflow states the exact inputs, processing nodes, main parameters, and output artifacts.

The current experiment runs the Klein methods locally with `FLUX.2-klein-9B` plus Try-On LoRA through Diffusers. It does not call an external image-generation API. In ComfyUI, IDM-VTON and Flux2Klein should be implemented as custom nodes that wrap the existing project code.

## Shared Inputs

| Input | Type | Required by | Notes |
|---|---|---|---|
| `person_image` | image | all methods | Full-body or near full-body source person image. |
| `garment_top` | image | upper/full outfit/Klein | Upper-body garment reference; also used for `women_bra`. |
| `garment_bottom` | image | lower/full outfit/Klein | Lower-body garment reference; also used for `men_underwear` and `women_underwear`. |
| `garment_dress` | image | dress/full outfit/Klein | Dress reference or composed outfit reference. |
| `accessory_hat` | image | Klein accessory/full outfit stress tests | Used only by Klein reference composition. |
| `accessory_shoes` | image | Klein accessory/full outfit stress tests | Used only by Klein reference composition. |
| `category` | string | all methods | `upper_body`, `lower_body`, `dress`, `full_outfit`, `men_underwear`, `women_underwear`, `women_bra`, or `accessory`. |
| `prompt` | text | all methods | Stronger for Klein 28 strong and IDM expanded. |
| `seed` | integer | all generative methods | Reproducibility seed. |

## Method 1: IDM

### Goal

Use IDM-VTON as the core garment-conditioned virtual try-on model. This method is best for standard clothing replacement: tops, bottoms, dresses, and simple full outfit cases.

### ComfyUI Node Graph

```text
LoadImage(person_image)
LoadImage(garment_reference)
LoadCategory(category)
BuildPrompt(IDM)
    -> ImagePreprocess(max_side)
    -> HumanParsingNode
    -> DensePoseNode
    -> AgnosticMaskNode(mask_expanded=false)
    -> GarmentSegmentationNode
    -> IDMVTonOfficialRunNode
    -> SaveImage(result.png)
    -> SaveJSON(status.json, quality_report.json)
```

### Node Details

| Node | Input | Output | Important parameters |
|---|---|---|---|
| `LoadImage:Person` | `person_image` | RGB person image | Keep original identity, pose, background. |
| `LoadImage:Garment` | selected garment image | RGB garment image | Chosen from top, bottom, dress, or composed outfit reference. |
| `BuildPrompt:IDM` | person description, garment description, category | core prompt | Uses garment replacement wording and preservation list. |
| `ImagePreprocess` | person + garment | normalized images | Project uses max-side resize before backend processing. |
| `HumanParsingNode` | person | parsing maps | Used to locate body/clothing regions. |
| `DensePoseNode` | person | DensePose map or placeholder | Helps IDM preserve body structure. |
| `AgnosticMaskNode` | person + parsing + category | `raw_mask`, `agnostic_mask`, `soft_mask`, `agnostic_image` | `mask_expanded=false`. |
| `GarmentSegmentationNode` | garment | `cloth_mask`, `garment_normalized` | Normalizes garment reference before IDM. |
| `IDMVTonOfficialRunNode` | agnostic image, mask, DensePose, garment normalized, prompt, category, seed | generated try-on image | Wraps official IDM-VTON inference. |
| `SaveArtifacts` | generated image + debug images | files | Saves result and debug artifacts. |

### Inputs And Outputs

| Item | Value |
|---|---|
| Primary input | `person_image + garment_reference + category + prompt + seed` |
| Intermediate outputs | `raw_mask.png`, `agnostic_mask.png`, `soft_mask.png`, `mask_preview.png`, `agnostic.png`, `cloth_mask.png`, `garment_normalized.png` |
| Final output | `result.png` |

## Method 2: IDM Expanded

### Goal

Use the same IDM-VTON core but with a larger replacement mask. This method is meant to reduce old-garment leftovers, especially near shirt hems, waistbands, lower torso, and dress boundaries.

### ComfyUI Node Graph

```text
LoadImage(person_image)
LoadImage(garment_reference)
LoadCategory(category)
BuildPrompt(IDM_EXPANDED)
    -> ImagePreprocess(max_side)
    -> HumanParsingNode
    -> DensePoseNode
    -> AgnosticMaskNode(mask_expanded=true)
    -> MaskDebugOverlayNode
    -> GarmentSegmentationNode
    -> IDMVTonOfficialRunNode
    -> SaveImage(result.png)
    -> SaveJSON(status.json, quality_report.json)
```

### Difference From IDM

| Part | IDM | IDM Expanded |
|---|---|---|
| Mask mode | Standard agnostic mask | Expanded agnostic mask |
| Prompt | Standard replacement prompt | Adds old-garment-removal and hemline coverage instruction |
| Best use | Stable standard garment transfer | Cases where old clothing remains visible |
| Risk | Less aggressive replacement | Higher risk of changing nearby body/clothing edges |

### Node Details

| Node | Input | Output | Important parameters |
|---|---|---|---|
| `BuildPrompt:IDM_EXPANDED` | base IDM prompt | stronger core prompt | Adds: replacement includes full garment area and hemline; do not preserve old garment inside mask. |
| `AgnosticMaskNode` | person + parsing + category | expanded mask artifacts | `mask_expanded=true`. |
| `MaskDebugOverlayNode` | original and expanded masks | overlay images | Saves original/expanded/diff overlays for visual QA. |
| `IDMVTonOfficialRunNode` | same as IDM | generated try-on image | Uses expanded mask during staging. |

### Inputs And Outputs

| Item | Value |
|---|---|
| Primary input | `person_image + garment_reference + category + expanded-mask prompt + seed` |
| Additional debug outputs | `mask_original_upper_body.png`, `mask_expanded_upper_body.png`, `mask_diff_upper_body.png`, overlay variants |
| Final output | `result.png` |

## Method 3: Klein 4-Step

### Goal

Run local `FLUX.2-klein-9B` with Try-On LoRA as a fast preview method. This is useful for quick comparison but can miss fine details or accessories.

### ComfyUI Node Graph

```text
LoadImage(person_image)
LoadReferenceImages(top/bottom/dress/accessories)
BuildKleinReferences(category)
BuildPrompt(KLEIN_DEFAULT)
LoadFlux2KleinPipeline(local_files_only=true)
LoadTryOnLoRA(local_files_only=true)
    -> Flux2KleinTryOnSampler(steps=4)
    -> SaveImage(result.png)
    -> SaveText(prompt.txt)
    -> SaveJSON(status.json, reference_config.json)
```

### Node Details

| Node | Input | Output | Important parameters |
|---|---|---|---|
| `LoadImage:Person` | `person_image` | RGB person image | Resized/fit to target canvas. |
| `LoadReferenceImages` | garment/accessory refs | raw refs | May include top, bottom, dress, hat, shoes. |
| `BuildKleinReferences` | refs + category | `person`, `top_reference`, `bottom_reference`, `primary_reference` | Creates the three-image conditioning layout expected by the local pipeline. |
| `BuildPrompt:KLEIN_DEFAULT` | sample metadata | text prompt | Describes replacement scope and preservation list. |
| `LoadFlux2KleinPipeline` | local model dir | pipeline object | `model_dir=models/flux2-klein-9b`, `torch_dtype=bfloat16`, `local_files_only=true`. |
| `LoadTryOnLoRA` | local LoRA path | LoRA adapter loaded | `flux-klein-tryon.safetensors`, `lora_scale=1.0`. |
| `Flux2KleinTryOnSampler` | `[person, top_ref, bottom_ref]`, prompt, seed | generated image | `steps=4`, `height=1024`, `width=768`, `guidance_scale=2.5`. |
| `SaveArtifacts` | generated image + configs | files | Saves result and run metadata. |

### Inputs And Outputs

| Item | Value |
|---|---|
| Primary input | `person_image + composed references + default Klein prompt + seed` |
| Model input shape | `image=[person, top_reference, bottom_reference]` |
| Main parameters | `steps=4`, `width=768`, `height=1024`, `lora_scale=1.0` |
| Final output | `result.png` |

## Method 4: Klein 28

### Goal

Run the same local Klein pipeline with more denoising steps for a more stable result. This is the main local open-source generative baseline.

### ComfyUI Node Graph

```text
LoadImage(person_image)
LoadReferenceImages(top/bottom/dress/accessories)
BuildKleinReferences(category)
BuildPrompt(KLEIN_DEFAULT)
LoadFlux2KleinPipeline(local_files_only=true)
LoadTryOnLoRA(local_files_only=true)
    -> Flux2KleinTryOnSampler(steps=28)
    -> SaveImage(result.png)
    -> SaveText(prompt.txt)
    -> SaveJSON(status.json, reference_config.json)
```

### Difference From Klein 4-Step

| Part | Klein 4-Step | Klein 28 |
|---|---|---|
| Denoising steps | 4 | 28 |
| Runtime | Fastest | Slower |
| Expected quality | Preview quality | More stable garment/body rendering |
| Prompt mode | Default | Default |
| Seed offset in experiments | `+0` | `+100` |

### Inputs And Outputs

| Item | Value |
|---|---|
| Primary input | `person_image + composed references + default Klein prompt + seed` |
| Main parameters | `steps=28`, `width=768`, `height=1024`, `guidance_scale=2.5`, `lora_scale=1.0` |
| Final output | `result.png` |

## Method 5: Klein 28 Strong

### Goal

Run Klein 28 with a stronger prompt. This is used for hard cases with multiple target items, accessories, or cases where the default prompt omits hats, shoes, logos, or garment details.

### ComfyUI Node Graph

```text
LoadImage(person_image)
LoadReferenceImages(top/bottom/dress/accessories)
BuildKleinReferences(category, accessory_aware=true)
BuildPrompt(KLEIN_STRONG)
LoadFlux2KleinPipeline(local_files_only=true)
LoadTryOnLoRA(local_files_only=true)
    -> Flux2KleinTryOnSampler(steps=28)
    -> SaveImage(result.png)
    -> SaveText(prompt.txt)
    -> SaveJSON(status.json, reference_config.json)
```

### Difference From Klein 28

| Part | Klein 28 | Klein 28 Strong |
|---|---|---|
| Prompt mode | Default | Strong |
| Target behavior | General try-on | Explicitly forces garment/accessory presence |
| Best use | Standard clothing transfer | Multi-item outfits, hats, shoes, watches, logos |
| Risk | More natural if prompt is enough | More prompt pressure, possible pose/body drift |
| Seed offset in experiments | `+100` | `+200` |

### Strong Prompt Requirements

For hard accessory cases, the strong prompt should explicitly state:

```text
The final person must clearly wear all target items.
Do not omit the hat.
Do not omit the shoes.
Keep the hat on the head.
Keep the shoes visible on both feet.
Preserve face, hair, body shape, pose, legs, feet, and background.
```

### Inputs And Outputs

| Item | Value |
|---|---|
| Primary input | `person_image + accessory-aware composed references + strong Klein prompt + seed` |
| Main parameters | `steps=28`, `width=768`, `height=1024`, `guidance_scale=2.5`, `lora_scale=1.0` |
| Final output | `result.png` |

## Klein Reference Composition Rules

Klein does not receive a single garment image in the same way IDM does. It receives a list of three images:

```text
image = [person_canvas, top_reference_canvas, bottom_reference_canvas]
```

The reference builder should follow these rules:

| Case | `top_reference_canvas` | `bottom_reference_canvas` | Notes |
|---|---|---|---|
| `upper_body` | upper garment | person or blank/neutral lower reference | Preserve lower body. |
| `lower_body` | person or blank/neutral upper reference | lower garment | Preserve upper body. |
| `men_underwear` | person or blank/neutral upper reference | adult men's underwear bottom | Preserve upper body and legs outside target region. |
| `women_underwear` | person or blank/neutral upper reference | adult women's underwear bottom | Preserve upper body and legs outside target region. |
| `women_bra` | adult women's bra or upper innerwear | person or blank/neutral lower reference | Preserve lower body and abdomen outside target region. |
| `dress` | dress or dress+hat composition | dress or dress+shoes composition | Used for one-piece garment. |
| `full_outfit` | top or top+hat composition | bottom or bottom+shoes composition | Used for multi-item outfit. |
| `accessory` | accessory composition | accessory or person-preserve reference | IDM is skipped; Klein is stress-test only. |
| `dress + hat + shoes` | hat + dress canvas | dress + shoes canvas | Used for sample 015 accessory fix. |

## Recommended ComfyUI Custom Nodes

To make this executable inside ComfyUI, create these custom nodes:

| Custom node | Wraps existing project logic |
|---|---|
| `VTON_LoadSample` | Loads person, garment, accessory paths and category. |
| `VTON_BuildIDMPrompt` | Wraps IDM prompt template. |
| `VTON_AgnosticMask` | Wraps `create_agnostic_mask`. |
| `VTON_GarmentSegment` | Wraps project garment segmentation/normalization. |
| `VTON_IDMRun` | Wraps official IDM-VTON command execution. |
| `VTON_BuildKleinReferences` | Wraps reference composition used by Klein experiments. |
| `VTON_BuildKleinPrompt` | Wraps default/strong Klein prompt construction. |
| `VTON_LoadFlux2KleinLocal` | Loads `Flux2KleinPipeline.from_pretrained(..., local_files_only=True)`. |
| `VTON_LoadTryOnLoRA` | Loads local Try-On LoRA safetensors. |
| `VTON_KleinSampler` | Calls the local Klein pipeline with person/top/bottom references. |
| `VTON_SaveRunArtifacts` | Writes `result.png`, `status.json`, `prompt.txt`, and `reference_config.json`. |

## Report Mapping

| Report column | Workflow |
|---|---|
| `IDM` | Method 1 |
| `IDM expanded` | Method 2 |
| `Klein 4-step` | Method 3 |
| `Klein 28` | Method 4 |
| `Klein 28 strong` | Method 5 |
