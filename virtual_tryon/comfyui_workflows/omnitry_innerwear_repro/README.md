# Omnitry Innerwear Reproduction Workflows

These ComfyUI workflows reproduce the adult, non-sexual innerwear try-on cases from `data/inputs/omnitry/output_omnitry`.

Default engine: `idm_klein_hybrid_pro`
Default resolution: `512x768`
Default steps: `4`

## Files

| File | Type | Notes |
|---|---|---|
| `manifest.json` | metadata | Copied ComfyUI input names and source paths. |
| `*_api.json` | API prompt | Queue with ComfyUI `/prompt`. |
| `*_ui.workflow.json` | UI workflow | Load in the ComfyUI editor. |

## Recommended reusable UI workflows

| File | Default engine | Notes |
|---|---|---|
| `backend_tryon_dropdown_ui.workflow.json` | `idm_vton` | One UI graph with category, engine, resolution, and step dropdowns. |
| `backend_tryon_hybrid_pro_ui.workflow.json` | `idm_klein_hybrid_pro` | IDM resident torch.compile plus Klein bnb 4-bit hybrid preset. |
| `backend_tryon_expanded_mask_ui.workflow.json` | `idm_mask_expanded` | Expanded IDM mask preset for broader replacement regions. |
| `backend_tryon_klein_4bit_ui.workflow.json` | `klein_bnb_4bit` | Pure local Klein LoRA bnb 4-bit preset. |

The per-case UI workflows are retained for reproducibility, but the reusable files are enough for manual ComfyUI use because the backend node exposes category, engine, resolution, and step controls as dropdowns/inputs.

UI workflows include ComfyUI's generated seed `control after generate` widget set to `fixed`, so later backend API fields stay aligned.

## Inputs

The script copies inputs into `/workspace/ComfyUI/input/vton_omnitry_repro/`.

## Cases

| case | category | target | person | garment | seed |
|---|---|---|---|---|---|
| `female_underwear_01_61k71bursel__ac_sl1440` | `women_underwear` | `lower` | `vton_omnitry_repro/female_underwear_01_61k71bursel__ac_sl1440_person.jpg` | `vton_omnitry_repro/female_underwear_01_61k71bursel__ac_sl1440_garment.jpg` | `2026070202` |
| `female_underwear_02_kga1151_pk001_p6_15` | `women_underwear` | `lower` | `vton_omnitry_repro/female_underwear_02_kga1151_pk001_p6_15_person.jpg` | `vton_omnitry_repro/female_underwear_02_kga1151_pk001_p6_15_garment.webp` | `2026070203` |
| `female_underwear_03_type_01` | `women_underwear` | `lower` | `vton_omnitry_repro/female_underwear_03_type_01_person.jpg` | `vton_omnitry_repro/female_underwear_03_type_01_garment.png` | `2026070204` |
| `male_model_1_men_underwear_04_34` | `men_underwear` | `lower` | `vton_omnitry_repro/male_model_1_men_underwear_04_34_person.jpg` | `vton_omnitry_repro/male_model_1_men_underwear_04_34_garment.jpg` | `2026070205` |
| `male_model_1_men_underwear_05_ny01_nvwht_0105_s123_jky_1` | `men_underwear` | `lower` | `vton_omnitry_repro/male_model_1_men_underwear_05_ny01_nvwht_0105_s123_jky_1_person.jpg` | `vton_omnitry_repro/male_model_1_men_underwear_05_ny01_nvwht_0105_s123_jky_1_garment.webp` | `2026070206` |
| `male_model_1_men_underwear_06_sample_14` | `men_underwear` | `lower` | `vton_omnitry_repro/male_model_1_men_underwear_06_sample_14_person.jpg` | `vton_omnitry_repro/male_model_1_men_underwear_06_sample_14_garment.png` | `2026070207` |
| `male_model_1_men_underwear_07_underwers` | `men_underwear` | `lower` | `vton_omnitry_repro/male_model_1_men_underwear_07_underwers_person.jpg` | `vton_omnitry_repro/male_model_1_men_underwear_07_underwers_garment.webp` | `2026070208` |
| `male_model_1_men_underwear_08_s_l1200` | `men_underwear` | `lower` | `vton_omnitry_repro/male_model_1_men_underwear_08_s_l1200_person.jpg` | `vton_omnitry_repro/male_model_1_men_underwear_08_s_l1200_garment.png` | `2026070209` |
| `male_model_1_men_underwear_09_superman_logo_sport_briefs_6__67136` | `men_underwear` | `lower` | `vton_omnitry_repro/male_model_1_men_underwear_09_superman_logo_sport_briefs_6__67136_person.jpg` | `vton_omnitry_repro/male_model_1_men_underwear_09_superman_logo_sport_briefs_6__67136_garment.jpg` | `2026070210` |
| `male_model_2_men_underwear_10_34` | `men_underwear` | `lower` | `vton_omnitry_repro/male_model_2_men_underwear_10_34_person.jpg` | `vton_omnitry_repro/male_model_2_men_underwear_10_34_garment.jpg` | `2026070211` |
| `male_model_2_men_underwear_11_ny01_nvwht_0105_s123_jky_1` | `men_underwear` | `lower` | `vton_omnitry_repro/male_model_2_men_underwear_11_ny01_nvwht_0105_s123_jky_1_person.jpg` | `vton_omnitry_repro/male_model_2_men_underwear_11_ny01_nvwht_0105_s123_jky_1_garment.webp` | `2026070212` |
| `male_model_2_men_underwear_12_sample_14` | `men_underwear` | `lower` | `vton_omnitry_repro/male_model_2_men_underwear_12_sample_14_person.jpg` | `vton_omnitry_repro/male_model_2_men_underwear_12_sample_14_garment.png` | `2026070213` |
| `male_model_2_men_underwear_13_underwers` | `men_underwear` | `lower` | `vton_omnitry_repro/male_model_2_men_underwear_13_underwers_person.jpg` | `vton_omnitry_repro/male_model_2_men_underwear_13_underwers_garment.webp` | `2026070214` |
| `male_model_2_men_underwear_14_s_l1200` | `men_underwear` | `lower` | `vton_omnitry_repro/male_model_2_men_underwear_14_s_l1200_person.jpg` | `vton_omnitry_repro/male_model_2_men_underwear_14_s_l1200_garment.png` | `2026070215` |
| `male_model_2_men_underwear_15_superman_logo_sport_briefs_6__67136` | `men_underwear` | `lower` | `vton_omnitry_repro/male_model_2_men_underwear_15_superman_logo_sport_briefs_6__67136_person.jpg` | `vton_omnitry_repro/male_model_2_men_underwear_15_superman_logo_sport_briefs_6__67136_garment.jpg` | `2026070216` |
