# Prompting

Virtual Try-On uses engine-specific prompts because the engines consume text differently:

- IDM-VTON is the default core engine and benefits from detailed person, garment, task, preservation, and old-garment-removal text.
- IDM-VTON expanded mask uses the same prompt plus a hemline/remnant-removal instruction.
- FLUX local refine is only a post-process editor, so its prompt says to refine only the masked clothing region and preserve all unmasked pixels.
- Klein LoRA is experimental; prompts must start with `TRYON` and include `The final image is a full body shot.`
- CatVTON treats text as auxiliary, so the prompt stays minimal and mask-focused.
- ADetailer-like repair is local artifact repair after a generated output exists.

The source of truth lives in:

```text
backend/app/prompts/
  prompt_types.py
  engine_prompt_templates.py
  prompt_builder.py
  prompt_safety.py
  testcase_prompt_library.py
configs/prompts.yaml
```

## Variants

Supported variants:

- `default`
- `strong_remove_old_garment`
- `identity_strict`
- `accessory_stress`
- `flux_local_refine`
- `catvton_minimal`
- `adetailer_repair`

## Safety Rules

Child test cases only use ordinary everyday clothing such as shirts, sweaters, skirts, pants, and dresses. Adult underwear/swimwear cases must use neutral terms such as `adult brief underwear`, `bikini bottom`, or `sports-bra style top`, and must avoid sexualized wording. Accessory cases are marked as stress tests because IDM-VTON and CatVTON are clothing-focused.

## Generate Prompts

```bash
uv run python scripts/generate_prompts.py \
  --testcase tc10 \
  --engine all \
  --variant strong_remove_old_garment \
  --output data/outputs/prompts_tc10
```

This writes per-engine prompt files plus `prompts_summary.json` and `prompts_summary.md`. It does not run inference.

## Prompt Ablation

```bash
uv run python scripts/run_prompt_ablation.py \
  --sample data/eval_set/sample_001 \
  --testcase-id tc10 \
  --engine idm,klein_lora \
  --variants default,strong_remove_old_garment,identity_strict \
  --output data/outputs/prompt_ablation_tc10
```

If an engine is unavailable, the row is skipped cleanly. Prompt artifacts are saved for every prompt variant, and subjective ratings stay blank in `manual_ratings_prompt_ablation.csv`.
