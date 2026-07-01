# Future Work

## Old-Garment Remnant Detection

The current quality report can measure image change inside and outside a mask, but it cannot determine whether visible pixels belong to the original garment. A future detector should compare the person input, garment reference, generated output, and human-parsing regions to identify old-garment remnants near collars, sleeves, waistlines, and hemlines.

Until that signal exists, old-garment bleed-through is a manual review criterion and belongs in `manual_ratings.csv` or `manual_ratings_mask_ablation.csv`, not `quality_report.json`.

## Mask Selection

Use the upper-body hem expansion ablation to collect evidence before changing the production mask. A candidate mask should remove more of the old garment while preserving identity, hair, hands, pose, body shape, background, and the intended length and silhouette of the reference garment.

## Refinement

Local refinement should remain constrained to safe garment or boundary masks. Future work can replace the current fallback safe mask with parser-derived face, hair, and hand exclusions, then evaluate whether local FLUX refinement improves garment cleanup without introducing over-editing.
