from __future__ import annotations


IDM_TEMPLATE = (
    "Person: {person_description}. Garment: {garment_description}. "
    "Task: Replace only the target clothing region with the reference garment. "
    "{removal_instruction} Preserve {preserve_list}. "
    "Match the reference garment's {detail_list} as accurately as possible."
)

IDM_EXPANDED_MASK_ADDON = (
    "The replacement region includes the full garment area and the garment hemline. "
    "Do not preserve any visible part of the old garment inside the replaced region."
)

FLUX_LOCAL_REFINE_TEMPLATE = (
    "Refine only the masked clothing region. Remove any remaining old garment artifacts inside the mask. "
    "Preserve the person's face, hair, hands, skin tone, body shape, pose, background, and all unmasked "
    "clothing exactly. Do not change the camera angle, identity, body proportions, or background. "
    "Improve garment boundary, fabric realism, folds, shadows, logo/pattern clarity, and fit."
)

KLEIN_TEMPLATE = (
    "TRYON {person_description}. Replace {replacement_scope} with {garment_description} as shown in the "
    "reference image(s). Preserve {preserve_list}. {removal_instruction} Match the garment color, material, "
    "cut, fit, pattern, logos, seams, waistband, collar, sleeves, and visible details as accurately as "
    "possible. The final image is a full body shot."
)

CATVTON_TEMPLATE = (
    "Replace the masked clothing region with the reference garment. "
    "Preserve identity, pose, body shape, background, and unmasked regions."
)

ADETAILER_REPAIR_TEMPLATE = (
    "Fix only the masked artifact region. Remove old garment remnants and blend the new garment naturally. "
    "Preserve all unmasked pixels exactly."
)

GLOBAL_NEGATIVE_PROMPT = (
    "Do not change face, identity, facial expression, hairstyle, skin tone, body proportions, hands, legs, "
    "pose, camera angle, background, lighting, or unmasked clothing. Do not add extra limbs, distorted hands, "
    "warped body, duplicate garments, random logos, wrong text, wrong patterns, or unrelated accessories."
)

