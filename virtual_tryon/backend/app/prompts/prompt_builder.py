from __future__ import annotations

import hashlib
import json

from app.prompts.engine_prompt_templates import (
    ADETAILER_REPAIR_TEMPLATE,
    CATVTON_TEMPLATE,
    FLUX_LOCAL_REFINE_TEMPLATE,
    GLOBAL_NEGATIVE_PROMPT,
    IDM_EXPANDED_MASK_ADDON,
    IDM_TEMPLATE,
    KLEIN_TEMPLATE,
)
from app.prompts.prompt_safety import validate_prompt_request
from app.prompts.prompt_types import (
    EngineMode,
    GarmentDescription,
    PersonDescription,
    PromptBuildRequest,
    PromptBuildResult,
    PromptVariant,
)


def _join_human(values: list[str]) -> str:
    clean = list(dict.fromkeys(value.strip() for value in values if value and value.strip()))
    if not clean:
        return "identity, pose, body shape, background, and unmasked regions"
    if len(clean) == 1:
        return clean[0]
    return ", ".join(clean[:-1]) + f", and {clean[-1]}"


def _person_description(person: PersonDescription) -> str:
    parts = []
    gender = person.gender_presentation or "person"
    if person.age_group == "adult" and "adult" not in gender.lower():
        gender = f"adult {gender}"
    parts.append(gender)
    if person.body_description:
        parts.append(person.body_description)
    parts.append(person.pose)
    if person.hair:
        parts.append(f"with {person.hair}")
    parts.append(f"currently wearing {person.current_outfit}")
    if person.background:
        parts.append(f"against {person.background}")
    return ", ".join(parts)


def _garment_description(garment: GarmentDescription) -> str:
    parts = [garment.color, garment.garment_type]
    qualifiers = [
        garment.material,
        garment.silhouette,
        garment.fit,
        garment.pattern,
    ]
    if any(qualifiers):
        parts.append("with " + _join_human([value for value in qualifiers if value]))
    if garment.logos_or_text:
        parts.append(f"featuring {garment.logos_or_text}")
    if garment.structural_details:
        parts.append(f"including {_join_human(garment.structural_details)}")
    return " ".join(parts)


def _all_garments(request: PromptBuildRequest) -> list[GarmentDescription]:
    return [
        garment
        for garment in [
            request.top_garment,
            request.bottom_garment,
            request.dress,
            *request.accessories,
        ]
        if garment is not None
    ]


def _combined_garment_description(request: PromptBuildRequest) -> str:
    garments = _all_garments(request)
    if not garments:
        return "the reference garment"
    return _join_human([_garment_description(garment) for garment in garments])


def _detail_list(request: PromptBuildRequest) -> str:
    details: list[str] = []
    for garment in _all_garments(request):
        details.extend(garment.preserve_detail_priority)
        details.extend(garment.structural_details)
    return _join_human(details or ["color", "material", "cut", "fit", "pattern", "logos", "seams"])


def _replacement_scope(request: PromptBuildRequest) -> str:
    if request.category == "upper_body":
        return "only the upper-body clothing region"
    if request.category == "lower_body":
        return "only the lower-body clothing region"
    if request.category in {"men_underwear", "women_underwear"}:
        return "only the adult underwear bottom region"
    if request.category == "women_bra":
        return "only the adult bra or upper innerwear region"
    if request.category == "dress":
        return "the current outfit"
    if request.category == "full_outfit":
        return "the full outfit"
    return "the specified accessory region"


def _preserve_list(request: PromptBuildRequest) -> str:
    preserve = [*request.person.must_preserve, *request.target.preserve_regions]
    if request.category in {"upper_body", "women_bra"} and request.preserve_original_bottom:
        preserve.append("original lower-body clothing")
    if request.category in {"lower_body", "men_underwear", "women_underwear"}:
        preserve.append("original upper-body clothing")
    if request.category in {"men_underwear", "women_underwear"}:
        preserve.append("legs outside the underwear target region")
    if request.category == "women_bra":
        preserve.append("abdomen outside the bra target region")
    return _join_human(preserve)


def _removal_instruction(request: PromptBuildRequest) -> str:
    strong = (
        request.use_strong_old_garment_removal
        or request.variant == PromptVariant.STRONG_REMOVE_OLD_GARMENT
    )
    if request.target.replacement_mode == "add_accessory":
        return "Add the requested accessories without replacing unrelated clothing."
    if strong:
        return (
            "Remove the original garment completely in the replaced region, including every visible remnant "
            "at boundaries, layers, and the hemline."
        )
    return "Remove the original garment completely in that region."


def _normalize_klein_manual_prompt(value: str) -> str:
    body = " ".join(value.strip().split())
    if body.upper().startswith("TRYON"):
        body = body[5:].strip(" .")
    prompt = f"TRYON {body}".rstrip()
    if "the final image is a full body shot." not in prompt.lower():
        prompt = f"{prompt.rstrip(' .')}. The final image is a full body shot."
    return prompt


def _prompt_hash(result_payload: dict) -> str:
    encoded = json.dumps(
        result_payload,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_prompt(request: PromptBuildRequest) -> PromptBuildResult:
    warnings = validate_prompt_request(request)
    person = _person_description(request.person)
    garment = _combined_garment_description(request)
    preserve = _preserve_list(request)
    removal = _removal_instruction(request)
    core_prompt: str | None = None
    refine_prompt: str | None = None

    if request.engine_mode == EngineMode.IDM:
        positive = IDM_TEMPLATE.format(
            person_description=person,
            garment_description=garment,
            removal_instruction=removal,
            preserve_list=preserve,
            detail_list=_detail_list(request),
        )
        core_prompt = positive
    elif request.engine_mode == EngineMode.IDM_MASK_EXPANDED:
        base = IDM_TEMPLATE.format(
            person_description=person,
            garment_description=garment,
            removal_instruction=removal,
            preserve_list=preserve,
            detail_list=_detail_list(request),
        )
        positive = f"{base} {IDM_EXPANDED_MASK_ADDON}"
        core_prompt = positive
    elif request.engine_mode == EngineMode.IDM_MASK_EXPANDED_FLUX:
        core_prompt = (
            IDM_TEMPLATE.format(
                person_description=person,
                garment_description=garment,
                removal_instruction=removal,
                preserve_list=preserve,
                detail_list=_detail_list(request),
            )
            + " "
            + IDM_EXPANDED_MASK_ADDON
        )
        refine_prompt = FLUX_LOCAL_REFINE_TEMPLATE
        positive = core_prompt
    elif request.engine_mode == EngineMode.KLEIN_LORA:
        if request.extra_user_instruction:
            positive = _normalize_klein_manual_prompt(request.extra_user_instruction)
        else:
            positive = KLEIN_TEMPLATE.format(
                person_description=person,
                replacement_scope=_replacement_scope(request),
                garment_description=garment,
                preserve_list=preserve,
                removal_instruction=removal,
            )
        core_prompt = positive
    elif request.engine_mode == EngineMode.CATVTON:
        positive = CATVTON_TEMPLATE
        core_prompt = positive
        if request.category in {"accessory", "accessory_stress"}:
            warnings.append("CatVTON does not reliably support accessory-only placement.")
    elif request.engine_mode == EngineMode.ADETAILER_REPAIR:
        positive = ADETAILER_REPAIR_TEMPLATE
        refine_prompt = positive
    else:
        raise ValueError(f"Unsupported engine mode: {request.engine_mode}")

    if request.variant == PromptVariant.IDENTITY_STRICT:
        positive = (
            f"{positive} Identity preservation is strict: keep facial identity, expression, hair, body "
            "proportions, pose, and all unmasked pixels unchanged."
        )
        if core_prompt:
            core_prompt = positive
    if request.variant == PromptVariant.FLUX_LOCAL_REFINE:
        refine_prompt = FLUX_LOCAL_REFINE_TEMPLATE
    if request.variant == PromptVariant.CATVTON_MINIMAL:
        positive = CATVTON_TEMPLATE
        core_prompt = positive
    if request.variant == PromptVariant.ADETAILER_REPAIR:
        positive = ADETAILER_REPAIR_TEMPLATE
        refine_prompt = positive

    metadata = {
        "testcase_id": request.testcase_id,
        "engine_mode": request.engine_mode.value,
        "variant": request.variant.value,
        "category": request.category,
        "target_regions": request.target.target_regions,
        "preserve_regions": request.target.preserve_regions,
        "is_accessory_stress_test": request.target.is_accessory_stress_test,
        "warnings": list(dict.fromkeys(warnings)),
    }
    hash_payload = {
        "positive_prompt": positive,
        "negative_prompt": GLOBAL_NEGATIVE_PROMPT,
        "core_prompt": core_prompt,
        "refine_prompt": refine_prompt,
        "metadata": metadata,
        "warnings": warnings,
    }
    metadata["prompt_hash"] = _prompt_hash(hash_payload)
    return PromptBuildResult(
        positive_prompt=positive,
        negative_prompt=GLOBAL_NEGATIVE_PROMPT,
        prompt_variant=request.variant,
        engine_mode=request.engine_mode,
        metadata=metadata,
        warnings=list(dict.fromkeys(warnings)),
        core_prompt=core_prompt,
        refine_prompt=refine_prompt,
    )
