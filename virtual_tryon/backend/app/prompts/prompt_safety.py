from __future__ import annotations

import re

from app.prompts.prompt_types import PromptBuildRequest


CHILD_FORBIDDEN_TERMS = {
    "underwear",
    "brief",
    "briefs",
    "bikini",
    "lingerie",
    "bra",
    "swimwear",
    "seductive",
    "sexy",
    "provocative",
}
SEXUALIZED_TERMS = {"seductive", "sexy", "provocative", "revealing"}


class PromptSafetyError(ValueError):
    pass


def _request_text(request: PromptBuildRequest) -> str:
    values = [
        request.person.current_outfit,
        request.person.body_description or "",
        request.extra_user_instruction or "",
    ]
    garments = [
        request.top_garment,
        request.bottom_garment,
        request.dress,
        *request.accessories,
    ]
    for garment in garments:
        if garment:
            values.extend(
                [
                    garment.garment_type,
                    garment.pattern or "",
                    garment.logos_or_text or "",
                    " ".join(garment.structural_details),
                ]
            )
    return " ".join(values).lower()


def validate_prompt_request(request: PromptBuildRequest) -> list[str]:
    text = _request_text(request)
    warnings: list[str] = []
    if request.person.age_group == "child":
        found = sorted(term for term in CHILD_FORBIDDEN_TERMS if re.search(rf"\b{re.escape(term)}\b", text))
        if found:
            raise PromptSafetyError(
                "Child test cases only support normal everyday clothing; prohibited terms: "
                + ", ".join(found)
            )
    sexualized = sorted(term for term in SEXUALIZED_TERMS if re.search(rf"\b{re.escape(term)}\b", text))
    if sexualized:
        raise PromptSafetyError("Sexualized prompt language is not allowed: " + ", ".join(sexualized))
    if request.target.is_accessory_stress_test or request.category in {"accessory", "accessory_stress"}:
        warnings.append(
            "Accessory stress test: IDM-VTON and CatVTON are clothing-focused and may be unreliable."
        )
    garments = [
        request.top_garment,
        request.bottom_garment,
        request.dress,
        *request.accessories,
    ]
    if any(garment and garment.logos_or_text for garment in garments):
        warnings.append(
            "Exact text or logo reproduction may be imperfect; preserve placement, colors, and pattern structure."
        )
    return warnings

