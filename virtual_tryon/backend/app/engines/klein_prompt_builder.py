from __future__ import annotations


FULL_BODY_SENTENCE = "The final image is a full body shot."
DEFAULT_PERSON_DESCRIPTION = "blonde woman standing front-facing"
DEFAULT_TOP_DESCRIPTION = "the blue velvet wrap V-neck short-sleeve top shown in the reference image"
DEFAULT_BOTTOM_DESCRIPTION = "black pants"
DEFAULT_INNERWEAR_BOTTOM_DESCRIPTIONS = {
    "men_underwear": "adult men's brief underwear",
    "women_underwear": "adult women's brief underwear",
}
DEFAULT_BRA_DESCRIPTION = "adult bra or upper innerwear garment"


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = " ".join(value.strip().split())
    return normalized or None


def _without_tryon_prefix(value: str) -> str:
    stripped = value.strip()
    if stripped.upper().startswith("TRYON"):
        return stripped[5:].strip(" .")
    return stripped.strip(" .")


def _ensure_full_body(prompt: str) -> str:
    if FULL_BODY_SENTENCE.lower() in prompt.lower():
        return prompt
    return f"{prompt.rstrip(' .')}. {FULL_BODY_SENTENCE}"


def _normalize_user_prompt(prompt: str) -> str:
    body = _without_tryon_prefix(prompt)
    if not body:
        return f"TRYON {FULL_BODY_SENTENCE}"
    return _ensure_full_body(f"TRYON {body}")


def build_klein_tryon_prompt(
    person_description: str | None,
    top_description: str | None,
    bottom_description: str | None,
    category: str,
    preserve_original_bottom: bool = True,
    extra_instruction: str | None = None,
) -> str:
    """Build the prompt format expected by the Klein Try-On LoRA baseline."""

    extra_instruction = _clean(extra_instruction)
    if extra_instruction:
        return _normalize_user_prompt(extra_instruction)

    person = _clean(person_description) or DEFAULT_PERSON_DESCRIPTION
    top = _clean(top_description) or (
        DEFAULT_BRA_DESCRIPTION if category == "women_bra" else DEFAULT_TOP_DESCRIPTION
    )
    bottom = _clean(bottom_description) or DEFAULT_INNERWEAR_BOTTOM_DESCRIPTIONS.get(
        category,
        DEFAULT_BOTTOM_DESCRIPTION,
    )

    if category == "full_outfit":
        body = (
            f"{person}. Replace the full outfit with {top} and {bottom} shown in the reference images. "
            "Preserve the person's face, hair, hands, body shape, pose, and background."
        )
    elif category == "upper_body":
        preserve_bottom = (
            f"Preserve the person's face, hair, hands, body shape, {bottom}, pose, and background."
            if preserve_original_bottom
            else "Preserve the person's face, hair, hands, body shape, pose, and background."
        )
        body = (
            f"{person}. Replace the upper outfit completely with {top}. "
            "Remove the original pink sleeveless shirt entirely. "
            f"{preserve_bottom}"
        )
    elif category == "lower_body":
        body = (
            f"{person}. Replace the lower outfit completely with {bottom} shown in the reference image. "
            "Preserve the person's face, hair, hands, body shape, upper outfit, pose, and background."
        )
    elif category in {"men_underwear", "women_underwear"}:
        body = (
            f"{person}. Replace only the adult underwear bottom region with {bottom} shown in the reference image. "
            "Preserve the person's face, hair, hands, body shape, upper outfit, legs outside the target region, pose, and background."
        )
    elif category == "women_bra":
        body = (
            f"{person}. Replace only the adult bra or upper innerwear region with {top}. "
            "Preserve the person's face, hair, hands, body shape, abdomen outside the target region, lower outfit, pose, and background."
        )
    elif category == "dress":
        body = (
            f"{person}. Replace the outfit completely with {top} shown in the reference image. "
            "Preserve the person's face, hair, hands, body shape, pose, and background."
        )
    else:
        body = (
            f"{person}. Replace the outfit with the garments shown in the reference images. "
            "Preserve the person's face, hair, hands, body shape, pose, and background."
        )

    return _ensure_full_body(f"TRYON {body}")
