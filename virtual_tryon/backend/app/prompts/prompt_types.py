from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


PromptCategory = Literal[
    "upper_body",
    "lower_body",
    "dress",
    "full_outfit",
    "men_underwear",
    "women_underwear",
    "women_bra",
    "accessory",
    "accessory_stress",
]
ReplacementMode = Literal[
    "replace_only_target",
    "replace_full_outfit",
    "refine_only_mask",
    "repair_artifact",
    "add_accessory",
]
AgeGroup = Literal["adult", "child"]


class PromptVariant(str, Enum):
    DEFAULT = "default"
    STRONG_REMOVE_OLD_GARMENT = "strong_remove_old_garment"
    IDENTITY_STRICT = "identity_strict"
    ACCESSORY_STRESS = "accessory_stress"
    FLUX_LOCAL_REFINE = "flux_local_refine"
    CATVTON_MINIMAL = "catvton_minimal"
    ADETAILER_REPAIR = "adetailer_repair"


class EngineMode(str, Enum):
    IDM = "idm"
    IDM_MASK_EXPANDED = "idm_mask_expanded"
    IDM_MASK_EXPANDED_FLUX = "idm_mask_expanded_flux"
    KLEIN_LORA = "klein_lora"
    CATVTON = "catvton"
    ADETAILER_REPAIR = "adetailer_repair"


class PromptTarget(BaseModel):
    category: PromptCategory
    target_regions: list[str] = Field(default_factory=list)
    preserve_regions: list[str] = Field(default_factory=list)
    replacement_mode: ReplacementMode = "replace_only_target"
    is_accessory_stress_test: bool = False


class GarmentDescription(BaseModel):
    garment_type: str
    color: str
    material: str | None = None
    silhouette: str | None = None
    fit: str | None = None
    pattern: str | None = None
    logos_or_text: str | None = None
    structural_details: list[str] = Field(default_factory=list)
    preserve_detail_priority: list[str] = Field(default_factory=list)


class PersonDescription(BaseModel):
    age_group: AgeGroup
    gender_presentation: str | None = None
    body_description: str | None = None
    pose: str
    hair: str | None = None
    current_outfit: str
    background: str | None = None
    must_preserve: list[str] = Field(default_factory=list)


class PromptBuildRequest(BaseModel):
    engine_mode: EngineMode
    testcase_id: str | None = None
    person: PersonDescription
    target: PromptTarget
    top_garment: GarmentDescription | None = None
    bottom_garment: GarmentDescription | None = None
    dress: GarmentDescription | None = None
    accessories: list[GarmentDescription] = Field(default_factory=list)
    category: PromptCategory
    variant: PromptVariant = PromptVariant.DEFAULT
    use_strong_old_garment_removal: bool = False
    preserve_original_bottom: bool = True
    extra_user_instruction: str | None = None


class PromptBuildResult(BaseModel):
    positive_prompt: str
    negative_prompt: str | None = None
    prompt_variant: PromptVariant
    engine_mode: EngineMode
    metadata: dict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    core_prompt: str | None = None
    refine_prompt: str | None = None
