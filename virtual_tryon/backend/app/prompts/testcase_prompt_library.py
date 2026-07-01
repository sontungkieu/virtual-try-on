from __future__ import annotations

from copy import deepcopy

from pydantic import BaseModel, Field

from app.prompts.prompt_types import (
    EngineMode,
    GarmentDescription,
    PersonDescription,
    PromptBuildRequest,
    PromptTarget,
    PromptVariant,
)


DEFAULT_VARIANTS = [
    PromptVariant.DEFAULT,
    PromptVariant.STRONG_REMOVE_OLD_GARMENT,
    PromptVariant.IDENTITY_STRICT,
    PromptVariant.FLUX_LOCAL_REFINE,
    PromptVariant.CATVTON_MINIMAL,
    PromptVariant.ADETAILER_REPAIR,
]
DEFAULT_ENGINES = [
    EngineMode.IDM,
    EngineMode.IDM_MASK_EXPANDED,
    EngineMode.IDM_MASK_EXPANDED_FLUX,
    EngineMode.KLEIN_LORA,
    EngineMode.CATVTON,
    EngineMode.ADETAILER_REPAIR,
]
BASE_PRESERVE = ["face", "identity", "hair", "hands", "skin tone", "body shape", "pose", "background"]


class TestcasePromptDefinition(BaseModel):
    testcase_id: str
    category: str
    person: PersonDescription
    target: PromptTarget
    top_garment: GarmentDescription | None = None
    bottom_garment: GarmentDescription | None = None
    dress: GarmentDescription | None = None
    accessories: list[GarmentDescription] = Field(default_factory=list)
    prompt_variants: list[PromptVariant] = Field(default_factory=lambda: list(DEFAULT_VARIANTS))
    recommended_engines: list[EngineMode] = Field(default_factory=lambda: list(DEFAULT_ENGINES))
    stress_test_flags: list[str] = Field(default_factory=list)

    def build_request(
        self,
        engine_mode: EngineMode,
        variant: PromptVariant = PromptVariant.DEFAULT,
    ) -> PromptBuildRequest:
        return PromptBuildRequest(
            engine_mode=engine_mode,
            testcase_id=self.testcase_id,
            person=deepcopy(self.person),
            target=deepcopy(self.target),
            top_garment=deepcopy(self.top_garment),
            bottom_garment=deepcopy(self.bottom_garment),
            dress=deepcopy(self.dress),
            accessories=deepcopy(self.accessories),
            category=self.target.category,
            variant=variant,
            use_strong_old_garment_removal=variant == PromptVariant.STRONG_REMOVE_OLD_GARMENT,
            preserve_original_bottom=self.target.category == "upper_body",
        )


def _person(
    *,
    age_group: str = "adult",
    gender: str,
    pose: str,
    current_outfit: str,
    hair: str | None = None,
    background: str = "the original background",
    body: str | None = None,
) -> PersonDescription:
    return PersonDescription(
        age_group=age_group,
        gender_presentation=gender,
        body_description=body,
        pose=pose,
        hair=hair,
        current_outfit=current_outfit,
        background=background,
        must_preserve=list(BASE_PRESERVE),
    )


def _target(category: str, regions: list[str], *, accessory: bool = False) -> PromptTarget:
    preserve = list(BASE_PRESERVE)
    if category == "upper_body":
        preserve.append("original lower-body clothing")
    elif category == "lower_body":
        preserve.append("original upper-body clothing")
    return PromptTarget(
        category=category,
        target_regions=regions,
        preserve_regions=preserve,
        replacement_mode=(
            "add_accessory"
            if accessory
            else "replace_full_outfit"
            if category in {"dress", "full_outfit"}
            else "replace_only_target"
        ),
        is_accessory_stress_test=accessory,
    )


def _garment(
    garment_type: str,
    color: str,
    *,
    material: str | None = None,
    silhouette: str | None = None,
    fit: str | None = None,
    pattern: str | None = None,
    logo: str | None = None,
    details: list[str] | None = None,
) -> GarmentDescription:
    priorities = ["color", "cut", "fit"]
    if pattern:
        priorities.append("pattern")
    if logo:
        priorities.append("logo placement")
    return GarmentDescription(
        garment_type=garment_type,
        color=color,
        material=material,
        silhouette=silhouette,
        fit=fit,
        pattern=pattern,
        logos_or_text=logo,
        structural_details=details or [],
        preserve_detail_priority=priorities,
    )


TESTCASES: dict[str, TestcasePromptDefinition] = {
    "tc1": TestcasePromptDefinition(
        testcase_id="tc1",
        category="upper_body",
        person=_person(gender="man", pose="standing front-facing", current_outfit="casual upper-body clothing"),
        target=_target("upper_body", ["torso", "shirt sleeves", "shirt hem"]),
        top_garment=_garment("short-sleeve button-up shirt", "navy", fit="regular", details=["button placket", "collar"]),
    ),
    "tc2": TestcasePromptDefinition(
        testcase_id="tc2",
        category="lower_body",
        person=_person(gender="man", pose="standing front-facing", current_outfit="casual trousers and top"),
        target=_target("lower_body", ["waist", "hips", "legs"]),
        bottom_garment=_garment("tailored trousers", "dark charcoal", material="woven suiting", fit="tailored", details=["waistband", "pressed legs"]),
    ),
    "tc3": TestcasePromptDefinition(
        testcase_id="tc3",
        category="upper_body",
        person=_person(gender="woman", pose="standing front-facing", current_outfit="casual top and trousers"),
        target=_target("upper_body", ["torso", "sleeves", "shirt hem"]),
        top_garment=_garment("oversized graphic T-shirt", "black", fit="oversized", logo="reference graphic print", details=["crew neck", "wide sleeves"]),
    ),
    "tc4": TestcasePromptDefinition(
        testcase_id="tc4",
        category="lower_body",
        person=_person(gender="woman", pose="standing front-facing", current_outfit="top and fitted bottoms"),
        target=_target("lower_body", ["waist", "hips", "legs"]),
        bottom_garment=_garment("leggings", "light pastel blue", material="stretch fabric", fit="close fitted", details=["high waistband"]),
    ),
    "tc5": TestcasePromptDefinition(
        testcase_id="tc5",
        category="dress",
        person=_person(gender="woman", pose="walking on a runway", current_outfit="runway outfit", background="runway scene"),
        target=_target("dress", ["full torso", "waist", "skirt"]),
        dress=_garment("sleeveless mini dress", "yellow", silhouette="fit-and-flare", fit="fitted bodice", details=["wide straps", "flared skirt"]),
        stress_test_flags=["runway_pose"],
    ),
    "tc6": TestcasePromptDefinition(
        testcase_id="tc6",
        category="upper_body",
        person=_person(gender="woman", pose="editorial standing pose", current_outfit="editorial outfit", background="editorial studio"),
        target=_target("upper_body", ["torso", "sleeves", "shirt hem"]),
        top_garment=_garment("oversized graphic T-shirt", "black", fit="oversized", logo="HAPPINESS graphic text", details=["crew neck", "long hem"]),
        stress_test_flags=["editorial_pose", "logo_text"],
    ),
    "tc7": TestcasePromptDefinition(
        testcase_id="tc7",
        category="upper_body",
        person=_person(gender="man", pose="seated", current_outfit="casual shirt and trousers"),
        target=_target("upper_body", ["torso", "long sleeves", "shirt hem"]),
        top_garment=_garment("long-sleeve plaid shirt", "blue and yellow", pattern="plaid", fit="regular", details=["collar", "button placket", "cuffs"]),
        stress_test_flags=["seated_pose"],
    ),
    "tc8": TestcasePromptDefinition(
        testcase_id="tc8",
        category="upper_body",
        person=_person(gender="adult person", pose="standing in a streetwear pose", current_outfit="streetwear outfit", background="outdoor street scene"),
        target=_target("upper_body", ["torso", "sleeves", "garment hem"]),
        top_garment=_garment("streetwear top", "reference colors", fit="relaxed", details=["reference neckline", "reference sleeve shape"]),
        stress_test_flags=["outdoor_background", "streetwear_pose"],
    ),
    "tc9": TestcasePromptDefinition(
        testcase_id="tc9",
        category="lower_body",
        person=_person(gender="adult man", pose="standing front-facing", current_outfit="adult boxer shorts", body="adult body"),
        target=_target("lower_body", ["waist", "hips", "brief region"]),
        bottom_garment=_garment("adult brief underwear", "white", pattern="colorful geometric pattern", details=["black waistband", "leg openings"]),
    ),
    "tc10": TestcasePromptDefinition(
        testcase_id="tc10",
        category="lower_body",
        person=_person(gender="adult woman", pose="standing front-facing", current_outfit="black crop top and adult brief underwear", body="adult body"),
        target=_target("lower_body", ["waist", "hips", "brief region"]),
        bottom_garment=_garment("adult brief underwear", "blue and red", pattern="Superman-themed print", logo="Superman emblem and MAN OF STEEL waistband text", details=["red trim", "waistband", "front panel"]),
    ),
    "tc11": TestcasePromptDefinition(
        testcase_id="tc11",
        category="full_outfit",
        person=_person(age_group="child", gender="young girl", pose="standing front-facing with hands on hips", current_outfit="blue everyday shirt and shorts", background="gray studio"),
        target=_target("full_outfit", ["upper body", "lower body"]),
        top_garment=_garment("everyday striped sweater", "pink and white", pattern="horizontal stripes", fit="relaxed", details=["long sleeves", "crew neck"]),
        bottom_garment=_garment("everyday skirt", "white, black, and red", pattern="floral print", details=["black waistband"]),
    ),
    "tc12": TestcasePromptDefinition(
        testcase_id="tc12",
        category="accessory_stress",
        person=_person(gender="adult woman", pose="standing front-facing", current_outfit="navy sports top and adult bikini bottom", body="adult body"),
        target=_target("accessory_stress", ["lower-body garment", "head"], accessory=True),
        bottom_garment=_garment("adult brief underwear", "blue and red", pattern="Superman-themed print", logo="Superman emblem", details=["red trim", "waistband"]),
        accessories=[_garment("baseball cap", "white", logo="black letter A", details=["curved brim"])],
        stress_test_flags=["accessory", "multi_reference"],
    ),
    "tc13": TestcasePromptDefinition(
        testcase_id="tc13",
        category="accessory",
        person=_person(gender="adult man", pose="standing in a bodybuilding pose", current_outfit="dark adult brief underwear", body="adult muscular body"),
        target=_target("accessory", ["head", "wrist"], accessory=True),
        accessories=[
            _garment("baseball cap", "black", logo="gold FORCE embroidery", details=["curved brim"]),
            _garment("wristwatch", "silver and navy", material="metal", details=["bracelet", "round chronograph face"]),
        ],
        stress_test_flags=["accessory_only"],
    ),
    "tc14": TestcasePromptDefinition(
        testcase_id="tc14",
        category="accessory_stress",
        person=_person(gender="adult woman", pose="standing front-facing", current_outfit="navy sports underwear set", body="adult body"),
        target=_target("accessory_stress", ["upper body", "lower body", "head"], accessory=True),
        top_garment=_garment("sports-bra style top", "gray", material="stretch fabric", fit="close fitted", details=["wide straps", "scoop neck"]),
        bottom_garment=_garment("adult brief underwear", "blue and red", pattern="Superman-themed print", logo="Superman emblem", details=["red trim", "waistband"]),
        accessories=[_garment("baseball cap", "white", logo="black letter A", details=["curved brim"])],
        stress_test_flags=["multi_garment", "accessory"],
    ),
    "tc15": TestcasePromptDefinition(
        testcase_id="tc15",
        category="accessory_stress",
        person=_person(gender="adult woman", pose="standing front-facing", current_outfit="navy sports underwear set", body="adult body"),
        target=_target("accessory_stress", ["full outfit", "feet", "head"], accessory=True),
        dress=_garment("sleeveless dress", "yellow", silhouette="fit-and-flare", fit="fitted bodice", details=["wide straps", "flared skirt"]),
        accessories=[
            _garment("high-heeled shoes", "white", details=["pointed toe", "stiletto heel"]),
            _garment("bucket hat", "pink", pattern="monogram print", logo="round side emblem"),
        ],
        stress_test_flags=["multi_garment", "accessory", "dress"],
    ),
}


def normalize_testcase_id(testcase_id: str) -> str:
    value = testcase_id.strip().lower().replace("testcase_", "tc").replace("testcase", "tc")
    if value.isdigit():
        value = f"tc{value}"
    return value


def get_testcase(testcase_id: str) -> TestcasePromptDefinition:
    normalized = normalize_testcase_id(testcase_id)
    if normalized not in TESTCASES:
        raise KeyError(f"Unknown testcase_id '{testcase_id}'. Expected tc1 through tc15.")
    return TESTCASES[normalized].model_copy(deep=True)


def list_testcases() -> list[TestcasePromptDefinition]:
    return [TESTCASES[f"tc{index}"].model_copy(deep=True) for index in range(1, 16)]

