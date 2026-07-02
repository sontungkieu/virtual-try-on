import type { Category } from "../store/tryonStore";

export type GarmentSlot = "top" | "bottom" | "dress";

export function garmentSlotsForCategory(category: Category): GarmentSlot[] {
  switch (category) {
    case "upper_body":
    case "women_bra":
      return ["top"];
    case "lower_body":
    case "men_underwear":
    case "women_underwear":
      return ["bottom"];
    case "dress":
      return ["dress"];
    case "full_outfit":
      return ["top", "bottom", "dress"];
    default:
      return ["top"];
  }
}

export function garmentLabelForSlot(slot: GarmentSlot, category: Category): string {
  if (slot === "top") return category === "women_bra" ? "Bra" : "Top";
  if (slot === "bottom") {
    if (category === "men_underwear") return "Men underwear";
    if (category === "women_underwear") return "Women underwear";
    return "Bottom";
  }
  return "Dress";
}
