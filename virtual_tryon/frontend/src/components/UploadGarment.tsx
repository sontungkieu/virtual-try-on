import { Shirt, Upload } from "lucide-react";
import { garmentLabelForSlot, garmentSlotsForCategory, type GarmentSlot } from "../lib/category";
import { Category, useTryOnStore } from "../store/tryonStore";
import { UploadTile } from "./UploadTile";

const categories: { value: Category; label: string }[] = [
  { value: "upper_body", label: "Top" },
  { value: "lower_body", label: "Bottom" },
  { value: "dress", label: "Dress" },
  { value: "full_outfit", label: "Outfit" },
  { value: "men_underwear", label: "Men underwear" },
  { value: "women_underwear", label: "Women underwear" },
  { value: "women_bra", label: "Bra" }
];

const slotIcons: Record<GarmentSlot, typeof Shirt> = {
  top: Shirt,
  bottom: Upload,
  dress: Upload
};

export function UploadGarment() {
  const state = useTryOnStore();
  const setField = state.setField;
  const visibleSlots = garmentSlotsForCategory(state.category);

  function fileForSlot(slot: GarmentSlot) {
    if (slot === "top") return state.topImage;
    if (slot === "bottom") return state.bottomImage;
    return state.dressImage;
  }

  function setSlotFile(slot: GarmentSlot, file?: File) {
    if (slot === "top") {
      setField("topImage", file);
    } else if (slot === "bottom") {
      setField("bottomImage", file);
    } else {
      setField("dressImage", file);
    }
  }

  return (
    <section className="garment-section">
      <label className="category-select">
        <span>Category</span>
        <select value={state.category} onChange={(event) => setField("category", event.target.value as Category)}>
          {categories.map((item) => (
            <option value={item.value} key={item.value}>{item.label}</option>
          ))}
        </select>
      </label>

      <div className={`garment-grid garment-grid-${visibleSlots.length}`}>
        {visibleSlots.map((slot) => {
          const label = garmentLabelForSlot(slot, state.category);
          return (
            <UploadTile
              key={slot}
              title={label}
              file={fileForSlot(slot)}
              ariaLabel={`${label} garment image`}
              icon={slotIcons[slot]}
              onChange={(file) => setSlotFile(slot, file)}
            />
          );
        })}
      </div>
    </section>
  );
}
