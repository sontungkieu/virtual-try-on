import { Shirt, Upload } from "lucide-react";
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

export function UploadGarment() {
  const state = useTryOnStore();
  const setField = state.setField;
  const topLabel = state.category === "women_bra" ? "Bra" : "Top";
  const bottomLabel =
    state.category === "men_underwear"
      ? "Men underwear"
      : state.category === "women_underwear"
        ? "Women underwear"
        : "Bottom";

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

      <div className="garment-grid">
        <UploadTile
          title={topLabel}
          file={state.topImage}
          ariaLabel={`${topLabel} garment image`}
          icon={Shirt}
          onChange={(file) => setField("topImage", file)}
        />
        <UploadTile
          title={bottomLabel}
          file={state.bottomImage}
          ariaLabel={`${bottomLabel} garment image`}
          icon={Upload}
          onChange={(file) => setField("bottomImage", file)}
        />
        <UploadTile
          title="Dress"
          file={state.dressImage}
          ariaLabel="Dress garment image"
          icon={Upload}
          onChange={(file) => setField("dressImage", file)}
        />
      </div>
    </section>
  );
}
