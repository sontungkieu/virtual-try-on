import { Upload } from "lucide-react";
import { useTryOnStore } from "../store/tryonStore";

export function UploadPerson() {
  const personImage = useTryOnStore((state) => state.personImage);
  const setField = useTryOnStore((state) => state.setField);

  return (
    <label className="upload-panel">
      <span className="upload-icon"><Upload size={18} /></span>
      <span className="upload-title">Person</span>
      <span className="upload-file">{personImage?.name ?? "No file selected"}</span>
      <input
        type="file"
        aria-label="Person image"
        accept="image/png,image/jpeg,image/webp"
        onChange={(event) => setField("personImage", event.target.files?.[0])}
      />
    </label>
  );
}
