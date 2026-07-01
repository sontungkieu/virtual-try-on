import { Upload } from "lucide-react";
import { useTryOnStore } from "../store/tryonStore";
import { UploadTile } from "./UploadTile";

export function UploadPerson() {
  const personImage = useTryOnStore((state) => state.personImage);
  const setField = useTryOnStore((state) => state.setField);

  return (
    <UploadTile
      title="Person"
      file={personImage}
      ariaLabel="Person image"
      icon={Upload}
      variant="person"
      onChange={(file) => setField("personImage", file)}
    />
  );
}
