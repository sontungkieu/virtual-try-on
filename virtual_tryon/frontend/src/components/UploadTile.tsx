import { ImageIcon, Upload } from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { useEffect, useState } from "react";

type UploadTileProps = {
  title: string;
  file?: File;
  ariaLabel: string;
  onChange: (file?: File) => void;
  variant?: "person" | "garment";
  icon?: LucideIcon;
};

function useObjectUrl(file?: File) {
  const [url, setUrl] = useState<string>();

  useEffect(() => {
    if (!file) {
      setUrl(undefined);
      return undefined;
    }
    const objectUrl = URL.createObjectURL(file);
    setUrl(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [file]);

  return url;
}

export function UploadTile({
  title,
  file,
  ariaLabel,
  onChange,
  variant = "garment",
  icon: Icon = Upload
}: UploadTileProps) {
  const previewUrl = useObjectUrl(file);
  const fileName = file?.name ?? "No file selected";

  return (
    <label className={`upload-panel upload-panel-${variant}`}>
      <span className="upload-panel-meta">
        <span className="upload-icon"><Icon size={18} /></span>
        <span className="upload-copy">
          <span className="upload-title">{title}</span>
          <span className="upload-file" title={file?.name}>{fileName}</span>
        </span>
      </span>
      <span className="upload-preview">
        {previewUrl ? <img src={previewUrl} alt="" /> : <span className="upload-placeholder"><ImageIcon size={22} /></span>}
      </span>
      <input
        type="file"
        aria-label={ariaLabel}
        accept="image/png,image/jpeg,image/webp"
        onChange={(event) => onChange(event.target.files?.[0])}
      />
    </label>
  );
}
