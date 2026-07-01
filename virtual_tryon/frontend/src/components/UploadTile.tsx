import { ImageIcon, Upload } from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { gsap } from "gsap";

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
  const panelRef = useRef<HTMLLabelElement | null>(null);
  const previewUrl = useObjectUrl(file);
  const fileName = file?.name ?? "No file selected";

  useEffect(() => {
    const panel = panelRef.current;
    if (!file || !panel) return undefined;

    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const targets = [panel.querySelector(".upload-preview"), panel.querySelector(".upload-icon")].filter(
      (target): target is Element => Boolean(target)
    );

    if (reduceMotion) {
      gsap.set(targets, { autoAlpha: 1, scale: 1, clearProps: "all" });
      return undefined;
    }

    const tween = gsap.fromTo(
      targets,
      { scale: 0.97, autoAlpha: 0.72 },
      {
        scale: 1,
        autoAlpha: 1,
        duration: 0.26,
        ease: "power2.out",
        overwrite: "auto",
        clearProps: "transform,visibility,opacity"
      }
    );

    return () => {
      tween.kill();
    };
  }, [file]);

  return (
    <label className={`upload-panel upload-panel-${variant}`} ref={panelRef}>
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
