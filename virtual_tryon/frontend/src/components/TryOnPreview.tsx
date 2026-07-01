import { ImageIcon } from "lucide-react";

function objectUrl(file?: File) {
  return file ? URL.createObjectURL(file) : undefined;
}

export function TryOnPreview({ title, file }: { title: string; file?: File }) {
  const url = objectUrl(file);
  return (
    <figure className="preview-tile">
      {url ? <img src={url} alt={title} /> : <div className="empty-preview"><ImageIcon size={24} /></div>}
      <figcaption>{title}</figcaption>
    </figure>
  );
}
