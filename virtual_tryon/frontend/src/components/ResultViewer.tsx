import { useEffect, useState } from "react";
import { Check, Clipboard, Download, FileArchive } from "lucide-react";
import { fetchJsonArtifact, resolveAssetUrl } from "../lib/api";
import { useTryOnStore } from "../store/tryonStore";

type QualityReport = {
  final_choice?: string;
  final_choice_reason?: string;
  engine_status?: Record<string, string>;
  outside_mask_delta?: number;
  garment_region_delta?: number;
  metrics?: {
    outside_mask_delta?: number;
    garment_region_delta?: number;
  };
};

export function ResultViewer() {
  const result = useTryOnStore((state) => state.result);
  const showDebug = useTryOnStore((state) => state.showDebug);
  const [qualityReport, setQualityReport] = useState<QualityReport>();
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    setQualityReport(undefined);
    if (!result?.debug?.quality_report_url) return;
    fetchJsonArtifact<QualityReport>(result.debug.quality_report_url)
      .then(setQualityReport)
      .catch(() => setQualityReport(undefined));
  }, [result?.debug?.quality_report_url]);

  if (!result) {
    return <section className="result-surface empty-state">No result yet</section>;
  }

  const resultUrl = resolveAssetUrl(result.result_url);
  const timeline = ["queued", "running", "generating", "refining", "completed"];
  const activeIndex =
    result.status === "queued"
      ? 0
      : result.status === "running" || result.status === "cancel_requested"
        ? 2
        : result.status === "completed"
          ? 4
          : -1;
  const debugItems: [string, string | null | undefined][] = [
    ["Mask", result.debug?.mask_url],
    ["Refine mask", result.debug?.refine_mask_url],
    ["Agnostic", result.debug?.agnostic_url],
    ["Core", result.debug?.core_output_url],
    ["Refined", result.debug?.refined_output_url]
  ];
  const engineStatus = qualityReport?.engine_status ?? result.engine_status ?? {};
  const outsideMaskDelta = qualityReport?.outside_mask_delta ?? qualityReport?.metrics?.outside_mask_delta;
  const garmentRegionDelta = qualityReport?.garment_region_delta ?? qualityReport?.metrics?.garment_region_delta;
  const jobId = result.job_id;

  async function copyJobId() {
    await navigator.clipboard.writeText(jobId);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1500);
  }

  return (
    <section className="result-surface">
      <div className="result-header">
        <div>
          <strong>{result.status}</strong>
          <span>{result.job_id}</span>
        </div>
        <div className="result-actions">
          <button className="icon-button neutral" type="button" onClick={copyJobId} title="Copy job ID">
            {copied ? <Check size={18} /> : <Clipboard size={18} />}
          </button>
          <button className="icon-button neutral" type="button" disabled title="Artifact ZIP export is not implemented yet">
            <FileArchive size={18} />
          </button>
          {resultUrl && (
            <a className="icon-button" href={resultUrl} download title="Download result">
              <Download size={18} />
            </a>
          )}
        </div>
      </div>

      <ol className="progress-timeline" aria-label="Job progress">
        {timeline.map((step, index) => (
          <li className={activeIndex >= index ? "done" : ""} key={step}>
            <span />
            {step}
          </li>
        ))}
      </ol>

      {result.error && <div className="error-box">{result.error}</div>}
      {resultUrl && <img className="result-image" src={resultUrl} alt="Try-on result" data-testid="tryon-result" />}

      {qualityReport && (
        <section className="quality-summary" aria-label="Quality report">
          <div><span>Final choice</span><strong>{qualityReport.final_choice ?? "unknown"}</strong></div>
          <div><span>Core engine</span><strong>{engineStatus.idm_vton ?? engineStatus.catvton ?? "unknown"}</strong></div>
          <div><span>Outside mask delta</span><strong>{outsideMaskDelta?.toFixed(4) ?? "n/a"}</strong></div>
          <div><span>Garment region delta</span><strong>{garmentRegionDelta?.toFixed(4) ?? "n/a"}</strong></div>
          {qualityReport.final_choice_reason && <p>{qualityReport.final_choice_reason}</p>}
        </section>
      )}

      {result.artifact_manifest?.files?.length ? (
        <section className="artifact-manifest">
          <h2>Artifacts</h2>
          <div>
            {result.artifact_manifest.files.map((artifact) => (
              <a href={resolveAssetUrl(artifact.url)} target="_blank" rel="noreferrer" key={artifact.name}>
                <span>{artifact.name}</span>
                <small>{Math.max(1, Math.round(artifact.size_bytes / 1024))} KB</small>
              </a>
            ))}
          </div>
        </section>
      ) : null}

      {showDebug && (
        <>
          <div className="debug-grid">
            {debugItems.map(([label, url]) => {
              const resolved = resolveAssetUrl(url);
              return (
                <figure key={label}>
                  {resolved ? <img src={resolved} alt={label} /> : <div className="empty-preview" />}
                  <figcaption>{label}</figcaption>
                </figure>
              );
            })}
            {(result.debug?.mask_urls ?? []).map((url, index) => {
              const resolved = resolveAssetUrl(url);
              return (
                <figure key={`${url}-${index}`}>
                  {resolved ? <img src={resolved} alt={`Mask ${index + 1}`} /> : <div className="empty-preview" />}
                  <figcaption>Mask {index + 1}</figcaption>
                </figure>
              );
            })}
          </div>
          {result.debug?.quality_report_url && (
            <a className="artifact-link" href={resolveAssetUrl(result.debug.quality_report_url)} target="_blank" rel="noreferrer">
              quality_report.json
            </a>
          )}
          {[
            ["mask_metadata.json", result.debug?.mask_metadata_url],
            ["prompt_core.txt", result.debug?.prompt_core_url],
            ["prompt_refine.txt", result.debug?.prompt_refine_url],
            ["prompt_metadata.json", result.debug?.prompt_metadata_url]
          ].map(([label, url]) =>
            url ? (
              <a className="artifact-link" href={resolveAssetUrl(url)} target="_blank" rel="noreferrer" key={label}>
                {label}
              </a>
            ) : null
          )}
          {qualityReport ? <pre className="quality-json">{JSON.stringify(qualityReport, null, 2)}</pre> : null}
        </>
      )}

      {result.quality?.notes?.length ? (
        <div className="notes">
          {result.quality.notes.map((note) => <span key={note}>{note}</span>)}
        </div>
      ) : null}
    </section>
  );
}
