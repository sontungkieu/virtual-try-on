import { useEffect, useState } from "react";
import { Check, ChevronDown, ChevronRight, Clipboard, Download, FileArchive } from "lucide-react";
import { fetchJsonArtifact, resolveAssetUrl } from "../lib/api";
import { PipelineStage, TryOnResult, useTryOnStore } from "../store/tryonStore";

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

const defaultTimeline = ["queued", "running", "generating", "refining", "completed"];
const stageLabels: Record<string, string> = {
  queued: "Queued",
  running: "Running",
  generating: "Generating",
  refining: "Refining",
  completed: "Completed"
};

function fallbackStages(result: TryOnResult): PipelineStage[] {
  return defaultTimeline.map((key): PipelineStage => {
    let status: PipelineStage["status"] = "pending";
    if (result.status === "queued") {
      status = key === "queued" ? "running" : "pending";
    } else if (result.status === "running" || result.status === "cancel_requested") {
      status = key === "generating" ? "running" : defaultTimeline.indexOf(key) < 2 ? "completed" : "pending";
    } else if (result.status === "completed") {
      status = "completed";
    } else if (result.status === "cancelled") {
      status = key === "completed" ? "cancelled" : defaultTimeline.indexOf(key) < 2 ? "completed" : "pending";
    } else if (result.status === "failed") {
      status = key === "completed" ? "failed" : defaultTimeline.indexOf(key) < 3 ? "completed" : "pending";
    }
    return { key, label: stageLabels[key], status };
  });
}

function formatRuntime(value?: number | null) {
  if (value == null) return null;
  return `${value.toFixed(value < 10 ? 1 : 0)}s`;
}

function stageMeta(stage: PipelineStage) {
  if (stage.status === "running") return "Running";
  if (stage.status === "pending") return "Pending";
  if (stage.status === "skipped") return "Skipped";
  if (stage.status === "failed") return "Failed";
  if (stage.status === "cancelled") return "Cancelled";
  return formatRuntime(stage.runtime_seconds) ?? "Done";
}

export function ResultViewer() {
  const result = useTryOnStore((state) => state.result);
  const showDebug = useTryOnStore((state) => state.showDebug);
  const [qualityReport, setQualityReport] = useState<QualityReport>();
  const [copied, setCopied] = useState(false);
  const [artifactsExpanded, setArtifactsExpanded] = useState(false);
  const [debugArtifactsExpanded, setDebugArtifactsExpanded] = useState(false);

  useEffect(() => {
    setQualityReport(undefined);
    if (!result?.debug?.quality_report_url) return;
    fetchJsonArtifact<QualityReport>(result.debug.quality_report_url)
      .then(setQualityReport)
      .catch(() => setQualityReport(undefined));
  }, [result?.debug?.quality_report_url]);

  useEffect(() => {
    setArtifactsExpanded(false);
    setDebugArtifactsExpanded(false);
  }, [result?.job_id]);

  if (!result) {
    return <section className="result-surface empty-state">No result yet</section>;
  }

  const resultUrl = resolveAssetUrl(result.result_url);
  const timeline = result.stages?.length ? result.stages : fallbackStages(result);
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
  const artifacts = result.artifact_manifest?.files ?? [];
  const debugLinks: [string, string | null | undefined][] = [
    ["quality_report.json", result.debug?.quality_report_url],
    ["mask_metadata.json", result.debug?.mask_metadata_url],
    ["prompt_core.txt", result.debug?.prompt_core_url],
    ["prompt_refine.txt", result.debug?.prompt_refine_url],
    ["prompt_metadata.json", result.debug?.prompt_metadata_url]
  ];
  const debugImageCount = debugItems.filter(([, url]) => Boolean(url)).length + (result.debug?.mask_urls?.length ?? 0);
  const debugLinkCount = debugLinks.filter(([, url]) => Boolean(url)).length;
  const hasDebugArtifacts = debugImageCount > 0 || debugLinkCount > 0 || Boolean(qualityReport);

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
          <span>{result.seed == null ? "seed n/a" : `seed ${result.seed}${result.deterministic ? " deterministic" : ""}`}</span>
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
        {timeline.map((stage) => (
          <li className={`stage-${stage.status}`} key={stage.key}>
            <span aria-hidden="true" />
            <strong>{stage.label || stage.key}</strong>
            <small>{stageMeta(stage)}</small>
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

      {artifacts.length ? (
        <section className="artifact-manifest">
          <button
            className="artifact-toggle"
            type="button"
            onClick={() => setArtifactsExpanded((value) => !value)}
            aria-expanded={artifactsExpanded}
          >
            <span>
              {artifactsExpanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
              Artifacts
            </span>
            <small>{artifacts.length} files</small>
          </button>
          {artifactsExpanded ? (
            <div className="artifact-list">
              {artifacts.map((artifact) => (
                <a href={resolveAssetUrl(artifact.url)} target="_blank" rel="noreferrer" key={artifact.name}>
                  <span>{artifact.name}</span>
                  <small>{Math.max(1, Math.round(artifact.size_bytes / 1024))} KB</small>
                </a>
              ))}
            </div>
          ) : null}
        </section>
      ) : null}

      {showDebug && hasDebugArtifacts ? (
        <section className="debug-artifacts">
          <button
            className="artifact-toggle"
            type="button"
            onClick={() => setDebugArtifactsExpanded((value) => !value)}
            aria-expanded={debugArtifactsExpanded}
          >
            <span>
              {debugArtifactsExpanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
              Debug artifacts
            </span>
            <small>{debugImageCount} images · {debugLinkCount} files</small>
          </button>
          {debugArtifactsExpanded ? (
            <div className="debug-artifact-body">
              {debugImageCount > 0 ? (
                <div className="debug-grid">
                  {debugItems.map(([label, url]) => {
                    const resolved = resolveAssetUrl(url);
                    return resolved ? (
                      <figure key={label}>
                        <img src={resolved} alt={label} />
                        <figcaption>{label}</figcaption>
                      </figure>
                    ) : null;
                  })}
                  {(result.debug?.mask_urls ?? []).map((url, index) => {
                    const resolved = resolveAssetUrl(url);
                    return resolved ? (
                      <figure key={`${url}-${index}`}>
                        <img src={resolved} alt={`Mask ${index + 1}`} />
                        <figcaption>Mask {index + 1}</figcaption>
                      </figure>
                    ) : null;
                  })}
                </div>
              ) : null}
              {debugLinkCount > 0 ? (
                <div className="debug-file-links">
                  {debugLinks.map(([label, url]) =>
                    url ? (
                      <a className="artifact-link" href={resolveAssetUrl(url)} target="_blank" rel="noreferrer" key={label}>
                        {label}
                      </a>
                    ) : null
                  )}
                </div>
              ) : null}
              {qualityReport ? <pre className="quality-json">{JSON.stringify(qualityReport, null, 2)}</pre> : null}
            </div>
          ) : null}
        </section>
      ) : null}

      {result.quality?.notes?.length ? (
        <div className="notes">
          {result.quality.notes.map((note) => <span key={note}>{note}</span>)}
        </div>
      ) : null}
    </section>
  );
}
