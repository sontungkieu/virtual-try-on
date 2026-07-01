import React from "react";
import ReactDOM from "react-dom/client";
import { Loader2, Play, X } from "lucide-react";
import { cancelTryOnJob, getTryOnJob, submitTryOn, TryOnApiError } from "./lib/api";
import { HistoryPanel } from "./components/HistoryPanel";
import { ResultViewer } from "./components/ResultViewer";
import { TryOnPreview } from "./components/TryOnPreview";
import { UploadGarment } from "./components/UploadGarment";
import { UploadPerson } from "./components/UploadPerson";
import { useTryOnStore } from "./store/tryonStore";
import "./styles.css";

const resolutionPresets = [
  { label: "Fast 512x768", width: 512, height: 768 },
  { label: "Balanced 640x896", width: 640, height: 896 },
  { label: "Quality 768x1024", width: 768, height: 1024 },
  { label: "Square 768x768", width: 768, height: 768 }
];

function App() {
  const state = useTryOnStore();
  const setField = state.setField;
  const resolutionValue = `${state.outputWidth}x${state.outputHeight}`;
  const isPresetResolution = resolutionPresets.some((item) => `${item.width}x${item.height}` === resolutionValue);
  const topPreviewTitle = state.category === "women_bra" ? "Bra" : "Top";
  const bottomPreviewTitle =
    state.category === "men_underwear"
      ? "Men underwear"
      : state.category === "women_underwear"
        ? "Women underwear"
        : "Bottom";

  function setResolution(value: string) {
    const preset = resolutionPresets.find((item) => `${item.width}x${item.height}` === value);
    if (!preset) return;
    setField("outputWidth", preset.width);
    setField("outputHeight", preset.height);
  }

  function displayError(error: unknown) {
    if (error instanceof TryOnApiError) {
      const labels: Record<string, string> = {
        INVALID_IMAGE: "The selected file is not a valid supported image.",
        FILE_TOO_LARGE: "The selected image exceeds the upload limit.",
        ENGINE_UNAVAILABLE: "The try-on engine is currently unavailable.",
        QUEUE_FULL: "The GPU queue is full. Please retry shortly.",
        TIMEOUT: "The job exceeded the configured runtime limit.",
        JOB_NOT_FOUND: "This job is no longer available."
      };
      return labels[error.code] ?? error.message;
    }
    if (error instanceof TypeError) return "Backend is offline or unreachable.";
    return error instanceof Error ? error.message : String(error);
  }

  async function generate() {
    state.resetResult();
    setField("loading", true);
    try {
      if (!state.personImage) throw new Error("Person image is required.");
      const form = new FormData();
      form.append("person_image", state.personImage);
      if (state.topImage) form.append("garment_top", state.topImage);
      if (state.bottomImage) form.append("garment_bottom", state.bottomImage);
      if (state.dressImage) form.append("garment_dress", state.dressImage);
      form.append("category", state.category);
      form.append("prompt", state.prompt);
      form.append("use_refiner", String(state.useRefiner));
      form.append("repair_mode", String(state.repairMode));
      form.append("run_mode", state.runMode);
      form.append("output_width", String(state.outputWidth));
      form.append("output_height", String(state.outputHeight));
      form.append("steps", String(state.steps));
      form.append("auto_prompt", String(state.autoPrompt));
      form.append("prompt_variant", state.promptVariant);
      if (state.testcaseId.trim()) form.append("testcase_id", state.testcaseId.trim());
      if (state.engineMode) form.append("engine_mode", state.engineMode);
      let result = await submitTryOn(form);
      setField("result", result);
      setField("jobId", result.job_id);
      while (["queued", "running", "cancel_requested"].includes(result.status)) {
        await new Promise((resolve) => window.setTimeout(resolve, 2000));
        result = await getTryOnJob(result.job_id);
        setField("result", result);
        setField("jobId", result.job_id);
      }
      if (result.error) {
        setField("error", result.error_code === "TIMEOUT" ? "The job exceeded the configured runtime limit." : result.error);
      }
    } catch (error) {
      setField("error", displayError(error));
    } finally {
      setField("loading", false);
    }
  }

  async function cancelJob() {
    if (!state.jobId) return;
    try {
      const result = await cancelTryOnJob(state.jobId);
      setField("result", result);
      if (result.error) setField("error", result.error);
    } catch (error) {
      setField("error", displayError(error));
    }
  }

  const canCancel = Boolean(
    state.loading && state.jobId && (state.result?.status === "queued" || state.result?.status === "running")
  );

  return (
    <main className="app-shell">
      <section className="workbench">
        <div className="toolbar">
          <div>
            <h1>Virtual Try-On</h1>
            <span>{state.jobId ?? "Ready"}</span>
          </div>
          <button className="primary-button" type="button" onClick={generate} disabled={state.loading}>
            {state.loading ? <Loader2 className="spin" size={18} /> : <Play size={18} />}
            Generate
          </button>
          {canCancel && (
            <button className="secondary-button" type="button" onClick={cancelJob}>
              <X size={18} />
              Cancel
            </button>
          )}
        </div>

        <div className="input-grid">
          <UploadPerson />
          <UploadGarment />
        </div>

        <div className="control-row">
          <label className="prompt-box">
            <span>Prompt</span>
            <textarea value={state.prompt} onChange={(e) => setField("prompt", e.target.value)} />
          </label>
          <div className="switches">
            <label><input type="checkbox" checked={state.useRefiner} onChange={(e) => setField("useRefiner", e.target.checked)} /> FLUX refine</label>
            <label><input type="checkbox" checked={state.repairMode} onChange={(e) => setField("repairMode", e.target.checked)} /> Repair</label>
            <label><input type="checkbox" checked={state.showDebug} onChange={(e) => setField("showDebug", e.target.checked)} /> Debug</label>
            <label>
              <span>Mode</span>
              <select value={state.runMode} onChange={(e) => setField("runMode", e.target.value as "sync" | "async")}>
                <option value="sync">Sync</option>
                <option value="async">Async</option>
              </select>
            </label>
          </div>
        </div>

        <section className="advanced-prompts" aria-label="Advanced prompt controls">
          <label><input type="checkbox" checked={state.autoPrompt} onChange={(e) => setField("autoPrompt", e.target.checked)} /> Use auto prompt</label>
          <label>
            <span>Testcase</span>
            <input
              value={state.testcaseId}
              onChange={(e) => setField("testcaseId", e.target.value)}
              placeholder="tc10"
            />
          </label>
          <label>
            <span>Variant</span>
            <select value={state.promptVariant} onChange={(e) => setField("promptVariant", e.target.value as typeof state.promptVariant)}>
              <option value="default">Default</option>
              <option value="strong_remove_old_garment">Strong remove old garment</option>
              <option value="identity_strict">Identity strict</option>
            </select>
          </label>
          <label>
            <span>Engine</span>
            <select value={state.engineMode} onChange={(e) => setField("engineMode", e.target.value as typeof state.engineMode)}>
              <option value="">IDM-VTON default</option>
              <option value="idm_vton">IDM-VTON</option>
              <option value="idm_mask_expanded">IDM-VTON expanded mask</option>
              <option value="idm_vton_flux">IDM-VTON + FLUX</option>
              <option value="idm_mask_expanded_flux">Expanded mask + FLUX</option>
              <option value="klein_lora">Klein LoRA experimental</option>
              <option value="catvton">CatVTON baseline</option>
            </select>
          </label>
        </section>

        <section className="generation-settings" aria-label="Generation settings">
          <label>
            <span>Resolution</span>
            <select value={resolutionValue} onChange={(e) => setResolution(e.target.value)}>
              {!isPresetResolution && <option value={resolutionValue}>Custom {resolutionValue}</option>}
              {resolutionPresets.map((preset) => (
                <option value={`${preset.width}x${preset.height}`} key={`${preset.width}x${preset.height}`}>
                  {preset.label}
                </option>
              ))}
            </select>
          </label>
          <label>
            <span>Width</span>
            <input
              type="number"
              min={384}
              max={1536}
              step={8}
              value={state.outputWidth}
              onChange={(e) => setField("outputWidth", Number(e.target.value))}
            />
          </label>
          <label>
            <span>Height</span>
            <input
              type="number"
              min={384}
              max={1536}
              step={8}
              value={state.outputHeight}
              onChange={(e) => setField("outputHeight", Number(e.target.value))}
            />
          </label>
          <label>
            <span>Steps</span>
            <input
              type="number"
              min={4}
              max={50}
              step={1}
              value={state.steps}
              onChange={(e) => setField("steps", Number(e.target.value))}
            />
          </label>
        </section>

        {state.error && <div className="error-box">{state.error}</div>}

        <div className="preview-grid">
          <TryOnPreview title="Person" file={state.personImage} />
          <TryOnPreview title={topPreviewTitle} file={state.topImage} />
          <TryOnPreview title={bottomPreviewTitle} file={state.bottomImage} />
          <TryOnPreview title="Dress" file={state.dressImage} />
        </div>
      </section>

      <aside className="right-rail">
        <ResultViewer />
        <HistoryPanel />
      </aside>
    </main>
  );
}

ReactDOM.createRoot(document.getElementById("root")!).render(<App />);
