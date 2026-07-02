import React from "react";
import ReactDOM from "react-dom/client";
import { Loader2, Play, Shuffle, X } from "lucide-react";
import { cancelTryOnJob, getTryOnJob, prepareTryOnModel, submitTryOn, TryOnApiError } from "./lib/api";
import { HistoryPanel } from "./components/HistoryPanel";
import { ResultViewer } from "./components/ResultViewer";
import { UploadGarment } from "./components/UploadGarment";
import { UploadPerson } from "./components/UploadPerson";
import { useWorkbenchMotion } from "./hooks/useWorkbenchMotion";
import { garmentSlotsForCategory } from "./lib/category";
import { useTryOnStore } from "./store/tryonStore";
import "./styles.css";

const resolutionPresets = [
  { label: "Fast 512x768", width: 512, height: 768 },
  { label: "Balanced 640x896", width: 640, height: 896 },
  { label: "Quality 768x1024", width: 768, height: 1024 },
  { label: "Square 768x768", width: 768, height: 768 }
];
const JOB_POLL_INTERVAL_MS = 500;
const JOB_TIMER_INTERVAL_MS = 250;

function formatElapsedSeconds(value?: number | null) {
  if (value == null) return null;
  return `${Math.max(0, Math.floor(value))}s`;
}

function generateButtonLabel(state: ReturnType<typeof useTryOnStore.getState>, elapsedSeconds?: number | null) {
  if (!state.loading) return "Generate";
  const runningStage = state.result?.stages?.find((stage) => stage.status === "running")?.key;
  const stage = state.result?.current_stage ?? runningStage ?? state.result?.status;
  const labels: Record<string, string> = {
    queued: "Queued",
    running: "Preprocess",
    loading_model: "Loading model",
    generating: "Generating",
    refining: "Refining",
    completed: "Finalizing",
    cancel_requested: "Cancelling"
  };
  const elapsed = formatElapsedSeconds(elapsedSeconds);
  return elapsed ? `${labels[stage ?? ""] ?? "Working"} · ${elapsed}` : `${labels[stage ?? ""] ?? "Working"}...`;
}

function engineModeLabel(value: ReturnType<typeof useTryOnStore.getState>["engineMode"]) {
  const labels: Record<string, string> = {
    "": "IDM-VTON default",
    idm_vton: "IDM-VTON",
    idm_mask_expanded: "IDM expanded mask",
    idm_vton_flux: "IDM + FLUX",
    idm_mask_expanded_flux: "Expanded mask + FLUX",
    klein_lora: "Klein LoRA",
    klein_bnb_4bit: "Klein LoRA bnb 4-bit",
    idm_klein_hybrid: "IDM + Klein hybrid",
    idm_klein_hybrid_pro: "IDM compile + Klein 4-bit pro",
    catvton: "CatVTON"
  };
  return labels[value] ?? value;
}

function App() {
  const state = useTryOnStore();
  const workbenchRef = useWorkbenchMotion();
  const [jobStartedAtMs, setJobStartedAtMs] = React.useState<number | null>(null);
  const [modelStartedAtMs, setModelStartedAtMs] = React.useState<number | null>(null);
  const [clockNowMs, setClockNowMs] = React.useState(() => Date.now());
  const prepareSeqRef = React.useRef(0);
  const setField = state.setField;
  const resolutionValue = `${state.outputWidth}x${state.outputHeight}`;
  const isPresetResolution = resolutionPresets.some((item) => `${item.width}x${item.height}` === resolutionValue);
  const liveElapsedSeconds =
    state.loading && jobStartedAtMs != null ? (clockNowMs - jobStartedAtMs) / 1000 : null;
  const modelElapsedSeconds =
    state.modelPreparing && modelStartedAtMs != null ? (clockNowMs - modelStartedAtMs) / 1000 : null;
  const selectedEngineToken = state.engineMode || null;
  const statusEngineToken = state.modelStatus?.engine_mode ?? null;
  const isModelReady =
    !state.modelPreparing && state.modelStatus?.status === "ready" && statusEngineToken === selectedEngineToken;

  React.useEffect(() => {
    if (!state.loading && !state.modelPreparing) return undefined;
    const timer = window.setInterval(() => setClockNowMs(Date.now()), JOB_TIMER_INTERVAL_MS);
    return () => window.clearInterval(timer);
  }, [state.loading, state.modelPreparing]);

  React.useEffect(() => {
    void prepareSelectedModel(state.engineMode);
  }, [state.engineMode]);

  function setResolution(value: string) {
    const preset = resolutionPresets.find((item) => `${item.width}x${item.height}` === value);
    if (!preset) return;
    setField("outputWidth", preset.width);
    setField("outputHeight", preset.height);
  }

  function randomizeSeed() {
    setField("seed", Math.floor(Math.random() * 2_147_483_647));
    setField("seedMode", "fixed");
  }

  function setEngineMode(value: typeof state.engineMode) {
    setField("engineMode", value);
    if (!["klein_lora", "klein_bnb_4bit", "idm_klein_hybrid", "idm_klein_hybrid_pro"].includes(value)) return;
    if (value === "idm_klein_hybrid" || value === "idm_klein_hybrid_pro") {
      setField("useRefiner", false);
      setField("repairMode", false);
    }
    if (state.steps >= 28) setField("steps", 4);
    if (state.outputWidth === 768 && state.outputHeight === 1024) {
      setField("outputWidth", 512);
      setField("outputHeight", 768);
    }
  }

  async function prepareSelectedModel(engineMode: typeof state.engineMode) {
    const requestSeq = prepareSeqRef.current + 1;
    prepareSeqRef.current = requestSeq;
    const startedAt = Date.now();
    setModelStartedAtMs(startedAt);
    setClockNowMs(startedAt);
    setField("modelPreparing", true);
    setField("modelError", undefined);
    setField("modelStatus", {
      status: "loading",
      engine: engineMode || "idm_vton",
      engine_mode: engineMode || null
    });
    try {
      const response = await prepareTryOnModel(engineMode);
      if (prepareSeqRef.current !== requestSeq) return;
      setField("modelStatus", response);
      setField("modelError", response.status === "ready" ? undefined : response.message ?? "Model did not become ready.");
    } catch (error) {
      if (prepareSeqRef.current !== requestSeq) return;
      setField("modelStatus", {
        status: "failed",
        engine: engineMode || "idm_vton",
        engine_mode: engineMode || null
      });
      setField("modelError", displayError(error));
    } finally {
      if (prepareSeqRef.current === requestSeq) setField("modelPreparing", false);
    }
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
    const startedAt = Date.now();
    setJobStartedAtMs(startedAt);
    setClockNowMs(startedAt);
    setField("loading", true);
    try {
      if (!isModelReady) throw new Error("Model is still loading.");
      if (!state.personImage) throw new Error("Person image is required.");
      const form = new FormData();
      const garmentSlots = new Set(garmentSlotsForCategory(state.category));
      const hasVisibleGarment =
        (garmentSlots.has("top") && Boolean(state.topImage)) ||
        (garmentSlots.has("bottom") && Boolean(state.bottomImage)) ||
        (garmentSlots.has("dress") && Boolean(state.dressImage));
      if (!hasVisibleGarment) throw new Error("Garment image is required for the selected category.");

      form.append("person_image", state.personImage);
      if (garmentSlots.has("top") && state.topImage) form.append("garment_top", state.topImage);
      if (garmentSlots.has("bottom") && state.bottomImage) form.append("garment_bottom", state.bottomImage);
      if (garmentSlots.has("dress") && state.dressImage) form.append("garment_dress", state.dressImage);
      form.append("category", state.category);
      form.append("prompt", state.prompt);
      form.append("use_refiner", String(state.useRefiner));
      form.append("repair_mode", String(state.repairMode));
      form.append("run_mode", state.runMode);
      form.append("output_width", String(state.outputWidth));
      form.append("output_height", String(state.outputHeight));
      form.append("steps", String(state.steps));
      form.append("deterministic", String(state.deterministic));
      form.append("save_intermediates", String(state.showDebug));
      if (state.seedMode === "fixed") form.append("seed", String(state.seed));
      form.append("auto_prompt", String(state.autoPrompt));
      form.append("prompt_variant", state.promptVariant);
      if (state.testcaseId.trim()) form.append("testcase_id", state.testcaseId.trim());
      if (state.engineMode) form.append("engine_mode", state.engineMode);
      let result = await submitTryOn(form);
      setField("result", result);
      setField("jobId", result.job_id);
      while (["queued", "running", "cancel_requested"].includes(result.status)) {
        await new Promise((resolve) => window.setTimeout(resolve, JOB_POLL_INTERVAL_MS));
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
  const modelRuntime =
    state.modelStatus?.runtime_seconds == null ? null : formatElapsedSeconds(state.modelStatus.runtime_seconds);
  const modelLabel = engineModeLabel(state.engineMode);
  const modelStatusText = state.modelPreparing
    ? `Loading model: ${modelLabel}${modelElapsedSeconds == null ? "" : ` · ${formatElapsedSeconds(modelElapsedSeconds)}`}`
    : state.modelError
      ? `Model failed: ${modelLabel}`
      : isModelReady
        ? `Model ready: ${modelLabel}${modelRuntime ? ` · ${modelRuntime}` : ""}`
        : `Model not loaded: ${modelLabel}`;
  const generateLabel = state.modelPreparing
    ? `Loading model${modelElapsedSeconds == null ? "" : ` · ${formatElapsedSeconds(modelElapsedSeconds)}`}`
    : isModelReady
      ? generateButtonLabel(state, liveElapsedSeconds)
      : "Load model first";
  const toolbarStatus = state.loading ? generateButtonLabel(state, liveElapsedSeconds) : modelStatusText;
  const disableGenerate = state.loading || state.modelPreparing || !isModelReady;

  return (
    <main className="app-shell">
      <section className="workbench" ref={workbenchRef}>
        <div className="toolbar">
          <div className="title-block">
            <h1>Virtual Try-On</h1>
            <div className={`toolbar-status${state.loading ? " is-working" : ""}`}>
              {state.loading ? <Loader2 className="spin" size={14} /> : null}
              <span>{toolbarStatus}</span>
            </div>
            {state.modelStatus?.message && !state.modelPreparing && !state.modelError ? (
              <div className="toolbar-note">{state.modelStatus.message}</div>
            ) : null}
            {state.error && <div className="error-box toolbar-error">{state.error}</div>}
            {state.modelError && <div className="error-box toolbar-error">{state.modelError}</div>}
          </div>
          <div className="toolbar-actions">
            <button className="primary-button" type="button" onClick={generate} disabled={disableGenerate}>
              {state.loading || state.modelPreparing ? <Loader2 className="spin" size={18} /> : <Play size={18} />}
              {generateLabel}
            </button>
            {canCancel && (
              <button className="secondary-button" type="button" onClick={cancelJob}>
                <X size={18} />
                Cancel
              </button>
            )}
          </div>
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
            <select value={state.engineMode} onChange={(e) => setEngineMode(e.target.value as typeof state.engineMode)}>
              <option value="">IDM-VTON default</option>
              <option value="idm_vton">IDM-VTON</option>
              <option value="idm_mask_expanded">IDM-VTON expanded mask</option>
              <option value="idm_vton_flux">IDM-VTON + FLUX</option>
              <option value="idm_mask_expanded_flux">Expanded mask + FLUX</option>
              <option value="klein_lora">Klein LoRA experimental</option>
              <option value="klein_bnb_4bit">Klein LoRA bnb 4-bit</option>
              <option value="idm_klein_hybrid">IDM + Klein hybrid</option>
              <option value="idm_klein_hybrid_pro">IDM compile + Klein 4-bit pro</option>
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
          <label>
            <span>Seed</span>
            <select value={state.seedMode} onChange={(e) => setField("seedMode", e.target.value as typeof state.seedMode)}>
              <option value="random">Random each job</option>
              <option value="fixed">Fixed seed</option>
            </select>
          </label>
          <label className="seed-control">
            <span>Value</span>
            <div>
              <input
                type="number"
                min={0}
                max={2147483646}
                step={1}
                value={state.seed}
                disabled={state.seedMode === "random"}
                onChange={(e) => setField("seed", Number(e.target.value))}
              />
              <button className="icon-button neutral" type="button" onClick={randomizeSeed} title="Randomize fixed seed">
                <Shuffle size={16} />
              </button>
            </div>
          </label>
          <label className="deterministic-toggle">
            <input
              type="checkbox"
              checked={state.deterministic}
              onChange={(e) => setField("deterministic", e.target.checked)}
            />
            Deterministic
          </label>
        </section>

      </section>

      <aside className="right-rail">
        <ResultViewer />
        <HistoryPanel />
      </aside>
    </main>
  );
}

ReactDOM.createRoot(document.getElementById("root")!).render(<App />);
