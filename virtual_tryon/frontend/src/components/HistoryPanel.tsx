import { RefreshCw } from "lucide-react";
import { useEffect, useState } from "react";
import { getTryOnHistory, getTryOnJob, resolveAssetUrl, type HistoryGenderFilter } from "../lib/api";
import { TryOnHistoryItem, useTryOnStore } from "../store/tryonStore";

const HISTORY_LIMIT = 100;

function formatTime(value?: string | null) {
  if (!value) return "pending";
  return new Intl.DateTimeFormat(undefined, {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit"
  }).format(new Date(value));
}

function formatRuntime(value?: number | null) {
  if (value == null) return "n/a";
  return `${value.toFixed(value < 10 ? 1 : 0)}s`;
}

function formatStageTimings(item: TryOnHistoryItem) {
  if (!item.stages?.length) return null;
  const labels: Record<string, string> = {
    queued: "Queue",
    running: "Prep",
    loading_model: "Load",
    generating: "Gen",
    refining: "Refine",
    completed: "Done"
  };
  return item.stages
    .filter((stage) => stage.status !== "pending")
    .map((stage) => {
      const value =
        stage.status === "skipped" || stage.status === "failed" || stage.status === "cancelled"
          ? stage.status
          : formatRuntime(stage.runtime_seconds);
      return `${labels[stage.key] ?? stage.label}: ${value}`;
    })
    .join(" · ");
}

function formatSeed(item: TryOnHistoryItem) {
  if (item.config.seed == null) return "seed n/a";
  return `seed ${item.config.seed}${item.config.deterministic ? " deterministic" : ""}`;
}

function inputImages(item: TryOnHistoryItem) {
  return [
    ["Person", item.inputs.person_url],
    ["Top", item.inputs.garment_top_url],
    ["Bottom", item.inputs.garment_bottom_url],
    ["Dress", item.inputs.garment_dress_url]
  ].filter((entry): entry is [string, string] => Boolean(entry[1]));
}

function inferGender(item: TryOnHistoryItem): Exclude<HistoryGenderFilter, "all"> | "unknown" {
  const category = item.config.category?.toLowerCase() ?? "";
  if (category.startsWith("women_") || category === "women" || category === "women_bra" || category.includes("bra")) {
    return "woman";
  }
  if (category.startsWith("men_") || category === "men") return "man";

  const searchText = [
    item.config.prompt,
    item.inputs.person_url,
    item.inputs.garment_url,
    item.inputs.garment_top_url,
    item.inputs.garment_bottom_url,
    item.inputs.garment_dress_url
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();

  if (/(^|[^a-z])(male|men|man)([^a-z]|$)/.test(searchText)) return "man";
  if (/(^|[^a-z])(female|women|woman|bra)([^a-z]|$)/.test(searchText)) return "woman";
  return "unknown";
}

function matchesGenderFilter(item: TryOnHistoryItem, filter: HistoryGenderFilter) {
  if (filter === "all") return true;
  return inferGender(item) === filter;
}

export function HistoryPanel() {
  const resultJobId = useTryOnStore((state) => state.result?.job_id);
  const resultStatus = useTryOnStore((state) => state.result?.status);
  const setField = useTryOnStore((state) => state.setField);
  const [items, setItems] = useState<TryOnHistoryItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [genderFilter, setGenderFilter] = useState<HistoryGenderFilter>("all");
  const [successOnly, setSuccessOnly] = useState(false);

  async function refresh() {
    setLoading(true);
    try {
      const response = await getTryOnHistory(HISTORY_LIMIT, { gender: genderFilter, successOnly });
      setItems(response.items);
    } finally {
      setLoading(false);
    }
  }

  async function loadJob(jobId: string) {
    const job = await getTryOnJob(jobId);
    setField("result", job);
    setField("jobId", job.job_id);
  }

  useEffect(() => {
    if (
      !resultJobId ||
      resultStatus === "queued" ||
      resultStatus === "completed" ||
      resultStatus === "failed" ||
      resultStatus === "cancelled"
    ) {
      void refresh();
    }
  }, [resultJobId, resultStatus, genderFilter, successOnly]);

  const filteredItems = items.filter((item) => {
    if (!matchesGenderFilter(item, genderFilter)) return false;
    if (successOnly && item.status !== "completed") return false;
    return true;
  });
  const hasFilters = genderFilter !== "all" || successOnly;

  return (
    <section className="history-panel" aria-label="History">
      <div className="history-header">
        <h2>History</h2>
        <button className="icon-button neutral" type="button" onClick={refresh} title="Refresh history">
          <RefreshCw className={loading ? "spin" : ""} size={17} />
        </button>
      </div>
      <div className="history-controls">
        <div className="history-segmented" role="group" aria-label="Gender filter">
          {(["all", "man", "woman"] as HistoryGenderFilter[]).map((filter) => (
            <button
              className={genderFilter === filter ? "active" : ""}
              type="button"
              onClick={() => setGenderFilter(filter)}
              key={filter}
            >
              {filter === "all" ? "All" : filter === "man" ? "Man" : "Woman"}
            </button>
          ))}
        </div>
        <label className="history-success-toggle">
          <input
            type="checkbox"
            checked={successOnly}
            onChange={(event) => setSuccessOnly(event.target.checked)}
          />
          Success only
        </label>
      </div>
      <div className="history-list">
        {filteredItems.map((item) => {
          const resultUrl = resolveAssetUrl(item.result_url);
          const stageTimings = formatStageTimings(item);
          return (
            <button className="history-item" type="button" onClick={() => loadJob(item.job_id)} key={item.job_id}>
              <div className="history-images">
                {inputImages(item).map(([label, url]) => (
                  <figure key={label}>
                    <img src={resolveAssetUrl(url)} alt={label} />
                    <figcaption>{label}</figcaption>
                  </figure>
                ))}
                <figure>
                  {resultUrl ? <img src={resultUrl} alt="Output" /> : <div className="empty-preview" />}
                  <figcaption>Output</figcaption>
                </figure>
              </div>
              <div className="history-meta">
                <strong>{item.config.category ?? "unknown"} · {item.status}</strong>
                <span>{item.config.output_width ?? "?"}x{item.config.output_height ?? "?"} · {item.config.steps ?? "?"} steps · {formatSeed(item)} · {item.config.engine ?? "engine"}</span>
                <span>{formatTime(item.finished_at ?? item.created_at)} · {formatRuntime(item.runtime_seconds)}</span>
                {stageTimings ? <span className="history-stage-times">{stageTimings}</span> : null}
                <small>{item.job_id}</small>
              </div>
            </button>
          );
        })}
        {!filteredItems.length && (
          <div className="history-empty">{items.length && hasFilters ? "No matching jobs" : "No jobs"}</div>
        )}
      </div>
    </section>
  );
}
