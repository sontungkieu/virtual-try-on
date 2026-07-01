import { create } from "zustand";

export type Category =
  | "upper_body"
  | "lower_body"
  | "dress"
  | "full_outfit"
  | "men_underwear"
  | "women_underwear"
  | "women_bra";
export type EngineMode = "" | "idm_vton" | "idm_vton_flux" | "idm_mask_expanded" | "idm_mask_expanded_flux" | "klein_lora" | "catvton";
export type PromptVariant = "default" | "strong_remove_old_garment" | "identity_strict";
export type StageStatus = "pending" | "running" | "completed" | "skipped" | "failed" | "cancelled";

export type PipelineStage = {
  key: string;
  label: string;
  status: StageStatus;
  started_at?: string | null;
  finished_at?: string | null;
  runtime_seconds?: number | null;
};

export type ArtifactManifest = {
  job_id: string;
  files: {
    name: string;
    url: string;
    type: "image" | "json" | "csv" | "html" | "text";
    size_bytes: number;
  }[];
};

export type TryOnResult = {
  job_id: string;
  status: "queued" | "running" | "completed" | "failed" | "cancelled" | "cancel_requested";
  result_url?: string | null;
  error?: string | null;
  error_code?: string | null;
  seed?: number | null;
  current_stage?: string | null;
  stages?: PipelineStage[];
  engine_status?: Record<string, string>;
  artifact_manifest?: ArtifactManifest | null;
  debug?: {
    mask_url?: string | null;
    mask_urls?: string[];
    agnostic_url?: string | null;
    core_output_url?: string | null;
    refined_output_url?: string | null;
    quality_report_url?: string | null;
    refine_mask_url?: string | null;
    mask_metadata_url?: string | null;
    prompt_core_url?: string | null;
    prompt_refine_url?: string | null;
    prompt_metadata_url?: string | null;
  };
  quality?: {
    needs_refine: boolean;
    notes: string[];
    background_preservation_score?: number | null;
    garment_similarity_score?: number | null;
    artifact_score?: number | null;
  } | null;
};

export type TryOnHistoryItem = {
  job_id: string;
  status: TryOnResult["status"];
  created_at?: string | null;
  started_at?: string | null;
  finished_at?: string | null;
  runtime_seconds?: number | null;
  current_stage?: string | null;
  stages?: PipelineStage[];
  result_url?: string | null;
  inputs: {
    person_url?: string | null;
    garment_url?: string | null;
    garment_top_url?: string | null;
    garment_bottom_url?: string | null;
    garment_dress_url?: string | null;
  };
  config: {
    output_width?: number | null;
    output_height?: number | null;
    steps?: number | null;
    seed?: number | null;
    engine?: string | null;
    category?: string | null;
    prompt?: string | null;
    use_refiner?: boolean | null;
    repair_mode?: boolean | null;
  };
  engine_status?: Record<string, string>;
  quality?: TryOnResult["quality"];
  error?: string | null;
};

export type TryOnHistoryResponse = {
  items: TryOnHistoryItem[];
};

type TryOnState = {
  personImage?: File;
  topImage?: File;
  bottomImage?: File;
  dressImage?: File;
  category: Category;
  prompt: string;
  autoPrompt: boolean;
  testcaseId: string;
  promptVariant: PromptVariant;
  engineMode: EngineMode;
  useRefiner: boolean;
  repairMode: boolean;
  runMode: "sync" | "async";
  outputWidth: number;
  outputHeight: number;
  steps: number;
  showDebug: boolean;
  loading: boolean;
  jobId?: string;
  result?: TryOnResult;
  error?: string;
  setField: <K extends keyof TryOnState>(key: K, value: TryOnState[K]) => void;
  resetResult: () => void;
};

export const useTryOnStore = create<TryOnState>((set) => ({
  category: "upper_body",
  prompt: "",
  autoPrompt: false,
  testcaseId: "",
  promptVariant: "default",
  engineMode: "",
  useRefiner: false,
  repairMode: true,
  runMode: "sync",
  outputWidth: 768,
  outputHeight: 1024,
  steps: 30,
  showDebug: true,
  loading: false,
  setField: (key, value) => set({ [key]: value } as Partial<TryOnState>),
  resetResult: () => set({ result: undefined, error: undefined, jobId: undefined })
}));
