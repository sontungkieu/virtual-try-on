import type { TryOnHistoryResponse, TryOnResult } from "../store/tryonStore";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "";

export type HistoryGenderFilter = "all" | "man" | "woman";

export type HistoryQueryOptions = {
  gender?: HistoryGenderFilter;
  successOnly?: boolean;
};

type ApiErrorPayload = {
  error?: {
    code?: string;
    message?: string;
    details?: Record<string, unknown>;
  };
};

export class TryOnApiError extends Error {
  code: string;

  constructor(code: string, message: string) {
    super(message);
    this.name = "TryOnApiError";
    this.code = code;
  }
}

async function throwApiError(response: Response, fallback: string): Promise<never> {
  const payload = (await response.json().catch(() => ({}))) as ApiErrorPayload;
  const code = payload.error?.code ?? "BACKEND_OFFLINE";
  const message = payload.error?.message ?? `${fallback} (${response.status})`;
  throw new TryOnApiError(code, message);
}

export async function submitTryOn(form: FormData): Promise<TryOnResult> {
  const response = await fetch(`${API_BASE_URL}/tryon`, {
    method: "POST",
    body: form
  });
  if (!response.ok) {
    return throwApiError(response, "Try-on request failed");
  }
  return response.json();
}

export async function getTryOnJob(jobId: string): Promise<TryOnResult> {
  const response = await fetch(`${API_BASE_URL}/tryon/${jobId}`);
  if (!response.ok) {
    return throwApiError(response, "Status request failed");
  }
  return response.json();
}

export async function cancelTryOnJob(jobId: string): Promise<TryOnResult> {
  const response = await fetch(`${API_BASE_URL}/tryon/${jobId}`, { method: "DELETE" });
  if (!response.ok) {
    return throwApiError(response, "Cancel request failed");
  }
  return response.json();
}

export async function getTryOnHistory(limit = 20, options: HistoryQueryOptions = {}): Promise<TryOnHistoryResponse> {
  const params = new URLSearchParams({ limit: String(limit) });
  if (options.gender && options.gender !== "all") params.set("gender", options.gender);
  if (options.successOnly) params.set("success_only", "true");
  const response = await fetch(`${API_BASE_URL}/tryon/history?${params.toString()}`);
  if (!response.ok) {
    return throwApiError(response, "History request failed");
  }
  return response.json();
}

export async function fetchJsonArtifact<T>(url?: string | null): Promise<T | undefined> {
  const resolved = resolveAssetUrl(url);
  if (!resolved) return undefined;
  const response = await fetch(resolved);
  if (!response.ok) return undefined;
  return response.json();
}

export function resolveAssetUrl(url?: string | null): string | undefined {
  if (!url) return undefined;
  if (url.startsWith("http")) return url;
  return `${API_BASE_URL}${url}`;
}
