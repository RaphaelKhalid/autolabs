import type { AgentId } from './types';

export const EXA_BUDGET_USD = 40;
export const EXA_REQUEST_AUTHORIZATION_USD = 0.02;

export interface ExaSource {
  title: string;
  url: string;
  author?: string;
  publishedDate?: string;
  highlights: string[];
}

export interface ExaRetrieval {
  agentId: AgentId;
  query: string;
  requestId?: string;
  costUsd: number;
  sources: ExaSource[];
  error?: string;
}

interface ExaPayload {
  requestId?: string;
  results?: Array<{
    title?: string;
    url?: string;
    author?: string;
    publishedDate?: string;
    highlights?: string[];
  }>;
  costDollars?: { total?: number };
  error?: string;
}

function boundedText(value: unknown, maximum: number) {
  return typeof value === 'string' ? value.slice(0, maximum) : '';
}

export async function searchExa(options: {
  apiKey: string;
  agentId: AgentId;
  query: string;
}): Promise<ExaRetrieval> {
  try {
    const response = await fetch('https://api.exa.ai/search', {
      method: 'POST',
      signal: AbortSignal.timeout(12_000),
      headers: { 'content-type': 'application/json', 'x-api-key': options.apiKey },
      body: JSON.stringify({
        query: options.query,
        type: 'auto',
        numResults: 5,
        moderation: true,
        contents: { highlights: true },
      }),
    });
    const payload = await response.json<ExaPayload>();
    if (!response.ok) throw new Error(payload.error ?? `Exa returned ${response.status}`);
    const sources = (payload.results ?? []).slice(0, 5).map((result) => ({
      title: boundedText(result.title, 300) || 'Untitled source',
      url: boundedText(result.url, 2_000),
      author: boundedText(result.author, 300) || undefined,
      publishedDate: boundedText(result.publishedDate, 80) || undefined,
      highlights: (result.highlights ?? []).slice(0, 4).map((item) => boundedText(item, 1_200)),
    })).filter((source) => /^https?:\/\//.test(source.url));
    return {
      agentId: options.agentId,
      query: options.query,
      requestId: payload.requestId,
      costUsd: Math.max(0, Number(payload.costDollars?.total ?? 0)),
      sources,
    };
  } catch (error) {
    return {
      agentId: options.agentId,
      query: options.query,
      costUsd: 0,
      sources: [],
      error: error instanceof Error ? error.message : 'Exa retrieval failed',
    };
  }
}

export function retrievalContext(retrieval: ExaRetrieval) {
  return {
    query: retrieval.query,
    sources: retrieval.sources.map((source) => ({
      title: source.title,
      url: source.url,
      author: source.author,
      publishedDate: source.publishedDate,
      highlights: source.highlights,
    })),
    retrievalError: retrieval.error,
  };
}
