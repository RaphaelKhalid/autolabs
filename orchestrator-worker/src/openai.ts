import type { AgentId, AgentResult, MeetingReport, ResearchReport, Usage, UsageRecord } from './types';

interface ResponsePayload {
  id?: string;
  output?: { type?: string; content?: { type?: string; text?: string }[] }[];
  usage?: {
    input_tokens?: number;
    output_tokens?: number;
    input_tokens_details?: { cached_tokens?: number };
  };
  error?: { message?: string };
}

interface StreamingEvent {
  type?: string;
  delta?: string;
  text?: string;
  response?: ResponsePayload;
  error?: { message?: string };
}

export interface ModelProgress {
  status: 'connecting' | 'streaming' | 'complete' | 'error';
  outputCharacters: number;
}

const EMPTY_USAGE: Usage = { inputTokens: 0, cachedInputTokens: 0, outputTokens: 0 };
const PROGRESS_INTERVAL_MS = 12_000;
const PROGRESS_CHARACTER_STEP = 128;

function outputText(payload: ResponsePayload) {
  for (const item of payload.output ?? []) {
    for (const content of item.content ?? []) {
      if (content.type === 'output_text' && content.text) return content.text;
    }
  }
  throw new Error(payload.error?.message ?? 'OpenAI response contained no output_text.');
}

function measuredUsage(payload: ResponsePayload): Usage {
  return {
    inputTokens: payload.usage?.input_tokens ?? 0,
    cachedInputTokens: payload.usage?.input_tokens_details?.cached_tokens ?? 0,
    outputTokens: payload.usage?.output_tokens ?? 0,
  };
}

function usageRecords(responseId: string | undefined, usage: Usage): UsageRecord[] {
  return responseId ? [{ responseId, usage }] : [];
}

function boundary(buffer: string) {
  const lf = buffer.indexOf('\n\n');
  const crlf = buffer.indexOf('\r\n\r\n');
  if (lf < 0 && crlf < 0) return null;
  if (crlf >= 0 && (lf < 0 || crlf < lf)) return { index: crlf, length: 4 };
  return { index: lf, length: 2 };
}

function eventData(block: string) {
  const data = block
    .split(/\r?\n/)
    .filter((line) => line.startsWith('data:'))
    .map((line) => line.slice(5).trimStart())
    .join('\n');
  if (!data || data === '[DONE]') return null;
  return JSON.parse(data) as StreamingEvent;
}

async function readStream(
  response: Response,
  reportProgress: (progress: ModelProgress) => Promise<void>,
) {
  if (!response.body) throw new Error('OpenAI streaming response had no body.');
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let text = '';
  let payload: ResponsePayload = {};
  let lastProgressAt = Date.now();
  let lastProgressCharacters = 0;

  const acceptBlock = async (block: string) => {
    const event = eventData(block);
    if (!event) return;
    if (event.response) payload = event.response;
    if (event.type === 'response.output_text.delta' && typeof event.delta === 'string') text += event.delta;
    if (event.type === 'response.output_text.done' && !text && typeof event.text === 'string') text = event.text;
    if (event.type === 'response.failed') throw new Error(event.response?.error?.message ?? event.error?.message ?? 'OpenAI response failed.');
    const now = Date.now();
    if (text.length - lastProgressCharacters >= PROGRESS_CHARACTER_STEP && now - lastProgressAt >= PROGRESS_INTERVAL_MS) {
      lastProgressAt = now;
      lastProgressCharacters = text.length;
      await reportProgress({ status: 'streaming', outputCharacters: text.length });
    }
  };

  while (true) {
    const chunk = await reader.read();
    buffer += decoder.decode(chunk.value ?? new Uint8Array(), { stream: !chunk.done });
    let marker = boundary(buffer);
    while (marker) {
      const block = buffer.slice(0, marker.index);
      buffer = buffer.slice(marker.index + marker.length);
      await acceptBlock(block);
      marker = boundary(buffer);
    }
    if (chunk.done) break;
  }
  if (buffer.trim()) await acceptBlock(buffer);
  return { payload, text };
}

export function addUsage(left: Usage, right: Usage): Usage {
  return {
    inputTokens: left.inputTokens + right.inputTokens,
    cachedInputTokens: left.cachedInputTokens + right.cachedInputTokens,
    outputTokens: left.outputTokens + right.outputTokens,
  };
}

export function mergeAttempts<T extends ResearchReport | MeetingReport>(
  first: AgentResult<T>,
  second: AgentResult<T>,
): AgentResult<T> {
  return {
    ...second,
    usage: addUsage(first.usage, second.usage),
    usageRecords: [...(first.usageRecords ?? []), ...(second.usageRecords ?? [])],
    error: second.ok ? undefined : `${first.error ?? 'First attempt failed'}; retry: ${second.error ?? 'failed'}`,
  };
}

export function costUsd(value: Usage) {
  const uncached = Math.max(0, value.inputTokens - value.cachedInputTokens);
  return (uncached * 0.2 + value.cachedInputTokens * 0.02 + value.outputTokens * 1.2) / 1_000_000;
}

export async function callStructured<T extends ResearchReport | MeetingReport>(options: {
  apiKey: string;
  model: 'gpt-5.6-luna';
  agentId: AgentId;
  system: string;
  user: string;
  schemaName: string;
  schema: object;
  maxOutputTokens: number;
  maxInputBytes: number;
  webSearch: boolean;
  timeoutMs?: number;
  onProgress?: (progress: ModelProgress) => void | Promise<void>;
}): Promise<AgentResult<T>> {
  let responseId: string | undefined;
  let usage = EMPTY_USAGE;
  let outputCharacters = 0;
  const inputBytes = new TextEncoder().encode(`${options.system}\n${options.user}`).byteLength;
  const reportProgress = async (progress: ModelProgress) => {
    try { await options.onProgress?.(progress); } catch { /* Telemetry must never interrupt mathematics. */ }
  };
  if (inputBytes > options.maxInputBytes) {
    await reportProgress({ status: 'error', outputCharacters });
    return { agentId: options.agentId, ok: false, usage, usageRecords: [], error: `Prompt exceeded ${options.maxInputBytes} bytes.` };
  }

  await reportProgress({ status: 'connecting', outputCharacters });
  try {
    const response = await fetch('https://api.openai.com/v1/responses', {
      method: 'POST',
      signal: AbortSignal.timeout(options.timeoutMs ?? 135_000),
      headers: {
        authorization: `Bearer ${options.apiKey}`,
        'content-type': 'application/json',
      },
      body: JSON.stringify({
        model: options.model,
        reasoning: { effort: 'high' },
        input: [
          { role: 'system', content: [{ type: 'input_text', text: options.system }] },
          { role: 'user', content: [{ type: 'input_text', text: options.user }] },
        ],
        text: {
          verbosity: 'low',
          format: {
            type: 'json_schema',
            name: options.schemaName,
            strict: true,
            schema: options.schema,
          },
        },
        tools: options.webSearch ? [{ type: 'web_search' }] : [],
        max_output_tokens: options.maxOutputTokens,
        truncation: 'auto',
        store: true,
        stream: true,
      }),
    });

    if (!response.ok) {
      const raw = await response.text();
      let payload: ResponsePayload = {};
      try { payload = JSON.parse(raw) as ResponsePayload; } catch { /* status below is enough */ }
      responseId = payload.id;
      usage = measuredUsage(payload);
      await reportProgress({ status: 'error', outputCharacters });
      return {
        agentId: options.agentId,
        ok: false,
        responseId,
        usage,
        usageRecords: usageRecords(responseId, usage),
        error: payload.error?.message ?? `OpenAI returned ${response.status}`,
      };
    }

    const streamed = await readStream(response, reportProgress);
    responseId = streamed.payload.id;
    usage = measuredUsage(streamed.payload);
    const rawOutput = streamed.text || outputText(streamed.payload);
    outputCharacters = rawOutput.length;

    try {
      const value = JSON.parse(rawOutput) as T;
      await reportProgress({ status: 'complete', outputCharacters });
      return {
        agentId: options.agentId,
        ok: true,
        responseId,
        value,
        usage,
        usageRecords: usageRecords(responseId, usage),
      };
    } catch (error) {
      await reportProgress({ status: 'error', outputCharacters });
      return {
        agentId: options.agentId,
        ok: false,
        responseId,
        usage,
        usageRecords: usageRecords(responseId, usage),
        error: error instanceof Error ? error.message : 'Invalid structured response',
      };
    }
  } catch (error) {
    await reportProgress({ status: 'error', outputCharacters });
    return {
      agentId: options.agentId,
      ok: false,
      responseId,
      usage,
      usageRecords: usageRecords(responseId, usage),
      error: error instanceof Error ? error.message : 'Unknown model error',
    };
  }
}
