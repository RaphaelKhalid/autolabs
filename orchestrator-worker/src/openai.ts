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

const EMPTY_USAGE: Usage = { inputTokens: 0, cachedInputTokens: 0, outputTokens: 0 };

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
}): Promise<AgentResult<T>> {
  let responseId: string | undefined;
  let usage = EMPTY_USAGE;
  const inputBytes = new TextEncoder().encode(`${options.system}\n${options.user}`).byteLength;
  if (inputBytes > options.maxInputBytes) {
    return { agentId: options.agentId, ok: false, usage, usageRecords: [], error: `Prompt exceeded ${options.maxInputBytes} bytes.` };
  }

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
      }),
    });

    const payload = await response.json<ResponsePayload>();
    responseId = payload.id;
    usage = measuredUsage(payload);

    if (!response.ok) {
      return {
        agentId: options.agentId,
        ok: false,
        responseId,
        usage,
        usageRecords: usageRecords(responseId, usage),
        error: payload.error?.message ?? `OpenAI returned ${response.status}`,
      };
    }

    try {
      const value = JSON.parse(outputText(payload)) as T;
      return {
        agentId: options.agentId,
        ok: true,
        responseId,
        value,
        usage,
        usageRecords: usageRecords(responseId, usage),
      };
    } catch (error) {
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
