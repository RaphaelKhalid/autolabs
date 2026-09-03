import type { AgentId, AgentResult, MeetingReport, ResearchReport, Usage } from './types';

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
    for (const content of item.content ?? []) if (content.type === 'output_text' && content.text) return content.text;
  }
  throw new Error(payload.error?.message ?? 'OpenAI response contained no output_text.');
}

function usage(payload: ResponsePayload): Usage {
  return {
    inputTokens: payload.usage?.input_tokens ?? 0,
    cachedInputTokens: payload.usage?.input_tokens_details?.cached_tokens ?? 0,
    outputTokens: payload.usage?.output_tokens ?? 0,
  };
}

export function costUsd(value: Usage) {
  const uncached = Math.max(0, value.inputTokens - value.cachedInputTokens);
  return (uncached * 0.2 + value.cachedInputTokens * 0.02 + value.outputTokens * 1.2) / 1_000_000;
}

export async function callStructured<T extends ResearchReport | MeetingReport>(options: {
  apiKey: string;
  model: string;
  agentId: AgentId;
  system: string;
  user: string;
  schemaName: string;
  schema: object;
  maxOutputTokens: number;
  webSearch: boolean;
}): Promise<AgentResult<T>> {
  try {
    const response = await fetch('https://api.openai.com/v1/responses', {
      method: 'POST',
      headers: { authorization: `Bearer ${options.apiKey}`, 'content-type': 'application/json' },
      body: JSON.stringify({
        model: options.model,
        reasoning: { effort: 'high' },
        input: [
          { role: 'system', content: [{ type: 'input_text', text: options.system }] },
          { role: 'user', content: [{ type: 'input_text', text: options.user }] },
        ],
        text: { format: { type: 'json_schema', name: options.schemaName, strict: true, schema: options.schema } },
        tools: options.webSearch ? [{ type: 'web_search' }] : [],
        max_output_tokens: options.maxOutputTokens,
        truncation: 'auto',
        store: true,
      }),
    });
    const payload = await response.json<ResponsePayload>();
    const measured = usage(payload);
    if (!response.ok) throw new Error(payload.error?.message ?? `OpenAI returned ${response.status}`);
    const value = JSON.parse(outputText(payload)) as T;
    return { agentId: options.agentId, ok: true, responseId: payload.id, value, usage: measured };
  } catch (error) {
    return { agentId: options.agentId, ok: false, usage: EMPTY_USAGE, error: error instanceof Error ? error.message : 'Unknown model error' };
  }
}
