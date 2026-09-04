import { afterEach, describe, expect, it, vi } from 'vitest';
import { callStructured, type ModelProgress } from '../src/openai';
import type { ResearchReport } from '../src/types';

afterEach(() => vi.unstubAllGlobals());

describe('OpenAI response streaming', () => {
  it('reassembles split SSE output and reports only bounded progress metadata', async () => {
    const report: ResearchReport = {
      headline: 'Exact branch checked',
      thesis: 'A deterministic test completed.',
      claims: [], equations: [], citations: [], failedAvenues: [], candidates: [], proposedJobs: [], nextQuestions: [],
    };
    const json = JSON.stringify(report);
    const sse = [
      `data: ${JSON.stringify({ type: 'response.created', response: { id: 'resp_test' } })}\n\n`,
      `data: ${JSON.stringify({ type: 'response.output_text.delta', delta: json.slice(0, 37) })}\n\n`,
      `data: ${JSON.stringify({ type: 'response.output_text.delta', delta: json.slice(37) })}\n\n`,
      `data: ${JSON.stringify({ type: 'response.completed', response: { id: 'resp_test', usage: { input_tokens: 11, output_tokens: 17, input_tokens_details: { cached_tokens: 3 } } } })}\n\n`,
      'data: [DONE]\n\n',
    ].join('');
    const bytes = new TextEncoder().encode(sse);
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(bytes.slice(0, 73));
        controller.enqueue(bytes.slice(73, 149));
        controller.enqueue(bytes.slice(149));
        controller.close();
      },
    });
    vi.stubGlobal('fetch', vi.fn(async () => new Response(stream, { status: 200, headers: { 'content-type': 'text/event-stream' } })));
    const progress: ModelProgress[] = [];

    const result = await callStructured<ResearchReport>({
      apiKey: 'test', model: 'gpt-5.6-luna', agentId: 'mira', system: 'system', user: 'user',
      schemaName: 'test', schema: {}, maxOutputTokens: 1000, maxInputBytes: 1000, webSearch: false,
      onProgress: (update) => { progress.push(update); },
    });

    expect(result.ok).toBe(true);
    expect(result.value).toEqual(report);
    expect(result.responseId).toBe('resp_test');
    expect(result.usage).toEqual({ inputTokens: 11, cachedInputTokens: 3, outputTokens: 17 });
    expect(progress[0]).toEqual({ status: 'connecting', outputCharacters: 0 });
    expect(progress.at(-1)).toEqual({ status: 'complete', outputCharacters: json.length });
    expect(progress.every((update) => Object.keys(update).sort().join(',') === 'outputCharacters,status')).toBe(true);
  });
});
