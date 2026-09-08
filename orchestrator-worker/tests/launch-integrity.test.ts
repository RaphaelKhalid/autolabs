import { createHmac } from 'node:crypto';
import { describe, expect, it } from 'vitest';
import { mergeAttempts } from '../src/openai';
import { AGENTS } from '../src/agents';
import { RESEARCH_SCHEMA, researchPrompt } from '../src/prompts';
import { verifyCallbackSignature } from '../src/security';
import type { AgentResult, ResearchReport } from '../src/types';

const emptyReport: ResearchReport = {
  headline: 'bounded',
  thesis: 'test',
  claims: [],
  claimAudit: [],
  equations: [],
  citations: [],
  failedAvenues: [],
  candidates: [],
  proposedJobs: [],
  nextQuestions: [],
};

describe('launch-integrity invariants', () => {
  it('retains usage from a failed first attempt and its successful retry', () => {
    const first: AgentResult<ResearchReport> = {
      agentId: 'mira',
      ok: false,
      responseId: 'resp_first',
      usage: { inputTokens: 100, cachedInputTokens: 20, outputTokens: 30 },
      usageRecords: [{ responseId: 'resp_first', usage: { inputTokens: 100, cachedInputTokens: 20, outputTokens: 30 } }],
      error: 'invalid JSON',
    };
    const second: AgentResult<ResearchReport> = {
      agentId: 'mira',
      ok: true,
      responseId: 'resp_second',
      usage: { inputTokens: 110, cachedInputTokens: 10, outputTokens: 40 },
      usageRecords: [{ responseId: 'resp_second', usage: { inputTokens: 110, cachedInputTokens: 10, outputTokens: 40 } }],
      value: emptyReport,
    };
    const merged = mergeAttempts(first, second);
    expect(merged.usage).toEqual({ inputTokens: 210, cachedInputTokens: 30, outputTokens: 70 });
    expect(merged.usageRecords?.map((record) => record.responseId)).toEqual(['resp_first', 'resp_second']);
  });

  it('uses a closed strict schema for deterministic job parameters', () => {
    const job = RESEARCH_SCHEMA.properties.proposedJobs.items;
    expect(job.additionalProperties).toBe(false);
    expect(job.properties.params.additionalProperties).toBe(false);
    expect(job.properties.params.required).toContain('maxChecks');
    expect(job.properties.params.properties.maxChecks.minimum).toBe(1_000);
    expect(job.properties.params.properties.maxChecks.maximum).toBe(5_000_000);
    const differencePattern = new RegExp(job.properties.params.properties.differences.pattern);
    const eighty = Array.from({ length: 80 }, (_, index) => String(index + 1)).join(' ');
    const eightyOne = `${eighty} 81`;
    expect(differencePattern.test(eighty)).toBe(true);
    expect(differencePattern.test(eightyOne)).toBe(false);
    expect(researchPrompt(AGENTS[0], 56, {}, [], []).system).toContain('80 distinct positive decimal differences');
    expect(job.required).toContain('manifest');
    expect(RESEARCH_SCHEMA.required).toContain('claimAudit');
  });

  it('accepts a fresh HMAC callback and rejects stale or changed bodies', async () => {
    const secret = 'test-only-callback-secret';
    const timestamp = '2000';
    const body = '{"id":"job-1","ok":true,"complete":true}';
    const signature = createHmac('sha256', secret).update(`${timestamp}.${body}`).digest('hex');
    await expect(verifyCallbackSignature({ secret, timestamp, signature, body, now: 2001 })).resolves.toBe(true);
    await expect(verifyCallbackSignature({ secret, timestamp, signature, body: `${body} `, now: 2001 })).resolves.toBe(false);
    await expect(verifyCallbackSignature({ secret, timestamp, signature, body, now: 2401 })).resolves.toBe(false);
  });
});
