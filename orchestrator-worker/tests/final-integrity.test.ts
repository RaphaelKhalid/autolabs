import { describe, expect, it } from 'vitest';
import { callStructured } from '../src/openai';
import { reapStaleJobs } from '../src/github-jobs';
import { readFileSync } from 'node:fs';

const indexSource = readFileSync(new URL('../src/index.ts', import.meta.url), 'utf8');

describe('final launch integrity', () => {
  it('rejects an oversized model prompt before issuing a billable request', async () => {
    const result = await callStructured({
      apiKey: 'must-not-be-used',
      model: 'gpt-5.6-luna',
      agentId: 'mira',
      system: 'system',
      user: 'x'.repeat(101),
      schemaName: 'never_sent',
      schema: { type: 'object', additionalProperties: false, properties: {} },
      maxOutputTokens: 10,
      maxInputBytes: 100,
      webSearch: false,
    });

    expect(result.ok).toBe(false);
    expect(result.responseId).toBeUndefined();
    expect(result.usageRecords).toEqual([]);
    expect(result.error).toContain('Prompt exceeded');
  });

  it('uses numeric SQLite time comparison for stale compute leases', async () => {
    let sql = '';
    const statement = {
      bind: () => statement,
      run: async () => ({ success: true }),
    };
    const db = {
      prepare: (value: string) => {
        sql = value;
        return statement;
      },
    } as unknown as D1Database;

    await reapStaleJobs(db, 'run-1');
    expect(sql).toContain("julianday(created_at) < julianday('now','-55 minutes')");
  });

  it('keeps the live state payload bounded while preserving paginated history', () => {
    expect(indexSource).toContain('recentEventSummaries(env.DB, row.id, 120)');
    expect(indexSource).toContain("url.searchParams.get('before')");
    expect(indexSource).toContain('nextBefore');
  });
});
