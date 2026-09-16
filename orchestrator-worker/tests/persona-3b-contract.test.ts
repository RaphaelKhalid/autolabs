import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { PERSONA_3B_CALL_CEILING, PERSONA_3B_DISAGREEMENT_RECORDS, PERSONA_3B_MANIFEST_HASH, PERSONA_3B_REPEAT_RECORDS, PERSONA_3B_SOURCE_ARTIFACT_HASH, PERSONA_3B_TOTAL_RECORDS } from '../src/persona-3b';

describe('Experiment 3B control-plane contract', () => {
  it('keeps the frozen cohort and call arithmetic bounded', () => {
    expect(PERSONA_3B_TOTAL_RECORDS).toBe(780);
    expect(PERSONA_3B_REPEAT_RECORDS).toBe(117);
    expect(PERSONA_3B_DISAGREEMENT_RECORDS).toBe(117);
    expect(PERSONA_3B_CALL_CEILING).toBe(1_014);
    expect(PERSONA_3B_CALL_CEILING - 885).toBe(129);
  });

  it('routes every score through the worker-owned blinded scorer', () => {
    const source = readFileSync(new URL('../src/persona-3b.ts', import.meta.url), 'utf8');
    const router = readFileSync(new URL('../src/index.ts', import.meta.url), 'utf8');
    expect(router).toContain("url.pathname === '/api/persona-3b/score'");
    expect(source).toContain("store:false");
    expect(source).toContain('reserved_usd=reserved_usd+?');
    expect(source).toContain('spent_usd+reserved_usd+?<=budget_usd');
    expect(source).toContain('chargedWorstCase: true');
  });

  it('pins the source and creates isolated telemetry tables', () => {
    expect(PERSONA_3B_MANIFEST_HASH).toMatch(/^[0-9a-f]{64}$/);
    expect(PERSONA_3B_SOURCE_ARTIFACT_HASH).toMatch(/^[0-9a-f]{64}$/);
    const migration = readFileSync(new URL('../migrations/0007_persona_3b_scoring.sql', import.meta.url), 'utf8');
    expect(migration).toContain('call_ceiling INTEGER NOT NULL CHECK (call_ceiling = 1014)');
    expect(migration).toContain('CREATE TABLE IF NOT EXISTS persona_3b_events');
    expect(migration).toContain("status IN ('blocked', 'queued', 'claimed', 'running', 'complete', 'failed')");
    const scores = readFileSync(new URL('../migrations/0008_persona_3b_scores.sql', import.meta.url), 'utf8');
    expect(scores).toContain('CREATE TABLE IF NOT EXISTS persona_3b_scores');
    expect(scores).toContain('UNIQUE (shard_id, record_id, repeat_index)');
  });
});
