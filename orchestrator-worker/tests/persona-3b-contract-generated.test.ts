import { existsSync, readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import {
  PERSONA_3B_BLIND_MEMBERSHIP,
  PERSONA_3B_CALL_CEILING,
  PERSONA_3B_FINAL_MANIFEST_HASH,
  PERSONA_3B_JUDGE_PROMPT,
  PERSONA_3B_PRIMARY_PAIR_COUNT,
  PERSONA_3B_PROMPT_SHA256,
  PERSONA_3B_REPEAT_RECORD_COUNT,
  PERSONA_3B_RETRY_ADJUDICATION_RESERVE,
  PERSONA_3B_SCHEMA_SHA256,
  PERSONA_3B_SCORE_SCHEMA,
  PERSONA_3B_SYNTHESIS_RESERVED_CALLS,
  PERSONA_3B_TOTAL_RECORDS,
  validatePersona3BInput,
  validPersona3BScore,
} from '../src/persona-3b-contract.generated';

type Pair = {
  record_id: string;
  scenario: string;
  A: string;
  B: string;
  truncation: { A: boolean; B: boolean };
  AResponseSha256: string;
  BResponseSha256: string;
};

const pairFile = new URL('../../research/experiment-003b/.local-test/blind-final/blind-pairs.jsonl', import.meta.url);
const localPairs: Pair[] = existsSync(pairFile) ? readFileSync(pairFile, 'utf8').trim().split(/\r?\n/).filter(Boolean).map((line) => JSON.parse(line) as Pair) : [];

function scoreSet() {
  return {
    relational_stance: null,
    self_positioning: null,
    assertiveness: null,
    affective_tone: null,
    voice_coherence: null,
    task_completion: null,
    topical_fidelity: null,
    specificity: null,
    refusal_safety: null,
    lexical_format_artifact: null,
  };
}

describe('generated Experiment 3B frozen contract', () => {
  it('exports synchronized bounded counts and strict assets', () => {
    expect(PERSONA_3B_TOTAL_RECORDS).toBe(780);
    expect(PERSONA_3B_PRIMARY_PAIR_COUNT).toBe(768);
    expect(PERSONA_3B_REPEAT_RECORD_COUNT).toBe(117);
    expect(PERSONA_3B_SYNTHESIS_RESERVED_CALLS).toBe(1);
    expect(PERSONA_3B_RETRY_ADJUDICATION_RESERVE).toBe(128);
    expect(PERSONA_3B_CALL_CEILING).toBe(1_014);
    expect(PERSONA_3B_PRIMARY_PAIR_COUNT + PERSONA_3B_REPEAT_RECORD_COUNT + PERSONA_3B_SYNTHESIS_RESERVED_CALLS + PERSONA_3B_RETRY_ADJUDICATION_RESERVE).toBe(PERSONA_3B_CALL_CEILING);
    expect(Object.keys(PERSONA_3B_BLIND_MEMBERSHIP)).toHaveLength(768);
    expect(PERSONA_3B_PROMPT_SHA256).toMatch(/^[0-9a-f]{64}$/);
    expect(PERSONA_3B_SCHEMA_SHA256).toMatch(/^[0-9a-f]{64}$/);
    expect(PERSONA_3B_FINAL_MANIFEST_HASH).toMatch(/^[0-9a-f]{64}$/);
    expect(PERSONA_3B_JUDGE_PROMPT).toContain('Response-level dimensions');
    expect(PERSONA_3B_JUDGE_PROMPT).toContain('Do not apply candidate-selection thresholds');
    expect(PERSONA_3B_SCORE_SCHEMA.additionalProperties).toBe(false);
  });

  it('rejects tampered or unbound inputs', async () => {
    const recordId = Object.keys(PERSONA_3B_BLIND_MEMBERSHIP)[0];
    const base = { recordId, scenario: 'wrong', A: 'wrong', B: 'wrong', truncation: { A: false, B: false }, AResponseSha256: '0'.repeat(64), BResponseSha256: '0'.repeat(64), repeatIndex: 0, orderSwap: false };
    expect(await validatePersona3BInput(base)).toBe(false);
    expect(await validatePersona3BInput({ ...base, repeatIndex: 1, orderSwap: true })).toBe(false);
  });

  it('requires disagreement membership and correct repeat orientation when local frozen pairs are available', async () => {
    if (localPairs.length === 0) return;
    const repeatId = Object.keys(PERSONA_3B_BLIND_MEMBERSHIP).find((id) => PERSONA_3B_BLIND_MEMBERSHIP[id].repeatEligible)!;
    const pair = localPairs.find((item) => item.record_id === repeatId)!;
    const primary = { recordId: pair.record_id, scenario: pair.scenario, A: pair.A, B: pair.B, truncation: pair.truncation, AResponseSha256: pair.AResponseSha256, BResponseSha256: pair.BResponseSha256, repeatIndex: 0, orderSwap: false };
    expect(await validatePersona3BInput(primary)).toBe(true);
    const repeat = { ...primary, A: pair.B, B: pair.A, truncation: { A: pair.truncation.B, B: pair.truncation.A }, AResponseSha256: pair.BResponseSha256, BResponseSha256: pair.AResponseSha256, repeatIndex: 1, orderSwap: true };
    expect(await validatePersona3BInput(repeat)).toBe(true);
    const disagreement = { ...primary, repeatIndex: 2 };
    expect(await validatePersona3BInput(disagreement)).toBe(false);
    expect(await validatePersona3BInput(disagreement, { disagreementRecordIds: new Set([pair.record_id]) })).toBe(true);
  });

  it('accepts only schema-shaped score objects', () => {
    const valid = { record_id: 'p0000001', scores: { A: scoreSet(), B: scoreSet() }, pairwise: Object.fromEntries(['relational_stance', 'self_positioning', 'assertiveness', 'affective_tone', 'voice_coherence'].map((key) => [key, { direction: 'unknown', magnitude: 0 }])), evidence: [], notes: '' };
    expect(validPersona3BScore(valid)).toBe(true);
    expect(validPersona3BScore({ ...valid, scores: { A: { ...scoreSet(), unknown: 1 }, B: scoreSet() } })).toBe(false);
  });
});