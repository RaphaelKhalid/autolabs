import { describe, expect, it } from 'vitest';
import { normalizeJob } from '../src/github-jobs';
import type { ProposedJob } from '../src/types';

function job(maxChecks: number | null): ProposedJob {
  return {
    jobType: 'family_scan',
    params: { differences: '1 2 3 4 5', maxChecks },
    reason: 'Exact bounded scan.',
    manifest: {
      familyFingerprint: 'test-family',
      registryOverlap: 'new',
      domain: 'Five fixed differences.',
      completenessTarget: 'bounded-complete',
      targetShape: '5x5',
      successCriterion: 'Return an exact target.',
      stopLoss: 'Stop at the requested exact-check limit.',
      symbolicIdentity: null,
      proofObligations: ['Verify every cell.', 'Restrict every negative claim to the domain.'],
    },
  };
}

describe('calculator dispatch normalization', () => {
  it('clamps an oversized request and records the effective stop-loss', () => {
    const normalized = normalizeJob(job(20_000_000));
    expect(normalized.params.maxChecks).toBe(5_000_000);
    expect(normalized.reason).toContain('adjusted maxChecks from 20000000');
    expect(normalized.manifest.stopLoss).toContain('Dispatcher-enforced stop: 5000000');
    expect(normalized.manifest.stopLoss).toContain('requested 20000000');
  });

  it('preserves an already valid request', () => {
    const valid = job(2_000_000);
    expect(normalizeJob(valid)).toBe(valid);
  });

  it('clamps undersized requests to the deterministic minimum', () => {
    expect(normalizeJob(job(10)).params.maxChecks).toBe(1_000);
  });
});
