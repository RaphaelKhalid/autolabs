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

function familyJob(differences: string): ProposedJob {
  const base = job(5_000_000);
  return { ...base, params: { ...base.params, differences } };
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

  it('reduces an oversized family to the numerically smallest 80 unique differences', () => {
    const shuffled = [...Array.from({ length: 82 }, (_, index) => String(index + 1)), '2', '100', '99'];
    const normalized = normalizeJob(familyJob(shuffled.reverse().join(' ')));
    const differences = String(normalized.params.differences).split(' ');

    expect(differences).toHaveLength(80);
    expect(differences.slice(0, 3)).toEqual(['1', '2', '3']);
    expect(differences.at(-1)).toBe('80');
    expect(normalized.reason).toContain('reduced 84 distinct differences');
    expect(normalized.reason).toContain('4 were omitted');
    expect(normalized.manifest.domain).toContain('numerically smallest 80');
    expect(normalized.manifest.domain).toContain('4 larger values were omitted');
    expect(normalized.manifest.stopLoss).toContain('requested 84');
  });

  it('preserves an in-limit family request by identity', () => {
    const valid = familyJob(Array.from({ length: 80 }, (_, index) => String(index + 1)).join(' '));
    expect(normalizeJob(valid)).toBe(valid);
  });
});
