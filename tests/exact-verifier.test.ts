import { describe, expect, it } from 'vitest';
import {
  BREMNER_K4_FIXTURE,
  differenceSet,
  factorizationFromDifference,
  scoreCandidateTable,
  verifyCommonWitness,
  verifyWitness,
} from '../lib/exact-verifier';

describe('exact Erdos 885 verifier', () => {
  it('recovers exact factor pairs without floating point arithmetic', () => {
    expect(differenceSet(26n)).toEqual([25n, 11n]);
    expect(factorizationFromDifference(26n, 25n)?.factorPair).toEqual({ a: 1n, b: 26n });
    expect(factorizationFromDifference(26n, 24n)).toBeUndefined();
  });

  it('verifies Bremner’s k=4 certificate', () => {
    const result = verifyCommonWitness(BREMNER_K4_FIXTURE, 4);
    expect(result.accepted, result.rejections.map((item) => item.detail).join('; ')).toBe(true);
    expect(result.cells).toHaveLength(16);
  });

  it('rejects structurally invalid k=5 claims deterministically', () => {
    expect(verifyWitness({ numbers: [26n], differences: [24n] }).rejections.map((item) => item.code))
      .toEqual(['wrong-number-count', 'wrong-difference-count']);
  });

  it('scores exact supports', () => {
    const score = scoreCandidateTable({ numbers: [26n, 15n], differences: [25n, 14n] });
    expect(score.supports).toEqual([[true, false], [false, true]]);
    expect(score.totalSupport).toBe(2);
  });
});
