import { describe, expect, it } from 'vitest';
import { extractExactBicliques } from '../../math-worker/biclique.mjs';

function candidate(number, supportMask) {
  return {
    number: String(number),
    supportMask,
    witnesses: supportMask.flatMap((supported, index) => supported
      ? [{ number: String(number), difference: String(index + 1), m: '1', a: '1', b: '1' }]
      : []),
  };
}

describe('exact biclique extraction', () => {
  it('finds an exact 5x5 incidence witness', () => {
    const candidates = Array.from({ length: 5 }, (_, index) => candidate(index + 1, [true, true, true, true, true]));
    const result = extractExactBicliques(candidates, [1n, 2n, 3n, 4n, 5n]);
    const match = result.matches.find((item) => item.target === '5x5');
    expect(match?.numbers).toHaveLength(5);
    expect(match?.differences).toEqual(['1', '2', '3', '4', '5']);
    expect(match?.witnesses).toHaveLength(25);
  });

  it('finds both strict SOTA target orientations', () => {
    const sixByFour = Array.from({ length: 6 }, (_, index) => candidate(index + 1, [true, true, true, true, false]));
    const fourByFive = Array.from({ length: 4 }, (_, index) => candidate(index + 11, [true, true, true, true, true]));
    expect(extractExactBicliques(sixByFour, [1n, 2n, 3n, 4n, 5n]).matches)
      .toContainEqual(expect.objectContaining({ target: '6x4', exactCells: 24 }));
    expect(extractExactBicliques(fourByFive, [1n, 2n, 3n, 4n, 5n]).matches)
      .toContainEqual(expect.objectContaining({ target: '4x5', exactCells: 20 }));
  });

  it('marks an interrupted extraction as partial', () => {
    const candidates = Array.from({ length: 5 }, (_, index) => candidate(index + 1, [true, true, true, true, true]));
    const result = extractExactBicliques(candidates, [1n, 2n, 3n, 4n, 5n], {
      consumeCheck: () => false,
      stopReason: () => 'check_limit',
    });
    expect(result.complete).toBe(false);
    expect(result.truncatedBy).toBe('check_limit');
    expect(result.matches).toEqual([]);
  });
});
