import { describe, expect, it } from 'vitest';
import { costUsd } from '../src/openai';
import { compareProgressMetric, verifyRectangle } from '../src/verifier';

describe('orchestration invariants', () => {
  it('prices Luna usage without double-counting cached or reasoning tokens', () => {
    expect(costUsd({ inputTokens: 1_000_000, cachedInputTokens: 250_000, outputTokens: 100_000 }))
      .toBeCloseTo(0.275, 8);
  });

  it('accepts the exact 4x4 Bremner rectangle but does not call it k=5', () => {
    const result = verifyRectangle({
      numbers: ['26128575', '291722431', '561117375', '713526975'],
      differences: ['126', '16110', '33390', '75390'],
      note: 'known fixture',
    });
    expect(result.accepted).toBe(true);
    expect(result.missing).toEqual([]);
    expect(result.isK5).toBe(false);
    expect(result.improvesSota).toBe(false);
    expect(result.support).toEqual([4, 4, 4, 4]);
  });

  it('never rewards numerical closeness', () => {
    const result = verifyRectangle({ numbers: ['26'], differences: ['24'], note: 'near square' });
    expect(result.exactCells).toBe(0);
    expect(result.missing).toEqual([{ number: '26', difference: '24' }]);
  });

  it('ranks a balanced 4x4 rectangle above an isolated 1x12 anchor', () => {
    const balanced = verifyRectangle({
      numbers: ['26128575', '291722431', '561117375', '713526975'],
      differences: ['126', '16110', '33390', '75390'],
      note: 'known fixture',
    });
    const anchor = verifyRectangle({
      numbers: ['105300'],
      differences: ['216', '243', '345', '405', '488', '519', '552', '680', '783', '953', '1080', '1329'],
      note: 'single column',
    });
    expect(compareProgressMetric(balanced.progressMetric, anchor.progressMetric)).toBeGreaterThan(0);
  });
});
