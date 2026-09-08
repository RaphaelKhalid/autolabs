import { describe, expect, it } from 'vitest';
import { erdosPilot, experiments, pilotSnapshot, pilotReportUrl, pilotLedgerUrl } from '../lib/experiment-catalog';
import { factorizationFromDifference } from '../lib/exact-verifier';
import { verifyRectangle } from '../orchestrator-worker/src/verifier';

describe('archived pilot', () => {
  it('pins evidence to a run rather than the mutable current endpoint', () => {
    expect(new Set(experiments.map(item => item.slug)).size).toBe(experiments.length);
    expect(pilotReportUrl).toContain(erdosPilot.runId);
    expect(pilotLedgerUrl).toContain(erdosPilot.runId);
    expect(erdosPilot.rounds).toBe(100);
  });
  it('reproduces the sparse 21-cell candidate without claiming a solution', () => {
    const { numbers, differences } = pilotSnapshot.candidate;
    const result = verifyRectangle({ numbers, differences, note: 'Archive regression' });
    expect(result.accepted).toBe(true);
    expect(result.exactCells).toBe(21);
    expect(result.missing).toHaveLength(21);
    expect(result.isK5).toBe(false);
    expect(result.improvesSota).toBe(false);
    for (const n of numbers) for (const d of differences) {
      const cell = factorizationFromDifference(n, d);
      if (cell) {
        expect(cell.factorPair.a * cell.factorPair.b).toBe(BigInt(n));
        expect(cell.factorPair.b - cell.factorPair.a).toBe(BigInt(d));
      }
    }
  });
  it('keeps job outcomes and pilot costs distinct from cumulative use', () => {
    expect(Object.values(pilotSnapshot.jobs).reduce((a, b) => a + b, 0)).toBe(606);
    expect(pilotSnapshot.openaiUsd).toBeLessThan(pilotSnapshot.cumulativeOpenaiUsd);
    expect(pilotSnapshot.exaUsd).toBeLessThan(pilotSnapshot.cumulativeExaUsd);
  });
});
