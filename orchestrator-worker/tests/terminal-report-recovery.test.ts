import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

const workflowSource = readFileSync(new URL('../src/workflow.ts', import.meta.url), 'utf8');

describe('terminal report recovery', () => {
  it('stores a compact report index instead of duplicating multi-megabyte job and retrieval payloads', () => {
    expect(workflowSource).toContain('schemaVersion: 2');
    expect(workflowSource).toContain('researchRetrieval: retrievalRows.map(({ sources, ...row }, offset)');
    expect(workflowSource).toContain('sourceCount: sources.length');
    expect(workflowSource).not.toContain('researchRetrieval: retrievalRows,');
    expect(workflowSource).not.toContain('params: JSON.parse(job.params)');
    expect(workflowSource).not.toContain('result: job.result ? JSON.parse(job.result) : null');
  });

  it('still releases every private plan and points to the complete paginated evidence ledger', () => {
    expect(workflowSource).toContain('privatePlan: JSON.parse(row.plan)');
    expect(workflowSource).toContain('completeEventLedger: `/api/experiments/${params.runId}/events`');
  });

  it('limits report-only recovery to a stopped run that completed all target rounds', () => {
    expect(workflowSource).toContain('if (params.finalizeOnly)');
    expect(workflowSource).toContain("row.status !== 'error' && row.status !== 'paused'");
    expect(workflowSource).toContain('completedRound < row.targetRounds');
    expect(workflowSource).toContain("finalReport(this.env, params, eureka ? 'eureka' : 'complete', completedRound)");
  });
});
