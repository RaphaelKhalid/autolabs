import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

const workflowSource = readFileSync(new URL('../src/workflow.ts', import.meta.url), 'utf8');

describe('free-plan workflow chaining', () => {
  it('runs one competition round per Workflow instance', () => {
    expect(workflowSource).toContain(
      'round <= Math.min(startRound, params.targetRounds)',
    );
  });

  it('starts the next round as a new durable instance', () => {
    expect(workflowSource).toContain('this.env.AUTOLABS_WORKFLOW.create');
    expect(workflowSource).toContain('params: { ...params, startRound: nextRound }');
  });
});
