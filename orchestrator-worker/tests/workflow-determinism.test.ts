import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

const workflowSource = readFileSync(new URL('../src/workflow.ts', import.meta.url), 'utf8');

describe('workflow replay determinism', () => {
  it('persists both phase deadlines as durable step outputs', () => {
    expect(workflowSource).toContain('const researchDeadline = await step.do');
    expect(workflowSource).toContain('const meetingDeadline = await step.do');
  });

  it('does not branch on wall-clock time outside a step', () => {
    expect(workflowSource).not.toContain('if (Date.now()');
  });
});
