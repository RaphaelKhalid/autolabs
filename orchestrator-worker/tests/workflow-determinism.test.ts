import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

const workflowSource = readFileSync(new URL('../src/workflow.ts', import.meta.url), 'utf8');

describe('workflow replay determinism', () => {
  it('persists both phase deadlines as durable step outputs', () => {
    expect(workflowSource).toContain('const researchDeadline = await step.do');
    expect(workflowSource).toContain('const meetingDeadline = await step.do');
    expect(workflowSource).toContain('const councilDeadline = await step.do');
  });

  it('does not branch on wall-clock time outside a step', () => {
    expect(workflowSource).not.toContain('if (Date.now()');
  });

  it('isolates failed parallel research calls instead of aborting the round', () => {
    expect(workflowSource).toContain('return isolatedResearchFailure(prompt, error);');
    expect(workflowSource).toContain('const research = await Promise.all(prepared.map(async (prompt) => {');
  });
});
