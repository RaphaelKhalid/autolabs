import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

const workflowSource = readFileSync(new URL('../src/workflow.ts', import.meta.url), 'utf8');

describe('workflow phase waits', () => {
  it('never asks Cloudflare to sleep until an already-past deadline', () => {
    expect(workflowSource).not.toContain('step.sleepUntil');
    expect(workflowSource).toContain('Math.max(0, researchDeadline - Date.now())');
    expect(workflowSource).toContain('Math.max(0, meetingDeadline - Date.now())');
  });
});
