import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

const indexSource = readFileSync(new URL('../src/index.ts', import.meta.url), 'utf8');
const workflowSource = readFileSync(new URL('../src/workflow.ts', import.meta.url), 'utf8');

describe('stopped-run continuation safety', () => {
  it('keeps resume private, bounded, and atomic', () => {
    expect(indexSource).toContain("url.pathname === '/api/experiments/resume'");
    expect(indexSource).toContain('secretEquals(bearer(request), env.ADMIN_TOKEN)');
    expect(indexSource).toContain("status IN ('error','paused')");
    expect(indexSource).toContain('MAX_TARGET_ROUNDS = 200');
    expect(indexSource).toContain('DEFAULT_SESSION_ROUNDS = 45');
    expect(indexSource).toContain('MAX_SESSION_ROUNDS = 145');
  });

  it('preserves a revealed interrupted round instead of rerunning its research', () => {
    expect(indexSource).toContain('seq % 1000=514');
    expect(indexSource).toContain('resumeMeetingRound');
    expect(workflowSource).toContain('revealedResearchBatch');
    expect(workflowSource).toContain('repeatedResearchCalls: 0');
  });

  it('pauses at a planned platform-capacity checkpoint', () => {
    expect(workflowSource).toContain('params.sessionEndRound');
    expect(workflowSource).toContain("status='paused',phase='paused'");
    expect(workflowSource).toContain("title: 'Cloudflare safety pause'");
  });

  it('keeps the hard OpenAI ceiling and protected reserve', () => {
    expect(indexSource).toContain('const BUDGET_USD = 50');
    expect(indexSource).toContain('const RESERVE_USD = 1.5');
    expect(indexSource).toContain('BUDGET_USD - RESERVE_USD');
  });
});
