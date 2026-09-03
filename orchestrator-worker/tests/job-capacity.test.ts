import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

const source = readFileSync(new URL('../src/github-jobs.ts', import.meta.url), 'utf8');

describe('exact computation capacity', () => {
  it('supports all 50 rounds without increasing concurrent execution', () => {
    expect(source).toContain('const MAX_ACTIVE_JOBS = 8');
    expect(source).toContain('const MAX_RUN_JOBS = 400');
  });
});
