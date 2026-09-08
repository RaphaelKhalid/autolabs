import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';
import { describe, expect, it } from 'vitest';

describe('exact calculator input normalization', () => {
  it('executes an 84-difference family over a disclosed deterministic 80-item domain', () => {
    const directory = mkdtempSync(join(tmpdir(), 'autolabs-family-'));
    try {
      const worker = fileURLToPath(new URL('../../math-worker/run.mjs', import.meta.url));
      const payload = {
        id: 'family-limit-test',
        sourceSha: '0'.repeat(40),
        jobType: 'family_scan',
        params: {
          differences: Array.from({ length: 84 }, (_, index) => String(84 - index)).join(' '),
          limit: 1,
          maxChecks: 1_000,
        },
        evidenceManifest: { domain: 'The submitted 84-value test family.' },
      };
      const processResult = spawnSync(process.execPath, [worker], {
        cwd: directory,
        env: { ...process.env, JOB_PAYLOAD: JSON.stringify(payload) },
        encoding: 'utf8',
      });
      const output = JSON.parse(readFileSync(join(directory, 'result.json'), 'utf8'));

      expect(processResult.status).toBe(0);
      expect(output.ok).toBe(true);
      expect(output.result.differences).toHaveLength(80);
      expect(output.result.differences.at(-1)).toBe('80');
      expect(output.result.inputNormalization).toEqual({
        requestedCount: 84,
        effectiveCount: 80,
        omittedCount: 4,
        selection: 'numerically-smallest',
      });
      expect(output.result.certificate.testedDomain).toContain('numerically smallest 80');
      expect(output.result.certificate.testedDomain).toContain('4 larger values were omitted');
    } finally {
      rmSync(directory, { recursive: true, force: true });
    }
  });
});
