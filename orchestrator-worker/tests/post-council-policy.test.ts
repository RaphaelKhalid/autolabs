import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

const promptSource = readFileSync(new URL('../src/prompts.ts', import.meta.url), 'utf8');
const jobSource = readFileSync(new URL('../src/github-jobs.ts', import.meta.url), 'utf8');
const verifierSource = readFileSync(new URL('../src/verifier.ts', import.meta.url), 'utf8');
const mathWorkerSource = readFileSync(new URL('../../math-worker/run.mjs', import.meta.url), 'utf8');
const indexSource = readFileSync(new URL('../src/index.ts', import.meta.url), 'utf8');

describe('approved round 56 research policy', () => {
  it('requires a machine-readable evidence manifest and audited claims', () => {
    expect(promptSource).toContain('const jobManifest');
    expect(promptSource).toContain("completenessTarget: { type: 'string'");
    expect(promptSource).toContain('proofObligations');
    expect(promptSource).toContain('claimAudit');
  });

  it('keeps distinct human mathematics over a shared pair-completion geometry', () => {
    expect(promptSource).toContain('Pair-completion incidence is the shared computational geometry');
    expect(promptSource).toContain('your assigned human-mathematics lane remains independent');
    expect(jobSource).toContain("job.jobType === 'family_scan' ? 30");
  });

  it('enforces a one-decisive-job portfolio and evidence ranking', () => {
    expect(jobSource).toContain('const perAgentLimit = options.round >= 56 ? 1 : 3');
    expect(jobSource).toContain('.filter(hasRound56Manifest)');
    expect(jobSource).toContain('postCouncilPriority');
  });

  it('publishes exact witnesses, replay hashes, and the active policy', () => {
    expect(verifierSource).toContain('exactWitnesses');
    expect(verifierSource).toContain('fnv1a32:');
    expect(mathWorkerSource).toContain('schemaVersion: 3');
    expect(mathWorkerSource).toContain('requestHash:');
    expect(mathWorkerSource).toContain('resultHash:');
    expect(mathWorkerSource).toContain('supportMask');
    expect(indexSource).toContain('POST_COUNCIL_POLICY_SUMMARY');
    expect(indexSource).toContain('state.researchPolicy = POST_COUNCIL_POLICY_SUMMARY');
  });
});
