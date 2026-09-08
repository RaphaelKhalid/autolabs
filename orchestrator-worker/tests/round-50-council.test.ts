import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

const workflowSource = readFileSync(new URL('../src/workflow.ts', import.meta.url), 'utf8');
const promptSource = readFileSync(new URL('../src/prompts.ts', import.meta.url), 'utf8');

describe('round 50 research council', () => {
  it('runs after the normal round and before the checkpoint and continuation', () => {
    const completed = workflowSource.indexOf('completedRound = round;');
    const council = workflowSource.indexOf('if (round === 50) await runRound50Council');
    const checkpoint = workflowSource.indexOf('checkpointRoundsThrough(completedRound)', council);
    expect(completed).toBeGreaterThan(-1);
    expect(council).toBeGreaterThan(completed);
    expect(checkpoint).toBeGreaterThan(council);
  });

  it('publishes advisory asks without applying them', () => {
    expect(workflowSource).toContain("title: consensus.ok ? 'Round 50 research council — unified asks'");
    expect(workflowSource).toContain('ownerReviewRequired: true');
    expect(workflowSource).toContain('automaticallyApplied: false');
  });

  it('continues round 51 under the existing policy', () => {
    expect(promptSource).toContain('round 51 must continue under the existing policy');
    expect(promptSource).toContain('not silent implementation of any ask');
  });

  it('rejects generic scale asks without a five-minute-loop mechanism', () => {
    expect(promptSource).toContain('Do not ask merely for a larger model, more agents, more budget, longer loops, or a larger context window');
    expect(promptSource).toContain('why retrieval or compression cannot preserve it');
    expect(promptSource).toContain('falsifiable expected benefit');
  });
});
