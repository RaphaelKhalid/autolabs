import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { checkpointRoundsThrough } from '../src/checkpoint-outbox';

const outboxSource = readFileSync(new URL('../src/checkpoint-outbox.ts', import.meta.url), 'utf8');
const migrationSource = readFileSync(new URL('../migrations/0004_checkpoint_email_outbox.sql', import.meta.url), 'utf8');

describe('checkpoint email outbox', () => {
  it('selects only completed five-round checkpoints', () => {
    expect(checkpointRoundsThrough(34)).toEqual([]);
    expect(checkpointRoundsThrough(35)).toEqual([35]);
    expect(checkpointRoundsThrough(44)).toEqual([35, 40]);
    expect(checkpointRoundsThrough(50)).toEqual([35, 40, 45, 50]);
  });

  it('deduplicates by run and checkpoint while leaving delivery pending', () => {
    expect(migrationSource).toContain('PRIMARY KEY (run_id, checkpoint_round)');
    expect(outboxSource).toContain('INSERT OR IGNORE INTO checkpoint_email_outbox');
    expect(outboxSource).toContain("VALUES(?,?,?,?,?,?,'pending',0,?)");
  });

  it('builds summaries exclusively from public revealed events', () => {
    expect(outboxSource).toContain('WHERE run_id=? AND visible=1 AND round BETWEEN ? AND ?');
    const queueSource = outboxSource.slice(
      outboxSource.indexOf('export async function queueCheckpointSummary'),
      outboxSource.indexOf('export async function queueCheckpointSummariesThrough'),
    );
    expect(queueSource).not.toContain('private_plans');
    expect(queueSource).not.toContain('privateNextPlan');
    expect(queueSource).not.toContain('payload_json');
  });
});
