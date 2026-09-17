import { beforeAll, describe, expect, it } from 'vitest';
import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { claimPersona3B, finalizePersona3B, scorePersona3B } from '../src/persona-3b';

const TOKEN = 'integration-test-token';
const RUN_ID = 'persona-3b-00000000-0000-4000-8000-000000000000';
const SHARD_ID = RUN_ID + '-primary-0';
// The frozen blind pair lives in a gitignored private-run file; on a machine
// without it (CI) this suite is skipped rather than failed.
const FROZEN_PAIRS_PATH = fileURLToPath(new URL('../../research/experiment-003b/.local-test/blind-final/blind-pairs.jsonl', import.meta.url));
const HAS_FROZEN_PAIRS = existsSync(FROZEN_PAIRS_PATH);
const FROZEN_PAIR = (HAS_FROZEN_PAIRS
  ? JSON.parse(readFileSync(FROZEN_PAIRS_PATH, 'utf8').split('\n')[0])
  : {}) as Record<string, unknown>;

function responseBody(response: Response) {
  return response.json() as Promise<Record<string, unknown>>;
}

function request(path: string, body: Record<string, unknown>) {
  return new Request('https://worker.test' + path, {
    method: 'POST',
    headers: { authorization: 'Bearer ' + TOKEN, 'content-type': 'application/json' },
    body: JSON.stringify(body),
  });
}

function scoreBody() {
  return {
    shardId: SHARD_ID,
    leaseToken: 'lease-token-123456',
    recordId: 'p0000001',
    scenario: FROZEN_PAIR.scenario,
    A: FROZEN_PAIR.A,
    B: FROZEN_PAIR.B,
    truncation: FROZEN_PAIR.truncation,
    AResponseSha256: FROZEN_PAIR.AResponseSha256,
    BResponseSha256: FROZEN_PAIR.BResponseSha256,
    repeatIndex: 0,
    orderSwap: false,
  };
}

class Statement {
  constructor(private readonly db: FakeD1, private readonly sql: string) {}
  private values: unknown[] = [];
  bind(...values: unknown[]) {
    this.values = values;
    return this;
  }
  async first<T = Record<string, unknown>>() {
    return this.db.first<T>(this.sql, this.values);
  }
  async all<T = Record<string, unknown>>() {
    return this.db.all<T>(this.sql);
  }
  async run() {
    return this.db.run(this.sql, this.values);
  }
}

class FakeD1 {
  runRow: Record<string, unknown> | null = null;
  shards: Record<string, unknown>[] = [];
  attempts: { status: string }[] = [];
  reviewCount = 0;
  scoreCounts = { 0: 768, 1: 117, 2: 0 };
  reservationChanges = 1;
  finalized = false;

  prepare(sql: string) {
    return new Statement(this, sql);
  }
  async batch(statements: Statement[]) {
    return Promise.all(statements.map((statement) => statement.run()));
  }
  async first<T>(sql: string, values: unknown[]) {
    if (sql.includes('FROM persona_3b_shards WHERE id=')) {
      return this.shards.find((shard) => shard.id === values[0]) as T | undefined;
    }
    if (sql.includes('SELECT response_json FROM persona_3b_scores')) return null;
    if (sql.includes('SELECT analyst_receipt_json')) return null;
    if (sql.includes('COUNT(*) AS n FROM persona_3b_review_records')) return { n: this.reviewCount } as T;
    if (sql.includes('COUNT(*) AS n FROM persona_3b_scores') && sql.includes('repeat_index=0')) return { n: this.scoreCounts[0] } as T;
    if (sql.includes('COUNT(*) AS n FROM persona_3b_scores') && sql.includes('repeat_index=1')) return { n: this.scoreCounts[1] } as T;
    if (sql.includes('COUNT(*) AS n FROM persona_3b_scores') && sql.includes('repeat_index=2')) return { n: this.scoreCounts[2] } as T;
    if (sql.includes('FROM persona_3b_runs WHERE id=')) return this.runRow as T;
    return null;
  }
  async all<T>(sql: string) {
    if (sql.includes('FROM persona_3b_runs WHERE status IN')) return { results: this.runRow ? [this.runRow as T] : [] };
    if (sql.includes('FROM persona_3b_shards WHERE run_id=')) return { results: this.shards as T[] };
    if (sql.includes('FROM persona_3b_runs WHERE status')) return { results: this.runRow ? [this.runRow as T] : [] };
    return { results: [] as T[] };
  }
  async run(sql: string, values: unknown[] = []) {
    if (sql.includes('INSERT OR IGNORE INTO persona_3b_attempts')) {
      this.attempts.push({ status: String(values[4]) });
      return { meta: { changes: 1 } };
    }
    if (sql.includes('UPDATE persona_3b_attempts SET status=') && sql.includes("status='failed'")) {
      for (const attempt of this.attempts) attempt.status = 'failed';
      return { meta: { changes: 1 } };
    }
    if (sql.includes('UPDATE persona_3b_runs SET call_count=call_count+1') || sql.includes('UPDATE persona_3b_runs SET call_count=call_count+?')) {
      return { meta: { changes: this.reservationChanges } };
    }
    if (sql.includes('UPDATE persona_3b_shards SET status=' + "'claimed'")) {
      const shard = this.shards.find((item) => item.id === values[3]);
      if (shard) shard.status = 'claimed';
      return { meta: { changes: 1 } };
    }
    if (sql.includes('UPDATE persona_3b_runs SET status=' + "'complete'")) {
      if (this.runRow?.status === 'cancelled') return { meta: { changes: 0 } };
      this.finalized = true;
      return { meta: { changes: 1 } };
    }
    if (sql.includes('INSERT INTO persona_3b_events')) return { meta: { changes: 1 } };
    return { meta: { changes: 1 } };
  }
}

const env = (db: FakeD1) => ({
  DB: db,
  PERSONA_3B_TOKEN: TOKEN,
  PERSONA_3B_SCORING_ENABLED: 'true',
  OPENAI_API_KEY: 'mock-provider-key',
  PERSONA_3B_MANIFEST_HASH: '7e3b333e93c0bd9c6d04ca3b7cc6f8a7ed4e00887a4080623abafec2ce8b20c7',
} as unknown as Env);

describe.skipIf(!HAS_FROZEN_PAIRS)('Experiment 3b zero-spend route integration', () => {
  beforeAll(() => {
    const subtle = crypto.subtle as unknown as { timingSafeEqual?: (left: ArrayBuffer, right: ArrayBuffer) => boolean };
    if (!subtle.timingSafeEqual) {
      subtle.timingSafeEqual = (left, right) => {
        const a = new Uint8Array(left);
        const b = new Uint8Array(right);
        return a.length === b.length && a.every((value, index) => value === b[index]);
      };
    }
  });

  it('releases repeat shards only after all primary shards are complete', async () => {
    const db = new FakeD1();
    db.runRow = { id: RUN_ID, status: 'running', phase: 'development-screen-scoring' };
    db.shards = [
      { id: SHARD_ID, phase: 'primary', shard_index: 0, status: 'complete', expected_records: 1 },
      { id: RUN_ID + '-repeat-0', phase: 'repeat', shard_index: 0, status: 'queued', expected_records: 1 },
      { id: RUN_ID + '-disagreement-0', phase: 'disagreement', shard_index: 0, status: 'blocked', expected_records: 0 },
      { id: RUN_ID + '-synthesis-0', phase: 'synthesis', shard_index: 0, status: 'blocked', expected_records: 0 },
    ];
    const response = await claimPersona3B(request('/api/persona-3b/next', {}), env(db), {});
    expect(response.status).toBe(200);
    expect((await responseBody(response)).shard).toMatchObject({ phase: 'repeat', shardIndex: 0 });
  });

  it('does not leave a failed budget reservation permanently in flight', async () => {
    const db = new FakeD1();
    db.runRow = { id: RUN_ID, status: 'running' };
    db.shards = [{ id: SHARD_ID, phase: 'primary', shard_index: 0, status: 'claimed', expected_records: 1, claim_token_hash: createHash('sha256').update('lease-token-123456').digest('hex') }];
    db.reservationChanges = 0;
    const response = await scorePersona3B(request('/api/persona-3b/score', scoreBody()), env(db), {});
    expect(response.status).toBe(409);
    expect(db.attempts.every((attempt) => attempt.status !== 'in_flight')).toBe(true);
  });

  it('rejects finalization when the persisted analyst receipt is missing', async () => {
    const db = new FakeD1();
    db.runRow = { id: RUN_ID, status: 'awaiting_adjudication' };
    const response = await finalizePersona3B(request('/api/persona-3b/finalize', {
      runId: RUN_ID,
      analysisDigest: 'a'.repeat(64),
      artifactHash: 'b'.repeat(64),
      manifestHash: '7e3b333e93c0bd9c6d04ca3b7cc6f8a7ed4e00887a4080623abafec2ce8b20c7',
      sourceArtifactHash: '6e479c2e02aa912f6bcf6b2e57554ed75711e93bb3fc3da813114bba871f3fe7',
      counts: { sourceRecords: 780, primary: 768, repeat: 117, disagreement: 0, synthesis: 1 },
      shortlistSummary: [],
      confirmationStatus: 'pending',
    }), env(db), {});
    expect(response.status).toBe(409);
  });
});
