import { beforeAll, describe, expect, it } from 'vitest';
import { createHash } from 'node:crypto';
import { canonicalJson, persona3CStatus, reportPersona3C, startPersona3C, stopPersona3C } from '../src/persona-3c';

const TOKEN = 'persona-3c-test-token';
const MANIFEST_HASH = 'a'.repeat(64);

function request(path: string, body: Record<string, unknown>, token = TOKEN) {
  return new Request('https://worker.test' + path, {
    method: 'POST',
    headers: { authorization: 'Bearer ' + token, 'content-type': 'application/json' },
    body: JSON.stringify(body),
  });
}

async function responseBody(response: Response) {
  return response.json() as Promise<Record<string, unknown>>;
}

function shaOf(payload: Record<string, unknown>) {
  return createHash('sha256').update(canonicalJson(payload)).digest('hex');
}

type RunRow = Record<string, unknown> & { id: string; idempotency_key: string; status: string };
type RecordRow = { run_id: string; stage: string; record_id: string; payload_json: string; sha256: string; created_at: string };
type ProgressRow = { run_id: string; stage: string; done: number; total: number; updated_at: string };
type EventRow = { id: number; run_id: string; at: string; stage: string | null; kind: string; title: string; summary: string; payload_json: string };

class Statement {
  private values: unknown[] = [];
  constructor(private readonly db: FakeD1, private readonly sql: string) {}
  bind(...values: unknown[]) {
    this.values = values;
    return this;
  }
  async first<T = Record<string, unknown>>() {
    return this.db.first<T>(this.sql, this.values);
  }
  async all<T = Record<string, unknown>>() {
    return this.db.all<T>(this.sql, this.values);
  }
  async run() {
    return this.db.run(this.sql, this.values);
  }
}

class FakeD1 {
  runs: RunRow[] = [];
  records: RecordRow[] = [];
  progress: ProgressRow[] = [];
  events: EventRow[] = [];
  nextEventId = 1;

  prepare(sql: string) {
    return new Statement(this, sql);
  }

  async batch(statements: Statement[]) {
    const results = [];
    for (const statement of statements) results.push(await statement.run());
    return results;
  }

  async first<T>(sql: string, values: unknown[]): Promise<T | null> {
    if (sql.includes('FROM persona_3c_runs WHERE idempotency_key=')) {
      return (this.runs.find((r) => r.idempotency_key === values[0]) as unknown as T) ?? null;
    }
    if (sql.includes('FROM persona_3c_runs WHERE id=')) {
      return (this.runs.find((r) => r.id === values[0]) as unknown as T) ?? null;
    }
    if (sql.includes('FROM persona_3c_runs ORDER BY created_at DESC LIMIT 1')) {
      const sorted = [...this.runs].sort((a, b) => String(a.created_at).localeCompare(String(b.created_at)));
      return (sorted[sorted.length - 1] as unknown as T) ?? null;
    }
    return null;
  }

  async all<T>(sql: string, values: unknown[]): Promise<{ results: T[] }> {
    if (sql.includes('FROM persona_3c_records WHERE run_id=')) {
      const rows = this.records.filter((r) => r.run_id === values[0]);
      const counts = new Map<string, number>();
      for (const row of rows) counts.set(row.stage, (counts.get(row.stage) ?? 0) + 1);
      return { results: [...counts.entries()].map(([stage, n]) => ({ stage, n })) as unknown as T[] };
    }
    if (sql.includes('FROM persona_3c_progress WHERE run_id=')) {
      const rows = this.progress.filter((r) => r.run_id === values[0]).sort((a, b) => a.stage.localeCompare(b.stage));
      return { results: rows.map((r) => ({ stage: r.stage, done: r.done, total: r.total, updatedAt: r.updated_at })) as unknown as T[] };
    }
    if (sql.includes('FROM persona_3c_events WHERE run_id=')) {
      const rows = this.events.filter((r) => r.run_id === values[0]).sort((a, b) => b.id - a.id).slice(0, 50);
      return { results: rows.map((r) => ({ id: r.id, at: r.at, stage: r.stage, kind: r.kind, title: r.title, summary: r.summary })) as unknown as T[] };
    }
    return { results: [] };
  }

  async run(sql: string, values: unknown[]) {
    if (sql.includes('INSERT INTO persona_3c_runs')) {
      const idempotencyKey = values[13] as string;
      if (this.runs.some((r) => r.idempotency_key === idempotencyKey)) throw new Error('UNIQUE constraint failed: persona_3c_runs.idempotency_key');
      const [id, study_id, status, stage, manifest_hash, budget_usd, spent_usd, gpu_hours, created_at, updated_at, completed_at, error_message, last_record_id] = values;
      this.runs.push({ id, study_id, status, stage, manifest_hash, budget_usd, spent_usd, gpu_hours, created_at, updated_at, completed_at, error_message, last_record_id, idempotency_key: idempotencyKey } as RunRow);
      return { meta: { changes: 1 } };
    }
    if (sql.includes('INSERT OR IGNORE INTO persona_3c_records')) {
      const [run_id, stage, record_id, payload_json, sha256, created_at] = values as [string, string, string, string, string, string];
      if (this.records.some((r) => r.run_id === run_id && r.stage === stage && r.record_id === record_id)) return { meta: { changes: 0 } };
      this.records.push({ run_id, stage, record_id, payload_json, sha256, created_at });
      return { meta: { changes: 1 } };
    }
    if (sql.includes('INSERT INTO persona_3c_progress')) {
      const [run_id, stage, done, total, updated_at] = values as [string, string, number, number, string];
      const existing = this.progress.find((r) => r.run_id === run_id && r.stage === stage);
      if (existing) {
        existing.done = done;
        existing.total = total;
        existing.updated_at = updated_at;
      } else {
        this.progress.push({ run_id, stage, done, total, updated_at });
      }
      return { meta: { changes: 1 } };
    }
    if (sql.includes("UPDATE persona_3c_runs SET stage=?,status=?")) {
      const [stage, status, spent_usd, gpu_hours, last_record_id, updated_at, completed_at, error_message, id] = values;
      const run = this.runs.find((r) => r.id === id && !['complete', 'failed', 'stopped'].includes(String(r.status)));
      if (!run) return { meta: { changes: 0 } };
      run.stage = stage; run.status = status; run.spent_usd = spent_usd; run.gpu_hours = gpu_hours;
      run.last_record_id = last_record_id; run.updated_at = updated_at; run.completed_at = completed_at; run.error_message = error_message;
      return { meta: { changes: 1 } };
    }
    if (sql.includes("UPDATE persona_3c_runs SET status='stopped'")) {
      const [completed_at, updated_at, id] = values;
      const run = this.runs.find((r) => r.id === id && !['complete', 'failed', 'stopped'].includes(String(r.status)));
      if (!run) return { meta: { changes: 0 } };
      run.status = 'stopped'; run.completed_at = completed_at; run.updated_at = updated_at;
      return { meta: { changes: 1 } };
    }
    if (sql.includes('INSERT INTO persona_3c_events')) {
      const [run_id, at, stage, kind, title, summary, payload_json] = values as [string, string, string | null, string, string, string, string];
      this.events.push({ id: this.nextEventId++, run_id, at, stage, kind, title, summary, payload_json });
      return { meta: { changes: 1 } };
    }
    return { meta: { changes: 1 } };
  }
}

function env(db: FakeD1, token: string = TOKEN) {
  return { DB: db, PERSONA_3C_TOKEN: token } as unknown as Env;
}

describe('Experiment 3C live progress reporting', () => {
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

  it('rejects start without a token configured', async () => {
    const db = new FakeD1();
    const response = await startPersona3C(request('/api/persona-3c/start', { studyId: 'experiment-003c-v1', manifestHash: MANIFEST_HASH, budgetUsd: 5, idempotencyKey: 'start-key-0000001' }, ''), env(db, ''), {});
    expect(response.status).toBe(503);
  });

  it('rejects an unauthorized start', async () => {
    const db = new FakeD1();
    const response = await startPersona3C(request('/api/persona-3c/start', { studyId: 'experiment-003c-v1', manifestHash: MANIFEST_HASH, budgetUsd: 5, idempotencyKey: 'start-key-0000001' }, 'wrong-token'), env(db), {});
    expect(response.status).toBe(401);
  });

  it('is idempotent on the idempotency key', async () => {
    const db = new FakeD1();
    const startBody = { studyId: 'experiment-003c-v1', manifestHash: MANIFEST_HASH, budgetUsd: 5, idempotencyKey: 'start-key-0000001' };
    const first = await startPersona3C(request('/api/persona-3c/start', startBody), env(db), {});
    expect(first.status).toBe(202);
    const firstPayload = await responseBody(first);
    expect(firstPayload.idempotent).toBe(false);
    const runId = (firstPayload.run as Record<string, unknown>).id as string;

    const second = await startPersona3C(request('/api/persona-3c/start', startBody), env(db), {});
    expect(second.status).toBe(200);
    const secondPayload = await responseBody(second);
    expect(secondPayload.idempotent).toBe(true);
    expect((secondPayload.run as Record<string, unknown>).id).toBe(runId);
    expect(db.runs).toHaveLength(1);
  });

  it('rejects a report whose record sha256 does not match its canonical payload', async () => {
    const db = new FakeD1();
    const started = await startPersona3C(request('/api/persona-3c/start', { studyId: 'experiment-003c-v1', manifestHash: MANIFEST_HASH, budgetUsd: 5, idempotencyKey: 'start-key-0000002' }), env(db), {});
    const runId = ((await responseBody(started)).run as Record<string, unknown>).id as string;

    const response = await reportPersona3C(request('/api/persona-3c/report', {
      runId, stage: 'harvest', progress: { done: 1, total: 10 },
      records: [{ recordId: 'r1', payload: { a: 1 }, sha256: 'f'.repeat(64) }],
    }), env(db), {});
    expect(response.status).toBe(400);
    expect(db.records).toHaveLength(0);
  });

  it('accepts a correctly hashed record, updates progress, and counts duplicates on resubmission', async () => {
    const db = new FakeD1();
    const started = await startPersona3C(request('/api/persona-3c/start', { studyId: 'experiment-003c-v1', manifestHash: MANIFEST_HASH, budgetUsd: 5, idempotencyKey: 'start-key-0000003' }), env(db), {});
    const runId = ((await responseBody(started)).run as Record<string, unknown>).id as string;

    const payload = { text: 'hello world', score: 3 };
    const sha256 = shaOf(payload);
    const reportBody = {
      runId, stage: 'harvest', status: 'running', progress: { done: 1, total: 10 },
      records: [{ recordId: 'r1', payload, sha256 }],
      gpuHours: 0.5, spentUsd: 0.1,
    };

    const first = await reportPersona3C(request('/api/persona-3c/report', reportBody), env(db), {});
    expect(first.status).toBe(200);
    const firstPayload = await responseBody(first);
    expect(firstPayload).toMatchObject({ ok: true, runId, stage: 'harvest', accepted: 1, duplicates: 0 });
    expect(db.progress.find((p) => p.run_id === runId && p.stage === 'harvest')).toMatchObject({ done: 1, total: 10 });
    expect(db.runs.find((r) => r.id === runId)).toMatchObject({ status: 'running', stage: 'harvest', gpu_hours: 0.5, spent_usd: 0.1, last_record_id: 'r1' });

    const second = await reportPersona3C(request('/api/persona-3c/report', { ...reportBody, progress: { done: 1, total: 10 } }), env(db), {});
    const secondPayload = await responseBody(second);
    expect(secondPayload).toMatchObject({ ok: true, accepted: 0, duplicates: 1 });
  });

  it('accepts payloadJson whose sha256 matches the exact string bytes, including exponent-notation floats', async () => {
    const db = new FakeD1();
    const started = await startPersona3C(request('/api/persona-3c/start', { studyId: 'experiment-003c-v1', manifestHash: MANIFEST_HASH, budgetUsd: 5, idempotencyKey: 'start-key-0000009' }), env(db), {});
    const runId = ((await responseBody(started)).run as Record<string, unknown>).id as string;

    // These float strings (1e-05, 0.435447) are exactly what a Python json.dumps caller would
    // send; the point of payloadJson is that the Worker never re-serializes and re-diffs them.
    const payloadJson = '{"dead_frac":1e-05,"fve":0.435447,"loss":1.251517}';
    const sha256 = createHash('sha256').update(payloadJson).digest('hex');

    const response = await reportPersona3C(request('/api/persona-3c/report', {
      runId, stage: 'train', progress: { done: 1, total: 10 },
      records: [{ recordId: 'train-checkpoint-1004535', payloadJson, sha256 }],
    }), env(db), {});
    expect(response.status).toBe(200);
    const payload = await responseBody(response);
    expect(payload).toMatchObject({ ok: true, accepted: 1, duplicates: 0 });
    expect(db.records[0]).toMatchObject({ record_id: 'train-checkpoint-1004535', payload_json: payloadJson, sha256 });
  });

  it('rejects payloadJson whose sha256 does not match the string bytes', async () => {
    const db = new FakeD1();
    const started = await startPersona3C(request('/api/persona-3c/start', { studyId: 'experiment-003c-v1', manifestHash: MANIFEST_HASH, budgetUsd: 5, idempotencyKey: 'start-key-0000010' }), env(db), {});
    const runId = ((await responseBody(started)).run as Record<string, unknown>).id as string;

    const response = await reportPersona3C(request('/api/persona-3c/report', {
      runId, stage: 'train', progress: { done: 1, total: 10 },
      records: [{ recordId: 'r1', payloadJson: '{"a":1}', sha256: 'f'.repeat(64) }],
    }), env(db), {});
    expect(response.status).toBe(400);
    expect(db.records).toHaveLength(0);
  });

  it('rejects payloadJson that does not parse to a plain object', async () => {
    const db = new FakeD1();
    const started = await startPersona3C(request('/api/persona-3c/start', { studyId: 'experiment-003c-v1', manifestHash: MANIFEST_HASH, budgetUsd: 5, idempotencyKey: 'start-key-0000011' }), env(db), {});
    const runId = ((await responseBody(started)).run as Record<string, unknown>).id as string;

    const payloadJson = '[1,2,3]';
    const sha256 = createHash('sha256').update(payloadJson).digest('hex');
    const response = await reportPersona3C(request('/api/persona-3c/report', {
      runId, stage: 'train', progress: { done: 1, total: 10 },
      records: [{ recordId: 'r1', payloadJson, sha256 }],
    }), env(db), {});
    expect(response.status).toBe(400);
    expect(db.records).toHaveLength(0);
  });

  it('rejects a record with neither payload nor payloadJson', async () => {
    const db = new FakeD1();
    const started = await startPersona3C(request('/api/persona-3c/start', { studyId: 'experiment-003c-v1', manifestHash: MANIFEST_HASH, budgetUsd: 5, idempotencyKey: 'start-key-0000012' }), env(db), {});
    const runId = ((await responseBody(started)).run as Record<string, unknown>).id as string;

    const response = await reportPersona3C(request('/api/persona-3c/report', {
      runId, stage: 'train', progress: { done: 1, total: 10 },
      records: [{ recordId: 'r1', sha256: 'f'.repeat(64) }],
    }), env(db), {});
    expect(response.status).toBe(400);
    expect(db.records).toHaveLength(0);
  });

  it('accepts done>total progress counters and stores them clamped to total', async () => {
    const db = new FakeD1();
    const started = await startPersona3C(request('/api/persona-3c/start', { studyId: 'experiment-003c-v1', manifestHash: MANIFEST_HASH, budgetUsd: 5, idempotencyKey: 'start-key-0000007' }), env(db), {});
    const runId = ((await responseBody(started)).run as Record<string, unknown>).id as string;

    const response = await reportPersona3C(request('/api/persona-3c/report', {
      runId, stage: 'harvest', progress: { done: 101, total: 100 },
    }), env(db), {});
    expect(response.status).toBe(200);
    expect(db.progress.find((p) => p.run_id === runId && p.stage === 'harvest')).toMatchObject({ done: 100, total: 100 });
  });

  it('accepts a status-only report with no progress and marks the run failed', async () => {
    const db = new FakeD1();
    const started = await startPersona3C(request('/api/persona-3c/start', { studyId: 'experiment-003c-v1', manifestHash: MANIFEST_HASH, budgetUsd: 5, idempotencyKey: 'start-key-0000008' }), env(db), {});
    const runId = ((await responseBody(started)).run as Record<string, unknown>).id as string;

    const response = await reportPersona3C(request('/api/persona-3c/report', {
      runId, stage: 'harvest', status: 'failed', message: 'GPU pod crashed before reporting progress.',
    }), env(db), {});
    expect(response.status).toBe(200);
    expect(db.progress.find((p) => p.run_id === runId && p.stage === 'harvest')).toBeUndefined();
    expect(db.runs.find((r) => r.id === runId)).toMatchObject({ status: 'failed', stage: 'harvest' });
  });

  it('hides payloads and tokens from the public status view', async () => {
    const db = new FakeD1();
    const started = await startPersona3C(request('/api/persona-3c/start', { studyId: 'experiment-003c-v1', manifestHash: MANIFEST_HASH, budgetUsd: 5, idempotencyKey: 'start-key-0000004' }), env(db), {});
    const runId = ((await responseBody(started)).run as Record<string, unknown>).id as string;
    const payload = { secret: 'do-not-leak' };
    await reportPersona3C(request('/api/persona-3c/report', {
      runId, stage: 'harvest', progress: { done: 1, total: 1 },
      records: [{ recordId: 'r1', payload, sha256: shaOf(payload) }],
    }), env(db), {});

    const status = await persona3CStatus(env(db), {}, null);
    const statusPayload = JSON.stringify(await responseBody(status));
    expect(statusPayload).not.toContain('do-not-leak');
    expect(statusPayload).not.toContain(TOKEN);
    expect(statusPayload).toContain(runId);
  });

  it('returns {run:null} when no run exists', async () => {
    const db = new FakeD1();
    const status = await persona3CStatus(env(db), {}, null);
    expect(await responseBody(status)).toEqual({ run: null });
  });

  it('rejects a report for a run that is already complete, failed, or stopped', async () => {
    const db = new FakeD1();
    const started = await startPersona3C(request('/api/persona-3c/start', { studyId: 'experiment-003c-v1', manifestHash: MANIFEST_HASH, budgetUsd: 5, idempotencyKey: 'start-key-0000005' }), env(db), {});
    const runId = ((await responseBody(started)).run as Record<string, unknown>).id as string;

    const complete = await reportPersona3C(request('/api/persona-3c/report', { runId, stage: 'done', status: 'complete', progress: { done: 1, total: 1 } }), env(db), {});
    expect(complete.status).toBe(200);

    const after = await reportPersona3C(request('/api/persona-3c/report', { runId, stage: 'done', progress: { done: 1, total: 1 } }), env(db), {});
    expect(after.status).toBe(409);
  });

  it('stops a run via the stop endpoint', async () => {
    const db = new FakeD1();
    const started = await startPersona3C(request('/api/persona-3c/start', { studyId: 'experiment-003c-v1', manifestHash: MANIFEST_HASH, budgetUsd: 5, idempotencyKey: 'start-key-0000006' }), env(db), {});
    const runId = ((await responseBody(started)).run as Record<string, unknown>).id as string;
    const stopped = await stopPersona3C(request('/api/persona-3c/stop', { runId }), env(db), {});
    expect(stopped.status).toBe(200);
    expect(((await responseBody(stopped)).run as Record<string, unknown>).status).toBe('stopped');

    const again = await stopPersona3C(request('/api/persona-3c/stop', { runId }), env(db), {});
    expect(again.status).toBe(409);
  });
});
