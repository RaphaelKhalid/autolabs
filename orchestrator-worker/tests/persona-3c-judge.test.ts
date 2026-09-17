import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest';
import {
  persona3CJudgeResults,
  persona3CStatus,
  planPersona3CJudge,
  runPersona3CJudge,
  startPersona3C,
} from '../src/persona-3c';

const TOKEN = 'persona-3c-judge-test-token';
const MANIFEST_HASH = 'a'.repeat(64);

function request(path: string, body: Record<string, unknown> | null, token = TOKEN, method: 'POST' | 'GET' = 'POST') {
  return new Request('https://worker.test' + path, {
    method,
    headers: { authorization: 'Bearer ' + token, 'content-type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
  });
}

async function responseBody(response: Response) {
  return response.json() as Promise<Record<string, unknown>>;
}

type RunRow = Record<string, unknown> & {
  id: string;
  idempotency_key: string;
  status: string;
  judge_budget_usd: number;
  judge_spent_usd: number;
  judge_reserved_usd: number;
  judge_calls: number;
  judge_call_ceiling: number;
};
type JudgeRow = {
  run_id: string;
  job_id: string;
  direction_key: string;
  scenario: string;
  order_swap: number;
  prompt_sha256: string | null;
  response_json: string | null;
  provider_response_id: string | null;
  input_tokens: number;
  cached_input_tokens: number;
  output_tokens: number;
  cost_usd: number;
  status: string;
  attempts: number;
  created_at: string;
  updated_at: string;
};
type JudgeInputRow = { run_id: string; job_id: string; text_a: string; text_b: string };

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
  judge: JudgeRow[] = [];
  judgeInputs: JudgeInputRow[] = [];
  events: { id: number; run_id: string; stage: string | null; kind: string; title: string; summary: string }[] = [];
  nextEventId = 1;
  nextCreated = 0;

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
    if (sql.includes('SELECT judge_spent_usd FROM persona_3c_runs WHERE id=')) {
      const run = this.runs.find((r) => r.id === values[0]);
      return (run ? { judge_spent_usd: run.judge_spent_usd } : null) as unknown as T;
    }
    if (sql.includes('FROM persona_3c_runs WHERE id=')) {
      return (this.runs.find((r) => r.id === values[0]) as unknown as T) ?? null;
    }
    if (sql.includes('FROM persona_3c_judge_inputs WHERE run_id=? AND job_id=')) {
      const row = this.judgeInputs.find((r) => r.run_id === values[0] && r.job_id === values[1]);
      return (row ? { textA: row.text_a, textB: row.text_b } : null) as unknown as T;
    }
    if (sql.includes("SELECT COUNT(*) AS n FROM persona_3c_judge WHERE run_id=? AND status='queued'")) {
      const n = this.judge.filter((j) => j.run_id === values[0] && j.status === 'queued').length;
      return { n } as unknown as T;
    }
    return null;
  }

  async all<T>(sql: string, values: unknown[]): Promise<{ results: T[] }> {
    if (sql.includes('FROM persona_3c_records WHERE run_id=')) {
      return { results: [] as unknown as T[] };
    }
    if (sql.includes('FROM persona_3c_progress WHERE run_id=')) {
      return { results: [] as unknown as T[] };
    }
    if (sql.includes('FROM persona_3c_events WHERE run_id=')) {
      const rows = this.events.filter((r) => r.run_id === values[0]).sort((a, b) => b.id - a.id).slice(0, 50);
      return { results: rows.map((r) => ({ id: r.id, at: '', stage: r.stage, kind: r.kind, title: r.title, summary: r.summary })) as unknown as T[] };
    }
    if (sql.includes("SELECT * FROM persona_3c_judge WHERE run_id=? AND status='queued' ORDER BY created_at")) {
      const rows = this.judge.filter((j) => j.run_id === values[0] && j.status === 'queued').sort((a, b) => a.created_at.localeCompare(b.created_at));
      const limit = Number(values[1]);
      return { results: rows.slice(0, limit) as unknown as T[] };
    }
    if (sql.includes('SELECT status,COUNT(*) AS n FROM persona_3c_judge WHERE run_id=? GROUP BY status')) {
      const rows = this.judge.filter((j) => j.run_id === values[0]);
      const counts = new Map<string, number>();
      for (const row of rows) counts.set(row.status, (counts.get(row.status) ?? 0) + 1);
      return { results: [...counts.entries()].map(([status, n]) => ({ status, n })) as unknown as T[] };
    }
    if (sql.includes("FROM persona_3c_judge WHERE run_id=? AND status='complete' ORDER BY created_at")) {
      const rows = this.judge.filter((j) => j.run_id === values[0] && j.status === 'complete').sort((a, b) => a.created_at.localeCompare(b.created_at));
      return {
        results: rows.map((r) => ({
          jobId: r.job_id, directionKey: r.direction_key, scenario: r.scenario, orderSwap: r.order_swap,
          responseJson: r.response_json, inputTokens: r.input_tokens, cachedInputTokens: r.cached_input_tokens,
          outputTokens: r.output_tokens, costUsd: r.cost_usd,
        })) as unknown as T[],
      };
    }
    return { results: [] };
  }

  async run(sql: string, values: unknown[]) {
    if (sql.includes('INSERT INTO persona_3c_runs')) {
      const idempotencyKey = values[13] as string;
      if (this.runs.some((r) => r.idempotency_key === idempotencyKey)) throw new Error('UNIQUE constraint failed: persona_3c_runs.idempotency_key');
      const [id, study_id, status, stage, manifest_hash, budget_usd, spent_usd, gpu_hours, created_at, updated_at, completed_at, error_message, last_record_id] = values;
      this.runs.push({
        id, study_id, status, stage, manifest_hash, budget_usd, spent_usd, gpu_hours, created_at, updated_at,
        completed_at, error_message, last_record_id, idempotency_key: idempotencyKey,
        judge_budget_usd: 0, judge_spent_usd: 0, judge_reserved_usd: 0, judge_calls: 0, judge_call_ceiling: 0,
      } as RunRow);
      return { meta: { changes: 1 } };
    }
    if (sql.includes('INSERT INTO persona_3c_events')) {
      const [run_id, , stage, kind, title, summary] = values as [string, string, string | null, string, string, string];
      this.events.push({ id: this.nextEventId++, run_id, stage, kind, title, summary });
      return { meta: { changes: 1 } };
    }
    if (sql.includes('INSERT OR IGNORE INTO persona_3c_judge(')) {
      const [run_id, job_id, direction_key, scenario, order_swap, status, attempts, created_at, updated_at] = values as [string, string, string, string, number, string, number, string, string];
      if (this.judge.some((j) => j.run_id === run_id && j.job_id === job_id)) return { meta: { changes: 0 } };
      this.judge.push({
        run_id, job_id, direction_key, scenario, order_swap, status: status as JudgeRow['status'], attempts,
        created_at: `${created_at}-${this.nextCreated++}`, updated_at,
        prompt_sha256: null, response_json: null, provider_response_id: null,
        input_tokens: 0, cached_input_tokens: 0, output_tokens: 0, cost_usd: 0,
      });
      return { meta: { changes: 1 } };
    }
    if (sql.includes('INSERT OR IGNORE INTO persona_3c_judge_inputs')) {
      const [run_id, job_id, text_a, text_b] = values as [string, string, string, string];
      if (this.judgeInputs.some((j) => j.run_id === run_id && j.job_id === job_id)) return { meta: { changes: 0 } };
      this.judgeInputs.push({ run_id, job_id, text_a, text_b });
      return { meta: { changes: 1 } };
    }
    if (sql.includes('UPDATE persona_3c_runs SET judge_budget_usd=?,judge_call_ceiling=?')) {
      const [judge_budget_usd, judge_call_ceiling, updated_at, id] = values as [number, number, string, string];
      const run = this.runs.find((r) => r.id === id);
      if (!run) return { meta: { changes: 0 } };
      run.judge_budget_usd = judge_budget_usd; run.judge_call_ceiling = judge_call_ceiling; run.updated_at = updated_at;
      return { meta: { changes: 1 } };
    }
    if (sql.includes("UPDATE persona_3c_judge SET status='queued',updated_at=? WHERE run_id=? AND status='in_flight'")) {
      const [updated_at, run_id, staleBefore] = values as [string, string, string];
      let changes = 0;
      for (const job of this.judge) {
        if (job.run_id === run_id && job.status === 'in_flight' && job.updated_at <= staleBefore) {
          job.status = 'queued'; job.updated_at = updated_at; changes += 1;
        }
      }
      return { meta: { changes } };
    }
    if (sql.includes("UPDATE persona_3c_judge SET status='in_flight',updated_at=? WHERE run_id=? AND job_id=? AND status='queued'")) {
      const [updated_at, run_id, job_id] = values as [string, string, string];
      const job = this.judge.find((j) => j.run_id === run_id && j.job_id === job_id && j.status === 'queued');
      if (!job) return { meta: { changes: 0 } };
      job.status = 'in_flight'; job.updated_at = updated_at;
      return { meta: { changes: 1 } };
    }
    if (sql.includes("UPDATE persona_3c_judge SET status='queued',updated_at=? WHERE run_id=? AND job_id=?")) {
      const [updated_at, run_id, job_id] = values as [string, string, string];
      const job = this.judge.find((j) => j.run_id === run_id && j.job_id === job_id);
      if (!job) return { meta: { changes: 0 } };
      job.status = 'queued'; job.updated_at = updated_at;
      return { meta: { changes: 1 } };
    }
    if (sql.includes("UPDATE persona_3c_judge SET status='failed',updated_at=? WHERE run_id=? AND job_id=?")) {
      const [updated_at, run_id, job_id] = values as [string, string, string];
      const job = this.judge.find((j) => j.run_id === run_id && j.job_id === job_id);
      if (!job) return { meta: { changes: 0 } };
      job.status = 'failed'; job.updated_at = updated_at;
      return { meta: { changes: 1 } };
    }
    if (sql.includes('UPDATE persona_3c_runs SET judge_calls=judge_calls+1,judge_reserved_usd=judge_reserved_usd+?')) {
      const [cap, updated_at, id, cap2] = values as [number, string, string, number];
      const run = this.runs.find((r) => r.id === id);
      if (!run) return { meta: { changes: 0 } };
      if (!(run.judge_calls < run.judge_call_ceiling && run.judge_spent_usd + run.judge_reserved_usd + cap2 <= run.judge_budget_usd)) {
        return { meta: { changes: 0 } };
      }
      run.judge_calls += 1; run.judge_reserved_usd += cap; run.updated_at = updated_at;
      return { meta: { changes: 1 } };
    }
    if (sql.includes("UPDATE persona_3c_judge SET status='complete',prompt_sha256=?")) {
      const [prompt_sha256, response_json, provider_response_id, input_tokens, cached_input_tokens, output_tokens, cost_usd, attempts, updated_at, run_id, job_id] = values as [
        string, string, string | null, number, number, number, number, number, string, string, string,
      ];
      const job = this.judge.find((j) => j.run_id === run_id && j.job_id === job_id);
      if (!job) return { meta: { changes: 0 } };
      job.status = 'complete'; job.prompt_sha256 = prompt_sha256; job.response_json = response_json;
      job.provider_response_id = provider_response_id; job.input_tokens = input_tokens; job.cached_input_tokens = cached_input_tokens;
      job.output_tokens = output_tokens; job.cost_usd = cost_usd; job.attempts = attempts; job.updated_at = updated_at;
      return { meta: { changes: 1 } };
    }
    if (sql.includes('UPDATE persona_3c_judge SET status=?,attempts=?,updated_at=? WHERE run_id=? AND job_id=?')) {
      const [status, attempts, updated_at, run_id, job_id] = values as [string, number, string, string, string];
      const job = this.judge.find((j) => j.run_id === run_id && j.job_id === job_id);
      if (!job) return { meta: { changes: 0 } };
      job.status = status as JudgeRow['status']; job.attempts = attempts; job.updated_at = updated_at;
      return { meta: { changes: 1 } };
    }
    if (sql.includes('UPDATE persona_3c_runs SET judge_reserved_usd=judge_reserved_usd-?,judge_spent_usd=judge_spent_usd+?')) {
      const [cap, actual, updated_at, id, capCheck] = values as [number, number, string, string, number];
      const run = this.runs.find((r) => r.id === id);
      if (!run) return { meta: { changes: 0 } };
      if (run.judge_reserved_usd < capCheck) return { meta: { changes: 0 } };
      run.judge_reserved_usd -= cap; run.judge_spent_usd += actual; run.updated_at = updated_at;
      return { meta: { changes: 1 } };
    }
    return { meta: { changes: 1 } };
  }
}

function env(db: FakeD1, extra: Record<string, unknown> = {}) {
  return { DB: db, PERSONA_3C_TOKEN: TOKEN, OPENAI_API_KEY: 'mock-provider-key', ...extra } as unknown as Env;
}

async function startRun(db: FakeD1, key: string) {
  const started = await startPersona3C(request('/api/persona-3c/start', { studyId: 'experiment-003c-v1', manifestHash: MANIFEST_HASH, budgetUsd: 5, idempotencyKey: key }), env(db), {});
  const runId = ((await responseBody(started)).run as Record<string, unknown>).id as string;
  const run = db.runs.find((r) => r.id === runId)!;
  run.status = 'running'; // judge routes require running/complete; the report route normally does this transition
  return runId;
}

function judgePlanBody(runId: string, overrides: Record<string, unknown> = {}) {
  return {
    runId,
    judgeBudgetUsd: 3,
    judgeCallCeiling: 200,
    pairs: [{ directionKey: 'feature-75-neg', scenario: 'scenario-1', textA: 'baseline text', textB: 'steered text', orderSwap: false }],
    ...overrides,
  };
}

function fetchOk(judgeResponse: Record<string, unknown>, usage?: Record<string, unknown>) {
  return vi.fn(async () => ({
    ok: true,
    status: 200,
    json: async () => ({
      id: 'resp_test_1',
      output: [{ content: [{ type: 'output_text', text: JSON.stringify(judgeResponse) }] }],
      usage: usage ?? { input_tokens: 500, output_tokens: 40, input_tokens_details: { cached_tokens: 0 } },
    }),
  }));
}

function fetchFailing() {
  return vi.fn(async () => ({ ok: false, status: 500, json: async () => ({ error: { type: 'server_error' } }) }));
}

describe('Experiment 3C judge (describe stage, judge pass one)', () => {
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

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('plan is idempotent: replanning the same pairs reports duplicates, not new jobs', async () => {
    const db = new FakeD1();
    const runId = await startRun(db, 'judge-key-0000001');

    const first = await planPersona3CJudge(request('/api/persona-3c/judge/plan', judgePlanBody(runId)), env(db), {});
    expect(first.status).toBe(200);
    expect(await responseBody(first)).toMatchObject({ queued: 1, duplicate: 0, judgeBudgetUsd: 3, judgeCallCeiling: 200 });
    expect(db.judge).toHaveLength(1);
    expect(db.judgeInputs).toHaveLength(1);

    const second = await planPersona3CJudge(request('/api/persona-3c/judge/plan', judgePlanBody(runId)), env(db), {});
    expect(second.status).toBe(200);
    expect(await responseBody(second)).toMatchObject({ queued: 0, duplicate: 1 });
    expect(db.judge).toHaveLength(1);
  });

  it('plan only raises the judge budget and ceiling, never lowers them', async () => {
    const db = new FakeD1();
    const runId = await startRun(db, 'judge-key-0000002');
    await planPersona3CJudge(request('/api/persona-3c/judge/plan', judgePlanBody(runId, { judgeBudgetUsd: 10, judgeCallCeiling: 1000, pairs: judgePlanBody(runId).pairs })), env(db), {});
    const lower = await planPersona3CJudge(request('/api/persona-3c/judge/plan', judgePlanBody(runId, {
      judgeBudgetUsd: 1, judgeCallCeiling: 5,
      pairs: [{ directionKey: 'feature-1019-pos', scenario: 'scenario-2', textA: 'a', textB: 'b', orderSwap: true }],
    })), env(db), {});
    expect((await responseBody(lower))).toMatchObject({ judgeBudgetUsd: 10, judgeCallCeiling: 1000 });
  });

  it('refuses a plan when the run is not running or complete', async () => {
    const db = new FakeD1();
    const started = await startPersona3C(request('/api/persona-3c/start', { studyId: 'experiment-003c-v1', manifestHash: MANIFEST_HASH, budgetUsd: 5, idempotencyKey: 'judge-key-0000003' }), env(db), {});
    const runId = ((await responseBody(started)).run as Record<string, unknown>).id as string; // still "queued"
    const response = await planPersona3CJudge(request('/api/persona-3c/judge/plan', judgePlanBody(runId)), env(db), {});
    expect(response.status).toBe(409);
  });

  it('refuses to run the judge when the call ceiling is already reached', async () => {
    const db = new FakeD1();
    const runId = await startRun(db, 'judge-key-0000004');
    await planPersona3CJudge(request('/api/persona-3c/judge/plan', judgePlanBody(runId)), env(db), {});
    const run = db.runs.find((r) => r.id === runId)!;
    run.judge_call_ceiling = 0; // simulate the ceiling already having been reached

    vi.stubGlobal('fetch', fetchOk({ property: 'B sounds warmer.', more_in: 'B', about: 'speaker', confidence: 'medium' }));
    const response = await runPersona3CJudge(request('/api/persona-3c/judge/run', { runId, maxJobs: 5 }), env(db), {});
    expect(response.status).toBe(200);
    const payload = await responseBody(response);
    expect(payload).toMatchObject({ ran: 0, complete: 0, failed: 0, remainingQueued: 1 });
    expect(db.judge[0].status).toBe('queued');
  });

  it('refuses to run the judge when spent+reserved+worst-case would exceed the budget', async () => {
    const db = new FakeD1();
    const runId = await startRun(db, 'judge-key-0000005');
    await planPersona3CJudge(request('/api/persona-3c/judge/plan', judgePlanBody(runId)), env(db), {});
    const run = db.runs.find((r) => r.id === runId)!;
    run.judge_budget_usd = 0.0000001; // any worst-case reservation exceeds this

    vi.stubGlobal('fetch', fetchOk({ property: 'B sounds warmer.', more_in: 'B', about: 'speaker', confidence: 'medium' }));
    const response = await runPersona3CJudge(request('/api/persona-3c/judge/run', { runId, maxJobs: 5 }), env(db), {});
    const payload = await responseBody(response);
    expect(payload).toMatchObject({ ran: 0, complete: 0, failed: 0, remainingQueued: 1 });
    expect(db.judge[0].status).toBe('queued');
  });

  it('a successful run settles cost and stores the parsed response', async () => {
    const db = new FakeD1();
    const runId = await startRun(db, 'judge-key-0000006');
    await planPersona3CJudge(request('/api/persona-3c/judge/plan', judgePlanBody(runId)), env(db), {});

    vi.stubGlobal('fetch', fetchOk({ property: 'B is more formal than A.', more_in: 'B', about: 'speaker', confidence: 'high' }, { input_tokens: 600, output_tokens: 30, input_tokens_details: { cached_tokens: 0 } }));
    const response = await runPersona3CJudge(request('/api/persona-3c/judge/run', { runId, maxJobs: 5 }), env(db), {});
    expect(response.status).toBe(200);
    const payload = await responseBody(response);
    expect(payload).toMatchObject({ ran: 1, complete: 1, failed: 0, remainingQueued: 0 });
    expect(Number(payload.judgeSpentUsd)).toBeGreaterThan(0);

    expect(db.judge[0].status).toBe('complete');
    expect(JSON.parse(db.judge[0].response_json!)).toMatchObject({ property: 'B is more formal than A.', more_in: 'B', about: 'speaker', confidence: 'high' });
    expect(db.judge[0].prompt_sha256).toMatch(/^[0-9a-f]{64}$/);

    const run = db.runs.find((r) => r.id === runId)!;
    expect(run.judge_reserved_usd).toBeCloseTo(0, 10);
    expect(run.judge_spent_usd).toBeGreaterThan(0);
    expect(run.judge_calls).toBe(1);

    // The results route never returns the blinded texts.
    const results = await persona3CJudgeResults(request('/api/persona-3c/judge/results', null, TOKEN, 'GET'), env(db), {}, runId);
    const resultsPayload = await responseBody(results);
    const serialized = JSON.stringify(resultsPayload);
    expect(serialized).not.toContain('baseline text');
    expect(serialized).not.toContain('steered text');
    expect((resultsPayload.results as unknown[])).toHaveLength(1);
    expect((resultsPayload.results as Record<string, unknown>[])[0]).toMatchObject({
      directionKey: 'feature-75-neg', scenario: 'scenario-1', orderSwap: false,
      response: { property: 'B is more formal than A.', more_in: 'B', about: 'speaker', confidence: 'high' },
    });
  });

  it('a failed provider call charges worst-case and requeues until 3 attempts, then marks failed', async () => {
    const db = new FakeD1();
    const runId = await startRun(db, 'judge-key-0000007');
    await planPersona3CJudge(request('/api/persona-3c/judge/plan', judgePlanBody(runId)), env(db), {});

    vi.stubGlobal('fetch', fetchFailing());

    for (let attempt = 1; attempt <= 3; attempt += 1) {
      const response = await runPersona3CJudge(request('/api/persona-3c/judge/run', { runId, maxJobs: 5 }), env(db), {});
      const payload = await responseBody(response);
      expect(payload).toMatchObject({ ran: 1, complete: 0, failed: 1 });
      expect(db.judge[0].attempts).toBe(attempt);
      if (attempt < 3) {
        expect(db.judge[0].status).toBe('queued');
        expect(payload.remainingQueued).toBe(1);
      } else {
        expect(db.judge[0].status).toBe('failed');
        expect(payload.remainingQueued).toBe(0);
      }
    }

    const run = db.runs.find((r) => r.id === runId)!;
    expect(run.judge_calls).toBe(3);
    expect(run.judge_reserved_usd).toBeCloseTo(0, 10);
    expect(run.judge_spent_usd).toBeGreaterThan(0);
  });

  it('re-queues stale in_flight jobs older than 10 minutes at the start of a run', async () => {
    const db = new FakeD1();
    const runId = await startRun(db, 'judge-key-0000008');
    await planPersona3CJudge(request('/api/persona-3c/judge/plan', judgePlanBody(runId)), env(db), {});
    db.judge[0].status = 'in_flight';
    db.judge[0].updated_at = new Date(Date.now() - 11 * 60_000).toISOString();

    vi.stubGlobal('fetch', fetchOk({ property: '', more_in: 'neither', about: 'speaker', confidence: 'low' }));
    const response = await runPersona3CJudge(request('/api/persona-3c/judge/run', { runId, maxJobs: 5 }), env(db), {});
    const payload = await responseBody(response);
    expect(payload).toMatchObject({ ran: 1, complete: 1, failed: 0, remainingQueued: 0 });
    expect(db.judge[0].status).toBe('complete');
  });

  it('status exposes judge counters but never the blinded texts', async () => {
    const db = new FakeD1();
    const runId = await startRun(db, 'judge-key-0000009');
    await planPersona3CJudge(request('/api/persona-3c/judge/plan', judgePlanBody(runId)), env(db), {});

    const status = await persona3CStatus(env(db), {}, runId);
    const payload = await responseBody(status);
    expect(payload.judge).toMatchObject({ queued: 1, inFlight: 0, complete: 0, failed: 0, calls: 0, budgetUsd: 3, ceiling: 200 });
    expect(JSON.stringify(payload)).not.toContain('baseline text');
    expect(JSON.stringify(payload)).not.toContain('steered text');
  });

  it('requires auth on the results route', async () => {
    const db = new FakeD1();
    const runId = await startRun(db, 'judge-key-0000010');
    const response = await persona3CJudgeResults(request('/api/persona-3c/judge/results', null, 'wrong-token', 'GET'), env(db), {}, runId);
    expect(response.status).toBe(401);
  });
});
