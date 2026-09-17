import { bearer, secretEquals } from './security';
import { cost as judgeCallCost } from './persona-3b';

// Experiment 3C: a RunPod GPU pod reports coarse stage progress into the ledger so the
// public homepage can show live status. The worker never runs GPU work itself; it only
// records what the pod attests, with the same auth/idempotency/append-only patterns as 3B.

export const PERSONA_3C_STUDY_ID = 'experiment-003c-v1';
export const PERSONA_3C_MAX_BUDGET_USD = 20;
export const PERSONA_3C_MAX_RECORDS = 200;
const MAX_BODY_BYTES = 400_000;

export const PERSONA_3C_STAGES = ['boot', 'harvest', 'train', 'calibrate', 'screen', 'judge', 'analysis', 'done'] as const;
export type Persona3CStage = typeof PERSONA_3C_STAGES[number];
const RUN_STATUSES = ['queued', 'running', 'complete', 'failed', 'stopped'] as const;
export type Persona3CRunStatus = typeof RUN_STATUSES[number];
const TERMINAL_STATUSES: readonly Persona3CRunStatus[] = ['complete', 'failed', 'stopped'];
const REPORTABLE_STATUSES: readonly string[] = ['running', 'complete', 'failed', 'stopped'];

export type Persona3CEnv = Env & { PERSONA_3C_TOKEN?: string };

type Run = {
  id: string;
  study_id: string;
  status: Persona3CRunStatus;
  stage: string;
  manifest_hash: string;
  budget_usd: number;
  spent_usd: number;
  gpu_hours: number;
  created_at: string;
  updated_at: string;
  completed_at: string | null;
  error_message: string | null;
  last_record_id: string | null;
  judge_budget_usd: number;
  judge_spent_usd: number;
  judge_reserved_usd: number;
  judge_calls: number;
  judge_call_ceiling: number;
};

function json(value: unknown, init: ResponseInit = {}, c: Record<string, string> = {}) {
  const headers = new Headers(init.headers);
  headers.set('content-type', 'application/json; charset=utf-8');
  headers.set('cache-control', 'no-store');
  for (const [key, item] of Object.entries(c)) headers.set(key, item);
  return new Response(JSON.stringify(value), { ...init, headers });
}

async function body(req: Request, maxBytes: number = MAX_BODY_BYTES): Promise<Record<string, unknown> | null> {
  const declared = Number(req.headers.get('content-length') ?? 0);
  if (!Number.isFinite(declared) || declared > maxBytes) return null;
  const text = await req.text();
  if (new TextEncoder().encode(text).byteLength > maxBytes || !text.trim()) return null;
  try {
    const value = JSON.parse(text);
    return value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : null;
  } catch {
    return null;
  }
}

async function authorized(req: Request, token: string | undefined) {
  return Boolean(token) && await secretEquals(bearer(req), token as string);
}

function str(value: unknown, min: number, max: number) {
  return typeof value === 'string' && value.trim().length >= min && value.length <= max ? value.trim() : '';
}

function integer(value: unknown, min: number, max: number) {
  return Number.isInteger(value) && Number(value) >= min && Number(value) <= max ? Number(value) : null;
}

function validSha(value: unknown) {
  return typeof value === 'string' && /^[0-9a-f]{64}$/i.test(value);
}

function validRun(value: unknown) {
  return typeof value === 'string' && /^persona-3c-[A-Za-z0-9-]{20,100}$/.test(value);
}

function validStage(value: unknown): value is Persona3CStage {
  return typeof value === 'string' && (PERSONA_3C_STAGES as readonly string[]).includes(value);
}

/** Canonical JSON per research/experiment-003b/HASHING.md: sorted keys, compact separators, UTF-8, non-ASCII unescaped. */
export function canonicalJson(value: unknown): string {
  if (value === null) return 'null';
  if (typeof value === 'number' || typeof value === 'boolean' || typeof value === 'string') return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`;
  if (typeof value === 'object') {
    const record = value as Record<string, unknown>;
    const keys = Object.keys(record).sort();
    return `{${keys.map((key) => `${JSON.stringify(key)}:${canonicalJson(record[key])}`).join(',')}}`;
  }
  throw new Error('Unsupported payload value.');
}

async function sha256Hex(text: string) {
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text));
  return Array.from(new Uint8Array(digest)).map((byte) => byte.toString(16).padStart(2, '0')).join('');
}

async function recordEvent(db: D1Database, runId: string, stage: string | null, kind: string, title: string, summary: string, payload: Record<string, unknown> = {}) {
  await db.prepare('INSERT INTO persona_3c_events(run_id,at,stage,kind,title,summary,payload_json) VALUES(?,?,?,?,?,?,?)')
    .bind(runId, new Date().toISOString(), stage, kind, title, summary.slice(0, 500), JSON.stringify(payload))
    .run();
}

function pubRun(run: Run | null) {
  return run ? {
    id: run.id,
    studyId: run.study_id,
    status: run.status,
    stage: run.stage,
    manifestHash: run.manifest_hash,
    budgetUsd: run.budget_usd,
    spentUsd: run.spent_usd,
    gpuHours: run.gpu_hours,
    createdAt: run.created_at,
    updatedAt: run.updated_at,
    completedAt: run.completed_at,
    error: run.error_message,
    lastRecordId: run.last_record_id,
  } : null;
}

export async function startPersona3C(req: Request, env: Persona3CEnv, c: Record<string, string>) {
  if (!env.PERSONA_3C_TOKEN) return json({ error: 'Experiment 3C reporting is not configured.' }, { status: 503 }, c);
  if (!await authorized(req, env.PERSONA_3C_TOKEN)) return json({ error: 'Unauthorized.' }, { status: 401 }, c);
  const b = await body(req);
  const key = str(b?.idempotencyKey, 16, 120);
  const manifest = str(b?.manifestHash, 64, 64).toLowerCase();
  const budget = typeof b?.budgetUsd === 'number' && Number.isFinite(b.budgetUsd) ? b.budgetUsd : null;
  if (!b || b.studyId !== PERSONA_3C_STUDY_ID || !validSha(manifest) || !key || budget === null || budget <= 0 || budget > PERSONA_3C_MAX_BUDGET_USD) {
    return json({ error: 'Invalid Experiment 3C start request.' }, { status: 400 }, c);
  }
  const prior = await env.DB.prepare('SELECT * FROM persona_3c_runs WHERE idempotency_key=?').bind(key).first<Run>();
  if (prior) return json({ accepted: true, idempotent: true, run: pubRun(prior) }, {}, c);
  const now = new Date().toISOString();
  const id = `persona-3c-${crypto.randomUUID()}`;
  const run: Run = {
    id, study_id: PERSONA_3C_STUDY_ID, status: 'queued', stage: 'created', manifest_hash: manifest,
    budget_usd: budget, spent_usd: 0, gpu_hours: 0, created_at: now, updated_at: now,
    completed_at: null, error_message: null, last_record_id: null,
    judge_budget_usd: 0, judge_spent_usd: 0, judge_reserved_usd: 0, judge_calls: 0, judge_call_ceiling: 0,
  };
  try {
    await env.DB.prepare(
      'INSERT INTO persona_3c_runs (id,study_id,status,stage,manifest_hash,budget_usd,spent_usd,gpu_hours,created_at,updated_at,completed_at,error_message,last_record_id,idempotency_key) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
    ).bind(id, run.study_id, run.status, run.stage, run.manifest_hash, run.budget_usd, run.spent_usd, run.gpu_hours, now, now, null, null, null, key).run();
    await recordEvent(env.DB, id, 'created', 'system', '3C run queued', 'GPU pod run queued; no progress has been reported yet.', { manifestHash: manifest, budgetUsd: budget });
  } catch {
    const raced = await env.DB.prepare('SELECT * FROM persona_3c_runs WHERE idempotency_key=?').bind(key).first<Run>();
    if (raced) return json({ accepted: true, idempotent: true, run: pubRun(raced) }, {}, c);
    return json({ error: 'The run could not be queued safely.' }, { status: 409 }, c);
  }
  return json({ accepted: true, idempotent: false, run: pubRun(run) }, { status: 202 }, c);
}

export async function reportPersona3C(req: Request, env: Persona3CEnv, c: Record<string, string>) {
  if (!env.PERSONA_3C_TOKEN) return json({ error: 'Experiment 3C reporting is not configured.' }, { status: 503 }, c);
  if (!await authorized(req, env.PERSONA_3C_TOKEN)) return json({ error: 'Unauthorized.' }, { status: 401 }, c);
  const b = await body(req);
  if (!b || !validRun(b.runId)) return json({ error: 'Unknown or invalid run id.' }, { status: 400 }, c);
  if (!validStage(b.stage)) return json({ error: 'Invalid or missing stage.' }, { status: 400 }, c);
  const runId = String(b.runId);
  const stage = b.stage;

  let status: Persona3CRunStatus | undefined;
  if (b.status !== undefined) {
    if (typeof b.status !== 'string' || !REPORTABLE_STATUSES.includes(b.status)) {
      return json({ error: 'Invalid status.' }, { status: 400 }, c);
    }
    status = b.status as Persona3CRunStatus;
  }

  // progress is optional: a status-only report (e.g. a failure notice) has no progress to
  // upsert. When present it must be an object with integer done/total counters; done>total
  // is clamped rather than rejected, since token counters can overshoot by one batch.
  const progressRaw = b.progress;
  let done: number | null = null;
  let total: number | null = null;
  if (progressRaw !== undefined && progressRaw !== null) {
    if (typeof progressRaw !== 'object' || Array.isArray(progressRaw)) {
      return json({ error: 'Invalid progress.' }, { status: 400 }, c);
    }
    const rawDone = integer((progressRaw as Record<string, unknown>).done, 0, 100_000_000);
    const rawTotal = integer((progressRaw as Record<string, unknown>).total, 0, 100_000_000);
    if (rawDone === null || rawTotal === null) return json({ error: 'Invalid progress counters.' }, { status: 400 }, c);
    total = rawTotal;
    done = Math.min(rawDone, rawTotal);
  }

  const recordsRaw = b.records;
  const records: { recordId: string; payloadJson: string; sha256: string }[] = [];
  if (recordsRaw !== undefined) {
    if (!Array.isArray(recordsRaw) || recordsRaw.length > PERSONA_3C_MAX_RECORDS) {
      return json({ error: 'Invalid records batch.' }, { status: 400 }, c);
    }
    for (const raw of recordsRaw) {
      if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return json({ error: 'Invalid record entry.' }, { status: 400 }, c);
      const entry = raw as Record<string, unknown>;
      const recordId = str(entry.recordId, 1, 120);
      const sha = typeof entry.sha256 === 'string' ? entry.sha256.toLowerCase() : '';
      if (!recordId || !validSha(sha)) return json({ error: 'Invalid record entry.' }, { status: 400 }, c);

      // Preferred path: the client sends the exact JSON string it hashed, so integrity is
      // checked over the literal bytes rather than a re-serialization that can disagree on
      // float formatting (e.g. Python's json.dumps vs. this file's canonicalJson: 1e-05 vs
      // 0.00001, 1.0 vs 1). Falls back to canonicalizing `payload` for older callers.
      const payloadJsonRaw = entry.payloadJson;
      const payload = entry.payload;
      if (payloadJsonRaw !== undefined) {
        if (typeof payloadJsonRaw !== 'string' || new TextEncoder().encode(payloadJsonRaw).byteLength > 65_536) {
          return json({ error: `Record ${recordId} payloadJson must be a string of at most 64 KB.` }, { status: 400 }, c);
        }
        let parsed: unknown;
        try {
          parsed = JSON.parse(payloadJsonRaw);
        } catch {
          return json({ error: `Record ${recordId} payloadJson is not valid JSON.` }, { status: 400 }, c);
        }
        if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
          return json({ error: `Record ${recordId} payloadJson must parse to a plain object.` }, { status: 400 }, c);
        }
        const computed = await sha256Hex(payloadJsonRaw);
        if (computed !== sha) return json({ error: `Record ${recordId} sha256 does not match the canonical JSON of its payload.` }, { status: 400 }, c);
        records.push({ recordId, payloadJson: payloadJsonRaw, sha256: sha });
      } else if (payload !== undefined && payload !== null) {
        if (typeof payload !== 'object' || Array.isArray(payload)) {
          return json({ error: 'Invalid record entry.' }, { status: 400 }, c);
        }
        let canonical: string;
        try {
          canonical = canonicalJson(payload);
        } catch {
          return json({ error: `Record ${recordId} payload could not be canonicalized.` }, { status: 400 }, c);
        }
        const computed = await sha256Hex(canonical);
        if (computed !== sha) return json({ error: `Record ${recordId} sha256 does not match the canonical JSON of its payload.` }, { status: 400 }, c);
        records.push({ recordId, payloadJson: canonical, sha256: sha });
      } else {
        return json({ error: `Record ${recordId} is missing payload or payloadJson.` }, { status: 400 }, c);
      }
    }
  }

  const gpuHours = typeof b.gpuHours === 'number' && Number.isFinite(b.gpuHours) && b.gpuHours >= 0 ? b.gpuHours : null;
  const spentUsd = typeof b.spentUsd === 'number' && Number.isFinite(b.spentUsd) && b.spentUsd >= 0 ? b.spentUsd : null;
  const message = str(b.message, 1, 500) || null;

  const run = await env.DB.prepare('SELECT * FROM persona_3c_runs WHERE id=?').bind(runId).first<Run>();
  if (!run) return json({ error: 'Unknown run.' }, { status: 404 }, c);
  if (TERMINAL_STATUSES.includes(run.status)) return json({ error: 'Run is already complete, failed, or stopped.' }, { status: 409 }, c);

  const now = new Date().toISOString();
  const nextStatus: Persona3CRunStatus = status ?? (run.status === 'queued' ? 'running' : run.status);
  const terminal = TERMINAL_STATUSES.includes(nextStatus);
  const completedAt = terminal ? now : run.completed_at;
  const nextSpent = spentUsd ?? run.spent_usd;
  const nextGpu = gpuHours ?? run.gpu_hours;
  const lastRecordId = records.length ? records[records.length - 1].recordId : run.last_record_id;
  const errorMessage = nextStatus === 'failed' ? (message ?? run.error_message ?? 'Run failed.') : nextStatus === 'stopped' ? run.error_message : run.error_message;

  const statements: D1PreparedStatement[] = records.map((record) => env.DB.prepare(
    'INSERT OR IGNORE INTO persona_3c_records(run_id,stage,record_id,payload_json,sha256,created_at) VALUES(?,?,?,?,?,?)',
  ).bind(runId, stage, record.recordId, record.payloadJson, record.sha256, now));
  if (done !== null && total !== null) {
    statements.push(env.DB.prepare(
      'INSERT INTO persona_3c_progress(run_id,stage,done,total,updated_at) VALUES(?,?,?,?,?) ON CONFLICT(run_id,stage) DO UPDATE SET done=excluded.done,total=excluded.total,updated_at=excluded.updated_at',
    ).bind(runId, stage, done, total, now));
  }
  statements.push(env.DB.prepare(
    "UPDATE persona_3c_runs SET stage=?,status=?,spent_usd=?,gpu_hours=?,last_record_id=?,updated_at=?,completed_at=?,error_message=? WHERE id=? AND status NOT IN ('complete','failed','stopped')",
  ).bind(stage, nextStatus, nextSpent, nextGpu, lastRecordId, now, completedAt, errorMessage, runId));

  const results = await env.DB.batch(statements);
  const recordResults = results.slice(0, records.length);
  let accepted = 0;
  let duplicates = 0;
  for (const result of recordResults) {
    if (Number(result.meta.changes ?? 0) === 1) accepted += 1;
    else duplicates += 1;
  }
  const runUpdate = results[results.length - 1];
  if (Number(runUpdate?.meta.changes ?? 0) !== 1) return json({ error: 'Run was already finalized.' }, { status: 409 }, c);

  const progressText = done !== null && total !== null ? `${stage}: ${done}/${total}` : stage;
  const recordsText = records.length ? `; ${accepted} new record(s), ${duplicates} duplicate(s)` : '';
  await recordEvent(
    env.DB, runId, stage, nextStatus === 'failed' ? 'error' : 'progress', `${stage} · ${nextStatus}`,
    message ?? `${progressText}${recordsText}.`,
    { done, total, gpuHours: nextGpu, spentUsd: nextSpent, recordCount: records.length },
  );

  return json({ ok: true, runId, stage, accepted, duplicates }, {}, c);
}

export async function persona3CStatus(env: Persona3CEnv, c: Record<string, string>, runId?: string | null) {
  const run = runId
    ? await env.DB.prepare('SELECT * FROM persona_3c_runs WHERE id=?').bind(runId).first<Run>()
    : await env.DB.prepare('SELECT * FROM persona_3c_runs ORDER BY created_at DESC LIMIT 1').first<Run>();
  if (!run) return json({ run: null }, {}, c);
  const counts = await env.DB.prepare('SELECT stage,COUNT(*) AS n FROM persona_3c_records WHERE run_id=? GROUP BY stage').bind(run.id).all<{ stage: string; n: number }>();
  const progress = await env.DB.prepare('SELECT stage,done,total,updated_at AS updatedAt FROM persona_3c_progress WHERE run_id=? ORDER BY stage').bind(run.id).all<{ stage: string; done: number; total: number; updatedAt: string }>();
  const events = await env.DB.prepare('SELECT id,at,stage,kind,title,summary FROM persona_3c_events WHERE run_id=? ORDER BY id DESC LIMIT 50').bind(run.id).all<{ id: number; at: string; stage: string | null; kind: string; title: string; summary: string }>();
  const judgeCounts = await env.DB.prepare('SELECT status,COUNT(*) AS n FROM persona_3c_judge WHERE run_id=? GROUP BY status').bind(run.id).all<{ status: string; n: number }>();
  const judgeByStatus = Object.fromEntries(judgeCounts.results.map((row) => [row.status, row.n]));
  return json({
    run: pubRun(run),
    recordCounts: Object.fromEntries(counts.results.map((row) => [row.stage, row.n])),
    progress: progress.results,
    events: events.results.slice().reverse(),
    judge: {
      queued: Number(judgeByStatus.queued ?? 0),
      inFlight: Number(judgeByStatus.in_flight ?? 0),
      complete: Number(judgeByStatus.complete ?? 0),
      failed: Number(judgeByStatus.failed ?? 0),
      calls: run.judge_calls ?? 0,
      spentUsd: run.judge_spent_usd ?? 0,
      budgetUsd: run.judge_budget_usd ?? 0,
      ceiling: run.judge_call_ceiling ?? 0,
    },
  }, {}, c);
}

export async function stopPersona3C(req: Request, env: Persona3CEnv, c: Record<string, string>) {
  if (!env.PERSONA_3C_TOKEN) return json({ error: 'Experiment 3C reporting is not configured.' }, { status: 503 }, c);
  if (!await authorized(req, env.PERSONA_3C_TOKEN)) return json({ error: 'Unauthorized.' }, { status: 401 }, c);
  const b = await body(req);
  if (!b || !validRun(b.runId)) return json({ error: 'Unknown or invalid run id.' }, { status: 400 }, c);
  const runId = String(b.runId);
  const now = new Date().toISOString();
  const update = await env.DB.prepare(
    "UPDATE persona_3c_runs SET status='stopped',completed_at=?,updated_at=? WHERE id=? AND status NOT IN ('complete','failed','stopped')",
  ).bind(now, now, runId).run();
  if (Number(update.meta.changes ?? 0) !== 1) return json({ error: 'Run was already finalized or is unknown.' }, { status: 409 }, c);
  await recordEvent(env.DB, runId, null, 'system', '3C run stopped', 'Run stopped by owner control.', {});
  const run = await env.DB.prepare('SELECT * FROM persona_3c_runs WHERE id=?').bind(runId).first<Run>();
  return json({ accepted: true, runId, run: pubRun(run ?? null) }, {}, c);
}

// ---------------------------------------------------------------------------
// DESCRIBE stage (judge pass one): a blinded pair (A/B, randomized order) is
// shown to the judge, which answers one question with strict JSON. It never
// sees a persona vocabulary, feature ids, or which side is steered. Reuses
// 3B's per-token pricing (`judgeCallCost`) and its reserve-worst-case /
// settle-to-actual / charge-worst-case-on-failure budget pattern, scoped to
// each run's own judge_* counters rather than a separate runs table.
// ---------------------------------------------------------------------------
export const PERSONA_3C_JUDGE_MODEL = 'gpt-5.6-luna';
const JUDGE_OUTPUT_TOKENS = 400;
const JUDGE_MAX_ATTEMPTS = 3;
const JUDGE_STALE_MINUTES = 10;
const JUDGE_MAX_PLAN_PAIRS = 500;
export const PERSONA_3C_JUDGE_MAX_BUDGET_USD = 20;
export const PERSONA_3C_JUDGE_MAX_CALL_CEILING = 4000;
const JUDGE_MAX_RUN_JOBS = 25;
const JUDGE_BODY_MAX_BYTES = 4_000_000;
const JUDGE_TEXT_MAX_CHARS = 8000;
const JUDGE_SCENARIO_MAX_CHARS = 4000;
const JUDGE_DIRECTION_KEY_MAX_CHARS = 200;

const JUDGE_SYSTEM_PROMPT =
  'You compare two responses to the same prompt. Name one property of how the speaker comes across that ' +
  'differs between them, and say which response shows more of it. If they do not meaningfully differ in how ' +
  'the speaker comes across, answer neither. Do not judge quality or correctness. Classify whether the ' +
  'property is about the speaker (voice, stance, self-presentation, relation to the reader), the content ' +
  '(facts, topics, arguments), or the format (length, structure, wording artifacts).';

const JUDGE_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['property', 'more_in', 'about', 'confidence'],
  properties: {
    property: { type: 'string', maxLength: 200 },
    more_in: { type: 'string', enum: ['A', 'B', 'neither'] },
    about: { type: 'string', enum: ['speaker', 'content', 'format'] },
    confidence: { type: 'string', enum: ['low', 'medium', 'high'] },
  },
};

type JudgeJobStatus = 'queued' | 'in_flight' | 'complete' | 'failed';

type JudgeJob = {
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
  status: JudgeJobStatus;
  attempts: number;
  created_at: string;
  updated_at: string;
};

type JudgeResponse = {
  property: string;
  more_in: 'A' | 'B' | 'neither';
  about: 'speaker' | 'content' | 'format';
  confidence: 'low' | 'medium' | 'high';
};

function judgeWorstCaseCost(promptChars: number): number {
  const estimatedInputTokens = Math.max(1, Math.ceil(promptChars / 3.5));
  return judgeCallCost(estimatedInputTokens, 0, JUDGE_OUTPUT_TOKENS);
}

function judgeText(value: unknown, maxChars: number): string | null {
  return typeof value === 'string' && value.length > 0 && value.length <= maxChars ? value : null;
}

type JudgePairInput = { directionKey: string; scenario: string; textA: string; textB: string; orderSwap: boolean };

function validJudgePair(value: unknown): JudgePairInput | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
  const r = value as Record<string, unknown>;
  const directionKey = str(r.directionKey, 1, JUDGE_DIRECTION_KEY_MAX_CHARS);
  const scenario = judgeText(r.scenario, JUDGE_SCENARIO_MAX_CHARS);
  const textA = judgeText(r.textA, JUDGE_TEXT_MAX_CHARS);
  const textB = judgeText(r.textB, JUDGE_TEXT_MAX_CHARS);
  if (!directionKey || !scenario || textA === null || textB === null || typeof r.orderSwap !== 'boolean') return null;
  return { directionKey, scenario, textA, textB, orderSwap: r.orderSwap };
}

async function judgeJobId(pair: { directionKey: string; scenario: string; orderSwap: boolean }): Promise<string> {
  return sha256Hex(canonicalJson({ directionKey: pair.directionKey, scenario: pair.scenario, orderSwap: pair.orderSwap }));
}

function validJudgeResponse(value: unknown): value is JudgeResponse {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false;
  const r = value as Record<string, unknown>;
  const keys = Object.keys(r);
  if (keys.length !== 4 || !['property', 'more_in', 'about', 'confidence'].every((k) => keys.includes(k))) return false;
  if (typeof r.property !== 'string' || r.property.length > 200) return false;
  if (!['A', 'B', 'neither'].includes(String(r.more_in))) return false;
  if (!['speaker', 'content', 'format'].includes(String(r.about))) return false;
  if (!['low', 'medium', 'high'].includes(String(r.confidence))) return false;
  return true;
}

function judgeUsage(payload: Record<string, unknown>) {
  const u = payload.usage && typeof payload.usage === 'object' ? payload.usage as Record<string, unknown> : {};
  const d = u.input_tokens_details && typeof u.input_tokens_details === 'object' ? u.input_tokens_details as Record<string, unknown> : {};
  const inputTokens = Number.isInteger(u.input_tokens) ? Number(u.input_tokens) : 0;
  const cachedInputTokens = Number.isInteger(d.cached_tokens) ? Number(d.cached_tokens) : 0;
  const outputTokens = Number.isInteger(u.output_tokens) ? Number(u.output_tokens) : 0;
  const known = Number.isInteger(u.input_tokens) && Number.isInteger(u.output_tokens);
  return { inputTokens, cachedInputTokens, outputTokens, known };
}

/** POST /api/persona-3c/judge/plan: append (INSERT OR IGNORE, keyed on a
 * job_id hashed from {directionKey, scenario, orderSwap}) up to 500 blinded
 * pairs to the judge queue. The run's judge budget/ceiling can only be
 * raised, never lowered, and only while the run is running or complete. */
export async function planPersona3CJudge(req: Request, env: Persona3CEnv, c: Record<string, string>) {
  if (!env.PERSONA_3C_TOKEN) return json({ error: 'Experiment 3C reporting is not configured.' }, { status: 503 }, c);
  if (!await authorized(req, env.PERSONA_3C_TOKEN)) return json({ error: 'Unauthorized.' }, { status: 401 }, c);
  const b = await body(req, JUDGE_BODY_MAX_BYTES);
  if (!b || !validRun(b.runId)) return json({ error: 'Unknown or invalid run id.' }, { status: 400 }, c);
  const runId = String(b.runId);
  const budget = typeof b.judgeBudgetUsd === 'number' && Number.isFinite(b.judgeBudgetUsd) ? b.judgeBudgetUsd : null;
  const ceiling = integer(b.judgeCallCeiling, 1, PERSONA_3C_JUDGE_MAX_CALL_CEILING);
  if (budget === null || budget <= 0 || budget > PERSONA_3C_JUDGE_MAX_BUDGET_USD || ceiling === null) {
    return json({ error: 'Invalid judge budget or call ceiling.' }, { status: 400 }, c);
  }
  const rawPairs = b.pairs;
  if (!Array.isArray(rawPairs) || rawPairs.length === 0 || rawPairs.length > JUDGE_MAX_PLAN_PAIRS) {
    return json({ error: 'Invalid judge pairs batch (1-500 pairs per call).' }, { status: 400 }, c);
  }
  const pairs: JudgePairInput[] = [];
  for (const raw of rawPairs) {
    const pair = validJudgePair(raw);
    if (!pair) return json({ error: 'Invalid judge pair entry.' }, { status: 400 }, c);
    pairs.push(pair);
  }

  const run = await env.DB.prepare('SELECT * FROM persona_3c_runs WHERE id=?').bind(runId).first<Run>();
  if (!run) return json({ error: 'Unknown run.' }, { status: 404 }, c);
  if (!['running', 'complete'].includes(run.status)) {
    return json({ error: 'Judge plans can only be appended while the run is running or complete.' }, { status: 409 }, c);
  }

  const now = new Date().toISOString();
  const nextBudget = Math.max(run.judge_budget_usd ?? 0, budget);
  const nextCeiling = Math.max(run.judge_call_ceiling ?? 0, ceiling);

  const statements: D1PreparedStatement[] = [];
  for (const pair of pairs) {
    const jobId = await judgeJobId(pair);
    statements.push(env.DB.prepare(
      'INSERT OR IGNORE INTO persona_3c_judge(run_id,job_id,direction_key,scenario,order_swap,status,attempts,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?)',
    ).bind(runId, jobId, pair.directionKey, pair.scenario, pair.orderSwap ? 1 : 0, 'queued', 0, now, now));
    statements.push(env.DB.prepare(
      'INSERT OR IGNORE INTO persona_3c_judge_inputs(run_id,job_id,text_a,text_b) VALUES(?,?,?,?)',
    ).bind(runId, jobId, pair.textA, pair.textB));
  }
  statements.push(env.DB.prepare(
    'UPDATE persona_3c_runs SET judge_budget_usd=?,judge_call_ceiling=?,updated_at=? WHERE id=?',
  ).bind(nextBudget, nextCeiling, now, runId));

  const results = await env.DB.batch(statements);
  let queued = 0;
  let duplicate = 0;
  for (let i = 0; i < pairs.length; i += 1) {
    const jobInsert = results[i * 2];
    if (Number(jobInsert?.meta.changes ?? 0) === 1) queued += 1;
    else duplicate += 1;
  }

  await recordEvent(
    env.DB, runId, 'judge', 'system', 'Judge plan appended',
    `${queued} queued, ${duplicate} duplicate judge job(s).`,
    { queued, duplicate, judgeBudgetUsd: nextBudget, judgeCallCeiling: nextCeiling },
  );

  return json({ queued, duplicate, judgeBudgetUsd: nextBudget, judgeCallCeiling: nextCeiling }, {}, c);
}

/** POST /api/persona-3c/judge/run: claims up to `maxJobs` (<=25) queued
 * jobs and scores them sequentially against the OpenAI Responses API.
 * Stale in_flight jobs (>10 minutes old) are re-queued first. Each call
 * reserves a worst-case cost before calling the provider, refuses when the
 * run's judge budget or call ceiling would be exceeded, settles to the
 * actual cost on success, and charges the worst-case on failure (retried
 * up to 3 attempts, then marked failed). */
export async function runPersona3CJudge(req: Request, env: Persona3CEnv, c: Record<string, string>) {
  if (!env.PERSONA_3C_TOKEN) return json({ error: 'Experiment 3C reporting is not configured.' }, { status: 503 }, c);
  if (!await authorized(req, env.PERSONA_3C_TOKEN)) return json({ error: 'Unauthorized.' }, { status: 401 }, c);
  if (!env.OPENAI_API_KEY) return json({ error: 'The Experiment 3C judge is not configured.' }, { status: 503 }, c);
  const b = await body(req);
  if (!b || !validRun(b.runId)) return json({ error: 'Unknown or invalid run id.' }, { status: 400 }, c);
  const runId = String(b.runId);
  const maxJobs = integer(b.maxJobs, 1, JUDGE_MAX_RUN_JOBS) ?? JUDGE_MAX_RUN_JOBS;

  const run = await env.DB.prepare('SELECT * FROM persona_3c_runs WHERE id=?').bind(runId).first<Run>();
  if (!run) return json({ error: 'Unknown run.' }, { status: 404 }, c);
  if (['failed', 'stopped'].includes(run.status)) return json({ error: 'Run is terminal; judge calls are disabled.' }, { status: 409 }, c);

  const nowMs = Date.now();
  const staleBefore = new Date(nowMs - JUDGE_STALE_MINUTES * 60_000).toISOString();
  await env.DB.prepare(
    "UPDATE persona_3c_judge SET status='queued',updated_at=? WHERE run_id=? AND status='in_flight' AND updated_at<=?",
  ).bind(new Date(nowMs).toISOString(), runId, staleBefore).run();

  const queuedJobs = await env.DB.prepare(
    "SELECT * FROM persona_3c_judge WHERE run_id=? AND status='queued' ORDER BY created_at LIMIT ?",
  ).bind(runId, maxJobs).all<JudgeJob>();

  let ran = 0;
  let complete = 0;
  let failed = 0;
  let budgetExhausted = false;

  for (const job of queuedJobs.results) {
    if (budgetExhausted) break;
    const claim = await env.DB.prepare(
      "UPDATE persona_3c_judge SET status='in_flight',updated_at=? WHERE run_id=? AND job_id=? AND status='queued'",
    ).bind(new Date().toISOString(), runId, job.job_id).run();
    if (Number(claim.meta.changes ?? 0) !== 1) continue;

    const inputs = await env.DB.prepare(
      'SELECT text_a AS textA, text_b AS textB FROM persona_3c_judge_inputs WHERE run_id=? AND job_id=?',
    ).bind(runId, job.job_id).first<{ textA: string; textB: string }>();
    if (!inputs) {
      await env.DB.prepare("UPDATE persona_3c_judge SET status='failed',updated_at=? WHERE run_id=? AND job_id=?")
        .bind(new Date().toISOString(), runId, job.job_id).run();
      failed += 1;
      continue;
    }

    const userText = `${job.scenario}\n\nA:\n${inputs.textA}\n\nB:\n${inputs.textB}`;
    const promptSha256 = await sha256Hex(canonicalJson({ system: JUDGE_SYSTEM_PROMPT, user: userText }));
    const cap = judgeWorstCaseCost(JUDGE_SYSTEM_PROMPT.length + userText.length);

    const reserve = await env.DB.prepare(
      'UPDATE persona_3c_runs SET judge_calls=judge_calls+1,judge_reserved_usd=judge_reserved_usd+?,updated_at=? WHERE id=? AND judge_calls<judge_call_ceiling AND judge_spent_usd+judge_reserved_usd+?<=judge_budget_usd',
    ).bind(cap, new Date().toISOString(), runId, cap).run();
    if (Number(reserve.meta.changes ?? 0) !== 1) {
      // Budget or ceiling reached: put the job back to queued and stop --
      // every later job would fail the same, monotonic check.
      await env.DB.prepare("UPDATE persona_3c_judge SET status='queued',updated_at=? WHERE run_id=? AND job_id=?")
        .bind(new Date().toISOString(), runId, job.job_id).run();
      budgetExhausted = true;
      break;
    }

    ran += 1;
    const attempts = job.attempts + 1;
    try {
      const response = await fetch('https://api.openai.com/v1/responses', {
        method: 'POST',
        signal: AbortSignal.timeout(60000),
        headers: { authorization: `Bearer ${env.OPENAI_API_KEY}`, 'content-type': 'application/json' },
        body: JSON.stringify({
          model: PERSONA_3C_JUDGE_MODEL,
          reasoning: { effort: 'high' },
          input: [
            { role: 'system', content: [{ type: 'input_text', text: JUDGE_SYSTEM_PROMPT }] },
            { role: 'user', content: [{ type: 'input_text', text: userText }] },
          ],
          text: { verbosity: 'low', format: { type: 'json_schema', name: 'experiment_3c_judge_pass1', strict: true, schema: JUDGE_SCHEMA } },
          max_output_tokens: JUDGE_OUTPUT_TOKENS,
          truncation: 'auto',
          store: false,
        }),
      });
      const payload = await response.json().catch(() => ({})) as Record<string, unknown>;
      if (!response.ok) throw new Error(`OpenAI returned ${response.status}`);
      let text = '';
      for (const o of (payload.output as Array<Record<string, unknown>> ?? [])) {
        for (const item of (o.content as Array<Record<string, unknown>> ?? [])) {
          if (item.type === 'output_text' && typeof item.text === 'string') text = item.text;
        }
      }
      if (!text) throw new Error('Judge response contained no structured output.');
      const parsed = JSON.parse(text) as unknown;
      if (!validJudgeResponse(parsed)) throw new Error('Judge returned invalid JSON.');

      const usage = judgeUsage(payload);
      const actual = usage.known ? judgeCallCost(usage.inputTokens, usage.cachedInputTokens, usage.outputTokens) : cap;
      const at = new Date().toISOString();
      await env.DB.batch([
        env.DB.prepare(
          "UPDATE persona_3c_judge SET status='complete',prompt_sha256=?,response_json=?,provider_response_id=?,input_tokens=?,cached_input_tokens=?,output_tokens=?,cost_usd=?,attempts=?,updated_at=? WHERE run_id=? AND job_id=?",
        ).bind(promptSha256, JSON.stringify(parsed), typeof payload.id === 'string' ? payload.id : null, usage.inputTokens, usage.cachedInputTokens, usage.outputTokens, actual, attempts, at, runId, job.job_id),
        env.DB.prepare(
          'UPDATE persona_3c_runs SET judge_reserved_usd=judge_reserved_usd-?,judge_spent_usd=judge_spent_usd+?,updated_at=? WHERE id=? AND judge_reserved_usd>=?',
        ).bind(cap, actual, at, runId, cap),
      ]);
      complete += 1;
    } catch {
      const at = new Date().toISOString();
      const nextStatus: JudgeJobStatus = attempts >= JUDGE_MAX_ATTEMPTS ? 'failed' : 'queued';
      await env.DB.batch([
        env.DB.prepare(
          'UPDATE persona_3c_judge SET status=?,attempts=?,updated_at=? WHERE run_id=? AND job_id=?',
        ).bind(nextStatus, attempts, at, runId, job.job_id),
        env.DB.prepare(
          'UPDATE persona_3c_runs SET judge_reserved_usd=judge_reserved_usd-?,judge_spent_usd=judge_spent_usd+?,updated_at=? WHERE id=? AND judge_reserved_usd>=?',
        ).bind(cap, cap, at, runId, cap),
      ]);
      failed += 1;
    }
  }

  const remaining = await env.DB.prepare(
    "SELECT COUNT(*) AS n FROM persona_3c_judge WHERE run_id=? AND status='queued'",
  ).bind(runId).first<{ n: number }>();
  const updatedRun = await env.DB.prepare('SELECT judge_spent_usd FROM persona_3c_runs WHERE id=?').bind(runId).first<{ judge_spent_usd: number }>();

  await recordEvent(
    env.DB, runId, 'judge', 'progress', 'Judge pass ran',
    `${ran} call(s): ${complete} complete, ${failed} failed.`,
    { ran, complete, failed, remainingQueued: Number(remaining?.n ?? 0) },
  );

  return json({
    ran, complete, failed,
    remainingQueued: Number(remaining?.n ?? 0),
    judgeSpentUsd: Number(updatedRun?.judge_spent_usd ?? run.judge_spent_usd ?? 0),
  }, {}, c);
}

/** GET /api/persona-3c/judge/results?runId=: auth required. Returns every
 * complete job's parsed response, usage, and cost -- never the blinded
 * texts (those live only in persona_3c_judge_inputs, which this route
 * never selects from). */
export async function persona3CJudgeResults(req: Request, env: Persona3CEnv, c: Record<string, string>, runId: string | null) {
  if (!env.PERSONA_3C_TOKEN) return json({ error: 'Experiment 3C reporting is not configured.' }, { status: 503 }, c);
  if (!await authorized(req, env.PERSONA_3C_TOKEN)) return json({ error: 'Unauthorized.' }, { status: 401 }, c);
  if (!validRun(runId)) return json({ error: 'Unknown or invalid run id.' }, { status: 400 }, c);
  const rows = await env.DB.prepare(
    "SELECT job_id AS jobId, direction_key AS directionKey, scenario, order_swap AS orderSwap, response_json AS responseJson, input_tokens AS inputTokens, cached_input_tokens AS cachedInputTokens, output_tokens AS outputTokens, cost_usd AS costUsd FROM persona_3c_judge WHERE run_id=? AND status='complete' ORDER BY created_at",
  ).bind(runId).all<{ jobId: string; directionKey: string; scenario: string; orderSwap: number; responseJson: string; inputTokens: number; cachedInputTokens: number; outputTokens: number; costUsd: number }>();
  const results = rows.results.map((row) => {
    let response: unknown = null;
    try { response = JSON.parse(row.responseJson); } catch { response = null; }
    return {
      jobId: row.jobId,
      directionKey: row.directionKey,
      scenario: row.scenario,
      orderSwap: Boolean(row.orderSwap),
      response,
      usage: { inputTokens: row.inputTokens, cachedInputTokens: row.cachedInputTokens, outputTokens: row.outputTokens },
      costUsd: row.costUsd,
    };
  });
  return json({ runId, results }, {}, c);
}
