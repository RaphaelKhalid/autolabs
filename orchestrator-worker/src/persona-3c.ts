import { bearer, secretEquals } from './security';

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
};

function json(value: unknown, init: ResponseInit = {}, c: Record<string, string> = {}) {
  const headers = new Headers(init.headers);
  headers.set('content-type', 'application/json; charset=utf-8');
  headers.set('cache-control', 'no-store');
  for (const [key, item] of Object.entries(c)) headers.set(key, item);
  return new Response(JSON.stringify(value), { ...init, headers });
}

async function body(req: Request): Promise<Record<string, unknown> | null> {
  const declared = Number(req.headers.get('content-length') ?? 0);
  if (!Number.isFinite(declared) || declared > MAX_BODY_BYTES) return null;
  const text = await req.text();
  if (new TextEncoder().encode(text).byteLength > MAX_BODY_BYTES || !text.trim()) return null;
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
  const records: { recordId: string; payload: Record<string, unknown>; sha256: string }[] = [];
  if (recordsRaw !== undefined) {
    if (!Array.isArray(recordsRaw) || recordsRaw.length > PERSONA_3C_MAX_RECORDS) {
      return json({ error: 'Invalid records batch.' }, { status: 400 }, c);
    }
    for (const raw of recordsRaw) {
      if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return json({ error: 'Invalid record entry.' }, { status: 400 }, c);
      const entry = raw as Record<string, unknown>;
      const recordId = str(entry.recordId, 1, 120);
      const payload = entry.payload;
      const sha = typeof entry.sha256 === 'string' ? entry.sha256.toLowerCase() : '';
      if (!recordId || !payload || typeof payload !== 'object' || Array.isArray(payload) || !validSha(sha)) {
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
      records.push({ recordId, payload: payload as Record<string, unknown>, sha256: sha });
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
  ).bind(runId, stage, record.recordId, JSON.stringify(record.payload), record.sha256, now));
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
  return json({
    run: pubRun(run),
    recordCounts: Object.fromEntries(counts.results.map((row) => [row.stage, row.n])),
    progress: progress.results,
    events: events.results.slice().reverse(),
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
