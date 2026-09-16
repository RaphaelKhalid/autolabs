import { bearer, secretEquals } from './security';

export const PERSONA_3A_STUDY_ID = 'experiment-003a-v1';
export const PERSONA_3A_PHASE = 'development';
export const PERSONA_3A_MAX_RUNTIME_SECONDS = 6_600;
export const PERSONA_3A_MANIFEST_HASH = 'sha256:9fd05fffe50a6f402fea4dab6f7cf51dcd66ef78ea9fa450374bd0f2411ca766';

type Persona3AEnv = Env & {
  PERSONA_3A_TOKEN?: string;
  PERSONA_3A_RELAY_TOKEN?: string;
};

type PersonaLaunchRow = {
  id: string;
  study_id: string;
  phase: 'preflight' | 'development';
  max_runtime_seconds: number;
  status: 'queued' | 'claimed' | 'started' | 'completed' | 'failed' | 'cancelled';
  idempotency_key: string;
  manifest_hash: string;
  requested_at: string;
  updated_at: string;
  started_at: string | null;
  completed_at: string | null;
  error_message: string | null;
};

const MAX_BODY_BYTES = 8_192;

function json(value: unknown, init: ResponseInit = {}, corsHeaders: Record<string, string>) {
  const headers = new Headers(init.headers);
  headers.set('content-type', 'application/json; charset=utf-8');
  headers.set('cache-control', 'no-store');
  for (const [key, item] of Object.entries(corsHeaders)) headers.set(key, item);
  return new Response(JSON.stringify(value), { ...init, headers });
}

function publicLaunch(row: PersonaLaunchRow) {
  return {
    id: row.id,
    studyId: row.study_id,
    phase: row.phase,
    maxRuntimeSeconds: row.max_runtime_seconds,
    status: row.status,
    manifestHash: row.manifest_hash,
    requestedAt: row.requested_at,
    updatedAt: row.updated_at,
    startedAt: row.started_at,
    completedAt: row.completed_at,
    error: row.error_message,
  };
}

async function authorized(request: Request, token: string | undefined) {
  return Boolean(token) && await secretEquals(bearer(request), token as string);
}

async function body(request: Request) {
  const contentLength = Number(request.headers.get('content-length') ?? 0);
  if (!Number.isFinite(contentLength) || contentLength > MAX_BODY_BYTES) return null;
  const text = await request.text();
  if (new TextEncoder().encode(text).byteLength > MAX_BODY_BYTES) return null;
  try {
    const value = JSON.parse(text) as unknown;
    return value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : null;
  } catch {
    return null;
  }
}

function idempotencyKey(value: unknown) {
  return typeof value === 'string' && value.length >= 16 && value.length <= 120 ? value.trim() : '';
}

function publicState(row: PersonaLaunchRow | null) {
  return row ? { launch: publicLaunch(row) } : {
    launch: {
      studyId: PERSONA_3A_STUDY_ID,
      phase: PERSONA_3A_PHASE,
      maxRuntimeSeconds: PERSONA_3A_MAX_RUNTIME_SECONDS,
      status: 'idle',
      manifestHash: PERSONA_3A_MANIFEST_HASH,
    },
  };
}

export async function startPersona3A(request: Request, env: Persona3AEnv, corsHeaders: Record<string, string>) {
  if (String(env.PERSONA_3A_LAUNCH_ENABLED) !== 'true') return json({ error: 'Experiment 3A launch is not enabled.' }, { status: 503 }, corsHeaders);
  if (!await authorized(request, env.PERSONA_3A_TOKEN ?? env.ADMIN_TOKEN)) return json({ error: 'Unauthorized.' }, { status: 401 }, corsHeaders);

  const parsed = await body(request);
  const key = idempotencyKey(parsed?.idempotencyKey);
  const phase = parsed?.phase;
  const maxRuntimeSeconds = parsed?.maxRuntimeSeconds;
  if (!parsed || parsed.studyId !== PERSONA_3A_STUDY_ID || phase !== PERSONA_3A_PHASE || !key ||
      !Number.isInteger(maxRuntimeSeconds) || (maxRuntimeSeconds as number) < 900 || (maxRuntimeSeconds as number) > PERSONA_3A_MAX_RUNTIME_SECONDS) {
    return json({ error: 'Only the fixed bounded Experiment 3A development launch is accepted.' }, { status: 400 }, corsHeaders);
  }

  const existing = await env.DB.prepare('SELECT * FROM persona_launch_requests WHERE idempotency_key=?')
    .bind(key)
    .first<PersonaLaunchRow>();
  if (existing) return json({ accepted: true, idempotent: true, ...publicState(existing) }, {}, corsHeaders);

  const now = new Date().toISOString();
  const row: PersonaLaunchRow = {
    id: `persona-3a-${crypto.randomUUID()}`,
    study_id: PERSONA_3A_STUDY_ID,
    phase: PERSONA_3A_PHASE,
    max_runtime_seconds: maxRuntimeSeconds as number,
    status: 'queued',
    idempotency_key: key,
    manifest_hash: PERSONA_3A_MANIFEST_HASH,
    requested_at: now,
    updated_at: now,
    started_at: null,
    completed_at: null,
    error_message: null,
  };

  try {
    await env.DB.prepare(`INSERT INTO persona_launch_requests
      (id,study_id,phase,max_runtime_seconds,status,idempotency_key,manifest_hash,requested_at,updated_at,started_at,completed_at,error_message)
      VALUES(?,?,?,?,?,?,?,?,?,?,?,?)`)
      .bind(row.id, row.study_id, row.phase, row.max_runtime_seconds, row.status, row.idempotency_key, row.manifest_hash, row.requested_at, row.updated_at, null, null, null)
      .run();
  } catch {
    const raced = await env.DB.prepare('SELECT * FROM persona_launch_requests WHERE idempotency_key=?').bind(key).first<PersonaLaunchRow>();
    if (raced) return json({ accepted: true, idempotent: true, ...publicState(raced) }, {}, corsHeaders);
    return json({ error: 'The launch request could not be queued safely.' }, { status: 409 }, corsHeaders);
  }
  return json({ accepted: true, idempotent: false, ...publicState(row) }, { status: 202 }, corsHeaders);
}

export async function persona3AStatus(env: Persona3AEnv, corsHeaders: Record<string, string>) {
  const row = await env.DB.prepare('SELECT * FROM persona_launch_requests WHERE study_id=? ORDER BY requested_at DESC LIMIT 1')
    .bind(PERSONA_3A_STUDY_ID)
    .first<PersonaLaunchRow>();
  return json(publicState(row), {}, corsHeaders);
}

export async function claimPersona3A(request: Request, env: Persona3AEnv, corsHeaders: Record<string, string>) {
  if (String(env.PERSONA_3A_LAUNCH_ENABLED) !== 'true') return json({ error: 'Experiment 3A launch is not enabled.' }, { status: 503 }, corsHeaders);
  if (!await authorized(request, env.PERSONA_3A_RELAY_TOKEN)) return json({ error: 'Unauthorized.' }, { status: 401 }, corsHeaders);
  const queued = await env.DB.prepare(`SELECT * FROM persona_launch_requests
    WHERE study_id=? AND status='queued' ORDER BY requested_at ASC LIMIT 1`).bind(PERSONA_3A_STUDY_ID).first<PersonaLaunchRow>();
  if (!queued) return json({ launch: null }, {}, corsHeaders);
  const now = new Date().toISOString();
  const claimed = await env.DB.prepare(`UPDATE persona_launch_requests SET status='claimed',updated_at=?
    WHERE id=? AND status='queued'`).bind(now, queued.id).run();
  if (Number(claimed.meta.changes ?? 0) !== 1) return json({ error: 'Launch request was claimed by another relay.' }, { status: 409 }, corsHeaders);
  return json({
    launch: {
      id: queued.id,
      studyId: queued.study_id,
      phase: queued.phase,
      maxRuntimeSeconds: queued.max_runtime_seconds,
      manifestHash: queued.manifest_hash,
      requestedAt: queued.requested_at,
    },
  }, {}, corsHeaders);
}

export async function updatePersona3A(request: Request, env: Persona3AEnv, corsHeaders: Record<string, string>) {
  if (!await authorized(request, env.PERSONA_3A_RELAY_TOKEN)) return json({ error: 'Unauthorized.' }, { status: 401 }, corsHeaders);
  const parsed = await body(request);
  const id = typeof parsed?.id === 'string' && /^persona-3a-[a-zA-Z0-9-]+$/.test(parsed.id) ? parsed.id : '';
  const status = parsed?.status;
  const allowed = status === 'started' || status === 'completed' || status === 'failed' || status === 'cancelled';
  const errorMessage = typeof parsed?.error === 'string' && parsed.error.length <= 500 ? parsed.error : null;
  if (!id || !allowed) return json({ error: 'Invalid launch update.' }, { status: 400 }, corsHeaders);
  const now = new Date().toISOString();
  const result = status === 'started'
    ? await env.DB.prepare(`UPDATE persona_launch_requests SET status='started',started_at=COALESCE(started_at,?),updated_at=? WHERE id=? AND status='claimed'`).bind(now, now, id).run()
    : await env.DB.prepare(`UPDATE persona_launch_requests SET status=?,completed_at=?,updated_at=?,error_message=? WHERE id=? AND status IN ('claimed','started')`).bind(status, now, now, errorMessage, id).run();
  if (Number(result.meta.changes ?? 0) !== 1) return json({ error: 'Launch update was stale or already finalized.' }, { status: 409 }, corsHeaders);
  return json({ accepted: true, status }, {}, corsHeaders);
}

