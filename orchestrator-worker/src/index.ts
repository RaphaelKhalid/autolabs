import { initialPublicState, globalSpend, nowIso, recentEvents } from './db';
import { secretEquals, bearer, cors } from './security';
import type { RunParams } from './types';
export { AutolabsWorkflow } from './workflow';

function json(value: unknown, init: ResponseInit = {}, corsHeaders?: Record<string, string>) {
  const headers = new Headers(init.headers);
  headers.set('content-type', 'application/json; charset=utf-8');
  headers.set('cache-control', 'no-store');
  for (const [key, item] of Object.entries(corsHeaders ?? {})) headers.set(key, item);
  return new Response(JSON.stringify(value), { ...init, headers });
}

function validStart(value: unknown): value is Partial<RunParams> & { mode: 'rehearsal' | 'competition' } {
  if (!value || typeof value !== 'object') return false;
  const mode = (value as Record<string, unknown>).mode;
  return mode === 'rehearsal' || mode === 'competition';
}

async function currentRun(env: Env) {
  return env.DB.prepare('SELECT id,public_state_json,report_json FROM runs ORDER BY created_at DESC LIMIT 1').first<{ id: string; public_state_json: string; report_json: string | null }>();
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);
    const corsHeaders = cors(request.headers.get('origin'), env.PUBLIC_SITE_ORIGIN);
    if (request.method === 'OPTIONS') return new Response(null, { status: 204, headers: corsHeaders });
    try {
      if (request.method === 'GET' && url.pathname === '/health') return json({ ok: true, service: 'autolabs-orchestrator', time: nowIso() }, {}, corsHeaders);

      if (request.method === 'GET' && url.pathname === '/api/experiments/current') {
        const row = await currentRun(env);
        if (!row) return json({ error: 'No experiment has started.' }, { status: 404 }, corsHeaders);
        const state = JSON.parse(row.public_state_json) as Record<string, unknown>;
        state.events = await recentEvents(env.DB, row.id);
        return json(state, {}, corsHeaders);
      }

      const eventMatch = url.pathname.match(/^\/api\/experiments\/([a-zA-Z0-9-]+)\/events$/);
      if (request.method === 'GET' && eventMatch) return json({ runId: eventMatch[1], events: await recentEvents(env.DB, eventMatch[1], 5000) }, {}, corsHeaders);

      const reportMatch = url.pathname.match(/^\/api\/experiments\/([a-zA-Z0-9-]+)\/report$/);
      if (request.method === 'GET' && reportMatch) {
        const row = await env.DB.prepare('SELECT report_json FROM runs WHERE id=?').bind(reportMatch[1]).first<{ report_json: string | null }>();
        if (!row?.report_json) return json({ error: 'Report is not available yet.' }, { status: 404 }, corsHeaders);
        return json(JSON.parse(row.report_json), {}, corsHeaders);
      }

      if (request.method === 'POST' && url.pathname === '/api/experiments/start') {
        if (!await secretEquals(bearer(request), env.ADMIN_TOKEN)) return json({ error: 'Unauthorized.' }, { status: 401 }, corsHeaders);
        if (Number(request.headers.get('content-length') ?? 0) > 16_384) return json({ error: 'Payload too large.' }, { status: 413 }, corsHeaders);
        const raw = await request.json();
        if (!validStart(raw)) return json({ error: 'Invalid start configuration.' }, { status: 400 }, corsHeaders);
        const active = await env.DB.prepare(`SELECT id FROM runs WHERE status='running' LIMIT 1`).first();
        if (active) return json({ error: 'An experiment is already running.' }, { status: 409 }, corsHeaders);
        const mode = raw.mode;
        const targetRounds = mode === 'rehearsal' ? 1 : Math.min(50, Math.max(25, Number(raw.targetRounds ?? 50)));
        const minimumRounds = mode === 'rehearsal' ? 1 : 25;
        const phaseMinutes = 5;
        const budgetUsd = 50;
        const reserveUsd = 1.5;
        const spent = await globalSpend(env.DB);
        const guaranteedCost = minimumRounds * 0.4;
        if (spent + guaranteedCost > budgetUsd - reserveUsd) return json({ error: 'Insufficient remaining API budget to guarantee the minimum run.' }, { status: 409 }, corsHeaders);
        const runId = `${mode}-${crypto.randomUUID()}`;
        const params: RunParams = { runId, mode, targetRounds, minimumRounds, phaseMinutes, budgetUsd, reserveUsd };
        const state = initialPublicState(params);
        state.spentUsd = spent;
        const created = nowIso();
        await env.DB.prepare(`INSERT INTO runs(id,workflow_id,status,mode,phase,round,target_rounds,minimum_rounds,phase_minutes,phase_ends_at,started_at,budget_usd,reserve_usd,spent_usd,calls,best_support_json,best_label,best_verified,sota_improved,public_state_json,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)`)
          .bind(runId, runId, 'running', mode, 'ribbon', 0, targetRounds, minimumRounds, phaseMinutes, state.phaseEndsAt, state.startedAt, budgetUsd, reserveUsd, spent, 0, JSON.stringify(state.bestSupport), state.bestLabel, 0, 0, JSON.stringify(state), created, created).run();
        const instance = await env.AUTOLABS_WORKFLOW.create({ id: runId, params, retention: { successRetention: '3 days', errorRetention: '3 days' } });
        return json({ accepted: true, runId, workflowId: instance.id, mode, targetRounds, minimumRounds, phaseMinutes, budgetUsd }, { status: 202 }, corsHeaders);
      }

      if (request.method === 'POST' && url.pathname === '/api/jobs/result') {
        if (!await secretEquals(request.headers.get('x-autolabs-callback') ?? '', env.CALLBACK_SECRET)) return json({ error: 'Unauthorized.' }, { status: 401 }, corsHeaders);
        if (Number(request.headers.get('content-length') ?? 0) > 1_000_000) return json({ error: 'Payload too large.' }, { status: 413 }, corsHeaders);
        const body = await request.json<{ id?: string; ok?: boolean; result?: unknown; error?: string }>();
        if (!body.id || typeof body.ok !== 'boolean') return json({ error: 'Invalid job result.' }, { status: 400 }, corsHeaders);
        const row = await env.DB.prepare('SELECT run_id,round,agent_id,job_type FROM jobs WHERE id=?').bind(body.id).first<{ run_id: string; round: number; agent_id: string; job_type: string }>();
        if (!row) return json({ error: 'Unknown job.' }, { status: 404 }, corsHeaders);
        await env.DB.prepare(`UPDATE jobs SET status=?,result_json=?,error=?,completed_at=? WHERE id=?`).bind(body.ok ? 'complete' : 'failed', JSON.stringify(body.result ?? null), body.error ?? null, nowIso(), body.id).run();
        await env.DB.prepare(`INSERT OR IGNORE INTO events(run_id,seq,at,round,phase,agent_id,kind,title,summary,payload_json,visible) VALUES(?,?,?,?,?,?,?,?,?,?,1)`)
          .bind(row.run_id, row.round * 1000 + 390 + Math.abs(body.id.split('').reduce((n, char) => n + char.charCodeAt(0), 0)) % 90, nowIso(), row.round, 'research', row.agent_id, body.ok ? 'tool' : 'error', `${row.job_type} ${body.ok ? 'completed' : 'failed'}`, body.ok ? 'The exact code-job result is available to its agent next round.' : body.error ?? 'Code job failed.', JSON.stringify(body.result ?? {})).run();
        return json({ accepted: true }, {}, corsHeaders);
      }

      return json({ error: 'Not found.' }, { status: 404 }, corsHeaders);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      console.error(JSON.stringify({ message: 'request failed', error: message, path: url.pathname }));
      return json({ error: 'Internal server error.' }, { status: 500 }, corsHeaders);
    }
  },
} satisfies ExportedHandler<Env>;
