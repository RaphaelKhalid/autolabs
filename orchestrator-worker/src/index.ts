import { initialPublicState, globalSpend, nowIso, publicJobs, recentEvents, recentEventSummaries } from './db';
import { secretEquals, bearer, cors, verifyCallbackSignature } from './security';
import { AGENT_IDS, type AgentId, type RunParams } from './types';
export { AutolabsWorkflow } from './workflow';

const COMPETITION_ROUNDS = 50;
const MINIMUM_ROUNDS = 25;
const PHASE_MINUTES = 5;
const BUDGET_USD = 50;
const RESERVE_USD = 1.5;
const ROUND_AUTHORIZATION_USD = 1.75;
const AGENT_ORDER: Record<AgentId, number> = { mira: 0, pip: 1, orum: 2, solvi: 3, tess: 4 };

function json(value: unknown, init: ResponseInit = {}, corsHeaders?: Record<string, string>) {
  const headers = new Headers(init.headers);
  headers.set('content-type', 'application/json; charset=utf-8');
  headers.set('cache-control', 'no-store');
  for (const [key, item] of Object.entries(corsHeaders ?? {})) headers.set(key, item);
  return new Response(JSON.stringify(value), { ...init, headers });
}

function validStart(value: unknown): value is { mode: 'rehearsal' | 'competition' } {
  if (!value || typeof value !== 'object') return false;
  const mode = (value as Record<string, unknown>).mode;
  return mode === 'rehearsal' || mode === 'competition';
}

function hasRuntimeSecrets(env: Env) {
  return Boolean(env.ADMIN_TOKEN && env.OPENAI_API_KEY && env.EXA_API_KEY && env.CALLBACK_SECRET && env.GITHUB_TOKEN);
}

async function currentRun(env: Env) {
  return env.DB.prepare('SELECT id,public_state_json,report_json FROM runs ORDER BY created_at DESC LIMIT 1')
    .first<{ id: string; public_state_json: string; report_json: string | null }>();
}

function callbackSequence(round: number, agentId: AgentId, jobId: string) {
  const jobIndex = Number(jobId.slice(jobId.lastIndexOf('-') + 1));
  if (!Number.isInteger(jobIndex) || jobIndex < 0 || jobIndex > 2) throw new Error('Invalid job id.');
  return round * 1000 + 400 + AGENT_ORDER[agentId] * 10 + jobIndex;
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);
    const corsHeaders = cors(request.headers.get('origin'), env.PUBLIC_SITE_ORIGIN);
    if (request.method === 'OPTIONS') return new Response(null, { status: 204, headers: corsHeaders });

    try {
      if (request.method === 'GET' && url.pathname === '/health') {
        return json({ ok: true, service: 'autolabs-orchestrator', time: nowIso() }, {}, corsHeaders);
      }

      if (request.method === 'GET' && url.pathname === '/api/experiments/current') {
        const row = await currentRun(env);
        if (!row) return json({ error: 'No experiment has started.' }, { status: 404 }, corsHeaders);
        const state = JSON.parse(row.public_state_json) as Record<string, unknown>;
        state.events = await recentEventSummaries(env.DB, row.id, 120);
        state.jobs = await publicJobs(env.DB, row.id);
        return json(state, {}, corsHeaders);
      }

      const jobsMatch = url.pathname.match(/^\/api\/experiments\/([a-zA-Z0-9-]+)\/jobs$/);
      if (request.method === 'GET' && jobsMatch) {
        return json({ runId: jobsMatch[1], jobs: await publicJobs(env.DB, jobsMatch[1]) }, {}, corsHeaders);
      }

      const eventMatch = url.pathname.match(/^\/api\/experiments\/([a-zA-Z0-9-]+)\/events$/);
      if (request.method === 'GET' && eventMatch) {
        const requestedLimit = Number(url.searchParams.get('limit') ?? 250);
        const limit = Number.isFinite(requestedLimit) ? Math.max(1, Math.min(500, Math.floor(requestedLimit))) : 250;
        const requestedBefore = Number(url.searchParams.get('before'));
        const before = Number.isSafeInteger(requestedBefore) && requestedBefore > 0 ? requestedBefore : undefined;
        const requestedAgent = url.searchParams.get('agentId');
        const agentId = requestedAgent && AGENT_IDS.includes(requestedAgent as AgentId) ? requestedAgent as AgentId : undefined;
        const events = await recentEvents(env.DB, eventMatch[1], limit, before, agentId);
        const nextBefore = events.length === limit ? Number(events[0]?.seq ?? 0) || null : null;
        return json({ runId: eventMatch[1], events, nextBefore }, {}, corsHeaders);
      }

      const reportMatch = url.pathname.match(/^\/api\/experiments\/([a-zA-Z0-9-]+)\/report$/);
      if (request.method === 'GET' && reportMatch) {
        const row = await env.DB.prepare('SELECT report_json FROM runs WHERE id=?')
          .bind(reportMatch[1])
          .first<{ report_json: string | null }>();
        if (!row?.report_json) return json({ error: 'Report is not available yet.' }, { status: 404 }, corsHeaders);
        return json(JSON.parse(row.report_json), {}, corsHeaders);
      }

      if (request.method === 'POST' && url.pathname === '/api/experiments/start') {
        if (!hasRuntimeSecrets(env)) return json({ error: 'The research engine is not fully configured.' }, { status: 503 }, corsHeaders);
        if (!await secretEquals(bearer(request), env.ADMIN_TOKEN)) return json({ error: 'Unauthorized.' }, { status: 401 }, corsHeaders);
        const length = Number(request.headers.get('content-length') ?? 0);
        if (!Number.isFinite(length) || length > 16_384) return json({ error: 'Payload too large.' }, { status: 413 }, corsHeaders);
        const rawText = await request.text();
        if (rawText.length > 16_384) return json({ error: 'Payload too large.' }, { status: 413 }, corsHeaders);
        const raw = JSON.parse(rawText) as unknown;
        if (!validStart(raw)) return json({ error: 'Invalid start configuration.' }, { status: 400 }, corsHeaders);

        const mode = raw.mode;
        const targetRounds = mode === 'rehearsal' ? 1 : COMPETITION_ROUNDS;
        const minimumRounds = mode === 'rehearsal' ? 1 : MINIMUM_ROUNDS;
        const guaranteedCost = minimumRounds * ROUND_AUTHORIZATION_USD;
        const spent = await globalSpend(env.DB);
        if (spent + guaranteedCost > BUDGET_USD - RESERVE_USD) {
          return json({ error: 'Insufficient remaining API budget to guarantee the minimum run.' }, { status: 409 }, corsHeaders);
        }

        const runId = `${mode}-${crypto.randomUUID()}`;
        const params: RunParams = {
          runId,
          mode,
          targetRounds,
          minimumRounds,
          phaseMinutes: mode === 'rehearsal' ? 1 : PHASE_MINUTES,
          budgetUsd: BUDGET_USD,
          reserveUsd: RESERVE_USD,
        };
        const state = initialPublicState(params);
        state.spentUsd = spent;
        const created = nowIso();

        try {
          await env.DB.prepare(`INSERT INTO runs(id,workflow_id,status,mode,phase,round,target_rounds,minimum_rounds,phase_minutes,phase_ends_at,started_at,budget_usd,reserve_usd,spent_usd,calls,best_support_json,best_label,best_verified,sota_improved,public_state_json,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)`)
            .bind(runId, runId, 'running', mode, 'ribbon', 0, targetRounds, minimumRounds, params.phaseMinutes, state.phaseEndsAt, state.startedAt, BUDGET_USD, RESERVE_USD, spent, 0, JSON.stringify(state.bestSupport), state.bestLabel, 0, 0, JSON.stringify(state), created, created)
            .run();
        } catch {
          return json({ error: 'An experiment is already running.' }, { status: 409 }, corsHeaders);
        }

        try {
          const instance = await env.AUTOLABS_WORKFLOW.create({
            id: runId,
            params,
            retention: { successRetention: '3 days', errorRetention: '3 days' },
          });
          return json({ accepted: true, runId, workflowId: instance.id, mode, targetRounds, minimumRounds, phaseMinutes: params.phaseMinutes, budgetUsd: BUDGET_USD }, { status: 202 }, corsHeaders);
        } catch (error) {
          const failedAt = nowIso();
          const failedState = { ...state, phase: 'error', phaseEndsAt: null };
          await env.DB.prepare(`UPDATE runs SET status='error',phase='error',phase_ends_at=NULL,completed_at=?,public_state_json=?,updated_at=? WHERE id=?`)
            .bind(failedAt, JSON.stringify(failedState), failedAt, runId)
            .run();
          throw error;
        }
      }

      if (request.method === 'POST' && url.pathname === '/api/jobs/result') {
        const length = Number(request.headers.get('content-length') ?? 0);
        if (!Number.isFinite(length) || length > 1_000_000) return json({ error: 'Payload too large.' }, { status: 413 }, corsHeaders);
        const rawBody = await request.text();
        if (new TextEncoder().encode(rawBody).byteLength > 1_000_000) return json({ error: 'Payload too large.' }, { status: 413 }, corsHeaders);
        const timestamp = request.headers.get('x-autolabs-timestamp') ?? '';
        const signature = request.headers.get('x-autolabs-signature') ?? '';
        if (!env.CALLBACK_SECRET || !await verifyCallbackSignature({ secret: env.CALLBACK_SECRET, timestamp, signature, body: rawBody })) {
          return json({ error: 'Unauthorized.' }, { status: 401 }, corsHeaders);
        }

        const body = JSON.parse(rawBody) as { id?: string; ok?: boolean; complete?: boolean; result?: unknown; error?: string };
        if (!body.id || typeof body.ok !== 'boolean' || typeof body.complete !== 'boolean') {
          return json({ error: 'Invalid job result.' }, { status: 400 }, corsHeaders);
        }
        const row = await env.DB.prepare('SELECT id,run_id,round,agent_id,job_type,status FROM jobs WHERE id=?')
          .bind(body.id)
          .first<{ id: string; run_id: string; round: number; agent_id: AgentId; job_type: string; status: string }>();
        if (!row) return json({ error: 'Unknown job.' }, { status: 404 }, corsHeaders);
        if (row.status === 'queued' || row.status === 'dispatching') return json({ error: 'Job lease is not ready for a result.' }, { status: 425 }, corsHeaders);
        if (row.status !== 'running') return json({ error: 'Job result already finalized.' }, { status: 409 }, corsHeaders);

        const status = body.ok ? (body.complete ? 'complete' : 'partial') : 'failed';
        const update = await env.DB.prepare(`UPDATE jobs SET status=?,result_json=?,error=?,completed_at=? WHERE id=? AND status='running'`)
          .bind(status, JSON.stringify(body.result ?? null), body.error ?? null, nowIso(), body.id)
          .run();
        if (Number(update.meta.changes ?? 0) !== 1) return json({ error: 'Job result already finalized.' }, { status: 409 }, corsHeaders);

        await env.DB.prepare(`INSERT OR IGNORE INTO events(run_id,seq,at,round,phase,agent_id,kind,title,summary,payload_json,visible) VALUES(?,?,?,?,?,?,?,?,?,?,1)`)
          .bind(
            row.run_id,
            callbackSequence(row.round, row.agent_id, row.id),
            nowIso(),
            row.round,
            'research',
            row.agent_id,
            body.ok ? 'tool' : 'error',
            `${row.job_type} ${status}`,
            body.ok ? (body.complete ? 'The exact code-job result is available to its agent next round.' : 'The bounded search returned partial evidence and is not an exhaustion certificate.') : body.error ?? 'Code job failed.',
            JSON.stringify({ id: body.id, status, result: body.result ?? null, error: body.error ?? null }),
          )
          .run();
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
