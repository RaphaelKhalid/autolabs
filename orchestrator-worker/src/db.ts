import { AGENTS } from './agents';
import type { AgentId, Phase, PublicEvent, RunParams, Usage } from './types';
import type { ExaRetrieval } from './exa';
import { costUsd } from './openai';

export function nowIso() {
  return new Date().toISOString();
}

export function publicAgent(id: AgentId, status = 'ready', bubble = 'Dormant geometry; waiting for the ribbon.') {
  const profile = AGENTS.find((agent) => agent.id === id)!;
  const visuals: Record<AgentId, { outfit: string; accent: string; tools: string[] }> = {
    mira: { outfit: 'Folded amber manifold', accent: '#9a4f36', tools: ['family_mapper', 'exact_verify'] },
    pip: { outfit: 'Moss divisor bloom', accent: '#657a4e', tools: ['divisor_completion', 'factor_sieve'] },
    orum: { outfit: 'Violet counter-knot', accent: '#60528b', tools: ['saturation_probe', 'certificate_builder'] },
    solvi: { outfit: 'Cyan refractive membrane', accent: '#27747a', tools: ['symmetry_reduce', 'curve_sampler'] },
    tess: { outfit: 'Crimson eclipse filament', accent: '#a14f59', tools: ['frontier_scheduler', 'exact_verify'] },
  };
  return {
    id,
    name: profile.name,
    epithet: profile.epithet,
    outfit: visuals[id].outfit,
    accent: visuals[id].accent,
    status,
    bubble,
    project: profile.prizeProject,
    approach: profile.style,
    bestSupport: [0, 0, 0, 0, 0],
    bestShape: [0, 0],
    bestMetric: [],
    citations: 0,
    tools: visuals[id].tools,
  };
}

export function initialPublicState(params: RunParams) {
  return {
    id: params.runId,
    mode: params.mode,
    phase: 'ribbon',
    round: 0,
    targetRounds: params.targetRounds,
    minimumRounds: params.minimumRounds,
    phaseEndsAt: new Date(Date.now() + 20_000).toISOString(),
    spentUsd: 0,
    budgetUsd: params.budgetUsd,
    reserveUsd: params.reserveUsd,
    calls: 0,
    exaSpentUsd: 0,
    exaBudgetUsd: 40,
    jobs: [],
    bestSupport: [0, 0, 0, 0, 0],
    bestLabel: 'No candidate verified yet',
    bestVerified: false,
    sotaImproved: false,
    startedAt: nowIso(),
    agents: AGENTS.map((agent) => publicAgent(agent.id)),
    events: [],
  };
}

export async function getState(db: D1Database, runId: string) {
  const row = await db.prepare('SELECT public_state_json FROM runs WHERE id=?')
    .bind(runId)
    .first<{ public_state_json: string }>();
  if (!row) throw new Error(`Run ${runId} does not exist.`);
  return JSON.parse(row.public_state_json) as Record<string, unknown>;
}

export async function patchState(db: D1Database, runId: string, patch: Record<string, unknown>) {
  const state = await getState(db, runId);
  const next = { ...state, ...patch };
  await db.prepare('UPDATE runs SET public_state_json=?, phase=?, round=?, phase_ends_at=?, updated_at=? WHERE id=?')
    .bind(JSON.stringify(next), String(next.phase), Number(next.round), next.phaseEndsAt ?? null, nowIso(), runId)
    .run();
  return next;
}

function eventStatement(db: D1Database, runId: string, seq: number, event: PublicEvent) {
  return db.prepare(`INSERT OR IGNORE INTO events(run_id,seq,at,round,phase,agent_id,kind,title,summary,payload_json,visible) VALUES(?,?,?,?,?,?,?,?,?,?,?)`)
    .bind(runId, seq, event.at, event.round, event.phase, event.agentId ?? null, event.kind, event.title, event.summary, JSON.stringify(event.payload ?? {}), event.visible ? 1 : 0);
}

export async function addEvent(db: D1Database, runId: string, seq: number, event: PublicEvent) {
  await eventStatement(db, runId, seq, event).run();
}

export async function addEvents(
  db: D1Database,
  runId: string,
  events: { seq: number; event: PublicEvent }[],
) {
  if (!events.length) return;
  await db.batch(events.map(({ seq, event }) => eventStatement(db, runId, seq, event)));
}

export async function addEventsAndPatchState(
  db: D1Database,
  runId: string,
  events: { seq: number; event: PublicEvent }[],
  patch: Record<string, unknown>,
) {
  const state = await getState(db, runId);
  const next = { ...state, ...patch };
  await db.batch([
    ...events.map(({ seq, event }) => eventStatement(db, runId, seq, event)),
    db.prepare('UPDATE runs SET public_state_json=?, phase=?, round=?, phase_ends_at=?, updated_at=? WHERE id=?')
      .bind(JSON.stringify(next), String(next.phase), Number(next.round), next.phaseEndsAt ?? null, nowIso(), runId),
  ]);
  return next;
}

export async function recordUsage(
  db: D1Database,
  runId: string,
  agentId: AgentId,
  round: number,
  phase: Phase,
  responseId: string,
  usage: Usage,
) {
  const price = costUsd(usage);
  await db.prepare(`INSERT OR IGNORE INTO usage(response_id,run_id,agent_id,round,phase,input_tokens,cached_input_tokens,output_tokens,cost_usd,at) VALUES(?,?,?,?,?,?,?,?,?,?)`)
    .bind(responseId, runId, agentId, round, phase, usage.inputTokens, usage.cachedInputTokens, usage.outputTokens, price, nowIso())
    .run();
  const total = await db.prepare('SELECT COALESCE(SUM(cost_usd),0) AS spent, COUNT(*) AS calls FROM usage')
    .first<{ spent: number; calls: number }>();
  const spent = total?.spent ?? 0;
  const calls = total?.calls ?? 0;
  const state = await getState(db, runId);
  await patchState(db, runId, { spentUsd: spent, calls, budgetUsd: state.budgetUsd });
  await db.prepare('UPDATE runs SET spent_usd=?, calls=? WHERE id=?').bind(spent, calls, runId).run();
  return { price, spent, calls };
}

export async function globalSpend(db: D1Database) {
  const row = await db.prepare('SELECT COALESCE(SUM(cost_usd),0) AS spent FROM usage')
    .first<{ spent: number }>();
  return row?.spent ?? 0;
}

export async function recentEvents(db: D1Database, runId: string, limit = 250, beforeSeq?: number): Promise<Array<PublicEvent & { seq: number }>> {
  const boundedLimit = Math.max(1, Math.min(5_000, Math.floor(limit)));
  const statement = beforeSeq === undefined
    ? db.prepare(`SELECT seq,at,round,phase,agent_id AS agentId,kind,title,summary,payload_json AS payloadJson,visible FROM events WHERE run_id=? AND visible=1 ORDER BY seq DESC LIMIT ?`)
      .bind(runId, boundedLimit)
    : db.prepare(`SELECT seq,at,round,phase,agent_id AS agentId,kind,title,summary,payload_json AS payloadJson,visible FROM events WHERE run_id=? AND visible=1 AND seq<? ORDER BY seq DESC LIMIT ?`)
      .bind(runId, beforeSeq, boundedLimit);
  const result = await statement.all<{
    seq: number;
    at: string;
    round: number;
    phase: Phase;
    agentId?: AgentId;
    kind: PublicEvent['kind'];
    title: string;
    summary: string;
    payloadJson: string;
    visible: number;
  }>();
  return result.results.reverse().map(({ payloadJson, ...event }) => {
    try {
      return { ...event, visible: Boolean(event.visible), payload: JSON.parse(payloadJson) as unknown };
    } catch {
      return { ...event, visible: Boolean(event.visible), payload: {} };
    }
  });
}

export async function globalExaSpend(db: D1Database) {
  const row = await db.prepare('SELECT COALESCE(SUM(cost_usd),0) AS spent FROM exa_usage')
    .first<{ spent: number }>();
  return row?.spent ?? 0;
}

export async function recordExaBatch(db: D1Database, runId: string, round: number, retrievals: ExaRetrieval[]) {
  const accepted = retrievals.filter((item) => item.requestId);
  if (accepted.length) {
    await db.batch(accepted.map((item) => db.prepare(`INSERT OR IGNORE INTO exa_usage(request_id,run_id,agent_id,round,query,cost_usd,sources_json,at) VALUES(?,?,?,?,?,?,?,?)`)
      .bind(item.requestId!, runId, item.agentId, round, item.query, item.costUsd, JSON.stringify(item.sources), nowIso())));
  }
  const spent = await globalExaSpend(db);
  await patchState(db, runId, { exaSpentUsd: spent, exaBudgetUsd: 40 });
  return spent;
}

export async function publicJobs(db: D1Database, runId: string) {
  const rows = await db.prepare(`SELECT id,agent_id AS agentId,round,job_type AS jobType,params_json AS params,status,result_json AS result,error,created_at AS createdAt,completed_at AS completedAt
      FROM jobs WHERE run_id=? ORDER BY created_at DESC LIMIT 100`)
    .bind(runId)
    .all<{ id: string; agentId: string; round: number; jobType: string; params: string; status: string; result: string | null; error: string | null; createdAt: string; completedAt: string | null }>();
  return rows.results.map((job) => ({
    ...job,
    params: JSON.parse(job.params) as unknown,
    result: job.result ? JSON.parse(job.result) as unknown : null,
  }));
}
