import { AGENTS } from './agents';
import type { AgentId, Phase, PublicEvent, RunParams, Usage } from './types';
import { costUsd } from './openai';

export function nowIso() { return new Date().toISOString(); }

export function publicAgent(id: AgentId, status = 'ready', bubble = 'Ready at the chalkboard.') {
  const profile = AGENTS.find((agent) => agent.id === id)!;
  const visuals: Record<AgentId, { outfit: string; accent: string; tools: string[] }> = {
    mira: { outfit: 'Amber survey cape', accent: '#ffc76b', tools: ['family_mapper', 'exact_verify'] },
    pip: { outfit: 'Moss utility vest', accent: '#9ee493', tools: ['divisor_completion', 'factor_sieve'] },
    orum: { outfit: 'Plum academic sash', accent: '#d8a8ff', tools: ['saturation_probe', 'certificate_builder'] },
    solvi: { outfit: 'Cyan raincoat', accent: '#7de5ef', tools: ['symmetry_reduce', 'curve_sampler'] },
    tess: { outfit: 'Red knit cap', accent: '#ff8d7b', tools: ['frontier_scheduler', 'exact_verify'] },
  };
  return { id, name: profile.name, epithet: profile.epithet, outfit: visuals[id].outfit, accent: visuals[id].accent, status, bubble, project: profile.prizeProject, approach: profile.style, bestSupport: [0, 0, 0, 0, 0], citations: 0, tools: visuals[id].tools };
}

export function initialPublicState(params: RunParams) {
  return {
    id: params.runId, mode: params.mode, phase: 'ribbon', round: 0,
    targetRounds: params.targetRounds, minimumRounds: params.minimumRounds,
    phaseEndsAt: new Date(Date.now() + 20_000).toISOString(), spentUsd: 0,
    budgetUsd: params.budgetUsd, reserveUsd: params.reserveUsd, calls: 0,
    bestSupport: [0, 0, 0, 0, 0], bestLabel: 'No candidate verified yet', bestVerified: false,
    sotaImproved: false, startedAt: nowIso(), agents: AGENTS.map((agent) => publicAgent(agent.id)), events: [],
  };
}

export async function getState(db: D1Database, runId: string) {
  const row = await db.prepare('SELECT public_state_json FROM runs WHERE id=?').bind(runId).first<{ public_state_json: string }>();
  if (!row) throw new Error(`Run ${runId} does not exist.`);
  return JSON.parse(row.public_state_json) as Record<string, unknown>;
}

export async function patchState(db: D1Database, runId: string, patch: Record<string, unknown>) {
  const state = await getState(db, runId);
  const next = { ...state, ...patch };
  await db.prepare('UPDATE runs SET public_state_json=?, phase=?, round=?, phase_ends_at=?, updated_at=? WHERE id=?')
    .bind(JSON.stringify(next), String(next.phase), Number(next.round), next.phaseEndsAt ?? null, nowIso(), runId).run();
  return next;
}

export async function addEvent(db: D1Database, runId: string, seq: number, event: PublicEvent) {
  await db.prepare(`INSERT OR IGNORE INTO events(run_id,seq,at,round,phase,agent_id,kind,title,summary,payload_json,visible) VALUES(?,?,?,?,?,?,?,?,?,?,?)`)
    .bind(runId, seq, event.at, event.round, event.phase, event.agentId ?? null, event.kind, event.title, event.summary, JSON.stringify(event.payload ?? {}), event.visible ? 1 : 0).run();
}

export async function recordUsage(db: D1Database, runId: string, agentId: AgentId, round: number, phase: Phase, responseId: string, usage: Usage) {
  const price = costUsd(usage);
  await db.prepare(`INSERT OR IGNORE INTO usage(response_id,run_id,agent_id,round,phase,input_tokens,cached_input_tokens,output_tokens,cost_usd,at) VALUES(?,?,?,?,?,?,?,?,?,?)`)
    .bind(responseId, runId, agentId, round, phase, usage.inputTokens, usage.cachedInputTokens, usage.outputTokens, price, nowIso()).run();
  const total = await db.prepare('SELECT COALESCE(SUM(cost_usd),0) AS spent, COUNT(*) AS calls FROM usage').first<{ spent: number; calls: number }>();
  const spent = total?.spent ?? 0;
  const calls = total?.calls ?? 0;
  const state = await getState(db, runId);
  await patchState(db, runId, { spentUsd: spent, calls, budgetUsd: state.budgetUsd });
  await db.prepare('UPDATE runs SET spent_usd=?, calls=? WHERE id=?').bind(spent, calls, runId).run();
  return { price, spent, calls };
}

export async function globalSpend(db: D1Database) {
  const row = await db.prepare('SELECT COALESCE(SUM(cost_usd),0) AS spent FROM usage').first<{ spent: number }>();
  return row?.spent ?? 0;
}

export async function recentEvents(db: D1Database, runId: string, limit = 400) {
  const result = await db.prepare(`SELECT seq,at,round,phase,agent_id AS agentId,kind,title,summary,visible FROM events WHERE run_id=? AND visible=1 ORDER BY seq DESC LIMIT ?`).bind(runId, limit).all();
  return result.results.reverse();
}
