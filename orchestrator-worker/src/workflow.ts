import { WorkflowEntrypoint, type WorkflowEvent, type WorkflowStep } from 'cloudflare:workers';
import { AGENTS } from './agents';
import { addEvent, getState, globalSpend, nowIso, patchState, recordUsage } from './db';
import { scheduleJobs } from './github-jobs';
import { callStructured } from './openai';
import { MEETING_SCHEMA, RESEARCH_SCHEMA, meetingPrompt, researchPrompt } from './prompts';
import type { AgentId, AgentResult, MeetingReport, ResearchReport, RunParams } from './types';
import { compareSupport, verifyRectangle, type RectangleCheck } from './verifier';

const AGENT_INDEX: Record<AgentId, number> = { mira: 0, pip: 1, orum: 2, solvi: 3, tess: 4 };
const RESEARCH_BATCH_RESERVE = 0.25;
const MEETING_BATCH_RESERVE = 0.15;

async function agentMemory(db: D1Database, runId: string, agentId: AgentId) {
  const row = await db.prepare('SELECT public_summary_json,private_plan_json FROM agent_memory WHERE run_id=? AND agent_id=?')
    .bind(runId, agentId).first<{ public_summary_json: string; private_plan_json: string }>();
  return row ? { publicSummary: JSON.parse(row.public_summary_json), privatePlan: JSON.parse(row.private_plan_json) } : {};
}

async function completedJobs(db: D1Database, runId: string, agentId: AgentId) {
  const rows = await db.prepare(`SELECT id,job_type AS jobType,result_json AS result FROM jobs WHERE run_id=? AND agent_id=? AND status='complete' ORDER BY completed_at DESC LIMIT 8`)
    .bind(runId, agentId).all<{ id: string; jobType: string; result: string }>();
  return rows.results.map((row) => ({ id: row.id, jobType: row.jobType, result: row.result ? JSON.parse(row.result) : null }));
}

async function researchOne(env: Env, params: RunParams, round: number, agentId: AgentId): Promise<AgentResult<ResearchReport>> {
  const profile = AGENTS[AGENT_INDEX[agentId]];
  const prompt = researchPrompt(profile, round, await agentMemory(env.DB, params.runId, agentId), await completedJobs(env.DB, params.runId, agentId));
  let result = await callStructured<ResearchReport>({ apiKey: env.OPENAI_API_KEY, model: env.MODEL_NAME, agentId, ...prompt, schemaName: 'erdos_885_research', schema: RESEARCH_SCHEMA, maxOutputTokens: 5000, webSearch: true });
  if (!result.ok) {
    result = await callStructured<ResearchReport>({ apiKey: env.OPENAI_API_KEY, model: env.MODEL_NAME, agentId, system: prompt.system, user: `${prompt.user}\nYour previous call failed. Return a smaller valid report now.`, schemaName: 'erdos_885_research_retry', schema: RESEARCH_SCHEMA, maxOutputTokens: 3500, webSearch: false });
  }
  return result;
}

async function meetingOne(env: Env, round: number, agentId: AgentId, reports: unknown): Promise<AgentResult<MeetingReport>> {
  const profile = AGENTS[AGENT_INDEX[agentId]];
  const prompt = meetingPrompt(profile, round, reports);
  let result = await callStructured<MeetingReport>({ apiKey: env.OPENAI_API_KEY, model: env.MODEL_NAME, agentId, ...prompt, schemaName: 'erdos_885_meeting', schema: MEETING_SCHEMA, maxOutputTokens: 2500, webSearch: false });
  if (!result.ok) {
    result = await callStructured<MeetingReport>({ apiKey: env.OPENAI_API_KEY, model: env.MODEL_NAME, agentId, system: prompt.system, user: `${prompt.user}\nReturn a shorter valid reaction and private plan.`, schemaName: 'erdos_885_meeting_retry', schema: MEETING_SCHEMA, maxOutputTokens: 1800, webSearch: false });
  }
  return result;
}

function agentCards(state: Record<string, unknown>, status: string, reports?: AgentResult<ResearchReport>[]) {
  const current = state.agents as Record<string, unknown>[];
  return current.map((agent) => {
    const report = reports?.find((item) => item.agentId === agent.id);
    return { ...agent, status: report?.ok ? status : report ? 'recovering' : status, bubble: report?.value?.headline ?? (report ? 'Call failed; isolating and retrying next round.' : agent.bubble), citations: report?.value?.citations.length ?? agent.citations };
  });
}

function bestRectangle(results: AgentResult<ResearchReport>[]) {
  let best: { agentId: AgentId; check: RectangleCheck; note: string } | undefined;
  for (const result of results) for (const candidate of result.value?.candidates ?? []) {
    const check = verifyRectangle(candidate);
    if (!check.accepted) continue;
    if (!best || compareSupport(check.support, best.check.support) > 0 || (compareSupport(check.support, best.check.support) === 0 && check.totalSupport > best.check.totalSupport)) best = { agentId: result.agentId, check, note: candidate.note };
  }
  return best;
}

async function chargeResults<T extends ResearchReport | MeetingReport>(env: Env, params: RunParams, round: number, phase: 'research' | 'meeting', results: AgentResult<T>[]) {
  for (const result of results) if (result.responseId) await recordUsage(env.DB, params.runId, result.agentId, round, phase, result.responseId, result.usage);
}

export class AutolabsWorkflow extends WorkflowEntrypoint<Env, RunParams> {
  async run(event: WorkflowEvent<RunParams>, step: WorkflowStep) {
    const params = event.payload;
    await step.do('ribbon cutting', async () => {
      await addEvent(this.env.DB, params.runId, 1, { at: nowIso(), round: 0, phase: 'ribbon', kind: 'system', title: 'Ribbon cut', summary: `The ${params.mode === 'rehearsal' ? 'real dress rehearsal' : 'competition'} began. Prompts, prizes, exact rules and budget ceiling are now committed.`, visible: true });
      await patchState(this.env.DB, params.runId, { phase: 'ribbon', phaseEndsAt: new Date(Date.now() + 20_000).toISOString() });
      return { opened: true };
    });
    await step.sleep('researchers enter the cabin', '20 seconds');

    for (let round = 1; round <= params.targetRounds; round += 1) {
      const allowed = await step.do(`budget preflight research ${round}`, async () => {
        const spent = await globalSpend(this.env.DB);
        return spent + RESEARCH_BATCH_RESERVE <= params.budgetUsd - params.reserveUsd;
      });
      if (!allowed) {
        await step.do(`budget stop ${round}`, async () => {
          await patchState(this.env.DB, params.runId, { phase: 'budget-stop', phaseEndsAt: null });
          await this.env.DB.prepare("UPDATE runs SET status='budget-stop',completed_at=?,updated_at=? WHERE id=?" ).bind(nowIso(), nowIso(), params.runId).run();
          await addEvent(this.env.DB, params.runId, round * 1000 + 990, { at: nowIso(), round, phase: 'budget-stop', kind: 'budget', title: 'Budget reserve reached', summary: 'The engine stopped before issuing a call that could breach the $50 ceiling.', visible: true });
        });
        return { status: 'budget-stop', round };
      }

      const researchDeadline = Date.now() + params.phaseMinutes * 60_000;
      await step.do(`open private research ${round}`, async () => {
        const state = await getState(this.env.DB, params.runId);
        await patchState(this.env.DB, params.runId, { phase: 'research', round, phaseEndsAt: new Date(researchDeadline).toISOString(), agents: agentCards(state, 'researching') });
        await addEvent(this.env.DB, params.runId, round * 1000 + 1, { at: nowIso(), round, phase: 'research', kind: 'system', title: `Private research loop ${round}`, summary: 'Five independent research calls opened in sync. Results remain sealed until the round table.', visible: true });
      });

      const research = await step.do(`five research calls ${round}`, { retries: { limit: 2, delay: '10 seconds', backoff: 'linear' }, timeout: '10 minutes' }, async () => {
        const results = await Promise.all(AGENTS.map((agent) => researchOne(this.env, params, round, agent.id)));
        await chargeResults(this.env, params, round, 'research', results);
        for (const result of results) {
          if (!result.ok || !result.value) continue;
          await scheduleJobs({ db: this.env.DB, githubToken: this.env.GITHUB_TOKEN, repository: this.env.GITHUB_REPOSITORY, callbackUrl: this.env.PUBLIC_WORKER_URL, callbackSecret: this.env.CALLBACK_SECRET, runId: params.runId, round, agentId: result.agentId, reports: result.value.proposedJobs });
        }
        const best = bestRectangle(results);
        if (best) {
          const state = await getState(this.env.DB, params.runId);
          const current = state.bestSupport as number[];
          if (compareSupport(best.check.support, current) > 0) await patchState(this.env.DB, params.runId, { bestSupport: best.check.support, bestLabel: best.note, bestVerified: true, sotaImproved: Boolean(state.sotaImproved) || best.check.improvesSota });
          await addEvent(this.env.DB, params.runId, round * 1000 + 100 + AGENT_INDEX[best.agentId] * 20, { at: nowIso(), round, phase: 'research', agentId: best.agentId, kind: 'candidate', title: best.check.isK5 ? 'Eureka: exact k=5 witness' : best.check.improvesSota ? 'Exact SOTA frontier improved' : 'Candidate checked exactly', summary: `${best.note} Support ${JSON.stringify(best.check.support)}; ${best.check.exactCells} exact cells.`, visible: true, payload: best.check });
          if (best.check.isK5) {
            await patchState(this.env.DB, params.runId, { phase: 'eureka', phaseEndsAt: null, bestSupport: best.check.support, bestVerified: true });
          }
        }
        return results;
      });

      const afterResearch = await getState(this.env.DB, params.runId);
      if (afterResearch.phase === 'eureka') {
        await this.env.DB.prepare("UPDATE runs SET status='eureka',completed_at=?,updated_at=? WHERE id=?" ).bind(nowIso(), nowIso(), params.runId).run();
        return { status: 'eureka', round };
      }
      await step.sleepUntil(`finish private research ${round}`, researchDeadline);

      const meetingDeadline = Date.now() + params.phaseMinutes * 60_000;
      const meeting = await step.do(`simultaneous reveal and meeting ${round}`, { retries: { limit: 2, delay: '10 seconds', backoff: 'linear' }, timeout: '10 minutes' }, async () => {
        const publicReports = research.map((result) => ({ agentId: result.agentId, ok: result.ok, report: result.value ?? { headline: 'Agent recovering', thesis: result.error ?? 'Call failed.' } }));
        for (const result of research) {
          const index = AGENT_INDEX[result.agentId];
          await addEvent(this.env.DB, params.runId, round * 1000 + 10 + index, { at: nowIso(), round, phase: 'meeting', agentId: result.agentId, kind: result.ok ? 'research' : 'error', title: result.value?.headline ?? `${AGENTS[index].name} is recovering`, summary: result.value?.thesis ?? result.error ?? 'The failed call was isolated; other agents continue.', visible: true, payload: result.value });
        }
        const state = await getState(this.env.DB, params.runId);
        await patchState(this.env.DB, params.runId, { phase: 'meeting', phaseEndsAt: new Date(meetingDeadline).toISOString(), agents: agentCards(state, 'meeting', research) });
        await addEvent(this.env.DB, params.runId, round * 1000 + 500, { at: nowIso(), round, phase: 'meeting', kind: 'meeting', title: `Round-table reveal ${round}`, summary: 'All five sealed reports were published simultaneously. Each researcher now gets one response.', visible: true });
        const results = await Promise.all(AGENTS.map((agent) => meetingOne(this.env, round, agent.id, publicReports)));
        await chargeResults(this.env, params, round, 'meeting', results);
        for (const result of results) {
          const index = AGENT_INDEX[result.agentId];
          await addEvent(this.env.DB, params.runId, round * 1000 + 510 + index, { at: nowIso(), round, phase: 'meeting', agentId: result.agentId, kind: result.ok ? 'meeting' : 'error', title: `${AGENTS[index].name} reacts`, summary: result.value?.reaction ?? 'Reaction call failed; the agent will rejoin next round.', visible: true, payload: result.value ? { agreements: result.value.agreements, objections: result.value.objections, collaborationCredits: result.value.collaborationCredits } : {} });
          if (result.value) await this.env.DB.prepare(`INSERT INTO agent_memory(run_id,agent_id,public_summary_json,private_plan_json,updated_at) VALUES(?,?,?,?,?) ON CONFLICT(run_id,agent_id) DO UPDATE SET public_summary_json=excluded.public_summary_json,private_plan_json=excluded.private_plan_json,updated_at=excluded.updated_at`)
            .bind(params.runId, result.agentId, JSON.stringify(publicReports.find((item) => item.agentId === result.agentId)?.report ?? {}), JSON.stringify(result.value.privateNextPlan), nowIso()).run();
        }
        return results;
      });
      void meeting;
      await step.sleepUntil(`finish round table ${round}`, meetingDeadline);
    }

    return step.do('publish final report', async () => {
      const state = await getState(this.env.DB, params.runId);
      const privateRows = await this.env.DB.prepare('SELECT agent_id AS agentId,private_plan_json AS privatePlan FROM agent_memory WHERE run_id=?').bind(params.runId).all<{ agentId: string; privatePlan: string }>();
      const report = {
        runId: params.runId,
        mode: params.mode,
        completedAt: nowIso(),
        rounds: params.targetRounds,
        bestSupport: Array.isArray(state.bestSupport) ? state.bestSupport.map(Number) : [],
        bestLabel: String(state.bestLabel ?? ''),
        exactVerified: Boolean(state.bestVerified),
        sotaImproved: Boolean(state.sotaImproved),
        privatePlansReleased: privateRows.results.map((row) => ({ agentId: row.agentId, privatePlan: JSON.parse(row.privatePlan) as { objective?: string; checks?: string[] } })),
        reproducibility: { verifier: 'lib/exact-verifier.ts', eventLedger: `/api/experiments/${params.runId}/events`, codeJobs: 'math-worker/' },
      };
      await this.env.DB.prepare(`UPDATE runs SET status='complete',phase='complete',completed_at=?,report_json=?,updated_at=? WHERE id=?`).bind(report.completedAt, JSON.stringify(report), report.completedAt, params.runId).run();
      await patchState(this.env.DB, params.runId, { phase: 'complete', phaseEndsAt: null, report });
      await addEvent(this.env.DB, params.runId, params.targetRounds * 1000 + 990, { at: nowIso(), round: params.targetRounds, phase: 'complete', kind: 'system', title: 'Experiment complete', summary: 'Budgeted research is complete. The replay, exact best result, programs, citations, failed avenues and private plans are now released.', visible: true });
      return report;
    });
  }
}
