import { WorkflowEntrypoint, type WorkflowEvent, type WorkflowStep } from 'cloudflare:workers';
import { AGENTS } from './agents';
import { addEvent, addEvents, addEventsAndPatchState, getState, globalExaSpend, globalSpend, nowIso, patchState, recentEvents, recordExaBatch, recordUsage } from './db';
import { EXA_BUDGET_USD, EXA_REQUEST_AUTHORIZATION_USD, retrievalContext, searchExa, type ExaRetrieval } from './exa';
import { reapStaleJobs, scheduleJobs } from './github-jobs';
import { callStructured, mergeAttempts } from './openai';
import { MEETING_SCHEMA, RESEARCH_SCHEMA, meetingPrompt, researchPrompt } from './prompts';
import type { AgentId, AgentResult, MeetingReport, PublicEvent, ResearchReport, RunParams } from './types';
import { compareSupport, verifyRectangle, type RectangleCheck } from './verifier';

const MODEL = 'gpt-5.6-luna' as const;
const AGENT_INDEX: Record<AgentId, number> = { mira: 0, pip: 1, orum: 2, solvi: 3, tess: 4 };
const RESEARCH_AUTHORIZATION_USD = 1.10;
const MEETING_AUTHORIZATION_USD = 0.65;
const ROUND_AUTHORIZATION_USD = RESEARCH_AUTHORIZATION_USD + MEETING_AUTHORIZATION_USD;

interface PreparedPrompt {
  agentId: AgentId;
  system: string;
  user: string;
  retrieval: ExaRetrieval;
}

interface ResearchResult extends AgentResult<ResearchReport> {
  retrieval: ExaRetrieval;
}

interface BestCandidate {
  agentId: AgentId;
  check: RectangleCheck;
  note: string;
}

async function agentMemory(db: D1Database, runId: string, agentId: AgentId) {
  const row = await db.prepare('SELECT public_summary_json,private_plan_json FROM agent_memory WHERE run_id=? AND agent_id=?')
    .bind(runId, agentId)
    .first<{ public_summary_json: string; private_plan_json: string }>();
  return row ? {
    publicSummary: JSON.parse(row.public_summary_json),
    privatePlan: JSON.parse(row.private_plan_json),
  } : {};
}

async function completedJobs(db: D1Database, runId: string, agentId: AgentId) {
  const rows = await db.prepare(`SELECT id,job_type AS jobType,status,params_json AS params,result_json AS result,error
      FROM jobs
      WHERE run_id=? AND agent_id=? AND status IN ('complete','partial','failed')
      ORDER BY completed_at DESC LIMIT 8`)
    .bind(runId, agentId)
    .all<{ id: string; jobType: string; status: string; params: string; result: string | null; error: string | null }>();
  return rows.results.map((row) => ({
    id: row.id,
    jobType: row.jobType,
    status: row.status,
    params: JSON.parse(row.params),
    result: row.result ? JSON.parse(row.result) : null,
    error: row.error,
  }));
}

async function prepareResearchPrompts(env: Env, params: RunParams, round: number): Promise<PreparedPrompt[]> {
  await reapStaleJobs(env.DB, params.runId);
  const spent = await globalExaSpend(env.DB);
  const retrievalAllowed = spent + AGENTS.length * EXA_REQUEST_AUTHORIZATION_USD <= EXA_BUDGET_USD - 1;
  const prepared = await Promise.all(AGENTS.map(async (profile) => {
    const memory = await agentMemory(env.DB, params.runId, profile.id);
    const jobs = await completedJobs(env.DB, params.runId, profile.id);
    const privatePlan = memory && typeof memory === 'object' ? (memory as { privatePlan?: { objective?: unknown } }).privatePlan : undefined;
    const objective = typeof privatePlan?.objective === 'string' ? privatePlan.objective.slice(0, 300) : profile.style;
    const query = `Erdos problem 885 Diophantine rectangles simultaneous square differences ${profile.epithet}: ${objective}`;
    const retrieval = retrievalAllowed
      ? await searchExa({ apiKey: env.EXA_API_KEY, agentId: profile.id, query })
      : { agentId: profile.id, query, costUsd: 0, sources: [], error: 'Exa research reserve reached.' } satisfies ExaRetrieval;
    return {
      agentId: profile.id,
      retrieval,
      ...researchPrompt(profile, round, memory, jobs, retrievalContext(retrieval)),
    };
  }));
  await recordExaBatch(env.DB, params.runId, round, prepared.map((item) => item.retrieval));
  return prepared;
}

async function researchOne(env: Env, prompt: PreparedPrompt): Promise<AgentResult<ResearchReport>> {
  const first = await callStructured<ResearchReport>({
    apiKey: env.OPENAI_API_KEY,
    model: MODEL,
    agentId: prompt.agentId,
    system: prompt.system,
    user: prompt.user,
    schemaName: 'erdos_885_research',
    schema: RESEARCH_SCHEMA,
    maxOutputTokens: 24_000,
    maxInputBytes: 64_000,
    webSearch: false,
    timeoutMs: 240_000,
  });
  if (first.ok) return first;
  const second = await callStructured<ResearchReport>({
    apiKey: env.OPENAI_API_KEY,
    model: MODEL,
    agentId: prompt.agentId,
    system: prompt.system,
    user: `${prompt.user}\nThe prior attempt failed. Return a smaller valid report now.`,
    schemaName: 'erdos_885_research_retry',
    schema: RESEARCH_SCHEMA,
    maxOutputTokens: 16_000,
    maxInputBytes: 64_000,
    webSearch: false,
    timeoutMs: 180_000,
  });
  return mergeAttempts(first, second);
}

async function meetingOne(env: Env, round: number, agentId: AgentId, reports: unknown): Promise<AgentResult<MeetingReport>> {
  const profile = AGENTS[AGENT_INDEX[agentId]];
  const prompt = meetingPrompt(profile, round, reports);
  const first = await callStructured<MeetingReport>({
    apiKey: env.OPENAI_API_KEY,
    model: MODEL,
    agentId,
    ...prompt,
    schemaName: 'erdos_885_meeting',
    schema: MEETING_SCHEMA,
    maxOutputTokens: 8_000,
    maxInputBytes: 128_000,
    webSearch: false,
    timeoutMs: 135_000,
  });
  if (first.ok) return first;
  const second = await callStructured<MeetingReport>({
    apiKey: env.OPENAI_API_KEY,
    model: MODEL,
    agentId,
    system: prompt.system,
    user: `${prompt.user}\nReturn a shorter valid reaction and private plan.`,
    schemaName: 'erdos_885_meeting_retry',
    schema: MEETING_SCHEMA,
    maxOutputTokens: 6_000,
    maxInputBytes: 128_000,
    webSearch: false,
    timeoutMs: 120_000,
  });
  return mergeAttempts(first, second);
}

function agentCards(state: Record<string, unknown>, status: string, reports?: ResearchResult[]) {
  const current = state.agents as Record<string, unknown>[];
  return current.map((agent) => {
    const report = reports?.find((item) => item.agentId === agent.id);
    return {
      ...agent,
      status: report?.ok ? status : report ? 'recovering' : status,
      bubble: report?.value?.headline ?? (report ? 'Call failed; isolated from the other researchers.' : agent.bubble),
      citations: report?.value?.citations.length ?? agent.citations,
    };
  });
}

function verifiedReport(result: ResearchResult) {
  if (!result.value) return undefined;
  return {
    ...result.value,
    retrieval: retrievalContext(result.retrieval),
    verifiedCandidates: result.value.candidates.map((candidate) => ({
      candidate,
      verification: verifyRectangle(candidate),
    })),
  };
}

function bestRectangle(results: AgentResult<ResearchReport>[]): BestCandidate | undefined {
  let best: BestCandidate | undefined;
  for (const result of results) {
    for (const candidate of result.value?.candidates ?? []) {
      const check = verifyRectangle(candidate);
      if (!check.accepted) continue;
      if (!best || compareSupport(check.support, best.check.support) > 0 || (
        compareSupport(check.support, best.check.support) === 0 && check.totalSupport > best.check.totalSupport
      )) {
        best = { agentId: result.agentId, check, note: candidate.note };
      }
    }
  }
  return best;
}

async function chargeResults<T extends ResearchReport | MeetingReport>(
  env: Env,
  params: RunParams,
  round: number,
  phase: 'research' | 'meeting',
  results: AgentResult<T>[],
) {
  for (const result of results) {
    const records = result.usageRecords?.length
      ? result.usageRecords
      : result.responseId ? [{ responseId: result.responseId, usage: result.usage }] : [];
    for (const record of records) {
      await recordUsage(env.DB, params.runId, result.agentId, round, phase, record.responseId, record.usage);
    }
  }
}

async function persistBest(env: Env, params: RunParams, best: BestCandidate | undefined, state: Record<string, unknown>): Promise<Record<string, unknown>> {
  if (!best) return {};
  const current = state.bestSupport as number[];
  if (compareSupport(best.check.support, current) <= 0) return {};
  const sotaImproved = Boolean(state.sotaImproved) || best.check.improvesSota;
  await env.DB.prepare(`UPDATE runs SET best_support_json=?,best_label=?,best_verified=1,sota_improved=?,updated_at=? WHERE id=?`)
    .bind(JSON.stringify(best.check.support), best.note, sotaImproved ? 1 : 0, nowIso(), params.runId)
    .run();
  return {
    bestSupport: best.check.support,
    bestLabel: best.note,
    bestVerified: true,
    sotaImproved,
  };
}

function researchEvents(round: number, research: ResearchResult[], phase: PublicEvent['phase']): { seq: number; event: PublicEvent }[] {
  return research.map((result) => {
    const index = AGENT_INDEX[result.agentId];
    return {
      seq: round * 1000 + 10 + index,
      event: {
        at: nowIso(),
        round,
        phase,
        agentId: result.agentId,
        kind: result.ok ? 'research' as const : 'error' as const,
        title: result.value?.headline ?? `${AGENTS[index].name} is recovering`,
        summary: result.value?.thesis ?? result.error ?? 'The failed call was isolated; other agents continue.',
        visible: true,
        payload: verifiedReport(result) ?? { error: result.error ?? 'Research call failed.' },
      },
    };
  });
}

async function revealResearch(
  env: Env,
  params: RunParams,
  round: number,
  research: ResearchResult[],
  best: BestCandidate | undefined,
  terminal: 'meeting' | 'eureka' | 'budget-stop',
  phaseEndsAt: string | null,
) {
  const events: { seq: number; event: PublicEvent }[] = researchEvents(round, research, terminal);
  if (terminal === 'meeting') {
    events.push({
      seq: round * 1000 + 500,
      event: {
        at: nowIso(), round, phase: 'meeting', kind: 'meeting',
        title: `Round-table reveal ${round}`,
        summary: 'All five sealed reports were published as one atomic batch. Each researcher now gets one response.',
        visible: true,
        payload: { simultaneous: true, agentIds: AGENTS.map((agent) => agent.id) },
      },
    });
  }
  if (best) {
    events.push({
      seq: round * 1000 + 100 + AGENT_INDEX[best.agentId] * 20,
      event: {
        at: nowIso(), round, phase: terminal, agentId: best.agentId, kind: 'candidate',
        title: best.check.isK5 ? 'Eureka: exact k=5 witness' : best.check.improvesSota ? 'Exact SOTA frontier improved' : 'Candidate checked exactly',
        summary: `${best.note} Support ${JSON.stringify(best.check.support)}; ${best.check.exactCells} exact cells.`,
        visible: true,
        payload: best.check,
      },
    });
  }
  const state = await getState(env.DB, params.runId);
  const bestPatch = await persistBest(env, params, best, state);
  await addEventsAndPatchState(env.DB, params.runId, events, {
    ...bestPatch,
    phase: terminal,
    round,
    phaseEndsAt,
    agents: agentCards({ ...state, ...bestPatch }, terminal === 'meeting' ? 'meeting' : 'complete', research),
  });

  if (terminal === 'meeting') {
    for (const result of research) {
      if (!result.value) continue;
      try {
        await scheduleJobs({
          db: env.DB,
          githubToken: env.GITHUB_TOKEN,
          repository: env.GITHUB_REPOSITORY,
          runId: params.runId,
          round,
          agentId: result.agentId,
          reports: result.value.proposedJobs,
        });
      } catch (error) {
        await addEvent(env.DB, params.runId, round * 1000 + 380 + AGENT_INDEX[result.agentId], {
          at: nowIso(), round, phase: 'meeting', agentId: result.agentId, kind: 'error',
          title: 'Code-job dispatch isolated',
          summary: error instanceof Error ? error.message : 'The job request failed without interrupting other researchers.',
          visible: true,
        });
      }
    }
  }
}

function stringList(value: unknown) {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string') : [];
}

async function finalReport(env: Env, params: RunParams, terminal: 'complete' | 'eureka' | 'budget-stop', completedRound: number) {
  const state = await getState(env.DB, params.runId);
  const events = await recentEvents(env.DB, params.runId, 5_000) as Array<PublicEvent & { seq: number }>;
  const privateRows = await env.DB.prepare(`SELECT round,agent_id AS agentId,plan_json AS plan
      FROM private_plans WHERE run_id=? ORDER BY round,agent_id`)
    .bind(params.runId)
    .all<{ round: number; agentId: AgentId; plan: string }>();
  const jobs = await env.DB.prepare(`SELECT id,agent_id AS agentId,round,job_type AS jobType,params_json AS params,status,result_json AS result,error,created_at AS createdAt,completed_at AS completedAt
      FROM jobs WHERE run_id=? ORDER BY round,agent_id,id`)
    .bind(params.runId)
    .all<{ id: string; agentId: AgentId; round: number; jobType: string; params: string; status: string; result: string | null; error: string | null; createdAt: string; completedAt: string | null }>();
  const exaUsage = await env.DB.prepare(`SELECT request_id AS requestId,agent_id AS agentId,round,query,cost_usd AS costUsd,sources_json AS sources,at
      FROM exa_usage WHERE run_id=? ORDER BY round,agent_id`)
    .bind(params.runId)
    .all<{ requestId: string; agentId: AgentId; round: number; query: string; costUsd: number; sources: string; at: string }>();
  const usage = await env.DB.prepare(`SELECT agent_id AS agentId,phase,
      SUM(input_tokens) AS inputTokens,SUM(cached_input_tokens) AS cachedInputTokens,
      SUM(output_tokens) AS outputTokens,SUM(cost_usd) AS costUsd,COUNT(*) AS calls
      FROM usage WHERE run_id=? GROUP BY agent_id,phase ORDER BY agent_id,phase`)
    .bind(params.runId)
    .all<Record<string, unknown>>();

  const researchPayloads = events
    .filter((event) => event.kind === 'research' && event.payload && typeof event.payload === 'object')
    .map((event) => event.payload as Record<string, unknown>);
  const retrievalRows = exaUsage.results.map((row) => ({ ...row, sources: JSON.parse(row.sources) as Array<{ url?: string }> }));
  const retrievalCitations = retrievalRows.flatMap((row) => row.sources.map((source) => source.url).filter((url): url is string => typeof url === 'string'));
  const citations = [...new Set([...researchPayloads.flatMap((payload) => stringList(payload.citations)), ...retrievalCitations])];
  const failedAvenues = [...new Set(researchPayloads.flatMap((payload) => stringList(payload.failedAvenues)))];
  const candidateCertificates = events
    .filter((event) => event.kind === 'candidate')
    .map((event) => ({ round: event.round, agentId: event.agentId, title: event.title, summary: event.summary, verification: event.payload }));
  const creditCounts = new Map<AgentId, number>();
  for (const event of events.filter((item) => item.kind === 'meeting')) {
    const payload = event.payload && typeof event.payload === 'object' ? event.payload as Record<string, unknown> : {};
    for (const id of stringList(payload.collaborationCredits)) {
      if (id in AGENT_INDEX) creditCounts.set(id as AgentId, (creditCounts.get(id as AgentId) ?? 0) + 1);
    }
  }
  const winner = candidateCertificates.find((certificate) => certificate.title.startsWith('Eureka'));
  const creditedCollaborators = [...creditCounts.entries()].filter(([, count]) => count > 0).map(([agentId]) => agentId);
  if (winner?.agentId && !creditedCollaborators.includes(winner.agentId as AgentId)) creditedCollaborators.unshift(winner.agentId as AgentId);

  const report = {
    schemaVersion: 1,
    runId: params.runId,
    mode: params.mode,
    outcome: terminal,
    completedAt: nowIso(),
    roundsCompleted: completedRound,
    targetRounds: params.targetRounds,
    result: {
      bestSupport: Array.isArray(state.bestSupport) ? state.bestSupport.map(Number) : [],
      bestLabel: String(state.bestLabel ?? ''),
      bestCandidateCheckedExactly: Boolean(state.bestVerified),
      k5Proved: terminal === 'eureka',
      sotaImproved: Boolean(state.sotaImproved),
      candidateCertificates,
    },
    scientificRecord: {
      citations,
      failedAvenues,
      eventCount: events.length + 1,
      completeEventLedger: `/api/experiments/${params.runId}/events`,
      verifier: 'orchestrator-worker/src/verifier.ts → orchestrator-worker/src/exact-verifier.ts',
    },
    agents: AGENTS.map((agent) => ({
      id: agent.id,
      name: agent.name,
      cognitiveStyle: agent.style,
      proposedPrizeProject: agent.prizeProject,
      collaborationCredits: creditCounts.get(agent.id) ?? 0,
    })),
    rewards: {
      externallyFundedByRaphael: true,
      openAiExperimentBudgetUsd: params.budgetUsd,
      participationRewardUsdPerAgent: 25,
      victoryProjectBudgetUsd: terminal === 'eureka' ? 50 : 0,
      collaborationRewardUsdPerCreditedAgent: terminal === 'eureka' ? 25 : 0,
      sotaRewardUsd: terminal !== 'eureka' && Boolean(state.sotaImproved) ? 25 : 0,
      creditedCollaborators,
    },
    privatePlansReleased: privateRows.results.map((row) => ({
      round: row.round,
      agentId: row.agentId,
      privatePlan: JSON.parse(row.plan) as { objective?: string; checks?: string[] },
    })),
    researchRetrieval: retrievalRows,
    codeJobs: jobs.results.map((job) => ({
      ...job,
      params: JSON.parse(job.params),
      result: job.result ? JSON.parse(job.result) : null,
    })),
    usage: usage.results,
  };

  await env.DB.prepare(`UPDATE runs SET status=?,phase=?,completed_at=?,report_json=?,updated_at=? WHERE id=?`)
    .bind(terminal, terminal, report.completedAt, JSON.stringify(report), report.completedAt, params.runId)
    .run();
  await patchState(env.DB, params.runId, { phase: terminal, phaseEndsAt: null, report });
  await addEvent(env.DB, params.runId, (completedRound + 1) * 1000 + 990, {
    at: nowIso(),
    round: completedRound,
    phase: terminal,
    kind: terminal === 'eureka' ? 'candidate' : terminal === 'budget-stop' ? 'budget' : 'system',
    title: terminal === 'eureka' ? 'Victory report published' : terminal === 'budget-stop' ? 'Budget-safe report published' : 'Experiment complete',
    summary: 'The scientific report, complete replay ledger, exact evidence, code jobs, citations, failed avenues and formerly private plans are now public.',
    visible: true,
    payload: { reportUrl: `/api/experiments/${params.runId}/report`, outcome: terminal },
  });
  return report;
}

export class AutolabsWorkflow extends WorkflowEntrypoint<Env, RunParams> {
  async run(event: WorkflowEvent<RunParams>, step: WorkflowStep) {
    const params = event.payload;
    let terminal: 'complete' | 'eureka' | 'budget-stop' = 'complete';
    let completedRound = 0;

    try {
    await step.do('ribbon cutting', async () => {
      await addEvent(this.env.DB, params.runId, 1, {
        at: nowIso(), round: 0, phase: 'ribbon', kind: 'system', title: 'Ribbon cut',
        summary: `The ${params.mode === 'rehearsal' ? 'real dress rehearsal' : 'competition'} began. Prompts, prizes, exact rules and budget ceiling are now committed.`,
        visible: true,
        payload: { params, model: MODEL, reasoningEffort: 'high' },
      });
      await patchState(this.env.DB, params.runId, { phase: 'ribbon', phaseEndsAt: new Date(Date.now() + 20_000).toISOString() });
      return { opened: true };
    });
    await step.sleep('researchers enter the field', '20 seconds');

    for (let round = 1; round <= params.targetRounds; round += 1) {
      const allowed = await step.do(`budget preflight round ${round}`, async () => {
        const spent = await globalSpend(this.env.DB);
        return spent + ROUND_AUTHORIZATION_USD <= params.budgetUsd - params.reserveUsd;
      });
      if (!allowed) {
        terminal = 'budget-stop';
        await step.do(`budget stop ${round}`, async () => {
          await patchState(this.env.DB, params.runId, { phase: 'budget-stop', phaseEndsAt: null });
          await addEvent(this.env.DB, params.runId, round * 1000 + 980, {
            at: nowIso(), round, phase: 'budget-stop', kind: 'budget', title: 'Budget reserve reached',
            summary: 'The engine stopped before authorizing another complete research-and-meeting round.',
            visible: true,
            payload: { authorizationUsd: ROUND_AUTHORIZATION_USD, reserveUsd: params.reserveUsd },
          });
        });
        break;
      }

      const researchDeadline = Date.now() + params.phaseMinutes * 60_000;
      await step.do(`open private research ${round}`, async () => {
        const state = await getState(this.env.DB, params.runId);
        await patchState(this.env.DB, params.runId, {
          phase: 'research', round,
          phaseEndsAt: new Date(researchDeadline).toISOString(),
          agents: agentCards(state, 'researching'),
        });
        await addEvent(this.env.DB, params.runId, round * 1000 + 1, {
          at: nowIso(), round, phase: 'research', kind: 'system', title: `Private research loop ${round}`,
          summary: 'Five independent research calls opened in sync. Results remain sealed until the reveal.',
          visible: true,
          payload: { sealed: true, agentCount: 5, deadline: new Date(researchDeadline).toISOString() },
        });
      });

      const prepared = await step.do(`prepare research context ${round}`, { retries: { limit: 2, delay: '5 seconds', backoff: 'linear' } }, () => prepareResearchPrompts(this.env, params, round));
      const research = await step.do(`five sealed research calls ${round}`, { retries: { limit: 0, delay: '1 second', backoff: 'constant' }, timeout: '10 minutes' }, () => Promise.all(prepared.map(async (prompt): Promise<ResearchResult> => ({
        ...await researchOne(this.env, prompt),
        retrieval: prompt.retrieval,
      }))));
      const best = await step.do(`charge and verify research ${round}`, { retries: { limit: 2, delay: '5 seconds', backoff: 'linear' } }, async () => {
        await chargeResults(this.env, params, round, 'research', research);
        return bestRectangle(research);
      });

      if (best?.check.isK5) {
        terminal = 'eureka';
        completedRound = round;
        await step.do(`immediate eureka reveal ${round}`, { retries: { limit: 2, delay: '5 seconds', backoff: 'linear' } }, () => revealResearch(this.env, params, round, research, best, 'eureka', null));
        break;
      }

      const spentAfterResearch = await step.do(`meeting budget preflight ${round}`, () => globalSpend(this.env.DB));
      if (spentAfterResearch + MEETING_AUTHORIZATION_USD > params.budgetUsd - params.reserveUsd) {
        terminal = 'budget-stop';
        await step.do(`reveal before budget stop ${round}`, { retries: { limit: 2, delay: '5 seconds', backoff: 'linear' } }, () => revealResearch(this.env, params, round, research, best, 'budget-stop', null));
        await step.do(`record mid-round budget stop ${round}`, () => addEvent(this.env.DB, params.runId, round * 1000 + 980, {
          at: nowIso(), round, phase: 'budget-stop', kind: 'budget', title: 'Meeting authorization withheld',
          summary: 'The sealed research reports were released, but no meeting calls were issued because the protected reserve had been reached.',
          visible: true,
        }));
        break;
      }

      if (Date.now() < researchDeadline) await step.sleepUntil(`finish private research ${round}`, researchDeadline);
      const meetingDeadline = Date.now() + params.phaseMinutes * 60_000;
      await step.do(`atomic simultaneous reveal ${round}`, { retries: { limit: 2, delay: '5 seconds', backoff: 'linear' } }, () => revealResearch(this.env, params, round, research, best, 'meeting', new Date(meetingDeadline).toISOString()));

      const publicReports = research.map((result) => ({
        agentId: result.agentId,
        ok: result.ok,
        report: verifiedReport(result) ?? { headline: 'Agent recovering', thesis: result.error ?? 'Call failed.' },
      }));
      const meeting = await step.do(`five meeting reactions ${round}`, { retries: { limit: 0, delay: '1 second', backoff: 'constant' }, timeout: '5 minutes' }, () => Promise.all(AGENTS.map((agent) => meetingOne(this.env, round, agent.id, publicReports))));

      await step.do(`publish meeting reactions ${round}`, { retries: { limit: 2, delay: '5 seconds', backoff: 'linear' } }, async () => {
        await chargeResults(this.env, params, round, 'meeting', meeting);
        await addEvents(this.env.DB, params.runId, meeting.map((result) => {
          const index = AGENT_INDEX[result.agentId];
          return {
            seq: round * 1000 + 510 + index,
            event: {
              at: nowIso(), round, phase: 'meeting' as const, agentId: result.agentId,
              kind: result.ok ? 'meeting' as const : 'error' as const,
              title: `${AGENTS[index].name} reacts`,
              summary: result.value?.reaction ?? 'Reaction call failed; this agent will rejoin next round.',
              visible: true,
              payload: result.value ? {
                agreements: result.value.agreements,
                objections: result.value.objections,
                collaborationCredits: result.value.collaborationCredits,
              } : { error: result.error ?? 'Meeting call failed.' },
            },
          };
        }));

        for (const result of meeting) {
          if (!result.value) continue;
          const ownReport = publicReports.find((item) => item.agentId === result.agentId)?.report ?? {};
          await this.env.DB.batch([
            this.env.DB.prepare(`INSERT INTO agent_memory(run_id,agent_id,public_summary_json,private_plan_json,updated_at) VALUES(?,?,?,?,?) ON CONFLICT(run_id,agent_id) DO UPDATE SET public_summary_json=excluded.public_summary_json,private_plan_json=excluded.private_plan_json,updated_at=excluded.updated_at`)
              .bind(params.runId, result.agentId, JSON.stringify(ownReport), JSON.stringify(result.value.privateNextPlan), nowIso()),
            this.env.DB.prepare(`INSERT OR IGNORE INTO private_plans(run_id,round,agent_id,plan_json,created_at) VALUES(?,?,?,?,?)`)
              .bind(params.runId, round, result.agentId, JSON.stringify(result.value.privateNextPlan), nowIso()),
          ]);
        }
      });

      if (Date.now() < meetingDeadline) await step.sleepUntil(`finish round table ${round}`, meetingDeadline);
      completedRound = round;
    }

    return await step.do('publish terminal scientific report', { retries: { limit: 3, delay: '10 seconds', backoff: 'linear' } }, async () => {
      await finalReport(this.env, params, terminal, completedRound);
      return { published: true, outcome: terminal, round: completedRound };
    });
    } catch (error) {
      const failureAt = nowIso();
      await step.do('record terminal workflow failure', { retries: { limit: 5, delay: '15 seconds', backoff: 'linear' } }, async () => {
        await patchState(this.env.DB, params.runId, { phase: 'error', phaseEndsAt: null });
        await this.env.DB.prepare(`UPDATE runs SET status='error',phase='error',phase_ends_at=NULL,completed_at=?,updated_at=? WHERE id=? AND status='running'`)
          .bind(failureAt, failureAt, params.runId)
          .run();
        await addEvent(this.env.DB, params.runId, (completedRound + 1) * 1000 + 998, {
          at: failureAt,
          round: completedRound,
          phase: 'error',
          kind: 'error',
          title: 'Experiment engine stopped safely',
          summary: 'A non-model infrastructure operation exhausted its retries. The run was closed so a new launch is never wedged behind it.',
          visible: true,
        });
      });
      throw error;
    }
  }
}
