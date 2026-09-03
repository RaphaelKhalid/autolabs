import { AGENTS, type AgentProfile } from './agents';
import { PROBLEM_CONTEXT } from './known-results';
import type { AgentId } from './types';

const stringArray = { type: 'array', items: { type: 'string' } } as const;
const candidate = {
  type: 'object', additionalProperties: false,
  properties: { numbers: stringArray, differences: stringArray, note: { type: 'string' } },
  required: ['numbers', 'differences', 'note'],
} as const;
const proposedJob = {
  type: 'object', additionalProperties: false,
  properties: {
    jobType: { type: 'string', enum: ['divisor_completion', 'family_scan', 'boundary_scan'] },
    params: { type: 'object', additionalProperties: { anyOf: [{ type: 'string' }, { type: 'number' }] } },
    reason: { type: 'string' },
  },
  required: ['jobType', 'params', 'reason'],
} as const;

export const RESEARCH_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    headline: { type: 'string' }, thesis: { type: 'string' }, claims: stringArray,
    equations: stringArray, citations: stringArray, failedAvenues: stringArray,
    candidates: { type: 'array', items: candidate, maxItems: 8 },
    proposedJobs: { type: 'array', items: proposedJob, maxItems: 3 }, nextQuestions: stringArray,
  },
  required: ['headline', 'thesis', 'claims', 'equations', 'citations', 'failedAvenues', 'candidates', 'proposedJobs', 'nextQuestions'],
} as const;

export const MEETING_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    reaction: { type: 'string' }, agreements: stringArray, objections: stringArray,
    collaborationCredits: { type: 'array', items: { type: 'string', enum: AGENTS.map((agent) => agent.id) } },
    privateNextPlan: {
      type: 'object', additionalProperties: false,
      properties: { objective: { type: 'string' }, checks: stringArray }, required: ['objective', 'checks'],
    },
  },
  required: ['reaction', 'agreements', 'objections', 'collaborationCredits', 'privateNextPlan'],
} as const;

function commonSystem(profile: AgentProfile) {
  return `You are ${profile.name}, ${profile.epithet}, one of five unrestricted expert mathematicians in Autolabs. Your alien cognitive style is: ${profile.style} This style changes idea-generation, not your access to any field of human mathematics. You are serious, precise, competitively collaborative, and determined. Your public later-prize project choice is: ${profile.prizeProject}.

PUBLIC-RECORD RULE: return inspectable research summaries—claims, equations, citations, tool/job requests, falsifications and conclusions. Never output hidden chain-of-thought or private deliberation. Label conjectures. Never claim a result from numerical closeness. Every candidate will be checked by deterministic bigint code.

${PROBLEM_CONTEXT}`;
}

export function researchPrompt(profile: AgentProfile, round: number, memory: unknown, completedJobs: unknown) {
  return {
    system: commonSystem(profile),
    user: `Round ${round} of at most 50. Produce one ambitious but bounded five-minute research contribution. Do not repeat covered scans. You may propose up to three asynchronous code jobs; their results can arrive in later rounds. Explicit candidate integers must be decimal strings. Look for a genuine k=5 rectangle or the strict known-frontier improvements 6x4 or 4x5.\n\nYour compact prior memory:\n${JSON.stringify(memory)}\n\nCompleted code jobs available now:\n${JSON.stringify(completedJobs)}`,
  };
}

export function meetingPrompt(profile: AgentProfile, round: number, reports: unknown) {
  return {
    system: commonSystem(profile),
    user: `Round ${round} simultaneous reveal follows. Give exactly one public reaction to the batch: concrete agreements, objections, and collaboration credit. Then produce a private next-round plan. The plan is stored internally and is not shown to other agents or the public until the experiment ends. Do not put private plan content in the public reaction.\n\nSIMULTANEOUS REPORT BATCH:\n${JSON.stringify(reports)}`,
  };
}

export function agentById(id: AgentId) {
  const profile = AGENTS.find((agent) => agent.id === id);
  if (!profile) throw new Error(`Unknown agent ${id}`);
  return profile;
}
