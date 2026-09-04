export const AGENT_IDS = ['mira', 'pip', 'orum', 'solvi', 'tess'] as const;
export type AgentId = (typeof AGENT_IDS)[number];
export type Phase = 'ribbon' | 'research' | 'meeting' | 'complete' | 'eureka' | 'budget-stop' | 'error';

export interface RunParams {
  runId: string;
  mode: 'rehearsal' | 'competition';
  targetRounds: number;
  minimumRounds: number;
  phaseMinutes: number;
  budgetUsd: number;
  reserveUsd: number;
  startRound?: number;
}

export interface CandidateInput {
  numbers: string[];
  differences: string[];
  note: string;
}

export type JobParameter = string | number | null;

export interface ProposedJob {
  jobType: 'divisor_completion' | 'family_scan' | 'boundary_scan';
  params: Record<string, JobParameter>;
  reason: string;
}

export interface ResearchReport {
  headline: string;
  thesis: string;
  claims: string[];
  equations: string[];
  citations: string[];
  failedAvenues: string[];
  candidates: CandidateInput[];
  proposedJobs: ProposedJob[];
  nextQuestions: string[];
}

export interface MeetingReport {
  reaction: string;
  agreements: string[];
  objections: string[];
  collaborationCredits: AgentId[];
  privateNextPlan: { objective: string; checks: string[] };
}

export interface Usage {
  inputTokens: number;
  cachedInputTokens: number;
  outputTokens: number;
}

export interface UsageRecord {
  responseId: string;
  usage: Usage;
}

export interface ModelTrace {
  agentId: AgentId;
  round: number;
  phase: 'research' | 'meeting';
  attempt: number;
  status: 'connecting' | 'streaming' | 'complete' | 'error';
  outputCharacters: number;
  updatedAt: string;
}

export interface AgentResult<T> {
  agentId: AgentId;
  ok: boolean;
  responseId?: string;
  value?: T;
  usage: Usage;
  usageRecords?: UsageRecord[];
  error?: string;
}

export interface PublicEvent {
  seq?: number;
  at: string;
  round: number;
  phase: Phase;
  agentId?: AgentId;
  kind: 'system' | 'research' | 'tool' | 'meeting' | 'candidate' | 'budget' | 'error';
  title: string;
  summary: string;
  visible: boolean;
  payload?: unknown;
}
