export const persona3bContract = {
  studyId: 'experiment-003b-v1',
  phase: 'development-screen-scoring',
  totalRecords: 780,
  pairedComparisons: 768,
  repeatRecords: 117,
  disagreementRecords: 117,
  callCeiling: 1014,
  budgetUsd: 10,
  model: 'gpt-5.6-luna',
  reasoningEffort: 'high',
  sourceArtifactHash: '6e479c2e02aa912f6bcf6b2e57554ed75711e93bb3fc3da813114bba871f3fe7',
} as const;

export type Persona3BRunStatus = 'queued' | 'running' | 'awaiting_adjudication' | 'synthesizing' | 'complete' | 'failed' | 'cancelled';
export type Persona3BShardStatus = 'blocked' | 'queued' | 'claimed' | 'running' | 'complete' | 'failed';
export type Persona3BShardPhase = 'primary' | 'repeat' | 'disagreement' | 'synthesis';

export interface Persona3BRun {
  id: string;
  studyId: string;
  status: Persona3BRunStatus;
  phase: string;
  manifestHash: string;
  sourceArtifactHash: string;
  totalRecords: number;
  primaryRecords: number;
  repeatRecords: number;
  disagreementRecords: number;
  model: string;
  reasoningEffort: string;
  /** Effective ceiling: an authorized same-run amendment supersedes the frozen manifest ceiling. */
  callCeiling: number;
  /** Frozen manifest ceiling; present when the worker reports an amendment. */
  originalCallCeiling?: number;
  amendment?: { amendmentHash: string; originalCallCeiling: number; effectiveCallCeiling: number; hardBudgetUsd: number; createdAt: string } | null;
  callCount: number;
  budgetUsd: number;
  reservedUsd: number;
  spentUsd: number;
  createdAt: string;
  updatedAt: string;
  completedAt: string | null;
  error: string | null;
}

export interface Persona3BShard {
  id: string;
  runId: string;
  phase: Persona3BShardPhase;
  shardIndex: number;
  agentId: string;
  expectedRecords: number;
  expectedCalls: number;
  status: Persona3BShardStatus;
  completedRecords: number;
  attempts: number;
  callCount: number;
  usage: { inputTokens: number; cachedInputTokens: number; outputTokens: number };
  claimedAt: string | null;
  updatedAt: string;
  completedAt: string | null;
  error: string | null;
}

export interface Persona3BEvent {
  id?: string;
  at: string;
  shardId?: string | null;
  phase?: Persona3BShardPhase | null;
  agentId?: string | null;
  kind: string;
  title: string;
  summary: string;
  payload?: unknown;
}

export interface Persona3BStatus {
  run: Persona3BRun | null;
  shards: Persona3BShard[];
  events: Persona3BEvent[];
}

export async function fetchPersona3BStatus(signal?: AbortSignal): Promise<Persona3BStatus> {
  const response = await fetch('/api/persona-3b/status', { signal, cache: 'no-store' });
  if (response.status === 404) return { run: null, shards: [], events: [] };
  if (!response.ok) throw new Error(`Experiment 3B status returned ${response.status}`);
  return response.json() as Promise<Persona3BStatus>;
}

export async function startPersona3B(ownerKey: string): Promise<Persona3BStatus & { accepted?: boolean; idempotent?: boolean }> {
  const response = await fetch('/api/control/persona-3b/start', {
    method: 'POST',
    headers: { 'content-type': 'application/json', 'x-autolabs-owner-key': ownerKey },
    body: JSON.stringify({ idempotencyKey: `autolabs-3b-${crypto.randomUUID()}` }),
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(typeof payload?.error === 'string' ? payload.error : `Start returned ${response.status}`);
  return payload as Persona3BStatus & { accepted?: boolean; idempotent?: boolean };
}

export function persona3bPhaseLabel(run: Persona3BRun | null) {
  if (!run) return 'AWAITING OWNER START';
  if (run.status === 'queued') return 'QUEUED · NO PROVIDER REQUEST';
  if (run.status === 'running' && run.callCount < run.primaryRecords) return 'PARALLEL BLIND SCORING';
  if (run.status === 'running') return 'REPEAT REVIEW';
  if (run.status === 'awaiting_adjudication') return 'ADJUDICATION';
  if (run.status === 'synthesizing') return 'FINAL ANALYST';
  if (run.status === 'complete') return 'SCORING COMPLETE';
  if (run.status === 'failed') return 'PAUSED · ERROR';
  if (run.status === 'cancelled') return 'CANCELLED';
  return String(run.status).toUpperCase();
}

export function phaseDisplay(phase: Persona3BShardPhase) {
  if (phase === 'primary') return 'Independent graders';
  if (phase === 'repeat') return 'Repeat reviewer';
  if (phase === 'disagreement') return 'Adjudicator';
  return 'Final analyst';
}
