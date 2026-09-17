export const PERSONA_3C_STAGES = ['boot', 'harvest', 'train', 'calibrate', 'screen', 'judge', 'analysis', 'done'] as const;
export type Persona3CStage = typeof PERSONA_3C_STAGES[number];
export type Persona3CRunStatus = 'queued' | 'running' | 'complete' | 'failed' | 'stopped';

export interface Persona3CRun {
  id: string;
  studyId: string;
  status: Persona3CRunStatus;
  stage: string;
  manifestHash: string;
  budgetUsd: number;
  spentUsd: number;
  gpuHours: number;
  createdAt: string;
  updatedAt: string;
  completedAt: string | null;
  error: string | null;
  lastRecordId: string | null;
}

export interface Persona3CProgress {
  stage: string;
  done: number;
  total: number;
  updatedAt: string;
}

export interface Persona3CEvent {
  id?: number;
  at: string;
  stage: string | null;
  kind: string;
  title: string;
  summary: string;
}

export interface Persona3CTrainPoint {
  tokensDone: number | null;
  tokensTarget: number | null;
  stepsDone: number | null;
  fve: number | null;
  deadFraction: number | null;
  loss: number | null;
}

export interface Persona3CStatus {
  run: Persona3CRun | null;
  recordCounts?: Record<string, number>;
  progress?: Persona3CProgress[];
  events?: Persona3CEvent[];
  trainCurve?: Persona3CTrainPoint[];
  staleMinutes?: number | null;
  error?: string;
}

/** Minutes without a report after which a running pod is treated as quiet (matches the Worker cron). */
export const PERSONA_3C_STALE_AFTER_MINUTES = 45;

export async function fetchPersona3CStatus(signal?: AbortSignal): Promise<Persona3CStatus> {
  const response = await fetch('/api/persona-3c/status', { signal, cache: 'no-store' });
  if (response.status === 404) return { run: null };
  if (!response.ok) throw new Error(`Experiment 3C status returned ${response.status}`);
  return response.json() as Promise<Persona3CStatus>;
}

export function persona3cStageLabel(stage: string) {
  const known: Record<Persona3CStage, string> = {
    boot: 'Booting pod',
    harvest: 'Harvesting data',
    train: 'Training',
    calibrate: 'Calibrating',
    screen: 'Screening',
    judge: 'Judging',
    analysis: 'Analysis',
    done: 'Done',
  };
  return known[stage as Persona3CStage] ?? stage;
}
