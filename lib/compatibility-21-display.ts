export const COMPATIBILITY21_API = 'https://autolabs-compatibility-21.raphaelbahadurkhan.workers.dev';
export interface Compatibility21Status {
  runId: string; version: string; model: string;
  status: 'ready'|'running'|'paused'|'complete'; stage: 'dev'|'eval'; reason: string|null;
  updatedAt: string; protocolHash: string|null; evaluationSealed: boolean;
  progress: {callsDone: number; callsTotal: number; cases: {split: string; done: number; total: number}[]};
  budget: {spentUsd: number; reservedUsd: number; priorCommittedUsd: number; totalCommittedUsd: number; capUsd: number; calls: number; knownSpendUsd?:number; uncertainChargeUpperBoundUsd?:number};
  execution: {concurrency: number}; ledger: {storageBytes: number; softLimitBytes: number};
  active: {id: string; started: number}[]; etaSeconds: number|null;
  gate: {pass?: boolean; parseRate?: number; forecastUsd?: number}|null;
  isolation: {passed?: boolean}|null;
  recent: {time: string; type: string; data: unknown}[];
}

const object = (value: unknown): value is Record<string, unknown> => value !== null && typeof value === 'object' && !Array.isArray(value);
const nonnegative = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value) && value >= 0;
export function validCompatibility21Status(value: unknown): value is Compatibility21Status {
  if (!object(value) || !['ready','running','paused','complete'].includes(String(value.status)) || !['dev','eval'].includes(String(value.stage))) return false;
  const {progress, budget, execution, ledger, active} = value;
  return typeof value.runId === 'string' && typeof value.version === 'string' && typeof value.model === 'string' && typeof value.updatedAt === 'string' &&
    typeof value.evaluationSealed === 'boolean' && (value.reason === null || typeof value.reason === 'string') && (value.protocolHash === null || typeof value.protocolHash === 'string') &&
    object(progress) && nonnegative(progress.callsDone) && nonnegative(progress.callsTotal) && Array.isArray(progress.cases) && progress.cases.every(row => object(row) && typeof row.split === 'string' && nonnegative(row.done) && nonnegative(row.total)) &&
    object(budget) && ['spentUsd','reservedUsd','priorCommittedUsd','totalCommittedUsd','capUsd','calls'].every(key => nonnegative(budget[key])) &&
    ['knownSpendUsd','uncertainChargeUpperBoundUsd'].every(key=>budget[key]===undefined||nonnegative(budget[key])) &&
    object(execution) && nonnegative(execution.concurrency) && object(ledger) && nonnegative(ledger.storageBytes) && nonnegative(ledger.softLimitBytes) &&
    Array.isArray(active) && active.every(row => object(row) && typeof row.id === 'string' && nonnegative(row.started)) &&
    (value.etaSeconds === null || nonnegative(value.etaSeconds)) && (value.gate === null || object(value.gate)) && (value.isolation === null || object(value.isolation)) &&
    Array.isArray(value.recent) && value.recent.every(row => object(row) && typeof row.time === 'string' && typeof row.type === 'string');
}

export function compatibility21Eta(status: Pick<Compatibility21Status,'status'|'etaSeconds'> | null): string {
  if (!status) return 'Waiting for the first checkpoint';
  if (status.status === 'complete') return 'Complete';
  if (status.status === 'paused') return 'Paused — estimate on hold';
  if (status.status === 'ready') return 'Not started';
  if (status.etaSeconds === null || !Number.isFinite(status.etaSeconds)) return 'Estimating after the first calls';
  const minutes = Math.max(1, Math.ceil(status.etaSeconds / 60));
  return minutes < 60 ? `About ${minutes} min remaining` : `About ${Math.floor(minutes/60)} h ${minutes%60} min remaining`;
}
export const methodNames: Record<string,string> = {description:'Description-only judgment',unguided:'Unguided search',guided:'Verifier-guided search'};
export function compatibility21CallTitle(id: string): string {
  const parts = id.split('/');
  const method = parts.find(part => Object.hasOwn(methodNames, part));
  const step = Number(parts.at(-1));
  return `${method ? methodNames[method] : 'Research call'}${Number.isInteger(step) && step >= 0 ? ` · step ${step+1}` : ''}`;
}
