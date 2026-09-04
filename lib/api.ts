import type { ExperimentEvent, ExperimentState } from './experiment';
import { initialExperiment } from './experiment';

function idleExperiment(): ExperimentState {
  return {
    ...initialExperiment,
    id: 'awaiting-launch',
    phase: 'idle',
    round: 0,
    phaseEndsAt: null,
    spentUsd: 0,
    calls: 0,
    startedAt: null,
    agents: initialExperiment.agents.map((agent) => ({ ...agent, status: 'ready' })),
    events: [],
  };
}

export async function fetchExperiment(signal?: AbortSignal): Promise<ExperimentState> {
  const endpoint = process.env.NEXT_PUBLIC_ORCHESTRATOR_URL;
  if (!endpoint) return initialExperiment;

  const response = await fetch(`${endpoint}/api/experiments/current`, {
    signal,
    cache: 'no-store',
  });

  // A freshly deployed engine has no run until the owner cuts the ribbon.
  if (response.status === 404) return idleExperiment();
  if (!response.ok) throw new Error(`Live engine returned ${response.status}`);
  return response.json() as Promise<ExperimentState>;
}
export async function fetchAgentEvents(runId: string, agentId: string, signal?: AbortSignal): Promise<ExperimentEvent[]> {
  const endpoint = process.env.NEXT_PUBLIC_ORCHESTRATOR_URL;
  if (!endpoint) return initialExperiment.events.filter((event) => !event.agentId || event.agentId === agentId);
  const url = new URL(`${endpoint}/api/experiments/${encodeURIComponent(runId)}/events`);
  url.searchParams.set('agentId', agentId);
  url.searchParams.set('limit', '500');
  const response = await fetch(url, { signal, cache: 'no-store' });
  if (!response.ok) throw new Error(`Research record returned ${response.status}`);
  const payload = await response.json() as { events?: ExperimentEvent[] };
  return payload.events ?? [];
}