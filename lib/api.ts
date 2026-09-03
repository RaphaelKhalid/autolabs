import type { ExperimentState } from './experiment';
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
