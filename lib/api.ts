import type { ExperimentState } from './experiment';
import { initialExperiment } from './experiment';

export async function fetchExperiment(signal?: AbortSignal): Promise<ExperimentState> {
  const endpoint = process.env.NEXT_PUBLIC_ORCHESTRATOR_URL;
  if (!endpoint) return initialExperiment;
  const response = await fetch(`${endpoint}/api/experiments/current`, { signal, cache: 'no-store' });
  if (!response.ok) throw new Error(`Live engine returned ${response.status}`);
  return response.json() as Promise<ExperimentState>;
}
