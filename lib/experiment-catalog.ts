/** Public experiment metadata. Keep historical runs independent of the current worker. */
export interface ExperimentDefinition {
  sequence: number;
  slug: string;
  title: string;
  label: string;
  runId: string;
  status: 'complete' | 'planned' | 'registered';
  model: string;
  reasoning: string;
  researchers: number;
  rounds: number;
  scheduleLabel?: string;
  objective: string;
  protocol: string;
  verifier: string;
  budgets: { openai: number; exa: number };
}

export const erdosPilot: ExperimentDefinition = {
  sequence: 1,
  slug: 'erdos-885', title: 'Erdős 885 · k = 5', label: 'Pilot 001',
  runId: 'competition-f6aadf00-aeb3-465b-9ba2-8679d70f28b1', status: 'complete',
  model: 'gpt-5.6-luna', reasoning: 'high', researchers: 5, rounds: 100,
  objective: 'Find five distinct positive integers sharing five distinct factor-pair differences.',
  protocol: 'Five-minute private research, then a five-minute shared round table. Asynchronous exact computation.',
  verifier: 'Exact integer factor-pair witnesses; no floating-point tolerance.',
  budgets: { openai: 50, exa: 40 },
};

export const rewardCompatibility: ExperimentDefinition = {
  sequence: 2,
  slug: 'reward-compatibility', title: 'Measuring reward compatibility', label: 'Experiment 002',
  runId: 'experiment-002-v01', status: 'registered',
  model: 'gpt-5.6-luna', reasoning: 'none (actor), high (evaluators)', researchers: 1, rounds: 12,
  scheduleLabel: '48 matched optimization histories',
  objective: 'Test whether readable high-reward strategies predict subsequent changes in monitorability.',
  protocol: 'Frozen coin-tracking study: feasibility gate, diagnostics, paired repeated optimization, sealed held-out evaluation.',
  verifier: 'Exact coin parity, blinded monitoring, grounded evidence audits and paired-history analysis.',
  budgets: {openai: 40, exa: 0},
};
/** Add published experiments here; ordering and the public landing page follow sequence. */
export const experiments: readonly ExperimentDefinition[] = [erdosPilot, rewardCompatibility].sort((a,b)=>b.sequence-a.sequence);
export const latestExperiment = experiments[0];
export const pilotReportUrl = `https://autolabs-orchestrator.raphaelbahadurkhan.workers.dev/api/experiments/${erdosPilot.runId}/report`;
export const pilotLedgerUrl = `https://autolabs-orchestrator.raphaelbahadurkhan.workers.dev/api/experiments/${erdosPilot.runId}/events`;

/** Audited from the public schema-2 terminal report on 2026-09-08. Estimates, not invoices. */
export const pilotSnapshot = {
  completedAt: '2026-09-08T16:15:24.849Z',
  openaiUsd: 7.41198192, exaUsd: 3.535,
  cumulativeOpenaiUsd: 9.32071384, cumulativeExaUsd: 4.536,
  events: 2611, certificates: 87, jobs: { complete: 297, partial: 297, failed: 12 },
  privatePlans: 496, retrievals: 505,
  candidate: {
    round: 40, agent: 'Pip Δ',
    numbers: ['12096', '30400', '51100', '21384', '7084', '61600', '10565100'],
    differences: ['30', '51', '117', '204', '225', '300'],
  },
};
