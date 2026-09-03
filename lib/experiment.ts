export type Phase = 'idle' | 'ribbon' | 'research' | 'meeting' | 'complete' | 'eureka';
export type AgentStatus = 'ready' | 'researching' | 'meeting' | 'recovering' | 'complete';

export interface ResearchAgent {
  id: string;
  name: string;
  epithet: string;
  outfit: string;
  accent: string;
  status: AgentStatus;
  bubble: string;
  project: string;
  approach: string;
  bestSupport: number[];
  citations: number;
  tools: string[];
}

export interface ExperimentEvent {
  seq: number;
  at: string;
  round: number;
  phase: Phase;
  agentId?: string;
  kind: 'system' | 'research' | 'tool' | 'meeting' | 'candidate' | 'budget' | 'error';
  title: string;
  summary: string;
  visible: boolean;
}

export interface ExperimentState {
  id: string;
  mode: 'rehearsal' | 'competition';
  phase: Phase;
  round: number;
  targetRounds: number;
  minimumRounds: number;
  phaseEndsAt: string | null;
  spentUsd: number;
  budgetUsd: number;
  reserveUsd: number;
  calls: number;
  bestSupport: number[];
  bestLabel: string;
  bestVerified: boolean;
  sotaImproved: boolean;
  startedAt: string | null;
  agents: ResearchAgent[];
  events: ExperimentEvent[];
}

const future = new Date(Date.now() + 4 * 60_000 + 23_000).toISOString();

export const demoAgents: ResearchAgent[] = [
  {
    id: 'mira', name: 'Mira-8', epithet: 'The Cartographer', outfit: 'Amber survey cape', accent: '#ffc76b', status: 'researching',
    bubble: 'Mapping the full elliptic family—no fixed specialization shortcuts.',
    project: 'A free atlas of conjectural mathematical landscapes.', approach: 'Global structure first; turns isolated hits into families.',
    bestSupport: [4, 4, 4, 4, 4], citations: 7, tools: ['family_mapper', 'exact_verify'],
  },
  {
    id: 'pip', name: 'Pip Δ', epithet: 'The Factor Gardener', outfit: 'Moss utility vest', accent: '#9ee493', status: 'researching',
    bubble: 'Smooth difference-of-squares seeds are producing dense exact branches.',
    project: 'Open-source recreational number theory workbenches.', approach: 'Cultivates smooth factorizations with high completion density.',
    bestSupport: [4, 4, 4, 4, 4], citations: 4, tools: ['divisor_completion', 'factor_sieve'],
  },
  {
    id: 'orum', name: 'Orum', epithet: 'The Skeptic', outfit: 'Plum academic sash', accent: '#d8a8ff', status: 'researching',
    bubble: 'Trying to kill each promising branch cheaply before we believe it.',
    project: 'Machine-checkable certificates for computational mathematics.', approach: 'Adversarial falsification and exact saturation certificates.',
    bestSupport: [4, 4, 4, 4, 4], citations: 9, tools: ['saturation_probe', 'certificate_builder'],
  },
  {
    id: 'solvi', name: 'Solvi', epithet: 'The Symmetry Thief', outfit: 'Cyan raincoat', accent: '#7de5ef', status: 'researching',
    bubble: 'A coordinate change exposes a symmetry the old scans never used.',
    project: 'Visual tools for teaching algebraic geometry.', approach: 'Reparameterizes search spaces to reveal hidden symmetries.',
    bestSupport: [4, 4, 4, 4, 4], citations: 5, tools: ['symmetry_reduce', 'curve_sampler'],
  },
  {
    id: 'tess', name: 'Tess-5', epithet: 'The Boundary Runner', outfit: 'Red knit cap', accent: '#ff8d7b', status: 'researching',
    bubble: 'Searching where the published cutoffs leave the thinnest frontier.',
    project: 'A public ledger of verified near-misses to open problems.', approach: 'Targets unsearched boundary regimes and Pareto improvements.',
    bestSupport: [4, 4, 4, 4, 4], citations: 6, tools: ['frontier_scheduler', 'exact_verify'],
  },
];

export const demoEvents: ExperimentEvent[] = [
  { seq: 1, at: new Date(Date.now() - 18 * 60_000).toISOString(), round: 0, phase: 'ribbon', kind: 'system', title: 'Ribbon cut', summary: 'The five researchers entered the lab. All prompts, prizes and stopping rules were committed before the first call.', visible: true },
  { seq: 2, at: new Date(Date.now() - 13 * 60_000).toISOString(), round: 1, phase: 'research', agentId: 'pip', kind: 'tool', title: 'Exact divisor completion', summary: 'Enumerated a smooth seed without floating-point arithmetic; 18,442 branches were rejected by square tests.', visible: true },
  { seq: 3, at: new Date(Date.now() - 10 * 60_000).toISOString(), round: 1, phase: 'meeting', kind: 'meeting', title: 'Round table #1', summary: 'Reports were revealed simultaneously. Orum challenged Solvi’s symmetry assumption; Mira proposed a family-level test.', visible: true },
  { seq: 4, at: new Date(Date.now() - 7 * 60_000).toISOString(), round: 2, phase: 'research', agentId: 'mira', kind: 'research', title: 'Family map expanded', summary: 'A non-fixed Bremner specialization survived the first exact constraints. No claim of novelty yet.', visible: true },
  { seq: 5, at: new Date(Date.now() - 3 * 60_000).toISOString(), round: 2, phase: 'research', agentId: 'orum', kind: 'candidate', title: 'Candidate verified: near miss', summary: 'Exact support vector remains (4,4,4,4,4). Certificate retained; size alone does not improve the frontier.', visible: true },
];

export const initialExperiment: ExperimentState = {
  id: 'dress-rehearsal', mode: 'rehearsal', phase: 'research', round: 2, targetRounds: 50, minimumRounds: 25,
  phaseEndsAt: future, spentUsd: 1.84, budgetUsd: 50, reserveUsd: 1.5, calls: 21,
  bestSupport: [4, 4, 4, 4, 4], bestLabel: 'five differences shared by four integers', bestVerified: true,
  sotaImproved: false, startedAt: new Date(Date.now() - 18 * 60_000).toISOString(), agents: demoAgents, events: demoEvents,
};

export function formatCountdown(end: string | null, now = Date.now()) {
  if (!end) return '--:--';
  const seconds = Math.max(0, Math.floor((new Date(end).getTime() - now) / 1000));
  return `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`;
}

export function supportLabel(values: number[]) {
  return `(${values.join(',')})`;
}
