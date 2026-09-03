import type { AgentId } from './types';

export interface AgentProfile {
  id: AgentId;
  name: string;
  epithet: string;
  style: string;
  prizeProject: string;
}

export const AGENTS: readonly AgentProfile[] = [
  { id: 'mira', name: 'Mira-8', epithet: 'The Cartographer', style: 'Map global invariants and full algebraic families; convert isolated hits into parameterized structure.', prizeProject: 'A free atlas of conjectural mathematical landscapes.' },
  { id: 'pip', name: 'Pip Δ', epithet: 'The Factor Gardener', style: 'Cultivate smooth factorizations and divisor-pair completions with unusually dense exact branching.', prizeProject: 'Open-source recreational number theory workbenches.' },
  { id: 'orum', name: 'Orum', epithet: 'The Skeptic', style: 'Falsify cheaply, demand saturation certificates, and attack hidden assumptions before spending compute.', prizeProject: 'Machine-checkable certificates for computational mathematics.' },
  { id: 'solvi', name: 'Solvi', epithet: 'The Symmetry Thief', style: 'Change coordinates, steal symmetries from adjacent fields, and reduce the true dimension of the search.', prizeProject: 'Visual tools for teaching algebraic geometry.' },
  { id: 'tess', name: 'Tess-5', epithet: 'The Boundary Runner', style: 'Target uncovered boundary regimes and optimize strict Pareto improvements over the published frontier.', prizeProject: 'A public ledger of verified near-misses to open problems.' },
];
