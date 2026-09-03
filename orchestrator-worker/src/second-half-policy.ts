import { AGENT_IDS, type AgentId } from './types';

export const SECOND_HALF_METHODS = [
  'constructive parametrization and algebraic families (including elliptic-curve or rational-point structure)',
  'modular, local, and congruence design to prune or force compatible rows',
  'incidence-hypergraph, SAT, or exact-cover search over verified cells',
  'stochastic and optimization-guided seed discovery followed by exact bigint verification',
  'adversarial falsification, saturation certificates, and adaptive experiment design',
] as const;

export interface SecondHalfPolicy {
  primaryMethod: string;
  designatedDivisorVerifier: AgentId;
  divisorRule: string;
  objective: string;
}

export function secondHalfPolicy(agentId: AgentId, round: number): SecondHalfPolicy {
  const agentIndex = AGENT_IDS.indexOf(agentId);
  const block = Math.max(0, Math.floor((round - 26) / 5));
  const primaryMethod = SECOND_HALF_METHODS[(agentIndex + block) % SECOND_HALF_METHODS.length];
  const designatedDivisorVerifier = AGENT_IDS[Math.max(0, round - 26) % AGENT_IDS.length];
  return {
    primaryMethod,
    designatedDivisorVerifier,
    divisorRule: agentId === designatedDivisorVerifier
      ? 'You may request at most one divisor_completion job, and only to verify a candidate first produced by a different primary method.'
      : `Do not request divisor_completion this round; ${designatedDivisorVerifier} is the sole validation-only verifier.`,
    objective: 'Optimize balanced progress toward a shared 5×5 biclique: 5-row and 5-column support floors first, then 4×5/5×4 structure, never isolated 1×n anchors.',
  };
}

export function sanitizeCollaborationCredits(author: AgentId, credits: readonly string[]) {
  const valid = new Set<AgentId>(AGENT_IDS);
  const cleaned: AgentId[] = [];
  for (const credit of credits) {
    if (!valid.has(credit as AgentId) || credit === author || cleaned.includes(credit as AgentId)) continue;
    cleaned.push(credit as AgentId);
    if (cleaned.length === 2) break;
  }
  return cleaned;
}

export function balancedCollaborationCredits(
  round: number,
  nominations: ReadonlyArray<{ agentId: AgentId; credits: readonly string[] }>,
  successfulContributors: readonly AgentId[],
) {
  const result = new Map<AgentId, AgentId[]>();
  for (const nomination of nominations) {
    result.set(nomination.agentId, sanitizeCollaborationCredits(nomination.agentId, nomination.credits));
  }

  const successful = new Set(successfulContributors);
  const targets: AgentId[] = ['solvi', 'tess'];
  for (const target of targets) {
    const alreadyRecognized = [...result.values()].some((credits) => credits.includes(target));
    if (alreadyRecognized || !successful.has(target)) continue;
    const start = Math.max(0, round - 26) % AGENT_IDS.length;
    const authors = [...AGENT_IDS.slice(start), ...AGENT_IDS.slice(0, start)];
    const author = authors.find((candidate) => candidate !== target && result.has(candidate) && (result.get(candidate)?.length ?? 0) < 2);
    if (author) result.set(author, [...(result.get(author) ?? []), target]);
  }
  return result;
}

export const SECOND_HALF_POLICY_SUMMARY = {
  metric: 'Balanced 5×5 support floors outrank isolated high-degree anchors.',
  divisorAtlas: 'Validation only: at most one rotating agent may submit one divisor-completion job per round.',
  lanes: 'Five distinct human-mathematics methods rotate every five rounds as a Latin-square assignment.',
  novelty: 'Completed negative scans and repeatedly falsified row sets are cooldown inputs, not prompts to repeat them.',
  collaboration: 'No self-credit; at most two peers per reaction; successful Solvi and Tess contributions receive a batch-level recognition floor.',
  collaborationRewardUsd: 10,
} as const;
