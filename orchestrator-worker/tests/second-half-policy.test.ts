import { describe, expect, it } from 'vitest';
import { AGENT_IDS } from '../src/types';
import { balancedCollaborationCredits, sanitizeCollaborationCredits, secondHalfPolicy } from '../src/second-half-policy';

describe('second-half diversification policy', () => {
  it('assigns five distinct primary methods each round and rotates them by block', () => {
    const round26 = AGENT_IDS.map((id) => secondHalfPolicy(id, 26).primaryMethod);
    const round31 = AGENT_IDS.map((id) => secondHalfPolicy(id, 31).primaryMethod);
    expect(new Set(round26).size).toBe(5);
    expect(new Set(round31).size).toBe(5);
    expect(round31).not.toEqual(round26);
  });

  it('permits only one rotating divisor verifier per round', () => {
    const verifiers = AGENT_IDS.filter((id) => secondHalfPolicy(id, 26).designatedDivisorVerifier === id);
    expect(verifiers).toEqual(['mira']);
    expect(secondHalfPolicy('pip', 27).designatedDivisorVerifier).toBe('pip');
  });

  it('removes self, invalid, duplicate, and excessive credits', () => {
    expect(sanitizeCollaborationCredits('mira', ['mira', 'solvi', 'solvi', 'nobody', 'tess', 'pip']))
      .toEqual(['solvi', 'tess']);
  });

  it('recognizes successful Solvi and Tess contributions at least once per batch', () => {
    const credits = balancedCollaborationCredits(26, [
      { agentId: 'mira', credits: ['pip'] },
      { agentId: 'pip', credits: ['orum'] },
      { agentId: 'orum', credits: ['mira'] },
      { agentId: 'solvi', credits: ['mira'] },
      { agentId: 'tess', credits: ['pip'] },
    ], [...AGENT_IDS]);
    expect([...credits.values()].some((items) => items.includes('solvi'))).toBe(true);
    expect([...credits.values()].some((items) => items.includes('tess'))).toBe(true);
    for (const [author, items] of credits) expect(items).not.toContain(author);
  });
});
