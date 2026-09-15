import { describe, expect, it } from 'vitest';
import { canReserveSponsoredBudget, validateSponsoredBudget } from '../lib/sponsored-budget';

describe('sponsored run budget contract', () => {
  it('uses the fixed $5 cap and carries uncertain charges separately', () => {
    expect(validateSponsoredBudget({ spentUsd: 1.25, reservedUsd: 0.5, uncertainChargeUpperBoundUsd: 0.75 })).toMatchObject({
      capUsd: 5,
      accountedUsd: 2.5,
      remainingUsd: 2.5,
      withinCap: true,
    });
  });

  it('rejects invalid amounts and cap overrides', () => {
    expect(() => validateSponsoredBudget({ capUsd: 6 })).toThrow();
    expect(() => validateSponsoredBudget({ spentUsd: Number.NaN })).toThrow();
    expect(() => validateSponsoredBudget({ reservedUsd: -0.1 })).toThrow();
  });

  it('does not reserve beyond the conservative accounted total', () => {
    expect(canReserveSponsoredBudget({ spentUsd: 4.8 }, 0.2)).toBe(true);
    expect(canReserveSponsoredBudget({ spentUsd: 4.8 }, 0.21)).toBe(false);
    expect(canReserveSponsoredBudget({ spentUsd: 5.1 }, 0)).toBe(false);
  });
});

