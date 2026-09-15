/**
 * Conservative accounting for the small sponsored test runs.
 *
 * Provider usage can be missing when a request fails or times out, so callers
 * must carry an explicit uncertain upper bound instead of silently treating
 * an unknown charge as zero.
 */
export const SPONSORED_RUN_CAP_USD = 5;

export interface SponsoredBudgetInput {
  capUsd?: unknown;
  spentUsd?: unknown;
  reservedUsd?: unknown;
  uncertainChargeUpperBoundUsd?: unknown;
}

export interface SponsoredBudget {
  capUsd: number;
  spentUsd: number;
  reservedUsd: number;
  uncertainChargeUpperBoundUsd: number;
  accountedUsd: number;
  remainingUsd: number;
  withinCap: boolean;
}

function nonNegativeFinite(value: unknown, fallback: number) {
  if (value === undefined) return fallback;
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : null;
}

/** Validate a budget snapshot against the fixed $5 sponsored-run contract. */
export function validateSponsoredBudget(input: SponsoredBudgetInput = {}): SponsoredBudget {
  const capUsd = nonNegativeFinite(input.capUsd, SPONSORED_RUN_CAP_USD);
  const spentUsd = nonNegativeFinite(input.spentUsd, 0);
  const reservedUsd = nonNegativeFinite(input.reservedUsd, 0);
  const uncertainChargeUpperBoundUsd = nonNegativeFinite(input.uncertainChargeUpperBoundUsd, 0);
  if (capUsd === null || capUsd !== SPONSORED_RUN_CAP_USD) throw new Error('Sponsored runs must use the fixed $5 cap.');
  if (spentUsd === null || reservedUsd === null || uncertainChargeUpperBoundUsd === null) {
    throw new Error('Budget amounts must be finite, non-negative numbers.');
  }
  const accountedUsd = spentUsd + reservedUsd + uncertainChargeUpperBoundUsd;
  return {
    capUsd,
    spentUsd,
    reservedUsd,
    uncertainChargeUpperBoundUsd,
    accountedUsd,
    remainingUsd: Math.max(0, capUsd - accountedUsd),
    withinCap: accountedUsd <= capUsd,
  };
}

/** Guard a new reservation without claiming that provider billing is exact. */
export function canReserveSponsoredBudget(input: SponsoredBudgetInput, reservationUsd: number) {
  if (!Number.isFinite(reservationUsd) || reservationUsd < 0) return false;
  const snapshot = validateSponsoredBudget(input);
  return snapshot.accountedUsd + reservationUsd <= snapshot.capUsd;
}

