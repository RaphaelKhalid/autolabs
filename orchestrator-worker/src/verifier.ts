import { factorizationFromDifference } from './exact-verifier';
import type { CandidateInput } from './types';

export interface RectangleCheck {
  accepted: boolean;
  numbers: string[];
  differences: string[];
  support: number[];
  totalSupport: number;
  exactCells: number;
  missing: { number: string; difference: string }[];
  isK5: boolean;
  improvesSota: boolean;
  rejection?: string;
}

function uniquePositive(values: string[]) {
  return values.length === new Set(values).size && values.every((value) => /^[1-9][0-9]*$/.test(value));
}

export function verifyRectangle(candidate: CandidateInput): RectangleCheck {
  const numbers = candidate.numbers.slice(0, 12);
  const differences = candidate.differences.slice(0, 12);
  if (!numbers.length || !differences.length || !uniquePositive(numbers) || !uniquePositive(differences)) {
    return { accepted: false, numbers, differences, support: [], totalSupport: 0, exactCells: 0, missing: [], isK5: false, improvesSota: false, rejection: 'Values must be distinct positive decimal integers.' };
  }
  const support: number[] = [];
  const missing: { number: string; difference: string }[] = [];
  for (const number of numbers) {
    let hits = 0;
    for (const difference of differences) {
      try {
        if (factorizationFromDifference(number, difference)) hits += 1;
        else missing.push({ number, difference });
      } catch {
        missing.push({ number, difference });
      }
    }
    support.push(hits);
  }
  const exactCells = support.reduce((sum, value) => sum + value, 0);
  const complete = exactCells === numbers.length * differences.length;
  return {
    accepted: true,
    numbers,
    differences,
    support: [...support].sort((a, b) => a - b),
    totalSupport: exactCells,
    exactCells,
    missing,
    isK5: complete && numbers.length >= 5 && differences.length >= 5,
    improvesSota: complete && ((numbers.length >= 6 && differences.length >= 4) || (numbers.length >= 4 && differences.length >= 5)),
  };
}

export function compareSupport(a: readonly number[], b: readonly number[]) {
  const length = Math.max(a.length, b.length);
  for (let index = 0; index < length; index += 1) {
    const av = a[index] ?? -1;
    const bv = b[index] ?? -1;
    if (av !== bv) return av - bv;
  }
  return 0;
}
