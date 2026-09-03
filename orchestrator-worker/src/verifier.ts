import { factorizationFromDifference } from './exact-verifier';
import type { CandidateInput } from './types';

export interface RectangleCheck {
  accepted: boolean;
  numbers: string[];
  differences: string[];
  support: number[];
  columnSupport: number[];
  progressMetric: number[];
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
    return { accepted: false, numbers, differences, support: [], columnSupport: [], progressMetric: [], totalSupport: 0, exactCells: 0, missing: [], isK5: false, improvesSota: false, rejection: 'Values must be distinct positive decimal integers.' };
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
  const missingKeys = new Set(missing.map((cell) => `${cell.number}:${cell.difference}`));
  const columnSupport = differences.map((difference) => numbers.reduce(
    (hits, number) => hits + (missingKeys.has(`${number}:${difference}`) ? 0 : 1),
    0,
  ));
  const progressMetric = rectangleProgressMetric(support, columnSupport, exactCells);
  const complete = exactCells === numbers.length * differences.length;
  return {
    accepted: true,
    numbers,
    differences,
    support: [...support].sort((a, b) => a - b),
    columnSupport: [...columnSupport].sort((a, b) => a - b),
    progressMetric,
    totalSupport: exactCells,
    exactCells,
    missing,
    isK5: complete && numbers.length >= 5 && differences.length >= 5,
    improvesSota: complete && ((numbers.length >= 6 && differences.length >= 4) || (numbers.length >= 4 && differences.length >= 5)),
  };
}

function supportFloor(values: readonly number[], size: number) {
  if (values.length < size) return 0;
  return [...values].sort((a, b) => b - a)[size - 1];
}

export function rectangleProgressMetric(rowSupport: readonly number[], columnSupport: readonly number[], exactCells: number) {
  const metric: number[] = [];
  for (let size = 5; size >= 1; size -= 1) {
    metric.push(Math.min(size, supportFloor(rowSupport, size), supportFloor(columnSupport, size)));
  }
  metric.push(exactCells);
  return metric;
}

export function compareProgressMetric(a: readonly number[], b: readonly number[]) {
  const length = Math.max(a.length, b.length);
  for (let index = 0; index < length; index += 1) {
    const av = a[index] ?? -1;
    const bv = b[index] ?? -1;
    if (av !== bv) return av - bv;
  }
  return 0;
}
