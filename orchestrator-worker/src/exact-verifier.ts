/** Exact bigint arithmetic for Erdos Problem 885 candidate tables. */
export type BigIntLike = bigint | number | string;
export interface FactorPair { readonly a: bigint; readonly b: bigint; }
export interface FactorPairWitness extends FactorPair { readonly n: BigIntLike; readonly difference: BigIntLike; }
export type RejectionCode = "wrong-number-count" | "wrong-difference-count" | "non-positive-number" | "non-positive-difference" | "duplicate-number" | "duplicate-difference" | "invalid-factor-pair" | "factor-pair-not-in-witness" | "duplicate-factor-pair" | "difference-not-realized" | "identity-mismatch" | "invalid-table-number" | "invalid-table-difference" | "duplicate-table-number" | "duplicate-table-difference";
export interface Rejection { readonly code: RejectionCode; readonly detail: string; readonly n?: bigint; readonly difference?: bigint; readonly index?: number; }
export interface WitnessInput { readonly numbers: readonly BigIntLike[]; readonly differences: readonly BigIntLike[]; readonly factorPairs?: readonly FactorPairWitness[]; }
export interface VerifiedCell { readonly n: bigint; readonly difference: bigint; readonly m: bigint; readonly factorPair: FactorPair; }
export interface WitnessVerification { readonly accepted: boolean; readonly numbers: readonly bigint[]; readonly differences: readonly bigint[]; readonly cells: readonly VerifiedCell[]; readonly rejections: readonly Rejection[]; }
export interface CandidateTable { readonly numbers: readonly BigIntLike[]; readonly differences: readonly BigIntLike[]; }
export interface CandidateTableScore { readonly accepted: boolean; readonly numbers: readonly bigint[]; readonly differences: readonly bigint[]; readonly supports: readonly (readonly boolean[])[]; readonly rowSupports: readonly number[]; readonly columnSupports: readonly number[]; readonly weakestColumnMetric: readonly number[]; readonly totalSupport: number; readonly rejections: readonly Rejection[]; }
export interface NearMissOptions { readonly targetK?: number; readonly limit?: number; }

export function toPositiveInteger(value: BigIntLike, label = "integer"): bigint {
  if (typeof value === "bigint") { if (value > 0n) return value; throw new RangeError(`${label} must be positive`); }
  if (typeof value === "number") { if (Number.isSafeInteger(value) && value > 0) return BigInt(value); throw new RangeError(`${label} must be a positive safe integer`); }
  if (typeof value === "string" && /^\+?[0-9]+$/.test(value)) { const n = BigInt(value); if (n > 0n) return n; }
  throw new RangeError(`${label} must be a positive integer`);
}
function positiveOrUndefined(value: BigIntLike, label: string): bigint | undefined { try { return toPositiveInteger(value, label); } catch { return undefined; } }
export function integerSqrt(n: bigint): bigint {
  if (n < 0n) throw new RangeError("square root of a negative integer"); if (n < 2n) return n;
  let x = 1n << BigInt(Math.ceil(n.toString(2).length / 2)); let y = (x + n / x) >> 1n;
  while (y < x) { x = y; y = (x + n / x) >> 1n; } return x;
}
export function isPerfectSquare(n: bigint): boolean { if (n < 0n) return false; const root = integerSqrt(n); return root * root === n; }
export function factorPairs(value: BigIntLike): FactorPair[] {
  const n = toPositiveInteger(value, "N"); const result: FactorPair[] = [];
  for (let a = 1n; a * a <= n; a++) { if (n % a === 0n) { const b = n / a; if (a < b) result.push({ a, b }); } }
  return result;
}
export function differenceSet(value: BigIntLike): bigint[] { return factorPairs(value).map(({ a, b }) => b - a); }
const cellKey = (n: bigint, d: bigint) => `${n}:${d}`;
export function factorizationFromDifference(nValue: BigIntLike, dValue: BigIntLike): VerifiedCell | undefined {
  const n = toPositiveInteger(nValue, "N"); const difference = toPositiveInteger(dValue, "difference");
  const radicand = difference * difference + 4n * n; const m = integerSqrt(radicand);
  if (m * m !== radicand || m < difference || (m - difference) % 2n !== 0n) return undefined;
  const a = (m - difference) / 2n; const b = (m + difference) / 2n;
  return a > 0n && a * b === n && b - a === difference ? { n, difference, m, factorPair: { a, b } } : undefined;
}
function structure(input: WitnessInput, k: number) {
  const numbers: bigint[] = [], differences: bigint[] = [], rejections: Rejection[] = [];
  if (input.numbers.length !== k) rejections.push({ code: "wrong-number-count", detail: `exactly ${k} N values are required` });
  if (input.differences.length !== k) rejections.push({ code: "wrong-difference-count", detail: `exactly ${k} positive differences are required` });
  input.numbers.forEach((raw, index) => { const n = positiveOrUndefined(raw, `N[${index}]`); if (n === undefined) rejections.push({ code: "non-positive-number", detail: `N[${index}] is not positive`, index }); else numbers.push(n); });
  input.differences.forEach((raw, index) => { const d = positiveOrUndefined(raw, `difference[${index}]`); if (d === undefined) rejections.push({ code: "non-positive-difference", detail: `difference[${index}] is not positive`, index }); else differences.push(d); });
  const ns = new Set<bigint>(), ds = new Set<bigint>();
  numbers.forEach((n, index) => { if (ns.has(n)) rejections.push({ code: "duplicate-number", detail: `N ${n} is repeated`, n, index }); ns.add(n); });
  differences.forEach((d, index) => { if (ds.has(d)) rejections.push({ code: "duplicate-difference", detail: `difference ${d} is repeated`, difference: d, index }); ds.add(d); });
  return { numbers, differences, rejections };
}
export function verifyCommonWitness(input: WitnessInput, expectedK: number): WitnessVerification {
  if (!Number.isInteger(expectedK) || expectedK <= 0) throw new RangeError("expectedK must be positive");
  const parsed = structure(input, expectedK); const rejections = [...parsed.rejections]; const cells: VerifiedCell[] = [];
  if (rejections.length) return { accepted: false, numbers: parsed.numbers, differences: parsed.differences, cells, rejections };
  const explicit = new Map<string, FactorPairWitness>(); const supplied = input.factorPairs ?? [];
  supplied.forEach((raw, index) => {
    const n = positiveOrUndefined(raw.n, "factor pair N"), d = positiveOrUndefined(raw.difference, "factor pair difference"), a = positiveOrUndefined(raw.a, "factor pair a"), b = positiveOrUndefined(raw.b, "factor pair b");
    if (n === undefined || d === undefined || a === undefined || b === undefined || a * b !== n || b - a !== d) { rejections.push({ code: "invalid-factor-pair", detail: `factor pair at index ${index} is not exact`, index }); return; }
    const k = cellKey(n, d); if (explicit.has(k)) rejections.push({ code: "duplicate-factor-pair", detail: `factor pair for (${n}, ${d}) is repeated`, n, difference: d, index }); else explicit.set(k, { n, difference: d, a, b });
    if (!parsed.numbers.includes(n) || !parsed.differences.includes(d)) rejections.push({ code: "factor-pair-not-in-witness", detail: `factor pair at index ${index} is outside the witness`, n, difference: d, index });
  });
  for (const n of parsed.numbers) for (const d of parsed.differences) {
    const cell = factorizationFromDifference(n, d); if (!cell) { rejections.push({ code: "difference-not-realized", detail: `${d} is not in D(${n})`, n, difference: d }); continue; }
    const pair = explicit.get(cellKey(n, d));
    if (!pair) { cells.push(cell); continue; }
    const m = pair.a + pair.b; if (m * m !== d * d + 4n * n) { rejections.push({ code: "identity-mismatch", detail: `d^2 + 4N != m^2 for (${n}, ${d})`, n, difference: d }); continue; }
    cells.push({ n, difference: d, m, factorPair: { a: pair.a, b: pair.b } });
  }
  return { accepted: rejections.length === 0, numbers: parsed.numbers, differences: parsed.differences, cells, rejections };
}
export function verifyWitness(input: WitnessInput): WitnessVerification { return verifyCommonWitness(input, 5); }

function tableStructure(input: CandidateTable) {
  const numbers: bigint[] = [], differences: bigint[] = [], rejections: Rejection[] = [];
  input.numbers.forEach((raw, index) => { const n = positiveOrUndefined(raw, `table N[${index}]`); if (n === undefined) rejections.push({ code: "invalid-table-number", detail: `table N[${index}] is invalid`, index }); else numbers.push(n); });
  input.differences.forEach((raw, index) => { const d = positiveOrUndefined(raw, `table difference[${index}]`); if (d === undefined) rejections.push({ code: "invalid-table-difference", detail: `table difference[${index}] is invalid`, index }); else differences.push(d); });
  const ns = new Set<bigint>(), ds = new Set<bigint>(); numbers.forEach((n, index) => { if (ns.has(n)) rejections.push({ code: "duplicate-table-number", detail: `table N ${n} is repeated`, n, index }); ns.add(n); }); differences.forEach((d, index) => { if (ds.has(d)) rejections.push({ code: "duplicate-table-difference", detail: `table difference ${d} is repeated`, difference: d, index }); ds.add(d); });
  return { numbers, differences, rejections };
}
export function scoreCandidateTable(input: CandidateTable, targetK = 5): CandidateTableScore {
  if (!Number.isInteger(targetK) || targetK <= 0) throw new RangeError("targetK must be positive");
  const parsed = tableStructure(input), supports: boolean[][] = [], cache = new Map<bigint, Set<bigint>>();
  for (const n of parsed.numbers) { let set = cache.get(n); if (!set) { set = new Set(differenceSet(n)); cache.set(n, set); } supports.push(parsed.differences.map((d) => set!.has(d))); }
  const rowSupports = supports.map((row) => row.filter(Boolean).length), columnSupports = parsed.differences.map((_, i) => supports.filter((row) => row[i]).length), weakestColumnMetric = [...columnSupports].map((x) => Math.min(targetK, x)).sort((a, b) => a - b);
  return { accepted: parsed.rejections.length === 0, numbers: parsed.numbers, differences: parsed.differences, supports, rowSupports, columnSupports, weakestColumnMetric, totalSupport: rowSupports.reduce((a, b) => a + b, 0), rejections: parsed.rejections };
}
function compareVector(a: readonly number[], b: readonly number[]) { for (let i = 0; i < Math.max(a.length, b.length); i++) { const x = a[i] ?? -1, y = b[i] ?? -1; if (x !== y) return x > y ? 1 : -1; } return 0; }
export function compareCandidateScores(a: CandidateTableScore, b: CandidateTableScore) { return compareVector(a.weakestColumnMetric, b.weakestColumnMetric) || (a.totalSupport === b.totalSupport ? 0 : a.totalSupport > b.totalSupport ? 1 : -1) || compareVector([...a.rowSupports].sort((x, y) => x - y), [...b.rowSupports].sort((x, y) => x - y)); }
export function rankNearMisses(inputs: readonly CandidateTable[], options: NearMissOptions = {}): CandidateTableScore[] {
  const result = inputs.map((input) => scoreCandidateTable(input, options.targetK ?? 5)); result.sort((a, b) => { const order = compareCandidateScores(b, a); return order || `${a.numbers.join(",")}|${a.differences.join(",")}`.localeCompare(`${b.numbers.join(",")}|${b.differences.join(",")}`); }); return options.limit === undefined ? result : result.slice(0, Math.max(0, options.limit));
}
function dominates(a: CandidateTableScore, b: CandidateTableScore) { if (a.columnSupports.length !== b.columnSupports.length || a.rowSupports.length !== b.rowSupports.length) return false; let strict = false; for (let i = 0; i < a.columnSupports.length; i++) { if (a.columnSupports[i] < b.columnSupports[i]) return false; strict ||= a.columnSupports[i] > b.columnSupports[i]; } for (let i = 0; i < a.rowSupports.length; i++) { if (a.rowSupports[i] < b.rowSupports[i]) return false; strict ||= a.rowSupports[i] > b.rowSupports[i]; } return strict; }
export function paretoFront(scores: readonly CandidateTableScore[]) { return scores.filter((score, i) => !scores.some((other, j) => i !== j && dominates(other, score))); }

export const BREMNER_K4_FIXTURE: Readonly<WitnessInput> = { numbers: [26128575n, 291722431n, 561117375n, 713526975n], differences: [126n, 16110n, 33390n, 75390n] };
