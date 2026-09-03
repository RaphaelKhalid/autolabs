import fs from 'node:fs';

const MAX_DECIMAL_DIGITS = 120;
const MAX_DIFFERENCES = 80;
const MAX_RESULT_NUMBERS = 500;
const MAX_RANK_CANDIDATES = 10_000;
const DEFAULT_CHECKS = 500_000;
const HARD_CHECK_LIMIT = 5_000_000;
const WALL_LIMIT_MS = 35 * 60_000;
const startedAt = new Date().toISOString();
const deadline = Date.now() + WALL_LIMIT_MS;
let checks = 0;
let maxChecks = DEFAULT_CHECKS;
let payload = {};

function positiveDecimal(value, label) {
  const text = String(value ?? '');
  if (!/^[1-9][0-9]*$/.test(text) || text.length > MAX_DECIMAL_DIGITS) {
    throw new Error(`${label} must be a positive decimal with at most ${MAX_DECIMAL_DIGITS} digits`);
  }
  return BigInt(text);
}

function boundedInteger(value, label, fallback, minimum, maximum) {
  if (value === null || value === undefined) return fallback;
  const number = Number(value);
  if (!Number.isSafeInteger(number) || number < minimum || number > maximum) {
    throw new Error(`${label} must be a safe integer from ${minimum} to ${maximum}`);
  }
  return number;
}

function completionLimit(value, fallback = 2_000) {
  if (value === null || value === undefined) return fallback;
  const number = Number(value);
  if (!Number.isFinite(number) || number <= 0) throw new Error('limit must be a positive number');
  return Math.min(5_000, Math.max(1, Math.floor(number)));
}
function consumeCheck() {
  if (checks >= maxChecks || Date.now() >= deadline) return false;
  checks += 1;
  return true;
}

function stopReason() {
  if (Date.now() >= deadline) return 'wall_time_limit';
  if (checks >= maxChecks) return 'check_limit';
  return null;
}

function sqrt(n) {
  if (n < 0n) throw new Error('negative square root');
  if (n < 2n) return n;
  let x = 1n << BigInt(Math.ceil(n.toString(2).length / 2));
  let y = (x + n / x) >> 1n;
  while (y < x) {
    x = y;
    y = (x + n / x) >> 1n;
  }
  return x;
}

function exactCellUnchecked(n, d) {
  const radicand = d * d + 4n * n;
  const m = sqrt(radicand);
  return m * m === radicand;
}

function exactCell(n, d) {
  if (!consumeCheck()) return null;
  return exactCellUnchecked(n, d);
}

function completions(rawD1, rawD2, requestedLimit = 2_000) {
  const d1 = rawD1 < rawD2 ? rawD1 : rawD2;
  const d2 = rawD1 < rawD2 ? rawD2 : rawD1;
  const limit = completionLimit(requestedLimit);
  const delta = d2 * d2 - d1 * d1;
  if (delta <= 0n) return { numbers: [], complete: true, truncatedBy: null };
  const results = new Map();
  const ceiling = sqrt(delta);
  let u = 1n;

  for (; u <= ceiling && results.size < limit; u += 1n) {
    if (!consumeCheck()) break;
    if (delta % u !== 0n) continue;
    const v = delta / u;
    if ((u + v) % 2n !== 0n) continue;
    const m1 = (v - u) / 2n;
    const fourN = m1 * m1 - d1 * d1;
    if (fourN <= 0n || fourN % 4n !== 0n) continue;
    const n = fourN / 4n;
    if (exactCellUnchecked(n, d1) && exactCellUnchecked(n, d2)) results.set(n.toString(), n);
  }

  const exhaustedDivisors = u > ceiling;
  const hitResultLimit = results.size >= limit && !exhaustedDivisors;
  return {
    numbers: [...results.values()],
    complete: exhaustedDivisors && !stopReason(),
    truncatedBy: hitResultLimit ? 'result_limit' : stopReason(),
  };
}

function parseDifferences(params) {
  const raw = Array.isArray(params.differences)
    ? params.differences
    : typeof params.differences === 'string'
      ? params.differences.split(/[ ,]+/)
      : [];
  const unique = new Map();
  for (const value of raw) {
    if (value === '') continue;
    const difference = positiveDecimal(value, 'difference');
    unique.set(difference.toString(), difference);
  }
  const values = [...unique.values()].sort((a, b) => a < b ? -1 : a > b ? 1 : 0);
  if (values.length > MAX_DIFFERENCES) throw new Error(`At most ${MAX_DIFFERENCES} distinct differences are accepted`);
  if (values.length < 2) throw new Error('At least two distinct differences are required');
  return values;
}

function rank(numbers, differences) {
  const ranked = [];
  let complete = true;
  let truncatedBy = null;
  const bounded = numbers.slice(0, MAX_RANK_CANDIDATES);
  if (numbers.length > bounded.length) {
    complete = false;
    truncatedBy = 'candidate_limit';
  }

  for (const n of bounded) {
    const support = [];
    for (const difference of differences) {
      const exact = exactCell(n, difference);
      if (exact === null) {
        complete = false;
        truncatedBy = stopReason();
        break;
      }
      if (exact) support.push(difference.toString());
    }
    ranked.push({ number: n.toString(), support });
    if (!complete && stopReason()) break;
  }

  ranked.sort((a, b) => b.support.length - a.support.length || a.number.localeCompare(b.number));
  return { candidates: ranked.slice(0, 50), complete, truncatedBy };
}

function divisorJob(params) {
  const d1 = positiveDecimal(params.d1, 'd1');
  const d2 = positiveDecimal(params.d2, 'd2');
  if (d1 === d2) throw new Error('d1 and d2 must be distinct');
  const found = completions(d1, d2, params.limit);
  return {
    d1: d1.toString(),
    d2: d2.toString(),
    count: found.numbers.length,
    numbers: found.numbers.slice(0, MAX_RESULT_NUMBERS).map(String),
    complete: found.complete && found.numbers.length <= MAX_RESULT_NUMBERS,
    truncatedBy: found.numbers.length > MAX_RESULT_NUMBERS ? 'output_limit' : found.truncatedBy,
  };
}

function familyJob(params) {
  const differences = parseDifferences(params);
  const numbers = new Map();
  let complete = true;
  let truncatedBy = null;

  outer: for (let i = 0; i < differences.length; i += 1) {
    for (let j = i + 1; j < differences.length; j += 1) {
      if (stopReason()) {
        complete = false;
        truncatedBy = stopReason();
        break outer;
      }
      const found = completions(differences[i], differences[j], 400);
      for (const n of found.numbers) {
        numbers.set(n.toString(), n);
        if (numbers.size >= MAX_RANK_CANDIDATES) {
          complete = false;
          truncatedBy = 'candidate_limit';
          break outer;
        }
      }
      if (!found.complete) {
        complete = false;
        truncatedBy = found.truncatedBy;
        break outer;
      }
    }
  }

  const ranked = rank([...numbers.values()], differences);
  return {
    differences: differences.map(String),
    candidates: ranked.candidates,
    distinctNumbers: numbers.size,
    complete: complete && ranked.complete,
    truncatedBy: truncatedBy ?? ranked.truncatedBy,
  };
}

function boundaryJob(params) {
  const start = boundedInteger(params.startDifference, 'startDifference', 161, 1, Number.MAX_SAFE_INTEGER - 5_000);
  const end = boundedInteger(params.endDifference, 'endDifference', start + 300, start, start + 5_000);
  const stride = boundedInteger(params.stride, 'stride', 1, 1, 5_000);
  const differences = [];
  for (let d = start; d <= end && differences.length < MAX_DIFFERENCES; d += stride) differences.push(String(d));
  return familyJob({ differences });
}

function safeWrite(result) {
  const encoded = JSON.stringify(result);
  if (Buffer.byteLength(encoded, 'utf8') > 900_000) {
    throw new Error('Result exceeded the 900KB callback limit');
  }
  fs.writeFileSync('result.json', encoded);
}

try {
  payload = JSON.parse(process.env.JOB_PAYLOAD ?? '{}');
  if (!payload || typeof payload !== 'object') throw new Error('JOB_PAYLOAD must be an object');
  if (!/^[a-zA-Z0-9-]{1,240}$/.test(String(payload.id ?? ''))) throw new Error('Invalid job id');
  if (!/^[0-9a-f]{40}$/.test(String(payload.sourceSha ?? ''))) throw new Error('Invalid source revision');
  maxChecks = boundedInteger(payload.params?.maxChecks, 'maxChecks', DEFAULT_CHECKS, 1_000, HARD_CHECK_LIMIT);

  let result;
  if (payload.jobType === 'divisor_completion') result = divisorJob(payload.params ?? {});
  else if (payload.jobType === 'family_scan') result = familyJob(payload.params ?? {});
  else if (payload.jobType === 'boundary_scan') result = boundaryJob(payload.params ?? {});
  else throw new Error(`Unsupported job type: ${payload.jobType}`);

  safeWrite({
    id: payload.id,
    ok: true,
    complete: Boolean(result.complete),
    result: { ...result, sourceSha: payload.sourceSha, checks, startedAt, completedAt: new Date().toISOString() },
  });
} catch (error) {
  safeWrite({
    id: String(payload?.id ?? 'unknown'),
    ok: false,
    complete: false,
    error: error instanceof Error ? error.message : String(error),
    result: { sourceSha: String(payload?.sourceSha ?? ''), checks, startedAt, completedAt: new Date().toISOString() },
  });
  process.exitCode = 1;
}
