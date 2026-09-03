import fs from 'node:fs';

const payload = JSON.parse(process.env.JOB_PAYLOAD ?? '{}');
const startedAt = new Date().toISOString();
const MAX_CHECKS = Math.min(5_000_000, Math.max(1_000, Number(payload.params?.maxChecks ?? 500_000)));
let checks = 0;

function sqrt(n) {
  if (n < 0n) throw new Error('negative square root');
  if (n < 2n) return n;
  let x = 1n << BigInt(Math.ceil(n.toString(2).length / 2));
  let y = (x + n / x) >> 1n;
  while (y < x) { x = y; y = (x + n / x) >> 1n; }
  return x;
}

function exactCell(n, d) {
  const m = sqrt(d * d + 4n * n);
  return m * m === d * d + 4n * n;
}

function completions(d1, d2, limit = 2000) {
  const delta = d2 * d2 - d1 * d1;
  if (delta <= 0n) return [];
  const results = [];
  const ceiling = sqrt(delta);
  for (let u = 1n; u <= ceiling && checks < MAX_CHECKS && results.length < limit; u += 1n) {
    checks += 1;
    if (delta % u !== 0n) continue;
    const v = delta / u;
    if ((u + v) % 2n !== 0n) continue;
    const m1 = (v - u) / 2n;
    const b = m1 * m1 - d1 * d1;
    if (b > 0n && b % 4n === 0n) results.push(b / 4n);
  }
  return [...new Set(results.map(String))].map(BigInt);
}

function parseDifferences(params) {
  if (Array.isArray(params.differences)) return params.differences.map(String).filter((x) => /^[1-9][0-9]*$/.test(x)).map(BigInt).slice(0, 80);
  if (typeof params.differences === 'string') return params.differences.split(/[ ,]+/).filter((x) => /^[1-9][0-9]*$/.test(x)).map(BigInt).slice(0, 80);
  return [];
}

function rank(numbers, differences) {
  return numbers.map((n) => ({
    number: n.toString(),
    support: differences.filter((d) => exactCell(n, d)).map(String),
  })).sort((a, b) => b.support.length - a.support.length || a.number.localeCompare(b.number)).slice(0, 50);
}

function divisorJob(params) {
  const d1 = BigInt(String(params.d1));
  const d2 = BigInt(String(params.d2));
  const numbers = completions(d1, d2, Number(params.limit ?? 5000));
  return { d1: String(d1), d2: String(d2), count: numbers.length, numbers: numbers.slice(0, 500).map(String), complete: checks < MAX_CHECKS };
}

function familyJob(params) {
  const differences = parseDifferences(params);
  const numbers = new Map();
  for (let i = 0; i < differences.length && checks < MAX_CHECKS; i += 1) for (let j = i + 1; j < differences.length && checks < MAX_CHECKS; j += 1) {
    for (const n of completions(differences[i], differences[j], 400)) numbers.set(String(n), n);
  }
  return { differences: differences.map(String), candidates: rank([...numbers.values()], differences), distinctNumbers: numbers.size, complete: checks < MAX_CHECKS };
}

function boundaryJob(params) {
  const start = Math.max(1, Number(params.startDifference ?? 161));
  const end = Math.min(start + 5000, Math.max(start, Number(params.endDifference ?? start + 300)));
  const stride = Math.max(1, Number(params.stride ?? 1));
  const differences = [];
  for (let d = start; d <= end; d += stride) differences.push(BigInt(d));
  return familyJob({ differences });
}

try {
  let result;
  if (payload.jobType === 'divisor_completion') result = divisorJob(payload.params ?? {});
  else if (payload.jobType === 'family_scan') result = familyJob(payload.params ?? {});
  else if (payload.jobType === 'boundary_scan') result = boundaryJob(payload.params ?? {});
  else throw new Error(`Unsupported job type: ${payload.jobType}`);
  fs.writeFileSync('result.json', JSON.stringify({ id: payload.id, ok: true, result: { ...result, checks, startedAt, completedAt: new Date().toISOString() } }));
} catch (error) {
  fs.writeFileSync('result.json', JSON.stringify({ id: payload.id, ok: false, error: error instanceof Error ? error.message : String(error), result: { checks, startedAt, completedAt: new Date().toISOString() } }));
  process.exitCode = 1;
}
