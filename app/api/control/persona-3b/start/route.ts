import { createHash, timingSafeEqual } from 'node:crypto';
import { NextResponse } from 'next/server';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const STUDY_ID = 'experiment-003b-v1';
const PHASE = 'development-screen-scoring';
const SOURCE_ARTIFACT_HASH = '6e479c2e02aa912f6bcf6b2e57554ed75711e93bb3fc3da813114bba871f3fe7';
const CALL_CEILING = 1_014;
const BUDGET_USD = 10;

function secretEquals(left: string, right: string) {
  const a = createHash('sha256').update(left).digest();
  const b = createHash('sha256').update(right).digest();
  return timingSafeEqual(a, b);
}

async function requestBody(request: Request) {
  const contentLength = Number(request.headers.get('content-length') ?? 0);
  if (!Number.isFinite(contentLength) || contentLength > 2_048) return {};
  const text = await request.text();
  if (new TextEncoder().encode(text).byteLength > 2_048 || text.trim() === '') return {};
  try {
    const value = JSON.parse(text) as unknown;
    return value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : {};
  } catch {
    return {};
  }
}

export async function POST(request: Request) {
  const ownerToken = process.env.AUTOLABS_3B_START_TOKEN ?? process.env.AUTOLABS_ADMIN_TOKEN;
  const workerToken = process.env.PERSONA_3B_WORKER_TOKEN ?? process.env.AUTOLABS_ADMIN_TOKEN;
  const workerUrl = process.env.ORCHESTRATOR_URL ?? process.env.NEXT_PUBLIC_ORCHESTRATOR_URL;
  const manifestHash = process.env.PERSONA_3B_MANIFEST_HASH;
  if (!ownerToken || !workerToken || !workerUrl || !manifestHash) return NextResponse.json({ error: 'Experiment 3B control is not configured.' }, { status: 503 });

  const supplied = request.headers.get('x-autolabs-owner-key') ?? '';
  if (!supplied || !secretEquals(supplied, ownerToken)) return NextResponse.json({ error: 'Owner key required.' }, { status: 401 });

  const raw = await requestBody(request);
  const suppliedKey = raw.idempotencyKey;
  const idempotencyKey = typeof suppliedKey === 'string' && suppliedKey.length >= 16 && suppliedKey.length <= 120
    ? suppliedKey
    : crypto.randomUUID();

  try {
    const response = await fetch(`${workerUrl.replace(/\/$/, '')}/api/persona-3b/start`, {
      method: 'POST',
      signal: AbortSignal.timeout(10_000),
      headers: { authorization: `Bearer ${workerToken}`, 'content-type': 'application/json' },
      body: JSON.stringify({ studyId: STUDY_ID, phase: PHASE, manifestHash, sourceArtifactHash: SOURCE_ARTIFACT_HASH, callCeiling: CALL_CEILING, budgetUsd: BUDGET_USD, idempotencyKey }),
    });
    const payload = await response.json().catch(() => ({}));
    return NextResponse.json(payload, { status: response.status, headers: { 'cache-control': 'no-store' } });
  } catch {
    return NextResponse.json({ error: 'The orchestrator is unavailable; no scoring launch result was assumed.' }, { status: 502 });
  }
}
