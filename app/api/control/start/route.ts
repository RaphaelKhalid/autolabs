import { createHash, timingSafeEqual } from 'node:crypto';
import { NextResponse } from 'next/server';

function secretEquals(left: string, right: string) {
  const a = createHash('sha256').update(left).digest();
  const b = createHash('sha256').update(right).digest();
  return timingSafeEqual(a, b);
}

export async function POST(request: Request) {
  const workerUrl = process.env.ORCHESTRATOR_URL ?? process.env.NEXT_PUBLIC_ORCHESTRATOR_URL;
  const adminToken = process.env.AUTOLABS_ADMIN_TOKEN;
  if (!workerUrl || !adminToken) {
    return NextResponse.json({ error: 'The live engine is not configured yet.' }, { status: 503 });
  }
  const supplied = request.headers.get('x-autolabs-owner-key');
  if (!supplied || !secretEquals(supplied, adminToken)) {
    return NextResponse.json({ error: 'Owner key required.' }, { status: 401 });
  }
  const body = await request.json().catch(() => ({}));
  const response = await fetch(`${workerUrl}/api/experiments/start`, {
    method: 'POST',
    headers: { 'content-type': 'application/json', authorization: `Bearer ${adminToken}` },
    body: JSON.stringify(body),
  });
  const payload = await response.json().catch(() => ({ error: 'Invalid engine response.' }));
  return NextResponse.json(payload, { status: response.status });
}
