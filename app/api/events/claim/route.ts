import { NextResponse } from 'next/server';

export async function POST(request: Request) {
  const workerUrl = process.env.ORCHESTRATOR_URL ?? process.env.NEXT_PUBLIC_ORCHESTRATOR_URL;
  if (!workerUrl) return NextResponse.json({ error: 'The event ledger is not configured yet.' }, { status: 503 });
  const contentLength = Number(request.headers.get('content-length') ?? 0);
  if (Number.isFinite(contentLength) && contentLength > 32_768) return NextResponse.json({ error: 'Payload too large.' }, { status: 413 });
  const raw = await request.text();
  if (raw.length > 32_768) return NextResponse.json({ error: 'Payload too large.' }, { status: 413 });
  const response = await fetch(`${workerUrl}/api/events/claim`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: raw,
  });
  const payload = await response.json().catch(() => ({ error: 'Invalid ledger response.' }));
  return NextResponse.json(payload, { status: response.status });
}
