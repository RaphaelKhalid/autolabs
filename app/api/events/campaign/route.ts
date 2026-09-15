import { NextResponse } from 'next/server';

export async function GET() {
  const workerUrl = process.env.ORCHESTRATOR_URL ?? process.env.NEXT_PUBLIC_ORCHESTRATOR_URL;
  if (!workerUrl) return NextResponse.json({ error: 'The event ledger is not configured yet.' }, { status: 503 });
  const response = await fetch(`${workerUrl}/api/events/campaign`, { cache: 'no-store' });
  const payload = await response.json().catch(() => ({ error: 'Invalid ledger response.' }));
  return NextResponse.json(payload, { status: response.status });
}
