import { NextResponse } from 'next/server';

export async function GET(_request: Request, { params }: { params: Promise<{ id: string }> }) {
  const workerUrl = process.env.ORCHESTRATOR_URL ?? process.env.NEXT_PUBLIC_ORCHESTRATOR_URL;
  if (!workerUrl) return NextResponse.json({ error: 'The event ledger is not configured yet.' }, { status: 503 });
  const { id } = await params;
  if (!/^event-[a-zA-Z0-9-]+$/.test(id)) return NextResponse.json({ error: 'Invalid run id.' }, { status: 400 });
  const response = await fetch(`${workerUrl}/api/events/runs/${id}`, { cache: 'no-store' });
  const payload = await response.json().catch(() => ({ error: 'Invalid ledger response.' }));
  return NextResponse.json(payload, { status: response.status });
}
