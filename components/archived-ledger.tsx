'use client';
import { useState } from 'react';
import { pilotLedgerUrl } from '@/lib/experiment-catalog';
import type { ExperimentEvent } from '@/lib/experiment';
export function ArchivedLedger() {
  const [events, setEvents] = useState<ExperimentEvent[]>([]);
  const [before, setBefore] = useState<number | null | undefined>(undefined);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  async function load() {
    if (loading || before === null) return;
    setLoading(true); setError('');
    try {
      const url = new URL(pilotLedgerUrl);
      url.searchParams.set('limit', '100');
      if (before !== undefined) url.searchParams.set('before', String(before));
      const response = await fetch(url, { signal: AbortSignal.timeout(15000) });
      if (!response.ok) throw new Error(`Archive returned ${response.status}. Please retry.`);
      const page = await response.json() as { events: ExperimentEvent[]; nextBefore: number | null };
      setEvents((current) => [...new Map([...current, ...page.events].map(event => [event.seq, event])).values()].sort((a, b) => b.seq - a.seq));
      setBefore(page.nextBefore);
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Could not load the archive.'); }
    finally { setLoading(false); }
  }
  return <div className="archived-ledger">
    <p className="archive-caption" aria-live="polite">{events.length ? `${events.length} events loaded, newest first.` : 'Load records on demand. Viewing the archive does not start agents or computations.'}</p>
    {events.map(event => <details key={event.seq}><summary><span>R{event.round} · {event.agentId ?? 'lab'} · {event.kind}</span>{event.title}</summary><p>{event.summary}</p><time>{event.at}</time>{event.payload != null && <pre>{JSON.stringify(event.payload, null, 2)}</pre>}</details>)}
    {error && <p role="alert">{error}</p>}
    {before !== null ? <button className="archive-button" onClick={() => void load()} disabled={loading}>{loading ? 'Loading records…' : events.length ? 'Load earlier events' : 'Load pilot ledger'}</button> : <p>All available events loaded.</p>}
  </div>;
}
