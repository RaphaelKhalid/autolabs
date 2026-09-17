'use client';

import Link from 'next/link';
import { useCallback, useEffect, useState } from 'react';
import { fetchPersona3CStatus, persona3cStageLabel, PERSONA_3C_STAGES, type Persona3CStatus } from '@/lib/persona-3c';

function number(value: number) { return new Intl.NumberFormat('en-US').format(value); }
function dollars(value: number) { return `$${value.toFixed(2)}`; }
function hours(value: number) { return `${value.toFixed(2)} h`; }

function statusTone(status: string) {
  if (status === 'complete') return 'is-complete';
  if (status === 'failed' || status === 'stopped') return 'is-failed';
  return 'is-live';
}

export function Persona3CProgress() {
  const [status, setStatus] = useState<Persona3CStatus>({ run: null });
  const [connection, setConnection] = useState<'loading' | 'ready' | 'error'>('loading');

  const reload = useCallback(async () => {
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 8_000);
    try {
      const next = await fetchPersona3CStatus(controller.signal);
      setStatus(next);
      setConnection('ready');
    } catch {
      setConnection('error');
    } finally {
      window.clearTimeout(timeout);
    }
  }, []);

  useEffect(() => {
    const initial = window.setTimeout(() => { void reload(); }, 0);
    return () => window.clearTimeout(initial);
  }, [reload]);
  useEffect(() => {
    const terminal = status.run && ['complete', 'failed', 'stopped'].includes(status.run.status);
    const timer = window.setInterval(() => { if (!terminal) void reload(); }, 5_000);
    return () => window.clearInterval(timer);
  }, [reload, status.run]);

  const run = status.run;

  if (connection === 'error') {
    return <section className="persona3c-card persona3c-quiet" aria-label="Experiment 3C status">
      <span className="persona3c-quiet-line">Experiment 3C: worker unavailable · <Link href="/experiments/persona-discovery-3c">pipeline walkthrough ↗</Link></span>
    </section>;
  }

  if (!run) {
    return <section className="persona3c-card persona3c-quiet" aria-label="Experiment 3C status">
      <span className="persona3c-quiet-line">{connection === 'loading' ? 'Experiment 3C: checking status…' : 'Experiment 3C: not started'} · <Link href="/experiments/persona-discovery-3c">pipeline walkthrough ↗</Link></span>
    </section>;
  }

  const progressByStage = new Map((status.progress ?? []).map((row) => [row.stage, row]));
  const events = (status.events ?? []).slice(-8).reverse();

  return <section className="persona3c-card" aria-live="polite" aria-label="Experiment 3C live progress">
    <div className="persona3c-top">
      <span className="persona3c-eyebrow">EXPERIMENT 3C · LIVE · <Link href="/experiments/persona-discovery-3c">PIPELINE WALKTHROUGH ↗</Link></span>
      <span className={`persona3c-status ${statusTone(run.status)}`}><i />{run.status}</span>
    </div>
    <div className="persona3c-headline">
      <strong>{persona3cStageLabel(run.stage)}</strong>
      <span>{run.id}</span>
    </div>
    <div className="persona3c-metrics">
      <div><span>GPU HOURS</span><strong>{hours(run.gpuHours)}</strong></div>
      <div><span>SPEND / CAP</span><strong>{dollars(run.spentUsd)} / {dollars(run.budgetUsd)}</strong></div>
      <div><span>LAST RECORD</span><strong>{run.lastRecordId ?? '—'}</strong></div>
    </div>
    {run.error && <p className="persona3c-error" role="alert">{run.error}</p>}
    <ol className="persona3c-stages">
      {PERSONA_3C_STAGES.map((stage) => {
        const row = progressByStage.get(stage);
        const done = row?.done ?? 0;
        const total = row?.total ?? 0;
        const active = run.stage === stage;
        return <li key={stage} className={active ? 'is-active' : row ? 'is-touched' : ''}>
          <span>{persona3cStageLabel(stage)}</span>
          <small>{row ? `${number(done)} / ${number(total)}` : '—'}</small>
        </li>;
      })}
    </ol>
    {events.length > 0 && <div className="persona3c-events">
      {events.map((event, index) => <article key={`${event.id ?? event.at}-${index}`}>
        <time dateTime={event.at}>{new Date(event.at).toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' })}</time>
        <div><b>{event.title}</b><p>{event.summary}</p></div>
      </article>)}
    </div>}
  </section>;
}
