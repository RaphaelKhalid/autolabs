'use client';

import { useCallback, useEffect, useState } from 'react';
import { fetchPersona3CStatus, persona3cStageLabel, PERSONA_3C_STAGES, PERSONA_3C_STALE_AFTER_MINUTES, type Persona3CStatus } from '@/lib/persona-3c';

function number(value: number) { return new Intl.NumberFormat('en-US').format(value); }
function dollars(value: number) { return `$${value.toFixed(2)}`; }
function hours(value: number) { return `${value.toFixed(2)} h`; }
// h:mm:ss (or m:ss under an hour) for the live elapsed / ETA readouts.
function clock(totalSeconds: number) {
  const s = Math.max(0, Math.floor(totalSeconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const ss = s % 60;
  const pad = (n: number) => String(n).padStart(2, '0');
  return h > 0 ? `${h}:${pad(m)}:${pad(ss)}` : `${m}:${pad(ss)}`;
}

// The pod reports every 1M tokens and never reports GPU time, so between reports
// the card estimates: tokens from the rate of the last two train reports, GPU hours
// from elapsed wall clock, spend from the pod's hourly price. Estimates are marked
// with ≈ and snap to the reported value whenever a report lands.
const GPU_USD_PER_HOUR = 1.59; // NVIDIA A100 SXM 80 GB secure cloud, the current pod
const TRAIN_EVENT = /^train: (\d+)\/(\d+)/;

function trainRate(status: Persona3CStatus): { tokensPerSecond: number; at: number; done: number; total: number } | null {
  const samples = (status.events ?? [])
    .map((event) => ({ at: Date.parse(event.at), match: TRAIN_EVENT.exec(event.summary ?? '') }))
    .filter((row) => row.match && Number.isFinite(row.at))
    .map((row) => ({ at: row.at, done: Number(row.match![1]), total: Number(row.match![2]) }))
    .sort((a, b) => a.at - b.at);
  if (samples.length < 2) return null;
  const last = samples[samples.length - 1];
  const prev = samples[samples.length - 2];
  const seconds = (last.at - prev.at) / 1000;
  if (seconds <= 0 || last.done <= prev.done) return null;
  return { tokensPerSecond: (last.done - prev.done) / seconds, at: last.at, done: last.done, total: last.total };
}

function statusTone(status: string) {
  if (status === 'complete') return 'is-complete';
  if (status === 'failed' || status === 'stopped') return 'is-failed';
  return 'is-live';
}

export function Persona3CProgress() {
  const [status, setStatus] = useState<Persona3CStatus>({ run: null });
  const [connection, setConnection] = useState<'loading' | 'ready' | 'error'>('loading');
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, []);

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
      <span className="persona3c-quiet-line">Experiment 3C: worker unavailable</span>
    </section>;
  }

  if (!run) {
    return <section className="persona3c-card persona3c-quiet" aria-label="Experiment 3C status">
      <span className="persona3c-quiet-line">{connection === 'loading' ? 'Experiment 3C: checking status…' : 'Experiment 3C: not started'}</span>
    </section>;
  }

  const progressByStage = new Map((status.progress ?? []).map((row) => [row.stage, row]));
  const events = (status.events ?? []).slice(-8).reverse();
  const live = run.status === 'running';
  const rate = live ? trainRate(status) : null;
  const elapsedHours = Math.max(0, ((live ? now : Date.parse(run.completedAt ?? run.updatedAt)) - Date.parse(run.createdAt)) / 3_600_000);
  const gpuHoursShown = run.gpuHours > 0 ? run.gpuHours : elapsedHours;
  const spendShown = run.spentUsd > 0 ? run.spentUsd : elapsedHours * GPU_USD_PER_HOUR;
  const estimated = run.gpuHours <= 0 || run.spentUsd <= 0;
  const estimateTokens = (row: { done: number; total: number } | undefined) => {
    if (!row || !rate || run.stage !== 'train') return row?.done ?? 0;
    const extra = Math.max(0, (now - rate.at) / 1000) * rate.tokensPerSecond;
    return Math.min(row.total, Math.round(Math.max(row.done, rate.done) + extra));
  };

  // Overall pipeline progress: completed stages plus the active stage's own
  // fraction, so the bar advances every time a stage lands even when a single
  // stage (rank, analysis) reports atomically. The elapsed clock and the ETA
  // it drives are the always-moving numbers the card was missing.
  const stageCount = PERSONA_3C_STAGES.length;
  const stageIdx = Math.max(0, PERSONA_3C_STAGES.indexOf(run.stage as (typeof PERSONA_3C_STAGES)[number]));
  const activeRow = progressByStage.get(run.stage);
  const activeFrac = activeRow && activeRow.total > 0
    ? Math.min(1, activeRow.done / activeRow.total)
    : (run.status === 'complete' ? 1 : 0);
  const overall = run.status === 'complete' ? 1 : Math.min(1, (stageIdx + activeFrac) / stageCount);
  const overallPct = Math.round(overall * 100);
  const stageNumber = Math.min(stageCount, stageIdx + 1);
  const endMs = live ? now : Date.parse(run.completedAt ?? run.updatedAt);
  const elapsedSeconds = Math.max(0, (endMs - Date.parse(run.createdAt)) / 1000);
  const etaSeconds = live && overall > 0.05 && overall < 0.995 ? (elapsedSeconds * (1 - overall)) / overall : null;
  const judge = status.judge;
  const judgeTotal = judge ? judge.queued + judge.inFlight + judge.complete + judge.failed : 0;

  return <section className="persona3c-card" aria-live="polite" aria-label="Experiment 3C live progress">
    <div className="persona3c-top">
      <span className="persona3c-eyebrow">EXPERIMENT 3C · LIVE</span>
      <span className={`persona3c-status ${statusTone(run.status)}`}><i />{run.status}</span>
    </div>
    <div className="persona3c-headline">
      <strong>{persona3cStageLabel(run.stage)}</strong>
      <span>{run.id}</span>
    </div>
    {run.status !== 'failed' && <div className="persona3c-progress">
      <div className="persona3c-progress-head">
        <span>STAGE {stageNumber} OF {stageCount} · {persona3cStageLabel(run.stage).toUpperCase()}</span>
        <span>{overallPct}%</span>
      </div>
      <div className="persona3c-bar" role="progressbar" aria-valuenow={overallPct} aria-valuemin={0} aria-valuemax={100}>
        <i style={{ width: `${overallPct}%` }} />
      </div>
      <div className="persona3c-progress-foot">
        <span>{live ? 'ELAPSED' : 'RAN FOR'} {clock(elapsedSeconds)}</span>
        {run.stage === 'judge' && judge && judgeTotal > 0 && <span>{number(judge.complete)} / {number(judgeTotal)} pairs judged</span>}
        {etaSeconds !== null && <span>≈ {clock(etaSeconds)} left</span>}
      </div>
    </div>}
    <div className="persona3c-metrics">
      <div><span>GPU HOURS{estimated ? ' · EST.' : ''}</span><strong>{estimated ? '≈ ' : ''}{hours(gpuHoursShown)}</strong></div>
      <div><span>SPEND / CAP{estimated ? ' · EST. AT $1.59/H' : ''}</span><strong>{estimated ? '≈ ' : ''}{dollars(spendShown)} / {dollars(run.budgetUsd)}</strong></div>
      <div><span>{rate ? 'RATE' : 'LAST RECORD'}</span><strong>{rate ? `${number(Math.round(rate.tokensPerSecond))} tok/s` : (run.lastRecordId ?? '—')}</strong></div>
    </div>
    {run.error && <p className="persona3c-error" role="alert">{run.error}</p>}
    {run.status === 'running' && typeof status.staleMinutes === 'number' && status.staleMinutes >= PERSONA_3C_STALE_AFTER_MINUTES && <p className="persona3c-error" role="alert">No report from the pod for {status.staleMinutes} minutes. The run is not marked failed; the pod may have died and needs a resume from the last uploaded checkpoint.</p>}
    <ol className="persona3c-stages">
      {PERSONA_3C_STAGES.map((stage) => {
        const row = progressByStage.get(stage);
        const done = stage === 'train' ? estimateTokens(row) : (row?.done ?? 0);
        const total = row?.total ?? 0;
        const active = run.stage === stage;
        return <li key={stage} className={active ? 'is-active' : row ? 'is-touched' : ''}>
          <span>{persona3cStageLabel(stage)}</span>
          <small>{row ? `${stage === 'train' && rate && active ? '≈ ' : ''}${number(done)} / ${number(total)}` : '—'}</small>
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
