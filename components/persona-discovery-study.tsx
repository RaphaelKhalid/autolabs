'use client';

import { useCallback, useEffect, useState } from 'react';

const STATUS_URL = '/api/persona-3a/status';
const QUESTION_URL = 'https://afterlight-research.vercel.app/#/questions/q-unsupervised-persona';
const NOTEBOOK_URL = 'https://www.kaggle.com/code/raphaelkhalid0/unsupervisedsaes';

type PersonaStudyStatus = {
  launch?: { id?: string; status?: string; phase?: string; requestedAt?: string; updatedAt?: string; maxRuntimeSeconds?: number; manifestHash?: string };
  telemetry?: { status?: string; phase?: string; completed?: number; total?: number; updatedAt?: string; message?: string; notebookUrl?: string; artifactUrl?: string | null } | null;
};

function timestampLabel(value?: string) {
  if (!value) return 'Waiting for a synchronized update';
  return `Updated ${new Date(value).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })}`;
}

export function PersonaDiscoveryStudy() {
  const [record, setRecord] = useState<PersonaStudyStatus | null>(null);
  const [syncState, setSyncState] = useState<'loading' | 'ready'>('loading');
  const [ownerKey, setOwnerKey] = useState('');
  const [launchState, setLaunchState] = useState('');

  const synchronize = useCallback(async () => {
    try {
      const response = await fetch(STATUS_URL, { cache: 'no-store' });
      if (!response.ok) throw new Error('status unavailable');
      setRecord(await response.json() as PersonaStudyStatus);
    } catch {
      setRecord(null);
    } finally {
      setSyncState('ready');
    }
  }, []);

  useEffect(() => {
    void synchronize();
    const interval = window.setInterval(() => void synchronize(), 30_000);
    return () => window.clearInterval(interval);
  }, [synchronize]);

  const launch = record?.launch;
  const telemetry = record?.telemetry;
  const launchStatus = launch?.status ?? 'unavailable';
  const statusLabel = launchStatus === 'started' && telemetry?.status ? telemetry.status : launchStatus;
  const completed = telemetry?.completed ?? 0;
  const total = telemetry?.total ?? 780;
  const hasProgress = typeof telemetry?.completed === 'number' && typeof telemetry?.total === 'number';
  const progress = hasProgress ? Math.min(100, Math.max(0, completed / Math.max(1, total) * 100)) : 0;
  const notebookUrl = telemetry?.notebookUrl ?? NOTEBOOK_URL;
  const monitorMessage = launchStatus === 'queued'
    ? 'Launch accepted. Waiting for the owner relay to start Kaggle.'
    : telemetry?.message ?? (launchStatus === 'started' ? 'Kaggle development run is active.' : 'No active launch.');

  async function start3A() {
    setLaunchState('Queueing the development run…');
    try {
      const response = await fetch('/api/control/persona-3a/start', { method: 'POST', headers: { 'content-type': 'application/json', 'x-autolabs-owner-key': ownerKey }, body: JSON.stringify({ idempotencyKey: crypto.randomUUID() }) });
      const data = await response.json().catch(() => ({})) as { launch?: { id?: string; status?: string }; error?: string };
      setLaunchState(response.ok ? `Queued · ${data.launch?.id ?? 'request accepted'} · ${data.launch?.status ?? 'queued'}` : data.error ?? 'Launch request failed.');
      void synchronize();
    } catch {
      setLaunchState('The orchestrator is unavailable; no launch was assumed.');
    }
  }

  return <main className="persona-page">
    <nav className="persona-nav" aria-label="Primary">
      <a href="/" className="persona-brand">AUTOLABS <span>/</span> EXPERIMENT 3A</a>
      <div><a href="/experiments">Experiments</a><a href="/research">Research</a></div>
    </nav>

    <header className="persona-header">
      <div><p className="persona-eyebrow">UNSUPERVISED REPRESENTATIONS · DEVELOPMENT STUDY</p><h1>Unsupervised <em>persona discovery</em></h1><p className="persona-lede">A label-free activation screen for repeatable behavioral directions in Qwen2.5‑7B‑Instruct.</p></div>
      <div className="persona-header-links"><a href={QUESTION_URL} target="_blank" rel="noreferrer">Protocol ↗</a><a href={notebookUrl} target="_blank" rel="noreferrer">Kaggle notebook ↗</a></div>
    </header>

    <section className="persona-monitor" aria-live="polite" aria-label="Live experiment monitoring">
      <div className="persona-monitor-top"><p className="persona-eyebrow">LIVE MONITOR</p><span className={'persona-status-pill ' + (statusLabel === 'started' || statusLabel === 'running' ? 'is-running' : '')}>{syncState === 'loading' ? 'syncing' : statusLabel}</span></div>
      <h2>{monitorMessage}</h2>
      <div className="persona-metrics"><div><span>PHASE</span><strong>{launch?.phase ?? 'development'}</strong></div><div><span>SCREEN UNITS</span><strong>{hasProgress ? `${completed} / ${total}` : `0 / ${total}`}</strong></div><div><span>COMPUTE</span><strong>2 × T4</strong></div><div><span>API SPEND</span><strong>$0</strong></div></div>
      <div className="persona-progress" aria-label={'Screen progress: ' + Math.round(progress) + ' percent'}><div className="persona-progress-label"><span>SCREEN PROGRESS</span><strong>{Math.round(progress)}%</strong></div><div className="persona-progress-track" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(progress)}><span style={{ width: progress + '%' }} /></div></div>
      <div className="persona-monitor-foot"><span>{timestampLabel(telemetry?.updatedAt ?? launch?.updatedAt ?? launch?.requestedAt)}</span><span>Auto-refresh · 30s</span></div>
    </section>

    <section className="persona-method"><div className="persona-section-heading"><p className="persona-eyebrow">METHOD</p><h2>Three bounded stages</h2></div><div className="persona-method-grid"><article><span>01</span><h3>Discover</h3><p>1,024 neutral responses pass through a pretrained layer‑19 BatchTopK SAE. Up to 32 features are selected without persona labels.</p></article><article><span>02</span><h3>Steer</h3><p>Each feature is added and subtracted across 12 neutral scenarios to measure repeatable, sign-sensitive behavior and record possible topic, wording, refusal, and style confounds.</p></article><article><span>03</span><h3>Confirm</h3><p>Candidates that pass the development controls may advance to a separately approved held-out screen. Confirmation is not part of this run.</p></article></div></section>

    <section className="persona-control"><div><p className="persona-eyebrow">OWNER CONTROL</p><h2>Development launch</h2><p>Fixed Kaggle run · 6,600-second ceiling · no paid API calls.</p></div><div className="persona-control-form"><label htmlFor="owner-key">Owner key</label><input id="owner-key" type="password" autoComplete="off" value={ownerKey} onChange={(event) => setOwnerKey(event.target.value)} /><button disabled={!ownerKey || launchState.startsWith('Queueing')} onClick={() => void start3A()}>{launchState.startsWith('Queueing') ? 'Queueing…' : 'Queue development run'}</button>{launchState && <p role="status">{launchState}</p>}</div></section>

    <footer className="persona-footer"><span>Qwen2.5‑7B‑Instruct · pretrained SAE · layer 19</span><a href={notebookUrl} target="_blank" rel="noreferrer">Working record ↗</a></footer>
  </main>;
}