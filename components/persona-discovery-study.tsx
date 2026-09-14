'use client';

import { useCallback, useEffect, useRef, useState } from 'react';

const STATUS_URL = 'https://afterlight-api.raphaelbahadurkhan.workers.dev/api/studies/persona-discovery/status';
const QUESTION_URL = 'https://afterlight-research.vercel.app/#/questions/q-unsupervised-persona';
const NOTEBOOK_URL = 'https://www.kaggle.com/code/raphaelkhalid0/unsupervisedsaes';

type PersonaStudyStatus = {
  status: string;
  phase: string;
  completed: number;
  total: number;
  updatedAt: string;
  message: string;
  notebookUrl: string;
  artifactUrl: string | null;
  telemetry: 'notebook-log' | 'kaggle-status';
};

type SyncState = 'loading' | 'ready' | 'unavailable';

function isHttpsUrl(value: unknown): value is string {
  if (typeof value !== 'string' || value.length === 0) return false;
  try {
    const url = new URL(value);
    return url.protocol === 'https:';
  } catch {
    return false;
  }
}

function parseStatus(value: unknown): PersonaStudyStatus | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
  const record = value as Record<string, unknown>;
  if (
    typeof record.status !== 'string' ||
    typeof record.phase !== 'string' ||
    !Number.isInteger(record.completed) ||
    !Number.isInteger(record.total) ||
    (record.completed as number) < 0 ||
    (record.total as number) < 0 ||
    (record.completed as number) > (record.total as number) ||
    typeof record.updatedAt !== 'string' ||
    Number.isNaN(Date.parse(record.updatedAt)) ||
    typeof record.message !== 'string' ||
    (record.telemetry !== 'notebook-log' && record.telemetry !== 'kaggle-status') ||
    !isHttpsUrl(record.notebookUrl) ||
    (record.artifactUrl !== null && !isHttpsUrl(record.artifactUrl))
  ) return null;
  return {
    status: record.status,
    phase: record.phase,
    completed: record.completed as number,
    total: record.total as number,
    updatedAt: record.updatedAt,
    message: record.message,
    notebookUrl: record.notebookUrl,
    artifactUrl: record.artifactUrl as string | null,
    telemetry: record.telemetry as 'notebook-log' | 'kaggle-status',
  };
}

function timestampLabel(value: string): string {
  const date = new Date(value);
  return date.toISOString() + ' · ' + date.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' });
}

export function PersonaDiscoveryStudy() {
  const [record, setRecord] = useState<PersonaStudyStatus | null>(null);
  const previousRecordRef = useRef<PersonaStudyStatus | null>(null);
  const [previousRecord, setPreviousRecord] = useState<PersonaStudyStatus | null>(null);
  const [syncState, setSyncState] = useState<SyncState>('loading');
  const [lastSynchronized, setLastSynchronized] = useState<string | null>(null);

  const synchronize = useCallback(async () => {
    try {
      const response = await fetch(STATUS_URL, { cache: 'no-store' });
      if (!response.ok) throw new Error('status unavailable');
      const next = parseStatus(await response.json());
      if (!next) throw new Error('invalid status record');
      setPreviousRecord(previousRecordRef.current);
      previousRecordRef.current = next;
      setRecord(next);
      setLastSynchronized(next.updatedAt);
      setSyncState('ready');
    } catch {
      setRecord(null);
      setSyncState('unavailable');
    }
  }, []);

  useEffect(() => {
    void synchronize();
    const interval = window.setInterval(() => void synchronize(), 30_000);
    return () => window.clearInterval(interval);
  }, [synchronize]);

  const available = syncState === 'ready' && record !== null;
  const screenOnly = record?.status === 'completed' && record.phase.toLowerCase() === 'discovery-and-development-screen';
  const stale = available && Date.now() - Date.parse(record.updatedAt) > 180_000;
  const notebookUrl = available ? record.notebookUrl : NOTEBOOK_URL;
  const measuredUnits = available && record && record.telemetry === 'notebook-log' ? record.completed + ' / ' + record.total : 'Awaiting notebook counts';
  const measured = available && record?.telemetry === 'notebook-log';
  const progress = measured && record ? Math.min(100, Math.max(0, (record.completed / Math.max(1, record.total)) * 100)) : 0;
  const elapsedSeconds = measured && previousRecord && record && record.completed > previousRecord.completed
    ? (Date.parse(record.updatedAt) - Date.parse(previousRecord.updatedAt)) / 1000
    : 0;
  const unitsPerSecond = elapsedSeconds > 0 && record && previousRecord ? (record.completed - previousRecord.completed) / elapsedSeconds : 0;
  const etaSeconds = unitsPerSecond > 0 && record && record.status === 'running' ? Math.ceil((record.total - record.completed) / unitsPerSecond) : null;
  const etaLabel = etaSeconds === null ? null : etaSeconds >= 3600
    ? Math.floor(etaSeconds / 3600) + 'h ' + Math.ceil((etaSeconds % 3600) / 60) + 'm'
    : Math.max(1, Math.ceil(etaSeconds / 60)) + 'm';

  return <main className="archive-page persona-study">
    <nav className="archive-nav" aria-label="Primary">
      <a href="/">A / AUTOLABS</a>
      <a href="/experiments">All experiments</a>
      <a href="/research">Research</a>
    </nav>

    <header className="archive-heading">
      <p className="archive-kicker">STUDY PROPOSAL / UNSUPERVISED REPRESENTATIONS</p>
      <h1>Unsupervised<br /><em>persona discovery.</em></h1>
      <p>Can label-free SAE feature selection recover generalizable persona tendencies that optimized prompting and prompt-extracted persona vectors struggle to recover? This page shows the planned protocol and latest study update.</p>
      <div className="persona-links">
        <a className="archive-button" href={QUESTION_URL} target="_blank" rel="noreferrer">Read the Afterlight question ↗</a>
        <a className="archive-button" href={notebookUrl} target="_blank" rel="noreferrer">Open the Kaggle notebook ↗</a>
      </div>
    </header>

    <section className="archive-verdict persona-status" aria-live="polite" aria-label="Public study status">
      <div className="persona-status-head">
        <p className="archive-kicker">PUBLIC STATUS</p>
        <span className={available ? 'persona-status-pill is-available' + (stale ? ' is-stale' : '') : 'persona-status-pill'}>{available ? (screenOnly ? 'confirmation pending' : record.status) : 'status unavailable'}</span>
      </div>
      {available ? <>
        <h2>{screenOnly ? 'Discovery and development screen recorded.' : record.message}</h2>
        {screenOnly && <p>The discovery and development screen is recorded. Confirmation remains pending.</p>}
        <div className="persona-status-grid">
          <div><span className="archive-kicker">PHASE</span><strong>{record.phase}</strong></div>
          <div><span className="archive-kicker">RECORDED UNITS</span><strong>{measuredUnits}</strong></div>
        </div>
        {measured && <div className="persona-progress" aria-label={'Discovery progress: ' + Math.round(progress) + ' percent'}>
          <div className="persona-progress-label"><span className="archive-kicker">DISCOVERY PROGRESS</span><strong>{Math.round(progress)}%</strong></div>
          <div className="persona-progress-track" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(progress)}><span style={{ width: progress + '%' }} /></div>
          <p className="persona-eta">{etaLabel ? 'Estimated time remaining · ' + etaLabel : 'ETA calibrating from the next synchronized update'}</p>
        </div>}
        <p className="persona-sync">Last synchronized status · {timestampLabel(record.updatedAt)}</p>
        {stale && <p className="persona-stale">Status may be stale. The last update is more than three minutes old.</p>}
        {record.artifactUrl ? <a href={record.artifactUrl} target="_blank" rel="noreferrer">Protocol and source ↗</a> : <p className="archive-caption">Protocol and source link pending.</p>}
      </> : <>
        <h2>Status unavailable</h2>
        <p>A synchronized study update is not available. This page does not estimate progress or report a result.</p>
        <p className="persona-sync">Last synchronized status · {lastSynchronized ? timestampLabel(lastSynchronized) : 'unavailable'}</p>
      </>}
    </section>

    <section className="archive-section">
      <p className="archive-kicker">01 / PLANNED METHOD</p>
      <h2>Three phases, with confirmation kept separate.</h2>
      <ol className="archive-lessons persona-phases">
        <li><h3>Discovery</h3><p>Use Qwen2.5-7B-Instruct activations from 1,024 discovery responses with a pretrained layer 19 BatchTopK SAE. Select up to 32 candidate features without labels; these are study leads, not findings.</p></li>
        <li><h3>Development</h3><p>Screen each feature on the same 12 neutral prompts with positive and negative activation steering. Freeze behavioral definitions and fair prompting baselines before carrying at most three candidates forward.</p></li>
        <li><h3>Confirmation</h3><p>The later plan covers 600 scenarios per candidate, six conditions, and two repeats per condition. Confirmation has not produced a result; any result will remain scoped to the tested setting.</p></li>
      </ol>
    </section>

    <section className="archive-section persona-boundary">
      <p className="archive-kicker">02 / RESEARCH BOUNDARY</p>
      <h2>Planned work is not completed evidence.</h2>
      <p>The Afterlight dossier is the source question. The Kaggle notebook is the working record. The confirmation phase has not produced a result, so this page reports the latest study update without presenting one.</p>
    </section>

    <footer className="archive-section">
      <a href={QUESTION_URL} target="_blank" rel="noreferrer">Return to the source question ↗</a>
      <p className="archive-caption">Notebook: <a href={notebookUrl} target="_blank" rel="noreferrer">{notebookUrl}</a></p>
    </footer>
  </main>;
}