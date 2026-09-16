'use client';

import { useEffect, useState } from 'react';
import { persona3aVerifiedRun } from '@/lib/persona-3a-results';

const STATUS_URL = '/api/persona-3a/status';
const QUESTION_URL = 'https://afterlight-research.vercel.app/#/questions/q-unsupervised-persona';
const NOTEBOOK_URL = 'https://www.kaggle.com/code/raphaelkhalid0/unsupervisedsaes';

type PersonaStudyStatus = {
  telemetry?: { notebookUrl?: string } | null;
};

function timestampLabel(value: string) {
  return `Completed ${new Date(value).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })}`;
}

export function PersonaDiscoveryStudy() {
  const [notebookUrl, setNotebookUrl] = useState(NOTEBOOK_URL);
  const [syncState, setSyncState] = useState<'loading' | 'ready'>('loading');

  useEffect(() => {
    let active = true;
    fetch(STATUS_URL, { cache: 'no-store' })
      .then((response) => response.ok ? response.json() as Promise<PersonaStudyStatus> : null)
      .then((record) => {
        if (!active) return;
        if (record?.telemetry?.notebookUrl) setNotebookUrl(record.telemetry.notebookUrl);
      })
      .catch(() => undefined)
      .finally(() => { if (active) setSyncState('ready'); });
    return () => { active = false; };
  }, []);

  const run = persona3aVerifiedRun;

  return <main className="persona-page">
    

    <header className="persona-header"><div><p className="persona-eyebrow">EXPERIMENT 3A · DEVELOPMENT STUDY</p><h1>Persona discovery</h1><p className="persona-lede">Development screen for repeatable behavioral directions in Qwen2.5‑7B‑Instruct.</p></div><div className="persona-header-links"><a href={QUESTION_URL} target="_blank" rel="noreferrer">Protocol ↗</a><a href={notebookUrl} target="_blank" rel="noreferrer">Kaggle notebook ↗</a><a href="/experiments/persona-discovery-scoring">Open Experiment 3B scoring room ↗</a></div></header>

    <section className="persona-monitor" aria-live="polite" aria-label="Experiment run receipt">
      <div className="persona-monitor-top"><p className="persona-eyebrow">RUN RECEIPT</p><span className="persona-status-pill is-complete">{syncState === 'loading' ? 'syncing' : run.status}</span></div>
      <h2>Kaggle development run completed: all 780 screen responses are recorded.</h2>
      <div className="persona-metrics"><div><span>PHASE</span><strong>development complete</strong></div><div><span>SCREEN UNITS</span><strong>{run.screenResponses} / {run.screenResponses}</strong></div><div><span>COMPUTE</span><strong>2 × T4</strong></div><div><span>API SPEND</span><strong>$0</strong></div></div>
      <div className="persona-progress" aria-label="Screen progress: 100 percent"><div className="persona-progress-label"><span>SCREEN PROGRESS</span><strong>100%</strong></div><div className="persona-progress-track" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={100}><span style={{ width: '100%' }} /></div></div>
      <div className="persona-monitor-foot"><span>{timestampLabel(run.completedAt)}</span><span>Terminal result · {syncState === 'loading' ? 'synchronizing' : 'verified artifact'}</span></div>
    </section>

    <section className="persona-results" aria-labelledby="preliminary-results">
      <div className="persona-section-heading"><p className="persona-eyebrow">PRELIMINARY RESULTS</p><h2 id="preliminary-results">Preliminary results</h2></div>
      <p className="persona-results-lede">The 32 label-free candidates completed a development screen across 12 neutral scenarios. Paired token changes show that steering affected generation, but this is only a perturbation diagnostic until blinded behavioral scoring is complete.</p>
      <div className="persona-results-grid">
        <article><span>32</span><h3>features screened</h3><p>One calibrated positive and one calibrated negative intervention per feature.</p></article>
        <article><span>660 / 780</span><h3>responses truncated</h3><p>Most outputs reached the 128-token cap, so length-sensitive conclusions are provisional.</p></article>
        <article><span>±1</span><h3>amplitude, not 4×</h3><p>This run did not contain a “positive 4×” condition. Feature IDs are SAE indices, not condition numbers.</p></article>
      </div>
      <div className="persona-results-columns">
        <div><h3>Paired token-change examples</h3><p className="persona-muted">Compared with the same scenario’s baseline; these figures do not measure semantic quality.</p><div className="persona-signal-table"><div className="persona-signal-row persona-signal-head"><span>FEATURE / SIGN</span><span>CHANGED TOKENS</span><span>EXACT MATCHES</span></div>{run.illustrativeSignals.map((signal) => <div className="persona-signal-row" key={`${signal.feature}-${signal.sign}`}><span>{signal.feature} {signal.sign}</span><span>{signal.changedFraction}</span><span>{signal.exactMatches}</span></div>)}</div></div>
        <div><h3>One before → after</h3><p className="persona-muted">Same Pug-template prompt, same batch-seed pairing.</p><div className="persona-example"><p><strong>Baseline</strong> {run.example.baseline}</p><p><strong>Feature 38812+</strong> {run.example.positive.replace('Feature 38812+: ', '')}</p></div><p className="persona-results-note">Interpretation: a concrete wording/topic shift is visible in this example. It is not evidence that feature 38812 represents a stable persona.</p></div>
      </div>
      <details className="persona-feature-list"><summary>Show the 32 selected SAE feature IDs</summary><code>{run.featureIds.join(' · ')}</code></details>
    </section>

    <section className="persona-scoring" aria-labelledby="scoring-plan">
      <div className="persona-section-heading"><p className="persona-eyebrow">NEXT GATE</p><h2 id="scoring-plan">How the 780 responses will be scored</h2></div>
      <ol className="persona-scoring-list"><li><strong>Blind the labels.</strong> Hide feature ID and sign from the evaluator, retain scenario and baseline pairing, and score outputs in randomized order.</li><li><strong>Score fixed dimensions.</strong> Rate task completion, topical drift, refusal/safety behavior, style/voice shift, and specificity on a preregistered ordinal rubric; separately flag truncation and obvious prompt artifacts.</li><li><strong>Compare within scenario.</strong> Estimate positive and negative effects against the matched baseline, with scenario-level uncertainty rather than pooling all tokens as if independent.</li><li><strong>Carry forward only robust candidates.</strong> Require cross-scenario consistency, sign-sensitive behavior, low topic/style confounding, and a meaningful complete-output sensitivity check before selecting at most three candidates for held-out confirmation.</li></ol>
      <p className="persona-results-note">Because 660 responses are truncated, the first pass can rank candidates but cannot justify a final persona interpretation. The next practical step is to score the complete portions, mark truncation as censored, and rerun the shortlist with a larger output cap before confirmation.</p>
    </section>

    <section className="persona-method"><div className="persona-section-heading"><p className="persona-eyebrow">METHOD</p><h2>Three bounded stages</h2></div><div className="persona-method-grid"><article><span>01</span><h3>Discover</h3><p>1,024 neutral responses pass through a pretrained layer‑19 BatchTopK SAE. Up to 32 features are selected without persona labels.</p></article><article><span>02</span><h3>Steer</h3><p>Each feature is added and subtracted across 12 neutral scenarios to measure repeatable, sign-sensitive behavior and record possible topic, wording, refusal, and style confounds.</p></article><article><span>03</span><h3>Confirm</h3><p>Candidates that pass the development controls may advance to a separately approved held-out screen. Confirmation is not part of this run.</p></article></div></section>

    <section className="persona-control"><div><p className="persona-eyebrow">OWNER CONTROL</p><h2>Development run complete</h2><p>The next owner action is scoring and review, not another launch. Confirmation remains separately gated.</p></div><div className="persona-control-form persona-control-form--complete"><span>780 responses · 32 features · 12 scenarios</span><a href={notebookUrl} target="_blank" rel="noreferrer">Open Kaggle record ↗</a></div></section>

    <footer className="persona-footer"><span>Qwen2.5‑7B‑Instruct · pretrained SAE · layer 19</span><a href={notebookUrl} target="_blank" rel="noreferrer">Working record ↗</a></footer>
  </main>;
}