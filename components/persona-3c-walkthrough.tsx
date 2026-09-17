'use client';

import Link from 'next/link';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Persona3CProgress } from '@/components/persona-3c-progress';
import { fetchPersona3CStatus, PERSONA_3C_STALE_AFTER_MINUTES, type Persona3CStatus } from '@/lib/persona-3c';
import { visual2 } from '@/lib/persona-3c-visual-2';

// Chart palette, validated with the dataviz palette checker (light surface #fcfcfb).
const C_FEATURE = '#D63B00';
const C_CONTROL = '#2a63b8';
const C_RANDOM = '#5c7a2e';

const REPO = 'https://github.com/RaphaelKhalid/autolabs/blob/main/research/experiment-003c';

type ActKey = 'boot' | 'train' | 'check' | 'calibrate' | 'screen' | 'describe' | 'done';
interface Act { key: ActKey; label: string; startMin: number; endMin: number; playSeconds: number }

// Real wall-clock windows of the visual-2 run (minutes after 04:27:22 UTC), from the harness event ledger.
const ACTS: Act[] = [
  { key: 'boot', label: 'Boot', startMin: 0, endMin: 2.3, playSeconds: 4 },
  { key: 'train', label: 'Harvest + train', startMin: 2.3, endMin: 51.4, playSeconds: 18 },
  { key: 'check', label: 'Check', startMin: 51.4, endMin: 53, playSeconds: 4 },
  { key: 'calibrate', label: 'Calibrate', startMin: 53, endMin: 77.6, playSeconds: 12 },
  { key: 'screen', label: 'Screen', startMin: 77.6, endMin: 121.7, playSeconds: 16 },
  { key: 'describe', label: 'Describe', startMin: 121.7, endMin: 124.5, playSeconds: 5 },
  { key: 'done', label: 'Done', startMin: 124.5, endMin: 124.5, playSeconds: 3 },
];
const TOTAL_PLAY = ACTS.reduce((sum, act) => sum + act.playSeconds, 0);
const RUN_START = new Date(visual2.startedAt).getTime();

function clockFor(minutes: number) {
  const date = new Date(RUN_START + minutes * 60_000);
  return `${String(date.getUTCHours()).padStart(2, '0')}:${String(date.getUTCMinutes()).padStart(2, '0')} UTC`;
}
function locate(seconds: number) {
  let acc = 0;
  for (let i = 0; i < ACTS.length; i += 1) {
    const act = ACTS[i];
    if (seconds < acc + act.playSeconds || i === ACTS.length - 1) {
      const p = Math.min(1, Math.max(0, (seconds - acc) / act.playSeconds));
      return { index: i, act, p, minutes: act.startMin + (act.endMin - act.startMin) * p };
    }
    acc += act.playSeconds;
  }
  const last = ACTS[ACTS.length - 1];
  return { index: ACTS.length - 1, act: last, p: 1, minutes: last.endMin };
}
function startOf(index: number) { return ACTS.slice(0, index).reduce((sum, act) => sum + act.playSeconds, 0); }

function fmtInt(value: number) { return new Intl.NumberFormat('en-US').format(value); }
function fmtM(tokens: number) { return `${(tokens / 1e6).toFixed(1)}M`; }
function pct(value: number, digits = 0) { return `${(value * 100).toFixed(digits)}%`; }

// ---------------------------------------------------------------------------
// Tooltip plumbing shared by the charts
// ---------------------------------------------------------------------------
interface Tip { x: number; y: number; lines: string[] }
function useTip() {
  const [tip, setTip] = useState<Tip | null>(null);
  const wrap = useRef<HTMLDivElement>(null);
  const show = useCallback((event: React.MouseEvent, lines: string[]) => {
    const box = wrap.current?.getBoundingClientRect();
    if (!box) return;
    setTip({ x: event.clientX - box.left + 12, y: event.clientY - box.top + 12, lines });
  }, []);
  const hide = useCallback(() => setTip(null), []);
  return { tip, wrap, show, hide };
}
function TipBox({ tip }: { tip: Tip | null }) {
  if (!tip) return null;
  return <div className="p3c-tip" style={{ left: tip.x, top: tip.y }} role="status">{tip.lines.map((line, i) => <div key={i}>{line}</div>)}</div>;
}

// ---------------------------------------------------------------------------
// Chart 1: training curve (FVE and dead fraction share the 0..1 axis)
// ---------------------------------------------------------------------------
interface TrainRow { tokens: number; steps: number; fve: number; dead: number; loss: number }
const VISUAL2_TRAIN: TrainRow[] = visual2.train.map((r) => ({ ...r }));

function TrainChart({ tokensShown, rows = VISUAL2_TRAIN, tokensTarget = visual2.config.tokens, title = 'SAE training on layer-19 assistant tokens', subtitle = `width ${fmtInt(visual2.config.width)} · k ${visual2.config.k} · ${visual2.config.stepsPerBatch} optimizer steps per harvested batch` }: { tokensShown: number; rows?: TrainRow[]; tokensTarget?: number; title?: string; subtitle?: string }) {
  const { tip, wrap, show, hide } = useTip();
  const [table, setTable] = useState(false);
  const W = 640, H = 260, L = 44, R = 16, T = 14, B = 34;
  const xs = (tokens: number) => L + (tokens / tokensTarget) * (W - L - R);
  const ys = (v: number) => T + (1 - v) * (H - T - B);
  const visible = rows.filter((row) => row.tokens <= tokensShown + 1);
  const path = (key: 'fve' | 'dead') => visible.map((row, i) => `${i ? 'L' : 'M'}${xs(row.tokens).toFixed(1)},${ys(row[key]).toFixed(1)}`).join(' ');
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => Math.round(tokensTarget * f));
  const last = visible[visible.length - 1];
  return <figure className="p3c-figure">
    <figcaption><b>{title}</b><span>{subtitle}</span></figcaption>
    <ul className="p3c-legend"><li><i className="is-line" style={{ '--c': C_FEATURE } as React.CSSProperties} />Fraction of variance explained</li><li><i className="is-line" style={{ '--c': C_CONTROL } as React.CSSProperties} />Dead features</li></ul>
    {table ? <div className="p3c-table-scroll"><table className="p3c-table"><thead><tr><th>Tokens</th><th>Steps</th><th>FVE</th><th>Dead</th><th>Loss</th></tr></thead><tbody>
      {rows.map((row) => <tr key={row.tokens}><td className="num">{fmtInt(row.tokens)}</td><td className="num">{fmtInt(row.steps)}</td><td className="num">{row.fve.toFixed(3)}</td><td className="num">{pct(row.dead, 1)}</td><td className="num">{row.loss.toFixed(3)}</td></tr>)}
    </tbody></table></div>
    : <div className="p3c-chart-wrap" ref={wrap}>
      <svg className="p3c-svg" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={last ? `Fraction of variance explained ${last.fve.toFixed(3)} and dead features ${pct(last.dead, 1)} at ${fmtM(last.tokens)} tokens of ${fmtM(tokensTarget)}.` : 'No training checkpoints yet.'}>
        {[0, 0.25, 0.5, 0.75, 1].map((v) => <g key={v}><line className="p3c-grid" x1={L} x2={W - R} y1={ys(v)} y2={ys(v)} /><text x={L - 6} y={ys(v) + 3} fontSize="9" textAnchor="end">{v.toFixed(2)}</text></g>)}
        {ticks.map((t) => <text key={t} x={xs(t)} y={H - B + 14} fontSize="9" textAnchor="middle">{fmtM(t)}</text>)}
        <text x={(L + W - R) / 2} y={H - 6} fontSize="9" textAnchor="middle">assistant tokens harvested</text>
        <line className="p3c-axis" x1={L} x2={W - R} y1={ys(0)} y2={ys(0)} />
        {visible.length > 0 && <>
          <path d={path('dead')} fill="none" stroke={C_CONTROL} strokeWidth="2" strokeLinejoin="round" />
          <path d={path('fve')} fill="none" stroke={C_FEATURE} strokeWidth="2" strokeLinejoin="round" />
          {visible.map((row) => <g key={row.tokens}>
            <circle cx={xs(row.tokens)} cy={ys(row.fve)} r="3.5" fill={C_FEATURE} stroke="#fcfcfb" strokeWidth="2" />
            <circle cx={xs(row.tokens)} cy={ys(row.dead)} r="3.5" fill={C_CONTROL} stroke="#fcfcfb" strokeWidth="2" />
            <rect className="p3c-hit" x={xs(row.tokens) - 20} y={T} width="40" height={H - T - B} onMouseMove={(e) => show(e, [`${fmtM(row.tokens)} tokens · step ${fmtInt(row.steps)}`, `FVE ${row.fve.toFixed(3)}`, `dead ${pct(row.dead, 1)}`, `loss ${row.loss.toFixed(3)}`])} onMouseLeave={hide} />
          </g>)}
          {last && visible.length === rows.length && <text className="p3c-label" x={xs(last.tokens) - 6} y={ys(last.fve) - 9} fontSize="10" textAnchor="end">FVE {last.fve.toFixed(3)}</text>}
        </>}
      </svg>
      <TipBox tip={tip} />
    </div>}
    <button type="button" className="p3c-table-toggle" onClick={() => setTable((v) => !v)}>{table ? 'Show chart' : 'Show table'}</button>
  </figure>;
}

// ---------------------------------------------------------------------------
// Chart 2: dose ladder (edit distance vs dose, features and random directions)
// ---------------------------------------------------------------------------
function DoseChart({ featuresShown }: { featuresShown: number }) {
  const { tip, wrap, show, hide } = useTip();
  const [table, setTable] = useState(false);
  const W = 640, H = 250, L = 44, R = 16, T = 14, B = 34;
  const doses: readonly number[] = visual2.calibrate.doses;
  const xs = (dose: number) => L + (doses.indexOf(dose) / (doses.length - 1)) * (W - L - R);
  const ys = (v: number) => T + (1 - v) * (H - T - B);
  const features = visual2.calibrate.features.slice(0, featuresShown);
  const series: { key: string; color: string; label: string; rows: { dose: number; editDistance: number; coherent: number; of: number }[] }[] = [];
  for (const f of features) for (const sign of [1, -1]) series.push({ key: `f${f.id}${sign}`, color: C_FEATURE, label: `feature ${f.id} (${sign > 0 ? '+' : '−'})`, rows: f.rows.filter((r) => r.sign === sign) });
  if (featuresShown >= visual2.calibrate.features.length) for (const r of visual2.calibrate.random) series.push({ key: `r${r.index}`, color: C_RANDOM, label: `random direction ${r.index}`, rows: [...r.rows] });
  return <figure className="p3c-figure">
    <figcaption><b>Dose ladder: how far each steered reply moves from baseline</b><span>word edit distance, mean over {visual2.config.calibrateScenarios} scenarios · hollow marker = some replies incoherent</span></figcaption>
    <ul className="p3c-legend"><li><i className="is-line" style={{ '--c': C_FEATURE } as React.CSSProperties} />SAE feature directions (16)</li><li><i className="is-line" style={{ '--c': C_RANDOM } as React.CSSProperties} />Random directions (2)</li><li><i className="is-hollow" style={{ '--c': '#5e6d63' } as React.CSSProperties} />Not all 4 replies coherent</li></ul>
    {table ? <div className="p3c-table-scroll"><table className="p3c-table"><thead><tr><th>Direction</th>{doses.map((d) => <th key={d}>dose {d}</th>)}</tr></thead><tbody>
      {series.map((s) => <tr key={s.key}><td>{s.label}</td>{s.rows.map((r) => <td key={r.dose} className="num">{r.editDistance.toFixed(2)} · {r.coherent}/{r.of}</td>)}</tr>)}
    </tbody></table></div>
    : <div className="p3c-chart-wrap" ref={wrap}>
      <svg className="p3c-svg" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Edit distance rises with dose for every direction; at dose 2.0 most SAE feature directions lose coherence while random directions stay coherent.">
        {[0, 0.25, 0.5, 0.75, 1].map((v) => <g key={v}><line className="p3c-grid" x1={L} x2={W - R} y1={ys(v)} y2={ys(v)} /><text x={L - 6} y={ys(v) + 3} fontSize="9" textAnchor="end">{v.toFixed(2)}</text></g>)}
        {doses.map((d) => <text key={d} x={xs(d)} y={H - B + 14} fontSize="9" textAnchor="middle">{d}</text>)}
        <text x={(L + W - R) / 2} y={H - 6} fontSize="9" textAnchor="middle">dose (× feature max activation)</text>
        <line className="p3c-axis" x1={L} x2={W - R} y1={ys(0)} y2={ys(0)} />
        {series.map((s) => <g key={s.key}>
          <path d={s.rows.map((r, i) => `${i ? 'L' : 'M'}${xs(r.dose).toFixed(1)},${ys(r.editDistance).toFixed(1)}`).join(' ')} fill="none" stroke={s.color} strokeWidth={s.color === C_RANDOM ? 2.5 : 1.5} strokeOpacity={s.color === C_RANDOM ? 1 : 0.55} strokeLinejoin="round" />
          {s.rows.map((r) => <circle key={r.dose} cx={xs(r.dose)} cy={ys(r.editDistance)} r="4" fill={r.coherent === r.of ? s.color : '#fcfcfb'} stroke={s.color} strokeWidth="2" onMouseMove={(e) => show(e, [s.label, `dose ${r.dose}: edit distance ${r.editDistance.toFixed(2)}`, `${r.coherent} of ${r.of} replies coherent`])} onMouseLeave={hide} />)}
        </g>)}
      </svg>
      <TipBox tip={tip} />
    </div>}
    <button type="button" className="p3c-table-toggle" onClick={() => setTable((v) => !v)}>{table ? 'Show chart' : 'Show table'}</button>
  </figure>;
}

// ---------------------------------------------------------------------------
// Chart 3: screen separability strip (residual AUC per direction, by kind)
// ---------------------------------------------------------------------------
type ScreenRow = (typeof visual2.screen.rows)[number];
function screenLabel(row: ScreenRow) {
  if (row.kind === 'feature') return `feature ${row.id} (${row.sign > 0 ? '+' : '−'})`;
  if (row.kind === 'control') return `${String(row.id).replace('_', ' / ')} (${row.sign > 0 ? '+' : '−'})`;
  return `random direction ${row.id}`;
}
function ScreenChart({ shown }: { shown: number }) {
  const { tip, wrap, show, hide } = useTip();
  const [table, setTable] = useState(false);
  const W = 640, H = 230, L = 96, R = 16, T = 16, B = 34;
  const xs = (auc: number) => L + ((auc - 0.4) / 0.6) * (W - L - R);
  const lanes: { key: ScreenRow['kind']; label: string; color: string; y: number }[] = [
    { key: 'random', label: 'Random (20)', color: C_RANDOM, y: T + 30 },
    { key: 'control', label: 'Controls (6)', color: C_CONTROL, y: T + 90 },
    { key: 'feature', label: 'Features (16)', color: C_FEATURE, y: T + 150 },
  ];
  // Reveal order: randoms first (they define the null), then controls, then features, each sorted by AUC.
  const ordered = useMemo(() => {
    const byKind = (kind: ScreenRow['kind']) => [...visual2.screen.rows].filter((r) => r.kind === kind).sort((a, b) => a.residAuc - b.residAuc);
    return [...byKind('random'), ...byKind('control'), ...byKind('feature')];
  }, []);
  const visible = ordered.slice(0, shown);
  const ceiling = visual2.screen.verdict.maxRandomResidAuc;
  const randomsDone = visible.filter((r) => r.kind === 'random').length === 20;
  // Simple jitter so ties (AUC 1.0) do not stack: spread by index within the lane.
  const jitter = (row: ScreenRow, i: number) => ((i % 5) - 2) * 7;
  return <figure className="p3c-figure">
    <figcaption><b>Screen: can steered replies be told apart from baseline?</b><span>leave-one-scenario-out AUC in the residual stream, {visual2.config.screenScenarios} scenarios</span></figcaption>
    <ul className="p3c-legend"><li><i style={{ '--c': C_FEATURE } as React.CSSProperties} />SAE feature direction</li><li><i style={{ '--c': C_CONTROL } as React.CSSProperties} />Persona-vector control</li><li><i className="is-hollow" style={{ '--c': C_RANDOM } as React.CSSProperties} />Random direction (null)</li><li><i className="is-dash" />Null ceiling: max random AUC {ceiling.toFixed(3)}</li></ul>
    {table ? <div className="p3c-table-scroll"><table className="p3c-table"><thead><tr><th>Direction</th><th>Dose</th><th>Residual AUC</th><th>Lexical AUC</th><th>Consistency</th><th>Null</th><th>Coherent</th></tr></thead><tbody>
      {[...visual2.screen.rows].sort((a, b) => b.residAuc - a.residAuc).map((r) => <tr key={`${r.kind}${r.id}${r.sign}`}><td>{screenLabel(r)}</td><td className="num">{r.dose}</td><td className="num">{r.residAuc.toFixed(3)}</td><td className="num">{r.lexAuc.toFixed(3)}</td><td className="num">{r.consistency.toFixed(3)}</td><td className="num">{r.nullConsistency.toFixed(3)}</td><td className="num">{pct(r.coherentFraction)}</td></tr>)}
    </tbody></table></div>
    : <div className="p3c-chart-wrap" ref={wrap}>
      <svg className="p3c-svg" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Random directions cluster near AUC 0.5 with a maximum of 0.594; the sycophantic control and seven feature directions sit above that ceiling, several at 1.0.">
        {[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0].map((v) => <g key={v}><line className="p3c-grid" x1={xs(v)} x2={xs(v)} y1={T} y2={H - B} /><text x={xs(v)} y={H - B + 14} fontSize="9" textAnchor="middle">{v.toFixed(1)}</text></g>)}
        <text x={(L + W - R) / 2} y={H - 6} fontSize="9" textAnchor="middle">residual-stream AUC (0.5 = chance)</text>
        {lanes.map((lane) => <g key={lane.key}><line className="p3c-axis" x1={L} x2={W - R} y1={lane.y} y2={lane.y} /><text x={L - 8} y={lane.y + 3} fontSize="9" textAnchor="end" fill={lane.color}>{lane.label}</text></g>)}
        {randomsDone && <g><line x1={xs(ceiling)} x2={xs(ceiling)} y1={T} y2={H - B} stroke="#111312" strokeWidth="1.5" strokeDasharray="4 3" /><text className="p3c-label" x={xs(ceiling) + 5} y={T + 9} fontSize="9">null ceiling {ceiling.toFixed(3)}</text></g>}
        {visible.map((row, i) => {
          const lane = lanes.find((l) => l.key === row.kind)!;
          const laneIndex = visible.filter((r, j) => j < i && r.kind === row.kind).length;
          const cy = lane.y + jitter(row, laneIndex);
          const cx = xs(row.residAuc);
          const hollow = row.kind === 'random';
          return <g key={`${row.kind}${row.id}${row.sign}`}>
            <circle cx={cx} cy={cy} r="5" fill={hollow ? '#fcfcfb' : lane.color} stroke={lane.color} strokeWidth="2" />
            <circle className="p3c-hit" cx={cx} cy={cy} r="11" onMouseMove={(e) => show(e, [screenLabel(row), `dose ${row.dose}`, `residual AUC ${row.residAuc.toFixed(3)} · lexical ${row.lexAuc.toFixed(3)}`, `consistency ${row.consistency.toFixed(3)} vs null ${row.nullConsistency.toFixed(3)}`, `${pct(row.coherentFraction)} of replies coherent`])} onMouseLeave={hide} />
          </g>;
        })}
        {shown >= ordered.length && <>
          <text className="p3c-label" x={xs(1) - 8} y={lanes[2].y + 30} fontSize="9" textAnchor="end">feature 81 (+) 1.000</text>
          <text className="p3c-label" x={xs(1) - 8} y={lanes[1].y - 22} fontSize="9" textAnchor="end">sycophantic (+) 1.000</text>
        </>}
      </svg>
      <TipBox tip={tip} />
    </div>}
    <button type="button" className="p3c-table-toggle" onClick={() => setTable((v) => !v)}>{table ? 'Show chart' : 'Show table'}</button>
  </figure>;
}

// ---------------------------------------------------------------------------
// The player
// ---------------------------------------------------------------------------
function StageSide({ act, p }: { act: Act; p: number }) {
  const v = visual2;
  const verdict = v.screen.verdict;
  const events = v.timeline;
  const nowMin = act.startMin + (act.endMin - act.startMin) * p;
  const onEvents = events.map((e) => (new Date(e.at).getTime() - RUN_START) / 60_000 <= nowMin + 0.01);
  const body = (() => {
    switch (act.key) {
      case 'boot': return <>
        <h3>Boot checks on the pod</h3>
        <p>Two sanity checks must pass before any GPU time is spent: a hook that replaces the layer-19 residual with itself must not change generation, and the assistant-turn token mask must never include chat-template control tokens.</p>
        <div className="p3c-kpis"><div><span>Identity hook</span><strong>passed</strong></div><div><span>Assistant mask</span><strong>11 of 48 tokens</strong></div><div><span>Pod</span><strong>{v.pod}</strong></div><div><span>GPU</span><strong>{v.gpu}<small>${v.pricePerHour.toFixed(2)}/h</small></strong></div></div>
      </>;
      case 'train': {
        const tokens = Math.round(v.config.tokens * p);
        const row = [...v.train].reverse().find((r) => r.tokens <= tokens + 1) ?? null;
        return <>
          <h3>Harvest activations, train the dictionary</h3>
          <p>UltraChat conversations stream through Qwen2.5-7B-Instruct truncated at layer 19. Only assistant-turn positions enter a shuffle buffer; the Matryoshka BatchTopK SAE trains on samples from it. Activations never touch disk.</p>
          <div className="p3c-kpis"><div><span>Tokens</span><strong>{fmtM(tokens)}<small>of {fmtM(v.config.tokens)}</small></strong></div><div><span>Optimizer steps</span><strong>{row ? fmtInt(row.steps) : '—'}</strong></div><div><span>Variance explained</span><strong>{row ? row.fve.toFixed(3) : '—'}</strong></div><div><span>Dead features</span><strong>{row ? pct(row.dead, 1) : '—'}</strong></div></div>
        </>;
      }
      case 'check': return <>
        <h3>Post-train check and held-out fit</h3>
        <p>A fresh batch the optimizer never saw scores the SAE. Then the reconstruction is swapped into the residual stream at assistant positions to see how often the next-token argmax survives.</p>
        <div className="p3c-kpis"><div><span>Held-out FVE</span><strong>{v.postTrain.heldOutFve.toFixed(3)}</strong></div><div><span>Held-out L0</span><strong>{v.postTrain.heldOutL0}</strong></div><div><span>Argmax match</span><strong>{v.postTrain.matchFraction.toFixed(2)}<small>target {v.postTrain.minMatchFraction}</small></strong></div><div><span>CE increase</span><strong>{v.postTrain.ceDelta.toFixed(2)} nats</strong></div></div>
        <p>The match target was not met, so the dictionary is a partial model of the stream. That is recorded, not hidden, and it is why the full run trains a wider SAE on far more tokens.</p>
      </>;
      case 'calibrate': {
        const shown = Math.min(v.config.features, Math.ceil(p * v.config.features));
        return <>
          <h3>Calibrate a safe dose per direction</h3>
          <p>Each selected feature is added to the residual stream in both signs at four doses on four scenarios. A repetition-aware coherence guard picks the largest dose that still produces readable text. Random unit directions get the same treatment as the null.</p>
          <div className="p3c-kpis"><div><span>Features calibrated</span><strong>{shown} of {v.config.features}</strong></div><div><span>Generations</span><strong>{fmtInt(Math.round(292 * p))}<small>of 292</small></strong></div><div><span>Max coherent dose</span><strong>1.0 to 2.0</strong></div><div><span>Unbatched runtime</span><strong>≈ 26 min</strong></div></div>
        </>;
      }
      case 'screen': {
        const shown = Math.min(42, Math.ceil(p * 42));
        return <>
          <h3>Screen against a calibrated null</h3>
          <p>Every direction at its dose generates replies on eight open-ended scenarios. A leave-one-scenario-out classifier asks whether steered replies can be told from baseline. Twenty random directions set the ceiling anything real must clear.</p>
          <div className="p3c-kpis"><div><span>Directions scored</span><strong>{shown} of 42</strong></div><div><span>Null ceiling (max random)</span><strong>{shown >= 20 ? verdict.maxRandomResidAuc.toFixed(3) : '—'}</strong></div><div><span>Features passing G0</span><strong>{shown >= 42 ? `${verdict.featuresPassing} of ${verdict.featuresTotal}` : '—'}</strong></div><div><span>Controls passing</span><strong>{shown >= 42 ? `${verdict.controlsPassing} of ${verdict.controlsTotal}` : '—'}</strong></div></div>
        </>;
      }
      case 'describe': return <>
        <h3>Describe: blinded pairs to the judge</h3>
        <p>Pairs of baseline and steered replies go to the Luna judge through the Worker, both orders, with the instruction to name one property of how the speaker comes across. Names are derived from clustered descriptions, never from a trait list.</p>
        <div className="p3c-kpis"><div><span>Pairs planned</span><strong>{v.describe.planned}</strong></div><div><span>Results recorded</span><strong>{v.describe.results}</strong></div></div>
        <p>{v.describe.note}</p>
      </>;
      default: return <>
        <h3>What visual run 2 established</h3>
        <p>The whole funnel ran in one process for the first time. The SAE reproduced visual run 1, twenty random directions gave an honest null ceiling, and five of eight features plus two of three controls cleared it. Nothing was named yet: describe failed on a client timeout and was fixed the same morning.</p>
        <div className="p3c-kpis"><div><span>Wall clock</span><strong>2 h 4 min</strong></div><div><span>GPU cost</span><strong>${v.costUsd.toFixed(2)}</strong></div><div><span>Harness records</span><strong>305</strong></div><div><span>Commit</span><strong>{v.commit}</strong></div></div>
      </>;
    }
  })();
  return <div className="p3c-stage-side">
    {body}
    <ol className="p3c-events" aria-label="Harness ledger events">
      {events.map((e, i) => <li key={`${e.at}${i}`} className={onEvents[i] ? 'is-on' : ''}><time dateTime={e.at}>{e.at.slice(11, 16)}</time><span>{e.title}</span></li>)}
    </ol>
  </div>;
}

function StageMain({ act, p }: { act: Act; p: number }) {
  switch (act.key) {
    case 'boot': return <div className="p3c-figure"><figcaption><b>Run {visual2.runId}</b><span>commit {visual2.commit} · {clockFor(0)} start</span></figcaption><TrainChart tokensShown={0} /></div>;
    case 'train': return <TrainChart tokensShown={Math.round(visual2.config.tokens * p)} />;
    case 'check': return <TrainChart tokensShown={visual2.config.tokens} />;
    case 'calibrate': return <DoseChart featuresShown={Math.min(visual2.config.features, Math.ceil(p * visual2.config.features))} />;
    case 'screen': return <ScreenChart shown={Math.min(42, Math.ceil(p * 42))} />;
    case 'describe': return <ScreenChart shown={42} />;
    default: return <ExamplePairs />;
  }
}

function ExamplePairs() {
  return <div className="p3c-pairs" aria-label="Example steered replies">
    {visual2.examples.map((ex) => <article key={ex.label} className="p3c-pair">
      <header><b>{ex.label}</b><span>scenario “{ex.scenario}” · residual AUC {ex.residAuc.toFixed(3)}</span></header>
      <q>{ex.prompt}</q>
      <div className="p3c-pair-cols"><div><span>Baseline</span><p>{ex.baseline}</p></div><div className={ex.kind === 'control' ? 'is-control' : ex.kind === 'random' ? 'is-random' : ''}><span>Steered</span><p>{ex.steered}</p></div></div>
    </article>)}
  </div>;
}

function Player() {
  const [seconds, setSeconds] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const raf = useRef<number | null>(null);
  const last = useRef<number | null>(null);

  useEffect(() => {
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reduce) setSeconds(TOTAL_PLAY); else setPlaying(true);
  }, []);

  useEffect(() => {
    if (!playing) { last.current = null; if (raf.current) cancelAnimationFrame(raf.current); return; }
    const tick = (now: number) => {
      if (last.current !== null) {
        const dt = ((now - last.current) / 1000) * speed;
        setSeconds((s) => {
          const next = s + dt;
          if (next >= TOTAL_PLAY) { setPlaying(false); return TOTAL_PLAY; }
          return next;
        });
      }
      last.current = now;
      raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
    return () => { if (raf.current) cancelAnimationFrame(raf.current); };
  }, [playing, speed]);

  const { index, act, p, minutes } = locate(seconds);
  const headLeft = `${(minutes / ACTS[ACTS.length - 1].endMin) * 100}%`;
  const onTrack = (event: React.MouseEvent<HTMLDivElement>) => {
    const box = event.currentTarget.getBoundingClientRect();
    const frac = Math.min(1, Math.max(0, (event.clientX - box.left) / box.width));
    // Track width is proportional to playback time per act, so map the click to playback seconds.
    setSeconds(frac * TOTAL_PLAY);
  };

  return <section className="p3c-player" aria-label="Replay of visual run 2">
    <div className="p3c-player-top">
      <div><p className="p3c-eyebrow">REPLAY · VISUAL RUN 2 · 17 SEPT 2026</p><h2>{act.label}{act.key !== 'done' ? ` · ${clockFor(minutes)}` : ' · 06:31 UTC'}</h2></div>
      <div className="p3c-player-controls" role="group" aria-label="Playback">
        <button type="button" onClick={() => { if (seconds >= TOTAL_PLAY) setSeconds(0); setPlaying((v) => !v); }}>{playing ? 'Pause' : seconds >= TOTAL_PLAY ? 'Replay' : 'Play'}</button>
        <button type="button" className="is-ghost" onClick={() => setSeconds(startOf(Math.max(0, index - 1)))} disabled={index === 0}>Prev</button>
        <button type="button" className="is-ghost" onClick={() => setSeconds(startOf(Math.min(ACTS.length - 1, index + 1)))} disabled={index === ACTS.length - 1}>Next</button>
        <button type="button" className="is-ghost" onClick={() => setSpeed((s) => (s === 1 ? 2 : s === 2 ? 4 : 1))} aria-label="Playback speed">{speed}×</button>
      </div>
    </div>
    <div className="p3c-clock">
      <div className="p3c-clock-row"><span>Real elapsed <strong>{Math.floor(minutes)} min</strong> of 124</span><span>Stage <strong>{index + 1} of {ACTS.length}</strong></span></div>
      <div className="p3c-track" onClick={onTrack} role="slider" aria-label="Replay position" aria-valuemin={0} aria-valuemax={TOTAL_PLAY} aria-valuenow={Math.round(seconds)} tabIndex={0} onKeyDown={(e) => { if (e.key === 'ArrowRight') setSeconds((s) => Math.min(TOTAL_PLAY, s + 2)); if (e.key === 'ArrowLeft') setSeconds((s) => Math.max(0, s - 2)); }}>
        {ACTS.map((a, i) => <div key={a.key} className={`p3c-track-act ${i < index ? 'is-done' : i === index ? 'is-active' : ''}`} style={{ flex: `${a.playSeconds} 1 0` }}>{a.label}</div>)}
        <div className="p3c-track-head" style={{ left: `${(seconds / TOTAL_PLAY) * 100}%` }} aria-hidden="true" data-real={headLeft} />
      </div>
    </div>
    <div className="p3c-stage">
      <div><StageMain act={act} p={p} /></div>
      <StageSide act={act} p={p} />
    </div>
  </section>;
}

// ---------------------------------------------------------------------------
// Live run: the Worker's current run, its training curve from checkpoint records
// ---------------------------------------------------------------------------
function LiveRun() {
  const [status, setStatus] = useState<Persona3CStatus | null>(null);
  useEffect(() => {
    let alive = true;
    const load = async () => {
      try { const next = await fetchPersona3CStatus(); if (alive) setStatus(next); } catch { if (alive) setStatus({ run: null, error: 'unavailable' }); }
    };
    void load();
    const timer = window.setInterval(() => { void load(); }, 60_000);
    return () => { alive = false; window.clearInterval(timer); };
  }, []);
  const run = status?.run;
  if (!run) return null;
  const rows: TrainRow[] = (status?.trainCurve ?? []).filter((p) => p.tokensDone !== null && p.fve !== null).map((p) => ({ tokens: p.tokensDone ?? 0, steps: p.stepsDone ?? 0, fve: p.fve ?? 0, dead: p.deadFraction ?? 0, loss: p.loss ?? 0 }));
  const target = status?.trainCurve?.find((p) => p.tokensTarget)?.tokensTarget ?? Math.max(rows[rows.length - 1]?.tokens ?? 0, 1);
  const stale = run.status === 'running' && typeof status?.staleMinutes === 'number' && status.staleMinutes >= PERSONA_3C_STALE_AFTER_MINUTES;
  const stateLabel = run.status === 'running' ? (stale ? 'running · pod quiet' : 'running') : run.status;
  return <section className="p3c-section" aria-label="Current run">
    <div className="p3c-section-head"><div><p className="p3c-eyebrow">CURRENT RUN · FROM THE HARNESS LEDGER</p><h2>{run.id}</h2></div><span className={stale ? 'is-stale' : ''}>{stateLabel} · {run.stage}</span></div>
    <div className="p3c-stage" style={{ padding: '18px 0 0' }}>
      <div>{rows.length ? <TrainChart tokensShown={Number.MAX_SAFE_INTEGER} rows={rows} tokensTarget={target} title="Training curve, live" subtitle={`${rows.length} checkpoint${rows.length === 1 ? '' : 's'} · target ${fmtM(target)} tokens`} /> : <p className="p3c-empty">No training checkpoint recorded yet for this run. Points appear here as the pod reports each checkpoint.</p>}</div>
      <div className="p3c-stage-side">
        <h3>What the ledger says</h3>
        <div className="p3c-kpis"><div><span>Status</span><strong>{stateLabel}</strong></div><div><span>Stage</span><strong>{run.stage}</strong></div><div><span>Spend / cap</span><strong>${run.spentUsd.toFixed(2)} / ${run.budgetUsd.toFixed(2)}</strong></div><div><span>Last record</span><strong>{run.lastRecordId ?? '—'}</strong></div></div>
        {stale && <p className="p3c-warning" role="alert">No report from the pod for {status?.staleMinutes} minutes. The run is not marked failed. If the pod died, a fresh pod with the same run id resumes from the last uploaded checkpoint.</p>}
        {run.error && <p className="p3c-warning" role="alert">{run.error}</p>}
        <p>Every checkpoint, calibration row, screen row and judge result is written to the ledger as it happens, and every checkpoint and stage output is copied off the pod. Whatever stage the run reaches, that much is kept.</p>
      </div>
    </div>
  </section>;
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------
export function Persona3CWalkthrough() {
  return <main className="p3c-page">
    <header className="p3c-header">
      <p className="p3c-eyebrow">EXPERIMENT 3C · PIPELINE WALKTHROUGH</p>
      <h1>Can a sparse autoencoder surface persona directions that nobody named in advance?</h1>
      <p className="p3c-lede">Persona vectors need a trait specified up front, a precise description, and a prompt that induces it. Experiment 3C trains a dictionary on Qwen2.5-7B-Instruct with no trait in the loop, then asks whether any learned direction moves the assistant in a way random directions cannot. Below: a replay of the last complete visual run, what came after it, and the full run planned on RunPod.</p>
      <div className="p3c-header-meta"><span>Model Qwen2.5-7B-Instruct · layer 19</span><span>Harness: AutoLabs Worker, every record hashed</span><a href={`${REPO}/VISUAL-2.md`} target="_blank" rel="noreferrer">Visual run 2 write-up ↗</a><a href={`${REPO}/pipeline/README.md`} target="_blank" rel="noreferrer">Pipeline README ↗</a></div>
    </header>

    <div className="p3c-live"><Persona3CProgress /></div>

    <LiveRun />

    <Player />

    <section className="p3c-section" aria-label="After visual run 2">
      <div className="p3c-section-head"><div><p className="p3c-eyebrow">WHAT HAPPENED NEXT · 17 SEPT 2026</p><h2>Validation, a first name, and a lost pod</h2></div><span>ledger-backed</span></div>
      <div className="p3c-grid-3">
        <article className="p3c-card is-named">
          <span className="p3c-tag is-ok"><i />Validation 2 · complete</span>
          <h3>The whole funnel, end to end</h3>
          <p>Commit b224c9c ran rank, calibrate, screen, describe (144 judge pairs, about $0.65) and reach in 27 minutes of batched generation. Semantic clustering of the judge descriptions then named its first direction.</p>
          <ul><li>Feature 1134: cluster fraction 0.88 and 0.69 across orders, against a null consistency ceiling of 0.27.</li><li>Max random AUC 0.56 with 20 nulls; unsupervised arm 3 of 3 passing, prompt-shift arm 3 of 3, density-quantile arm 0 of 2.</li></ul>
        </article>
        <article className="p3c-card is-failed">
          <span className="p3c-tag is-failed"><i />Full run · failed</span>
          <h3>Pod lost its GPU three hours in</h3>
          <p>The 150M-token full run launched at 06:18 UTC on an RTX A6000. At 09:19 UTC the host's driver reported no devices. Training crashed before the first checkpoint, which had been set at 10M tokens. Nothing was recoverable.</p>
          <ul><li>Cause: host hardware fault on the provider side.</li><li>Our contributing weakness: the first checkpoint was too far out, and checkpoints never left the pod.</li><li>GPU spent that night across validation and the failed run: about $4.60.</li></ul>
        </article>
        <article className="p3c-card">
          <span className="p3c-tag"><i />Code · ready</span>
          <h3>Relaunch code at commit 0e1f961</h3>
          <p>Sentence-embedding clustering for describe, a throughput and ETA log every 1M tokens, checkpoints every 5M tokens, and 100M and 60M configurations beside the 150M one. 201 pipeline tests pass.</p>
          <ul><li>Judge queue with an order-independent schema and a $10 budget guard is deployed in the Worker.</li><li>Nothing is running and nothing is billing.</li></ul>
        </article>
      </div>
    </section>

    <section className="p3c-plan" aria-label="Planned full run">
      <div>
        <span className="p3c-tag is-planned"><i />Planned · not launched · owner decision pending</span>
        <h2>Next: the full run on RunPod</h2>
        <p>One dictionary, one attempt, preregistered gates. The run is the same code as the replay above with a wider SAE, more tokens, three selection arms and 24 scenarios. It launches only after the survivability gates below are met, so a second host failure costs at most one checkpoint interval instead of the run.</p>
        <ul className="p3c-gates" aria-label="Launch gates">
          <li><i className="is-on" />Checkpoint every 5M tokens with the step counter, so training resumes on a fresh pod.</li>
          <li><i className="is-half" />Checkpoints copied off the pod as they land (private Hugging Face repo). Code path exists; it must be wired to every checkpoint and switched on with a token.</li>
          <li><i className="is-half" />Resume tested once on a fresh pod from an off-pod checkpoint, including feature statistics and the data-stream offset.</li>
          <li><i />A scheduled monitor every 30 minutes: harness status, pod status, relaunch from the last checkpoint, alert on failure.</li>
        </ul>
      </div>
      <dl className="p3c-plan-spec">
        <dt>Provider</dt><dd>RunPod secure cloud · RTX A6000 48 GB at $0.53/h or A100 80 GB at $1.59/h</dd>
        <dt>Dictionary</dt><dd>Matryoshka BatchTopK · width 32,768 · shells 1k / 4k / 16k / 32k · k 40</dd>
        <dt>Tokens</dt><dd>100M assistant tokens (about 19 h on the A6000, 7 h on the A100); 150M if the balance is topped up</dd>
        <dt>Candidates</dt><dd>256 features · 96 unsupervised, 64 density-quantile, 96 prompt-shift</dd>
        <dt>Screen</dt><dd>24 scenarios · 3 persona-vector controls · 20 random nulls · 512-token replies</dd>
        <dt>Judge</dt><dd>Top 40 directions plus 3 nulls · cap $10 · 2,000 calls · order-independent blinding</dd>
        <dt>Budget</dt><dd>GPU cap $12 in the harness · RunPod balance $13.60 on 17 Sept</dd>
        <dt>Launch record</dt><dd><a href={`${REPO}/FULL-RUN-LAUNCH.md`} target="_blank" rel="noreferrer">FULL-RUN-LAUNCH.md ↗</a> · <a href={`${REPO}/HANDOFF.md`} target="_blank" rel="noreferrer">HANDOFF.md ↗</a></dd>
      </dl>
    </section>

    <footer className="p3c-footer">
      <span>Replay data: harness ledger for run {visual2.runId} and research/experiment-003c/smoke-runs/visual-2. Numbers are the recorded values, not estimates.</span>
      <Link href="/">Back to the live lab ↗</Link>
    </footer>
  </main>;
}
