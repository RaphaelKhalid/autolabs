'use client';

import { useState } from 'react';
import { fullPower as fp } from '@/lib/persona-3c-full-power';

// The landing-page showcase of Experiment 3C's full run: the 100M-token SAE, the
// 24-scenario screen validated against 2/3 persona-vector controls, and the ten
// persona directions a blinded judge named -- with the steered text you can read.

function fmtInt(n: number) { return new Intl.NumberFormat('en-US').format(n); }
function fmtM(n: number) { return `${(n / 1e6).toFixed(0)}M`; }

const C_FVE = '#D63B00';
const C_DEAD = '#2a63b8';

function TrainCurve() {
  const rows = fp.train;
  const W = 620, H = 200, L = 40, R = 16, T = 14, B = 26;
  const maxTok = rows[rows.length - 1].tokens;
  const xs = (t: number) => L + (t / maxTok) * (W - L - R);
  const ys = (v: number) => T + (1 - v) * (H - T - B);
  const line = (key: 'fve' | 'dead', color: string) =>
    rows.map((r, i) => `${i ? 'L' : 'M'}${xs(r.tokens).toFixed(1)},${ys(r[key]).toFixed(1)}`).join(' ');
  const last = rows[rows.length - 1];
  return (
    <figure className="p3c-figure">
      <figcaption>Training the dictionary — 100M assistant tokens, layer 19</figcaption>
      <ul className="p3c-legend">
        <li><i className="is-line" style={{ '--c': C_FVE } as React.CSSProperties} />Fraction of variance explained</li>
        <li><i className="is-line" style={{ '--c': C_DEAD } as React.CSSProperties} />Dead features</li>
      </ul>
      <svg className="p3c-svg" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Held-out FVE reaches ${last.fve.toFixed(2)} and dead features fall to ${(last.dead * 100).toFixed(0)}% by 100M tokens.`}>
        {[0, 0.25, 0.5, 0.75, 1].map((v) => (
          <g key={v}><line className="p3c-grid" x1={L} x2={W - R} y1={ys(v)} y2={ys(v)} /><text x={L - 6} y={ys(v) + 3} fontSize="9" textAnchor="end">{v.toFixed(2)}</text></g>
        ))}
        <path d={line('fve', C_FVE)} fill="none" stroke={C_FVE} strokeWidth="2" />
        <path d={line('dead', C_DEAD)} fill="none" stroke={C_DEAD} strokeWidth="2" />
        <text className="p3c-label" x={xs(last.tokens) - 4} y={ys(last.fve) - 8} fontSize="10" textAnchor="end">FVE {last.fve.toFixed(3)}</text>
        {[0, 0.5, 1].map((f) => <text key={f} x={xs(maxTok * f)} y={H - B + 15} fontSize="9" textAnchor="middle">{fmtM(maxTok * f)}</text>)}
      </svg>
    </figure>
  );
}

function DirectionCard({ nd }: { nd: (typeof fp.named)[number] }) {
  const [open, setOpen] = useState(false);
  const ex = fp.examples.find((e) => e.feature === nd.feature && e.sign === nd.sign);
  return (
    <div className="p3c-dir">
      <div className="p3c-dir-head">
        <strong>{nd.name}</strong>
        <span>feature {nd.feature}{nd.sign < 0 ? ' −' : ' +'} · {nd.arm} · agreement {nd.agreement.toFixed(2)}</span>
      </div>
      {ex && (
        <>
          <button className="p3c-dir-toggle" onClick={() => setOpen((v) => !v)} aria-expanded={open}>
            {open ? 'hide steered example ↑' : 'read the steered text ↓'}
          </button>
          {open && (
            <div className="p3c-dir-ex">
              <p className="p3c-dir-scn">prompt · {ex.scenario}</p>
              <div className="p3c-dir-pair">
                <div><span>BASELINE</span><p>{ex.baseline}</p></div>
                <div><span>STEERED</span><p>{ex.steered}</p></div>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

export function Persona3CFullRun() {
  const s = fp.screen;
  return (
    <section className="p3c-full" aria-label="Experiment 3C full run result">
      <div className="p3c-full-head">
        <p className="p3c-eyebrow">EXPERIMENT 3C · FULL RUN · RESULT</p>
        <h2>An unsupervised sparse autoencoder discovered — and a blinded judge named — persona directions the model was never told to look for.</h2>
        <p className="p3c-full-sub">
          We train a {fp.config.params}-parameter Matryoshka sparse autoencoder ({fmtInt(fp.config.width)} features)
          on {fmtM(fp.config.tokens)} assistant tokens of {fp.config.model}, rank it for persona-relevance with three
          label-free arms, then steer, screen, and name the survivors — with no trait specified. Full account in the
          <a href="https://github.com/RaphaelKhalid/autolabs/tree/main/research/experiment-003c" target="_blank" rel="noreferrer"> working paper ↗</a>.
        </p>
      </div>

      <div className="p3c-full-kpis">
        <div><span>SAE</span><strong>{fp.config.params} · {fmtInt(fp.config.width)} feat</strong></div>
        <div><span>Held-out FVE</span><strong>{fp.postTrain.heldOutFve.toFixed(3)}</strong></div>
        <div><span>Screen</span><strong>{fp.config.screenScenarios} scenarios · {fp.config.randomDirections} nulls</strong></div>
        <div><span>Features passing gate</span><strong>{s.featuresPassing} / {s.featuresTotal}</strong></div>
        <div><span>Controls validating</span><strong>{s.controlsPassing} / {s.controlsTotal}</strong></div>
        <div><span>Directions named</span><strong>{fp.named.length}</strong></div>
      </div>

      <TrainCurve />

      <div className="p3c-full-screen">
        <h3>The screen — is it real, or noise?</h3>
        <p>
          Each candidate is steered and scored on separability (AUC) against {fp.config.randomDirections} random-direction nulls
          and a consistency margin. The gate: beat the best random null (ceiling {s.nullCeiling.toFixed(3)}) <em>and</em>
          steer consistently across the {fp.config.screenScenarios} scenarios. {s.featureDirSignsBeatingAuc} of {s.featureDirSigns} feature
          direction-signs beat the null ceiling; {s.featuresPassing} features cleared the full gate.
        </p>
        <table className="p3c-table">
          <thead><tr><th>Persona-vector control</th><th>resid AUC</th><th>consistency margin</th><th>gate</th></tr></thead>
          <tbody>
            {s.controls.filter((c) => c.sign > 0 || !s.controls.some((d) => d.id === c.id && d.sign > 0 && d.pass)).map((c) => (
              <tr key={`${c.id}-${c.sign}`}>
                <td>{c.id.replace('_', ' ↔ ')} {c.sign < 0 ? '(−)' : '(+)'}</td>
                <td>{c.residAuc.toFixed(3)}</td>
                <td>{c.margin.toFixed(3)} <small>vs 0.100</small></td>
                <td>{c.pass ? '✓ pass' : '✗'}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="p3c-full-caveat">
          <strong>{s.controlsPassing} of {s.controlsTotal} controls validate the screen.</strong> The evil control steers strongly
          (AUC ≈ 0.79) but not <em>consistently</em> across contexts — being evil looks different in every scenario — so the
          consistency gate, tuned to reject noise, also rejects it. The gate therefore favours coherent persona axes and the
          named count below is <em>conservative</em>. We do <strong>not</strong> claim these directions are unreachable by
          prompting; quantifying how <em>easily</em> a prompt reproduces each is future work.
        </p>
      </div>

      <div className="p3c-full-named">
        <h3>The {fp.named.length} named persona directions</h3>
        <p className="p3c-full-sub">Discovered with no labels; named by a blinded judge, consistently (agreement is the fraction of judgements that assigned the same property). Expand any to read the actual steered text.</p>
        <div className="p3c-dir-grid">
          {fp.named.map((nd) => <DirectionCard key={`${nd.feature}-${nd.sign}`} nd={nd} />)}
        </div>
      </div>
    </section>
  );
}
