'use client';

import { useState } from 'react';
import { fullPower as fp } from '@/lib/persona-3c-full-power';

// The AutoLabs landing page: Experiment 3C presented as a long-form working-paper
// blog post -- the full run's training curve and screen woven into the narrative,
// with the ten named persona directions as readable, expandable evidence.

function fmtInt(n: number) { return new Intl.NumberFormat('en-US').format(n); }
function fmtM(n: number) { return `${(n / 1e6).toFixed(0)}M`; }
const C_FVE = '#c2410c';
const C_DEAD = '#2a63b8';

function H2({ id, children }: { id: string; children: React.ReactNode }) {
  return <h2 id={id} className="art-h2"><a href={`#${id}`} aria-label="section link">{children}</a></h2>;
}

function TrainCurve() {
  const rows = fp.train;
  const W = 640, H = 210, L = 42, R = 16, T = 14, B = 30;
  const maxTok = rows[rows.length - 1].tokens;
  const xs = (t: number) => L + (t / maxTok) * (W - L - R);
  const ys = (v: number) => T + (1 - v) * (H - T - B);
  const line = (key: 'fve' | 'dead') => rows.map((r, i) => `${i ? 'L' : 'M'}${xs(r.tokens).toFixed(1)},${ys(r[key]).toFixed(1)}`).join(' ');
  const last = rows[rows.length - 1];
  return (
    <figure className="art-fig">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Held-out FVE reaches ${last.fve.toFixed(2)} and dead features fall below 1% by 100M tokens.`}>
        {[0, 0.25, 0.5, 0.75].map((v) => (
          <g key={v}><line x1={L} x2={W - R} y1={ys(v)} y2={ys(v)} stroke="#e7e3dd" /><text x={L - 6} y={ys(v) + 3} fontSize="9" textAnchor="end" fill="#8a8178">{v.toFixed(2)}</text></g>
        ))}
        <line x1={L} x2={W - R} y1={ys(0)} y2={ys(0)} stroke="#c9c3ba" />
        <path d={line('fve')} fill="none" stroke={C_FVE} strokeWidth="2.2" />
        <path d={line('dead')} fill="none" stroke={C_DEAD} strokeWidth="2.2" />
        <text x={xs(last.tokens) - 4} y={ys(last.fve) - 8} fontSize="10" textAnchor="end" fill={C_FVE}>FVE {last.fve.toFixed(3)}</text>
        <text x={xs(rows[1].tokens) + 6} y={ys(rows[1].dead) - 6} fontSize="10" fill={C_DEAD}>dead features</text>
        {[0, 25, 50, 75, 100].map((f) => <text key={f} x={xs(1e6 * f)} y={H - B + 16} fontSize="9" textAnchor="middle" fill="#8a8178">{f}M</text>)}
      </svg>
      <figcaption><em>Figure 1.</em> Training the width-{fmtInt(fp.config.width)} Matryoshka SAE on {fmtM(fp.config.tokens)} assistant-position tokens of {fp.config.model}, layer {fp.config.layer}. Held-out fraction of variance explained climbs to ≈0.75 and the dead-feature fraction falls below 1%; both plateau well inside the token budget.</figcaption>
    </figure>
  );
}

function DirectionCard({ nd }: { nd: (typeof fp.named)[number] }) {
  const [open, setOpen] = useState(false);
  const ex = fp.examples.find((e) => e.feature === nd.feature && e.sign === nd.sign);
  return (
    <div className="art-dir">
      <div className="art-dir-head">
        <strong>{nd.name}</strong>
        <span>feature {nd.feature}{nd.sign < 0 ? ' −' : ' +'} · {nd.arm} arm · judge agreement {nd.agreement.toFixed(2)}</span>
      </div>
      {ex && (
        <>
          <button className="art-dir-toggle" onClick={() => setOpen((v) => !v)} aria-expanded={open}>
            {open ? '− hide steered text' : '+ read the steered text'}
          </button>
          {open && (
            <div className="art-dir-ex">
              <p className="art-dir-scn">Prompt · {ex.scenario}</p>
              <div className="art-dir-pair">
                <div><span>UNSTEERED</span><p>{ex.baseline}</p></div>
                <div><span>STEERED ALONG THE DIRECTION</span><p>{ex.steered}</p></div>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

export function Persona3CArticle() {
  const s = fp.screen;
  const controlRows = s.controls.filter((c) => c.sign > 0 || !s.controls.some((d) => d.id === c.id && d.sign > 0 && d.pass));
  return (
    <article className="art">
      <header className="art-header">
        <p className="art-kicker">AUTOLABS · EXPERIMENT 3C · WORKING PAPER</p>
        <h1>Unsupervised Discovery of Persona-Relevant Directions with Matryoshka Sparse Autoencoders</h1>
        <p className="art-byline">Raphael Khalid · <a href="https://autolabs-ebon.vercel.app" target="_blank" rel="noreferrer">AutoLabs</a> · September 2026 · <span className="art-badge">working draft</span></p>
        <p className="art-links">
          <a href="https://github.com/RaphaelKhalid/autolabs/blob/main/research/experiment-003c/paper/persona-directions.pdf" target="_blank" rel="noreferrer">Paper (PDF)</a>
          <a href="https://github.com/RaphaelKhalid/autolabs/tree/main/research/experiment-003c" target="_blank" rel="noreferrer">Code &amp; artifacts</a>
          <a href="https://huggingface.co/RaphaelRaphaelRaphael/autolabs-3c-sae" target="_blank" rel="noreferrer">SAE on the Hub</a>
        </p>
      </header>

      <div className="art-takeaways">
        <h2>Core takeaways</h2>
        <ul>
          <li>A sparse autoencoder trained only to reconstruct a chat model&apos;s activations, with <strong>no trait ever specified</strong>, contains persona-relevant directions — we recover and name them.</li>
          <li>A powered screen validated the method against <strong>{s.controlsPassing} of {s.controlsTotal}</strong> supervised persona-vector controls; <strong>{s.featuresPassing} of {s.featuresTotal}</strong> features cleared the full gate; a blinded judge gave <strong>{fp.named.length}</strong> of them consistent names — <em>warmth, empathy, formality, playfulness, confidence</em>.</li>
          <li>We do <strong>not</strong> claim these are unreachable by prompting — that&apos;s unfalsifiable. The open question, in the Persona Vectors authors&apos; own words, is whether SAEs surface traits &ldquo;that cannot be <em>easily</em> elicited through prompting.&rdquo; Measuring that is future work.</li>
        </ul>
      </div>

      <section>
        <H2 id="question">Can you find a model&apos;s personality without naming it first?</H2>
        <p>Persona vectors — directions in a language model&apos;s activation space that control traits like sycophancy or dishonesty (Chen et al., 2025) — are found by a supervised recipe: name the trait, write prompts that elicit and suppress it, and difference the activations. The authors list three requirements of this recipe: it &ldquo;requires specifying a target trait in advance,&rdquo; it &ldquo;depends on providing a precise natural-language description,&rdquo; and it &ldquo;requires that the specified trait is inducible by system prompting.&rdquo; So you can only find traits you already thought to name.</p>
        <p>We take up a conjecture the same authors make: that &ldquo;SAEs may therefore enable unsupervised discovery of persona-relevant directions, including specific traits that cannot be easily elicited through prompting.&rdquo; We build that conjecture into a full discover-and-screen pipeline, and drop their first two requirements — <strong>no trait is specified and no description is written at any point</strong>.</p>
      </section>

      <section>
        <H2 id="method">The pipeline</H2>
        <p>Everything runs on <code>{fp.config.model}</code>, hooking the residual stream at layer {fp.config.layer}.</p>
        <ol className="art-steps">
          <li><strong>Train.</strong> A Matryoshka BatchTopK sparse autoencoder — {fp.config.params} parameters, {fmtInt(fp.config.width)} features, k={fp.config.k} — on {fmtM(fp.config.tokens)} assistant-position tokens. Each decoder row is a candidate direction.</li>
          <li><strong>Rank.</strong> Score the {fmtInt(fp.config.width)}-feature dictionary for persona-relevance with three <em>label-free</em> arms (assistant-specificity, density-quantile, prompt-shift) → {fp.config.candidates} candidates.</li>
          <li><strong>Calibrate &amp; screen.</strong> Steer along each candidate, then test how separably it shifts behaviour across {fp.config.screenScenarios} scenarios, against {fp.config.randomDirections} random-direction nulls and {fp.config.controls} persona-vector controls.</li>
          <li><strong>Name.</strong> A blinded judge names the survivors — but only if the supervised controls first clear the same gate.</li>
        </ol>
      </section>

      <section>
        <H2 id="training">Training the dictionary</H2>
        <p>The recipe trains cleanly at production scale: held-out fraction of variance explained {fp.postTrain.heldOutFve.toFixed(3)}, L0 exactly {fp.postTrain.heldOutL0}, at width {fmtInt(fp.config.width)}. The trained SAE is reused for every downstream screen without retraining.</p>
        <TrainCurve />
      </section>

      <section>
        <H2 id="screen">The screen: signal, or noise?</H2>
        <p>Each direction is steered and scored on residual-stream separability (a logistic-regression AUC) against {fp.config.randomDirections} random-direction nulls, plus a consistency margin — does the steering effect point the same way across scenarios? The gate <strong>G0</strong>: beat the best random null (ceiling {s.nullCeiling.toFixed(3)}) <em>and</em> clear the consistency margin. {s.featureDirSignsBeatingAuc} of {s.featureDirSigns} feature direction-signs beat the null ceiling; <strong>{s.featuresPassing} of {s.featuresTotal}</strong> features cleared the full gate.</p>
        <p>The crucial check is the positive control: the supervised persona-vectors run through the identical screen. Nothing gets named unless they validate it.</p>
        <div className="art-tablewrap">
          <table className="art-table">
            <thead><tr><th>Persona-vector control</th><th>resid AUC</th><th>consistency margin</th><th>G0</th></tr></thead>
            <tbody>
              {controlRows.map((c) => (
                <tr key={`${c.id}-${c.sign}`}>
                  <td>{c.id.replace('_', ' ↔ ')} {c.sign < 0 ? '(−)' : '(+)'}</td>
                  <td>{c.residAuc.toFixed(3)} <span className="art-mut">&gt; {s.nullCeiling.toFixed(2)}</span></td>
                  <td>{c.margin.toFixed(3)} <span className="art-mut">vs 0.10</span></td>
                  <td>{c.pass ? '✓' : '✗'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="art-callout">
          <strong>{s.controlsPassing} of {s.controlsTotal} controls validate the screen.</strong> Sycophantic and hallucinating clear the gate cleanly. The <em>evil</em> control separates strongly on AUC (≈0.79) but fails the <em>consistency</em> margin: being evil looks different in every scenario, so its steering direction is heterogeneous, and the consistency gate — tuned to reject noise — rejects it too. That means our gate favours coherent persona axes and penalises diffuse ones, so the named count below is <strong>conservative</strong>: an &ldquo;evil-like&rdquo; feature would likely be gated out as well.
        </div>
      </section>

      <section>
        <H2 id="directions">The {fp.named.length} named persona directions</H2>
        <p>Discovered with no labels, then named by a blinded judge — consistently. &ldquo;Agreement&rdquo; is the fraction of blinded judgements that assigned the direction the same property. Expand any card to read the model&apos;s actual output, unsteered versus steered along the direction.</p>
        <div className="art-dir-grid">
          {fp.named.map((nd) => <DirectionCard key={`${nd.feature}-${nd.sign}`} nd={nd} />)}
        </div>
      </section>

      <section>
        <H2 id="discussion">Discussion &amp; limitations</H2>
        <p>The claim is scoped to one model, one layer, and one training mixture; the directions are causal under steering, but the map from an SAE feature to a human-named persona is judged, not proven. The consistency gate is sensitive to scenario count — an earlier 8-scenario screen was under-powered, its controls fell just under the margin, and the pipeline correctly named nothing; going to {fp.config.screenScenarios} scenarios is what lifted the controls over the gate.</p>
        <p>We deliberately do <strong>not</strong> claim to overcome the third requirement, prompt-inducibility. A direction we can discover and steer is almost certainly reproducible by <em>some</em> prompt; &ldquo;prompt-unreachable&rdquo; is close to unfalsifiable and we avoid it. What the result establishes is independence from the first two requirements — no trait named, no description written.</p>
        <div className="art-callout art-callout-future">
          <strong>Future work — quantifying &ldquo;easily.&rdquo;</strong> The sharp question is the graded form of the third requirement: not <em>whether</em> a prompt can induce a discovered direction, but <em>how easily</em> — how specific, how long, or how many attempts a prompt needs to match the steering effect. A calibrated elicitation-cost metric would rank directions from trivially prompt-inducible to stubbornly hard-to-elicit, and ask whether that hard tail is where unsupervised discovery earns its keep.
        </div>
      </section>

      <section>
        <H2 id="contributions">Related work &amp; contributions</H2>
        <p>The idea that SAEs might surface persona directions is Chen et al.&apos;s conjecture, with an initial appendix exploration; we take it up rather than originate it. Unsupervised behaviour-steering directions have also been found without SAEs, and SAE feature steering and LLM-judge naming are standard tools. Our contribution is the <strong>operationalisation and validation</strong>: a purpose-built SAE ranked by three label-free arms, a null-calibrated causal screen, and — most distinctively — a <strong>positive-control gate</strong> that runs the supervised persona-vectors through the identical screen and withholds all naming unless they clear it. That turns &ldquo;these features look persona-like&rdquo; into a falsifiable validity check, and is exactly what made the under-powered screen correctly name nothing.</p>
      </section>

      <footer className="art-foot">
        <p>Run <code>{fp.runId}</code> on an {fp.gpu}. Every stage is ledger-backed with frozen protocol hashes; the SAE, checkpoints, screen records, and judge outputs are on the <a href="https://huggingface.co/RaphaelRaphaelRaphael/autolabs-3c-sae" target="_blank" rel="noreferrer">Hugging Face Hub</a>. Built on AutoLabs, a reproducible harness for observable agent experiments.</p>
        <p className="art-cite">Chen, Arditi, Sleight et al. <em>Persona Vectors: Monitoring and Controlling Character Traits in Language Models.</em> arXiv:2507.21509 (2025).</p>
      </footer>
    </article>
  );
}
