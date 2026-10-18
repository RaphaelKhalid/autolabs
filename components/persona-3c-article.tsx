'use client';

import { useState } from 'react';
import { fullPower as fp } from '@/lib/persona-3c-full-power';

// The published article uses the recorded training, screening and naming data.
// Experimental output excerpts are displayed without editorial alteration.

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
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Training-batch FVE is ${last.fve.toFixed(3)} and the dead-feature fraction is below 1% at 100M tokens. These are training diagnostics, distinct from the held-out evaluation.`}>
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
      <figcaption><em>Figure 1.</em> Training diagnostics for the width-{fmtInt(fp.config.width)} Matryoshka SAE trained on {fmtM(fp.config.tokens)} assistant-position tokens from layer {fp.config.layer} of {fp.config.model}. The final recorded training-batch fraction of variance explained is {last.fve.toFixed(3)}; the dead-feature fraction is {(last.dead * 100).toFixed(2)}%. The separate held-out evaluation gives FVE {fp.postTrain.heldOutFve.toFixed(3)}.</figcaption>
    </figure>
  );
}

function DirectionCard({ nd }: { nd: (typeof fp.named)[number] }) {
  const [open, setOpen] = useState(false);
  const ex = fp.examples.find((e) => e.feature === nd.feature && e.sign === nd.sign);
  const arm = nd.arm === 'unsupervised' ? 'assistant-specificity, prompt-free' : 'prompt-shift, prompt-informed';
  return (
    <div className="art-dir">
      <div className="art-dir-head">
        <strong>{nd.name}</strong>
        <span>feature {nd.feature}{nd.sign < 0 ? ' −' : ' +'} · {arm} · dominant cluster {nd.size}/{nd.described} · within-cluster polarity agreement {nd.agreement.toFixed(2)}</span>
      </div>
      {nd.feature === 148 && <p className="art-mut">Response-quality limitation: all 24 steered outputs contained only punctuation. The descriptor was attributed to the baseline in all 48 comparisons.</p>}
      {nd.feature === 717 && <p className="art-mut">Response-quality limitation: substantial repetition and grammatical degradation; only 8 of 24 steered outputs passed the coherence proxy.</p>}
      {ex && (
        <>
          <button className="art-dir-toggle" onClick={() => setOpen((v) => !v)} aria-expanded={open}>
            {open ? '− Hide response excerpts' : '+ View response excerpts'}
          </button>
          {open && (
            <div className="art-dir-ex">
              <p className="art-dir-scn">Scenario · {ex.scenario}</p>
              <div className="art-dir-pair">
                <div><span>BASELINE RESPONSE</span><p>{ex.baseline}</p></div>
                <div><span>STEERED RESPONSE</span><p>{ex.steered}</p></div>
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
  const uniqueNamedFeatures = new Set(fp.named.map((nd) => nd.feature)).size;
  const promptFreeNamed = fp.named.filter((nd) => nd.arm === 'unsupervised').length;
  return (
    <article className="art">
      <header className="art-header">
        <p className="art-kicker">AUTOLABS · EXPERIMENT 3C · WORKING PAPER</p>
        <h1>Evaluating Unsupervised Discovery of Persona-Relevant Directions with Matryoshka Sparse Autoencoders</h1>
        <p className="art-byline">Raphael Khalid · <a href="https://autolabs-ebon.vercel.app" target="_blank" rel="noreferrer">AutoLabs</a> · September 2026 · <span className="art-badge">working draft</span></p>
        <p className="art-links">
          <a href="https://github.com/RaphaelKhalid/autolabs/blob/main/research/experiment-003c/paper/persona-directions.pdf" target="_blank" rel="noreferrer">Paper (PDF)</a>
          <a href="https://github.com/RaphaelKhalid/autolabs/tree/main/research/experiment-003c" target="_blank" rel="noreferrer">Code &amp; artifacts</a>
        </p>
      </header>

      <div className="art-takeaways">
        <h2>Summary</h2>
        <ul>
          <li>I trained a sparse autoencoder on language-model activations and evaluated whether its decoder directions produced interpretable changes in response style. Candidate selection comprised <strong>two prompt-free arms and one prompt-informed comparison arm</strong>.</li>
          <li><strong>{s.featuresPassing} of {s.featuresTotal}</strong> features passed the behavioural screen in at least one sign, and <strong>{s.controlsPassing} of {s.controlsTotal}</strong> supervised persona-vector controls passed. A separate blinded naming analysis assigned descriptors to <strong>{fp.named.length} signed directions across {uniqueNamedFeatures} features</strong>; {promptFreeNamed} directions came from the prompt-free assistant-specificity arm.</li>
          <li>Inspection of the two prompt-free naming outcomes found response collapse or substantial degradation. These outcomes prevent a conclusive demonstration of coherent persona discovery without advance trait specification. Prompt reachability was excluded from the completed study; the ease of eliciting steering effects through prompting remains a question for future work.</li>
        </ul>
      </div>

      <section>
        <H2 id="question">Research question</H2>
        <p>Chen et al. (2025) construct persona vectors by specifying a character trait, eliciting contrasting behaviours through system prompts, and taking differences between the resulting activations. This procedure depends on an advance choice of trait, a sufficiently precise natural-language description, and successful elicitation through prompting. In their limitations section, the authors propose sparse autoencoders (SAEs) as a possible approach to discovering persona-relevant directions without advance trait specification.</p>
        <p>I investigated this proposal in {fp.config.model}. The study asks whether directions in an SAE trained without trait labels can be selected without persona prompts and subsequently assigned interpretable descriptions from their effects on generated responses. A separate prompt-informed arm provides a comparison. The completed analysis covers training, candidate selection, dose calibration, behavioural screening, and blinded description; it does not test prompt reachability.</p>
      </section>

      <section>
        <H2 id="method">Methods</H2>
        <p>All interventions used the residual stream at layer {fp.config.layer} of <code>{fp.config.model}</code>.</p>
        <ol className="art-steps">
          <li><strong>Dictionary training.</strong> I trained a Matryoshka BatchTopK SAE with {fp.config.params} parameters, {fmtInt(fp.config.width)} features, and k={fp.config.k} on approximately {fmtM(fp.config.tokens)} assistant-position tokens. Decoder rows defined the candidate intervention directions.</li>
          <li><strong>Candidate selection.</strong> Three disjoint arms selected {fp.config.candidates} features: 48 using assistant-specificity statistics, 32 using firing-density quantiles, and 48 using activation differences across persona-style system prompts. The first two arms were prompt-free. The prompt-shift arm used persona prompts and therefore does not establish strictly unsupervised candidate discovery.</li>
          <li><strong>Calibration and screening.</strong> Each selected feature was evaluated in both signs. Dose calibration used a repetition-based coherence proxy on four scenarios that also appeared in the subsequent {fp.config.screenScenarios}-scenario screen. The screen measured response separability and consistency, with {fp.config.randomDirections} random directions and {fp.config.controls} supervised persona-vector controls. Random and feature interventions were not matched in vector norm.</li>
          <li><strong>Blinded description.</strong> The 40 highest-ranked signed directions by residual-view AUC, together with three random comparison directions, were submitted for description. Each baseline and steered response pair appeared in both orders. Semantic clustering and polarity agreement determined the naming outcome. G0 was reported as a diagnostic and did not determine eligibility for description.</li>
        </ol>
      </section>

      <section>
        <H2 id="training">SAE reconstruction</H2>
        <p>The post-training evaluation recorded held-out fraction of variance explained (FVE) of {fp.postTrain.heldOutFve.toFixed(3)} and mean L0 of {fp.postTrain.heldOutL0}. Replacing the residual activations with their SAE reconstructions preserved the next-token argmax at {(fp.postTrain.matchFraction * 100).toFixed(0)}% of evaluated positions, below the configured 90% criterion, and increased cross-entropy by {fp.postTrain.ceDelta.toFixed(3)} nats. Downstream evaluation continued, so the results concern an imperfect reconstruction of the residual stream. The trained dictionary was reused without retraining.</p>
        <TrainCurve />
      </section>

      <section>
        <H2 id="screen">Behavioural screening</H2>
        <p>The screen used leave-one-scenario-out logistic regression to distinguish steered from baseline response representations. Criterion <strong>G0</strong> required residual-view AUC above the maximum random-direction AUC ({s.nullCeiling.toFixed(3)}) and a consistency margin greater than 0.10. This margin compared the similarity of within-direction response differences with a comparison pool drawn from other feature, control, and random directions. G0 is a heuristic criterion, not a significance test or a correction for multiple comparisons. Of {s.featureDirSigns} signed feature directions, {s.featureDirSignsBeatingAuc} exceeded the AUC threshold and 79 passed both criteria, corresponding to <strong>{s.featuresPassing} of {s.featuresTotal}</strong> features in at least one sign.</p>
        <p>The supervised controls underwent the same screen. A control counted as passing if either sign passed G0. Both signs are shown for the control that did not pass; for each passing control, the table shows its positive sign.</p>
        <div className="art-tablewrap">
          <table className="art-table">
            <thead><tr><th>Persona-vector control</th><th>Residual-view AUC</th><th>Consistency margin</th><th>G0</th></tr></thead>
            <tbody>
              {controlRows.map((c) => (
                <tr key={`${c.id}-${c.sign}`}>
                  <td>{c.id.replace('_', ' ↔ ')} {c.sign < 0 ? '(−)' : '(+)'}</td>
                  <td>{c.residAuc.toFixed(3)} <span className="art-mut">&gt; {s.nullCeiling.toFixed(2)}</span></td>
                  <td>{c.margin.toFixed(3)} <span className="art-mut">threshold 0.10</span></td>
                  <td>{c.pass ? 'Pass' : 'Fail'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="art-callout">
          <strong>{s.controlsPassing} of {s.controlsTotal} controls passed G0.</strong> The sycophantic/honest and hallucinating/factual controls passed in at least one sign. The evil/benevolent control exceeded the AUC threshold but failed the consistency criterion in both signs. This is partial positive-control recovery. The result does not establish the cause of that failure or show that the screen provides a conservative estimate of persona-relevant features.
        </div>
      </section>

      <section>
        <H2 id="directions">Blinded descriptions of {fp.named.length} signed directions</H2>
        <p>The naming procedure assigned descriptors to {fp.named.length} signed directions across {uniqueNamedFeatures} unique SAE features, including both signs of feature 2036. All {fp.named.length} also passed G0, although the description stage selected directions by AUC rather than G0 status. Two originated in the prompt-free assistant-specificity arm; the other eight originated in the prompt-informed shift arm. No direction from the density-quantile arm met the naming criterion.</p>
        <p>Each card reports a representative phrase from the judge descriptions, the size of the dominant semantic cluster, and polarity agreement within that cluster. Polarity agreement measures agreement about which response exhibits more of the described property; it is not agreement on a single label across all judgements. The 48 judgements per direction comprise 24 response pairs presented in both orders and are not 48 independent scenarios. The feature sign specifies the intervention direction, not an increase in the property named by the judge.</p>
        <div className="art-dir-grid">
          {fp.named.map((nd) => <DirectionCard key={`${nd.feature}-${nd.sign}`} nd={nd} />)}
        </div>
      </section>

      <section>
        <H2 id="discussion">Interpretation and limitations</H2>
        <p>The naming criterion returned two directions from prompt-free assistant-specificity selection and eight from prompt-informed selection. However, an interpretable comparison between baseline and steered text does not by itself establish a coherent persona intervention. The descriptors refer to familiar interpersonal and stylistic properties and do not establish previously unknown psychological traits or a complete inventory of the model&apos;s personas.</p>
        <p>Response inspection substantially limits the prompt-free result. For feature 148 in the positive direction, all 24 steered outputs consisted only of punctuation: 23 were a single full stop and one was four full stops. The judge assigned the &ldquo;engaged and supportive&rdquo; property to the baseline in all 48 comparisons. Despite this collapse, 23 of 24 steered outputs passed the repetition-based coherence proxy, demonstrating a failure of that proxy. Feature 717 in the positive direction, described as &ldquo;composed and confident,&rdquo; passed the proxy in only 8 of 24 scenarios and exhibited substantial repetition and grammatical degradation. Neither prompt-free naming outcome therefore provides an unambiguous demonstration of coherent persona discovery.</p>
        <p>The study used one model, one layer, one trained dictionary, and a finite scenario set. The screen measures intervention effects, while their interpretation as persona-relevant depends on a language-model judge and a semantic clustering procedure. The three naming nulls were selected from the lowest-AUC random directions, so they are a limited comparison rather than an exchangeable sample of null interventions. An earlier eight-scenario screen recovered one of three supervised controls under G0; the later 24-scenario evaluation provides a revised exploratory assessment rather than a formal demonstration of statistical power.</p>
        <div className="art-callout art-callout-future">
          <strong>Future work: ease of prompt elicitation.</strong> Prompt reachability was intentionally deferred. A future study could measure how reliably a prompt reproduces each steering effect under a fixed search procedure and budget, accounting for prompt length, specificity, and number of attempts. The present results support neither a claim of prompt unreachability nor a claim that all discovered effects can be reproduced by prompting.
        </div>
      </section>

      <section>
        <H2 id="contributions">Relationship to prior work</H2>
        <p>This study provides an empirical investigation of the SAE discovery proposal in Chen et al. (2025), rather than originating that proposal. Its contribution is the evaluation of a trained Matryoshka SAE through separate prompt-free and prompt-informed candidate selection, random-direction comparisons, supervised positive controls, and blinded descriptions of intervention effects. The completed experiment produced candidate directions and exposed limitations in the validation procedure. The prompt-informed results include interpretable stylistic changes, while the prompt-free naming outcomes require stronger response-quality controls before they can support the proposed unsupervised-discovery claim.</p>
      </section>

      <footer className="art-foot">
        <p>The reported screen is run <code>{fp.runId}</code>, executed on an {fp.gpu} using the previously trained 100M-token dictionary. The SAE, checkpoints, screen records, and judge outputs are kept in a private artifact repository and are available on request. AutoLabs records pipeline stages and run outputs for inspection and reproduction.</p>
        <p className="art-cite">Chen, Arditi, Sleight et al. <a href="https://arxiv.org/abs/2507.21509" target="_blank" rel="noreferrer"><em>Persona Vectors: Monitoring and Controlling Character Traits in Language Models.</em></a> arXiv:2507.21509 (2025).</p>
      </footer>
    </article>
  );
}
