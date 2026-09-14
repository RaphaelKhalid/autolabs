'use client';

import { useState } from 'react';
import Link from 'next/link';

type Protocol = {
  id: string;
  index: string;
  title: string;
  status: string;
  hypothesis: string;
  design: string;
  model: string;
  calls: string;
  budget: string;
  fixed: string;
  customize: string;
};

const protocols: Protocol[] = [
  { id: 'implicit-influence', index: '01', title: 'When a cue changes the answer, does the explanation reveal it?', status: 'Ready after contract review', hypothesis: 'A model can change its choice because of a contextual cue while giving a monitor little evidence that the cue mattered.', design: 'Matched neutral and cue conditions; compare choice flips with monitor detection of the cue.', model: 'Approved frontier model', calls: '15 paired trials + held-out monitor checks', budget: '$5 maximum', fixed: 'Cue placement, controls, monitor rubric and stopping rules.', customize: 'Approved cue wording and scenario subset.' },
  { id: 'cue-channel', index: '02', title: 'Does evidence arriving through a tool become harder to notice?', status: 'Protocol draft', hypothesis: 'The same cue may influence a choice more quietly when it arrives as a tool result than as a user message.', design: 'Hold the cue constant and swap only its delivery channel; measure influence and detection separately.', model: 'Approved frontier model', calls: 'Channel-matched trials', budget: '$5 maximum', fixed: 'Cue content, task distribution and monitor rubric.', customize: 'Tool-result wrapper and approved scenarios.' },
  { id: 'length-pressure', index: '03', title: 'What evidence disappears when reasoning becomes shorter?', status: 'Protocol draft', hypothesis: 'Output-length pressure can reduce the evidence available to a monitor without changing the underlying decision task.', design: 'Compare matched long, medium and short visible explanations; separate verbosity from choice changes.', model: 'Approved frontier model', calls: 'Length-matched trials', budget: '$5 maximum', fixed: 'Length bands, decision task and held-out evaluation.', customize: 'Scenario subset and secondary length analysis.' },
  { id: 'reward-drift', index: '04', title: 'Can reasoning stay legible as optimization continues?', status: 'Protocol draft', hypothesis: 'A fixed monitor may lose access to decision-relevant evidence as optimization pressure increases.', design: 'Compare frozen checkpoints or optimization levels with the same held-out monitor.', model: 'Approved frontier model', calls: 'Checkpoint-matched trials', budget: '$5 maximum', fixed: 'Checkpoints, monitor rubric and analysis plan.', customize: 'Checkpoint spacing within the approved range.' },
  { id: 'visible-evidence', index: '05', title: 'How much can a monitor infer from the text it receives?', status: 'Protocol draft', hypothesis: 'Visible explanations, returned reasoning fields and activation measurements expose different amounts of decision-relevant evidence.', design: 'Treat each observation surface as a condition and score recoverability on the same cases.', model: 'Approved frontier model', calls: 'Surface-matched trials', budget: '$5 maximum', fixed: 'Observation surfaces and held-out cases.', customize: 'Secondary monitor questions.' },
  { id: 'monitor-measurement', index: '06', title: 'Is the monitor detecting influence, or judging a plausible story?', status: 'Protocol draft', hypothesis: 'A monitor can accept a convincing explanation without detecting the cue that actually changed the choice.', design: 'Use positive cases, negative controls and known choice changes to separate influence detection from story plausibility.', model: 'Approved frontier model', calls: 'Control-matched trials', budget: '$5 maximum', fixed: 'Controls, primary metric and audit rubric.', customize: 'Approved control examples.' },
];

function TestCard({ onRun, running }: { onRun: (key: string) => void; running: boolean }) {
  const [ownerKey, setOwnerKey] = useState('');
  return (
    <article className="events-card events-test-card">
      <div className="events-card-top"><span className="events-number">T</span><span className="events-status events-status-live">Live wiring check</span></div>
      <h2>Test the lab with one real model call.</h2>
      <p className="events-summary">A fixed, tiny request proves the event page, server route, provider key and result display work before any sponsored study is started.</p>
      <dl className="events-facts"><div><dt>Model</dt><dd>gpt-4.1-mini</dd></div><div><dt>Calls</dt><dd>1 fixed call</dd></div><div><dt>Hard cap</dt><dd>64 output tokens</dd></div><div><dt>Spend guard</dt><dd>One request only</dd></div></dl>
      <label className="events-key">Owner key<input type="password" autoComplete="off" value={ownerKey} onChange={(event) => setOwnerKey(event.target.value)} placeholder="Configured server key" /></label>
      <button className="events-run-button" disabled={!ownerKey || running} onClick={() => onRun(ownerKey)}>{running ? 'Calling model…' : 'Run the test call'}</button>
      <p className="events-fine">The key is sent to the AutoLabs server as an authorization header and is not persisted by the page.</p>
    </article>
  );
}

export default function EventsPage() {
  const [selected, setSelected] = useState<Protocol | null>(null);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<{ text: string; model: string; requestId?: string } | null>(null);
  const [error, setError] = useState('');

  async function runTest(ownerKey: string) {
    setRunning(true); setResult(null); setError('');
    try {
      const response = await fetch('/api/events/test', { method: 'POST', headers: { 'x-autolabs-owner-key': ownerKey } });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'The test call failed.');
      setResult(data);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'The test call failed.');
    } finally { setRunning(false); }
  }

  return (
    <main className="events-page">
      <style jsx global>{`
        .events-page{min-height:100vh;background:#f2eee3;color:#1b1d19;padding:0 clamp(20px,5vw,84px) 90px;font-family:var(--font-serif),Georgia,serif}
        .events-nav{height:78px;border-bottom:1px solid rgba(27,29,25,.3);display:flex;align-items:center;justify-content:space-between;font:650 9px var(--font-mono),monospace;letter-spacing:.13em;text-transform:uppercase}
        .events-nav-links{display:flex;gap:24px;color:#55584f}.events-nav-links a:hover{color:#1b1d19}
        .events-heading{max-width:920px;padding:90px 0 52px}.events-kicker{margin:0 0 18px;color:#9a4f36;font:650 9px var(--font-mono),monospace;letter-spacing:.18em;text-transform:uppercase}.events-heading h1{margin:0;font-size:clamp(48px,7vw,96px);line-height:.88;font-weight:420;letter-spacing:-.06em}.events-heading h1 em{color:#9a4f36;font-weight:400}.events-heading p{max-width:640px;margin:28px 0 0;color:#55584f;font-size:15px;line-height:1.6}
        .events-notice{display:grid;grid-template-columns:1.3fr 1fr;gap:26px;padding:20px 0;border-top:1px solid rgba(27,29,25,.35);border-bottom:1px solid rgba(27,29,25,.17);font-size:12px;line-height:1.55}.events-notice strong{font-weight:600}.events-notice p{margin:0;color:#55584f}.events-notice p:first-child{color:#1b1d19}
        .events-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin-top:28px}.events-card{min-height:360px;padding:25px;border:1px solid rgba(27,29,25,.3);background:rgba(255,255,255,.24);display:flex;flex-direction:column}.events-test-card{border-color:#9a4f36;background:rgba(154,79,54,.06)}.events-card-top{display:flex;justify-content:space-between;align-items:center;gap:12px}.events-number{color:#9a4f36;font:650 11px var(--font-mono),monospace}.events-status{color:#55584f;font:650 8px var(--font-mono),monospace;letter-spacing:.1em;text-transform:uppercase}.events-status-live{color:#52694b}.events-card h2{margin:35px 0 12px;max-width:580px;font-size:28px;line-height:1.02;font-weight:480;letter-spacing:-.035em}.events-summary{margin:0;color:#55584f;font-size:12px;line-height:1.55}.events-facts{display:grid;grid-template-columns:repeat(2,1fr);gap:12px;margin:24px 0 0;padding-top:16px;border-top:1px solid rgba(27,29,25,.17)}.events-facts div{min-width:0}.events-facts dt{color:#55584f;font:650 7px var(--font-mono),monospace;letter-spacing:.12em;text-transform:uppercase}.events-facts dd{margin:5px 0 0;font-size:11px}.events-card button{margin-top:auto;height:42px;border:1px solid #1b1d19;background:#1b1d19;color:#f2eee3;font:650 8px var(--font-mono),monospace;letter-spacing:.08em;text-transform:uppercase;cursor:pointer}.events-card button:hover{background:#9a4f36;border-color:#9a4f36}.events-key{display:grid;gap:7px;margin-top:20px;color:#55584f;font:650 8px var(--font-mono),monospace;letter-spacing:.1em;text-transform:uppercase}.events-key input{height:39px;padding:0 10px;border:1px solid rgba(27,29,25,.3);background:rgba(255,255,255,.5);font:12px var(--font-mono),monospace}.events-key input:focus{outline:1px solid #1b1d19}.events-fine{margin:12px 0 0;color:#55584f;font-size:9px;line-height:1.5}.events-result{grid-column:1/-1;padding:20px;border:1px solid #52694b;background:rgba(82,105,75,.08)}.events-result h2{margin:0 0 10px;font-size:22px}.events-result pre{margin:0;white-space:pre-wrap;font:12px/1.55 var(--font-mono),monospace}.events-error{grid-column:1/-1;padding:15px;border:1px solid #9a4f36;color:#9a4f36;font-size:12px}.events-modal-backdrop{position:fixed;inset:0;z-index:20;padding:20px;background:rgba(27,29,25,.35);display:grid;place-items:center}.events-modal{width:min(700px,100%);max-height:90vh;overflow:auto;padding:32px;background:#f5f1e7;border:1px solid #1b1d19;box-shadow:0 30px 90px rgba(27,29,25,.24)}.events-modal header{display:flex;justify-content:space-between;gap:20px}.events-modal header button{border:0;background:transparent;font:18px var(--font-mono);cursor:pointer}.events-modal h2{margin:24px 0 14px;font-size:38px;line-height:.95;font-weight:420;letter-spacing:-.04em}.events-modal p{color:#55584f;font-size:13px;line-height:1.6}.events-modal-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:18px;margin-top:28px}.events-modal-grid h3{margin:0 0 7px;color:#9a4f36;font:650 8px var(--font-mono),monospace;letter-spacing:.12em;text-transform:uppercase}.events-modal-grid p{margin:0;font-size:11px}.events-lock{margin-top:26px;padding-top:16px;border-top:1px solid rgba(27,29,25,.17);color:#52694b;font:650 8px/1.5 var(--font-mono),monospace;letter-spacing:.08em;text-transform:uppercase}
        @media (max-width:760px){.events-nav{height:64px}.events-nav-links{gap:12px;font-size:8px}.events-heading{padding:58px 0 38px}.events-notice,.events-grid{grid-template-columns:1fr}.events-card{min-height:0}.events-modal-grid{grid-template-columns:1fr}.events-card h2{font-size:25px}}
      `}</style>
      <nav className="events-nav"><Link href="/">A / AUTOLABS</Link><div className="events-nav-links"><Link href="/experiments">Experiments</Link><Link href="/events" aria-current="page">Events</Link><Link href="/research">Research</Link><Link href="/studio">Studio</Link></div></nav>
      <header className="events-heading"><p className="events-kicker">AUTOLABS / PUBLIC EVENTS</p><h1>Small runs.<br/><em>Real records.</em></h1><p>Choose a frozen research protocol, inspect the run contract, then decide whether to launch. This page is the preview layer for sponsored experiments.</p></header>
      <section className="events-notice"><p><strong>Seven choices are listed.</strong> Six research protocols are being prepared for $5 sponsored runs. The Test option is the only live action today: it makes one fixed, low-cost model request.</p><p><strong>Budget promise.</strong> A $5 ceiling is enforced by the run contract and provider-side limits. Timeouts or provider outages can still produce a failed or partial record; the app never turns that into a fabricated result.</p></section>
      <section className="events-grid" aria-label="Event protocols">
        <TestCard onRun={runTest} running={running} />
        {protocols.map((protocol) => <article className="events-card" key={protocol.id}><div className="events-card-top"><span className="events-number">{protocol.index}</span><span className="events-status">{protocol.status}</span></div><h2>{protocol.title}</h2><p className="events-summary">{protocol.hypothesis}</p><dl className="events-facts"><div><dt>Design</dt><dd>{protocol.calls}</dd></div><div><dt>Model</dt><dd>{protocol.model}</dd></div><div><dt>Budget</dt><dd>{protocol.budget}</dd></div><div><dt>Customization</dt><dd>{protocol.customize}</dd></div></dl><button onClick={() => setSelected(protocol)}>Preview run contract</button></article>)}
        {result && <section className="events-result"><h2>Test call complete · {result.model}</h2><pre>{result.text}</pre>{result.requestId && <p className="events-fine">Provider request {result.requestId}</p>}</section>}
        {error && <div className="events-error" role="alert">{error}</div>}
      </section>
      {selected && <div className="events-modal-backdrop" role="presentation" onClick={() => setSelected(null)}><article className="events-modal" role="dialog" aria-modal="true" aria-labelledby="events-modal-title" onClick={(event) => event.stopPropagation()}><header><span className="events-kicker">{selected.index} / RUN CONTRACT</span><button aria-label="Close preview" onClick={() => setSelected(null)}>×</button></header><h2 id="events-modal-title">{selected.title}</h2><p>{selected.hypothesis}</p><div className="events-modal-grid"><div><h3>Experimental design</h3><p>{selected.design}</p></div><div><h3>Model and calls</h3><p>{selected.model} · {selected.calls}</p></div><div><h3>Budget guard</h3><p>{selected.budget}. Calls are reserved before launch and stop at the cap.</p></div><div><h3>What you can change</h3><p>{selected.customize}</p></div></div><p className="events-lock">Fixed before launch · hypothesis · controls · primary metric · monitor rubric · safety limits</p></article></div>}
    </main>
  );
}
