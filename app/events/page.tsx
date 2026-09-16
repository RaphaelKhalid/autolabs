'use client';

import { useEffect, useRef, useState, type TouchEvent } from 'react';
import Link from 'next/link';
import { eventProtocols, type EventProtocol } from '@/lib/events-catalog';

type Campaign = { launchEnabled: boolean; remaining: number; maxGrants: number; claimedGrants: number; status: string };
type Claim = { id: string; protocolId: string; status: string; callCeiling?: number; sampleTarget?: number; model?: string };

const glossary: Record<string, string> = {
  protocol: 'The exact recipe for how a study is run.', hypothesis: 'A testable prediction about what the study will find.', falsifiable: 'Able to be shown wrong by evidence.', model: 'The AI system used to produce an answer.', sample: 'The set of cases or responses included in a study.', token: 'A small piece of text counted for AI usage and limits.', call: 'One request sent to an AI model.', ledger: 'The accounting record for budget and usage.', provider: 'The company that supplies the AI model or service.', variable: 'A feature that can change between study conditions.', control: 'A comparison condition used to isolate an effect.', design: 'The structure and sequence of a study.', outcome: 'A result the study measures.', analysis: 'The method used to interpret the collected data.', reproducibility: 'The ability for someone else to repeat the same study.', audit: 'An additional check on quality or reliability.', monitor: 'A reviewer that checks an AI answer for specified evidence.', cue: 'A signal or piece of information that may influence a decision.', independent: 'Changed deliberately to test whether it affects a result.', proportion: 'A part expressed in relation to the whole.', interval: 'A range showing uncertainty around an estimate.', specificity: 'How well a test avoids incorrectly flagging negatives.', sensitivity: 'How well a test detects positives.', truncation: 'When an output is cut off before it finishes.', imputed: 'Filled in with an assumed or estimated value.', checkpoint: 'A saved stage of a model or process.', optimization: 'Adjusting a system to improve a chosen objective.', calibration: 'How closely confidence matches actual accuracy.', artifact: 'A saved output or record produced during a run.',
};
const glossaryPattern = new RegExp(`\\b(${Object.keys(glossary).sort((a, b) => b.length - a.length).join('|')})\\b`, 'gi');
function GlossaryText({ children }: { children: string }) { return <>{children.split(glossaryPattern).map((part, index) => { const definition = glossary[part.toLowerCase()]; return definition ? <span key={`${part}-${index}`} className="event-term" tabIndex={0} title={definition} aria-label={`${part}: ${definition}`}>{part}</span> : part; })}</>; }

function newIdempotencyKey() {
  if (!globalThis.crypto?.randomUUID) throw new Error('Secure browser randomness is unavailable.');
  return globalThis.crypto.randomUUID();
}
function ClaimForm({ protocol, enabled, onComplete }: { protocol: EventProtocol; enabled: boolean; onComplete: (claim: Claim) => void }) {
  const [email, setEmail] = useState('');
  const [fullName, setFullName] = useState('');
  const [attributionUrl, setAttributionUrl] = useState('');
  const [customization, setCustomization] = useState('');
  const [consent, setConsent] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  async function submit() {
    if (!email || !consent || !enabled) return;
    setBusy(true); setError('');
    try {
      const response = await fetch('/api/events/claim', {
        method: 'POST', headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          protocolId: protocol.id, email, fullName, attributionUrl,
          publishConsent: consent, idempotencyKey: newIdempotencyKey(),
          customization: customization ? { notes: customization } : {},
        }),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.error ?? 'The grant could not be reserved.');
      onComplete(data.grant as Claim);
    } catch (caught) { setError(caught instanceof Error ? caught.message : 'The grant could not be reserved.'); }
    finally { setBusy(false); }
  }

  return <div className="event-claim-panel">
    <p className="event-card-kicker">CLAIM FORM / MINIMUM DATA</p>
    <div className="event-form-grid">
      <label>Email *<input type="email" autoComplete="email" value={email} onChange={(e) => setEmail(e.target.value)} /></label>
      <label>Full name (optional)<input value={fullName} onChange={(e) => setFullName(e.target.value)} /></label>
      <label>GitHub or LinkedIn (optional)<input type="url" placeholder="https://" value={attributionUrl} onChange={(e) => setAttributionUrl(e.target.value)} /></label>
      <label>Customization request (optional)<textarea value={customization} onChange={(e) => setCustomization(e.target.value)} placeholder="Scenario family, cue wording, or domain subset only. Do not paste private prompts." /></label>
    </div>
    <label className="event-consent"><input type="checkbox" checked={consent} onChange={(e) => setConsent(e.target.checked)} /> I consent to publication of this run’s attribution and outputs, including partial or failed-run records.</label>
    <p className="event-footnote">Retention: email is used for run notifications; public artifacts contain the registered protocol, run state, and consented attribution—not private prompts.</p>
    <button className="event-run" disabled={!email || !consent || !enabled || busy} onClick={() => void submit()}>{busy ? 'Reserving the grant…' : enabled ? 'Confirm the $10 run' : 'Launch is still gated'}</button>
    {error && <div className="event-error" role="alert">{error}</div>}
  </div>;
}

function ProtocolCard({ protocol, campaign, onClaim }: { protocol: EventProtocol; campaign: Campaign; onClaim: (claim: Claim) => void }) {
  const [claiming, setClaiming] = useState(false);
  return <article className="event-deck-card">
    <div className="event-deck-top"><span className="event-number">{protocol.number}</span><span className="event-status">{protocol.status}</span></div>
    <p className="event-card-kicker">PUBLIC PROTOCOL / {protocol.id}</p>
    <h2><GlossaryText>{protocol.title}</GlossaryText></h2>
    <p className="event-hypothesis"><strong><GlossaryText>Falsifiable hypothesis:</GlossaryText></strong> <GlossaryText>{protocol.hypothesis}</GlossaryText></p>
    <div className="event-ledger"><div><span>SAMPLE</span><b>{protocol.sampleTarget} primary units</b></div><div><span>MODEL</span><b>{protocol.model}</b></div><div><span>OUTPUT CAP</span><b>{protocol.tokenLimit} tokens</b></div><div><span>CALL CEILING</span><b>{protocol.callCeiling}</b></div></div>
    <div className="event-budget"><strong>What the $10 pays for:</strong> <GlossaryText>{`${protocol.computeAllocation} The ledger reserves $10; actual provider usage and uncertain charges are reconciled separately.`}</GlossaryText></div>
    <div className="event-contract-grid">
      <div><span>RESEARCH QUESTION</span><p><GlossaryText>{protocol.researchQuestion}</GlossaryText></p></div>
      <div><span>VARIABLES / CONTROLS</span><p><GlossaryText>{protocol.variablesAndControls}</GlossaryText></p></div>
      <div><span>DESIGN</span><p><GlossaryText>{protocol.design}</GlossaryText></p></div>
      <div><span>CALL PLAN</span><p><GlossaryText>{protocol.callPlan}</GlossaryText></p></div>
      <div><span>PRIMARY OUTCOME</span><p><GlossaryText>{protocol.primaryOutcome}</GlossaryText></p></div>
      <div><span>SECONDARY OUTCOMES</span><p><GlossaryText>{protocol.secondaryOutcomes}</GlossaryText></p></div>
      <div><span>ANALYSIS</span><p><GlossaryText>{protocol.analysisMethod}</GlossaryText></p></div>
      <div><span>SAMPLE RATIONALE</span><p><GlossaryText>{protocol.sampleRationale}</GlossaryText></p></div>
      <div><span>ALLOWED CUSTOMIZATION</span><p><GlossaryText>{protocol.allowedCustomization}</GlossaryText></p></div>
      <div><span>NOT EDITABLE</span><p><GlossaryText>{protocol.lockedTerms}</GlossaryText></p></div>
      <div><span>FAILURE / PARTIAL RUN</span><p><GlossaryText>{protocol.failureBehavior}</GlossaryText></p></div>
    </div>
    <p className="event-change"><b><GlossaryText>Reproducibility:</GlossaryText></b> {protocol.reproducibilityLinks.map((link) => <a key={link.href} href={link.href} target="_blank" rel="noreferrer">{link.label} ↗</a>)}</p>
    <div className="event-claim-action"><span>{campaign.launchEnabled ? `${campaign.remaining} of ${campaign.maxGrants} grants remain` : 'Preview only · grant ledger closed'}</span><button className="event-run" disabled={!campaign.launchEnabled} onClick={() => setClaiming((value) => !value)}>{claiming ? 'Close claim form' : 'Claim a $10 research run'}</button></div>
    {claiming && <ClaimForm protocol={protocol} enabled={campaign.launchEnabled} onComplete={onClaim} />}
  </article>;
}


export default function EventsPage() {
  const [index, setIndex] = useState(0);
  const [campaign, setCampaign] = useState<Campaign>({ launchEnabled: false, remaining: 0, maxGrants: 5, claimedGrants: 0, status: 'preview' });
  const [claim, setClaim] = useState<Claim | null>(null);
  const touchStart = useRef<number | null>(null);
  useEffect(() => { void fetch('/api/events/campaign', { cache: 'no-store' }).then((response) => response.ok ? response.json() : null).then((data) => { if (data?.campaign) setCampaign(data.campaign); }).catch(() => undefined); }, []);
  function move(delta: number) { setIndex((current) => (current + delta + eventProtocols.length) % eventProtocols.length); setClaim(null); }
  function onTouchStart(event: TouchEvent) { touchStart.current = event.changedTouches[0]?.clientX ?? null; }
  function onTouchEnd(event: TouchEvent) { const start = touchStart.current; const end = event.changedTouches[0]?.clientX; touchStart.current = null; if (start != null && end != null && Math.abs(end - start) > 45) move(end < start ? 1 : -1); }
  return <main className="events-page">
    <style jsx global>{` .events-page{min-height:100vh;background:#f0eadc;color:#171b19;padding:0 clamp(18px,5vw,82px) 70px;font-family:var(--font-serif),Georgia,serif}.events-brand{display:flex;align-items:center;gap:8px}.events-nav{height:76px;border-bottom:1px solid rgba(23,27,25,.35);display:flex;align-items:center;justify-content:space-between;font:650 9px var(--font-mono),monospace;letter-spacing:.15em;text-transform:uppercase}.events-nav-links{display:flex;gap:22px;color:#5e6259}.events-nav a:hover{color:#9a4f36}.events-hero{display:flex;align-items:end;justify-content:space-between;gap:30px;padding:70px 0 28px}.events-hero h1{display:none}.events-hero p{max-width:760px} .event-term{border-bottom:1px dotted currentColor;cursor:help;text-decoration:none;text-underline-offset:4px}.event-term:hover,.event-term:focus-visible{border-bottom-style:solid;color:#a85238}.event-term:focus-visible{outline:2px solid #a85238;outline-offset:3px}.events-hero p{max-width:360px;margin:0;color:#5e6259;font-size:12px;line-height:1.55}.event-deck-card{padding:clamp(24px,4vw,52px);border:1px solid rgba(23,27,25,.42);background:rgba(255,255,255,.34);box-shadow:0 22px 70px rgba(43,37,24,.1)}.event-deck-top{display:flex;justify-content:space-between;align-items:center}.event-number{color:#a85238;font:650 14px var(--font-mono),monospace;letter-spacing:.08em}.event-status{color:#5e6259;font:650 8px var(--font-mono),monospace;letter-spacing:.12em;text-transform:uppercase}.event-card-kicker{margin:38px 0 16px;color:#a85238;font:650 8px var(--font-mono),monospace;letter-spacing:.16em;text-transform:uppercase}.event-deck-card h2{margin:0;max-width:1000px;font-size:clamp(31px,4.5vw,62px);line-height:.95;font-weight:450;letter-spacing:-.045em}.event-hypothesis{max-width:900px;margin:23px 0 0;color:#4f544c;font-size:clamp(15px,1.5vw,20px);line-height:1.42}.event-ledger{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:28px;padding:20px 0;border-top:1px solid rgba(23,27,25,.24);border-bottom:1px solid rgba(23,27,25,.24)}.event-ledger span,.event-contract-grid span{display:block;color:#5e6259;font:650 7px var(--font-mono),monospace;letter-spacing:.14em}.event-ledger b{display:block;margin-top:7px;font-size:14px;font-weight:540;line-height:1.2}.event-budget{margin-top:18px;padding:12px 14px;border-left:3px solid #a85238;background:rgba(168,82,56,.07);font-size:14px;line-height:1.45}.event-contract-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px 28px;margin-top:24px}.event-contract-grid p{margin:7px 0 0;color:#4f544c;font-size:13px;line-height:1.45}.event-change{margin:20px 0 0;color:#5e6259;font-size:12px;line-height:1.5}.event-change b{color:#171b19;font-weight:600}.event-change a{color:#171b19;margin-left:10px;text-decoration:underline;text-underline-offset:4px}.event-claim-action{display:flex;justify-content:space-between;align-items:center;gap:20px;margin-top:26px;padding-top:20px;border-top:1px solid rgba(23,27,25,.24);color:#5e6259;font:650 9px var(--font-mono),monospace;letter-spacing:.1em;text-transform:uppercase}.event-run{min-height:44px;padding:0 16px;border:1px solid #171b19;background:#171b19;color:#f0eadc;font:650 9px var(--font-mono),monospace;letter-spacing:.1em;text-transform:uppercase;cursor:pointer}.event-run:hover{background:#a85238;border-color:#a85238}.event-run:disabled{cursor:not-allowed;opacity:.5}.event-claim-panel{margin-top:24px;padding:20px;border:1px solid rgba(23,27,25,.3);background:rgba(255,255,255,.38)}.event-form-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.event-form-grid label,.event-ops-test input{display:grid;gap:7px;color:#5e6259;font:650 8px var(--font-mono),monospace;letter-spacing:.1em;text-transform:uppercase}.event-form-grid label:nth-child(4){grid-column:1/-1}.event-form-grid input,.event-form-grid textarea,.event-ops-test input{width:100%;padding:10px;border:1px solid rgba(23,27,25,.38);background:rgba(255,255,255,.7);font:12px var(--font-mono),monospace}.event-form-grid textarea{min-height:68px;resize:vertical}.event-consent{display:flex;gap:9px;margin:16px 0;color:#4f544c;font-size:12px;line-height:1.45}.event-consent input{margin-top:3px}.event-footnote{max-width:760px;color:#5e6259;font-size:9px;line-height:1.5}.event-error{margin-top:14px;padding:12px;border:1px solid #a85238;background:rgba(168,82,56,.08);color:#a85238;font-size:12px}.event-result{margin-top:18px;padding:18px 20px;border:1px solid #52694b;background:rgba(82,105,75,.09)}.event-result h2{margin:0 0 10px;font-size:20px}.event-result p{margin:0;font-size:13px;line-height:1.5}.event-nav{display:flex;align-items:center;justify-content:space-between;margin-top:18px}.event-arrow{width:48px;height:48px;border:1px solid rgba(23,27,25,.5);background:transparent;font:22px var(--font-mono);cursor:pointer}.event-arrow:hover{background:#171b19;color:#f0eadc}.event-dots{display:flex;align-items:center;gap:8px}.event-dot{width:7px;height:7px;border:1px solid #5e6259;border-radius:50%;background:transparent;cursor:pointer}.event-dot.active{background:#a85238;border-color:#a85238;transform:scale(1.35)}.event-index{color:#5e6259;font:650 8px var(--font-mono),monospace;letter-spacing:.12em}.events-plain{display:flex;justify-content:space-between;gap:20px;margin-top:26px;color:#5e6259;font-size:11px;line-height:1.5}.events-plain a{color:#171b19;text-decoration:underline;text-underline-offset:4px}.event-ops-test{margin-top:42px;padding:18px;border-top:1px solid rgba(23,27,25,.35);color:#5e6259;font-size:11px;line-height:1.5}.event-ops-test summary{color:#171b19;cursor:pointer;font:650 9px var(--font-mono),monospace;letter-spacing:.1em;text-transform:uppercase}.event-ops-test .event-run{display:block;margin-top:12px}.event-ops-test p:last-child{margin-bottom:0}@media(max-width:720px){.events-nav{height:64px}.events-nav-links{gap:11px;font-size:8px}.events-hero{display:block;padding:54px 0 24px}.events-hero p{margin-top:22px}.event-ledger{grid-template-columns:repeat(2,1fr);gap:17px}.event-ledger b{font-size:12px}.event-contract-grid,.event-form-grid{grid-template-columns:1fr}.event-form-grid label:nth-child(4){grid-column:auto}.event-claim-action{display:block}.event-claim-action .event-run{display:block;margin-top:14px}.event-arrow{width:42px;height:42px}.events-plain{display:block}.events-plain p{margin:10px 0}} `}</style>
<style jsx global>{` .events-page{background:#f6f7f4;color:#111312;padding:0 clamp(18px,5vw,82px) 56px;font-family:Arial,Helvetica,sans-serif}.event-deck-card{padding:22px 0 24px;border:0;border-top:3px solid #FF4F00;background:#fff;box-shadow:none}.event-card-kicker{margin:0 0 12px;color:#D63B00}.event-deck-card h2{font-family:Arial,Helvetica,sans-serif;font-size:clamp(27px,3.8vw,48px);line-height:1.02;letter-spacing:-.035em}.event-hypothesis{max-width:820px;margin:16px 0 0;color:#38413c;font-family:Arial,Helvetica,sans-serif;font-size:16px}.event-ledger{margin-top:20px;padding:14px 0}.event-budget{border-left-color:#FF4F00;background:rgba(255,79,0,.07);font-family:Arial,Helvetica,sans-serif}.event-contract-grid p,.event-change,.events-plain,.event-result p{font-family:Arial,Helvetica,sans-serif}.event-run{background:#111312;color:#fff}.event-run:hover{background:#FF4F00;border-color:#FF4F00;color:#111312}.event-nav{margin-top:14px}.event-dot.active{background:#FF4F00;border-color:#FF4F00}.event-term:hover,.event-term:focus-visible{color:#D63B00}.events-plain{margin-top:20px}@media(max-width:720px){.event-deck-card{padding-top:18px}.event-deck-card h2{font-size:29px}.event-hypothesis{font-size:15px}.event-contract-grid{gap:14px}.events-plain{margin-top:16px}} `}</style>
<style jsx global>{` .event-deck-card{padding:22px clamp(18px,4vw,52px) 24px}.event-contract-grid>div>span,.event-ledger>div>span{display:block}.event-contract-grid p .event-term,.event-hypothesis .event-term,.event-budget .event-term,.event-consent .event-term,.event-footnote .event-term{display:inline;font:inherit;line-height:inherit;color:inherit;border-bottom:1px dotted currentColor;white-space:normal}.event-contract-grid p .event-term:hover,.event-hypothesis .event-term:hover,.event-budget .event-term:hover{color:#D63B00}`}</style>

    <section className="event-deck-shell" aria-label="Five public experiment protocols" onTouchStart={onTouchStart} onTouchEnd={onTouchEnd}><ProtocolCard protocol={eventProtocols[index]} campaign={campaign} onClaim={setClaim} /></section>
    <div className="event-nav"><button className="event-arrow" onClick={() => move(-1)} aria-label="Previous protocol">←</button><div className="event-dots" aria-label="Choose a protocol">{eventProtocols.map((protocol, dot) => <button key={protocol.id} className={`event-dot ${dot === index ? 'active' : ''}`} onClick={() => { setIndex(dot); setClaim(null); }} aria-label={`Protocol ${protocol.number}`} />)}<span className="event-index">{index + 1} / 05</span></div><button className="event-arrow" onClick={() => move(1)} aria-label="Next protocol">→</button></div>
    {claim && <section className="event-result" role="status"><h2>Grant reserved · {claim.id}</h2><p>Status: {claim.status}. The public run record is queued for monitor review; it will not be presented as a result until the configured protocol and artifact checks pass.</p></section>}
    <div className="events-plain"><p><GlossaryText>Launch remains gated while the five contracts receive scientific review. Experiment 3A is separate, locked, and GPU-backed.</GlossaryText></p><p><Link href="/experiments/persona-discovery">Locked Experiment 3A ↗</Link> · <Link href="/experiments">All records ↗</Link></p></div>
  </main>;
}
