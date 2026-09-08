import Link from 'next/link';
import type { Metadata } from 'next';
import { experiments } from '@/lib/experiment-catalog';

export const metadata: Metadata = { title: 'Experiments · AutoLabs' };
export default function ExperimentsPage() {
  return <main className="archive-page">
    <nav className="archive-nav" aria-label="Primary"><Link href="/">A / AUTOLABS</Link><Link href="/journal">Lab journal</Link></nav>
    <header className="archive-heading"><p className="archive-kicker">THE EXPERIMENT REGISTER</p><h1>One lab.<br/><em>Many questions.</em></h1><p>AutoLabs is a home for observable, repeatable agent experiments. Each experiment keeps its own question, model configuration, tools, budget and evidence.</p></header>
    <section aria-label="Experiments">{experiments.map((experiment) => <article className="experiment-entry" key={experiment.slug}>
      <div className="archive-kicker">{experiment.label} / {experiment.status} / {experiment.rounds} rounds</div>
      <h2><Link href={`/experiments/${experiment.slug}`}>{experiment.title} ↗</Link></h2>
      <p>{experiment.objective}</p><p className="archive-caption">{experiment.researchers} researchers · {experiment.model} · {experiment.reasoning} reasoning</p>
      <Link className="archive-button" href={`/experiments/${experiment.slug}`}>Read the results</Link>
    </article>)}</section>
    <section className="archive-section"><h2>The next experiment is not scheduled.</h2><p>The animated lab stays. Future experiments can use different questions, models, tools and success criteria while preserving this pilot’s record. The current execution engine is still specialized for Erdős 885; a general-purpose launch workflow is the next development step, not an existing feature.</p><p>Experiments are owner-run. Visitors can observe the published records.</p></section>
  </main>;
}
