import Link from 'next/link';
import type { Metadata } from 'next';
import { experiments } from '@/lib/experiment-catalog';

export const metadata: Metadata = { title: 'Experiments · AutoLabs' };
export default function ExperimentsPage() {
  return <main className="archive-page">
    <nav className="archive-nav" aria-label="Primary"><Link href="/">A / AUTOLABS</Link><Link href="/journal">Lab journal</Link></nav>
    <header className="archive-heading"><p className="archive-kicker">AUTOLABS</p><h1>Experiments</h1><p>Research objectives, configurations, results and supporting records for AutoLabs experiments.</p></header>
    <section aria-label="Experiments">{experiments.map((experiment) => <article className="experiment-entry" key={experiment.slug}>
      <div className="archive-kicker">{experiment.label} / {experiment.status} / {experiment.rounds} rounds</div>
      <h2><Link href={`/experiments/${experiment.slug}`}>{experiment.title} ↗</Link></h2>
      <p>{experiment.objective}</p><p className="archive-caption">{experiment.researchers} researchers · {experiment.model} · {experiment.reasoning} reasoning</p>
      <Link className="archive-button" href={`/experiments/${experiment.slug}`}>Read the results</Link>
    </article>)}</section>
    <section className="archive-section"><h2>Create an experiment</h2><p>Arrange agents, assign models and export a configuration in the visual creator. Run it with your own credentials on the self-hosted runner.</p><p>The initial templates support a known-answer systems test and research notes requiring human review. Study-specific evaluators and tools must be implemented separately. The public creator cannot launch jobs or access private credentials.</p><Link className="archive-button" href="/studio">Open the creator</Link></section>
  </main>;
}
