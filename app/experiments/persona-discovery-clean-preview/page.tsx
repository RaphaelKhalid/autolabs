import type { Metadata } from 'next';
import './clean-preview.css';

export const metadata: Metadata = {
  title: 'Experiment 3A · Clean page preview · AutoLabs',
  description: 'A simpler, more readable preview of the Experiment 3A study page.',
};

export default function PersonaDiscoveryCleanPreviewPage() {
  return <main className="clean-study">
    

    <header className="clean-study__header">
      <p className="clean-label">Experiment 3A</p>
      <h1>Unsupervised persona discovery</h1>
      <p className="clean-study__lede">Can SAE features selected without persona labels reveal repeatable behavior that prompting-based persona extraction misses?</p>
      <p className="clean-study__status"><strong>Current status:</strong> Discovery and development are recorded. Confirmation has not been run, so there is no confirmed finding yet.</p>
      <div className="clean-study__links"><a href="https://afterlight-research.vercel.app/#/questions/q-unsupervised-persona" target="_blank" rel="noreferrer">Source question ↗</a><a href="https://www.kaggle.com/code/raphaelkhalid0/unsupervisedsaes" target="_blank" rel="noreferrer">Working notebook ↗</a></div>
    </header>

    <section className="clean-study__section">
      <p className="clean-label">What we are testing</p>
      <ul>
        <li><strong>Label-free discovery:</strong> select up to 32 SAE features from 1,024 responses without using a persona label or target trait.</li>
        <li><strong>Development screen:</strong> steer candidates in both directions on 12 neutral prompts and keep only candidates with a repeatable behavioral signal.</li>
        <li><strong>Held-out confirmation:</strong> test at most three candidates across 600 scenarios each. This phase remains gated.</li>
      </ul>
    </section>

    <section className="clean-study__section">
      <p className="clean-label">How to read this page</p>
      <ul>
        <li><strong>A candidate is a hypothesis, not a persona.</strong> A human-readable label is an interpretation that must be tested.</li>
        <li><strong>A confirmed finding requires held-out evidence.</strong> Steering must produce consistent behavior across new contexts, not just familiar examples.</li>
        <li><strong>This page will separate protocol, status, and results.</strong> No confirmation result is being claimed at this stage.</li>
      </ul>
    </section>
  </main>;
}
