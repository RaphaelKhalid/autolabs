import type { Metadata } from 'next';
import { Caveat } from 'next/font/google';
import './handwritten-preview.css';

const hand = Caveat({
  subsets: ['latin'],
  weight: ['400', '500', '600', '700'],
  variable: '--font-hand',
  display: 'swap',
});

export const metadata: Metadata = {
  title: 'Handwritten font preview · AutoLabs',
  description: 'A one-page preview of a handwritten display font for AutoLabs.',
};

export default function PersonaDiscoveryHandwrittenPreviewPage() {
  return <main className={`handwritten-study ${hand.variable}`}>
    

    <header className="handwritten-study__header">
      <p className="hand-label">Experiment 3A</p>
      <h1>Unsupervised persona discovery</h1>
      <p className="handwritten-study__lede">Can SAE features selected without persona labels reveal repeatable behavior that prompting-based persona extraction misses?</p>
      <p className="handwritten-study__status"><strong>Current status:</strong> Discovery and development are recorded. Confirmation has not been run, so there is no confirmed finding yet.</p>
      <div className="handwritten-study__links"><a href="https://afterlight-research.vercel.app/#/questions/q-unsupervised-persona" target="_blank" rel="noreferrer">Source question ↗</a><a href="https://www.kaggle.com/code/raphaelkhalid0/unsupervisedsaes" target="_blank" rel="noreferrer">Working notebook ↗</a></div>
    </header>

    <section className="handwritten-study__section">
      <p className="hand-label">What we are testing</p>
      <ul>
        <li><strong>Label-free discovery:</strong> select up to 32 SAE features from 1,024 responses without using a persona label or target trait.</li>
        <li><strong>Development screen:</strong> steer candidates in both directions on 12 neutral prompts and keep only candidates with a repeatable behavioral signal.</li>
        <li><strong>Held-out confirmation:</strong> test at most three candidates across 600 scenarios each. This phase remains gated.</li>
      </ul>
    </section>

    <p className="handwritten-study__note"><span>Font sample</span> Handwritten display type for the brand and headings; regular text stays in the existing readable typeface.</p>
  </main>;
}
