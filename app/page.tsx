import type { Metadata } from 'next';
import { Persona3CFullRun } from '@/components/persona-3c-full-run';
import { Persona3CWalkthrough } from '@/components/persona-3c-walkthrough';
import './experiments/persona-discovery-scoring/persona-3b.css';
import './experiments/persona-discovery-3c/persona-3c-walkthrough.css';

export const metadata: Metadata = {
  title: 'Experiment 3C · AutoLabs',
  description: 'An unsupervised Matryoshka sparse autoencoder discovered and named persona-relevant directions in Qwen2.5-7B — validated against 2 of 3 persona-vector controls. Full run, ledger-backed, with the steered text.',
};

export default function Home() {
  return (
    <>
      <Persona3CFullRun />
      <Persona3CWalkthrough />
    </>
  );
}
