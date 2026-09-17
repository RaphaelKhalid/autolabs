import type { Metadata } from 'next';
import { Persona3CWalkthrough } from '@/components/persona-3c-walkthrough';
import './experiments/persona-discovery-scoring/persona-3b.css';
import './experiments/persona-discovery-3c/persona-3c-walkthrough.css';

export const metadata: Metadata = {
  title: 'Experiment 3C · AutoLabs',
  description: 'Live: can a Matryoshka sparse autoencoder surface persona-relevant directions that prompting cannot reach? Current run, ledger-backed, plus a replay of the last visual run.',
};

export default function Home() {
  return <Persona3CWalkthrough />;
}
