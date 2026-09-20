import type { Metadata } from 'next';
import { Persona3CWalkthrough } from '@/components/persona-3c-walkthrough';
import '../persona-discovery-scoring/persona-3b.css';
import './persona-3c-walkthrough.css';

export const metadata: Metadata = {
  title: 'Experiment 3C · Pipeline walkthrough · AutoLabs',
  description: 'An archived replay of Experiment 3C visual run 2, with training history and ledger status. The completed study and findings are available on the AutoLabs landing page.',
};

export default function PersonaDiscovery3CPage() {
  return <Persona3CWalkthrough />;
}
