import type { Metadata } from 'next';
import { Persona3CWalkthrough } from '@/components/persona-3c-walkthrough';
import '../persona-discovery-scoring/persona-3b.css';
import './persona-3c-walkthrough.css';

export const metadata: Metadata = {
  title: 'Experiment 3C · Pipeline walkthrough · AutoLabs',
  description: 'A replay of the Experiment 3C visual run 2 through every pipeline stage, the failed full-run attempt, and the planned RunPod relaunch.',
};

export default function PersonaDiscovery3CPage() {
  return <Persona3CWalkthrough />;
}
