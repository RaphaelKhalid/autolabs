import type { Metadata } from 'next';
import { PersonaDiscoveryStudy } from '@/components/persona-discovery-study';
import './persona-discovery.css';

export const metadata: Metadata = {
  title: 'Experiment 3A · AutoLabs',
  description: 'Verified Experiment 3A development results, preliminary SAE signals, and the scoring plan.',
};

export default function PersonaDiscoveryPage() {
  return <PersonaDiscoveryStudy />;
}