import type { Metadata } from 'next';
import { PersonaDiscoveryStudy } from '@/components/persona-discovery-study';
import './persona-discovery.css';

export const metadata: Metadata = {
  title: 'Unsupervised persona discovery · AutoLabs',
  description: 'Live monitoring for the bounded Experiment 3A unsupervised persona discovery study.',
};

export default function PersonaDiscoveryPage() {
  return <PersonaDiscoveryStudy />;
}