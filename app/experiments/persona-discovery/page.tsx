import type { Metadata } from 'next';
import { PersonaDiscoveryStudy } from '@/components/persona-discovery-study';
import './persona-discovery.css';

export const metadata: Metadata = {
  title: 'Unsupervised persona discovery · AutoLabs',
  description: 'A read-only proposal and public status surface for the planned unsupervised persona discovery study.',
};

export default function PersonaDiscoveryPage() {
  return <PersonaDiscoveryStudy />;
}