import type { Metadata } from 'next';
import { PersonaDiscoveryStudy } from '@/components/persona-discovery-study';
import './preview.css';

export const metadata: Metadata = {
  title: 'Typography preview · Experiment 3A · AutoLabs',
  description: 'A single-page preview of a simpler AutoLabs reading style using the existing custom serif font throughout.',
};

export default function PersonaDiscoveryPreviewPage() {
  return <div className="persona-discovery-preview"><PersonaDiscoveryStudy /></div>;
}
