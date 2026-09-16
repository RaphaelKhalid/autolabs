import type { Metadata } from 'next';
import { Persona3BLab } from '@/components/persona-3b-lab';
import './experiments/persona-discovery-scoring/persona-3b.css';

export const metadata: Metadata = {
  title: 'Experiment 3B · AutoLabs',
  description: 'Live blinded Luna High scoring of the completed Experiment 3A screen.',
};

export default function Home() {
  return <Persona3BLab />;
}