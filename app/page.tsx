import type { Metadata } from 'next';
import { Persona3CArticle } from '@/components/persona-3c-article';
import './experiments/persona-discovery-3c/persona-3c-article.css';

export const metadata: Metadata = {
  title: 'Unsupervised Discovery of Persona-Relevant Directions · AutoLabs',
  description: 'A Matryoshka sparse autoencoder trained on Qwen2.5-7B discovered — and a blinded judge named — 10 persona-relevant directions with no labels, validated against 2 of 3 persona-vector controls. Working paper.',
};

export default function Home() {
  return <Persona3CArticle />;
}
