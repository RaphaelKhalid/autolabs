import type { Metadata } from 'next';
import { Persona3CArticle } from '@/components/persona-3c-article';
import './experiments/persona-discovery-3c/persona-3c-article.css';

export const metadata: Metadata = {
  title: 'Evaluating Unsupervised Discovery of Persona-Relevant Directions · AutoLabs',
  description: 'A study of persona-relevant directions in a Matryoshka sparse autoencoder trained on Qwen2.5-7B-Instruct, with separate prompt-free and prompt-informed candidate selection. Working paper.',
};

export default function Home() {
  return <Persona3CArticle />;
}
