import type { Metadata } from 'next';
import { AutolabsHome } from '@/components/autolabs-home';

export const metadata: Metadata = {
  title: 'AutoLabs · Observable agent experiments',
  description: 'A public laboratory for agent experiments, research records and reproducible evidence.',
};

export default function Home() {
  if (process.env.AUTOLABS_SELF_HOSTED === '1') return <AutolabsHome />;
  return <AutolabsHome />;
}
