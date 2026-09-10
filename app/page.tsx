import { latestExperiment } from '../lib/experiment-catalog';
import { redirect } from 'next/navigation';

export const dynamic = 'force-dynamic';

export default function Home() {
  if (process.env.AUTOLABS_SELF_HOSTED === '1') redirect('/studio');
  // Temporary redirect: the newest published experiment changes over time.
  redirect(`/experiments/${latestExperiment.slug}`);
}
