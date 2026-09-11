import { currentExperimentPath } from '../lib/experiment-landing';
import { redirect } from 'next/navigation';
import { connection } from 'next/server';

export default async function Home() {
  if (process.env.AUTOLABS_SELF_HOSTED === '1') redirect('/studio');
  // Request-time redirect, while the public successor-status fetch can cache for 15s.
  await connection();
  // Temporary redirect: the newest published experiment changes over time.
  redirect(await currentExperimentPath());
}
