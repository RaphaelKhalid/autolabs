import { LivingLab } from '@/components/living-lab';
import { redirect } from 'next/navigation';

export const dynamic = 'force-dynamic';

export default function Home() {
  if (process.env.AUTOLABS_SELF_HOSTED === '1') redirect('/studio');
  return <LivingLab />;
}
