import type { Metadata } from 'next';
import { LivingLab } from '@/components/living-lab';

export const metadata: Metadata = {title: 'Animated pilot archive · AutoLabs', description: 'Archived observatory for the completed 100-round Erdős 885 pilot.'};
export default function PilotLabPage() {return <LivingLab />;}
