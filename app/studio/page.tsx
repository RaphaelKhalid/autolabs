import type { Metadata } from 'next';
import { notFound } from 'next/navigation';
import { ExperimentStudio } from '@/components/experiment-studio';
export const metadata:Metadata={title:'Experiment creator · AutoLabs'};
export const dynamic='force-dynamic';
export default function StudioPage(){if(process.env.AUTOLABS_SELF_HOSTED!=='1')notFound();return <ExperimentStudio/>;}
