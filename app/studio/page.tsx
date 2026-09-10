import type { Metadata } from 'next';
import { ExperimentWorkbench } from '@/components/experiment-workbench';
export const metadata:Metadata={title:'Experiment creator · AutoLabs'};
export const dynamic='force-dynamic';
export default function StudioPage(){return <ExperimentWorkbench selfHosted={process.env.AUTOLABS_SELF_HOSTED==='1'}/>;}
