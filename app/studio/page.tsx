import type { Metadata } from 'next';
import { ExperimentWorkbench } from '@/components/experiment-workbench';
import { parseQuestionContext } from '@/lib/afterlight-contract';

export const metadata: Metadata = { title: 'Experiment creator · AutoLabs' };
export const dynamic = 'force-dynamic';

export default async function StudioPage({ searchParams }: { searchParams: Promise<{ question?: string; title?: string; source?: string }> }) {
  const params = await searchParams;
  const questionContext = parseQuestionContext({ id: params.question, title: params.title, sourceUrl: params.source });
  return <ExperimentWorkbench selfHosted={process.env.AUTOLABS_SELF_HOSTED === '1'} questionContext={questionContext} />;
}