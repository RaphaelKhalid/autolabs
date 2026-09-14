import type { Metadata } from 'next';
import { ResearchSurface } from '@/components/research-surface';
import { parseParentView } from '@/lib/afterlight-contract';

export const metadata: Metadata = { title: 'Research · AutoLabs', description: 'Browse the embedded Afterlight research atlas inside Auto Labs.' };
export const dynamic = 'force-dynamic';

export default async function ResearchPage({ searchParams }: { searchParams: Promise<{ view?: string | string[] }> }) {
  const params = await searchParams;
  const view = Array.isArray(params.view) ? params.view[0] : params.view;
  return <ResearchSurface initialPath={parseParentView(view)} />;
}
