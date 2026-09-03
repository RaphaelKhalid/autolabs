import { afterEach, describe, expect, it, vi } from 'vitest';
import { searchExa } from '../src/exa';

afterEach(() => vi.unstubAllGlobals());

describe('bounded Exa retrieval', () => {
  it('requests five highlighted sources and bounds returned evidence', async () => {
    const fetchMock = vi.fn(async (_url: string, init?: RequestInit) => {
      const request = JSON.parse(String(init?.body)) as Record<string, unknown>;
      expect(request).toMatchObject({ type: 'auto', numResults: 5, contents: { highlights: true } });
      return new Response(JSON.stringify({
        requestId: 'exa-request-1',
        costDollars: { total: 0.012 },
        results: Array.from({ length: 7 }, (_, index) => ({
          title: `Source ${index}`,
          url: `https://example.com/${index}`,
          highlights: ['x'.repeat(1_500), 'second'],
        })),
      }), { status: 200, headers: { 'content-type': 'application/json' } });
    });
    vi.stubGlobal('fetch', fetchMock);

    const result = await searchExa({ apiKey: 'test-only', agentId: 'mira', query: 'Erdos 885' });
    expect(result.requestId).toBe('exa-request-1');
    expect(result.costUsd).toBe(0.012);
    expect(result.sources).toHaveLength(5);
    expect(result.sources[0].highlights[0]).toHaveLength(1_200);
  });
});
