import { describe, expect, it } from 'vitest';
import { normalizeChildPath, parseAfterlightMessage, parseParentView, parseQuestionContext } from '../lib/afterlight-contract';

describe('Afterlight parent contract', () => {
  it('accepts the supported child routes and rejects unbounded paths', () => {
    expect(normalizeChildPath('/atlas')).toBe('/atlas');
    expect(normalizeChildPath('/questions/q-implicit-influence')).toBe('/questions/q-implicit-influence');
    expect(normalizeChildPath('/designer/q-implicit-influence')).toBe('/designer/q-implicit-influence');
    expect(normalizeChildPath('/lab/demo')).toBe('/lab/demo');
    expect(normalizeChildPath('/results/run_123')).toBe('/results/run_123');
    expect(normalizeChildPath('/questions/q-1?redirect=https://evil.test')).toBeNull();
    expect(normalizeChildPath(`/questions/${'x'.repeat(129)}`)).toBeNull();
    expect(parseParentView('/unknown')).toBe('/atlas');
  });

  it('requires version 1 and validates each incoming envelope', () => {
    expect(parseAfterlightMessage({ type: 'afterlight:ready', version: 1 })).toEqual({ type: 'afterlight:ready', version: 1 });
    expect(parseAfterlightMessage({ type: 'afterlight:navigation', version: 1, path: '/atlas' })).toEqual({ type: 'afterlight:navigation', version: 1, path: '/atlas' });
    expect(parseAfterlightMessage({ type: 'afterlight:navigation', version: 2, path: '/atlas' })).toBeNull();
    expect(parseAfterlightMessage({ type: 'afterlight:design', version: 1, question: { id: 'q-1', title: 'A bounded question', sourceUrl: 'https://example.com/paper' } })).toMatchObject({ type: 'afterlight:design', question: { id: 'q-1' } });
    expect(parseAfterlightMessage({ type: 'afterlight:design', version: 1, question: { id: 'q-1', title: 'A question', sourceUrl: 'javascript:alert(1)' } })).toBeNull();
  });

  it('matches child identifier and payload bounds at their limits', () => {
    const id = 'q-' + 'a'.repeat(125) + '~';
    expect(normalizeChildPath('/questions/' + id)).toBe('/questions/' + id);
    expect(parseAfterlightMessage({ type: 'afterlight:design', version: 1, question: { id, title: 'x'.repeat(300), sourceUrl: 'https://example.com/' + 'a'.repeat(2028) } })).toMatchObject({ type: 'afterlight:design' });
    expect(parseAfterlightMessage({ type: 'afterlight:design', version: 1, question: { id, title: 'x'.repeat(301), sourceUrl: 'https://example.com/source' } })).toBeNull();
  });

  it('keeps Studio handoff context bounded and source linked', () => {
    expect(parseQuestionContext({ id: 'q-1', title: '  A question  ', sourceUrl: 'https://example.com/source' })).toEqual({ id: 'q-1', title: 'A question', sourceUrl: 'https://example.com/source' });
    expect(parseQuestionContext({ id: 'q-1', title: '', sourceUrl: 'https://example.com/source' })).toBeNull();
  });
});
