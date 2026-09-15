import { describe, expect, it } from 'vitest';
import { eventProtocols } from '../lib/events-catalog';
import manifest from '../research/experiment-003a/launch-manifest.json';

describe('public event and locked 3A contracts', () => {
  it('exposes exactly five public protocols with unique IDs', () => {
    expect(eventProtocols).toHaveLength(5);
    expect(new Set(eventProtocols.map((protocol) => protocol.id)).size).toBe(5);
    expect(eventProtocols.map((protocol) => protocol.number)).toEqual(['01', '02', '03', '04', '05']);
  });

  it('does not advertise the undefined optimization treatment as ready', () => {
    const protocol = eventProtocols.find((item) => item.id === 'event-04-optimization-legibility');
    expect(protocol?.status).toContain('HOLD');
  });

  it('keeps the corrected tool and surface envelopes explicit', () => {
    expect(eventProtocols.find((item) => item.id === 'event-02-tool-evidence')?.callCeiling).toBe(72);
    expect(eventProtocols.find((item) => item.id === 'event-05-plausible-story')?.callCeiling).toBe(144);
  });

  it('keeps Experiment 3A bounded and confirmation-disabled', () => {
    expect(manifest.studyId).toBe('experiment-003a-v1');
    expect(manifest.phase).toBe('development');
    expect(manifest.confirmationEnabled).toBe(false);
    expect(manifest.maxRuntimeSeconds).toBe(6600);
    expect(manifest.apiBudgetUsd).toBe(0);
    expect(manifest.compute.accelerator).toBe('NvidiaTeslaT4');
  });
});

