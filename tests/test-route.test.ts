import { afterEach, describe, expect, it, vi } from 'vitest';
import { POST } from '../app/api/test/route';

const ownerToken = 'owner-token-for-sponsored-test-route';
const providerKey = 'server-only-provider-key';

afterEach(() => {
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
});

describe('sponsored test route', () => {
  it('requires the owner credential before contacting the provider', async () => {
    vi.stubEnv('AUTOLABS_ADMIN_TOKEN', ownerToken);
    vi.stubEnv('OPENAI_API_KEY', providerKey);
    const provider = vi.spyOn(globalThis, 'fetch');
    const response = await POST(new Request('https://autolabs.example/api/test', { method: 'POST' }));
    expect(response.status).toBe(401);
    expect(provider).not.toHaveBeenCalled();
  });

  it('rejects caller supplied prompts and model settings', async () => {
    vi.stubEnv('AUTOLABS_ADMIN_TOKEN', ownerToken);
    vi.stubEnv('OPENAI_API_KEY', providerKey);
    const provider = vi.spyOn(globalThis, 'fetch');
    const response = await POST(new Request('https://autolabs.example/api/test', {
      method: 'POST',
      headers: { 'x-autolabs-owner-key': ownerToken, 'content-type': 'application/json' },
      body: JSON.stringify({ prompt: 'spend more', model: 'anything', max_output_tokens: 99999 }),
    }));
    expect(response.status).toBe(400);
    expect(provider).not.toHaveBeenCalled();
  });

  it('makes one fixed small request and keeps secrets server-side', async () => {
    vi.stubEnv('AUTOLABS_ADMIN_TOKEN', ownerToken);
    vi.stubEnv('OPENAI_API_KEY', providerKey);
    const provider = vi.spyOn(globalThis, 'fetch').mockResolvedValue(Response.json({
      id: 'resp-test',
      output: [{ content: [{ type: 'output_text', text: 'OK' }] }],
      usage: { input_tokens: 7, output_tokens: 2 },
    }));

    const response = await POST(new Request('https://autolabs.example/api/test', {
      method: 'POST',
      headers: { authorization: `Bearer ${ownerToken}` },
    }));
    const payload = await response.json() as Record<string, unknown>;
    expect(response.status).toBe(200);
    expect(provider).toHaveBeenCalledTimes(1);
    expect(String(payload.output)).toBe('OK');
    expect(JSON.stringify(payload)).not.toContain(ownerToken);
    expect(JSON.stringify(payload)).not.toContain(providerKey);
    const [url, init] = provider.mock.calls[0] as [string, RequestInit];
    expect(url).toBe('https://api.openai.com/v1/responses');
    expect(JSON.parse(String(init.body))).toEqual({
      model: 'gpt-5.6-luna',
      input: 'Reply with exactly the word OK.',
      max_output_tokens: 32,
      store: false,
    });
  });
});

