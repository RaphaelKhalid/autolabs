export async function secretEquals(provided: string, expected: string) {
  const encoder = new TextEncoder();
  const [left, right] = await Promise.all([
    crypto.subtle.digest('SHA-256', encoder.encode(provided)),
    crypto.subtle.digest('SHA-256', encoder.encode(expected)),
  ]);
  return crypto.subtle.timingSafeEqual(left, right);
}

export function bearer(request: Request) {
  const value = request.headers.get('authorization') ?? '';
  return value.startsWith('Bearer ') ? value.slice(7) : '';
}

function hexBytes(value: string) {
  if (!/^[0-9a-f]{64}$/i.test(value)) return null;
  const bytes = new Uint8Array(32);
  for (let index = 0; index < bytes.length; index += 1) {
    bytes[index] = Number.parseInt(value.slice(index * 2, index * 2 + 2), 16);
  }
  return bytes;
}

export async function verifyCallbackSignature(options: {
  secret: string;
  timestamp: string;
  signature: string;
  body: string;
  now?: number;
}) {
  const timestamp = Number(options.timestamp);
  const now = options.now ?? Math.floor(Date.now() / 1000);
  if (!Number.isSafeInteger(timestamp) || Math.abs(now - timestamp) > 300) return false;
  const signature = hexBytes(options.signature);
  if (!signature) return false;
  const encoder = new TextEncoder();
  const key = await crypto.subtle.importKey(
    'raw',
    encoder.encode(options.secret),
    { name: 'HMAC', hash: 'SHA-256' },
    false,
    ['verify'],
  );
  return crypto.subtle.verify('HMAC', key, signature, encoder.encode(`${options.timestamp}.${options.body}`));
}

export function cors(origin: string | null, allowed: string) {
  const accepted = origin === allowed || origin === 'http://localhost:3000' || origin === 'http://localhost:3001';
  return {
    'access-control-allow-origin': accepted && origin ? origin : allowed,
    'access-control-allow-methods': 'GET,POST,OPTIONS',
    'access-control-allow-headers': 'content-type,authorization,x-autolabs-timestamp,x-autolabs-signature',
    'access-control-max-age': '86400',
    vary: 'Origin',
  };
}
