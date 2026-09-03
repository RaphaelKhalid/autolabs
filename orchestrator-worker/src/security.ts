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

export function cors(origin: string | null, allowed: string) {
  const accepted = origin === allowed || origin === 'http://localhost:3000' || origin === 'http://localhost:3001';
  return {
    'access-control-allow-origin': accepted && origin ? origin : allowed,
    'access-control-allow-methods': 'GET,POST,OPTIONS',
    'access-control-allow-headers': 'content-type,authorization,x-autolabs-callback',
    'access-control-max-age': '86400',
    vary: 'Origin',
  };
}
