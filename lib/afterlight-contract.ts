export const AFTERLIGHT_ORIGIN = 'https://afterlight-research.vercel.app';
export const AFTERLIGHT_URL = `${AFTERLIGHT_ORIGIN}/?embed=autolabs`;
export const MAX_CHILD_PATH_LENGTH = 256;

const ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._~-]{0,127}$/;

export type AfterlightQuestion = { id: string; title: string; sourceUrl: string };
export type AfterlightChildMessage =
  | { type: 'afterlight:ready'; version: 1 }
  | { type: 'afterlight:navigation'; version: 1; path: string }
  | { type: 'afterlight:design'; version: 1; question: AfterlightQuestion };
export type AutolabsNavigateMessage = { type: 'autolabs:navigate'; version: 1; path: string };

export function isQuestionId(value: unknown): value is string {
  return typeof value === 'string' && ID_PATTERN.test(value);
}

export function normalizeChildPath(value: unknown): string | null {
  if (typeof value !== 'string' || value.length === 0 || value.length > MAX_CHILD_PATH_LENGTH) return null;
  if (value.includes('?') || value.includes('#') || value.includes('\\')) return null;
  const parts = value.split('/');
  if (parts[0] !== '' || parts.length < 2 || parts.length > 3 || parts.some((part) => part.length > 128)) return null;
  if (parts[1] === 'atlas') return parts.length === 2 ? '/atlas' : null;
  if (!['questions', 'designer', 'lab', 'results'].includes(parts[1]) || parts.length !== 3 || !isQuestionId(parts[2])) return null;
  return value;
}

function isSourceUrl(value: unknown): value is string {
  if (typeof value !== 'string' || value.length === 0 || value.length > 2048) return false;
  try {
    const url = new URL(value);
    return url.protocol === 'http:' || url.protocol === 'https:';
  } catch { return false; }
}

export function parseAfterlightMessage(value: unknown): AfterlightChildMessage | null {
  if (!value || typeof value !== 'object') return null;
  const message = value as Record<string, unknown>;
  if (message.version !== 1 || typeof message.type !== 'string') return null;
  if (message.type === 'afterlight:ready') return { type: message.type, version: 1 };
  if (message.type === 'afterlight:navigation') {
    const path = normalizeChildPath(message.path);
    return path ? { type: message.type, version: 1, path } : null;
  }
  if (message.type === 'afterlight:design') {
    if (!message.question || typeof message.question !== 'object') return null;
    const question = message.question as Record<string, unknown>;
    if (!isQuestionId(question.id) || typeof question.title !== 'string' || question.title.trim().length === 0 || question.title.length > 300 || !isSourceUrl(question.sourceUrl)) return null;
    return { type: message.type, version: 1, question: { id: question.id, title: question.title.trim(), sourceUrl: question.sourceUrl } };
  }
  return null;
}

export function parseParentView(value: unknown): string {
  return normalizeChildPath(value) ?? '/atlas';
}

export function parseQuestionContext(value: { id?: unknown; title?: unknown; sourceUrl?: unknown } | null | undefined): AfterlightQuestion | null {
  if (!value || !isQuestionId(value.id) || typeof value.title !== 'string' || value.title.trim().length === 0 || value.title.length > 300 || !isSourceUrl(value.sourceUrl)) return null;
  return { id: value.id, title: value.title.trim(), sourceUrl: value.sourceUrl };
}
