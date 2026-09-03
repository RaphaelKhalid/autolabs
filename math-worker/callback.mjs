import crypto from 'node:crypto';
import fs from 'node:fs';

const EXPECTED_CALLBACK = 'https://autolabs-orchestrator.raphaelbahadurkhan.workers.dev';
let payload = {};
try {
  payload = JSON.parse(process.env.JOB_PAYLOAD ?? '{}');
} catch {
  payload = {};
}

const callback = String(process.env.CALLBACK_URL ?? '').replace(/\/$/, '');
const secret = process.env.CALLBACK_SECRET;
if (callback !== EXPECTED_CALLBACK || !secret) process.exit(1);

if (!fs.existsSync('result.json')) {
  fs.writeFileSync('result.json', JSON.stringify({
    id: String(payload.id ?? 'unknown'),
    ok: false,
    complete: false,
    error: 'Compute process ended before producing a result artifact.',
    result: null,
  }));
}

const body = fs.readFileSync('result.json', 'utf8');
if (Buffer.byteLength(body, 'utf8') > 1_000_000) process.exit(1);

for (let attempt = 0; attempt < 3; attempt += 1) {
  const timestamp = String(Math.floor(Date.now() / 1000));
  const signature = crypto.createHmac('sha256', secret).update(`${timestamp}.${body}`).digest('hex');
  try {
    const response = await fetch(`${callback}/api/jobs/result`, {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        'x-autolabs-timestamp': timestamp,
        'x-autolabs-signature': signature,
      },
      body,
    });
    if (response.ok || response.status === 409) process.exit(0);
    if (response.status < 500 && response.status !== 425 && response.status !== 429) process.exit(1);
  } catch {
    // Retry a transient network failure below.
  }
  await new Promise((resolve) => setTimeout(resolve, 2 ** attempt * 2_000));
}

process.exit(1);
