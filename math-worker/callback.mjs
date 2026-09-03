import fs from 'node:fs';

const payload = JSON.parse(process.env.JOB_PAYLOAD ?? '{}');
const callback = payload.callbackUrl;
const secret = process.env.CALLBACK_SECRET;
if (!callback || !secret || !fs.existsSync('result.json')) process.exit(1);
const result = fs.readFileSync('result.json', 'utf8');
const response = await fetch(`${callback}/api/jobs/result`, {
  method: 'POST',
  headers: { 'content-type': 'application/json', 'x-autolabs-callback': secret },
  body: result,
});
if (!response.ok) process.exit(1);
