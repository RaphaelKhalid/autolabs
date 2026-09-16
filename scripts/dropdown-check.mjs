import fs from 'node:fs';

const endpoint = process.argv[2] ?? 'http://127.0.0.1:9333';
const url = process.argv[3] ?? 'https://autolabs-ebon.vercel.app/';
const tabs = await (await fetch(`${endpoint}/json/list`)).json();
let tab = tabs.find((item) => item.type === 'page');
if (!tab) {
  throw new Error('No page target');
}
const ws = new WebSocket(tab.webSocketDebuggerUrl);
let nextId = 1;
const pending = new Map();
ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  if (message.id && pending.has(message.id)) {
    pending.get(message.id)(message);
    pending.delete(message.id);
  }
};
await new Promise((resolve, reject) => { ws.onopen = resolve; ws.onerror = reject; });
function command(method, params = {}) {
  return new Promise((resolve) => {
    const id = nextId++;
    pending.set(id, resolve);
    ws.send(JSON.stringify({ id, method, params }));
  });
}
async function evaluate(expression) {
  const result = await command('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
  return result.result?.result?.value;
}
await command('Page.navigate', { url });
await new Promise((resolve) => setTimeout(resolve, 2500));
const before = await evaluate(`(() => { const b=document.querySelector('.site-header__dropdown-trigger'); const m=document.querySelector('.site-header__dropdown-menu'); return {url:location.href, ready:document.readyState, button:!!b, expanded:b?.getAttribute('aria-expanded'), menu:!!m, menuRect:m?.getBoundingClientRect().toJSON(), headerRect:document.querySelector('.site-header')?.getBoundingClientRect().toJSON(), computed:m?{display:getComputedStyle(m).display,visibility:getComputedStyle(m).visibility,zIndex:getComputedStyle(m).zIndex,position:getComputedStyle(m).position,opacity:getComputedStyle(m).opacity}:null}; })()`);
const click = await evaluate(`(() => { const b=document.querySelector('.site-header__dropdown-trigger'); b?.click(); return !!b; })()`);
await new Promise((resolve) => setTimeout(resolve, 400));
const after = await evaluate(`(() => { const b=document.querySelector('.site-header__dropdown-trigger'); const m=document.querySelector('.site-header__dropdown-menu'); const r=m?.getBoundingClientRect(); const links=[...document.querySelectorAll('.site-header__dropdown-menu a')].map(a=>({text:a.textContent.trim(),href:a.href,rect:a.getBoundingClientRect().toJSON(),visible:!!(a.offsetWidth||a.offsetHeight||a.getClientRects().length)})); const hit=r?document.elementFromPoint(Math.max(1,r.left+10),Math.max(1,r.top+10))?.outerHTML?.slice(0,160):null; return {url:location.href,expanded:b?.getAttribute('aria-expanded'),menu:!!m,rect:r?.toJSON(),links,hit,scrollWidth:document.documentElement.scrollWidth,innerWidth:innerWidth}; })()`);
const shot = await command('Page.captureScreenshot', { format: 'png' });
if (shot.result?.data) fs.writeFileSync('output/dropdown-production.png', Buffer.from(shot.result.data, 'base64'));
console.log(JSON.stringify({before, after}, null, 2));
ws.close();

