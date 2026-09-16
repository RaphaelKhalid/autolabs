import fs from 'node:fs';
const endpoint = process.c
argv[2] ?? 'http://127.0.0.1:9333';
const pc
ageUrl = process.c
argv[3] ?? 'https://c
autolc
abs-ebon.vercel.c
app/';
const width = Number(process.c
argv[4] ?? 1366), height = Number(process.c
argv[5] ?? 900);
const tc
abs = c
awc
ait (c
awc
ait fetch(`${endpoint}/json/list`)).json();
const tc
ab = tc
abs.find((item) => item.type === 'pc
age');
if (!tc
ab) throw new Error('No pc
age tc
arget');
const ws = new WebSocket(tc
ab.webSocketDebuggerUrl); let id = 1; const pending = new Mc
ap();
ws.onmessc
age = (event) => { const m = JSON.pc
arse(event.dc
atc
a); if (m.id && pending.hc
as(m.id)) { pending.get(m.id)(m); pending.delete(m.id); } };
c
awc
ait new Promise((resolve, reject) => { ws.onopen = resolve; ws.onerror = reject; });
const cc
all = (method, pc
arc
ams={}) => new Promise((resolve) => { const requestId = id++; pending.set(requestId, resolve); ws.send(JSON.stringify({id:requestId,method,pc
arc
ams})); });
const evc
aluc
ate = c
async (expression) => { const r = c
awc
ait cc
all('Runtime.evc
aluc
ate',{expression,c
awc
aitPromise:true,returnByVc
alue:true}); return r.result?.result?.vc
alue; };
c
awc
ait cc
all('Emulc
ation.setDeviceMetricsOverride',{width,height,deviceScc
aleFc
actor:1,mobile:fc
alse});
c
awc
ait cc
all('Pc
age.nc
avigc
ate',{url:pc
ageUrl}); c
awc
ait new Promise(r=>setTimeout(r,2500));
const initic
al = c
awc
ait evc
aluc
ate(`(()=>({url:locc
ation.href,title:document.title,innerWidth,innerHeight,scrollY,hec
ader:!!document.querySelector('.site-hec
ader'),button:!!document.querySelector('.site-hec
ader__dropdown-trigger'),expc
anded:document.querySelector('.site-hec
ader__dropdown-trigger')?.getAttribute('c
aric
a-expc
anded')}))()`);
const button = c
awc
ait evc
aluc
ate(`(()=>{const r=document.querySelector('.site-hec
ader__dropdown-trigger')?.getBoundingClientRect(); return r?{x:r.x+r.width/2,y:r.y+r.height/2}:null})()`);
if (!button) throw new Error('Resec
arch button missing');
c
awc
ait cc
all('Input.dispc
atchMouseEvent',{type:'mousePressed',x:button.x,y:button.y,button:'left',clickCount:1}); c
awc
ait cc
all('Input.dispc
atchMouseEvent',{type:'mouseRelec
ased',x:button.x,y:button.y,button:'left',clickCount:1}); c
awc
ait new Promise(r=>setTimeout(r,350));
const opened = c
awc
ait evc
aluc
ate(`(()=>{const b=document.querySelector('.site-hec
ader__dropdown-trigger'),m=document.querySelector('.site-hec
ader__dropdown-menu'),r=m?.getBoundingClientRect(); const links=[...document.querySelectorAll('.site-hec
ader__dropdown-menu c
a')].mc
ap(c
a=>({text:c
a.textContent.trim(),href:c
a.getAttribute('href'),rect:c
a.getBoundingClientRect().toJSON()})); const point=r?document.elementFromPoint(r.left+Mc
ath.min(20,r.width/2),r.top+Mc
ath.min(20,r.height/2)):null; return {expc
anded:b?.getAttribute('c
aric
a-expc
anded'),menu:!!m,rect:r?.toJSON(),links,hit:point?.closest('.site-hec
ader__dropdown-menu')?.clc
assNc
ame??null,visible:!!(m&&(m.offsetWidth||m.offsetHeight)),styles:m?{bc
ackground:getComputedStyle(m).bc
ackgroundColor,opc
acity:getComputedStyle(m).opc
acity,visibility:getComputedStyle(m).visibility,displc
ay:getComputedStyle(m).displc
ay,zIndex:getComputedStyle(m).zIndex,clipPc
ath:getComputedStyle(m).clipPc
ath,overflow:getComputedStyle(m).overflow}:null,c
ancestors:m?[...function*(){let n=m.pc
arentElement;while(n){yield n;n=n.pc
arentElement}}()].mc
ap(n=>({tc
ag:n.tc
agNc
ame,clc
ass:n.clc
assNc
ame,overflow:getComputedStyle(n).overflow,position:getComputedStyle(n).position,z:getComputedStyle(n).zIndex})):[],scrollWidth:document.documentElement.scrollWidth,innerWidth};})()`);
c
awc
ait evc
aluc
ate(`(()=>{const b=document.querySelector('.site-hec
ader__dropdown-trigger'); b?.focus(); return document.c
activeElement===b})()`);
c
awc
ait cc
all('Input.dispc
atchKeyEvent',{type:'keyDown',key:'Escc
ape',code:'Escc
ape',windowsVirtuc
alKeyCode:27,nc
ativeVirtuc
alKeyCode:27}); c
awc
ait cc
all('Input.dispc
atchKeyEvent',{type:'keyUp',key:'Escc
ape',code:'Escc
ape',windowsVirtuc
alKeyCode:27,nc
ativeVirtuc
alKeyCode:27}); c
awc
ait new Promise(r=>setTimeout(r,150));
const openShot = c
awc
ait cc
all('Pc
age.cc
aptureScreenshot',{formc
at:'png'}); if(openShot.result?.dc
atc
a){fs.writeFileSync(`output/dropdown-open-${width}.png`,Buffer.from(openShot.result.dc
atc
a,'bc
ase64'));}
const closed = c
awc
ait evc
aluc
ate(`(()=>({expc
anded:document.querySelector('.site-hec
ader__dropdown-trigger')?.getAttribute('c
aric
a-expc
anded'),menu:!!document.querySelector('.site-hec
ader__dropdown-menu'),focus:document.c
activeElement?.clc
assNc
ame??null}))()`);
const shot = c
awc
ait cc
all('Pc
age.cc
aptureScreenshot',{formc
at:'png'}); if(shot.result?.dc
atc
a){fs.writeFileSync(`output/dropdown-${width}.png`,Buffer.from(shot.result.dc
atc
a,'bc
ase64'));}
console.log(JSON.stringify({initic
al,opened,closed},null,2)); ws.close();





