#!/usr/bin/env node
/* Smoke-test landing_demo.js: synchronous build + full chat timeline via fake clock.
 * Usage: node scripts/test_landing_demo_dom.js
 */
const { createDom, makeClock } = require('./_dom_shim.js');

const clock = makeClock();
const dom = createDom(clock);

const hubRoot = dom.document.createElement('div'); hubRoot.setAttribute('id', 'landing-demo-hub');
const chatRoot = dom.document.createElement('div'); chatRoot.setAttribute('id', 'landing-demo-chat');
const statusEl = dom.document.createElement('span'); statusEl.setAttribute('id', 'landing-demo-hub-status');
dom.root.appendChild(hubRoot); dom.root.appendChild(chatRoot); dom.root.appendChild(statusEl);

const LD = require('../static/js/landing_demo.js');

/* 1. Geometry must pass validation before it is drawn */
const g = LD.computeModelHubGeometry(LD.MODEL_HUB_CONFIG);
const errs = LD.validateModelHubGeometry(g);
if (errs.length) { console.error('GEOM FAIL'); errs.forEach((e) => console.error('  -', e)); process.exit(1); }

/* 2. Build the demo (not reduced) */
LD.initDemo({ hub: hubRoot, chat: chatRoot, status: statusEl, reduced: false });

const cards = hubRoot.querySelectorAll('.hub-node-card');
const routes = hubRoot.querySelectorAll('.hub-route');
if (cards.length !== 6) { console.error('expected 6 hub cards, got', cards.length); process.exit(1); }
if (routes.length !== 6) { console.error('expected 6 routes, got', routes.length); process.exit(1); }
const PATH_RE = /^M\s+([-\d.]+)\s+([-\d.]+)\s+C\s+([-\d.]+)\s+([-\d.]+),\s*([-\d.]+)\s+([-\d.]+),\s*([-\d.]+)\s+([-\d.]+)$/;
for (const r of routes) {
    const m = PATH_RE.exec(r.getAttribute('d'));
    if (!m || !m.slice(1).map(Number).every(Number.isFinite)) { console.error('bad path:', r.getAttribute('d')); process.exit(1); }
}
/* packets present (not reduced) */
const packets = hubRoot.querySelectorAll('.hub-packet');
if (packets.length !== 12) { console.error('expected 12 packets, got', packets.length); process.exit(1); }

/* 3. Run the full chat timeline (~21s) and inspect the final state */
clock.advanceTo(21300);

const body = chatRoot.querySelector('.chat-demo-body');
const rows = body.querySelectorAll('.ld-row');
const userBubbles = body.querySelectorAll('.ld-user .ld-bubble');
const aiBubbles = body.querySelectorAll('.ld-ai-bubble');
const footers = body.querySelectorAll('.ld-msg-footer');
const badges = body.querySelectorAll('.ld-model-badge');

if (userBubbles.length !== 2) { console.error('expected 2 user bubbles, got', userBubbles.length); process.exit(1); }
if (aiBubbles.length !== 2) { console.error('expected 2 AI bubbles, got', aiBubbles.length); process.exit(1); }
if (footers.length !== 2) { console.error('expected 2 footers, got', footers.length); process.exit(1); }
if (badges.length !== 2) { console.error('expected 2 model badges, got', badges.length); process.exit(1); }
const input = chatRoot.querySelector('#ld-input');
if (input.value !== '') { console.error('input not cleared after send'); process.exit(1); }
const selectName = chatRoot.querySelector('#ld-select-name');
if (selectName.textContent !== 'Grok 4.3') { console.error('model select should end at Grok 4.3, got', selectName.textContent); process.exit(1); }

/* gemini list = 5 items, grok list = 4 items */
const ols = body.querySelectorAll('.ld-ai-content ol');
const lis = body.querySelectorAll('.ld-ai-content ol li');
if (lis.length !== 9) { console.error('expected 9 list items, got', lis.length); process.exit(1); }

const footersText = footers.map((f) => f.textContent);
if (!footersText.includes('In 2,310 / Out 1,845')) { console.error('footer A missing'); process.exit(1); }
if (!footersText.includes('In 1,020 / Out 3,402 (Thought 1,104)')) { console.error('footer B missing'); process.exit(1); }

/* hub status initial (cycles are no-op intervals in fake clock) */
if (statusEl.textContent !== 'Gemini 3.6 Flash に接続中') { console.error('hub status wrong:', statusEl.textContent); process.exit(1); }

console.log('DOM+TL SMOKE PASS: hub(cards=6 routes=6 packets=12) chat(2 user, 2 AI, 2 footers, 9 list items)');
