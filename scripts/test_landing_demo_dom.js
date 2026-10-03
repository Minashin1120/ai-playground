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

/* 3. Run the full chat timeline and inspect the state just before the loop restarts */
const chatBody = chatRoot.querySelector('#chat-container');
let guard = 0;
for (let t = 100; !chatRoot.querySelector('#chat-stage').classList.contains('ld-demo-dimming'); t += 100) {
    if (++guard > 600) { console.error('demo never reached the end of its timeline'); process.exit(1); }
    clock.advanceTo(t);
}

/* the demo mirrors the real chat screen: sidebar, main column, composer dock */
for (const id of ['#sidebar', '#thread-list', '#chat-container', '#welcome-screen', '#model-selector-btn', '#prompt-input', '#send-btn']) {
    if (!chatRoot.querySelector(id)) { console.error('missing real-UI element', id); process.exit(1); }
}
if (!chatRoot.querySelector('.composer-dock') || !chatRoot.querySelector('.composer-input-shell')) { console.error('composer dock missing'); process.exit(1); }

const messages = chatBody.querySelectorAll('.message-group');
const userBubbles = chatBody.querySelectorAll('.message-bubble.bg-blue-600');
const aiBubbles = chatBody.querySelectorAll('.message-bubble.bg-gray-700');
const footers = chatBody.querySelectorAll('.message-footer-meta');
const thoughts = chatBody.querySelectorAll('.thought-container');
if (messages.length !== 4) { console.error('expected 4 message groups, got', messages.length); process.exit(1); }
if (userBubbles.length !== 2) { console.error('expected 2 user bubbles, got', userBubbles.length); process.exit(1); }
if (aiBubbles.length !== 2) { console.error('expected 2 AI bubbles, got', aiBubbles.length); process.exit(1); }
if (footers.length !== 2) { console.error('expected 2 footers, got', footers.length); process.exit(1); }
if (thoughts.length !== 1) { console.error('expected 1 Thinking Process box (Gemini only), got', thoughts.length); process.exit(1); }
if (chatBody.querySelectorAll('.skeleton-pending').length !== 0) { console.error('pending skeleton left behind'); process.exit(1); }
const input = chatRoot.querySelector('#prompt-input');
if (input.value !== '') { console.error('input not cleared after send'); process.exit(1); }
if (chatRoot.querySelector('#welcome-screen').classList.contains('ld-hidden') === false) { console.error('welcome screen should be hidden after sending'); process.exit(1); }
if (chatRoot.querySelector('#ld-new-thread').classList.contains('ld-hidden')) { console.error('new thread row should be visible in the sidebar'); process.exit(1); }
const selectName = chatRoot.querySelector('#model-selector-text');
if (selectName.textContent !== 'Grok 4.3') { console.error('model selector should end at Grok 4.3, got', selectName.textContent); process.exit(1); }
if (chatRoot.getAttribute('data-model') !== 'grok') { console.error('data-model should end at grok'); process.exit(1); }

/* gemini list = 5 items, grok list = 3 items */
const lis = chatBody.querySelectorAll('.prose ol li');
if (lis.length !== 8) { console.error('expected 8 list items, got', lis.length); process.exit(1); }

const footersText = footers.map((f) => f.innerHTML);
if (!footersText.some((f) => f.includes('In 2310 / Out 1845 (Thought 612)'))) { console.error('footer A missing'); process.exit(1); }
if (!footersText.some((f) => f.includes('In 1020 / Out 3402 (Thought 1104)'))) { console.error('footer B missing'); process.exit(1); }

/* hub status initial (cycles are no-op intervals in fake clock) */
if (statusEl.textContent !== 'Gemini 3.6 Flash に接続中') { console.error('hub status wrong:', statusEl.textContent); process.exit(1); }

console.log('DOM+TL SMOKE PASS: hub(cards=6 routes=6 packets=12) chat(2 user, 2 AI, 2 footers, 8 list items)');
