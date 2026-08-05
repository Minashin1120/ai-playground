/* Shared minimal DOM shim for node smoke tests of static/js/landing_demo.js */

const VOID_TAGS = new Set(['input', 'br', 'img', 'hr', 'meta', 'link']);
const ATTR_RE = /([a-zA-Z_:][-a-zA-Z0-9_:.]*)(?:="([^"]*)")?/g;

function makeEl(tag, ns) {
    const el = {
        tagName: tag,
        ns: ns || null,
        attrs: {},
        _class: '',
        children: [],
        parent: null,
        textContent: '',
        scrollTop: 0,
        scrollHeight: 0,
        value: '',
    };
    el.style = { setProperty: (k, v) => { el.style[k] = v; } };
    el.classList = {
        add: (...c) => { for (const x of c) { if (!el._class.split(/\s+/).includes(x)) el._class = (el._class + ' ' + x).trim(); } },
        remove: (...c) => { const s = el._class.split(/\s+/).filter(Boolean); for (const x of c) { const i = s.indexOf(x); if (i >= 0) s.splice(i, 1); } el._class = s.join(' '); },
        toggle: (c, force) => { const s = el._class.split(/\s+/).filter(Boolean); const has = s.includes(c); const on = force === undefined ? !has : !!force; if (on && !has) s.push(c); if (!on && has) s.splice(s.indexOf(c), 1); el._class = s.join(' '); return on; },
        contains: (c) => el._class.split(/\s+/).includes(c),
    };
    el.setAttribute = (k, v) => { el.attrs[k] = String(v); if (k === 'class') el._class = String(v); };
    el.getAttribute = (k) => (k in el.attrs ? el.attrs[k] : null);
    el.appendChild = (c) => { if (c.parent) c.parent.children = c.parent.children.filter((x) => x !== c); c.parent = el; el.children.push(c); return c; };
    el.remove = () => { if (el.parent) { el.parent.children = el.parent.children.filter((x) => x !== el); el.parent = null; } };
    el.querySelectorAll = (sel) => { const out = []; (function walk(n) { for (const c of n.children) { if (matches(c, sel)) out.push(c); walk(c); } })(el); return out; };
    el.querySelector = (sel) => el.querySelectorAll(sel)[0] || null;
    Object.defineProperty(el, 'className', { get: () => el._class, set: (v) => { el._class = String(v); } });
    Object.defineProperty(el, 'innerHTML', {
        get: () => (el._innerHTML !== undefined ? el._innerHTML : ''),
        set: (html) => { el._innerHTML = String(html); el.children = []; parseHTMLInto(el, String(html)); },
    });
    return el;
}

function parseHTMLInto(container, html) {
    const re = /<(\/?)([a-zA-Z][a-zA-Z0-9]*)((?:\s+[a-zA-Z_:][-a-zA-Z0-9_:.]*(?:="[^"]*")?)*)\s*(\/?)>/g;
    const stack = [container];
    let lastIndex = 0;
    let m;
    while ((m = re.exec(html))) {
        const text = html.slice(lastIndex, m.index);
        if (text) appendText(stack[stack.length - 1], text);
        const isClose = m[1] === '/';
        const tag = m[2];
        const selfClose = m[4] === '/' || VOID_TAGS.has(tag);
        if (isClose) {
            if (stack.length > 1) stack.pop();
        } else {
            const el = makeEl(tag, null);
            ATTR_RE.lastIndex = 0;
            let am;
            while ((am = ATTR_RE.exec(m[3]))) el.setAttribute(am[1], am[2] === undefined ? '' : am[2]);
            stack[stack.length - 1].children.push(el);
            el.parent = stack[stack.length - 1];
            if (!selfClose) stack.push(el);
        }
        lastIndex = re.lastIndex;
    }
    const tail = html.slice(lastIndex);
    if (tail) appendText(stack[stack.length - 1], tail);
}

function appendText(el, text) {
    if (!text) return;
    if (el._textNodes === undefined) el._textNodes = [];
    el._textNodes.push(text);
}

function matchSimple(el, part) {
    if (!el) return false;
    if (part.startsWith('.')) return el.classList.contains(part.slice(1));
    if (part.startsWith('#')) return el.attrs.id === part.slice(1);
    return el.tagName === part.toLowerCase() || el.tagName === part;
}

function matches(el, sel) {
    for (const p of sel.split(',').map((s) => s.trim())) {
        const segs = p.split(/\s+/).filter(Boolean);
        if (segs.length === 1) {
            if (matchSimple(el, segs[0])) return true;
            continue;
        }
        if (!matchSimple(el, segs[segs.length - 1])) continue;
        let ok = true;
        let idx = segs.length - 2;
        let node = el.parent;
        while (idx >= 0) {
            while (node && !matchSimple(node, segs[idx])) node = node.parent;
            if (!node) { ok = false; break; }
            idx--;
            node = node.parent;
        }
        if (ok) return true;
    }
    return false;
}

function createDom(clock) {
    const document = {
        createElement: (t) => makeEl(t, null),
        createElementNS: (ns, t) => makeEl(t, ns),
        getElementById: (id) => {
            const r = document._root.querySelectorAll('#' + id);
            return r[0] || null;
        },
        _root: null,
        addEventListener: () => {},
    };
    const root = makeEl('div', null);
    root.setAttribute('id', 'landing-demo-root');
    document._root = root;

    const window = {
        matchMedia: () => ({ matches: clock.reduced }),
        setTimeout: (cb, delay) => clock.setTimeout(cb, delay),
        clearTimeout: (id) => clock.clearTimeout(id),
        setInterval: () => clock.intervalId++,
        clearInterval: () => {},
    };
    global.document = document;
    global.window = window;
    global.self = window;
    return { document, window, root };
}

/* Fake clock: timers fire at their scheduled time when advanceTo() is called. */
function makeClock() {
    const clock = { now: 0, reduced: false, intervalId: 0 };
    let nextId = 1;
    const timers = new Map();
    clock.setTimeout = (cb, delay) => {
        const id = nextId++;
        timers.set(id, { fireAt: clock.now + (delay || 0), cb, alive: true });
        return id;
    };
    clock.clearTimeout = (id) => { const t = timers.get(id); if (t) t.alive = false; };
    clock.advanceTo = (limit) => {
        let guard = 0;
        for (;;) {
            if (++guard > 20000) throw new Error('advanceTo guard: too many timer hops');
            let next = null;
            for (const t of timers.values()) {
                if (t.alive && (next === null || t.fireAt < next.fireAt)) next = t;
            }
            if (next === null || next.fireAt > limit) break;
            clock.now = next.fireAt;
            next.alive = false;
            next.cb();
        }
        clock.now = limit;
        return clock.now;
    };
    return clock;
}

module.exports = { createDom, makeClock, makeEl, matches };
