/**
 * AI Chat Playground — Landing page demo
 *
 * - Animated chat-UI walkthrough (video-like demo of the operation screen)
 * - Animated SVG "model hub" showing the chat hub connected to multiple models
 *
 * SVG geometry is NOT hand-written: every coordinate / bezier path is computed
 * by code and validated by `scripts/verify_landing_geometry.js` (node) before
 * it is drawn on the page. See MODEL_HUB_CONFIG below.
 *
 * Usage on the page:
 *   <div id="landing-demo-chat"></div>   (chat walkthrough)
 *   <div id="landing-demo-hub"></div>    (animated SVG model hub)
 *   <div id="landing-demo-hub-status"></div>  (e.g. "Gemini 3.6 Flash に接続中")
 */
(function (root, factory) {
    if (typeof module !== 'undefined' && module.exports) {
        module.exports = factory();
    } else if (root) {
        root.LandingDemo = factory();
    }
})(typeof window !== 'undefined' ? window : null, function () {
    'use strict';

    /* ─────────────────────────────────────────────────────────────
     * 1. Model registry (shared between hub SVG and chat demo)
     * ───────────────────────────────────────────────────────────── */
    var MODELS = [
        { key: 'gemini', name: 'Gemini 3.6 Flash', short: 'Gemini', color: '#0dd4bf', glyph: '\uF005', faIcon: 'fa-star' },
        { key: 'gpt', name: 'GPT-5.6 Sol', short: 'GPT-5.6', color: '#34d399', glyph: '\uF5DC', faIcon: 'fa-brain' },
        { key: 'grok', name: 'Grok 4.3', short: 'Grok', color: '#e2e8f0', glyph: '\uF135', faIcon: 'fa-rocket' },
        { key: 'claude', name: 'Claude Opus 4.6', short: 'Claude', color: '#f59e0b', glyph: '\uF06D', faIcon: 'fa-fire' },
        { key: 'deepseek', name: 'DeepSeek V4', short: 'DeepSeek', color: '#22d3ee', glyph: '\uF0E7', faIcon: 'fa-bolt' },
        { key: 'kimi', name: 'Kimi K3', short: 'Kimi', color: '#a78bfa', glyph: '\uF3A5', faIcon: 'fa-gem' }
    ];

    /* Geometry configuration for the SVG model hub (680 x 420 viewBox).
     * Verified by scripts/verify_landing_geometry.js — do not edit numbers
     * without re-running that script. */
    var MODEL_HUB_CONFIG = {
        width: 680,
        height: 420,
        hubX: 340,
        hubY: 210,
        hubR: 36,
        rx: 230,
        ry: 118,
        nodeW: 150,
        nodeH: 48,
        startDeg: -90,
        models: MODELS
    };

    /* ─────────────────────────────────────────────────────────────
     * 2. Pure geometry (shared with the node verification script)
     * ───────────────────────────────────────────────────────────── */
    function computeModelHubGeometry(cfg) {
        var N = cfg.models.length;
        var halfW = cfg.nodeW / 2;
        var halfH = cfg.nodeH / 2;
        var nodes = [];
        for (var i = 0; i < N; i++) {
            var deg = cfg.startDeg + (i * 360) / N;
            var rad = (deg * Math.PI) / 180;
            var cx = cfg.hubX + cfg.rx * Math.cos(rad);
            var cy = cfg.hubY + cfg.ry * Math.sin(rad);
            var dx = cx - cfg.hubX;
            var dy = cy - cfg.hubY;
            var len = Math.hypot(dx, dy);
            var ux = dx / len;
            var uy = dy / len;
            /* Distance from node center to its rect border along the hub direction */
            var tEdge = Math.min(halfW / Math.abs(ux || 1e-9), halfH / Math.abs(uy || 1e-9));
            var pad = 6;
            var ex = cx - ux * (tEdge + pad);
            var ey = cy - uy * (tEdge + pad);
            var sx = cfg.hubX + ux * (cfg.hubR + 12);
            var sy = cfg.hubY + uy * (cfg.hubR + 12);
            var dist = len - cfg.hubR - tEdge - pad;
            var base = Math.max(18, dist * 0.28);
            var curve = (i % 2 === 0 ? 1 : -1) * 16;
            var px = -uy;
            var py = ux;
            var c1x = sx + ux * base + px * curve;
            var c1y = sy + uy * base + py * curve;
            var c2x = ex - ux * base + px * curve;
            var c2y = ey - uy * base + py * curve;
            var d = 'M ' + sx.toFixed(2) + ' ' + sy.toFixed(2) +
                ' C ' + c1x.toFixed(2) + ' ' + c1y.toFixed(2) +
                ', ' + c2x.toFixed(2) + ' ' + c2y.toFixed(2) +
                ', ' + ex.toFixed(2) + ' ' + ey.toFixed(2);
            nodes.push({
                i: i,
                key: cfg.models[i].key,
                name: cfg.models[i].name,
                short: cfg.models[i].short,
                color: cfg.models[i].color,
                glyph: cfg.models[i].glyph,
                cx: cx,
                cy: cy,
                rect: { x: cx - halfW, y: cy - halfH, w: cfg.nodeW, h: cfg.nodeH },
                start: { x: sx, y: sy },
                end: { x: ex, y: ey },
                c1: { x: c1x, y: c1y },
                c2: { x: c2x, y: c2y },
                d: d
            });
        }
        return { width: cfg.width, height: cfg.height, hub: { x: cfg.hubX, y: cfg.hubY, r: cfg.hubR }, nodes: nodes };
    }

    function sampleCubicBezier(p0, p1, p2, p3, steps) {
        var out = [];
        for (var i = 0; i <= steps; i++) {
            var t = i / steps;
            var mt = 1 - t;
            out.push({
                x: mt * mt * mt * p0.x + 3 * mt * mt * t * p1.x + 3 * mt * t * t * p2.x + t * t * t * p3.x,
                y: mt * mt * mt * p0.y + 3 * mt * mt * t * p1.y + 3 * mt * t * t * p2.y + t * t * t * p3.y
            });
        }
        return out;
    }

    function validateModelHubGeometry(g) {
        var errors = [];
        var keys = g.nodes.map(function (n) { return n.key; });
        if (new Set(keys).size !== keys.length) errors.push('duplicate model key');
        for (var k = 0; k < g.nodes.length; k++) {
            var n = g.nodes[k];
            if (!Number.isFinite(n.cx) || !Number.isFinite(n.cy)) errors.push(n.key + ': non-finite center');
            var r = n.rect;
            if (!Number.isFinite(r.x) || !Number.isFinite(r.y)) errors.push(n.key + ': non-finite rect');
            if (r.x < 0 || r.y < 0 || r.x + r.w > g.width || r.y + r.h > g.height) {
                errors.push(n.key + ': rect out of bounds ' + [r.x, r.y, r.x + r.w, r.y + r.h].join(','));
            }
            var samples = sampleCubicBezier(n.start, n.c1, n.c2, n.end, 24);
            for (var s = 0; s < samples.length; s++) {
                var p = samples[s];
                if (!Number.isFinite(p.x) || !Number.isFinite(p.y)) errors.push(n.key + ': non-finite bezier point');
                if (p.x < 0 || p.x > g.width || p.y < 0 || p.y > g.height) {
                    errors.push(n.key + ': bezier out of bounds ' + p.x.toFixed(1) + ',' + p.y.toFixed(1));
                }
            }
        }
        for (var i = 0; i < g.nodes.length; i++) {
            for (var j = i + 1; j < g.nodes.length; j++) {
                var a = g.nodes[i];
                var b = g.nodes[j];
                var ox = Math.max(0, Math.min(a.rect.x + a.rect.w, b.rect.x + b.rect.w) - Math.max(a.rect.x, b.rect.x));
                var oy = Math.max(0, Math.min(a.rect.y + a.rect.h, b.rect.y + b.rect.h) - Math.max(a.rect.y, b.rect.y));
                if (ox > 0 && oy > 0) errors.push('nodes overlap: ' + a.key + ' & ' + b.key);
            }
        }
        if (g.hub.x - g.hub.r < 0 || g.hub.y - g.hub.r < 0 || g.hub.x + g.hub.r > g.width || g.hub.y + g.hub.r > g.height) {
            errors.push('hub out of bounds');
        }
        return errors;
    }

    /* ─────────────────────────────────────────────────────────────
     * 3. Browser-side rendering (no-op under node)
     * ───────────────────────────────────────────────────────────── */
    var NS = 'http://www.w3.org/2000/svg';
    var REDUCED = typeof window !== 'undefined' &&
        window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    var DEFAULTS = { reduced: REDUCED };

    function svgEl(name, attrs) {
        var el = document.createElementNS(NS, name);
        if (attrs) {
            Object.keys(attrs).forEach(function (k) {
                el.setAttribute(k, attrs[k]);
            });
        }
        return el;
    }

    /* ── SVG model hub ── */
    function buildModelHub() {
        var g = computeModelHubGeometry(MODEL_HUB_CONFIG);
        var svg = svgEl('svg', {
            viewBox: '0 0 ' + g.width + ' ' + g.height,
            role: 'img',
            'aria-label': '複数のAIモデルとチャットを接続する図'
        });
        svg.classList.add('hub-svg');

        var defs = svgEl('defs');
        var grad = svgEl('linearGradient', { id: 'ld-hub-grad', x1: '0', y1: '0', x2: '1', y2: '1' });
        grad.appendChild(svgEl('stop', { offset: '0%', 'stop-color': '#0dd4bf' }));
        grad.appendChild(svgEl('stop', { offset: '100%', 'stop-color': '#34d399' }));
        defs.appendChild(grad);
        svg.appendChild(defs);

        /* Connection routes (drawn first, under everything) */
        g.nodes.forEach(function (n) {
            var grp = svgEl('g', { class: 'hub-node', 'data-key': n.key, style: '--nc:' + n.color });
            var route = svgEl('path', { class: 'hub-route', d: n.d, fill: 'none' });
            grp.appendChild(route);
            /* Data packets travelling along the computed bezier */
            if (!DEFAULTS.reduced) {
                [0, 1.6].forEach(function (delay) {
                    var dot = svgEl('circle', { class: 'hub-packet', r: 3.2, fill: n.color });
                    var motion = svgEl('animateMotion', {
                        dur: '3.4s',
                        begin: delay + 's',
                        repeatCount: 'indefinite',
                        path: n.d
                    });
                    dot.appendChild(motion);
                    grp.appendChild(dot);
                });
            }
            svg.appendChild(grp);
        });

        /* Hub center */
        var hubG = svgEl('g', { class: 'hub-core' });
        var glow = svgEl('circle', { cx: g.hub.x, cy: g.hub.y, r: g.hub.r + 16, class: 'hub-glow' });
        var disc = svgEl('circle', { cx: g.hub.x, cy: g.hub.y, r: g.hub.r, class: 'hub-disc', fill: 'url(#ld-hub-grad)' });
        var icon = svgEl('text', { class: 'hub-icon', x: g.hub.x, y: g.hub.y + 7, 'text-anchor': 'middle' });
        icon.textContent = '\uF075';
        hubG.appendChild(glow);
        hubG.appendChild(disc);
        hubG.appendChild(icon);
        svg.appendChild(hubG);

        /* Model nodes */
        g.nodes.forEach(function (n) {
            var grp = svgEl('g', { class: 'hub-node-card', 'data-key': n.key, style: '--nc:' + n.color });
            var card = svgEl('rect', { class: 'hub-card', x: n.rect.x, y: n.rect.y, width: n.rect.w, height: n.rect.h, rx: 12 });
            var ring = svgEl('circle', { class: 'hub-ring', cx: n.cx, cy: n.cy, r: n.rect.w / 2 + 9, fill: 'none' });
            var dot = svgEl('circle', { class: 'hub-model-dot', cx: n.rect.x + 18, cy: n.cy, r: 5, fill: n.color });
            var icon = svgEl('text', { class: 'hub-model-glyph', x: n.rect.x + 33, y: n.cy + 5, 'text-anchor': 'middle', fill: n.color });
            icon.textContent = n.glyph;
            var label = svgEl('text', { class: 'hub-model-label', x: n.rect.x + 42, y: n.cy + 5 });
            label.textContent = n.name;
            var status = svgEl('circle', { class: 'hub-status-dot', cx: n.rect.x + n.rect.w - 15, cy: n.cy, r: 2.5, fill: n.color });
            grp.appendChild(card);
            grp.appendChild(ring);
            grp.appendChild(dot);
            grp.appendChild(icon);
            grp.appendChild(label);
            grp.appendChild(status);
            svg.appendChild(grp);
        });

        return { svg: svg, graph: g, activate: function (key) { activateHubNode(svg, key); } };
    }

    function activateHubNode(svg, key) {
        if (!svg) return;
        var nodes = svg.querySelectorAll('.hub-node-card, .hub-node');
        for (var i = 0; i < nodes.length; i++) {
            nodes[i].classList.toggle('active', nodes[i].getAttribute('data-key') === key);
        }
    }

    /* ── Chat walkthrough demo ── */
    var CHAT_SCRIPT = [
        { key: 'gemini', statuses: ['APIに送信中...', '回答を生成中...'], thinking: true },
        { key: 'grok', statuses: ['接続完了。モデル応答を待機中...'], thinking: true }
    ];

    function modelInfo(key) {
        for (var i = 0; i < MODELS.length; i++) if (MODELS[i].key === key) return MODELS[i];
        return MODELS[0];
    }

    function buildChatDemo(root) {
        var systemVersion = (typeof window !== 'undefined' && window.LANDING_SYSTEM_VERSION) || 'V4.8.745';
        var chrome = document.createElement('div');
        chrome.className = 'chat-demo-chrome';
        chrome.innerHTML =
            '<div class="chat-demo-title">' +
            '<span class="chat-demo-dots"><i></i><i></i><i></i></span>' +
            '<span class="chat-demo-title-text">AI Chat Playground</span>' +
            '</div>' +
            '<div class="chat-demo-version">' + systemVersion + '</div>';

        var body = document.createElement('div');
        body.className = 'chat-demo-body';
        var composer = document.createElement('div');
        composer.className = 'chat-demo-composer';
        composer.innerHTML =
            '<div class="chat-demo-select" id="ld-select">' +
            '<i class="chat-demo-select-icon"></i>' +
            '<span class="chat-demo-select-name" id="ld-select-name"></span>' +
            '<i class="fas fa-chevron-down chat-demo-select-caret"></i>' +
            '</div>' +
            '<input class="chat-demo-input" id="ld-input" type="text" autocomplete="off" readonly placeholder="メッセージを入力...">' +
            '<button class="chat-demo-send" id="ld-send" type="button" aria-label="送信"><i class="fas fa-paper-plane"></i></button>';

        root.appendChild(chrome);
        root.appendChild(body);
        root.appendChild(composer);

        var setModel = function (key) {
            var m = modelInfo(key);
            var icon = composer.querySelector('.chat-demo-select-icon');
            icon.className = 'chat-demo-select-icon fas ' + m.faIcon;
            icon.style.color = m.color;
            composer.querySelector('#ld-select-name').textContent = m.name;
            composer.querySelector('#ld-select').style.setProperty('--sc', m.color);
            return m;
        };

        return { root: root, body: body, setModel: setModel };
    }

    function chatScroll(body) {
        body.scrollTop = body.scrollHeight;
    }

    function appendUserBubble(body, text) {
        var wrap = document.createElement('div');
        wrap.className = 'ld-row ld-user';
        var bubble = document.createElement('div');
        bubble.className = 'ld-bubble ld-user-bubble';
        bubble.textContent = text;
        wrap.appendChild(bubble);
        body.appendChild(wrap);
        chatScroll(body);
        return { wrap: wrap };
    }

    function appendPending(body, status) {
        var wrap = document.createElement('div');
        wrap.className = 'ld-row ld-ai';
        var bubble = document.createElement('div');
        bubble.className = 'ld-bubble ld-ai-bubble';
        bubble.innerHTML =
            '<div class="ld-model-badge" style="--mc:#8e9aaf"><span class="ld-model-badge-dot"></span><span>生成中</span></div>' +
            '<div class="ld-skeleton">' +
            '<div class="ld-skeleton-line" style="width:88%"></div>' +
            '<div class="ld-skeleton-line" style="width:72%"></div>' +
            '<div class="ld-skeleton-line" style="width:94%"></div>' +
            '<div class="ld-skeleton-line" style="width:58%"></div>' +
            '</div>' +
            '<div class="ld-skeleton-status">' + status + '</div>';
        wrap.appendChild(bubble);
        body.appendChild(wrap);
        chatScroll(body);
        return { wrap: wrap, setStatus: function (s) {
            var el = bubble.querySelector('.ld-skeleton-status');
            if (el) el.textContent = s;
        } };
    }

    function appendThinking(body, m, label) {
        var wrap = document.createElement('div');
        wrap.className = 'ld-row ld-ai';
        var pill = document.createElement('div');
        pill.className = 'ld-thinking';
        pill.style.setProperty('--mc', m.color);
        pill.innerHTML =
            '<span class="ld-thinking-label">' + label + '</span>' +
            '<span class="ld-dots"><i></i><i></i><i></i></span>';
        wrap.appendChild(pill);
        body.appendChild(wrap);
        chatScroll(body);
        return { wrap: wrap };
    }

    function appendAIBubble(body, m, blocks, footer, done) {
        var wrap = document.createElement('div');
        wrap.className = 'ld-row ld-ai';
        var bubble = document.createElement('div');
        bubble.className = 'ld-bubble ld-ai-bubble ld-streaming';
        var badge = document.createElement('div');
        badge.className = 'ld-model-badge';
        badge.style.setProperty('--mc', m.color);
        badge.innerHTML =
            '<i class="' + ('fas ' + m.faIcon) + '"></i>' +
            '<span>' + m.name + '</span>' +
            (m.key === 'gemini' ? '<span class="ld-model-tag">Thinking</span>' : '');
        var content = document.createElement('div');
        content.className = 'ld-ai-content';
        bubble.appendChild(badge);
        bubble.appendChild(content);
        wrap.appendChild(bubble);
        body.appendChild(wrap);

        /* Flatten blocks into individual reveal units.
         * A block that is an array of <li> items is rendered into one shared
         * <ol>/<ul> so numbering stays continuous while items stream in. */
        var units = [];
        var listTag = null;
        var currentList = null;
        for (var bi = 0; bi < blocks.length; bi++) {
            if (Array.isArray(blocks[bi])) {
                listTag = /^<ul/i.test(blocks[bi][0]) ? 'ul' : 'ol';
                for (var li = 0; li < blocks[bi].length; li++) {
                    units.push({ list: true, html: blocks[bi][li] });
                }
            } else {
                units.push({ list: false, html: blocks[bi] });
            }
        }

        var idx = 0;
        var cursor = document.createElement('span');
        cursor.className = 'ld-cursor';
        var footerEl = null;

        function revealUnit(unit) {
            if (unit.list) {
                if (!currentList) {
                    currentList = document.createElement(listTag);
                    content.appendChild(currentList);
                }
                var li = document.createElement('div');
                li.className = 'ld-block';
                li.innerHTML = unit.html;
                currentList.appendChild(li);
            } else {
                var b = document.createElement('div');
                b.className = 'ld-block';
                b.innerHTML = unit.html;
                content.appendChild(b);
            }
            if (cursor.parentNode) cursor.parentNode.removeChild(cursor);
            content.appendChild(cursor);
            chatScroll(body);
        }

        function next() {
            if (idx < units.length) {
                revealUnit(units[idx]);
                idx++;
                window.setTimeout(next, 420);
            } else {
                if (cursor.parentNode) cursor.parentNode.removeChild(cursor);
                content.classList.remove('ld-streaming-active');
                bubble.classList.remove('ld-streaming');
                if (footer) {
                    footerEl = document.createElement('div');
                    footerEl.className = 'ld-msg-footer';
                    footerEl.textContent = footer;
                    bubble.appendChild(footerEl);
                }
                chatScroll(body);
                done();
            }
        }
        content.classList.add('ld-streaming-active');
        next();
        return { wrap: wrap };
    }

    function typeInto(input, text, per, done) {
        var i = 0;
        function tick() {
            if (i <= text.length) {
                input.value = text.slice(0, i);
                i++;
                window.setTimeout(tick, per);
            } else {
                done();
            }
        }
        tick();
    }

    function runChatDemo(api, root) {
        var body = api.body;
        var reduced = DEFAULTS.reduced;

        var stage = function (steps, onDone) {
            var timers = [];
            for (var i = 0; i < steps.length; i++) {
                (function (s) {
                    timers.push(window.setTimeout(s.fn, s.t));
                })(steps[i]);
            }
            timers.push(window.setTimeout(onDone, steps[steps.length - 1].t + 100));
            return function cancel() { timers.forEach(function (t) { window.clearTimeout(t); }); };
        };

        function sequence() {
            body.textContent = '';
            var current = null;

            var steps = [];
            var ua = '2026年の日本で注目のテックトレンドを、理由付きで5つ教えて';
            var gm = modelInfo('gemini');
            var gk = modelInfo('grok');
            var t = 0;

            steps.push({ t: 0, fn: function () { api.setModel('gemini'); } });
            t += 450;
            steps.push({ t: t, fn: function () { current = appendUserBubble(body, ua); } });
            t += 1300;
            steps.push({ t: t, fn: function () { current = appendPending(body, 'APIに送信中...'); } });
            t += 1200;
            steps.push({ t: t, fn: function () { if (current && current.setStatus) current.setStatus('回答を生成中...'); } });
            t += 900;
            steps.push({ t: t, fn: function () { if (current && current.wrap) current.wrap.remove(); current = appendThinking(body, gm, '思考中'); } });
            t += 1000;
            steps.push({ t: t, fn: function () {
                if (current && current.wrap) current.wrap.remove();
                var blocks = [
                    '<h4>2026年の注目テックトレンド（日本）</h4>',
                    [
                        '<li><strong>AIエージェントの実務浸透</strong> — 経理・カスタマーサポートなどの定型業務でエージェント運用が標準化。AI法対応のガバナンスツールも拡大。</li>',
                        '<li><strong>モバイル型データセンター</strong> — 電力制約への対策として、遊休地を活用したコンパクトDCの建設計画が全国で進行。</li>',
                        '<li><strong>次世代半導体パッケージング</strong> — 2nm世代で日本勢の後工程受託が拡大し、供給網の再編が加速。</li>',
                        '<li><strong>AIによる個別最適医療</strong> — 自治体と病院の連携で予防医療のAI診断が拡大し、診療データ連携基盤が整備。</li>',
                        '<li><strong>クリエイティブ生成の民主化</strong> — 映像・3D・音声の生成コストが大幅低下し、中小企業のプロモーション制作が変革。</li>'
                    ]
                ];
                appendAIBubble(body, gm, blocks, 'In 2,310 / Out 1,845', function () {});
            } });

            /* Model switch → Grok 4.3 */
            t += 4700;
            steps.push({ t: t, fn: function () { api.setModel('grok'); } });
            var ub = 'じゃあ2026年のAIエージェント事情は、日本の導入状況を踏まえてまとめて';
            var typed = false;
            t += 600;
            steps.push({ t: t, fn: function () {
                var input = root.querySelector('#ld-input');
                if (reduced) { input.value = ub; typed = true; }
                else typeInto(input, ub, 50, function () { typed = true; });
            } });
            t += 2400;
            steps.push({ t: t, fn: function () { if (!typed) return; var input = root.querySelector('#ld-input'); input.value = ''; current = appendUserBubble(body, ub); } });
            t += 800;
            steps.push({ t: t, fn: function () { current = appendPending(body, '接続完了。モデル応答を待機中...'); } });
            t += 1600;
            steps.push({ t: t, fn: function () { if (current && current.wrap) current.wrap.remove(); current = appendThinking(body, gk, '思考を整理中'); } });
            t += 1000;
            steps.push({ t: t, fn: function () {
                if (current && current.wrap) current.wrap.remove();
                var blocks = [
                    '<h4>2026年のAIエージェント（概観）</h4>',
                    [
                        '<li><strong>MCPなどの標準プロトコル</strong>が普及し、異なるベンダーのエージェントが相互運用できる時代へ。</li>',
                        '<li>エージェント同士がタスクを委譲する<strong>オーケストレーション</strong>が一般化。企業は人間の承認ワークフローと統合。</li>',
                        '<li>セキュリティ面では、エージェント専用の監視体制<strong>Agent SOC</strong>という新職種が登場。</li>',
                        '<li>個人向けには、カレンダー・メール・購買を横断する<strong>パーソナルエージェント</strong>の定額サービスが拡大。</li>'
                    ]
                ];
                appendAIBubble(body, gk, blocks, 'In 1,020 / Out 3,402 (Thought 1,104)', function () {});
            } });

            t += 4600;
            steps.push({ t: t, fn: function () {
                body.classList.add('ld-demo-dimming');
                window.setTimeout(function () {
                    body.classList.remove('ld-demo-dimming');
                    sequence();
                }, 900);
            } });

            stage(steps, function () {});
        }

        sequence();
    }

    /* ── Scroll-reveal (IntersectionObserver) ──
     * Elements marked with .ld-reveal / .ld-reveal-fade fade+slide in when
     * they enter the viewport. Safe under Rocket Loader (self-booted). */
    function initReveal() {
        if (typeof document === 'undefined') return;
        var nodes = document.querySelectorAll('.ld-reveal, .ld-reveal-fade');
        if (!nodes.length) return;

        /* Reduced motion: show everything immediately, no transition. */
        if (DEFAULTS.reduced || typeof IntersectionObserver === 'undefined') {
            for (var i = 0; i < nodes.length; i++) {
                nodes[i].classList.add('ld-reveal-visible');
            }
            return;
        }

        var io = new IntersectionObserver(function (entries) {
            for (var e = 0; e < entries.length; e++) {
                var entry = entries[e];
                if (entry.isIntersecting) {
                    entry.target.classList.add('ld-reveal-visible');
                    io.unobserve(entry.target);
                }
            }
        }, { root: null, rootMargin: '0px 0px -8% 0px', threshold: 0.12 });

        for (var n = 0; n < nodes.length; n++) {
            /* Already in (or near) the first viewport: reveal without waiting
             * a second paint so hero content does not sit invisible. */
            var rect = nodes[n].getBoundingClientRect();
            var vh = window.innerHeight || document.documentElement.clientHeight || 800;
            if (rect.top < vh * 0.92 && rect.bottom > 0) {
                nodes[n].classList.add('ld-reveal-visible');
            } else {
                io.observe(nodes[n]);
            }
        }
    }

    /* ── Public API ── */
    function initDemo(options) {
        options = options || {};
        DEFAULTS.reduced = options.reduced === true || REDUCED;
        if (typeof document === 'undefined') return;

        var hubRoot = options.hub || document.getElementById('landing-demo-hub');
        var chatRoot = options.chat || document.getElementById('landing-demo-chat');
        var statusEl = options.status || document.getElementById('landing-demo-hub-status');

        /* Idempotency guard: never build the same root twice. */
        if (hubRoot && !hubRoot.getAttribute('data-ld-built')) {
            hubRoot.setAttribute('data-ld-built', '1');
            var hub = buildModelHub();
            hubRoot.appendChild(hub.svg);
            var idx = 0;
            hub.activate(MODELS[0].key);
            if (statusEl) statusEl.textContent = MODELS[0].name + ' に接続中';
            if (!DEFAULTS.reduced) {
                window.setInterval(function () {
                    idx = (idx + 1) % MODELS.length;
                    hub.activate(MODELS[idx].key);
                    if (statusEl) {
                        statusEl.textContent = MODELS[idx].name + ' に接続中';
                        statusEl.style.color = MODELS[idx].color;
                    }
                }, 3600);
            }
        }

        if (chatRoot && !chatRoot.getAttribute('data-ld-built')) {
            chatRoot.setAttribute('data-ld-built', '1');
            var api = buildChatDemo(chatRoot);
            runChatDemo(api, chatRoot);
        }

        /* Reveal runs once per page load (guard via data attr on <html>).
         * documentElement may be absent under the node DOM shim used by tests. */
        var rootEl = document.documentElement;
        if (rootEl && !rootEl.getAttribute('data-ld-reveal')) {
            rootEl.setAttribute('data-ld-reveal', '1');
            initReveal();
        }
    }

    /* ── Self-boot ──
     * Do not rely on a separate inline <script> + DOMContentLoaded listener:
     * Cloudflare Rocket Loader rewrites inline/external scripts to a custom
     * type and executes them after the document has already loaded, so a
     * DOMContentLoaded listener registered from that script never fires.
     * Instead, boot here based on document.readyState. */
    function bootDemo() {
        initDemo();
    }
    if (typeof document !== 'undefined' && typeof window !== 'undefined') {
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', bootDemo);
        } else {
            bootDemo();
        }
    }

    return {
        MODELS: MODELS,
        MODEL_HUB_CONFIG: MODEL_HUB_CONFIG,
        computeModelHubGeometry: computeModelHubGeometry,
        sampleCubicBezier: sampleCubicBezier,
        validateModelHubGeometry: validateModelHubGeometry,
        initDemo: initDemo,
        initReveal: initReveal
    };
});
