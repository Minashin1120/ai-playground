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
        { key: 'gemini', name: 'Gemini 3.6 Flash', short: 'Gemini', color: '#0dd4bf', glyph: '\uF005' },
        { key: 'gpt', name: 'GPT-5.6 Sol', short: 'GPT-5.6', color: '#34d399', glyph: '\uF5DC' },
        { key: 'grok', name: 'Grok 4.3', short: 'Grok', color: '#e2e8f0', glyph: '\uF135' },
        { key: 'claude', name: 'Claude Opus 4.6', short: 'Claude', color: '#f59e0b', glyph: '\uF06D' },
        { key: 'deepseek', name: 'DeepSeek V4.1', short: 'DeepSeek', color: '#22d3ee', glyph: '\uF0E7' },
        { key: 'kimi', name: 'Kimi K3', short: 'Kimi', color: '#a78bfa', glyph: '\uF3A5' }
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

    /* ── Chat walkthrough demo ──
     * A miniature of the real chat screen. Markup, class names and ids follow
     * templates/chat/*.html and renderMessage() / renderPendingMessage() in
     * static/js/chat_core_parts, so the shared design tokens and the light theme
     * apply the same way. Keep this in sync when the real chat UI changes. */
    function modelInfo(key) {
        for (var i = 0; i < MODELS.length; i++) if (MODELS[i].key === key) return MODELS[i];
        return MODELS[0];
    }

    function esc(s) {
        return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }

    var DEMO_THREADS = ['Pythonでのデータ分析入門', '週末の京都旅行プラン', '英文メールの添削と言い換え', 'SQLのパフォーマンス改善'];
    var QUICK_START = [
        ['⚡', 'Gemini 3.6 Flash'],
        ['☀️', 'GPT-5.6 Sol'],
        ['🧠', 'Kimi K3'],
        ['⚡', 'DeepSeek V4.1 Flash']
    ];
    var NEW_THREAD_TITLE = '2026年日本のテックトレンド5選';

    function toolButton(icon) {
        return '<button type="button" tabindex="-1" class="sidebar-icon-btn text-gray-400 p-1.5 rounded btn-hover"><i class="' + icon + '"></i></button>';
    }

    function threadRow(title, extraClass) {
        return '<div class="p-2 rounded cursor-pointer text-sm text-gray-300 truncate flex justify-between items-center group' + (extraClass ? ' ' + extraClass : '') + '">' +
            '<div class="flex items-center gap-1 truncate flex-1">' +
            '<button type="button" tabindex="-1" class="text-gray-500 px-1"><i class="fas fa-star text-[10px]"></i></button>' +
            '<span class="truncate">' + esc(title) + '</span></div></div>';
    }

    function chatOption(label) {
        return '<label class="composer-opt flex items-center gap-1 select-none">' +
            '<input type="checkbox" class="accent-blue-500 w-3 h-3"><span>' + label + '</span></label>';
    }

    function chatChip(id, label, cls, accent) {
        return '<label id="' + id + '" class="composer-chip flex items-center gap-1 select-none px-2 py-0.5 rounded-full border ' + cls + '">' +
            '<input type="checkbox" class="' + accent + ' w-3 h-3"><span>' + label + '</span></label>';
    }

    function sidebarHtml(version) {
        return '<div id="sidebar" class="sidebar w-64 bg-gray-800 border-r border-gray-700 flex flex-col shrink-0">' +
            '<div class="sidebar-header p-3 border-b border-gray-700 flex flex-col gap-2.5">' +
            '<div class="min-w-0"><div class="flex items-center gap-2 min-w-0">' +
            '<h2 id="sidebar-chat-title" class="font-bold text-base text-gray-200 truncate tracking-tight">AI Chat</h2></div></div>' +
            '<div class="sidebar-toolbar flex items-center justify-start gap-0.5 flex-wrap">' +
            toolButton('fab fa-github') +
            toolButton('fas fa-history text-xs') +
            toolButton('fas fa-clock-rotate-left text-xs') +
            toolButton('fas fa-tachometer-alt text-xs') +
            toolButton('fas fa-cog text-xs') +
            toolButton('fas fa-folder text-xs') +
            '<button type="button" id="new-chat-btn" tabindex="-1" class="sidebar-icon-btn text-gray-400 p-1.5 rounded bg-blue-600 btn-hover"><i class="fas fa-plus text-xs"></i></button>' +
            toolButton('fas fa-sitemap text-xs') +
            toolButton('fas fa-layer-group text-xs') +
            toolButton('fas fa-file-pdf text-xs') +
            '</div>' +
            '<div class="relative sidebar-search"><input type="search" id="search-box" readonly tabindex="-1" placeholder="チャットを検索..." class="w-full bg-gray-700 rounded-lg px-3 py-1.5 text-sm outline-none">' +
            '<i class="fas fa-search absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 text-xs pointer-events-none"></i></div>' +
            '</div>' +
            '<div class="sidebar-gems px-2.5 py-2"><div class="flex justify-between items-center px-1.5 mb-1.5">' +
            '<span class="text-xs font-bold text-gray-400 uppercase tracking-wider">Gems</span>' +
            '<span class="text-xs text-blue-400"><i class="fas fa-plus"></i> New</span></div></div>' +
            '<div class="sidebar-divider border-t border-gray-700 mx-2 my-0.5"></div>' +
            '<div class="flex-1 overflow-y-auto p-2 space-y-0.5" id="thread-list">' +
            '<div id="ld-new-thread" class="ld-hidden">' + threadRow(NEW_THREAD_TITLE, 'bg-gray-700/60 border-l-2 border-blue-500') + '</div>' +
            DEMO_THREADS.map(function (t) { return threadRow(t, ''); }).join('') +
            '</div>' +
            '<div class="sidebar-footer p-3 border-t border-gray-700 flex flex-col gap-2 text-center text-xs text-gray-500 relative">' +
            '<div id="version-display">' + esc(version) + ' <span class="text-green-500 font-bold">Stable</span></div>' +
            '<div class="sidebar-legal flex justify-center gap-2.5"><span>ヘルプ</span><span class="text-gray-500">·</span><span>利用規約</span><span class="text-gray-500">·</span><span>プライバシー</span></div>' +
            '<div class="w-full block bg-gray-700 py-2 rounded-lg sidebar-logout-btn">ログアウト</div>' +
            '</div></div>';
    }

    function mobileHeaderHtml() {
        return '<header class="main-chrome-header bg-gray-800 border-b border-gray-700 px-3 py-2.5 shadow-md flex justify-between items-center gap-2 shrink-0">' +
            '<span class="text-gray-300 p-2 shrink-0"><i class="fas fa-bars text-xl"></i></span>' +
            '<div class="min-w-0 flex-1"><div class="flex items-center gap-1.5 min-w-0"><span id="mobile-chat-title" class="font-bold text-sm truncate">AI Playground</span></div></div>' +
            '<div class="flex items-center shrink-0"><span class="text-blue-400 font-bold text-xl p-2">+</span>' +
            '<span class="text-gray-400 p-2 rounded"><i class="fas fa-file-pdf text-xs"></i></span></div></header>';
    }

    function welcomeHtml() {
        return '<div id="welcome-screen" class="flex flex-col items-center justify-center text-gray-500 z-10">' +
            '<div class="flex flex-col items-center justify-center w-full">' +
            '<h2 class="text-2xl font-bold flex items-center gap-2.5 tracking-tight"><i class="fas fa-gem"></i> AI Gems &amp; Chat</h2>' +
            '<p class="welcome-subtitle text-sm text-gray-400/90">使いたいモデルを選んで、すぐに会話を始められます</p>' +
            '<div id="welcome-quick-start" class="welcome-grid grid grid-cols-1 gap-2 w-full px-4">' +
            QUICK_START.map(function (q) {
                return '<div class="welcome-btn p-3 rounded text-sm text-left transition btn-hover">' + q[0] + ' ' + esc(q[1]) + '</div>';
            }).join('') +
            '</div></div></div>';
    }

    function composerHtml() {
        return '<div class="composer-dock bg-gray-800 border-t border-gray-700 px-3 pt-2 pb-2.5 shrink-0 z-10">' +
            '<div class="composer-shell max-w-3xl mx-auto space-y-1.5 relative">' +
            '<div id="prompt-controls-row" class="composer-controls flex flex-wrap items-center text-xs text-gray-400 gap-1.5">' +
            '<div id="prompt-primary-controls" class="flex items-center gap-1.5 min-w-0 flex-wrap">' +
            '<button type="button" id="model-selector-btn" tabindex="-1" class="bg-gray-700 border border-gray-600 rounded-full px-2.5 py-1 text-white text-xs transition flex items-center gap-1.5 max-w-[180px] truncate">' +
            '<i class="fas fa-robot text-blue-400"></i><span id="model-selector-text"></span>' +
            '<i class="fas fa-chevron-down text-[10px] text-gray-400 ml-auto"></i></button>' +
            chatChip('canvas-mode-container', 'Canvas', 'border-cyan-500/30 bg-cyan-900/15 text-cyan-200', 'accent-cyan-400') +
            chatChip('coding-mode-container', 'Coding', 'border-emerald-500/30 bg-emerald-900/15 text-emerald-200', 'accent-emerald-400') +
            chatChip('browser-fast-mode-container', '高速', 'border-amber-500/40 bg-amber-900/20 text-amber-200', 'accent-amber-400') +
            '<span id="prompt-controls-toggle-btn" class="bg-gray-700 border border-gray-600 rounded-full px-2 py-0.5 text-[10px] text-gray-200 flex items-center gap-1">' +
            '<span>詳細</span><i class="fas fa-chevron-down text-[10px]"></i></span>' +
            '</div>' +
            '<div id="prompt-details-controls"><div class="prompt-details-inner flex items-center gap-1 flex-wrap min-w-0">' +
            '<div id="standard-chat-controls" class="composer-detail-chips flex gap-1 items-center flex-wrap">' +
            chatOption('Search') +
            '<label class="composer-opt flex items-center gap-1 select-none"><input type="checkbox" class="accent-yellow-500 w-3 h-3" checked><span class="text-yellow-200">Python</span></label>' +
            '<label class="composer-opt flex items-center gap-1 select-none"><input type="checkbox" class="accent-orange-500 w-3 h-3" checked><span class="text-orange-300">File</span></label>' +
            '<div class="composer-opt flex items-center gap-0.5 select-none"><label class="flex items-center gap-1"><input type="checkbox" class="accent-green-500 w-3 h-3"><span class="text-green-300">SysPrompt</span></label>' +
            '<span class="text-gray-500 p-1"><i class="fas fa-cog text-[10px]"></i></span></div>' +
            '<div class="composer-opt composer-opt-group ld-only-gemini flex items-center gap-1.5"><label class="flex items-center gap-1 select-none"><input type="checkbox" class="accent-purple-500 w-3 h-3" checked><span class="text-purple-300">Thinking</span></label>' +
            '<select class="bg-gray-700 border border-gray-600 rounded px-1 py-0.5 text-xs text-white outline-none"><option>High</option></select>' +
            '<label class="text-[10px] text-purple-300/80">Budget</label>' +
            '<input type="number" value="4096" readonly tabindex="-1" class="w-20 bg-gray-700 border border-gray-600 rounded px-1 py-0.5 text-[10px] text-white outline-none"></div>' +
            '<div class="composer-opt composer-opt-group ld-only-grok flex items-center gap-1"><span class="text-gray-400">Effort:</span>' +
            '<select class="bg-gray-700 border border-gray-600 rounded px-1 py-0.5 text-xs text-white outline-none"><option>Med</option></select></div>' +
            '<div class="composer-opt composer-opt-group ld-only-gemini flex items-center gap-1"><span class="text-gray-400">Safety:</span>' +
            '<select class="bg-gray-700 border border-gray-600 rounded px-1 py-0.5 text-xs text-white outline-none"><option>Default</option></select></div>' +
            '<div class="composer-opt flex items-center gap-0.5 select-none"><label class="flex items-center gap-1"><input type="checkbox" class="accent-teal-500 w-3 h-3"><span class="text-teal-300 text-[10px]">PromptCache</span></label></div>' +
            '<div class="composer-opt flex items-center gap-0.5 select-none"><label class="flex items-center gap-1"><input type="checkbox" class="accent-blue-500 w-3 h-3" checked><span class="text-gray-400 text-[10px]">Compress</span></label>' +
            '<span class="text-gray-500 p-1"><i class="fas fa-cog text-[10px]"></i></span></div>' +
            '</div></div></div></div>' +
            '<div id="input-container" class="relative"><div id="input-row" class="composer-input-row flex gap-1.5 items-end">' +
            '<div class="composer-input-shell flex flex-1 items-end gap-1 min-w-0">' +
            '<span id="upload-btn" class="composer-tool-btn bg-gray-700 p-2.5 rounded-xl text-gray-300 shrink-0"><i class="fas fa-paperclip"></i></span>' +
            '<span id="rich-paste-btn" class="composer-tool-btn bg-amber-600 p-2.5 rounded-xl text-white shrink-0"><i class="fas fa-paste"></i></span>' +
            '<span id="mic-btn" class="composer-tool-btn bg-gray-700 p-2.5 rounded-xl text-gray-300 shrink-0"><i class="fas fa-microphone"></i></span>' +
            '<textarea id="prompt-input" rows="1" readonly tabindex="-1" class="flex-1 bg-transparent border-none rounded-xl px-2.5 py-2.5 outline-none resize-none text-white transition text-sm" placeholder="Ctrl + Enter で送信..."></textarea>' +
            '<span id="send-btn" class="composer-send-btn bg-blue-600 p-2.5 rounded-xl text-white font-bold shadow-lg w-11 flex justify-center items-center shrink-0"><i class="fas fa-paper-plane"></i></span>' +
            '</div></div></div>' +
            '</div></div>';
    }

    function buildChatDemo(root) {
        var systemVersion = (typeof window !== 'undefined' && window.LANDING_SYSTEM_VERSION) || 'V4.8.747';
        root.innerHTML =
            sidebarHtml(systemVersion) +
            '<div class="ld-main">' +
            mobileHeaderHtml() +
            '<div class="ld-stage">' +
            '<div id="chat-stage" class="relative flex-1 min-w-0 overflow-hidden">' +
            '<div id="chat-container" class="absolute inset-0 overflow-y-auto px-4 py-5"></div>' +
            welcomeHtml() +
            '</div></div>' +
            composerHtml() +
            '</div>';

        var setModel = function (key) {
            var m = modelInfo(key);
            root.setAttribute('data-model', m.key);
            var name = root.querySelector('#model-selector-text');
            if (name) name.textContent = m.name;
            return m;
        };

        return {
            root: root,
            body: root.querySelector('#chat-container'),
            stage: root.querySelector('#chat-stage'),
            welcome: root.querySelector('#welcome-screen'),
            input: root.querySelector('#prompt-input'),
            newThread: root.querySelector('#ld-new-thread'),
            setModel: setModel
        };
    }

    function chatScroll(body) {
        body.scrollTop = body.scrollHeight;
    }

    function appendUserMessage(body, text) {
        var wrap = document.createElement('div');
        wrap.className = 'flex justify-end mb-4 fade-in relative message-group group';
        wrap.innerHTML =
            '<div class="message-bubble bg-blue-600 text-white p-4 rounded-2xl shadow-md relative">' +
            '<div class="content-area whitespace-pre-wrap font-sans text-sm break-words">' + esc(text) + '</div></div>';
        body.appendChild(wrap);
        chatScroll(body);
        return { wrap: wrap };
    }

    /* renderPendingMessage() + buildPendingSkeletonHtml() */
    function appendPending(body, status) {
        var wrap = document.createElement('div');
        wrap.className = 'flex justify-start mb-4 fade-in';
        wrap.innerHTML =
            '<div class="message-bubble ai-pending-bubble bg-gray-700 text-white p-4 rounded-2xl rounded-tl-none shadow-md relative">' +
            '<div class="content-area pending-shimmer skeleton-pending" data-skeleton-kind="text">' +
            '<div class="skeleton-lines">' +
            '<div class="skeleton-line" style="width:92%"></div>' +
            '<div class="skeleton-line" style="width:78%"></div>' +
            '<div class="skeleton-line" style="width:86%"></div>' +
            '<div class="skeleton-line" style="width:64%"></div>' +
            '<div class="skeleton-line" style="width:48%"></div>' +
            '</div><div class="skeleton-status">' + esc(status) + '</div></div></div>';
        body.appendChild(wrap);
        chatScroll(body);
        return { wrap: wrap, setStatus: function (s) {
            var el = wrap.querySelector('.skeleton-status');
            if (el) el.textContent = s;
        } };
    }

    /* renderMessage() for an assistant reply. `opts.thought` is a list of
     * thought lines streamed into the "Thinking Process" box first (reasoning
     * models); `opts.blocks` are revealed one by one like streamed markdown. */
    function appendAIMessage(body, opts, done) {
        var wrap = document.createElement('div');
        wrap.className = 'flex justify-start mb-4 fade-in relative message-group group';
        var bubble = document.createElement('div');
        bubble.className = 'message-bubble bg-gray-700 text-white p-4 rounded-2xl shadow-md relative';
        var thoughtEl = null;
        var thoughtHeader = null;
        var thoughtBody = null;
        if (opts.thought && opts.thought.length) {
            thoughtEl = document.createElement('div');
            thoughtEl.className = 'thought-container';
            thoughtEl.innerHTML =
                '<div class="thought-header thinking-shimmer"><i class="fas fa-brain text-purple-400"></i> Thinking Process</div>' +
                '<div class="thought-content"></div>';
            thoughtHeader = thoughtEl.querySelector('.thought-header');
            thoughtBody = thoughtEl.querySelector('.thought-content');
            bubble.appendChild(thoughtEl);
        }
        var content = document.createElement('div');
        content.className = 'content-area prose prose-invert text-sm break-words';
        bubble.appendChild(content);
        wrap.appendChild(bubble);
        body.appendChild(wrap);
        chatScroll(body);

        var currentList = null;
        function revealBlock(block) {
            if (block.li) {
                if (!currentList) {
                    currentList = document.createElement('ol');
                    content.appendChild(currentList);
                }
                var li = document.createElement('li');
                li.className = 'ld-block';
                li.innerHTML = block.li;
                currentList.appendChild(li);
            } else {
                var p = document.createElement('p');
                p.className = 'ld-block';
                p.innerHTML = block.p;
                content.appendChild(p);
            }
            chatScroll(body);
        }

        function finish() {
            var footer = document.createElement('div');
            footer.className = 'text-[10px] text-slate-300/90 mt-2 text-right font-mono message-footer-meta';
            footer.innerHTML = esc(opts.model) + ' • <span class="underline decoration-dotted">' + esc(opts.tokens) + '</span>';
            bubble.appendChild(footer);
            chatScroll(body);
            done();
        }

        function streamBlocks() {
            var idx = 0;
            (function next() {
                if (idx < opts.blocks.length) {
                    revealBlock(opts.blocks[idx]);
                    idx++;
                    window.setTimeout(next, 420);
                } else {
                    finish();
                }
            })();
        }

        if (!thoughtEl) {
            streamBlocks();
            return { wrap: wrap };
        }

        var lineIdx = 0;
        (function nextThought() {
            if (lineIdx < opts.thought.length) {
                thoughtBody.textContent = opts.thought.slice(0, lineIdx + 1).join('\n');
                chatScroll(body);
                lineIdx++;
                window.setTimeout(nextThought, 650);
            } else {
                window.setTimeout(function () {
                    thoughtHeader.classList.remove('thinking-shimmer');
                    thoughtBody.classList.add('collapsed');
                    streamBlocks();
                }, 650);
            }
        })();
        return { wrap: wrap };
    }

    function fitInput(input) {
        input.style.height = 'auto';
        input.style.height = Math.min(input.scrollHeight || 0, 150) + 'px';
    }

    function typeInto(input, text, per, done) {
        var i = 0;
        function tick() {
            if (i <= text.length) {
                input.value = text.slice(0, i);
                fitInput(input);
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

        function schedule(steps) {
            for (var i = 0; i < steps.length; i++) {
                window.setTimeout(steps[i].fn, steps[i].t);
            }
        }

        function sequence() {
            body.textContent = '';
            api.input.value = '';
            fitInput(api.input);
            api.welcome.classList.remove('ld-hidden');
            api.newThread.classList.add('ld-hidden');
            api.stage.classList.remove('ld-demo-dimming');
            api.setModel('gemini');

            var steps = [];
            var current = null;
            var q1 = '2026年の日本で注目のテックトレンドを、理由付きで5つ教えて';
            var q2 = 'じゃあ2026年のAIエージェント事情は、日本の導入状況を踏まえてまとめて';
            var typed = false;
            var t = 0;

            function typeStep(text, start) {
                steps.push({ t: start, fn: function () {
                    typed = false;
                    root.classList.add('ld-typing');
                    if (reduced) { api.input.value = text; fitInput(api.input); typed = true; }
                    else typeInto(api.input, text, 50, function () { typed = true; });
                } });
            }
            function sendStep(text, at) {
                steps.push({ t: at, fn: function () {
                    if (!typed) return;
                    api.input.value = '';
                    fitInput(api.input);
                    root.classList.remove('ld-typing');
                    api.welcome.classList.add('ld-hidden');
                    api.newThread.classList.remove('ld-hidden');
                    current = appendUserMessage(body, text);
                } });
            }
            function removeCurrent() {
                if (current && current.wrap) current.wrap.remove();
                current = null;
            }

            /* 1) Gemini 3.6 Flash (Thinking) */
            t = 900;
            typeStep(q1, t);
            t += 2300;
            sendStep(q1, t);
            t += 700;
            steps.push({ t: t, fn: function () { current = appendPending(body, 'APIに送信中...'); } });
            t += 1200;
            steps.push({ t: t, fn: function () { if (current && current.setStatus) current.setStatus('回答を生成中...'); } });
            t += 900;
            steps.push({ t: t, fn: function () {
                removeCurrent();
                appendAIMessage(body, {
                    model: 'Gemini 3.6 Flash',
                    tokens: 'In 2310 / Out 1845 (Thought 612)',
                    thought: [
                        '日本の2026年のテックトレンドを5つに絞って整理する。',
                        'AI、半導体、データセンター、医療、生成コンテンツの5領域を選び、',
                        'それぞれに理由を添えて日本の事情に沿って説明する。'
                    ],
                    blocks: [
                        { p: '2026年の日本で注目されるテックトレンドを、理由とあわせて5つ挙げます。' },
                        { li: '<strong>AIエージェントの実務浸透</strong> — 経理・カスタマーサポートなどの定型業務でエージェント運用が標準化。' },
                        { li: '<strong>モバイル型データセンター</strong> — 電力制約への対策として、遊休地を活用したコンパクトDCの計画が全国で進行。' },
                        { li: '<strong>次世代半導体パッケージング</strong> — 2nm世代で日本勢の後工程受託が拡大し、供給網の再編が加速。' },
                        { li: '<strong>AIによる個別最適医療</strong> — 自治体と病院の連携で予防医療のAI診断が拡大。' },
                        { li: '<strong>クリエイティブ生成の民主化</strong> — 映像・3D・音声の生成コストが大幅に低下。' }
                    ]
                }, function () {});
            } });

            /* 2) Switch the model to Grok 4.3 and ask a follow-up */
            t += 5600;
            steps.push({ t: t, fn: function () { api.setModel('grok'); } });
            t += 700;
            typeStep(q2, t);
            t += 2700;
            sendStep(q2, t);
            t += 700;
            steps.push({ t: t, fn: function () { current = appendPending(body, '接続完了。モデル応答を待機中...'); } });
            t += 1500;
            steps.push({ t: t, fn: function () {
                removeCurrent();
                appendAIMessage(body, {
                    model: 'Grok 4.3',
                    tokens: 'In 1020 / Out 3402 (Thought 1104)',
                    thought: null,
                    blocks: [
                        { p: '2026年のAIエージェント動向を、日本の導入状況を踏まえてまとめます。' },
                        { li: '<strong>MCPなどの標準プロトコル</strong>が普及し、異なるベンダーのエージェントが相互運用できる時代へ。' },
                        { li: 'エージェント同士がタスクを委譲する<strong>オーケストレーション</strong>が一般化。' },
                        { li: 'セキュリティ面では、エージェント専用の監視体制<strong>Agent SOC</strong>という新職種が登場。' }
                    ]
                }, function () {});
            } });

            t += 3200;
            steps.push({ t: t, fn: function () {
                api.stage.classList.add('ld-demo-dimming');
                window.setTimeout(function () { sequence(); }, 900);
            } });

            schedule(steps);
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

    /* ── FAQ accordion open/close animation ──
     * Native <details> swaps content visibility instantly. Intercept summary
     * clicks and animate height/opacity so answers expand and collapse smoothly. */
    function initFaqAccordion() {
        if (typeof document === 'undefined') return;
        /* Node DOM shim used by tests may lack querySelectorAll. */
        if (typeof document.querySelectorAll !== 'function') return;
        var items = document.querySelectorAll('details.ld-faq');
        if (!items.length) return;

        function clearInline(body) {
            body.style.height = '';
            body.style.opacity = '';
            body.style.overflow = '';
            body.style.paddingTop = '';
            body.style.paddingBottom = '';
            body.classList.remove('ld-faq-animating');
        }

        for (var i = 0; i < items.length; i++) {
            (function (details) {
                if (details.getAttribute('data-ld-faq-anim')) return;
                details.setAttribute('data-ld-faq-anim', '1');
                var body = details.querySelector('.ld-faq-body');
                var summary = details.querySelector('summary');
                if (!body || !summary) return;
                var busy = false;
                var endTimer = null;

                function afterTransition(fn) {
                    var finished = false;
                    var wrap = function (ev) {
                        if (ev && ev.target !== body) return;
                        if (ev && ev.propertyName && ev.propertyName !== 'height') return;
                        if (finished) return;
                        finished = true;
                        body.removeEventListener('transitionend', wrap);
                        if (endTimer) { window.clearTimeout(endTimer); endTimer = null; }
                        fn();
                    };
                    body.addEventListener('transitionend', wrap);
                    endTimer = window.setTimeout(function () { wrap(null); }, 420);
                }

                summary.addEventListener('click', function (ev) {
                    /* Let keyboard / assistive tech use the native toggle when
                     * we are mid-animation or reduced-motion is preferred. */
                    if (busy) {
                        ev.preventDefault();
                        return;
                    }
                    if (DEFAULTS.reduced) {
                        /* Native toggle; still sync opened class for styling. */
                        window.setTimeout(function () {
                            body.classList.toggle('ld-faq-opened', details.open);
                        }, 0);
                        return;
                    }

                    ev.preventDefault();

                    if (details.open) {
                        /* ── Close ── */
                        busy = true;
                        body.classList.remove('ld-faq-opened');
                        body.classList.add('ld-faq-animating');
                        var startH = body.scrollHeight;
                        body.style.height = startH + 'px';
                        body.style.opacity = '1';
                        body.style.overflow = 'hidden';
                        void body.offsetHeight;
                        body.style.height = '0px';
                        body.style.opacity = '0';
                        body.style.paddingTop = '0px';
                        body.style.paddingBottom = '0px';
                        afterTransition(function () {
                            details.open = false;
                            clearInline(body);
                            busy = false;
                        });
                    } else {
                        /* ── Open ── */
                        busy = true;
                        details.open = true;
                        body.classList.add('ld-faq-animating');
                        body.classList.remove('ld-faq-opened');
                        /* Measure natural height first (height:auto), then collapse
                         * to 0 and animate up — scrollHeight is unreliable while
                         * height is forced to 0 in some engines. */
                        body.style.height = 'auto';
                        body.style.opacity = '0';
                        body.style.overflow = 'hidden';
                        var endH = body.scrollHeight;
                        body.style.height = '0px';
                        void body.offsetHeight;
                        body.style.height = endH + 'px';
                        body.style.opacity = '1';
                        afterTransition(function () {
                            clearInline(body);
                            body.classList.add('ld-faq-opened');
                            busy = false;
                        });
                    }
                });
            })(items[i]);
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

        /* FAQ accordion animation (idempotent via data-ld-faq-anim). */
        initFaqAccordion();
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
        initReveal: initReveal,
        initFaqAccordion: initFaqAccordion
    };
});
