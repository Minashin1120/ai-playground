/*
 * ログの収集を強化 (activity log).
 *
 * While the user turns it on (settings > フィードバック), this records the browser's own
 * operations to localStorage: clicks (element ids/classes, never text), setting switches,
 * requests (method, path without query, status, time), modals, toasts, page lifecycle,
 * errors and console warnings. It never records what was typed, prompts, answers or file
 * contents. Sending a feedback attaches the last hour (`window.ActivityLog.recent()`).
 *
 * The log is not part of the site cache: clearing the cache keeps it, and it is removed
 * only by the "ログを削除" button, turning the setting off, or another account signing in.
 */
(function () {
    'use strict';
    if (window.ActivityLog) return;

    var ENABLED_KEY = 'aip_activity_log_enabled';
    var DATA_KEY = 'aip_activity_log_v1';
    var OWNER_KEY = 'aip_activity_log_owner';
    var KEEP_MS = 60 * 60 * 1000;
    var MAX_STORED_CHARS = 1500000;
    var MAX_FIELD = 500;
    var FLUSH_MS = 2000;

    var enabled = false;
    var buffer = [];
    var flushTimer = null;
    var seq = 0;
    var listeners = [];

    function readStorage(key) {
        try { return window.localStorage.getItem(key); } catch (e) { return null; }
    }
    function writeStorage(key, value) {
        try { window.localStorage.setItem(key, value); return true; } catch (e) { return false; }
    }
    function removeStorage(key) {
        try { window.localStorage.removeItem(key); } catch (e) {}
    }

    enabled = readStorage(ENABLED_KEY) === '1';

    function clip(value, max) {
        var text = String(value == null ? '' : value);
        var limit = max || MAX_FIELD;
        return text.length > limit ? text.slice(0, limit) + '…(' + text.length + ')' : text;
    }

    function readStored() {
        var raw = readStorage(DATA_KEY);
        if (!raw) return [];
        try {
            var parsed = JSON.parse(raw);
            return Array.isArray(parsed) ? parsed : [];
        } catch (e) {
            return [];
        }
    }

    function prune(entries) {
        var cutoff = Date.now() - KEEP_MS;
        var kept = entries.filter(function (entry) { return entry && entry.t >= cutoff; });
        var text = JSON.stringify(kept);
        while (text.length > MAX_STORED_CHARS && kept.length > 1) {
            kept = kept.slice(Math.ceil(kept.length / 4));
            text = JSON.stringify(kept);
        }
        return { entries: kept, text: text };
    }

    function flush() {
        if (flushTimer) { clearTimeout(flushTimer); flushTimer = null; }
        if (!buffer.length) return;
        var pending = buffer;
        buffer = [];
        if (!enabled) return;
        var result = prune(readStored().concat(pending));
        if (!writeStorage(DATA_KEY, result.text)) {
            // The site's storage is full: keep the newest half so recording can continue.
            var half = result.entries.slice(Math.floor(result.entries.length / 2));
            writeStorage(DATA_KEY, JSON.stringify(half));
        }
    }

    function scheduleFlush() {
        if (!flushTimer) flushTimer = setTimeout(flush, FLUSH_MS);
    }

    function log(ev, fields) {
        if (!enabled) return;
        try {
            var entry = { t: Date.now(), ev: clip(ev, 80), seq: ++seq };
            if (fields && typeof fields === 'object') {
                Object.keys(fields).forEach(function (key) {
                    var value = fields[key];
                    if (value === undefined) return;
                    if (value === null || typeof value === 'number' || typeof value === 'boolean') entry[key] = value;
                    else entry[key] = clip(typeof value === 'object' ? JSON.stringify(value) : value);
                });
            }
            buffer.push(entry);
            scheduleFlush();
        } catch (e) {}
    }

    function recent(windowMs, maxChars) {
        flush();
        var cutoff = Date.now() - (windowMs || KEEP_MS);
        var entries = readStored().filter(function (entry) { return entry && entry.t >= cutoff; });
        var limit = maxChars || 3000000;
        var size = JSON.stringify(entries).length;
        while (size > limit && entries.length > 1) {
            entries = entries.slice(Math.ceil(entries.length / 10));
            size = JSON.stringify(entries).length;
        }
        return entries;
    }

    function stats() {
        var raw = readStorage(DATA_KEY) || '';
        return { count: readStored().length + buffer.length, chars: raw.length };
    }

    function clear() {
        buffer = [];
        if (flushTimer) { clearTimeout(flushTimer); flushTimer = null; }
        removeStorage(DATA_KEY);
        notify();
    }

    function setEnabled(value) {
        var next = !!value;
        if (next === enabled) return;
        if (!next) log('activity_log.disabled');
        flush();
        enabled = next;
        writeStorage(ENABLED_KEY, next ? '1' : '0');
        if (next) {
            log('activity_log.enabled', pageInfo());
        } else {
            clear();
        }
        notify();
    }

    function onChange(callback) {
        if (typeof callback === 'function') listeners.push(callback);
    }
    function notify() {
        listeners.slice().forEach(function (callback) { try { callback(enabled); } catch (e) {} });
    }

    // --- Descriptions that never include user text -------------------------------------

    function pathOnly(url) {
        try {
            var parsed = new URL(String(url), window.location.href);
            if (parsed.origin === window.location.origin) return parsed.pathname;
            return parsed.origin + parsed.pathname;
        } catch (e) {
            return clip(String(url).split('?')[0].split('#')[0], 200);
        }
    }

    function describe(el) {
        if (!el || el.nodeType !== 1) return null;
        var tag = el.tagName.toLowerCase();
        var parts = tag;
        if (el.id) parts += '#' + el.id;
        var cls = typeof el.className === 'string' ? el.className.trim().split(/\s+/).filter(Boolean) : [];
        if (cls.length) parts += '.' + cls.slice(0, 4).join('.');
        return clip(parts, 200);
    }

    function iconOf(el) {
        var icon = el && el.querySelector ? el.querySelector('i[class*="fa-"]') : null;
        if (!icon) return undefined;
        var names = String(icon.className).split(/\s+/).filter(function (c) { return /^fa-/.test(c) && !/^fa-(solid|regular|brands|fw)$/.test(c); });
        return names.length ? names.join(' ') : undefined;
    }

    function ancestry(el) {
        var ids = [];
        var node = el ? el.parentElement : null;
        while (node && ids.length < 3) {
            if (node.id) ids.push(node.id);
            node = node.parentElement;
        }
        return ids.length ? ids.join('<') : undefined;
    }

    function actionable(target) {
        if (!target || !target.closest) return target;
        return target.closest('button, a, [role="button"], [onclick], input, select, label, summary, [data-action], [data-tab]') || target;
    }

    function staticAttr(el, name) {
        // onclick / data-action / data-tab are template code, not user text.
        var value = el && el.getAttribute ? el.getAttribute(name) : null;
        return value ? clip(value, 120) : undefined;
    }

    function isSecretField(el) {
        var type = String((el && el.type) || '').toLowerCase();
        return type === 'password' || type === 'text' || type === 'search' || type === 'email' || type === 'url'
            || type === 'textarea' || type === 'number' || type === 'tel' || el.isContentEditable;
    }

    function pageInfo() {
        var nav = window.navigator || {};
        var info = {
            path: window.location.pathname,
            version: (window.CHAT_CONFIG && window.CHAT_CONFIG.appVersion) || undefined,
            ua: clip(nav.userAgent || '', 300),
            lang: nav.language,
            online: nav.onLine,
            viewport: window.innerWidth + 'x' + window.innerHeight,
            dpr: window.devicePixelRatio,
            standalone: !!(window.matchMedia && window.matchMedia('(display-mode: standalone)').matches),
            visibility: document.visibilityState,
        };
        try { info.sw = !!(nav.serviceWorker && nav.serviceWorker.controller); } catch (e) {}
        return info;
    }

    // --- Account ownership: another account's log is never sent ------------------------

    function checkOwner() {
        var owner = window.CHAT_CONFIG ? window.CHAT_CONFIG.currentUsername : undefined;
        if (owner === undefined || owner === null) return;
        var key = String(owner);
        var stored = readStorage(OWNER_KEY);
        if (stored !== null && stored !== key) removeStorage(DATA_KEY);
        writeStorage(OWNER_KEY, key);
    }

    // --- Hooks ---------------------------------------------------------------------------

    function hookFetch() {
        if (typeof window.fetch !== 'function') return;
        var original = window.fetch;
        window.fetch = function (input, init) {
            if (!enabled) return original.apply(this, arguments);
            var started = Date.now();
            var method = 'GET';
            var url = '';
            try {
                url = typeof input === 'string' ? input : (input && input.url) || String(input);
                method = String((init && init.method) || (input && input.method) || 'GET').toUpperCase();
            } catch (e) {}
            var path = pathOnly(url);
            var promise = original.apply(this, arguments);
            try {
                promise.then(function (res) {
                    log('fetch', { method: method, path: path, status: res.status, ms: Date.now() - started,
                        type: clip(res.headers && res.headers.get('content-type') || '', 60) || undefined,
                        redirected: res.redirected || undefined });
                }, function (err) {
                    log('fetch.error', { method: method, path: path, ms: Date.now() - started,
                        error: err && err.name, message: err && err.message });
                });
            } catch (e) {}
            return promise;
        };
    }

    function hookXhr() {
        if (!window.XMLHttpRequest) return;
        var proto = window.XMLHttpRequest.prototype;
        var open = proto.open;
        var send = proto.send;
        proto.open = function (method, url) {
            try { this.__aipLog = { method: String(method || 'GET').toUpperCase(), path: pathOnly(url) }; } catch (e) {}
            return open.apply(this, arguments);
        };
        proto.send = function () {
            var info = this.__aipLog;
            if (enabled && info) {
                var started = Date.now();
                var xhr = this;
                xhr.addEventListener('loadend', function () {
                    log('xhr', { method: info.method, path: info.path, status: xhr.status, ms: Date.now() - started });
                });
            }
            return send.apply(this, arguments);
        };
    }

    function hookHistory() {
        ['pushState', 'replaceState'].forEach(function (name) {
            var original = window.history && window.history[name];
            if (typeof original !== 'function') return;
            window.history[name] = function (state, title, url) {
                var result = original.apply(this, arguments);
                if (enabled && url !== undefined && url !== null) log('nav.' + name, { path: pathOnly(url) });
                return result;
            };
        });
    }

    function hookConsole() {
        ['error', 'warn'].forEach(function (level) {
            var original = window.console && window.console[level];
            if (typeof original !== 'function') return;
            window.console[level] = function () {
                if (enabled) {
                    try {
                        // Objects can hold chat data, so only strings, numbers and errors are kept.
                        var text = Array.prototype.map.call(arguments, function (arg) {
                            if (arg instanceof Error) return arg.name + ': ' + arg.message;
                            if (typeof arg === 'string') return clip(arg, 300);
                            if (typeof arg === 'number' || typeof arg === 'boolean') return String(arg);
                            return '[' + (arg === null ? 'null' : typeof arg) + ']';
                        }).join(' ');
                        log('console.' + level, { message: clip(text, 1000) });
                    } catch (e) {}
                }
                return original.apply(this, arguments);
            };
        });
    }

    function watchToasts() {
        // chat_core calls its own showToast, so the toast stack is watched instead.
        var start = function () {
            var stack = document.getElementById && document.getElementById('toast-stack');
            if (!stack || !window.MutationObserver) return;
            new MutationObserver(function (mutations) {
                if (!enabled) return;
                mutations.forEach(function (mutation) {
                    Array.prototype.forEach.call(mutation.addedNodes || [], function (node) {
                        if (node.nodeType !== 1) return;
                        var match = /\b(error|success|warning|info)\b/.exec(String(node.className));
                        var kind = match ? match[1] : undefined;
                        log('toast', { kind: kind, message: clip((node.textContent || '').trim(), 300) });
                    });
                });
            }).observe(stack, { childList: true });
        };
        if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start, { once: true });
        else start();
    }

    function watchPanels() {
        if (!window.MutationObserver) return;
        var pattern = /modal|overlay|dialog|panel|popup|popover|sidebar|menu|banner|drawer/i;
        var observer = new MutationObserver(function (mutations) {
            if (!enabled) return;
            mutations.forEach(function (mutation) {
                var el = mutation.target;
                if (!el || !el.id || !pattern.test(el.id)) return;
                var wasHidden = /(^|\s)hidden(\s|$)/.test(mutation.oldValue || '');
                var isHidden = el.classList.contains('hidden');
                if (wasHidden !== isHidden) log(isHidden ? 'ui.hide' : 'ui.show', { id: el.id });
            });
        });
        var start = function () {
            if (document.body) observer.observe(document.body, { subtree: true, attributes: true, attributeFilter: ['class'], attributeOldValue: true });
        };
        if (document.body) start(); else document.addEventListener('DOMContentLoaded', start, { once: true });
    }

    function watchLongTasks() {
        try {
            if (!window.PerformanceObserver) return;
            var observer = new PerformanceObserver(function (list) {
                if (!enabled) return;
                list.getEntries().forEach(function (entry) {
                    if (entry.duration >= 200) log('perf.longtask', { ms: Math.round(entry.duration) });
                });
            });
            observer.observe({ type: 'longtask', buffered: false });
        } catch (e) {}
    }

    function heartbeat() {
        setInterval(function () {
            if (!enabled || document.visibilityState !== 'visible') return;
            var fields = { nodes: document.getElementsByTagName('*').length };
            try {
                var memory = window.performance && window.performance.memory;
                if (memory) fields.heap_mb = Math.round(memory.usedJSHeapSize / 1048576);
            } catch (e) {}
            log('heartbeat', fields);
        }, 60000);
    }

    function listen() {
        document.addEventListener('click', function (event) {
            if (!enabled) return;
            var el = actionable(event.target);
            log('click', {
                target: describe(el), icon: iconOf(el), within: ancestry(el),
                onclick: staticAttr(el, 'onclick'), action: staticAttr(el, 'data-action') || staticAttr(el, 'data-tab'),
            });
        }, true);
        document.addEventListener('change', function (event) {
            if (!enabled) return;
            var el = event.target;
            if (!el || !el.tagName) return;
            var fields = { target: describe(el), within: ancestry(el) };
            var type = String(el.type || '').toLowerCase();
            if (type === 'checkbox' || type === 'radio') fields.checked = !!el.checked;
            else if (el.tagName === 'SELECT') fields.value = clip(el.value, 120);
            else if (type === 'file') fields.files = el.files ? el.files.length : 0;
            else if (type === 'range' || type === 'color') fields.value = clip(el.value, 40);
            else if (isSecretField(el) || el.tagName === 'TEXTAREA') fields.length = String(el.value || '').length;
            log('change', fields);
        }, true);
        document.addEventListener('keydown', function (event) {
            if (!enabled) return;
            var key = event.key;
            if (key !== 'Enter' && key !== 'Escape' && !(event.ctrlKey || event.metaKey || event.altKey)) return;
            if ((event.ctrlKey || event.metaKey || event.altKey) && String(key).length === 1 && !/[a-z0-9\/\.,]/i.test(key)) return;
            log('key', { key: clip(key, 20), ctrl: event.ctrlKey || undefined, meta: event.metaKey || undefined,
                alt: event.altKey || undefined, shift: event.shiftKey || undefined, target: describe(event.target) });
        }, true);
        document.addEventListener('submit', function (event) {
            log('submit', { target: describe(event.target) });
        }, true);
        document.addEventListener('visibilitychange', function () {
            log('page.visibility', { state: document.visibilityState });
            if (document.visibilityState === 'hidden') flush();
        });
        window.addEventListener('pagehide', function (event) {
            log('page.hide', { persisted: !!event.persisted });
            flush();
        });
        window.addEventListener('pageshow', function (event) {
            if (event.persisted) log('page.show', { persisted: true });
        });
        window.addEventListener('popstate', function () { log('nav.popstate', { path: window.location.pathname }); });
        window.addEventListener('hashchange', function () { log('nav.hashchange', { path: window.location.pathname }); });
        window.addEventListener('online', function () { log('net.online'); });
        window.addEventListener('offline', function () { log('net.offline'); });
        window.addEventListener('resize', (function () {
            var timer = null;
            return function () {
                if (!enabled) return;
                clearTimeout(timer);
                timer = setTimeout(function () { log('page.resize', { viewport: window.innerWidth + 'x' + window.innerHeight }); }, 500);
            };
        })());
        window.addEventListener('error', function (event) {
            if (!enabled) return;
            if (event && event.target && event.target !== window && event.target.tagName) {
                var src = event.target.src || event.target.href;
                log('resource.error', { target: describe(event.target), url: src ? pathOnly(src) : undefined });
                return;
            }
            log('error', {
                message: event && event.message, source: event && event.filename ? pathOnly(event.filename) : undefined,
                line: event && event.lineno, col: event && event.colno,
                stack: event && event.error && event.error.stack ? clip(event.error.stack, 2000) : undefined,
            });
        }, true);
        window.addEventListener('unhandledrejection', function (event) {
            var reason = event && event.reason;
            log('error.unhandled_rejection', {
                error: reason && reason.name, message: reason && reason.message ? reason.message : clip(reason, 300),
                stack: reason && reason.stack ? clip(reason.stack, 2000) : undefined,
            });
        });
        try {
            if (navigator.serviceWorker) {
                navigator.serviceWorker.addEventListener('controllerchange', function () { log('sw.controllerchange'); });
            }
        } catch (e) {}
        window.addEventListener('storage', function (event) {
            if (event.key !== ENABLED_KEY) return;
            enabled = event.newValue === '1';
            if (!enabled) buffer = [];
            notify();
        });
    }

    checkOwner();
    hookFetch();
    hookXhr();
    hookHistory();
    hookConsole();
    watchToasts();
    watchPanels();
    watchLongTasks();
    heartbeat();
    listen();
    log('page.load', pageInfo());
    window.addEventListener('load', function () {
        var timing = {};
        try {
            var nav = window.performance.getEntriesByType('navigation')[0];
            if (nav) {
                timing.dom_ms = Math.round(nav.domContentLoadedEventEnd);
                timing.load_ms = Math.round(nav.loadEventStart);
                timing.nav_type = nav.type;
            }
        } catch (e) {}
        log('page.loaded', timing);
    });

    function feedbackPayload() {
        if (!enabled) return null;
        return {
            client: 'web',
            version: (window.CHAT_CONFIG && window.CHAT_CONFIG.appVersion) || '',
            window_seconds: Math.round(KEEP_MS / 1000),
            entries: recent(KEEP_MS),
        };
    }

    /**
     * Shows the result of a feedback sent with [feedbackPayload] (null when the log is off) and, when
     * [chatCopy], a copy of the open chat; false keeps the form (the request failed).
     */
    function reportFeedback(res, payload, chatCopy) {
        var toast = typeof window.showToast === 'function' ? window.showToast : function () {};
        return res.json().catch(function () { return {}; }).then(function (data) {
            if (!res.ok) {
                toast(data.error || 'フィードバックの送信に失敗しました', 'error', true);
                return false;
            }
            var failed = [];
            if (payload && data.logs_saved === false) failed.push('ログ');
            if (chatCopy && data.chat_copy_saved !== true) failed.push('チャットのコピー');
            if (failed.length) {
                toast('フィードバックを送信しました（' + failed.join('と') + 'は保存できませんでした）', 'error', true);
                return true;
            }
            var sent = ['フィードバック'];
            if (payload) sent.push('直近1時間のログ（' + payload.entries.length + '件）');
            if (chatCopy) sent.push('チャットのコピー');
            toast(sent.join(sent.length > 2 ? '、' : 'と') + 'を送信しました', 'success');
            return true;
        });
    }

    // --- Settings: the switch (feedback tab) and the delete button (data tab cache card) ---

    function usageText() {
        var info = stats();
        return '操作ログ: ' + info.count + '件 (' + (info.chars * 2 / (1024 * 1024)).toFixed(1) + 'MB)';
    }

    function refreshUi() {
        if (typeof document.getElementById !== 'function') return;
        var toggle = document.getElementById('set-activity-log');
        if (toggle) toggle.checked = enabled;
        var status = document.getElementById('activity-log-feedback-status');
        if (status) {
            status.textContent = enabled
                ? '記録中です。フィードバックの送信時に直近1時間のログを送信します。（' + usageText() + '）'
                : '記録していません。';
        }
        var usage = document.getElementById('activity-log-usage-text');
        if (usage) usage.textContent = usageText();
    }

    function bindUi() {
        try { bindControls(); } catch (e) {}
    }

    function bindControls() {
        if (typeof document.getElementById !== 'function') return;
        var toggle = document.getElementById('set-activity-log');
        if (toggle) toggle.addEventListener('change', function () { setEnabled(!!toggle.checked); });
        var button = document.getElementById('clear-activity-log-btn');
        if (button) {
            button.addEventListener('click', function () {
                if (!window.confirm('操作ログを削除しますか？')) return;
                clear();
                if (typeof window.showToast === 'function') window.showToast('操作ログを削除しました', 'success');
            });
        }
        // Read the size again whenever the settings modal or one of the two tabs is shown.
        if (window.MutationObserver) {
            var observer = new MutationObserver(refreshUi);
            ['settings-modal', 'tab-feedback', 'tab-data'].forEach(function (id) {
                var el = document.getElementById(id);
                if (el) observer.observe(el, { attributes: true, attributeFilter: ['class'] });
            });
        }
        onChange(refreshUi);
        refreshUi();
    }

    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', bindUi, { once: true });
    else bindUi();

    window.ActivityLog = {
        isEnabled: function () { return enabled; },
        setEnabled: setEnabled,
        log: log,
        recent: recent,
        stats: stats,
        clear: clear,
        flush: flush,
        onChange: onChange,
        feedbackPayload: feedbackPayload,
        reportFeedback: reportFeedback,
        WINDOW_MS: KEEP_MS,
    };
})();
