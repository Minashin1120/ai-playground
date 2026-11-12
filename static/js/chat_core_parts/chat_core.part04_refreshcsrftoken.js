        const refreshCsrfToken = async () => {
            if (csrfRefreshPromise) return csrfRefreshPromise;
            csrfRefreshPromise = (async () => {
                const response = await fetch('/api/csrf_token', {
                    method: 'GET',
                    credentials: 'include',
                    cache: 'no-store',
                    headers: {'Accept': 'application/json'}
                });
                if (!response.ok) return false;
                const data = await response.json().catch(() => ({}));
                const refreshedToken = data && typeof data.csrf_token === 'string'
                    ? data.csrf_token
                    : '';
                if (!refreshedToken) return false;
                csrfToken = refreshedToken;
                const meta = document.querySelector('meta[name="csrf-token"]');
                if (meta) meta.setAttribute('content', refreshedToken);
                return true;
            })().catch(() => false).finally(() => {
                csrfRefreshPromise = null;
            });
            return csrfRefreshPromise;
        };
        const apiFetch = async (url, opts = {}) => {
            const method = (opts.method || 'GET').toUpperCase();
            const headers = Object.assign({}, opts.headers || {});
            const requiresCsrf = !['GET', 'HEAD', 'OPTIONS'].includes(method);
            if (requiresCsrf) {
                headers['X-CSRF-Token'] = csrfToken;
            }
            const credentials = opts.credentials || 'include';
            let response = await fetch(url, Object.assign({}, opts, { headers, credentials }));
            // Apache's error override can surface an application CSRF 403 as a 404 page.
            // Refresh from the current signed session and replay an unsafe request at most once.
            if (requiresCsrf && (response.status === 403 || response.status === 404)) {
                let errBody = null;
                try { errBody = await response.clone().json(); } catch (e) {}
                // Bot / lock / ban responses are intentional application errors —
                // never treat them as CSRF failures and never retry the same
                // body (Turnstile tokens are single-use; retrying a spent token
                // used to stack verify_fail counts and ban new accounts).
                const botErr = errBody && errBody.error;
                if (botErr === 'account_locked') {
                    // Admins are never subject to temporary locks (server-side
                    // also skips them); never show the lock overlay for admins.
                    if (!isAdminUser && !document.getElementById('bot-lock-overlay')) {
                        showBotLockOverlay(errBody.message || 'アカウントが一時的にロックされています。', errBody.remaining_seconds);
                    }
                    return response;
                }
                if (botErr === 'banned' || botErr === 'turnstile_failed' || botErr === 'rate_limit') {
                    return response;
                }
                if (botErr === 'turnstile_required' && isBotDetectionActive()) {
                    // Marker expired mid-session: force a fresh verification, then retry once.
                    botDetectionVerified = false;
                    const verified = await Promise.race([
                        runBotDetectionGate(),
                        new Promise((resolve) => setTimeout(() => resolve(false), 30000))
                    ]);
                    if (verified) {
                        headers['X-CSRF-Token'] = csrfToken;
                        response = await fetch(url, Object.assign({}, opts, { headers, credentials }));
                    }
                    return response;
                }
                const refreshed = await refreshCsrfToken();
                if (refreshed) {
                    headers['X-CSRF-Token'] = csrfToken;
                    response = await fetch(url, Object.assign({}, opts, { headers, credentials }));
                }
            }
            return response;
        };
        const manualSpinnerRequestOptions = (options) => window.ProgressSpinner
            ? window.ProgressSpinner.manualRequestOptions(options)
            : options;

        window.updateGoogleLinkUI = (d) => {
            const linkText = get('google-link-text');
            const emailText = get('google-email-text');
            const actionArea = get('google-action-area');
            const icon = get('google-link-icon');
            if (!linkText || !actionArea) return;

            if (d.google_id) {
                linkText.innerText = '連携済み';
                linkText.classList.replace('text-gray-200', 'text-green-400');
                emailText.innerText = d.google_email || '連携中の Google アカウント';
                icon.classList.replace('bg-gray-800', 'bg-green-900/30');
                icon.classList.add('text-green-400');
                actionArea.innerHTML = `<button onclick="unlinkGoogleAccount()" class="px-4 py-2 bg-red-900/20 hover:bg-red-900/40 text-red-400 border border-red-800 rounded text-xs font-bold transition btn-hover">連携を解除</button>`;
            } else {
                linkText.innerText = '未連携';
                linkText.classList.replace('text-green-400', 'text-gray-200');
                emailText.innerText = 'Google アカウントでログインできるようになります。';
                icon.classList.replace('bg-green-900/30', 'bg-gray-800');
                icon.classList.remove('text-green-400');
                actionArea.innerHTML = `<a href="/login/google" class="inline-block px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded text-xs font-bold transition btn-hover">Google と連携する</a>`;
            }
        };

        window.unlinkGoogleAccount = async () => {
            if (!confirm('Google 連携を解除しますか？\n解除後は Google ログインが利用できなくなります（パスワードが設定されていない場合はログインできなくなる可能性があります）。')) return;
            try {
                const res = await apiFetch(CHAT_CONFIG.urls.unlinkGoogleAccount, {method: 'POST'});
                if (res.ok) {
                    showToast('Google 連携を解除しました');
                    // Refresh UI
                    apiFetch(CHAT_CONFIG.urls.handleSettingsQuery).then(r=>r.json()).then(d=>updateGoogleLinkUI(d));
                } else {
                    const err = await res.json();
                    showToast(err.error || '解除に失敗しました', 'error', true);
                }
            } catch (e) {
                showToast('ネットワークエラーが発生しました', 'error', true);
            }
        };

        window.updateMinashinLinkUI = (d) => {
            const linkText = get('minashin-link-text');
            const emailText = get('minashin-email-text');
            const actionArea = get('minashin-action-area');
            const icon = get('minashin-link-icon');
            if (!linkText || !actionArea) return;

            if (d.minashin_sub) {
                linkText.innerText = '連携済み';
                linkText.classList.replace('text-gray-200', 'text-green-400');
                emailText.innerText = d.minashin_email || '連携中の Minashin アカウント';
                icon.classList.replace('bg-gray-800', 'bg-green-900/30');
                actionArea.innerHTML = `<button onclick="unlinkMinashinAccount()" class="px-4 py-2 bg-red-900/20 hover:bg-red-900/40 text-red-400 border border-red-800 rounded text-xs font-bold transition btn-hover">連携を解除</button>`;
            } else {
                linkText.innerText = '未連携';
                linkText.classList.replace('text-green-400', 'text-gray-200');
                emailText.innerText = 'Minashin アカウントでログインできるようになります。';
                icon.classList.replace('bg-green-900/30', 'bg-gray-800');
                actionArea.innerHTML = `<a href="/login/minashin" class="inline-block px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded text-xs font-bold transition btn-hover">Minashin と連携する</a>`;
            }
        };

        window.unlinkMinashinAccount = async () => {
            if (!confirm('Minashin 連携を解除しますか？\n解除後は Minashin ログインが利用できなくなります（パスワードが設定されていない場合はログインできなくなる可能性があります）。')) return;
            try {
                const res = await apiFetch(CHAT_CONFIG.urls.unlinkMinashinAccount, {method: 'POST'});
                if (res.ok) {
                    showToast('Minashin 連携を解除しました');
                    // Refresh UI
                    apiFetch(CHAT_CONFIG.urls.handleSettingsQuery).then(r=>r.json()).then(d=>updateMinashinLinkUI(d));
                } else {
                    const err = await res.json();
                    showToast(err.error || '解除に失敗しました', 'error', true);
                }
            } catch (e) {
                showToast('ネットワークエラーが発生しました', 'error', true);
            }
        };
        let lastClientDebugEnabled = null;
        const isClientDebugLogEnabled = () => {
            const settingEl = get('set-client-debug-log');
            return !!(settingEl && settingEl.checked);
        };
        const sendClientDebugLog = (level, message) => {
            if (!isClientDebugLogEnabled()) return;
            const payload = {
                ev: 'server.client_log'
            };
            apiFetch('/api/debug/client_log', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            }).catch(() => {});
        };
        const syncClientDebugLogToggle = (enabled, source) => {
            const settingEl = get('set-client-debug-log');
            if (settingEl) {
                settingEl.checked = !!enabled;
            }
            const normalized = !!enabled;
            if (normalized && lastClientDebugEnabled !== true) {
                sendClientDebugLog('info', `Client debug logging enabled (${source}).`);
            }
            lastClientDebugEnabled = normalized;
        };
        const nowPerfMs = () => {
            if (window.performance && typeof window.performance.now === 'function') {
                return window.performance.now();
            }
            return Date.now();
        };
        const reportFirstTokenLatency = (payload) => {
            if (!enableLatencyMetrics) return;
            try {
                if (!payload || typeof payload !== 'object') return;
                const secRaw = Number(payload.latency_seconds);
                if (!Number.isFinite(secRaw) || secRaw < 0 || secRaw > 600) return;
                const msRaw = Number(payload.latency_ms);
                const body = {
                    latency_seconds: Number(secRaw.toFixed(6)),
                    latency_ms: Number.isFinite(msRaw) ? Math.max(0, Math.round(msRaw)) : Math.round(secRaw * 1000),
                    thread_id: payload.thread_id ? String(payload.thread_id) : null,
                    job_id: payload.job_id ? String(payload.job_id) : null,
                    model: payload.model ? String(payload.model) : null,
                    first_event_type: payload.first_event_type ? String(payload.first_event_type) : "content",
                    client_sent_at_ms: Number.isFinite(Number(payload.client_sent_at_ms))
                        ? Math.round(Number(payload.client_sent_at_ms))
                        : Date.now(),
                    is_total: !!payload.is_total,
                    client_done_at_ms: Number.isFinite(Number(payload.client_done_at_ms))
                        ? Math.round(Number(payload.client_done_at_ms))
                        : null
                };
                apiFetch('/api/metrics/first_token', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(body)
                }).catch(() => {});
            } catch (_e) {}
        };
        let currentThreadId = CHAT_CONFIG.initialThreadId;
        if (currentThreadId !== null && currentThreadId !== undefined) {
            currentThreadId = String(currentThreadId);
        }
        const UPLOAD_CONCURRENCY = Math.max(1, Number(CHAT_CONFIG.uploadConcurrency) || 3);
        const TEMP_CHAT_TIMEOUT_MIN_SECONDS = 10;
        const TEMP_CHAT_TIMEOUT_MAX_SECONDS = 3600;
        const TEMP_CHAT_DEFAULT_TIMEOUT_SECONDS = 90;
        const TEMP_CHAT_HEARTBEAT_MIN_MS = 4000;
        const TEMP_CHAT_HEARTBEAT_MAX_MS = 15000;
        let activeSkill = null, editingSkillUuid = null, currentImageUrls = [], currentMaskImage = null, abortController = null, richPasteAbortController = null, userAutoScroll = true, searchTimeout;

        // Prompt History
        let promptHistory = [];
        let historyIndex = -1;
        let tempPrompt = "";

        const markerAppliedUploads = new Set();
        const attachmentSourceByPath = new Map();
        const attachmentNameByPath = new Map();
        const BROWSER_FAST_IGNORE_WARNING_STORAGE = 'browser_fast_mode_ignore_warning';
        const BROWSER_FAST_MAX_IMAGES = 4;
        const BROWSER_FAST_MAX_BYTES = 12 * 1024 * 1024;
        const browserFastLocalFiles = new Map();
        let browserFastModeEnabled = false;
        let browserFastApiKey = '';
        let browserFastApiKeyModel = '';
        let browserFastBootstrap = null;
        let browserFastPreviousOptions = null;
        let cameraCaptureStream = null;
        let cameraCaptureFacingMode = 'environment';
        let cameraCaptureBusy = false;
        let cameraCaptureSequence = 0;
        const cameraCapturePendingFiles = [];
        const cameraCapturePendingPreviewUrls = [];
        let modalThreadId = null;
        const MARKER_HINT_TEXT = "編集済みの画像を見てください。";
        const MARKER_OPACITY_MIN_PCT = 0.1;
        const MARKER_OPACITY_MAX_PCT = 100;
        const MARKER_OPACITY_MIN_ALPHA = MARKER_OPACITY_MIN_PCT / 100;
        const markerState = {
            row: null,
            filename: '',
            maskMode: false,
            hasStroke: false,
            naturalWidth: 0,
            naturalHeight: 0,
            colorHex: '#facc15',
            opacity: 0.6,
            history: [],
            mode: 'draw',
            cropRect: null,
            mosaicRects: [],
            mosaicPreviewRect: null,
            baseCanvas: null,
            baseImageData: null
        };
        const markerView = { scale: 1, offsetX: 0, offsetY: 0, minScale: 1, maxScale: 4 };
        const threadSkillMap = {};
        let pendingSkillForNewThread = null;
        let loadedSkills = [];
        let currentJobId = null;
        let currentThreadPending = null;
        let currentVisionModel = null;
        let activeStreamingBubbleId = null;
        let manualStopContext = null;
        let manualStopSeq = 0;
        let isStopMode = false;
        const suppressedPendingJobIds = new Set();
        const pendingStreamReconnectJobs = new Set();
        let editingMessageId = null; // Track message being edited
        // True once the user picks a model in this chat; editing, regenerating and branch/cancel
        // restores then keep it instead of switching to the prompt's own model.
        let modelChosenManually = false;
        const messageStore = {}, lib = { modal: get('lib-modal'), grid: get('lib-grid'), files: [], selected: new Set(), attachMode: false, searchQuery: '', favoritesOnly: false, totalCount: 0, hasMore: false, loading: false, nextOffset: 0 };
        const LIBRARY_PAGE_SIZE = 40;
        const LIB_SORT_KEY = 'lib_sort_order';
        const LIB_FAVORITES_ONLY_KEY = 'lib_favorites_only';
        let threadPage = 1, threadLoading = false, hasMoreThreads = true;
        let threadObserver = null;
        let currentQuote = "";
        let currentQuoteMessageId = null;
        let currentThreadTitle = null;
        let temporaryChatEnabled = false;
        let temporaryChatTimeoutSeconds = TEMP_CHAT_DEFAULT_TIMEOUT_SECONDS;
        let tempChatExpiresAtMs = null;
        let tempChatHeartbeatTimer = null;
        let tempChatHeartbeatIntervalMs = 0;
        let tempChatHeartbeatInFlight = false;
        let tempChatHeaderTicker = null;
        let enterToSend = CHAT_CONFIG.enterToSend;
        let autoSearchOnLinks = CHAT_CONFIG.autoSearchOnLinks;
        let useSwCache = CHAT_CONFIG.useSwCache;
        let compactPromptMode = CHAT_CONFIG.compactPromptMode;
        let minimalPromptMode = !!CHAT_CONFIG.minimalPromptMode;
        let voiceStudioUiEnabled = true;
        const CANVAS_MODE_STORAGE_KEY = 'canvas_mode_enabled_v1';
        const CODING_MODE_STORAGE_KEY = 'coding_mode_enabled_v1';
        let canvasModeEnabled = false;
        let codingModeEnabled = false;
        // UI preference and model-facing behavior are intentionally separate.
        // This becomes true only after a complete code block exists.
        let codingModeEffective = false;
        let codingTargetSelection = null;
        const canvasPreviewState = {
            blocks: [],
            rawText: '',
            renderText: '',
            selectedIndex: -1,
            selectedKey: '',
            selectionMode: 'auto',
            mobileView: 'preview',
            sourceScrollTop: 0,
            sourceScrollLeft: 0,
            frameScrollX: 0,
            frameScrollY: 0,
            frameRenderToken: 0,
            panelAnimationToken: 0,
            panelHideTimer: null,
            viewAnimationToken: 0,
            viewAnimationTimer: null,
            lastCanvasData: null
        };
        try {
            canvasModeEnabled = localStorage.getItem(CANVAS_MODE_STORAGE_KEY) === 'true';
        } catch (e) {
            canvasModeEnabled = false;
        }
        try {
            codingModeEnabled = localStorage.getItem(CODING_MODE_STORAGE_KEY) === 'true';
        } catch (e) {
            codingModeEnabled = false;
        }
        let enableLatencyMetrics = CHAT_CONFIG.enableLatencyMetrics;
        let promptControlsExpanded = false;
        const appVersion = CHAT_CONFIG.appVersion;
        const botConfig = CHAT_CONFIG.botConfig;
        const isAdminUser = botConfig && botConfig.isAdmin;
        const currentUsername = CHAT_CONFIG.currentUsername;
        let turnstileWidgetId = null;
        let turnstileToken = null;
        let turnstilePending = false;
        let botDetectionVerified = false;
        let botDetectionGatePromise = null;
        let botDetectionOverlayShown = false;
        let botDetectionDialogWidgetId = null;
        let sendButtonSpamTimestamps = [];
        let chatDefaultsLoaded = false;
        let modelApiKeyMap = {};
        const THREAD_INITIAL_MESSAGE_LIMIT = 50;
        const THREAD_OLDER_PAGE_SIZE = 50;
        const LOW_BANDWIDTH_INITIAL_MESSAGE_LIMIT = 40;
        const LOW_BANDWIDTH_OLDER_PAGE_SIZE = 60;
        const LOW_BANDWIDTH_MODE_STORAGE_KEY = 'low_bandwidth_mode_pref_v1';
        const LOW_BANDWIDTH_DECORATION_VISIBILITY_THRESHOLD = 0.02;
        const MATHJAX_SRC = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js';
        const HLJS_JS_SRC = 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js';
        const HLJS_CSS_SRC = 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/atom-one-dark.min.css';
        let mathJaxLoadPromise = null;
        let incrementalMathTypesetChain = Promise.resolve();
        let highlightLoadPromise = null;
        let lowBandwidthModePreference = 'auto';
        let lowBandwidthModeAuto = false;
        let lowBandwidthMode = false;
        let lowBandwidthModeReason = '';
        let lowBandwidthConnectionListenerAttached = false;
        let deferredDecorationObserver = null;
        const deferredDecorationTextMap = new WeakMap();
        let threadHasOlderMessages = false;
        let oldestLoadedMessageId = null;
        let loadingOlderMessages = false;
        let threadLoadSequence = 0;

        // Tree State
        let allMessages = [];
        let currentLeafId = null;
        let currentParentId = null;

        function loadScriptOnce(src, id) {
            const existingById = id ? document.getElementById(id) : null;
            if (existingById) {
                if (existingById.dataset.loaded === '1') return Promise.resolve(existingById);
                return new Promise((resolve, reject) => {
                    existingById.addEventListener('load', () => resolve(existingById), { once: true });
                    existingById.addEventListener('error', reject, { once: true });
                });
            }
            return new Promise((resolve, reject) => {
                const s = document.createElement('script');
                if (id) s.id = id;
                s.src = src;
                s.async = true;
                s.onload = () => {
                    s.dataset.loaded = '1';
                    resolve(s);
                };
                s.onerror = reject;
                document.head.appendChild(s);
            });
        }
        function loadStylesheetOnce(href, id) {
            const existingById = id ? document.getElementById(id) : null;
            if (existingById) return Promise.resolve(existingById);
            const existing = Array.from(document.querySelectorAll('link[rel="stylesheet"]')).find((el) => el.href === href);
            if (existing) return Promise.resolve(existing);
            return new Promise((resolve, reject) => {
                const link = document.createElement('link');
                if (id) link.id = id;
                link.rel = 'stylesheet';
                link.href = href;
                link.onload = () => resolve(link);
                link.onerror = reject;
                document.head.appendChild(link);
            });
        }
        async function ensureMathJaxLoaded() {
            if (window.MathJax && typeof window.MathJax.typesetPromise === 'function') return window.MathJax;
            if (!mathJaxLoadPromise) {
                window.MathJax = window.MathJax || {
                    tex: {
                        inlineMath: [['\\(', '\\)'], ['$', '$']],
                        displayMath: [['$$', '$$'], ['\\[', '\\]']],
                        processEscapes: true
                    },
                    options: {
                        // skipHtmlTags の pre/code を再処理する用途のみ。全要素は通常どおり走査する
                        ignoreHtmlClass: 'tex2jax_ignore|mathjax_ignore',
                        processHtmlClass: 'tex2jax_process|mathjax_process'
                    },
                    startup: {
                        typeset: false
                    }
                };
                mathJaxLoadPromise = loadScriptOnce(MATHJAX_SRC, 'MathJax-script').catch((err) => {
                    mathJaxLoadPromise = null;
                    throw err;
                });
            }
            await mathJaxLoadPromise;
            return window.MathJax || null;
        }
        async function ensureHighlightLoaded() {
            if (window.hljs) return window.hljs;
            if (!highlightLoadPromise) {
                highlightLoadPromise = Promise.all([
                    loadStylesheetOnce(HLJS_CSS_SRC, 'hljs-theme-chat'),
                    loadScriptOnce(HLJS_JS_SRC, 'hljs-script')
                ]).then(() => window.hljs || null).catch((err) => {
                    highlightLoadPromise = null;
                    throw err;
                });
            }
            return await highlightLoadPromise;
        }
        function maybeNeedsMathJax(text) {
            const t = String(text || '');
            if (t.includes('$$') || t.includes('\\(') || t.includes('\\（') || t.includes('\\[') || t.includes('\\begin{')) return true;
            // 単独 $...$ （通貨 $12 は除外し、数式らしい中身のみ）
            return /(?<!\$)\$(?!\$)(?=[\s\S]*?[A-Za-z\\^_{}])(?:[^$\n\\]|\\.)+?\$(?!\$)/.test(t);
        }
        /**
         * marked は \( \[ などのバックスラッシュを Markdown エスケープとして除去するため、
         * 数式セグメントをプレースホルダへ退避してから parse する。
         */
        function protectMathSegments(src) {
            const source = String(src || '');
            const blocks = [];
            const stash = (match) => {
                const key = `@@MATHJAX_BLOCK_${blocks.length}@@`;
                blocks.push(match);
                return key;
            };
            // コードフェンス内は触らない（表示用の LaTeX ソースを壊さない）
            const parts = [];
            const fenceRe = /(^|\n)([ \t]*)(`{3,}|~{3,})[^\n]*\n[\s\S]*?(?:\n\2\3[ \t]*(?:\n|$)|$)/g;
            let last = 0;
            let m;
            while ((m = fenceRe.exec(source)) !== null) {
                const start = m.index;
                if (start > last) {
                    parts.push({ type: 'text', value: source.slice(last, start) });
                }
                parts.push({ type: 'code', value: m[0] });
                last = start + m[0].length;
            }
            if (last < source.length) {
                parts.push({ type: 'text', value: source.slice(last) });
            }
            if (!parts.length) {
                parts.push({ type: 'text', value: source });
            }
            const protectedText = parts.map((part) => {
                if (part.type === 'code') return part.value;
                let t = part.value;
                // 長い／優先度の高いデリミタから順に退避
                t = t.replace(/\$\$([\s\S]+?)\$\$/g, stash);
                // モデルが全角の括弧で \（ … \） と書くことがあるため、半角の区切りへ直して退避
                t = t.replace(/\\[(（]([\s\S]+?)\\[)）]/g, (match, body) => stash(`\\(${body}\\)`));
                t = t.replace(/\\\[([\s\S]+?)\\\]/g, stash);
                t = t.replace(/\\begin\{([a-zA-Z*]+)\}([\s\S]+?)\\end\{\1\}/g, stash);
                // 単独 $...$ （空や空白のみ、および $$ は除外）
                t = t.replace(/(?<!\$)\$(?!\$)([^\s$](?:(?:[^$\n\\]|\\.)*?[^\s$])?)\$(?!\$)/g, stash);
                return t;
            }).join('');
            return { text: protectedText, blocks };
        }
        function getStreamMathSegmentKey(index, raw) {
            const source = String(raw || '');
            let hash = 2166136261;
            for (let i = 0; i < source.length; i++) {
                hash ^= source.charCodeAt(i);
                hash = Math.imul(hash, 16777619);
            }
            return `${index}-${source.length}-${(hash >>> 0).toString(16)}`;
        }
        function restoreMathSegments(html, blocks, opts = {}) {
            if (!blocks || !blocks.length) return String(html || '');
            return String(html || '').replace(/@@MATHJAX_BLOCK_(\d+)@@/g, (_, idx) => {
                const raw = blocks[Number(idx)];
                if (raw === undefined || raw === null) return '';
                // HTML へ埋め込むため特殊文字のみエスケープ（textContent 上は元の TeX に戻る）
                const escaped = String(raw)
                    .replace(/&/g, '&amp;')
                    .replace(/</g, '&lt;')
                    .replace(/>/g, '&gt;');
                if (!opts.streamMathSegments) return escaped;
                const key = getStreamMathSegmentKey(Number(idx), raw);
                return `<span class="stream-math-segment mathjax_process" data-stream-math-key="${key}">${escaped}</span>`;
            });
        }
        function maybeNeedsHighlight(text, container = null) {
            const t = String(text || '');
            if (t.includes('```')) return true;
            if (!container || typeof container.querySelector !== 'function') return false;
            return !!container.querySelector('pre code');
        }
        function queueMathTypeset(container, text = '', opts = {}) {
            if (lowBandwidthMode && !opts.force) return;
            if (!container || !maybeNeedsMathJax(text)) return;
            return ensureMathJaxLoaded()
                .then(() => {
                    if (!window.MathJax || typeof window.MathJax.typesetPromise !== 'function') return;
                    // ストリーム再描画で DOM が差し替わったあとに古い math リストが残ると typeset が失敗するためクリア
                    try {
                        if (typeof window.MathJax.typesetClear === 'function') {
                            window.MathJax.typesetClear([container]);
                        }
                    } catch (e) {}
                    return window.MathJax.typesetPromise([container]).catch(() => {
                        console.warn('Math typesetting failed');
                    });
                })
                .catch(() => { console.warn('Math renderer loading failed'); });
        }
        function queueIncrementalMathTypeset(elements) {
            const candidates = Array.from(elements || []).filter((el) => (
                el && el.isConnected && !el.getAttribute('data-stream-math-state')
            ));
            if (!candidates.length || lowBandwidthMode) return;
            candidates.forEach((el) => el.setAttribute('data-stream-math-state', 'queued'));
            incrementalMathTypesetChain = incrementalMathTypesetChain
                .catch(() => {})
                .then(async () => {
                    await ensureMathJaxLoaded();
                    const connected = candidates.filter((el) => el.isConnected && el.getAttribute('data-stream-math-state') === 'queued');
                    if (!connected.length || !window.MathJax || typeof window.MathJax.typesetPromise !== 'function') return;
                    connected.forEach((el) => el.setAttribute('data-stream-math-state', 'rendering'));
                    try {
                        await window.MathJax.typesetPromise(connected);
                        connected.forEach((el) => {
                            if (el.isConnected) el.setAttribute('data-stream-math-state', 'rendered');
                        });
                    } catch (e) {
                        connected.forEach((el) => el.removeAttribute('data-stream-math-state'));
                    }
                })
                .catch(() => {
                    candidates.forEach((el) => el.removeAttribute('data-stream-math-state'));
                });
        }
        function queueHighlight(container, text = '', opts = {}) {
            if (lowBandwidthMode && !opts.force) return;
            if (!container || !maybeNeedsHighlight(text, container)) return;
            // Skip redundant highlighting during active streaming (already handled by custom renderer)
            if (activeStreamingBubbleId && container.closest(`#${activeStreamingBubbleId}`)) return;

            ensureHighlightLoaded()
                .then(() => {
                    if (!window.hljs) return;
                    container.querySelectorAll('pre code').forEach((codeEl) => {
                        if (codeEl.getAttribute('data-highlighted') === 'true' && !opts.force) return;
                        try {
                            window.hljs.highlightElement(codeEl);
                        } catch (e) {}
                    });
                })
                .catch(() => {});
        }
        function getNetworkConnectionInfo() {
            return navigator.connection || navigator.mozConnection || navigator.webkitConnection || null;
        }
        function detectLowBandwidthModeAuto() {
            const conn = getNetworkConnectionInfo();
            if (!conn) {
                return { enabled: false, reason: '' };
            }
            const saveData = !!conn.saveData;
            const effectiveType = String(conn.effectiveType || '').toLowerCase();
            const downlink = Number(conn.downlink || 0);
            const isSlowType = effectiveType === 'slow-2g' || effectiveType === '2g' || effectiveType === '3g';
            const isLowDownlink = Number.isFinite(downlink) && downlink > 0 && downlink < 1.3;
            const enabled = saveData || isSlowType || isLowDownlink;
            const parts = [];
            if (saveData) parts.push('データ節約');
            if (effectiveType) parts.push(`回線:${effectiveType}`);
            if (isLowDownlink) parts.push(`下り:${downlink}Mbps`);
            return { enabled, reason: parts.join(' / ') };
        }
        function normalizeLowBandwidthModePreference(raw) {
            const v = String(raw || '').trim().toLowerCase();
            if (v === 'on' || v === 'off' || v === 'auto') return v;
            return 'auto';
        }
        function readLowBandwidthModePreference() {
            try {
                return normalizeLowBandwidthModePreference(localStorage.getItem(LOW_BANDWIDTH_MODE_STORAGE_KEY) || 'auto');
            } catch (e) {
                return 'auto';
            }
        }
        function persistLowBandwidthModePreference(pref) {
            const normalized = normalizeLowBandwidthModePreference(pref);
            lowBandwidthModePreference = normalized;
            try {
                if (normalized === 'auto') localStorage.removeItem(LOW_BANDWIDTH_MODE_STORAGE_KEY);
                else localStorage.setItem(LOW_BANDWIDTH_MODE_STORAGE_KEY, normalized);
            } catch (e) {}
        }
        function getEffectiveThreadInitialMessageLimit() {
            return lowBandwidthMode ? LOW_BANDWIDTH_INITIAL_MESSAGE_LIMIT : THREAD_INITIAL_MESSAGE_LIMIT;
        }
        function getEffectiveThreadOlderPageSize() {
            return lowBandwidthMode ? LOW_BANDWIDTH_OLDER_PAGE_SIZE : THREAD_OLDER_PAGE_SIZE;
        }
        function mergeBtnClasses(btn, add = [], remove = []) {
            if (!btn) return;
            remove.forEach(c => btn.classList.remove(c));
            add.forEach(c => btn.classList.add(c));
        }
        function updateLowBandwidthModeUi() {
            const btn = get('low-bandwidth-toggle-btn');
            const pill = get('low-bandwidth-status-pill');
            const prefLabel = lowBandwidthModePreference === 'auto' ? '自動' : (lowBandwidthModePreference === 'on' ? '固定ON' : '固定OFF');
            const modeLabel = lowBandwidthMode ? 'ON' : 'OFF';
            const reasonText = lowBandwidthModeReason ? ` (${lowBandwidthModeReason})` : '';
            if (btn) {
                btn.setAttribute('title', `低速回線モード ${modeLabel} / ${prefLabel}${reasonText}`);
                btn.setAttribute('aria-pressed', lowBandwidthMode ? 'true' : 'false');
                if (lowBandwidthMode) {
                    mergeBtnClasses(btn, ['text-amber-200', 'bg-amber-900/30', 'border', 'border-amber-600/40'], ['text-gray-400']);
                } else {
                    mergeBtnClasses(btn, ['text-gray-400'], ['text-amber-200', 'bg-amber-900/30', 'border', 'border-amber-600/40']);
                }
            }
            if (pill) {
                if (lowBandwidthMode) {
                    pill.classList.remove('hidden');
                    const autoBadge = lowBandwidthModePreference === 'auto' ? ' (自動)' : ' (手動)';
                    pill.innerHTML = `<i class="fas fa-wifi mr-1"></i>低速回線モード${autoBadge}${lowBandwidthModeReason ? `: ${escapeHtml(lowBandwidthModeReason)}` : ''}`;
                } else {
                    pill.classList.add('hidden');
                    pill.innerHTML = '<i class="fas fa-wifi mr-1"></i>低速回線モード';
                }
            }
        }
        function refreshDecorationsForVisibleChat() {
            const container = get('chat-container');
            if (!container) return;
            queueHighlight(container, container.textContent || '', { force: true });
            queueMathTypeset(container, container.textContent || '', { force: true });
        }
        function applyLowBandwidthModeState(nextMode, opts = {}) {
            const prev = lowBandwidthMode;
            lowBandwidthMode = !!nextMode;
            updateLowBandwidthModeUi();
            if (prev && !lowBandwidthMode) {
                // When leaving low-bandwidth mode, progressively decorate existing visible content.
                refreshDecorationsForVisibleChat();
            }
            if (opts.notify) {
                const prefText = lowBandwidthModePreference === 'auto' ? '自動' : '手動';
                const suffix = lowBandwidthModeReason ? ` (${lowBandwidthModeReason})` : '';
                showToast(`低速回線モードを${lowBandwidthMode ? 'ON' : 'OFF'}にしました [${prefText}]${suffix}`, 'info', false);
            }
        }
        function recomputeLowBandwidthMode(opts = {}) {
            const auto = detectLowBandwidthModeAuto();
            lowBandwidthModeAuto = !!auto.enabled;
            lowBandwidthModeReason = auto.reason || '';
            const effective = lowBandwidthModePreference === 'on'
                ? true
                : (lowBandwidthModePreference === 'off' ? false : lowBandwidthModeAuto);
            applyLowBandwidthModeState(effective, opts);
        }
        function cycleLowBandwidthModePreference() {
            const current = normalizeLowBandwidthModePreference(lowBandwidthModePreference);
            const next = current === 'auto' ? 'on' : (current === 'on' ? 'off' : 'auto');
            persistLowBandwidthModePreference(next);
            recomputeLowBandwidthMode({ notify: true });
        }
        function ensureDeferredDecorationObserver() {
            if (deferredDecorationObserver || typeof IntersectionObserver === 'undefined') return deferredDecorationObserver;
            const rootEl = get('chat-container') || null;
            deferredDecorationObserver = new IntersectionObserver((entries) => {
                entries.forEach((entry) => {
                    if (!entry.isIntersecting || !entry.target) return;
                    runDeferredDecorations(entry.target);
                });
            }, { root: rootEl, threshold: LOW_BANDWIDTH_DECORATION_VISIBILITY_THRESHOLD });
            return deferredDecorationObserver;
        }
        function runDeferredDecorations(container) {
            if (!container) return;
            if (deferredDecorationObserver) {
                try { deferredDecorationObserver.unobserve(container); } catch (e) {}
            }
            const text = deferredDecorationTextMap.get(container) || '';
            queueHighlight(container, text, { force: true });
            queueMathTypeset(container, text, { force: true });
        }
        function queueMessageDecorations(container, text = '') {
            if (!container) return;
            if (!lowBandwidthMode) {
                queueHighlight(container, text);
                queueMathTypeset(container, text);
                return;
            }
            if (!maybeNeedsHighlight(text, container) && !maybeNeedsMathJax(text)) return;
            deferredDecorationTextMap.set(container, String(text || ''));
            const chatRoot = get('chat-container');
            if (chatRoot && container === chatRoot) {
                window.setTimeout(() => runDeferredDecorations(container), 250);
                return;
            }
            if (!container.isConnected) return;
            const observer = ensureDeferredDecorationObserver();
            if (observer) {
                observer.observe(container);
                return;
            }
            window.setTimeout(() => runDeferredDecorations(container), 250);
        }
        function initLowBandwidthMode() {
            lowBandwidthModePreference = readLowBandwidthModePreference();
            recomputeLowBandwidthMode({ notify: false });
            const btn = get('low-bandwidth-toggle-btn');
            if (btn && !btn.__lowBandwidthBound) {
                btn.__lowBandwidthBound = true;
                btn.addEventListener('click', (e) => {
                    if (e) e.preventDefault();
                    cycleLowBandwidthModePreference();
                });
            }
            const conn = getNetworkConnectionInfo();
            if (conn && typeof conn.addEventListener === 'function' && !lowBandwidthConnectionListenerAttached) {
                lowBandwidthConnectionListenerAttached = true;
                conn.addEventListener('change', () => {
                    if (lowBandwidthModePreference === 'auto') {
                        recomputeLowBandwidthMode({ notify: true });
                    } else {
                        const auto = detectLowBandwidthModeAuto();
                        lowBandwidthModeAuto = !!auto.enabled;
                        lowBandwidthModeReason = auto.reason || '';
                        updateLowBandwidthModeUi();
                    }
                });
            }
        }

        function escapeHtml(t) {
            if (t === null || t === undefined) return '';
            return String(t).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
        }

        // === Security: Block compromised polyfill.io and similar + password-phishing script patterns ===
        // See: http://polyfill.io/ was taken over and serves malware; LLMs sometimes emit it in generated HTML/JS code.
        const BLOCKED_SCRIPT_HOSTS = ['polyfill.io', 'cdn.polyfill.io'];
        function isBlockedScriptSrc(src) {
            if (!src) return false;
            const s = String(src).trim();
            if (!s) return false;
            let testUrl = s;
            if (s.startsWith('//')) testUrl = 'https:' + s;
            else if (!/^https?:\/\//i.test(s) && !s.startsWith('data:') && !s.startsWith('blob:')) {
                testUrl = 'https://' + s;
            }
            try {
                const u = new URL(testUrl, 'https://example.com');
                const host = (u.hostname || '').toLowerCase();
                return BLOCKED_SCRIPT_HOSTS.some(bad => host === bad || host.endsWith('.' + bad));
            } catch (e) {
                return /polyfill\.io/i.test(s);
            }
        }
        function isPasswordPromptingScript(code) {
            if (!code) return false;
            const c = String(code);
            const lower = c.toLowerCase();
            // Patterns commonly seen in malicious credential-harvesting "demo" scripts
            if (/prompt\s*\(\s*(['"`]).{0,40}(pass|pwd|password|secret|credential|認証|パスワード|login|pin|暗証)/i.test(c)) return true;
            if (/confirm\s*\(\s*(['"`]).{0,40}(pass|password|削除|重要|delete all|全削除)/i.test(c)) return true;
            if (/(type\s*=\s*['"]?password|name\s*=\s*['"]?password|password.*input|input.*password|getPassword|promptForPass)/i.test(lower)) return true;
            // Prompt + data exfil heuristic (common in bad generated phishing pages)
            if (/prompt\s*\(/.test(c) && /(fetch\(|XMLHttpRequest|\.send\(|navigator\.sendBeacon|location\s*\.\s*(href|replace)|document\.cookie\s*=)/i.test(c)) return true;
            return false;
        }
        function detectBlockedScriptsInCode(code) {
            if (!code) return false;
            const html = String(code);
            // 1) External script srcs
            const srcRe = /<script\b[^>]*\bsrc\s*=\s*["']?([^"'\s>]+)/gi;
            let m;
            while ((m = srcRe.exec(html)) !== null) {
                if (isBlockedScriptSrc(m[1])) return true;
            }
            // 2) Inline <script> content that asks for passwords / credentials
            const inlineRe = /<script\b(?![^>]*\bsrc\s*=)[^>]*>([\s\S]*?)<\/script>/gi;
            while ((m = inlineRe.exec(html)) !== null) {
                if (isPasswordPromptingScript(m[1])) return true;
            }
            // 3) Direct references in strings or attributes (e.g. const u = 'https://polyfill.io/...')
            if (/["'`]https?:\/\/[^"'`\s]*polyfill\.io/i.test(html) || /src\s*=\s*["'`][^"'`]*polyfill\.io/i.test(html)) return true;
            return false;
        }
        function sanitizeHtmlForPreview(rawHtml) {
            if (!rawHtml) return '';
            const hadBlocked = detectBlockedScriptsInCode(rawHtml);
            let output = String(rawHtml);
            try {
                const parser = new DOMParser();
                const doc = parser.parseFromString(output, 'text/html');
                let modified = false;
                doc.querySelectorAll('script').forEach((scriptEl) => {
                    const src = scriptEl.getAttribute('src') || '';
                    let replaced = false;
                    if (src && isBlockedScriptSrc(src)) {
                        const warn = doc.createElement('div');
                        warn.setAttribute('data-blocked-script', 'true');
                        warn.style.cssText = 'background:#fee2e2;border:1px solid #ef4444;color:#991b1b;padding:6px 10px;border-radius:6px;font-size:12px;margin:6px 0;font-family:system-ui;';
                        const short = src.length > 70 ? src.slice(0, 67) + '...' : src;
                        warn.textContent = '⚠ ブロック済み: ' + short + ' （polyfill.io などの危険ドメインはプレビューで無効化されます）';
                        if (scriptEl.parentNode) scriptEl.parentNode.replaceChild(warn, scriptEl);
                        modified = true;
                        replaced = true;
                    } else if (!src) {
                        const inner = scriptEl.textContent || '';
                        if (isPasswordPromptingScript(inner)) {
                            const warn = doc.createElement('div');
                            warn.setAttribute('data-blocked-script', 'true');
                            warn.style.cssText = 'background:#fef3c7;border:1px solid #f59e0b;color:#92400e;padding:6px 10px;border-radius:6px;font-size:12px;margin:6px 0;font-family:system-ui;';
                            warn.textContent = '⚠ ブロック済み: パスワード入力要求などの疑わしいインラインスクリプトを無効化しました';
                            if (scriptEl.parentNode) scriptEl.parentNode.replaceChild(warn, scriptEl);
                            modified = true;
                            replaced = true;
                        }
                    }
                    // non-blocked scripts are intentionally left intact so that legitimate HTML/JS demos continue to work
                });
                // Neutralize javascript: links (defense in depth)
                doc.querySelectorAll('a[href^="javascript:" i], area[href^="javascript:" i]').forEach(a => {
                    a.setAttribute('href', '#');
                    a.setAttribute('title', (a.getAttribute('title') || '') + ' [javascript: disabled in preview]');
                });
                const head = doc.head || doc.querySelector('head');
                if (head && !head.querySelector('base')) {
                    const base = doc.createElement('base');
                    base.setAttribute('href', `${window.location.origin}/`);
                    head.insertBefore(base, head.firstChild);
                }
                if (hadBlocked || modified) {
                    const body = doc.body || doc.documentElement;
                    if (body) {
                        const banner = doc.createElement('div');
                        banner.style.cssText = 'position:sticky;top:0;left:0;right:0;z-index:2147483647;background:#7f1d1d;color:#fff;padding:8px 12px;text-align:center;font-size:12px;font-family:system-ui;border-bottom:1px solid #b91c1c;';
                        banner.innerHTML = '⚠ <strong>安全プレビュー</strong>: polyfill.io などの危険なスクリプトをブロックしています。実行は自己責任で。';
                        if (body.firstChild) {
                            body.insertBefore(banner, body.firstChild);
                        } else {
                            body.appendChild(banner);
                        }
                    }
                }
                output = '<!DOCTYPE html>\n' + (doc.documentElement ? doc.documentElement.outerHTML : output);
            } catch (err) {
                // Regex fallback: at least strip the worst external polyfill scripts
                output = output.replace(/<script\b([^>]*\bsrc\s*=\s*["']?[^"'\s>]*polyfill\.io[^"'\s>]*)["']?[^>]*>[\s\S]*?<\/script>/gi,
                    '<!-- blocked polyfill.io script for safety -->');
            }
            return output;
        }
        // === end security helpers ===

        function wrapTextWave(t) {
            if (!t) return "";
            return t.split("").map((c, i) => `<span class="wave-char" style="animation-delay: ${i * 0.028}s">${escapeHtml(c)}</span>`).join("");
        }

        /** Skeleton kind for pending answer bubble, based on model / content type */
        function getPendingSkeletonKind(modelId) {
            let m = String(modelId || '').toLowerCase();
            if (!m) {
                try { m = String((get('model-select') && get('model-select').value) || '').toLowerCase(); } catch (e) { m = ''; }
            }
            if (m.includes('video')) return 'video';
            if (
                m.includes('tts') ||
                m.includes('transcribe') ||
                m.includes('realtime') ||
                m.includes('voice') ||
                m.includes('native-audio') ||
                (m.includes('live') && m.includes('gemini'))
            ) return 'audio';
            if (
                m.includes('gpt-image') ||
                m.includes('imagine-image') ||
                m.startsWith('ideogram-') ||
                (m.includes('image') && !m.includes('vision')) ||
                (m.includes('gemini') && (m.includes('image') || m.includes('nano')))
            ) return 'image';
            if (m.includes('ocr') || m.includes('mistral-ocr')) return 'text';
            if (m.includes('build') || m.includes('code-fast') || m.includes('coding')) return 'code';
            return 'text';
        }

        function buildPendingSkeletonBody(kind) {
            if (kind === 'image') {
                return `<div class="skeleton-media skeleton-image" aria-hidden="true"><div class="skeleton-media-icon"><i class="fas fa-image"></i></div></div>`;
            }
            if (kind === 'video') {
                return `<div class="skeleton-media skeleton-video" aria-hidden="true"><div class="skeleton-media-icon"><i class="fas fa-play"></i></div><div class="skeleton-video-progress"></div></div>`;
            }
            if (kind === 'audio') {
                return `<div class="skeleton-audio" aria-hidden="true"><div class="skeleton-audio-disc"><i class="fas fa-volume-up"></i></div><div class="skeleton-wave"><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span></div></div>`;
            }
            if (kind === 'code') {
                return `<div class="skeleton-code" aria-hidden="true"><div class="skeleton-code-header"><span class="skeleton-code-dot"></span><span class="skeleton-code-dot"></span><span class="skeleton-code-dot"></span><div class="skeleton-code-title"></div></div><div class="skeleton-lines skeleton-code-lines"><div class="skeleton-line" style="width:72%"></div><div class="skeleton-line" style="width:88%"></div><div class="skeleton-line" style="width:54%"></div><div class="skeleton-line" style="width:76%"></div><div class="skeleton-line" style="width:41%"></div></div></div>`;
            }
            // text (default)
            return `<div class="skeleton-lines" aria-hidden="true"><div class="skeleton-line" style="width:92%"></div><div class="skeleton-line" style="width:78%"></div><div class="skeleton-line" style="width:86%"></div><div class="skeleton-line" style="width:64%"></div><div class="skeleton-line" style="width:48%"></div></div>`;
        }

        function buildPendingSkeletonHtml(modelId, statusText) {
            const kind = getPendingSkeletonKind(modelId);
            const status = (statusText === null || statusText === undefined || statusText === '')
                ? '回答を生成中...'
                : String(statusText);
            return `<div class="content-area pending-shimmer skeleton-pending" data-skeleton-kind="${escapeHtml(kind)}">${buildPendingSkeletonBody(kind)}<div class="skeleton-status">${escapeHtml(status)}</div></div>`;
        }

        // A monotonic timestamp is kept only while this page owns the submission.
        function startPendingElapsed(bubbleEl, startedAt) {
            if (!bubbleEl || !Number.isFinite(startedAt)) return;
            const tick = () => {
                const pending = bubbleEl.querySelector('.content-area.skeleton-pending');
                if (!bubbleEl.isConnected || !pending) { clearInterval(timer); return; }
                let elapsed = pending.querySelector('.pending-elapsed');
                if (!elapsed) {
                    elapsed = document.createElement('div');
                    elapsed.className = 'skeleton-status-sub pending-elapsed';
                    pending.appendChild(elapsed);
                }
                const seconds = Math.max(0, Math.floor((performance.now() - startedAt) / 1000));
                elapsed.textContent = `経過時間: ${Math.floor(seconds / 60)}分${seconds % 60}秒`;
            };
            const timer = setInterval(tick, 1000);
            tick();
            return () => clearInterval(timer);
        }

