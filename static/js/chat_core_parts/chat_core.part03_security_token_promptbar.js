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
            ensureMathJaxLoaded()
                .then(() => {
                    if (!window.MathJax || typeof window.MathJax.typesetPromise !== 'function') return;
                    // ストリーム再描画で DOM が差し替わったあとに古い math リストが残ると typeset が失敗するためクリア
                    try {
                        if (typeof window.MathJax.typesetClear === 'function') {
                            window.MathJax.typesetClear([container]);
                        }
                    } catch (e) {}
                    return window.MathJax.typesetPromise([container]).catch(() => {});
                })
                .catch(() => {});
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

        function updatePendingSkeletonStatus(bubbleEl, statusText, subText) {
            if (!bubbleEl) return false;
            const ca = bubbleEl.querySelector('.content-area.skeleton-pending');
            if (!ca) return false;
            let statusEl = ca.querySelector('.skeleton-status');
            if (!statusEl) {
                statusEl = document.createElement('div');
                statusEl.className = 'skeleton-status';
                ca.appendChild(statusEl);
            }
            const main = (statusText === null || statusText === undefined) ? '' : String(statusText);
            const sub = (subText === null || subText === undefined || subText === '') ? '' : String(subText);
            if (sub) {
                statusEl.innerHTML = `${escapeHtml(main)}<span class="skeleton-status-sub">${escapeHtml(sub)}</span>`;
            } else {
                statusEl.textContent = main;
            }
            return true;
        }

        /** Skeleton placeholders shown while a thread history is loading (open from history / page reload). */
        function buildChatLoadingSkeletonHtml() {
            const rows = [
                { role: 'user', widths: ['62%', '44%'] },
                { role: 'ai', widths: ['88%', '76%', '92%', '58%'] },
                { role: 'user', widths: ['48%'] },
                { role: 'ai', widths: ['82%', '70%', '54%'] }
            ];
            const items = rows.map((row, idx) => {
                const isUser = row.role === 'user';
                const align = isUser ? 'justify-end' : 'justify-start';
                const bubbleCls = isUser
                    ? 'message-bubble chat-load-skeleton-bubble chat-load-skeleton-user text-white p-4 rounded-2xl rounded-tr-none shadow-md relative'
                    : 'message-bubble chat-load-skeleton-bubble chat-load-skeleton-ai bg-gray-700 text-white p-4 rounded-2xl rounded-tl-none shadow-md relative';
                const lines = row.widths.map((w, i) =>
                    `<div class="skeleton-line" style="width:${w};animation-delay:${(idx * 0.08 + i * 0.06).toFixed(2)}s"></div>`
                ).join('');
                return `<div class="flex ${align} mb-4 chat-load-skeleton-row" style="animation-delay:${(idx * 0.07).toFixed(2)}s" aria-hidden="true"><div class="${bubbleCls}"><div class="content-area pending-shimmer skeleton-pending chat-load-skeleton-body" data-skeleton-kind="text"><div class="skeleton-lines">${lines}</div></div></div></div>`;
            }).join('');
            return `<div class="chat-load-skeleton" role="status" aria-live="polite" aria-label="チャットを読み込み中">${items}<div class="chat-load-skeleton-caption"><span class="chat-load-skeleton-caption-dot"></span>チャットを読み込み中...</div></div>`;
        }
        function showChatLoadError(threadId) {
            const container = get('chat-container');
            if (!container) return;
            container.innerHTML = '<div class="min-h-[45vh] flex items-center justify-center px-4"><div class="max-w-md w-full rounded-2xl border border-red-500/40 bg-red-950/30 p-5 text-center" role="alert"><i class="fas fa-triangle-exclamation text-red-300 text-xl mb-3"></i><p class="text-sm font-semibold text-red-100">チャットを読み込めませんでした</p><p class="mt-2 text-xs text-red-200/80">通信状態を確認して、もう一度お試しください。</p><button type="button" data-chat-load-retry class="mt-4 rounded-lg border border-red-300/40 px-4 py-2 text-sm text-red-100 hover:bg-red-500/20"><i class="fas fa-rotate-right mr-1"></i>再試行</button></div></div>';
            const retry = container.querySelector('[data-chat-load-retry]');
            if (retry) retry.addEventListener('click', () => loadMessages(threadId));
        }
        function hashString(str) {
            let h = 0;
            if (!str) return '0';
            for (let i = 0; i < str.length; i++) {
                h = ((h << 5) - h) + str.charCodeAt(i);
                h |= 0;
            }
            return Math.abs(h).toString(36);
        }
        function decodeCodeButtonValue(value) {
            if (!value) return '';
            try {
                return decodeURIComponent(value);
            } catch (e) {
                return '';
            }
        }
        function getCodingTargetFromButton(btn) {
            if (!btn) return null;
            const code = decodeCodeButtonValue(btn.getAttribute('data-code') || '');
            if (!code) return null;
            const wrapper = btn.closest('.code-wrapper');
            const group = btn.closest('.message-group');
            return {
                code,
                language: String(btn.getAttribute('data-coding-lang') || 'text').trim().slice(0, 40) || 'text',
                key: String(btn.getAttribute('data-code-key') || wrapper?.getAttribute('data-code-key') || hashString(code)),
                message_id: group?.id ? group.id.replace(/^msg-/, '') : null,
                thread_id: currentThreadId ? String(currentThreadId) : null
            };
        }
        function findLatestCodingTarget() {
            const root = get('chat-container');
            if (!root) return null;
            const buttons = Array.from(root.querySelectorAll('.message-group .coding-target-btn'));
            for (let i = buttons.length - 1; i >= 0; i--) {
                const target = getCodingTargetFromButton(buttons[i]);
                if (target) return target;
            }
            return null;
        }
        function extractPromptCodingTargets(promptText) {
            const lines = String(promptText || '').replace(/\r\n?/g, '\n').split('\n');
            const completed = [];
            let active = null;
            for (const line of lines) {
                if (!active) {
                    const opening = line.match(/^\s*(`{3,}|~{3,})(.*)$/);
                    if (!opening) continue;
                    const info = String(opening[2] || '').trim();
                    active = {
                        markerChar: opening[1][0],
                        markerLength: opening[1].length,
                        language: (info.split(/\s+/)[0] || 'text').replace(/^\{?\.?/, '').replace(/\}$/, '') || 'text',
                        buffer: []
                    };
                    continue;
                }
                const trimmed = String(line || '').trim();
                const closingPattern = new RegExp(`^\\${active.markerChar}{${active.markerLength},}\\s*$`);
                if (closingPattern.test(trimmed)) {
                    const code = active.buffer.join('\n');
                    if (code.trim()) {
                        completed.push({
                            code,
                            language: active.language,
                            key: hashString(`prompt\\n${active.language}\\n${code}`),
                            candidate_id: `prompt-${completed.length + 1}`,
                            prompt_index: completed.length,
                            message_id: null,
                            thread_id: currentThreadId ? String(currentThreadId) : null,
                            prompt_source: true
                        });
                    }
                    active = null;
                    continue;
                }
                active.buffer.push(line);
            }
            return completed;
        }
        function extractLatestPromptCodingTarget(promptText) {
            const completed = extractPromptCodingTargets(promptText);
            return completed.length ? completed[completed.length - 1] : null;
        }
        function collectCodingCandidates(promptText) {
            if (codingTargetSelection) {
                const selectedThread = codingTargetSelection.thread_id;
                if (!selectedThread || !currentThreadId || String(selectedThread) === String(currentThreadId)) {
                    return [{
                        ...codingTargetSelection,
                        candidate_id: 'selected-1',
                        source: 'history',
                        explicit: true
                    }];
                }
                codingTargetSelection = null;
            }
            const candidates = extractPromptCodingTargets(promptText);
            const seen = new Set(candidates.map(item => `${item.language}\n${item.code}`));
            const root = get('chat-container');
            const historyTargets = [];
            if (root) {
                Array.from(root.querySelectorAll('.message-group .coding-target-btn')).forEach((btn) => {
                    const target = getCodingTargetFromButton(btn);
                    if (!target) return;
                    const signature = `${target.language}\n${target.code}`;
                    if (seen.has(signature)) return;
                    seen.add(signature);
                    historyTargets.push(target);
                });
            }
            historyTargets.slice(-20).forEach((target, index) => {
                candidates.push({
                    ...target,
                    candidate_id: `history-${index + 1}`,
                    source: 'history',
                    explicit: false
                });
            });
            return candidates;
        }
        function resolveCodingTarget(promptText = null) {
            const inputText = promptText === null
                ? String(get('prompt-input')?.value || '')
                : String(promptText || '');
            if (codingTargetSelection) {
                const selectedThread = codingTargetSelection.thread_id;
                if (!selectedThread || !currentThreadId || String(selectedThread) === String(currentThreadId)) {
                    return { ...codingTargetSelection, explicit: true };
                }
                codingTargetSelection = null;
            }
            const promptTarget = extractLatestPromptCodingTarget(inputText);
            if (promptTarget) {
                return { ...promptTarget, explicit: false };
            }
            const latest = findLatestCodingTarget();
            return latest ? { ...latest, explicit: false } : null;
        }
        function syncCodingTargetButtons(root = document) {
            if (!root || typeof root.querySelectorAll !== 'function') return;
            const selectedKey = codingTargetSelection ? String(codingTargetSelection.key || '') : '';
            root.querySelectorAll('.coding-target-btn').forEach((btn) => {
                const active = !!selectedKey && String(btn.getAttribute('data-code-key') || '') === selectedKey;
                btn.classList.toggle('coding-target-active', active);
                btn.setAttribute('aria-pressed', active ? 'true' : 'false');
                btn.innerHTML = active
                    ? '<i class="fas fa-thumbtack"></i>'
                    : '<i class="fas fa-quote-right"></i>';
                btn.title = active ? '編集対象に設定済み' : 'Coding Modeの編集対象に指定';
                btn.setAttribute('aria-label', active ? '編集対象に設定済み' : '編集対象に指定');
            });
        }
        function syncCodingModeUi(enabled = codingModeEnabled, options = {}) {
            codingModeEnabled = !!enabled;
            if (options.persist !== false) {
                try {
                    localStorage.setItem(CODING_MODE_STORAGE_KEY, codingModeEnabled ? 'true' : 'false');
                } catch (e) {}
            }
            const checkbox = get('enable-coding-mode');
            if (checkbox && checkbox.checked !== codingModeEnabled) checkbox.checked = codingModeEnabled;
            const bar = get('coding-target-bar');
            const textEl = get('coding-target-text');
            const clearBtn = get('clear-coding-target-btn');
            if (bar) bar.classList.toggle('visible', codingModeEnabled);
            const target = resolveCodingTarget();
            const candidates = codingTargetSelection
                ? [target].filter(Boolean)
                : collectCodingCandidates(String(get('prompt-input')?.value || ''));
            codingModeEffective = codingModeEnabled && candidates.length > 0;
            if (textEl) {
                if (codingTargetSelection && target) {
                    textEl.textContent = `編集対象: ${target.language || 'text'} コードブロック`;
                } else if (candidates.length > 1) {
                    const promptCount = candidates.filter(item => item.prompt_source).length;
                    const historyCount = candidates.length - promptCount;
                    textEl.textContent = `モデルが編集対象を判断: 入力${promptCount}件 / 履歴${historyCount}件`;
                } else if (target && target.prompt_source) {
                    textEl.textContent = `入力中: ${target.language || 'text'} コードブロック`;
                } else if (target) {
                    textEl.textContent = `自動選択: 最新の ${target.language || 'text'} コードブロック`;
                } else {
                    textEl.textContent = 'コードブロック生成後に自動有効化';
                }
            }
            if (clearBtn) clearBtn.classList.toggle('hidden', !codingTargetSelection);
            syncCodingTargetButtons();
        }
        function activateDeferredCodingModeFromStream(markdownText) {
            if (!codingModeEnabled || codingModeEffective) return false;
            if (extractPromptCodingTargets(markdownText).length === 0) return false;
            codingModeEffective = true;
            const textEl = get('coding-target-text');
            if (textEl) textEl.textContent = 'コードブロックを検出: 次の送信から有効';
            return true;
        }
        function selectCodingTargetFromButton(btn) {
            const target = getCodingTargetFromButton(btn);
            if (!target) {
                showToast('このコードブロックを編集対象にできません', 'error', true);
                return;
            }
            codingTargetSelection = target;
            syncCodingModeUi(codingModeEnabled, { persist: false });
            if (codingModeEnabled) {
                showToast('Coding Modeの編集対象に設定しました', 'success');
            } else {
                showToast('編集対象を選択しました。プロンプトバーのCodingをオンにすると使用します', 'info');
            }
        }
        function renderCodingDiffLines(diffText) {
            return String(diffText || '').split('\n').map((line) => {
                let cls = 'coding-diff-context';
                if (line.startsWith('+++') || line.startsWith('---')) cls = 'coding-diff-file';
                else if (line.startsWith('@@')) cls = 'coding-diff-hunk';
                else if (line.startsWith('+')) cls = 'coding-diff-added';
                else if (line.startsWith('-')) cls = 'coding-diff-removed';
                return `<span class="${cls}">${escapeHtml(line || ' ')}</span>`;
            }).join('\n');
        }
        function appendCodingLiveDiff(root, payload) {
            if (!root || !payload || !payload.diff) return;
            let panel = root.querySelector('.coding-live-diff');
            if (!panel) {
                panel = document.createElement('div');
                panel.className = 'coding-live-diff';
                panel.innerHTML = '<div class="coding-live-diff-header"><span><i class="fas fa-code-branch"></i> Live Code Changes</span><span class="coding-live-diff-count">0 edits</span></div><div class="coding-live-diff-list"></div>';
                root.appendChild(panel);
            }
            const editIndex = Math.max(0, Number(payload.edit_index || 0));
            if (editIndex && panel.querySelector(`[data-coding-edit-index="${editIndex}"]`)) return;
            const list = panel.querySelector('.coding-live-diff-list');
            const edit = document.createElement('div');
            edit.className = 'coding-live-diff-edit';
            if (editIndex) edit.setAttribute('data-coding-edit-index', String(editIndex));
            const repairLabel = Number(payload.repair_attempt || 0) > 0
                ? ` · Auto repair ${Number(payload.repair_attempt)}`
                : '';
            edit.innerHTML = `<div class="coding-live-diff-meta">Edit ${editIndex} · ${escapeHtml(payload.language || 'text')}${repairLabel}</div><pre>${renderCodingDiffLines(payload.diff)}</pre>`;
            if (list) list.appendChild(edit);
            const count = panel.querySelector('.coding-live-diff-count');
            const editCount = panel.querySelectorAll('.coding-live-diff-edit').length;
            if (count) count.textContent = `${editCount} edit${editCount === 1 ? '' : 's'}`;
            panel.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
        }
        function isHtmlPreviewCandidate(lang, codeRaw) {
            const token = String(lang || '').trim().toLowerCase();
            if (token === 'html' || token === 'htm' || token === 'xhtml') return true;
            if (token) return false;
            return /<!doctype\s+html/i.test(codeRaw || '');
        }
        function openHtmlCodePreview(encodedCode) {
            if (!encodedCode) return;
            let html = '';
            try {
                html = decodeURIComponent(encodedCode);
            } catch (e) {
                showToast('HTMLプレビューの読み込みに失敗しました', 'error', true);
                return;
            }
            const hadBad = detectBlockedScriptsInCode(html);
            if (hadBad) {
                showToast('⚠ 危険な外部スクリプトを検知 (polyfill.io など)。プレビューではブロックして開きます。', 'warning', true);
            }
            const safe = sanitizeHtmlForPreview(html);
            openSandboxedHtmlTab(safe);
        }
        function snapshotCodeCollapse(container) {
            if (!container) return [];
            const states = [];
            container.querySelectorAll('.code-wrapper').forEach((wrapper, idx) => {
                const key = String(idx);
                const collapsed = wrapper.classList.contains('collapsed') || wrapper.getAttribute('data-collapsed') === 'true';
                states.push({ key, collapsed });
            });
            return states;
        }
        function applyCodeCollapse(container, states = [], defaultCollapsed = false) {
            if (!container) return;
            const map = new Map();
            states.forEach(s => map.set(s.key, s.collapsed));
            container.querySelectorAll('.code-wrapper').forEach((wrapper, idx) => {
                const key = String(idx);
                const collapsed = map.has(key) ? map.get(key) : defaultCollapsed;
                wrapper.setAttribute('data-collapsed', collapsed ? 'true' : 'false');
                wrapper.classList.toggle('collapsed', !!collapsed);
                const btn = wrapper.querySelector('.code-toggle');
                if (btn) {
                    btn.setAttribute('aria-expanded', collapsed ? 'false' : 'true');
                    btn.innerHTML = collapsed
                        ? '<i class="fas fa-chevron-down"></i>'
                        : '<i class="fas fa-chevron-up"></i>';
                    btn.title = collapsed ? '展開' : '折りたたむ';
                    btn.setAttribute('aria-label', collapsed ? '展開' : '折りたたむ');
                }
            });
        }
        function snapshotCodeCollapseByMessage(container) {
            if (!container) return new Map();
            const map = new Map();
            container.querySelectorAll('.message-group').forEach(group => {
                const msgId = group.getAttribute('id') || '';
                group.querySelectorAll('.code-wrapper').forEach((wrapper, idx) => {
                    const key = wrapper.getAttribute('data-code-key') || String(idx);
                    const collapsed = wrapper.classList.contains('collapsed') || wrapper.getAttribute('data-collapsed') === 'true';
                    map.set(`${msgId}:${key}`, collapsed);
                });
            });
            return map;
        }
        function applyCodeCollapseByMessage(container, map, defaultCollapsed = false) {
            if (!container) return;
            container.querySelectorAll('.message-group').forEach(group => {
                const msgId = group.getAttribute('id') || '';
                group.querySelectorAll('.code-wrapper').forEach((wrapper, idx) => {
                    const key = wrapper.getAttribute('data-code-key') || String(idx);
                    const stateKey = `${msgId}:${key}`;
                    const collapsed = map && map.has(stateKey) ? map.get(stateKey) : defaultCollapsed;
                    wrapper.setAttribute('data-collapsed', collapsed ? 'true' : 'false');
                    wrapper.classList.toggle('collapsed', !!collapsed);
                    const btn = wrapper.querySelector('.code-toggle');
                    if (btn) {
                        btn.setAttribute('aria-expanded', collapsed ? 'false' : 'true');
                        btn.innerHTML = collapsed
                            ? '<i class="fas fa-chevron-down"></i>'
                            : '<i class="fas fa-chevron-up"></i>';
                        btn.title = collapsed ? '展開' : '折りたたむ';
                        btn.setAttribute('aria-label', collapsed ? '展開' : '折りたたむ');
                    }
                });
            });
        }
        function buildTokenTotals(messages) {
            const totals = {
                tokens_total: 0,
                tokens_in: 0,
                tokens_out: 0,
                tokens_content: 0,
                tokens_thought: 0
            };
            let hasTotal = false;
            let hasIn = false;
            let hasOut = false;
            let hasContent = false;
            let hasThought = false;
            (messages || []).forEach(m => {
                if (!m) return;
                let rowTotal = null;
                if (m.tokens !== null && m.tokens !== undefined) {
                    rowTotal = Number(m.tokens || 0);
                } else if (
                    (m.tokens_in !== null && m.tokens_in !== undefined) ||
                    (m.tokens_out !== null && m.tokens_out !== undefined)
                ) {
                    rowTotal = Number(m.tokens_in || 0) + Number(m.tokens_out || 0);
                }
                if (rowTotal !== null) {
                    totals.tokens_total += rowTotal;
                    hasTotal = true;
                }
                if (m.tokens_in !== null && m.tokens_in !== undefined) {
                    totals.tokens_in += Number(m.tokens_in || 0);
                    hasIn = true;
                }
                if (m.tokens_out !== null && m.tokens_out !== undefined) {
                    totals.tokens_out += Number(m.tokens_out || 0);
                    hasOut = true;
                }
                if (m.tokens_content !== null && m.tokens_content !== undefined) {
                    totals.tokens_content += Number(m.tokens_content || 0);
                    hasContent = true;
                }
                if (m.tokens_thought !== null && m.tokens_thought !== undefined) {
                    totals.tokens_thought += Number(m.tokens_thought || 0);
                    hasThought = true;
                }
            });
            return {
                tokens_total: hasTotal ? totals.tokens_total : 0,
                tokens_in: hasIn ? totals.tokens_in : null,
                tokens_out: hasOut ? totals.tokens_out : null,
                tokens_content: hasContent ? totals.tokens_content : null,
                tokens_thought: hasThought ? totals.tokens_thought : null
            };
        }
        function updateTotalTokenBar(total, details = null, allBranchDetails = null) {
            const bar = get('total-token-bar');
            const count = get('total-token-count');
            const allBranchCount = get('total-token-count-all-branches');
            if (!bar || !count) return;
            const val = Number(total || 0);
            const allBranchVal = Number((allBranchDetails && allBranchDetails.tokens_total) || 0);
            if (val > 0 || allBranchVal > 0) {
                bar.classList.remove('hidden');
                count.innerText = `Total: ${val} tokens`;
                if (details) {
                    count.classList.add('cursor-pointer', 'underline', 'decoration-dotted');
                    messageMeta.__total__ = {
                        tokens_total: val,
                        tokens_in: details.tokens_in,
                        tokens_out: details.tokens_out,
                        tokens_content: details.tokens_content,
                        tokens_thought: details.tokens_thought,
                        is_encrypted: null,
                        role: 'total',
                        model: 'Conversation'
                    };
                    count.onclick = () => openTokenDetail('__total__');
                } else {
                    count.classList.remove('cursor-pointer', 'underline', 'decoration-dotted');
                    count.onclick = null;
                    delete messageMeta.__total__;
                }
                if (allBranchCount) {
                    if (allBranchDetails && allBranchVal > 0) {
                        allBranchCount.classList.remove('hidden');
                        allBranchCount.classList.add('cursor-pointer', 'underline', 'decoration-dotted');
                        allBranchCount.innerText = `All branches: ${allBranchVal} tokens`;
                        messageMeta.__total_all_branches__ = {
                            tokens_total: allBranchVal,
                            tokens_in: allBranchDetails.tokens_in,
                            tokens_out: allBranchDetails.tokens_out,
                            tokens_content: allBranchDetails.tokens_content,
                            tokens_thought: allBranchDetails.tokens_thought,
                            is_encrypted: null,
                            role: 'total',
                            model: 'Conversation (All branches)'
                        };
                        allBranchCount.onclick = () => openTokenDetail('__total_all_branches__');
                    } else {
                        allBranchCount.classList.add('hidden');
                        allBranchCount.classList.remove('cursor-pointer', 'underline', 'decoration-dotted');
                        allBranchCount.innerText = 'All branches: 0 tokens';
                        allBranchCount.onclick = null;
                        delete messageMeta.__total_all_branches__;
                    }
                }
            } else {
                bar.classList.add('hidden');
                count.innerText = 'Total: 0 tokens';
                count.classList.remove('cursor-pointer', 'underline', 'decoration-dotted');
                count.onclick = null;
                delete messageMeta.__total__;
                if (allBranchCount) {
                    allBranchCount.classList.add('hidden');
                    allBranchCount.classList.remove('cursor-pointer', 'underline', 'decoration-dotted');
                    allBranchCount.innerText = 'All branches: 0 tokens';
                    allBranchCount.onclick = null;
                }
                delete messageMeta.__total_all_branches__;
            }
        }
        const PROMPT_TOKEN_ESTIMATE_DEBOUNCE_MS = 300;
        let promptTokenEstimateTimer = null;
        let promptTokenEstimateAbort = null;
        let promptTokenEstimateSeq = 0;
        let promptTokenEstimateLastKey = '';
        let promptTokenEstimateLastData = null;
        function setPromptTokenEstimateText(text, colorClass = 'text-gray-400') {
            const el = get('prompt-token-estimate');
            if (!el) return;
            if (!text) {
                el.classList.add('hidden');
                el.innerText = '';
                return;
            }
            el.className = `mt-1 px-1 text-[10px] ${colorClass}`;
            el.classList.remove('hidden');
            el.innerText = text;
        }
        function buildPromptTokenEstimatePayload() {
            return {
                model: (get('model-select') && get('model-select').value) ? get('model-select').value : '',
                message: (get('prompt-input') && get('prompt-input').value) ? get('prompt-input').value : '',
                quote_text: currentQuote || '',
                image_urls: collectImageUrlsForSend()
            };
        }
        function renderPromptTokenEstimate(data, payload = null) {
            const p = payload || buildPromptTokenEstimatePayload();
            const hasPrompt = !!((p.message || '').trim() || (p.quote_text || '').trim());
            const hasFiles = Array.isArray(p.image_urls) && p.image_urls.length > 0;
            if (!hasPrompt && !hasFiles) {
                setPromptTokenEstimateText('');
                return;
            }
            if (data && data.pending) {
                setPromptTokenEstimateText('入力トークンを計算中...', 'text-gray-500');
                return;
            }
            if (!data) {
                setPromptTokenEstimateText('入力トークンを計算できませんでした', 'text-red-300');
                return;
            }
            if (!data.countable) {
                setPromptTokenEstimateText('このモデルは入力トークン表示対象外です', 'text-gray-500');
                return;
            }
            const total = Number(data.tokens_total || 0);
            const prompt = Number(data.tokens_prompt || 0);
            const files = Number(data.tokens_files || 0);
            const notes = [];
            if (Number(data.files_non_text || 0) > 0) notes.push(`非テキスト${data.files_non_text}件は0換算`);
            if (Number(data.files_missing || 0) > 0) notes.push(`未検出${data.files_missing}件`);
            if (Number(data.files_error || 0) > 0) notes.push(`失敗${data.files_error}件`);
            const noteText = notes.length ? ` ・ ${notes.join(' / ')}` : '';
            setPromptTokenEstimateText(`入力見積: ${total} tokens (本文 ${prompt} / ファイル ${files})${noteText}`, 'text-cyan-300');
        }
        function schedulePromptTokenEstimate(immediate = false) {
            const payload = buildPromptTokenEstimatePayload();
            const hasPrompt = !!((payload.message || '').trim() || (payload.quote_text || '').trim());
            const hasFiles = Array.isArray(payload.image_urls) && payload.image_urls.length > 0;
            if (!hasPrompt && !hasFiles) {
                promptTokenEstimateLastKey = '';
                promptTokenEstimateLastData = null;
                if (promptTokenEstimateTimer) {
                    clearTimeout(promptTokenEstimateTimer);
                    promptTokenEstimateTimer = null;
                }
                if (promptTokenEstimateAbort) {
                    promptTokenEstimateAbort.abort();
                    promptTokenEstimateAbort = null;
                }
                renderPromptTokenEstimate(null, payload);
                return;
            }
            const key = JSON.stringify([payload.model || '', payload.message || '', payload.quote_text || '', payload.image_urls || []]);
            if (key === promptTokenEstimateLastKey && promptTokenEstimateLastData) {
                renderPromptTokenEstimate(promptTokenEstimateLastData, payload);
                return;
            }
            if (promptTokenEstimateTimer) {
                clearTimeout(promptTokenEstimateTimer);
                promptTokenEstimateTimer = null;
            }
            const run = async () => {
                if (promptTokenEstimateAbort) promptTokenEstimateAbort.abort();
                promptTokenEstimateAbort = new AbortController();
                const seq = ++promptTokenEstimateSeq;
                renderPromptTokenEstimate({ pending: true }, payload);
                try {
                    const r = await apiFetch(CHAT_CONFIG.urls.estimatePromptTokensApi, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload),
                        signal: promptTokenEstimateAbort.signal
                    });
                    if (!r.ok) throw new Error(`HTTP ${r.status}`);
                    const d = await r.json();
                    if (seq !== promptTokenEstimateSeq) return;
                    promptTokenEstimateLastKey = key;
                    promptTokenEstimateLastData = d;
                    renderPromptTokenEstimate(d, payload);
                } catch (e) {
                    if (e && e.name === 'AbortError') return;
                    if (seq !== promptTokenEstimateSeq) return;
                    promptTokenEstimateLastKey = '';
                    promptTokenEstimateLastData = null;
                    renderPromptTokenEstimate(null, payload);
                }
            };
            if (immediate) run();
            else promptTokenEstimateTimer = setTimeout(run, PROMPT_TOKEN_ESTIMATE_DEBOUNCE_MS);
        }
        function updatePromptPlaceholder() {
            const input = get('prompt-input');
            if (!input) return;
            if (editingMessageId) {
                input.placeholder = "編集中... (Enter送信は設定に従います)";
            } else if (enterToSend) {
                input.placeholder = "Enter で送信 (Shift+Enter で改行)";
            } else {
                input.placeholder = "Ctrl + Enter で送信...";
            }
        }
        function readPromptBarModeFromForm() {
            if (get('set-minimal-prompt-mode') && get('set-minimal-prompt-mode').checked) {
                return { compact_prompt_mode: false, minimal_prompt_mode: true };
            }
            if (get('set-compact-prompt-mode') && get('set-compact-prompt-mode').checked) {
                return { compact_prompt_mode: true, minimal_prompt_mode: false };
            }
            return { compact_prompt_mode: false, minimal_prompt_mode: false };
        }
        function writePromptBarModeToForm(compact, minimal) {
            const normalEl = get('set-prompt-bar-mode-normal');
            const compactEl = get('set-compact-prompt-mode');
            const minimalEl = get('set-minimal-prompt-mode');
            if (minimal && minimalEl) minimalEl.checked = true;
            else if (compact && compactEl) compactEl.checked = true;
            else if (normalEl) normalEl.checked = true;
        }
        function placeModelSelectorButton() {
            const btn = get('model-selector-btn');
            const topBar = get('top-model-bar');
            const home = get('prompt-primary-controls');
            const hiddenSelect = get('model-select');
            if (!btn || !topBar || !home) return;
            if (minimalPromptMode) {
                if (btn.parentElement !== topBar) topBar.appendChild(btn);
                return;
            }
            if (hiddenSelect && hiddenSelect.parentElement === home) {
                if (btn.previousElementSibling !== hiddenSelect) {
                    hiddenSelect.insertAdjacentElement('afterend', btn);
                }
                return;
            }
            if (btn.parentElement !== home) {
                home.insertBefore(btn, home.firstChild);
            }
        }
        function applyMinimalPromptMode() {
            const enabled = !!minimalPromptMode;
            document.body.classList.toggle('minimal-prompt-mode', enabled);
            const topBar = get('top-model-bar');
            if (topBar) {
                topBar.classList.toggle('hidden', !enabled);
                topBar.classList.toggle('flex', enabled);
            }
            const uploadBtn = get('upload-btn');
            const uploadIcon = uploadBtn ? uploadBtn.querySelector('i') : null;
            if (uploadIcon) uploadIcon.className = enabled ? 'fas fa-plus' : 'fas fa-paperclip';
            if (uploadBtn) uploadBtn.title = enabled ? 'オプション' : 'Upload';
            if (!enabled) {
                closeMinimalOptions();
                hideThinkingSlider();
            }
            placeModelSelectorButton();
        }
        function applyPromptControlMode() {
            const details = get('prompt-details-controls');
            const toggleBtn = get('prompt-controls-toggle-btn');
            const toggleText = get('prompt-controls-toggle-text');
            const toggleIcon = get('prompt-controls-toggle-icon');
            const row = get('prompt-controls-row');
            applyMinimalPromptMode();
            if (!details || !toggleBtn) return;
            const compactActive = compactPromptMode && !minimalPromptMode;
            const showDetails = !compactActive || promptControlsExpanded;
            if (row) row.classList.toggle('compact-collapsed', compactActive && !showDetails);

            if (compactActive) {
                if (showDetails) {
                    details.classList.remove('collapsed');
                    details.classList.add('expanded');
                    details.classList.remove('hidden');
                } else {
                    details.classList.remove('expanded');
                    details.classList.add('collapsed');
                }
            } else {
                details.classList.remove('hidden');
                details.classList.remove('collapsed');
                details.classList.remove('expanded');
            }

            if (compactActive) {
                toggleBtn.classList.remove('hidden');
                toggleBtn.classList.add('inline-flex');
                toggleBtn.setAttribute('aria-expanded', showDetails ? 'true' : 'false');
                if (toggleText) toggleText.textContent = showDetails ? '折りたたむ' : '詳細';
                if (toggleIcon) toggleIcon.className = showDetails ? 'fas fa-chevron-up text-[10px]' : 'fas fa-chevron-down text-[10px]';
            } else {
                toggleBtn.classList.add('hidden');
                toggleBtn.classList.remove('inline-flex');
                toggleBtn.setAttribute('aria-expanded', 'true');
                if (toggleText) toggleText.textContent = '詳細';
                if (toggleIcon) toggleIcon.className = 'fas fa-chevron-down text-[10px]';
            }
        }
        function setCompactPromptMode(enabled, keepExpanded = false) {
            compactPromptMode = !!enabled;
            if (compactPromptMode) minimalPromptMode = false;
            if (!compactPromptMode) {
                promptControlsExpanded = true;
            } else if (!keepExpanded) {
                promptControlsExpanded = false;
            }
            applyPromptControlMode();
        }
        function setMinimalPromptMode(enabled) {
            minimalPromptMode = !!enabled;
            if (minimalPromptMode) {
                compactPromptMode = false;
                promptControlsExpanded = false;
            }
            applyPromptControlMode();
        }
        function togglePromptControlDetails() {
            if (!compactPromptMode) return;
            promptControlsExpanded = !promptControlsExpanded;
            applyPromptControlMode();
        }

        // ------------------------------------------------------------------
        // Minimal mode: plus-button options popup + temporary Thinking slider
        // ------------------------------------------------------------------
        const MINIMAL_MODEL_PANEL_IDS = [
            'gpt-image-options', 'gemini-image-options', 'grok-image-options',
            'xai-chat-options', 'grok-video-options', 'mistral-ocr-options', 'image-input-limits', 'audio-gen-options'
        ];
        const THINKING_LEVELS = [
            { value: 'minimal', label: 'Min' },
            { value: 'low', label: 'Low' },
            { value: 'medium', label: 'Mid' },
            { value: 'high', label: 'High' }
        ];
        const MINIMAL_POPUP_ITEMS = [
            { key: 'attach', icon: 'fa-paperclip', label: 'ファイルを添付', action: 'upload' },
            { key: 'canvas', icon: 'fa-window-restore', label: 'Canvas', checkboxId: 'enable-canvas-mode', containerId: 'canvas-mode-container' },
            { key: 'coding', icon: 'fa-code-branch', label: 'Coding', checkboxId: 'enable-coding-mode', containerId: 'coding-mode-container' },
            { key: 'fast', icon: 'fa-bolt', label: '高速', checkboxId: 'enable-browser-fast-mode', containerId: 'browser-fast-mode-container' },
            { key: 'search', icon: 'fa-search', label: 'Search', checkboxId: 'enable-search', containerId: 'search-container' },
            { key: 'urls', icon: 'fa-link', label: 'URLs', checkboxId: 'enable-url-context', containerId: 'url-context-container' },
            { key: 'maps', icon: 'fa-map-location-dot', label: 'Maps', checkboxId: 'enable-maps', containerId: 'maps-grounding-container' },
            { key: 'python', icon: 'fa-code', label: 'Python', checkboxId: 'enable-python', containerId: 'python-container' },
            { key: 'file', icon: 'fa-file-lines', label: 'File', checkboxId: 'enable-file-creation', containerId: 'file-creation-container' },
            { key: 'mcp', icon: 'fa-plug', label: 'MCP', checkboxId: 'enable-mcp', containerId: 'mcp-container' },
            { key: 'sysprompt', icon: 'fa-terminal', label: 'SysPrompt', checkboxId: 'enable-sys-prompt', containerId: 'sys-prompt-option', gear: true, gearAction: () => { if (window.openThreadModal) window.openThreadModal(); } },
            { key: 'thinking', icon: 'fa-brain', label: 'Thinking', checkboxId: 'enable-thinking', containerId: 'thinking-options', special: 'thinking' },
            { key: 'effort', icon: 'fa-sliders-h', label: 'Effort', containerId: 'reasoning-effort-container', selectId: 'reasoning-effort' },
            { key: 'safety', icon: 'fa-shield-halved', label: 'Safety', selectId: 'safety-setting' },
            { key: 'promptcache', icon: 'fa-database', label: 'PromptCache', checkboxId: 'enable-prompt-cache', containerId: 'prompt-cache-container' },
            { key: 'compress', icon: 'fa-compress-alt', label: 'Compress', checkboxId: 'enable-compression', containerId: 'compression-option', gear: true, gearAction: () => { if (window.openCompressionModal) window.openCompressionModal(); } },
            { key: 'tempchat', icon: 'fa-hourglass-half', label: '一時チャット', checkboxId: 'enable-temporary-chat', containerId: 'temporary-chat-container', gear: true, gearAction: () => openTemporaryChatSettings() }
        ];
        let minimalOptionsOpen = false;
        let thinkingSliderOpen = false;
        let thinkingSliderTimer = null;
        let thinkingSliderStartY = 0;
        let thinkingSliderStartX = 0;
        let thinkingSliderDragging = false;
        let thinkingSliderAxis = null;
        let popupSwipeStartY = 0;
        let popupSwipeStartX = 0;
        let popupSwipeDragging = false;
        let popupSwipeAtTop = false;
        let popupSwipeAxis = null;
        const minimalPanelOrigins = new Map();

        function minimalOptionVisible(item) {
            if (item.containerId) {
                const cont = get(item.containerId);
                if (!cont || cont.classList.contains('hidden')) return false;
            }
            return true;
        }
        function minimalOptionDisabled(item) {
            if (item.special === 'thinking') {
                // The Thinking row is handled specially: a disabled checkbox only
                // means thinking is forced on for the current model (e.g. Gemini 3.x),
                // so the row must stay tappable to open the level slider.
                const cont = get(item.containerId);
                return !!(cont && cont.classList.contains('pointer-events-none'));
            }
            if (item.checkboxId) {
                const chk = get(item.checkboxId);
                if (chk && chk.disabled) return true;
            }
            if (item.containerId) {
                const cont = get(item.containerId);
                if (cont && cont.classList.contains('pointer-events-none')) return true;
            }
            return false;
        }
        function minimalOptionChecked(item) {
            if (!item.checkboxId) return false;
            const chk = get(item.checkboxId);
            return !!chk && chk.checked;
        }

        function buildMinimalOptionItem(item) {
            const row = document.createElement('div');
            row.className = 'minimal-option-item';
            row.dataset.key = item.key;
            if (item.action) row.classList.add('action-' + item.action);
            if (minimalOptionChecked(item)) row.classList.add('on');
            else row.classList.add('off');
            if (minimalOptionDisabled(item)) row.classList.add('disabled');
            const icon = document.createElement('i');
            icon.className = 'fas ' + item.icon + ' minimal-option-icon';
            row.appendChild(icon);
            const label = document.createElement('span');
            label.className = 'minimal-option-label';
            label.textContent = item.label;
            row.appendChild(label);
            if (item.selectId) {
                const src = get(item.selectId);
                if (src) {
                    const clone = src.cloneNode(true);
                    clone.removeAttribute('id');
                    clone.className = 'minimal-option-select';
                    clone.addEventListener('change', () => {
                        src.value = clone.value;
                        src.dispatchEvent(new Event('change', { bubbles: true }));
                        refreshMinimalOptionItems();
                    });
                    row.appendChild(clone);
                }
            }
            if (item.gear) {
                const gear = document.createElement('button');
                gear.type = 'button';
                gear.className = 'minimal-option-gear';
                gear.title = item.label + '設定';
                const gicon = document.createElement('i');
                gicon.className = 'fas fa-cog';
                gear.appendChild(gicon);
                gear.addEventListener('click', (e) => {
                    e.stopPropagation();
                    closeMinimalOptions();
                    if (typeof item.gearAction === 'function') item.gearAction();
                });
                row.appendChild(gear);
            }
            row.addEventListener('click', () => handleMinimalOptionClick(item));
            return row;
        }

        function renderMinimalOptionItems() {
            const wrap = get('minimal-options-items');
            if (!wrap) return;
            const frag = document.createDocumentFragment();
            MINIMAL_POPUP_ITEMS.forEach((item) => {
                if (!minimalOptionVisible(item)) return;
                frag.appendChild(buildMinimalOptionItem(item));
            });
            wrap.innerHTML = '';
            wrap.appendChild(frag);
        }

        function refreshMinimalOptionItems() {
            const wrap = get('minimal-options-items');
            if (!wrap || !minimalOptionsOpen) return;
            const rows = wrap.querySelectorAll('.minimal-option-item');
            const rowByKey = {};
            rows.forEach((row) => { rowByKey[row.dataset.key] = row; });
            MINIMAL_POPUP_ITEMS.forEach((item) => {
                const row = rowByKey[item.key];
                if (!row) return;
                if (!minimalOptionVisible(item)) { row.classList.add('hidden'); return; }
                row.classList.remove('hidden');
                row.classList.toggle('on', minimalOptionChecked(item));
                row.classList.toggle('off', !minimalOptionChecked(item));
                row.classList.toggle('disabled', minimalOptionDisabled(item));
                if (item.selectId) {
                    const src = get(item.selectId);
                    const clone = row.querySelector('.minimal-option-select');
                    if (src && clone && document.activeElement !== clone && clone.value !== src.value) {
                        clone.value = src.value;
                    }
                }
            });
        }

        function handleMinimalOptionClick(item) {
            if (item.action === 'upload') {
                closeMinimalOptions();
                openUploadModal();
                return;
            }
            if (item.special === 'thinking') {
                // Thinking needs special handling: for models where thinking is
                // forced on (e.g. Gemini 3.x) the checkbox is disabled, but the
                // row must still open the level slider.
                const chk = get(item.checkboxId);
                if (chk && !chk.disabled) {
                    const turningOn = !chk.checked;
                    chk.checked = turningOn;
                    chk.dispatchEvent(new Event('change', { bubbles: true }));
                    if (turningOn) {
                        closeMinimalOptions();
                        showThinkingSlider();
                    } else {
                        hideThinkingSlider();
                    }
                    refreshMinimalOptionItems();
                } else {
                    // Thinking is forced on (or the checkbox is unavailable):
                    // open the slider so the level can still be adjusted.
                    closeMinimalOptions();
                    showThinkingSlider();
                }
                return;
            }
            if (minimalOptionDisabled(item)) return;
            if (item.selectId) return; // select rows change via their own <select>
            const chk = get(item.checkboxId);
            if (!chk) return;
            if (chk.disabled) return;
            chk.checked = !chk.checked;
            chk.dispatchEvent(new Event('change', { bubbles: true }));
            refreshMinimalOptionItems();
            // Fast mode may open a modal and settles asynchronously; temporary
            // chat confirmation is async too, so refresh the rows afterwards.
            if (item.key === 'fast') {
                closeMinimalOptions();
                setTimeout(() => refreshMinimalOptionItems(), 350);
            } else if (item.key === 'tempchat') {
                setTimeout(() => refreshMinimalOptionItems(), 350);
            }
        }

        function moveModelPanelsIntoPopup() {
            const body = get('minimal-options-model-body');
            if (!body) return;
            let anyVisible = false;
            MINIMAL_MODEL_PANEL_IDS.forEach((id) => {
                const el = get(id);
                if (!el) return;
                if (el.parentElement === body) {
                    if (!el.classList.contains('hidden')) anyVisible = true;
                    return;
                }
                if (minimalPanelOrigins.has(el)) return;
                minimalPanelOrigins.set(el, { parent: el.parentElement, next: el.nextSibling });
                body.appendChild(el);
                if (!el.classList.contains('hidden')) anyVisible = true;
            });
            refreshMinimalModelSection();
        }
        function restoreModelPanelsFromPopup() {
            const body = get('minimal-options-model-body');
            if (!body) return;
            minimalPanelOrigins.forEach((origin, el) => {
                if (origin.parent && origin.parent.contains(el)) {
                    if (origin.next && origin.next.parentNode === origin.parent) origin.parent.insertBefore(el, origin.next);
                    else origin.parent.appendChild(el);
                }
            });
            minimalPanelOrigins.clear();
        }
        function refreshMinimalModelSection() {
            const body = get('minimal-options-model-body');
            const section = get('minimal-options-model-section');
            if (!body || !section) return;
            let anyVisible = false;
            Array.from(body.children).forEach((child) => {
                if (!child.classList.contains('hidden')) anyVisible = true;
            });
            section.classList.toggle('hidden', !anyVisible);
        }

        function openMinimalOptions() {
            if (minimalOptionsOpen || !minimalPromptMode) return;
            hideThinkingSlider();
            minimalOptionsOpen = true;
            renderMinimalOptionItems();
            moveModelPanelsIntoPopup();
            const popup = get('minimal-options-popup');
            if (!popup) return;
            // Clear any inline transform/opacity left by a swipe-to-close so a
            // quick reopen starts from the clean open state.
            const panel = get('minimal-options-panel');
            if (panel) {
                panel.style.transform = '';
                panel.style.opacity = '';
            }
            popup.classList.remove('hidden');
            popup.setAttribute('aria-hidden', 'false');
            void popup.offsetWidth;
            popup.classList.add('minimal-options-open');
        }
        function closeMinimalOptions() {
            if (!minimalOptionsOpen) return;
            minimalOptionsOpen = false;
            const popup = get('minimal-options-popup');
            if (popup) {
                popup.classList.remove('minimal-options-open');
                popup.setAttribute('aria-hidden', 'true');
                setTimeout(() => {
                    if (!minimalOptionsOpen) popup.classList.add('hidden');
                }, 320);
            }
            restoreModelPanelsFromPopup();
            hideThinkingSlider();
        }
        function toggleMinimalOptions() {
            if (minimalOptionsOpen) closeMinimalOptions();
            else openMinimalOptions();
        }
        function refreshMinimalOptionsIfOpen() {
            if (!minimalOptionsOpen) return;
            renderMinimalOptionItems();
            refreshMinimalModelSection();
        }

        // --- Thinking level slide bar ---
        function allowedThinkingValues() {
            const sel = get('thinking-level');
            if (!sel) return THINKING_LEVELS.map((l) => l.value);
            const allowed = Array.from(sel.options)
                .filter((o) => !o.disabled && !o.classList.contains('hidden'))
                .map((o) => o.value);
            // All options disabled (e.g. Claude uses budget only) -> nothing to pick.
            return allowed;
        }
        function thinkingIndexFromValue(value) {
            const idx = THINKING_LEVELS.findIndex((l) => l.value === value);
            return idx < 0 ? 3 : idx;
        }
        function syncThinkingSliderUi() {
            const slider = get('thinking-slider');
            const label = get('thinking-slide-value');
            const sel = get('thinking-level');
            const idx = thinkingIndexFromValue(sel ? sel.value : 'high');
            if (slider) slider.value = String(idx);
            if (label) label.textContent = THINKING_LEVELS[idx].label;
        }
        function scheduleThinkingSliderHide() {
            if (thinkingSliderTimer) clearTimeout(thinkingSliderTimer);
            thinkingSliderTimer = setTimeout(() => {
                thinkingSliderTimer = null;
                hideThinkingSlider();
            }, 2500);
        }
        function showThinkingSlider() {
            if (thinkingSliderOpen) {
                scheduleThinkingSliderHide();
                return;
            }
            const bar = get('thinking-slide-bar');
            if (!bar) return;
            // Clear any leftover drag transform from a previous swipe-close so
            // the slider always reappears in the open position.
            const inner = get('thinking-slide-inner');
            if (inner) inner.style.transform = '';
            thinkingSliderOpen = true;
            bar.classList.remove('hidden');
            bar.setAttribute('aria-hidden', 'false');
            syncThinkingSliderUi();
            void bar.offsetWidth;
            bar.classList.add('thinking-slide-open');
            scheduleThinkingSliderHide();
        }
        function hideThinkingSlider() {
            if (thinkingSliderTimer) { clearTimeout(thinkingSliderTimer); thinkingSliderTimer = null; }
            const bar = get('thinking-slide-bar');
            if (!bar) return;
            thinkingSliderOpen = false;
            bar.classList.remove('thinking-slide-open');
            bar.setAttribute('aria-hidden', 'true');
            setTimeout(() => {
                if (!thinkingSliderOpen) bar.classList.add('hidden');
                // Clear any leftover drag transform only after the close
                // transition has finished, so a swipe-to-close fades out from
                // the released position instead of bouncing back to translateY(0).
                const inner = get('thinking-slide-inner');
                if (inner) inner.style.transform = '';
            }, 360);
        }
