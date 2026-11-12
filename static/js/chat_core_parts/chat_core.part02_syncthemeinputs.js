        const syncThemeInputs = (value) => {
            const hex = normalizeHex(value) || THEME_DEFAULT;
            const colorInput = get('set-theme-color');
            const textInput = get('set-theme-color-text');
            if (colorInput) colorInput.value = hex;
            if (textInput) textInput.value = hex;
            const swatches = document.querySelectorAll('#theme-presets .theme-swatch');
            swatches.forEach((btn) => {
                const c = normalizeHex(btn.getAttribute('data-color'));
                btn.classList.toggle('active', c === hex);
            });
        };
        const initThemeFromServer = () => {
            if (INITIAL_LIGHT_MODE_ENABLED) applyLightMode(true);
            const serverTheme = normalizeHex(INITIAL_THEME_COLOR);
            if (serverTheme) {
                applyThemeColor(serverTheme, false);
                return;
            }
            const stored = normalizeHex(localStorage.getItem(THEME_STORAGE_KEY));
            if (stored) {
                applyThemeColor(stored, false);
            } else {
                applyThemeColor(THEME_DEFAULT, false);
            }
        };
        const LIQUID_GLASS_SURFACE_SELECTOR = [
            '#sidebar',
            '.composer-dock',
            'body > .flex-1 > header',
            '#top-model-bar',
            '.modal-panel',
            '.modal-glass-panel',
            '.viewer-toolbar',
            '.viewer-meta',
            '#quote-bar',
            '#slash-command-suggestions',
            '#skill-suggestions',
            '#total-token-bar'
        ].join(',');
        const refreshLiquidGlassSurfaces = () => {
            document.querySelectorAll(LIQUID_GLASS_SURFACE_SELECTOR).forEach((element) => {
                element.classList.add('liquid-glass-surface');
                if (element.matches('.viewer-toolbar, .viewer-meta')) {
                    element.classList.add('liquid-glass-clear');
                }
                const hasNoBackground = element.matches('[data-liquid-glass-background="none"]')
                    || !!element.closest('.liquid-glass-no-backdrop');
                element.classList.toggle('liquid-glass-no-background', hasNoBackground);
            });
        };
        const applyLiquidGlassMode = (enabled) => {
            if (!document.body) return;
            document.body.classList.toggle('liquid-glass-mode', !!enabled);
            if (enabled) refreshLiquidGlassSurfaces();
        };
        let pendingLiquidGlassPointer = null;
        let liquidGlassPointerFrame = 0;
        let liquidGlassPointerPaintAt = 0;
        let liquidGlassPointerSurface = null;
        let liquidGlassPointerRect = null;
        const paintLiquidGlassPointer = (timestamp) => {
            if (!pendingLiquidGlassPointer || !document.body || !document.body.classList.contains('liquid-glass-mode')) {
                liquidGlassPointerFrame = 0;
                return;
            }
            // Expensive translucent surfaces do not benefit from repainting faster than ~30fps.
            if (timestamp - liquidGlassPointerPaintAt < 30) {
                liquidGlassPointerFrame = requestAnimationFrame(paintLiquidGlassPointer);
                return;
            }
            const pointer = pendingLiquidGlassPointer;
            pendingLiquidGlassPointer = null;
            const surface = pointer.target && pointer.target.closest
                ? pointer.target.closest(LIQUID_GLASS_SURFACE_SELECTOR)
                : null;
            if (!surface) {
                liquidGlassPointerFrame = 0;
                return;
            }
            if (surface !== liquidGlassPointerSurface || !liquidGlassPointerRect) {
                liquidGlassPointerSurface = surface;
                liquidGlassPointerRect = surface.getBoundingClientRect();
            }
            const rect = liquidGlassPointerRect;
            if (rect.width && rect.height) {
                const x = Math.max(0, Math.min(100, ((pointer.clientX - rect.left) / rect.width) * 100));
                const y = Math.max(0, Math.min(100, ((pointer.clientY - rect.top) / rect.height) * 100));
                surface.style.setProperty('--glass-light-x', `${x.toFixed(1)}%`);
                surface.style.setProperty('--glass-light-y', `${y.toFixed(1)}%`);
                liquidGlassPointerPaintAt = timestamp;
            }
            liquidGlassPointerFrame = pendingLiquidGlassPointer
                ? requestAnimationFrame(paintLiquidGlassPointer)
                : 0;
        };
        document.addEventListener('pointermove', (event) => {
            if (!document.body || !document.body.classList.contains('liquid-glass-mode')) return;
            pendingLiquidGlassPointer = {
                target: event.target,
                clientX: event.clientX,
                clientY: event.clientY,
            };
            if (!liquidGlassPointerFrame) {
                liquidGlassPointerFrame = requestAnimationFrame(paintLiquidGlassPointer);
            }
        }, { passive: true });
        document.addEventListener('pointerout', (event) => {
            const surface = event.target.closest ? event.target.closest(LIQUID_GLASS_SURFACE_SELECTOR) : null;
            if (!surface || (event.relatedTarget && surface.contains(event.relatedTarget))) return;
            pendingLiquidGlassPointer = null;
            surface.style.removeProperty('--glass-light-x');
            surface.style.removeProperty('--glass-light-y');
            surface.classList.remove('liquid-glass-pressed');
            if (surface === liquidGlassPointerSurface) {
                liquidGlassPointerSurface = null;
                liquidGlassPointerRect = null;
            }
        }, { passive: true });
        document.addEventListener('pointerdown', (event) => {
            if (!document.body || !document.body.classList.contains('liquid-glass-mode')) return;
            const surface = event.target.closest ? event.target.closest(LIQUID_GLASS_SURFACE_SELECTOR) : null;
            if (surface) surface.classList.add('liquid-glass-pressed');
        }, { passive: true });
        const releaseLiquidGlassPress = (event) => {
            const surface = event.target.closest ? event.target.closest(LIQUID_GLASS_SURFACE_SELECTOR) : null;
            if (surface) surface.classList.remove('liquid-glass-pressed');
        };
        document.addEventListener('pointerup', releaseLiquidGlassPress, { passive: true });
        document.addEventListener('pointercancel', releaseLiquidGlassPress, { passive: true });
        let liquidGlassScrollTimer = 0;
        document.addEventListener('scroll', () => {
            if (!document.body || !document.body.classList.contains('liquid-glass-mode')) return;
            liquidGlassPointerRect = null;
            document.body.classList.add('liquid-glass-scrolling');
            window.clearTimeout(liquidGlassScrollTimer);
            liquidGlassScrollTimer = window.setTimeout(() => {
                if (document.body) document.body.classList.remove('liquid-glass-scrolling');
            }, 140);
        }, { passive: true, capture: true });
        window.addEventListener('resize', () => {
            liquidGlassPointerRect = null;
        }, { passive: true });
        const MODAL_ANIM_MS = 280;
        const formatBytes = (bytes) => {
            if (bytes === null || bytes === undefined) return '0MB';
            const mb = bytes / (1024 * 1024);
            if (mb < 1024) return `${mb.toFixed(1)}MB`;
            const gb = mb / 1024;
            return `${gb.toFixed(2)}GB`;
        };
        const inspectSiteCacheStorage = async () => {
            const summary = {
                cacheCount: 0,
                entryCount: 0,
                totalBytes: 0,
                storageUsageBytes: null,
                storageQuotaBytes: null,
            };
            if ('caches' in window) {
                try {
                    const names = await caches.keys();
                    summary.cacheCount = names.length;
                    for (const name of names) {
                        const cache = await caches.open(name);
                        const requests = await cache.keys();
                        summary.entryCount += requests.length;
                        for (const request of requests) {
                            try {
                                const response = await cache.match(request);
                                if (!response) continue;
                                const contentLength = parseInt(response.headers.get('content-length') || '', 10);
                                if (Number.isFinite(contentLength) && contentLength >= 0) {
                                    summary.totalBytes += contentLength;
                                } else {
                                    const blob = await response.clone().blob();
                                    summary.totalBytes += blob.size || 0;
                                }
                            } catch (err) {}
                        }
                    }
                } catch (err) {}
            }
            if (navigator.storage && navigator.storage.estimate) {
                try {
                    const estimate = await navigator.storage.estimate();
                    summary.storageUsageBytes = Number(estimate.usage || 0);
                    summary.storageQuotaBytes = Number(estimate.quota || 0);
                } catch (err) {}
            }
            return summary;
        };
        const loadSiteCacheUsage = async () => {
            const textEl = get('site-cache-usage-text');
            const detailEl = get('site-cache-usage-detail');
            if (!textEl && !detailEl) return;
            if (textEl) textEl.innerText = '読み込み中...';
            if (detailEl) detailEl.innerText = '';
            try {
                const summary = await inspectSiteCacheStorage();
                const main = `キャッシュ使用量: ${formatBytes(summary.totalBytes)} (${summary.cacheCount}キャッシュ / ${summary.entryCount}件)`;
                if (summary.storageQuotaBytes) {
                    const quotaPct = Math.min(100, Math.round((summary.totalBytes / summary.storageQuotaBytes) * 100));
                    if (textEl) textEl.innerText = `${main} / 保存領域上限 ${formatBytes(summary.storageQuotaBytes)} (${quotaPct}%)`;
                    if (detailEl) {
                        const usageLine = summary.storageUsageBytes !== null
                            ? `保存領域使用量: ${formatBytes(summary.storageUsageBytes)}`
                            : '保存領域使用量: 取得できませんでした';
                        detailEl.innerText = `${usageLine} / ブラウザの実測値です`;
                    }
                } else {
                    if (textEl) textEl.innerText = main;
                    if (detailEl) {
                        detailEl.innerText = summary.storageUsageBytes !== null
                            ? `保存領域使用量: ${formatBytes(summary.storageUsageBytes)}`
                            : '保存領域上限はこのブラウザでは取得できません';
                    }
                }
            } catch (e) {
                if (textEl) textEl.innerText = 'キャッシュ容量の取得に失敗しました';
                if (detailEl) detailEl.innerText = '';
            }
        };
        let versionUpdateCachePreferenceSavePromise = Promise.resolve();
        const loadStorageUsage = async () => {
            const textEl = get('storage-usage-text');
            const barEl = get('storage-usage-bar');
            if (!textEl || !barEl) return;
            textEl.innerText = '読み込み中...';
            try {
                const r = await apiFetch('/api/storage', { cache: 'no-store' });
                if (!r.ok) throw new Error('HTTP ' + r.status);
                const d = await r.json();
                const used = Number(d.used_bytes || 0);
                const limit = Number(d.limit_bytes || 0);
                if (d.is_unlimited || !limit) {
                    textEl.innerText = `使用量: ${formatBytes(used)} (無制限)`;
                    barEl.style.width = '0%';
                    barEl.style.opacity = '0.5';
                } else {
                    const pct = Math.min(100, Math.round((used / limit) * 100));
                    textEl.innerText = `使用量: ${formatBytes(used)} / ${formatBytes(limit)} (${pct}%)`;
                    barEl.style.width = `${pct}%`;
                    barEl.style.opacity = '1';
                }
            } catch (e) {
                textEl.innerText = '読み込みに失敗しました';
                barEl.style.width = '0%';
                barEl.style.opacity = '0.5';
            }
        };
        const clearSiteCacheAndReload = async (triggerEl, options = {}) => {
            const { scanFirst = true } = options || {};
            const oldLabel = triggerEl ? triggerEl.innerText : '';
            if (triggerEl) {
                triggerEl.disabled = true;
                triggerEl.innerText = '削除中...';
            }
            try {
                const summary = scanFirst ? await inspectSiteCacheStorage() : null;
                await purgeCaches();
                const cacheMsg = summary ? `ローカルキャッシュ ${formatBytes(summary.totalBytes)} を削除しました。` : 'ローカルキャッシュを削除しました。';
                showToast(`${cacheMsg} 再読み込みします。`, 'success');
                window.setTimeout(() => location.reload(), 900);
            } catch (e) {
                showToast('ローカルキャッシュの削除に失敗しました', 'error', true);
            } finally {
                if (triggerEl) {
                    triggerEl.disabled = false;
                    triggerEl.innerText = oldLabel || 'サイトキャッシュを削除';
                }
            }
        };
        const syncVersionUpdateCachePreferenceUi = () => {
            const el = get('version-update-clear-cache');
            if (!el) return;
            el.checked = !!(window.CHAT_CONFIG && window.CHAT_CONFIG.clearCacheOnVersionUpdate);
        };
        const saveVersionUpdateCachePreference = async (enabled) => {
            if (window.CHAT_CONFIG) {
                window.CHAT_CONFIG.clearCacheOnVersionUpdate = !!enabled;
            }
            try {
                await apiFetch(CHAT_CONFIG.urls.handleSettings, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ clear_cache_on_version_update: !!enabled })
                });
            } catch (e) {}
        };
        initThemeFromServer();
        applyLiquidGlassMode(INITIAL_LIQUID_GLASS_ENABLED);
        measureInteractionFrames(true);
        const modalCloseTimers = new WeakMap();
        const modalOpenFrames = new WeakMap();
        const cancelModalTransitions = (el) => {
            const closeTimer = modalCloseTimers.get(el);
            if (closeTimer) {
                clearTimeout(closeTimer);
                modalCloseTimers.delete(el);
            }
            const frames = modalOpenFrames.get(el);
            if (frames) {
                cancelAnimationFrame(frames.first);
                if (frames.second) cancelAnimationFrame(frames.second);
                modalOpenFrames.delete(el);
            }
        };
        const showModal = (id) => {
            const el = get(id);
            if (!el) return;
            if (el.classList.contains('modal-open')) return;
            const quotePopover = get('quote-popover');
            if (quotePopover) {
                quotePopover.style.display = 'none';
                quotePopover.classList.remove('show');
            }
            cancelModalTransitions(el);
            el.classList.remove('hidden');
            el.style.display = 'flex';
            el.classList.remove('modal-close');
            el.classList.remove('modal-open');
            el.classList.add('modal-prep');
            // Two compositor frames establish the start state without a synchronous layout flush.
            const frames = { first: 0, second: 0 };
            frames.first = requestAnimationFrame(() => {
                frames.second = requestAnimationFrame(() => {
                    modalOpenFrames.delete(el);
                    el.classList.remove('modal-prep');
                    el.classList.add('modal-open');
                });
            });
            modalOpenFrames.set(el, frames);
        };
        window.showModal = showModal;
        const hideModal = (id, options = {}) => {
            if (id === 'token-detail-modal' && typeof cancelPromptTokenDetail === 'function') {
                cancelPromptTokenDetail();
                if (get('token-detail-estimate') && !get('token-detail-estimate').classList.contains('hidden')) get('prompt-token-estimate')?.focus();
            }
            const el = get(id);
            if (!el) return;
            cancelModalTransitions(el);
            const skipConfirm = !!(options && options.skipConfirm);
            const skipReset = !!(options && options.skipReset);
            if (id === 'camera-capture-modal' && cameraCapturePendingFiles.length > 0) {
                if (!skipConfirm && !cameraCaptureBusy) {
                    // 自動的に添付を開始
                    attachCameraCapturedFiles();
                    return;
                }
            }
            if (id === 'rich-paste-modal') {
                if (!skipConfirm && hasRichPasteContent() && !confirm('貼り付けた内容を破棄して閉じますか？')) {
                    return;
                }
            }
            if (id === 'marker-modal') {
                markerState.row = null;
            }
            if (id === 'camera-capture-modal') {
                if (!skipReset) {
                    resetCameraCapturePending();
                }
                stopCameraCaptureStream();
            }
            if (!el.classList.contains('modal-open')) {
                el.style.display = 'none';
                el.classList.remove('modal-close');
                el.classList.remove('modal-prep');
                el.classList.add('hidden');
                return;
            }
            el.classList.remove('modal-open');
            el.classList.add('modal-close');
            const closeTimer = setTimeout(() => {
                el.style.display = 'none';
                el.classList.remove('modal-close');
                el.classList.remove('modal-prep');
                el.classList.add('hidden');
                modalCloseTimers.delete(el);
            }, MODAL_ANIM_MS);
            modalCloseTimers.set(el, closeTimer);
        };
        window.hideModal = hideModal;
        const RICH_PASTE_ALLOWED_TAGS = [
            'a', 'abbr', 'address', 'article', 'b', 'blockquote', 'br', 'caption', 'cite', 'code',
            'col', 'colgroup', 'dd', 'del', 'details', 'div', 'dl', 'dt', 'em', 'figcaption',
            'figure', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'hr', 'i', 'img', 'kbd', 'li', 'main',
            'mark', 'ol', 'p', 'pre', 'q', 's', 'samp', 'section', 'small', 'span', 'strong',
            'sub', 'summary', 'sup', 'table', 'tbody', 'td', 'th', 'thead', 'tfoot', 'time', 'tr',
            'u', 'ul', 'var'
        ];
        const RICH_PASTE_ALLOWED_ATTR = [
            'align', 'alt', 'cellpadding', 'cellspacing', 'class', 'colspan', 'datetime', 'dir',
            'headers', 'height', 'href', 'lang', 'open', 'rel', 'reversed', 'rowspan', 'scope',
            'src', 'start', 'style', 'target', 'title', 'type', 'value', 'width'
        ];
        const RICH_PASTE_SAFE_STYLE_PROPS = new Set([
            'align-items', 'align-self', 'background', 'background-color', 'background-image',
            'border', 'border-block-color', 'border-block-style', 'border-block-width',
            'border-bottom', 'border-bottom-color', 'border-bottom-left-radius',
            'border-bottom-right-radius', 'border-bottom-style', 'border-bottom-width',
            'border-collapse', 'border-color', 'border-image', 'border-inline-color',
            'border-inline-style', 'border-inline-width', 'border-left', 'border-left-color',
            'border-left-style', 'border-left-width', 'border-radius', 'border-right',
            'border-right-color', 'border-right-style', 'border-right-width', 'border-spacing',
            'border-style', 'border-top', 'border-top-color', 'border-top-left-radius',
            'border-top-right-radius', 'border-top-style', 'border-top-width', 'border-width',
            'box-shadow', 'box-sizing', 'break-after', 'break-before', 'break-inside', 'clear',
            'clip-path', 'color', 'column-gap', 'direction', 'display', 'flex', 'flex-basis',
            'flex-direction', 'flex-grow', 'flex-shrink', 'flex-wrap', 'float', 'font',
            'font-family', 'font-feature-settings', 'font-kerning', 'font-language-override',
            'font-optical-sizing', 'font-size', 'font-size-adjust', 'font-stretch', 'font-style',
            'font-variant', 'font-variant-caps', 'font-variant-ligatures',
            'font-variation-settings', 'font-weight', 'gap', 'grid', 'grid-auto-columns',
            'grid-auto-flow', 'grid-auto-rows', 'grid-column', 'grid-column-end',
            'grid-column-start', 'grid-row', 'grid-row-end', 'grid-row-start', 'grid-template',
            'grid-template-areas', 'grid-template-columns', 'grid-template-rows', 'height',
            'hyphens', 'justify-content', 'justify-items', 'justify-self', 'letter-spacing',
            'line-break', 'line-height', 'list-style', 'list-style-position', 'list-style-type',
            'margin', 'margin-block', 'margin-block-end', 'margin-block-start', 'margin-bottom',
            'margin-inline', 'margin-inline-end', 'margin-inline-start', 'margin-left',
            'margin-right', 'margin-top', 'max-height', 'max-width', 'min-height', 'min-width',
            'object-fit', 'object-position', 'opacity', 'order', 'orphans', 'outline',
            'outline-color', 'outline-offset', 'outline-style', 'outline-width', 'overflow',
            'overflow-wrap', 'overflow-x', 'overflow-y', 'padding', 'padding-block',
            'padding-block-end', 'padding-block-start', 'padding-bottom', 'padding-inline',
            'padding-inline-end', 'padding-inline-start', 'padding-left', 'padding-right',
            'padding-top', 'page-break-after', 'page-break-before', 'page-break-inside',
            'row-gap', 'table-layout', 'text-align', 'text-decoration', 'text-decoration-color',
            'text-decoration-line', 'text-decoration-style', 'text-decoration-thickness',
            'text-indent', 'text-overflow', 'text-shadow', 'text-transform',
            'text-underline-offset', 'vertical-align', 'visibility', 'white-space', 'widows',
            'width', 'word-break', 'word-spacing', 'writing-mode', '-webkit-text-stroke',
            '-webkit-text-stroke-color', '-webkit-text-stroke-width'
        ]);
        const RICH_PASTE_NOISE_TAGS = new Set(['script', 'style', 'link', 'meta', 'noscript', 'iframe', 'canvas', 'svg', 'object', 'embed']);
        let userSettingsSnapshot = null;
        let userSettingsSnapshotPromise = null;
        let richPastePromptSaveTimer = null;
        let richPastePromptPreferenceSyncing = false;
        const getRichPasteEditor = () => get('rich-paste-storage');
        const getRichPasteCapture = () => get('rich-paste-capture');
        const getRichPastePrompt = () => get('rich-paste-prompt');
        const getRichPasteUseDefaultCheckbox = () => get('rich-paste-use-default');
        const getRichPasteStatus = () => get('rich-paste-status');
        const downloadBlob = (blob, fileName) => {
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = fileName;
            document.body.appendChild(a);
            a.click();
            setTimeout(() => {
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            }, 100);
        };
        const getRichPasteEffectivePrompt = (d = null) => {
            if (d && d.rich_paste_prompt_use_custom_default) {
                const text = String(d.rich_paste_prompt_default || '').trim();
                if (text) return text;
            }
            return RICH_PASTE_DEFAULT_PROMPT;
        };
        const syncRichPastePromptPreferencesUi = (d = null, options = {}) => {
            const preservePrompt = !!options.preservePrompt;
            const prompt = getRichPastePrompt();
            const checkbox = getRichPasteUseDefaultCheckbox();
            if (checkbox) checkbox.checked = !!(d && d.rich_paste_prompt_use_custom_default);
            if (prompt && !richPastePromptPreferenceSyncing && !preservePrompt) {
                prompt.value = getRichPasteEffectivePrompt(d);
            }
        };
        const cacheUserSettings = (d, options = {}) => {
            userSettingsSnapshot = d || null;
            syncRichPastePromptPreferencesUi(userSettingsSnapshot, options);
            return userSettingsSnapshot;
        };
        const SETTINGS_LOAD_TIMEOUT_MS = 15000;
        const fetchSettingsSnapshot = async () => {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), SETTINGS_LOAD_TIMEOUT_MS);
            try {
                const response = await apiFetch(CHAT_CONFIG.urls.handleSettingsQuery, {
                    cache: 'no-store',
                    signal: controller.signal
                });
                if (!response.ok) {
                    const error = new Error('HTTP ' + response.status);
                    error.retryable = response.status >= 500 || response.status === 429;
                    throw error;
                }
                const data = await response.json();
                if (!data || typeof data !== 'object') throw new Error('Invalid settings response');
                return cacheUserSettings(data);
            } finally {
                clearTimeout(timeoutId);
            }
        };
        const ensureUserSettingsSnapshot = async () => {
            if (userSettingsSnapshot) return userSettingsSnapshot;
            if (!userSettingsSnapshotPromise) {
                userSettingsSnapshotPromise = (async () => {
                    for (let attempt = 0; attempt < 2; attempt++) {
                        try { return await fetchSettingsSnapshot(); }
                        catch (error) {
                            if (error.retryable === false || attempt === 1) return null;
                            await new Promise(resolve => setTimeout(resolve, 500));
                        }
                    }
                    return null;
                })()
                    .finally(() => {
                        userSettingsSnapshotPromise = null;
                    });
            }
            return await userSettingsSnapshotPromise;
        };
        // ユーザーシステムプロンプトは設定モーダルとチャット設定の2か所から保存される。
        // どちらから保存しても、キャッシュと、プロンプトバーの SysPrompt スイッチを揃える。
        const isUserSystemPromptActive = (s) => !!(s && String(s.system_prompt || '').trim() && s.system_prompt_enabled !== false);
        window.applySavedUserSystemPromptSettings = (saved) => {
            if (!saved) return;
            const prev = userSettingsSnapshot || null;
            const next = Object.assign({}, prev || {}, saved);
            cacheUserSettings(next);
            try {
                const chk = document.getElementById('enable-sys-prompt');
                // 文面または有効/無効を変えて保存したときは、その内容をすぐ使えるようにスイッチを合わせる。
                const promptChanged = !prev
                    || String(prev.system_prompt || '') !== String(next.system_prompt || '')
                    || (prev.system_prompt_enabled !== false) !== (next.system_prompt_enabled !== false);
                if (chk && promptChanged) {
                    const active = isUserSystemPromptActive(next);
                    if (chk.disabled) {
                        // モデルの都合で一時的に使えない間は、使えるようになった時点で反映する。
                        if (active) chk.dataset.restoreChecked = '1';
                        else delete chk.dataset.restoreChecked;
                    } else if (chk.checked !== active) {
                        chk.checked = active;
                        chk.dispatchEvent(new Event('change', { bubbles: true }));
                    }
                }
            } catch (e) {}
            // サーバー側で整えられた値（自動注入の既定文面など）をキャッシュへ取り込む。
            fetchSettingsSnapshot().catch(() => {});
        };
        const saveRichPastePromptPreferences = async () => {
            const prompt = getRichPastePrompt();
            const checkbox = getRichPasteUseDefaultCheckbox();
            if (!prompt || !checkbox) return;
            const payload = {
                rich_paste_prompt_default: prompt.value || '',
                rich_paste_prompt_use_custom_default: !!checkbox.checked
            };
            try {
                await apiFetch(CHAT_CONFIG.urls.handleSettings, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(payload)
                });
                cacheUserSettings(Object.assign({}, userSettingsSnapshot || {}, payload), { preservePrompt: true });
            } catch (e) {}
        };
        const queueRichPastePromptPreferenceSave = () => {
            if (richPastePromptSaveTimer) clearTimeout(richPastePromptSaveTimer);
            richPastePromptSaveTimer = setTimeout(() => {
                richPastePromptSaveTimer = null;
                saveRichPastePromptPreferences();
            }, 500);
        };
        const hasRichPasteContent = () => {
            const editor = getRichPasteEditor();
            if (!editor) return false;
            if ((editor.textContent || '').trim()) return true;
            return !!editor.querySelector('img,table,ul,ol,blockquote,h1,h2,h3,h4,h5,h6,pre,code');
        };
        const updateRichPasteStatus = () => {
            const editor = getRichPasteEditor();
            const status = getRichPasteStatus();
            if (!status || !editor) return;
            const text = (editor.innerText || '').trim();
            if (!text) {
                status.textContent = 'まだ内容がありません。';
                return;
            }
            const images = editor.querySelectorAll('img').length;
            const tables = editor.querySelectorAll('table').length;
            const links = editor.querySelectorAll('a').length;
            const headings = editor.querySelectorAll('h1,h2,h3,h4,h5,h6').length;
            status.textContent = `${text.length} 文字 / 画像 ${images} / 表 ${tables} / リンク ${links} / 見出し ${headings}`;
        };
        const focusRichPasteEditor = () => {
            const capture = getRichPasteCapture();
            if (!capture) return;
            capture.focus();
            capture.value = capture.value || '';
            const selection = window.getSelection && window.getSelection();
            if (selection && capture.select) capture.select();
        };
        const clearRichPasteEditor = (keepPrompt = true) => {
            const editor = getRichPasteEditor();
            if (editor) editor.innerHTML = '';
            const capture = getRichPasteCapture();
            if (capture) capture.value = '';
            if (!keepPrompt) {
                const prompt = getRichPastePrompt();
                if (prompt) prompt.value = RICH_PASTE_DEFAULT_PROMPT;
            }
            updateRichPasteStatus();
        };
        const sanitizeRichPasteStyle = (styleText) => {
            if (!styleText) return '';
            const safe = [];
            String(styleText).split(';').forEach((decl) => {
                const part = decl.trim();
                if (!part) return;
                const idx = part.indexOf(':');
                if (idx <= 0) return;
                const prop = part.slice(0, idx).trim().toLowerCase();
                const value = part.slice(idx + 1).trim();
                if (!RICH_PASTE_SAFE_STYLE_PROPS.has(prop)) return;
                if (!value || value.length > 1000) return;
                const lower = value.toLowerCase();
                if (
                    lower.includes('url(') ||
                    lower.includes('expression(') ||
                    lower.includes('javascript:') ||
                    lower.includes('@import') ||
                    lower.includes('behavior:') ||
                    lower.includes('-moz-binding') ||
                    lower.includes('var(') ||
                    lower.includes('env(')
                ) return;
                safe.push(`${prop}: ${value}`);
            });
            return safe.join('; ');
        };
        let richPasteColorCanvasContext = null;
        const parseRichPasteCssColor = (value) => {
            const cssValue = String(value || '').trim();
            if (!cssValue || cssValue === 'inherit' || cssValue === 'currentcolor' || cssValue === 'transparent') {
                return null;
            }
            if (window.CSS && typeof window.CSS.supports === 'function' && !window.CSS.supports('color', cssValue)) {
                return null;
            }
            try {
                if (!richPasteColorCanvasContext) {
                    const canvas = document.createElement('canvas');
                    canvas.width = 1;
                    canvas.height = 1;
                    richPasteColorCanvasContext = canvas.getContext('2d', { willReadFrequently: true });
                }
                const context = richPasteColorCanvasContext;
                if (!context) return null;
                context.clearRect(0, 0, 1, 1);
                context.fillStyle = 'rgba(1, 2, 3, 0.004)';
                context.fillStyle = cssValue;
                context.fillRect(0, 0, 1, 1);
                const pixel = context.getImageData(0, 0, 1, 1).data;
                if (!pixel || pixel[3] === 0) return null;
                return {
                    r: pixel[0],
                    g: pixel[1],
                    b: pixel[2],
                    a: pixel[3] / 255
                };
            } catch (e) {
                return null;
            }
        };
        const richPasteColorLuminance = (color) => {
            if (!color) return 0;
            const channel = (value) => {
                const normalized = Math.max(0, Math.min(255, Number(value) || 0)) / 255;
                return normalized <= 0.04045
                    ? normalized / 12.92
                    : Math.pow((normalized + 0.055) / 1.055, 2.4);
            };
            return (0.2126 * channel(color.r)) + (0.7152 * channel(color.g)) + (0.0722 * channel(color.b));
        };
        const richPasteColorContrast = (first, second) => {
            const firstLum = richPasteColorLuminance(first);
            const secondLum = richPasteColorLuminance(second);
            return (Math.max(firstLum, secondLum) + 0.05) / (Math.min(firstLum, secondLum) + 0.05);
        };
        const richPasteColorCss = (color) => {
            if (!color) return '';
            return `rgb(${Math.round(color.r)}, ${Math.round(color.g)}, ${Math.round(color.b)})`;
        };
        const makeRichPasteTheme = (background, foreground) => {
            const dark = richPasteColorLuminance(background) < 0.32;
            let resolvedForeground = foreground;
            if (!resolvedForeground || richPasteColorContrast(background, resolvedForeground) < 3) {
                resolvedForeground = dark
                    ? { r: 244, g: 244, b: 245, a: 1 }
                    : { r: 17, g: 24, b: 39, a: 1 };
            }
            return {
                mode: dark ? 'dark' : 'light',
                background: richPasteColorCss(background),
                foreground: richPasteColorCss(resolvedForeground),
                muted: dark ? 'rgb(161, 161, 170)' : 'rgb(100, 116, 139)',
                border: dark ? 'rgb(63, 63, 70)' : 'rgb(203, 213, 225)',
                surface: dark ? 'rgb(33, 33, 33)' : 'rgb(248, 250, 252)',
                quote: dark ? 'rgb(39, 39, 42)' : 'rgb(255, 249, 235)',
                link: dark ? 'rgb(125, 211, 252)' : 'rgb(15, 118, 110)'
            };
        };
        const detectRichPasteTheme = (contentHtml) => {
            const fallbackBackground = { r: 255, g: 255, b: 255, a: 1 };
            const fallbackForeground = { r: 17, g: 24, b: 39, a: 1 };
            const template = document.createElement('template');
            template.innerHTML = String(contentHtml || '');
            if (!template.content.querySelector('*')) {
                return makeRichPasteTheme(fallbackBackground, fallbackForeground);
            }

            const probe = document.createElement('div');
            probe.setAttribute('aria-hidden', 'true');
            probe.style.position = 'fixed';
            probe.style.left = '-100000px';
            probe.style.top = '0';
            probe.style.width = '794px';
            probe.style.visibility = 'hidden';
            probe.style.pointerEvents = 'none';
            probe.style.color = '#111827';
            probe.style.background = 'transparent';
            probe.appendChild(template.content.cloneNode(true));
            document.body.appendChild(probe);

            try {
                const nodes = [probe, ...Array.from(probe.querySelectorAll('*')).slice(0, 5000)];
                const backgroundCandidates = [];
                const foregroundWeights = new Map();
                let totalForegroundWeight = 0;
                const directTextLength = (node) => Array.from(node.childNodes || []).reduce((total, child) => {
                    if (child && child.nodeType === Node.TEXT_NODE) {
                        return total + String(child.textContent || '').replace(/\s+/g, ' ').trim().length;
                    }
                    return total;
                }, 0);

                nodes.forEach((node) => {
                    if (!node || node === probe || !node.style) return;
                    const computed = window.getComputedStyle(node);
                    const directWeight = directTextLength(node);
                    if (directWeight > 0) {
                        const foreground = parseRichPasteCssColor(computed.color);
                        if (foreground && foreground.a >= 0.5) {
                            const key = richPasteColorCss(foreground);
                            const previous = foregroundWeights.get(key) || { color: foreground, weight: 0 };
                            previous.weight += directWeight;
                            foregroundWeights.set(key, previous);
                            totalForegroundWeight += directWeight;
                        }
                    }

                    const hasInlineBackground = !!(
                        String(node.style.backgroundColor || '').trim() ||
                        String(node.style.background || '').trim()
                    );
                    if (hasInlineBackground) {
                        const background = parseRichPasteCssColor(computed.backgroundColor);
                        if (background && background.a >= 0.72) {
                            const subtreeWeight = String(node.textContent || '').replace(/\s+/g, ' ').trim().length;
                            backgroundCandidates.push({
                                color: background,
                                weight: Math.max(1, subtreeWeight)
                            });
                        }
                    }
                });

                const foregroundEntries = Array.from(foregroundWeights.values()).sort((a, b) => b.weight - a.weight);
                const dominantForeground = foregroundEntries.length ? foregroundEntries[0].color : null;
                const lightTextWeight = foregroundEntries.reduce((total, entry) => (
                    total + (richPasteColorLuminance(entry.color) >= 0.6 ? entry.weight : 0)
                ), 0);
                backgroundCandidates.sort((a, b) => b.weight - a.weight);
                let background = backgroundCandidates.length ? backgroundCandidates[0].color : null;
                if (!background) {
                    background = totalForegroundWeight > 0 && lightTextWeight / totalForegroundWeight >= 0.55
                        ? { r: 11, g: 11, b: 12, a: 1 }
                        : fallbackBackground;
                }
                return makeRichPasteTheme(background, dominantForeground || fallbackForeground);
            } catch (e) {
                return makeRichPasteTheme(fallbackBackground, fallbackForeground);
            } finally {
                if (probe.parentNode) probe.parentNode.removeChild(probe);
            }
        };
        const prepareRichPastePdfClone = (clonedDoc, theme) => {
            if (!clonedDoc) return;
            const head = clonedDoc.head || clonedDoc.querySelector('head');
            if (head) {
                Array.from(head.querySelectorAll('link[rel="stylesheet"]')).forEach((node) => {
                    try { node.remove(); } catch (e) {}
                });
            }
            if (clonedDoc.body) {
                clonedDoc.body.style.margin = '0';
                clonedDoc.body.style.background = theme.background;
                clonedDoc.body.style.color = theme.foreground;
            }
        };
        const normalizeRichPasteTree = (root) => {
            if (!root || typeof root.querySelectorAll !== 'function') return;
            root.querySelectorAll('*').forEach((node) => {
                if (!node || !node.getAttribute || !node.parentNode) return;
                const tag = String(node.tagName || '').toLowerCase();
                if (RICH_PASTE_NOISE_TAGS.has(tag)) {
                    node.remove();
                    return;
                }
                node.removeAttribute('class');
                node.removeAttribute('id');
                node.removeAttribute('role');
                node.removeAttribute('aria-label');
                if (tag === 'img') {
                    node.setAttribute('loading', 'eager');
                    node.setAttribute('decoding', 'sync');
                    node.removeAttribute('srcset');
                    node.removeAttribute('sizes');
                }
                const styleText = node.getAttribute('style');
                if (styleText) {
                    const safeStyle = sanitizeRichPasteStyle(styleText);
                    if (safeStyle) node.setAttribute('style', safeStyle);
                    else node.removeAttribute('style');
                }
            });
        };
        const extractRichPasteArticleHtml = (html) => {
            const parser = new DOMParser();
            const doc = parser.parseFromString(String(html || ''), 'text/html');
            if (!doc.body) return '';
            const bodyTextLength = (doc.body.textContent || '').replace(/\s+/g, ' ').trim().length;
            const bodyTagCount = doc.body.querySelectorAll('*').length;
            if (bodyTextLength < 1000 || bodyTagCount < 120) {
                return doc.body.innerHTML;
            }
            const candidates = [
                ...Array.from(doc.body.querySelectorAll('article')),
                ...Array.from(doc.body.querySelectorAll('main')),
                ...Array.from(doc.body.querySelectorAll('[role="main"],[role="article"]'))
            ];
            const eligible = candidates.filter((node) => {
                const textLength = (node.textContent || '').replace(/\s+/g, ' ').trim().length;
                return textLength >= bodyTextLength * 0.65;
            });
            eligible.sort((a, b) => {
                const headingDelta = Number(!!b.querySelector('h1')) - Number(!!a.querySelector('h1'));
                if (headingDelta) return headingDelta;
                return a.querySelectorAll('*').length - b.querySelectorAll('*').length;
            });
            const best = eligible[0] || null;
            if (best) {
                return best.outerHTML;
            }
            return doc.body.innerHTML;
        };
        const sanitizeRichPasteHtml = (html) => {
            if (!window.DOMPurify || typeof window.DOMPurify.sanitize !== 'function') {
                const fallbackDoc = new DOMParser().parseFromString(String(html || ''), 'text/html');
                return escapeHtml(fallbackDoc.body ? fallbackDoc.body.textContent : '');
            }
            let articleHtml = extractRichPasteArticleHtml(html);
            let safeHtml = window.DOMPurify.sanitize(articleHtml || '', {
                ALLOWED_TAGS: RICH_PASTE_ALLOWED_TAGS,
                ALLOWED_ATTR: RICH_PASTE_ALLOWED_ATTR,
                KEEP_CONTENT: true
            });
            if ((!safeHtml || safeHtml.trim() === '') && html && html.trim() !== '') {
                safeHtml = window.DOMPurify.sanitize(html, {
                    ALLOWED_TAGS: RICH_PASTE_ALLOWED_TAGS,
                    ALLOWED_ATTR: RICH_PASTE_ALLOWED_ATTR,
                    KEEP_CONTENT: true
                });
            }
            if (!safeHtml) return '';
            const template = document.createElement('template');
            template.innerHTML = safeHtml;
            normalizeRichPasteTree(template.content);
            return template.innerHTML;
        };
        const normalizeRichPastePrintHtml = (contentHtml) => {
            const template = document.createElement('template');
            template.innerHTML = String(contentHtml || '');
            const nodes = Array.from(template.content.querySelectorAll('*'));
            const complexLayoutCount = nodes.reduce((total, node) => {
                const display = String(node.style && node.style.display || '').trim().toLowerCase();
                return total + (['flex', 'inline-flex', 'grid', 'inline-grid'].includes(display) ? 1 : 0);
            }, 0);
            const screenLayoutCount = nodes.reduce((total, node) => {
                if (!node || !node.style || !['article', 'div', 'main', 'section'].includes(String(node.tagName || '').toLowerCase())) {
                    return total;
                }
                const styleText = String(node.getAttribute('style') || '');
                const largeSidePadding = Array.from(
                    styleText.matchAll(/(?:^|;)\s*padding(?:-left|-right|-inline|-inline-start|-inline-end)?\s*:\s*([^;]+)/gi)
                ).some((match) => Array.from(match[1].matchAll(/(-?\d+(?:\.\d+)?)px/gi))
                    .some((pixelMatch) => Math.abs(Number(pixelMatch[1]) || 0) >= 96));
                const oversizedWidth = Array.from(
                    styleText.matchAll(/(?:^|;)\s*(?:width|min-width)\s*:\s*(-?\d+(?:\.\d+)?)px/gi)
                ).some((match) => Math.abs(Number(match[1]) || 0) > 720);
                return total + (largeSidePadding || oversizedWidth ? 1 : 0);
            }, 0);
            if (nodes.length <= 500 && complexLayoutCount <= 24 && screenLayoutCount === 0) {
                return template.innerHTML;
            }

            const layoutProps = new Set([
                'align-items', 'align-self', 'column-gap', 'flex', 'flex-basis', 'flex-direction',
                'flex-grow', 'flex-shrink', 'flex-wrap', 'gap', 'grid', 'grid-auto-columns',
                'grid-auto-flow', 'grid-auto-rows', 'grid-column', 'grid-column-end',
                'grid-column-start', 'grid-row', 'grid-row-end', 'grid-row-start', 'grid-template',
                'grid-template-areas', 'grid-template-columns', 'grid-template-rows',
                'justify-content', 'justify-items', 'justify-self', 'order', 'row-gap'
            ]);
            const blockWidthTags = new Set(['article', 'div', 'main', 'section']);
            const sidePaddingProps = new Set([
                'padding', 'padding-left', 'padding-right', 'padding-inline',
                'padding-inline-start', 'padding-inline-end'
            ]);
            nodes.forEach((node) => {
                if (!node || !node.style) return;
                const tag = String(node.tagName || '').toLowerCase();
                const declarations = [];
                String(node.getAttribute('style') || '').split(';').forEach((declaration) => {
                    if (!declaration || declaration.indexOf(':') < 0) return;
                    const separator = declaration.indexOf(':');
                    const prop = declaration.slice(0, separator).trim().toLowerCase();
                    let value = declaration.slice(separator + 1).trim();
                    if (!prop || !value || layoutProps.has(prop)) return;
                    if (['height', 'max-height', 'min-height', 'overflow', 'overflow-x', 'overflow-y'].includes(prop)) {
                        return;
                    }
                    if (['width', 'min-width'].includes(prop) && blockWidthTags.has(tag)) {
                        return;
                    }
                    if (sidePaddingProps.has(prop) && blockWidthTags.has(tag)) {
                        const pixelValues = Array.from(value.matchAll(/(-?\d+(?:\.\d+)?)px/gi))
                            .map((match) => Math.abs(Number(match[1]) || 0));
                        if (pixelValues.some((pixelValue) => pixelValue >= 96)) {
                            value = '0px';
                        }
                    }
                    if (prop === 'display') {
                        const displayValue = value.toLowerCase();
                        if (['flex', 'grid'].includes(displayValue)) value = 'block';
                        else if (['inline-flex', 'inline-grid'].includes(displayValue)) value = 'inline-block';
                    }
                    declarations.push(`${prop}: ${value}`);
                });
                if (declarations.length) node.setAttribute('style', declarations.join('; '));
                else node.removeAttribute('style');
            });
            return template.innerHTML;
        };
        const getRichPasteSelectionRange = (editor) => {
            const selection = window.getSelection && window.getSelection();
            if (!selection || !selection.rangeCount) return null;
            const range = selection.getRangeAt(0);
            if (editor && editor.contains(range.commonAncestorContainer)) return range;
            const fallback = document.createRange();
            fallback.selectNodeContents(editor);
            fallback.collapse(false);
            return fallback;
        };
        const insertNodeIntoRichPasteEditor = (node) => {
            const editor = getRichPasteEditor();
            if (!editor || !node) return;
            editor.appendChild(node);
            updateRichPasteStatus();
        };
        const insertHtmlIntoRichPasteEditor = (html) => {
            const safeHtml = sanitizeRichPasteHtml(html);
            if (!safeHtml || safeHtml.trim() === '') return false;
            const template = document.createElement('template');
            template.innerHTML = safeHtml;
            const frag = template.content.cloneNode(true);
            insertNodeIntoRichPasteEditor(frag);
            return true;
        };
        const insertTextIntoRichPasteEditor = (text) => {
            if (text === null || text === undefined) return;
            const node = document.createTextNode(String(text));
            insertNodeIntoRichPasteEditor(node);
        };
        const blobToDataUrl = (blob) => new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(String(reader.result || ''));
            reader.onerror = () => reject(reader.error || new Error('clipboard_image_read_failed'));
            reader.readAsDataURL(blob);
        });
