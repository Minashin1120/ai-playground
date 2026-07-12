        const get = (id) => document.getElementById(id);
        const externalScriptLoads = new Map();
        const loadExternalScript = (src, globalReady) => {
            if (typeof globalReady === 'function' && globalReady()) return Promise.resolve();
            if (externalScriptLoads.has(src)) return externalScriptLoads.get(src);
            const pending = new Promise((resolve, reject) => {
                const script = document.createElement('script');
                script.src = src;
                script.async = true;
                script.crossOrigin = 'anonymous';
                script.referrerPolicy = 'no-referrer';
                script.onload = () => resolve();
                script.onerror = () => reject(new Error(`ライブラリを読み込めませんでした: ${src}`));
                document.head.appendChild(script);
            });
            externalScriptLoads.set(src, pending);
            return pending;
        };
        const ensurePdfLibraries = () => Promise.all([
            loadExternalScript(
                'https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js',
                () => typeof window.html2canvas === 'function'
            ),
            loadExternalScript(
                'https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js',
                () => !!(window.jspdf && window.jspdf.jsPDF)
            )
        ]);
        const ensureImageCompression = () => loadExternalScript(
            'https://cdn.jsdelivr.net/npm/browser-image-compression@2.0.2/dist/browser-image-compression.js',
            () => typeof window.imageCompression === 'function'
        );
        let webauthnJsonLoad = null;
        const ensureWebAuthnJson = async () => {
            if (window.webauthnJSON) return window.webauthnJSON;
            if (!webauthnJsonLoad) {
                webauthnJsonLoad = import('https://esm.sh/@github/webauthn-json@2.1.1')
                    .then(({ create, get: getCredential }) => ({ create, get: getCredential }));
            }
            window.webauthnJSON = await webauthnJsonLoad;
            return window.webauthnJSON;
        };
        // Media is allowed, but untrusted chat content must never create frames.
        if (window.DOMPurify) {
            window.DOMPurify.setConfig(window.CHAT_DOMPURIFY_CONFIG || {
                ADD_TAGS: ['video', 'source'],
                ADD_ATTR: ['controls', 'src', 'class', 'autoplay', 'loop', 'muted', 'poster', 'width', 'height', 'start', 'type', 'reversed'],
                FORBID_TAGS: ['iframe', 'object', 'embed']
            });
        }
        const THEME_DEFAULT = '#0dd4bf';
        const THEME_STORAGE_KEY = 'theme_color';
        const INITIAL_THEME_COLOR = (window.CHAT_CONFIG && window.CHAT_CONFIG.initialThemeColor) || null;
        const INITIAL_LIQUID_GLASS_ENABLED = !!(window.CHAT_CONFIG && window.CHAT_CONFIG.initialLiquidGlassEnabled);
        const RICH_PASTE_DEFAULT_PROMPT = 'このPDFをMarkdown形式に変換し、コードブロックに書き出してください。';
        const GEMINI_LOCAL_PY_DIALOG_KEY = 'gemini_local_py_dialog_enabled';
        const COMPRESSION_SIZE_KEY = 'compression_max_size_mb';
        const COMPRESSION_DIM_KEY = 'compression_max_dim';
        const COMPRESSION_TYPE_KEY = 'compression_output_type';
        const COMPRESSION_FORMAT_ONLY_KEY = 'compression_format_only';
        const getCompressionMaxSizeMB = () => parseFloat(localStorage.getItem(COMPRESSION_SIZE_KEY) || '1.0');
        const getCompressionMaxDim = () => parseInt(localStorage.getItem(COMPRESSION_DIM_KEY) || '1920');
        const getCompressionOutputType = () => localStorage.getItem(COMPRESSION_TYPE_KEY) || 'original';
        const getCompressionFormatOnly = () => localStorage.getItem(COMPRESSION_FORMAT_ONLY_KEY) === 'true';
        const setCompressionSettings = (size, dim, type, formatOnly) => {
            localStorage.setItem(COMPRESSION_SIZE_KEY, size);
            localStorage.setItem(COMPRESSION_DIM_KEY, dim);
            localStorage.setItem(COMPRESSION_TYPE_KEY, type);
            localStorage.setItem(COMPRESSION_FORMAT_ONLY_KEY, formatOnly);
        };
        const syncCompressionSettingsUi = () => {
            const sizeEl = get('compression-max-size');
            const dimEl = get('compression-max-dim');
            const typeEl = get('compression-output-type');
            const onlyEl = get('compression-format-only');

            if (sizeEl) sizeEl.value = getCompressionMaxSizeMB();
            if (dimEl) dimEl.value = getCompressionMaxDim();
            if (typeEl) typeEl.value = getCompressionOutputType();
            if (onlyEl) {
                onlyEl.checked = getCompressionFormatOnly();
                const disabled = onlyEl.checked;
                if (sizeEl) sizeEl.disabled = disabled;
                if (dimEl) dimEl.disabled = disabled;
                const sizeWrap = get('compression-size-wrap');
                const dimWrap = get('compression-dim-wrap');
                if (sizeWrap) sizeWrap.style.opacity = disabled ? '0.4' : '1';
                if (dimWrap) dimWrap.style.opacity = disabled ? '0.4' : '1';
            }

            // Sync model-specific settings from the prompt bar to the modal
            const sync = (srcId, destId) => { if(get(srcId) && get(destId)) get(destId).value = get(srcId).value; };
            sync('gpt-image-size', 'modal-gpt-image-size');
            sync('gpt-image-quality', 'modal-gpt-image-quality');
            sync('gpt-image-format', 'modal-gpt-image-format');
            sync('gpt-image-compression', 'modal-gpt-image-compression');
            sync('gemini-image-aspect', 'modal-gemini-image-aspect');
            sync('gemini-image-size', 'modal-gemini-image-size');
            sync('grok-image-aspect', 'modal-grok-image-aspect');
            sync('grok-image-resolution', 'modal-grok-image-resolution');

            // Hide/Show sections based on current model support
            const model = get('model-select').value;
            const isGpt = isGptImageModel(model);
            const isGemini = isGeminiImageModel(model);
            const isGrok = isGrokImageModel(model);
            if(get('modal-gpt-image-options')) get('modal-gpt-image-options').classList.toggle('hidden', !isGpt);
            if(get('modal-gemini-image-options')) get('modal-gemini-image-options').classList.toggle('hidden', !isGemini);
            if(get('modal-grok-image-options')) get('modal-grok-image-options').classList.toggle('hidden', !isGrok);
        };
        const isGeminiLocalPyDialogEnabled = () => {
            const v = localStorage.getItem(GEMINI_LOCAL_PY_DIALOG_KEY);
            if (v === null) return true;
            return v === '1' || v === 'true';
        };
        const setGeminiLocalPyDialogEnabled = (enabled) => {
            localStorage.setItem(GEMINI_LOCAL_PY_DIALOG_KEY, enabled ? '1' : '0');
        };
        const syncGeminiLocalPyDialogSetting = () => {
            const el = get('set-gemini-local-python-dialog');
            if (el) el.checked = isGeminiLocalPyDialogEnabled();
        };
        const normalizeGeminiBackend = (value) => {
            const raw = String(value || '').trim().toLowerCase().replace('-', '_');
            return raw === 'vertex_ai' || raw === 'vertex' || raw === 'vertexai' ? 'vertex_ai' : 'gemini_api';
        };
        const normalizeAdminApiKeyMode = (value) => {
            const raw = String(value || '').trim().toLowerCase().replace('-', '_');
            return raw === 'user_only' || raw === 'user' || raw === 'settings' || raw === 'user_settings'
                ? 'user_only'
                : 'env_fallback';
        };
        const syncToggleButtons = (buttons, activeValue, attrName) => {
            (buttons || []).forEach((btn) => {
                const isActive = btn.getAttribute(attrName) === activeValue;
                btn.classList.toggle('border-cyan-400', isActive);
                btn.classList.toggle('bg-cyan-900/30', isActive);
                btn.classList.toggle('text-white', isActive);
                btn.classList.toggle('border-gray-600', !isActive);
                btn.classList.toggle('bg-gray-800/70', !isActive);
            });
        };
        const syncAdminApiKeyModeUi = () => {
            const modeEl = get('set-admin-api-key-mode');
            const noteEl = get('admin-api-key-mode-note');
            const statusEl = get('admin-api-key-mode-status');
            const toggleWrap = get('admin-api-key-mode-toggle');
            if (!modeEl) return;
            const mode = normalizeAdminApiKeyMode(modeEl.value);
            modeEl.value = mode;
            if (toggleWrap && !toggleWrap.dataset.bound) {
                toggleWrap.dataset.bound = '1';
                toggleWrap.querySelectorAll('[data-admin-api-key-mode]').forEach((btn) => {
                    btn.addEventListener('click', () => {
                        modeEl.value = normalizeAdminApiKeyMode(btn.getAttribute('data-admin-api-key-mode'));
                        syncAdminApiKeyModeUi();
                    });
                });
            }
            syncToggleButtons(
                toggleWrap ? toggleWrap.querySelectorAll('[data-admin-api-key-mode]') : [],
                mode,
                'data-admin-api-key-mode'
            );
            if (noteEl) {
                noteEl.textContent = mode === 'user_only'
                    ? '通常ユーザーと同じく、この画面で保存したAPIキー/Vertex設定のみを使用します。'
                    : '管理者設定が空欄のときだけ .env をフォールバック利用します（既定）。';
            }
            if (statusEl) {
                statusEl.textContent = mode === 'user_only'
                    ? '現在: ユーザー設定のみ（推奨: 設定値を明示管理）'
                    : '現在: .env フォールバック有効（管理者設定が空欄なら .env）';
            }
        };
        const ensureGeminiVertexCredentialsField = () => {
            const vertexWrap = get('gemini-vertex-settings');
            if (!vertexWrap || get('set-gemini-vertex-credentials-json')) return;
            const block = document.createElement('div');
            block.innerHTML = `
                <label class="text-xs text-gray-500 block">Vertex Service Account JSON (任意)</label>
                <textarea id="set-gemini-vertex-credentials-json" class="w-full h-28 bg-gray-800 border border-gray-600 rounded px-2 py-1 text-[11px] text-white font-mono" placeholder='{"type":"service_account", ...}'></textarea>
                <div class="text-[10px] text-gray-500 mt-1">未入力時はサーバー側ADCを使用します。入力するとこのユーザーの設定だけでVertex認証できます。</div>
            `;
            vertexWrap.appendChild(block);
        };
        const syncGeminiBackendUi = () => {
            const backendEl = get('set-gemini-backend');
            const vertexWrap = get('gemini-vertex-settings');
            const noteEl = get('gemini-backend-note');
            const statusEl = get('gemini-backend-status');
            const toggleWrap = get('gemini-backend-toggle');
            if (!backendEl) return;
            ensureGeminiVertexCredentialsField();
            const backend = normalizeGeminiBackend(backendEl.value);
            backendEl.value = backend;
            if (toggleWrap && !toggleWrap.dataset.bound) {
                toggleWrap.dataset.bound = '1';
                toggleWrap.querySelectorAll('[data-gemini-backend]').forEach((btn) => {
                    btn.addEventListener('click', () => {
                        backendEl.value = normalizeGeminiBackend(btn.getAttribute('data-gemini-backend'));
                        syncGeminiBackendUi();
                    });
                });
            }
            syncToggleButtons(
                toggleWrap ? toggleWrap.querySelectorAll('[data-gemini-backend]') : [],
                backend,
                'data-gemini-backend'
            );
            if (vertexWrap) vertexWrap.classList.toggle('hidden', backend !== 'vertex_ai');
            if (noteEl) {
                noteEl.textContent = backend === 'vertex_ai'
                    ? 'Vertex AI を利用します。Project ID / Location を設定し、ADC または Vertex Service Account JSON を用意してください。'
                    : 'Gemini API を利用します。API Key を設定してください。';
            }
            if (statusEl) {
                statusEl.textContent = backend === 'vertex_ai'
                    ? '現在: Vertex AI（Project ID / Location / 認証情報が必要）'
                    : '現在: Gemini API（Gemini API Key を使用）';
            }
        };
        const normalizeHex = (value) => {
            if (!value) return null;
            let v = String(value).trim();
            if (!v) return null;
            if (!v.startsWith('#')) v = `#${v}`;
            if (v.length === 4) v = `#${v[1]}${v[1]}${v[2]}${v[2]}${v[3]}${v[3]}`;
            if (!/^#[0-9a-fA-F]{6}$/.test(v)) return null;
            return v.toLowerCase();
        };
        const hexToRgb = (hex) => {
            const h = hex.replace('#', '');
            const r = parseInt(h.slice(0, 2), 16);
            const g = parseInt(h.slice(2, 4), 16);
            const b = parseInt(h.slice(4, 6), 16);
            return [r, g, b];
        };
        const mix = (a, b, p) => Math.round(a + (b - a) * p);
        const rgbToHex = (r, g, b) => `#${[r, g, b].map(v => v.toString(16).padStart(2, '0')).join('')}`;
        const deriveTheme = (hex) => {
            const [r, g, b] = hexToRgb(hex);
            const light = rgbToHex(mix(r, 255, 0.45), mix(g, 255, 0.45), mix(b, 255, 0.45));
            const lighter = rgbToHex(mix(r, 255, 0.7), mix(g, 255, 0.7), mix(b, 255, 0.7));
            const dark = rgbToHex(mix(r, 0, 0.18), mix(g, 0, 0.18), mix(b, 0, 0.18));
            const darker = rgbToHex(mix(r, 0, 0.32), mix(g, 0, 0.32), mix(b, 0, 0.32));
            return { base: hex, light, lighter, dark, darker, rgb: `${r}, ${g}, ${b}` };
        };
        const applyThemeColor = (value, persist = false) => {
            const hex = normalizeHex(value) || THEME_DEFAULT;
            const theme = deriveTheme(hex);
            const root = document.documentElement;
            root.style.setProperty('--theme-500', theme.base);
            root.style.setProperty('--theme-600', theme.dark);
            root.style.setProperty('--theme-700', theme.darker);
            root.style.setProperty('--theme-300', theme.light);
            root.style.setProperty('--theme-200', theme.lighter);
            root.style.setProperty('--theme-rgb', theme.rgb);
            if (persist) localStorage.setItem(THEME_STORAGE_KEY, hex);
        };
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
            '.modal-panel',
            '.modal-glass-panel',
            '.viewer-toolbar',
            '.viewer-meta',
            '#quote-bar',
            '#slash-command-suggestions',
            '#gem-suggestions',
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
        document.addEventListener('pointermove', (event) => {
            if (!document.body || !document.body.classList.contains('liquid-glass-mode')) return;
            const surface = event.target.closest ? event.target.closest(LIQUID_GLASS_SURFACE_SELECTOR) : null;
            if (!surface) return;
            const rect = surface.getBoundingClientRect();
            if (!rect.width || !rect.height) return;
            const x = Math.max(0, Math.min(100, ((event.clientX - rect.left) / rect.width) * 100));
            const y = Math.max(0, Math.min(100, ((event.clientY - rect.top) / rect.height) * 100));
            surface.style.setProperty('--glass-light-x', `${x.toFixed(1)}%`);
            surface.style.setProperty('--glass-light-y', `${y.toFixed(1)}%`);
        }, { passive: true });
        document.addEventListener('pointerout', (event) => {
            const surface = event.target.closest ? event.target.closest(LIQUID_GLASS_SURFACE_SELECTOR) : null;
            if (!surface || (event.relatedTarget && surface.contains(event.relatedTarget))) return;
            surface.style.removeProperty('--glass-light-x');
            surface.style.removeProperty('--glass-light-y');
            surface.classList.remove('liquid-glass-pressed');
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
            document.body.classList.add('liquid-glass-scrolling');
            window.clearTimeout(liquidGlassScrollTimer);
            liquidGlassScrollTimer = window.setTimeout(() => {
                if (document.body) document.body.classList.remove('liquid-glass-scrolling');
            }, 140);
        }, { passive: true, capture: true });
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
        const showModal = (id) => {
            const el = get(id);
            if (!el) return;
            el.classList.remove('hidden');
            el.style.display = 'flex';
            el.classList.remove('modal-close');
            el.classList.remove('modal-open');
            el.classList.add('modal-prep');
            // Force a reflow so the first open animates reliably, then double-rAF for a smoother first paint.
            void el.offsetHeight;
            requestAnimationFrame(() => {
                requestAnimationFrame(() => {
                    el.classList.remove('modal-prep');
                    el.classList.add('modal-open');
                });
            });
        };
        window.showModal = showModal;
        const hideModal = (id, options = {}) => {
            const el = get(id);
            if (!el) return;
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
            setTimeout(() => {
                el.style.display = 'none';
                el.classList.remove('modal-close');
                el.classList.remove('modal-prep');
                el.classList.add('hidden');
            }, MODAL_ANIM_MS);
        };
        const RICH_PASTE_ALLOWED_TAGS = [
            'a', 'abbr', 'b', 'blockquote', 'br', 'caption', 'code', 'col', 'colgroup', 'div',
            'em', 'figcaption', 'figure', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'hr', 'i', 'img',
            'li', 'mark', 'ol', 'p', 'pre', 's', 'section', 'small', 'span', 'strong', 'sub',
            'sup', 'table', 'tbody', 'td', 'th', 'thead', 'tfoot', 'tr', 'u', 'ul'
        ];
        const RICH_PASTE_ALLOWED_ATTR = [
            'align', 'alt', 'cellpadding', 'cellspacing', 'class', 'colspan', 'href', 'height',
            'rel', 'rowspan', 'src', 'style', 'target', 'title', 'width'
        ];
        const RICH_PASTE_SAFE_STYLE_PROPS = new Set([
            'background-color', 'color', 'font-family', 'font-size', 'font-style', 'font-weight',
            'letter-spacing', 'line-height', 'text-align', 'text-decoration', 'text-transform',
            'vertical-align', 'white-space'
        ]);
        const RICH_PASTE_NOISE_TAGS = new Set(['script', 'style', 'link', 'meta', 'noscript', 'iframe', 'canvas', 'svg', 'object', 'embed']);
        const RICH_PASTE_NOISE_ROLE_RE = /(^|[\s_-])(nav|navbar|menu|footer|header|aside|sidebar|ads?|ad-|promo|cookie|banner|share|social|comments?|related|recommend|breadcrumb|subscription|popup|modal|overlay|toolbar|dialog|toast|sponsor)([\s_-]|$)/i;
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
        const ensureUserSettingsSnapshot = async () => {
            if (userSettingsSnapshot) return userSettingsSnapshot;
            if (!userSettingsSnapshotPromise) {
                userSettingsSnapshotPromise = apiFetch(CHAT_CONFIG.urls.handleSettingsQuery)
                    .then((r) => r.json())
                    .then((d) => cacheUserSettings(d))
                    .catch(() => null)
                    .finally(() => {
                        userSettingsSnapshotPromise = null;
                    });
            }
            return await userSettingsSnapshotPromise;
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
            const hasUnsupportedColorSyntax = (value) => {
                const lower = String(value || '').toLowerCase();
                return (
                    lower.includes('oklab(') ||
                    lower.includes('oklch(') ||
                    lower.includes('color-mix(') ||
                    lower.includes('lab(') ||
                    lower.includes('lch(')
                );
            };
            String(styleText).split(';').forEach((decl) => {
                const part = decl.trim();
                if (!part) return;
                const idx = part.indexOf(':');
                if (idx <= 0) return;
                const prop = part.slice(0, idx).trim().toLowerCase();
                const value = part.slice(idx + 1).trim();
                if (!RICH_PASTE_SAFE_STYLE_PROPS.has(prop)) return;
                const lower = value.toLowerCase();
                if (lower.includes('position:') || lower.includes('fixed') || lower.includes('absolute') || lower.includes('sticky')) return;
                if (lower.includes('z-index') || lower.includes('overflow') || lower.includes('transform') || lower.includes('filter')) return;
                if (lower.includes('url(') || lower.includes('expression(')) return;
                if (hasUnsupportedColorSyntax(value)) return;
                if (prop === 'font-size' && /calc\s*\(/i.test(value)) return;
                safe.push(`${prop}: ${value}`);
            });
            return safe.join('; ');
        };
        const prepareRichPastePdfClone = (clonedDoc) => {
            if (!clonedDoc) return;
            const head = clonedDoc.head || clonedDoc.querySelector('head');
            if (head) {
                Array.from(head.querySelectorAll('link[rel="stylesheet"]')).forEach((node) => {
                    try { node.remove(); } catch (e) {}
                });
            }
            if (clonedDoc.body) {
                clonedDoc.body.style.margin = '0';
                clonedDoc.body.style.background = '#ffffff';
                clonedDoc.body.style.color = '#111827';
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
                const textLen = (node.textContent || '').trim().length;
                if (textLen > 160) return; 

                const role = String(node.getAttribute('role') || '').trim().toLowerCase();
                const aria = String(node.getAttribute('aria-label') || '').trim();
                const cls = String(node.className || '').trim();
                const id = String(node.id || '').trim();
                const noiseText = `${role} ${aria} ${cls} ${id}`;
                if (RICH_PASTE_NOISE_ROLE_RE.test(noiseText)) {
                    node.remove();
                    return;
                }
                node.removeAttribute('class');
                node.removeAttribute('id');
                node.removeAttribute('width');
                node.removeAttribute('height');
                if (tag === 'img') {
                    node.setAttribute('loading', 'lazy');
                    node.setAttribute('decoding', 'async');
                    node.removeAttribute('srcset');
                    node.removeAttribute('sizes');
                }
                const styleText = node.getAttribute('style');
                if (styleText) {
                    const safeStyle = sanitizeRichPasteStyle(styleText);
                    if (safeStyle) node.setAttribute('style', safeStyle);
                    else node.removeAttribute('style');
                }
                if (tag === 'table') {
                    node.removeAttribute('border');
                }
            });
        };
        const extractRichPasteArticleHtml = (html) => {
            const parser = new DOMParser();
            const doc = parser.parseFromString(String(html || ''), 'text/html');
            const candidates = [];
            const pushCandidate = (node, boost = 0) => {
                if (!node) return;
                const textLen = (node.textContent || '').replace(/\s+/g, ' ').trim().length;
                if (textLen <= 0) return;
                const mediaCount = node.querySelectorAll ? node.querySelectorAll('img,table,figure,video,picture,pre,blockquote').length : 0;
                candidates.push({ node, score: textLen + (mediaCount * 120) + boost });
            };
            pushCandidate(doc.body, 0);
            const queryList = [
                'article',
                'main',
                '[role="main"]',
                '[role="article"]',
                'section',
                '.article',
                '.article-body',
                '.post',
                '.post-content',
                '.content',
                '.entry-content',
                '.article-content',
                '.story-body'
            ];
            queryList.forEach((sel, idx) => {
                doc.querySelectorAll(sel).forEach((node) => pushCandidate(node, 1000 - idx * 10));
            });
            if (!candidates.length) {
                return doc.body ? doc.body.innerHTML : '';
            }
            candidates.sort((a, b) => b.score - a.score);
            const best = candidates[0].node ? candidates[0].node.cloneNode(true) : doc.body.cloneNode(true);
            if (best && best.querySelectorAll) {
                best.querySelectorAll('header,footer,nav,aside,form,button,input,textarea,select,label,script,style,link,meta,noscript').forEach((node) => {
                    const textLen = (node.textContent || '').trim().length;
                    if (textLen > 150) return;
                    if (node && node.remove) node.remove();
                });
                best.querySelectorAll('[role="navigation"],[role="complementary"],[role="banner"],[role="contentinfo"],[aria-hidden="true"]').forEach((node) => {
                    const textLen = (node.textContent || '').trim().length;
                    if (textLen > 150) return;
                    if (node && node.remove) node.remove();
                });
                best.querySelectorAll('div,section,article,main,span,p').forEach((node) => {
                    const cls = String(node.className || '').trim();
                    const id = String(node.id || '').trim();
                    const role = String(node.getAttribute('role') || '').trim().toLowerCase();
                    const label = String(node.getAttribute('aria-label') || '').trim();
                    const noiseText = `${cls} ${id} ${role} ${label}`;
                    const textLen = (node.textContent || '').trim().length;
                    if (textLen > 150) return;
                    if (RICH_PASTE_NOISE_ROLE_RE.test(noiseText) && node.parentNode) {
                        node.remove();
                    }
                });
            }
            return best ? best.innerHTML : (doc.body ? doc.body.innerHTML : '');
        };
        const sanitizeRichPasteHtml = (html) => {
            let articleHtml = extractRichPasteArticleHtml(html);
            let safeHtml = DOMPurify.sanitize(articleHtml || '', {
                ALLOWED_TAGS: RICH_PASTE_ALLOWED_TAGS,
                ALLOWED_ATTR: RICH_PASTE_ALLOWED_ATTR,
                KEEP_CONTENT: true
            });
            if ((!safeHtml || safeHtml.trim() === '') && html && html.trim() !== '') {
                safeHtml = DOMPurify.sanitize(html, {
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
        const insertClipboardImageBlob = async (blob, alt = 'clipboard-image') => {
            if (!blob) return false;
            const dataUrl = await blobToDataUrl(blob);
            if (!dataUrl) return false;
            insertHtmlIntoRichPasteEditor(`<p><img src="${escapeHtml(dataUrl)}" alt="${escapeHtml(alt)}"></p>`);
            return true;
        };
        const readClipboardRichContent = async () => {
            if (!navigator.clipboard || !navigator.clipboard.read) {
                throw new Error('このブラウザはリッチクリップボード読み取りに対応していません');
            }
            const capture = getRichPasteCapture();
            if (capture) capture.value = '';
            const items = await navigator.clipboard.read();
            if (!items || !items.length) return false;
            let inserted = false;
            for (const item of items) {
                if (!item) continue;
                const itemTypes = Array.from(item.types || []);
                let htmlInserted = false;
                if (itemTypes.includes('text/html')) {
                    const htmlBlob = await item.getType('text/html');
                    const html = await htmlBlob.text();
                    if (html && insertHtmlIntoRichPasteEditor(html)) {
                        inserted = true;
                        htmlInserted = true;
                    }
                }
                if (!htmlInserted && itemTypes.includes('text/plain')) {
                    const textBlob = await item.getType('text/plain');
                    const text = await textBlob.text();
                    if (text) {
                        insertTextIntoRichPasteEditor(text);
                        inserted = true;
                    }
                }
                const imageType = itemTypes.find((type) => type && type.startsWith('image/'));
                if (imageType) {
                    const imageBlob = await item.getType(imageType);
                    if (await insertClipboardImageBlob(imageBlob, 'clipboard-image')) {
                        inserted = true;
                    }
                }
            }
            return inserted;
        };
        const ingestRichPasteClipboardData = async (clipboardData) => {
            if (!clipboardData) return false;
            let inserted = false;
            const html = clipboardData.getData && clipboardData.getData('text/html');
            const text = clipboardData.getData && clipboardData.getData('text/plain');
            let htmlInserted = false;
            if (html) {
                if (insertHtmlIntoRichPasteEditor(html)) {
                    inserted = true;
                    htmlInserted = true;
                }
            }
            if (!htmlInserted && text) {
                insertTextIntoRichPasteEditor(text);
                inserted = true;
            }
            const items = Array.from(clipboardData.items || []);
            const imageFiles = items
                .filter((item) => item && item.kind === 'file')
                .map((item) => item.getAsFile())
                .filter((file) => file && file.type && file.type.startsWith('image/'));
            if (imageFiles.length) {
                for (const imageFile of imageFiles) {
                    try {
                        if (await insertClipboardImageBlob(imageFile, imageFile.name || 'clipboard-image')) {
                            inserted = true;
                        }
                    } catch (e) {}
                }
            }
            return inserted;
        };
        const buildRichPastePdfFilename = () => {
            const now = new Date();
            const pad = (n) => String(n).padStart(2, '0');
            return `clipboard_rich_${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}_${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}.pdf`;
        };
        const getRichPasteProgressElements = () => {
            return {
                container: get('rich-paste-progress-container'),
                bar: get('rich-paste-progress-bar'),
                text: get('rich-paste-progress-text')
            };
        };
        const setRichPasteProgress = (pct, text = null) => {
            const { container, bar, text: textEl } = getRichPasteProgressElements();
            const safePct = Math.max(0, Math.min(100, Number(pct) || 0));
            if (container) {
                container.classList.remove('hidden');
                container.style.setProperty('display', 'block', 'important');
            }
            if (bar) {
                bar.style.width = `${safePct}%`;
                bar.style.transform = 'none';
            }
            if (textEl) {
                textEl.textContent = `${Math.round(safePct)}%`;
            }
            if (text && container) {
                const label = container.querySelector('.text-amber-400');
                if (label) label.innerHTML = `<i class="fas fa-spinner fa-spin"></i> ${escapeHtml(text)}`;
            }
        };
        const hideRichPasteProgress = () => {
            const { container, bar } = getRichPasteProgressElements();
            if (bar) {
                bar.style.transform = 'scaleX(0)';
            }
            if (container) {
                container.classList.add('hidden');
                container.style.display = 'none';
            }
        };
        const inferRichPasteTitle = () => {
            const editor = getRichPasteEditor();
            if (!editor) return 'Clipboard Export';
            const heading = editor.querySelector('h1, h2, h3, h4, h5, h6');
            if (heading && heading.textContent && heading.textContent.trim()) {
                return heading.textContent.trim().slice(0, 48);
            }
            const text = (editor.innerText || '').trim().replace(/\s+/g, ' ');
            if (!text) return 'Clipboard Export';
            return text.slice(0, 48);
        };
        const waitForRichPasteMedia = async (root, timeoutMs = 2500) => {
            if (!root) return;
            const settle = new Promise((resolve) => setTimeout(resolve, Math.max(0, timeoutMs)));
            const loadPromise = Promise.all(Array.from(root.querySelectorAll('img') || []).map((img) => {
                if (!img) return Promise.resolve();
                if (img.complete) return Promise.resolve();
                return new Promise((resolve) => {
                    let done = false;
                    const finish = () => {
                        if (done) return;
                        done = true;
                        resolve();
                    };
                    img.addEventListener('load', finish, { once: true });
                    img.addEventListener('error', finish, { once: true });
                    setTimeout(finish, Math.max(250, Math.min(timeoutMs, 2000)));
                });
            }));
            await Promise.race([loadPromise, settle]);
            if (document.fonts && document.fonts.ready) {
                try {
                    await Promise.race([
                        document.fonts.ready,
                        settle
                    ]);
                } catch (e) {}
            }
        };
        const normalizeRichPastePdfText = (value) => {
            return String(value || '')
                .replace(/\u00a0/g, ' ')
                .replace(/\r\n?/g, '\n')
                .replace(/[ \t\f\v]+/g, ' ')
                .replace(/\n[ \t]+/g, '\n')
                .replace(/[ \t]+\n/g, '\n')
                .replace(/\n{3,}/g, '\n\n')
                .trim();
        };
        const normalizeRichPastePdfCodeText = (value) => {
            return String(value || '').replace(/\u00a0/g, ' ').replace(/\r\n?/g, '\n');
        };
        const collectRichPasteInlineSegments = (node, options = {}) => {
            if (!node) return [];
            const allowLinks = options.allowLinks !== false;
            const segments = [];
            const walk = (current, state) => {
                if (!current) return;
                if (current.nodeType === Node.TEXT_NODE) {
                    const text = current.textContent || '';
                    if (text) {
                        segments.push(Object.assign({}, state, { text }));
                    }
                    return;
                }
                if (current.nodeType !== Node.ELEMENT_NODE) return;
                const tag = String(current.tagName || '').toLowerCase();
                if (RICH_PASTE_NOISE_TAGS.has(tag)) return;
                if (tag === 'br') {
                    segments.push({ text: '\n' });
                    return;
                }
                const newState = Object.assign({}, state);
                if (['b', 'strong'].includes(tag)) newState.bold = true;
                if (['i', 'em'].includes(tag)) newState.italic = true;
                if (tag === 'a' && allowLinks) {
                    newState.link = String(current.getAttribute('href') || '').trim();
                }
                if (tag === 'code') newState.monospace = true;

                Array.from(current.childNodes || []).forEach((child) => walk(child, newState));
            };
            walk(node, { bold: !!options.bold, italic: !!options.italic });
            return segments;
        };
        const collectRichPasteInlineText = (node, options = {}) => {
            const segments = collectRichPasteInlineSegments(node, options);
            return segments.map(s => s.text).join('');
        };
        const collectRichPasteTableRows = (tableEl) => {
            const rows = [];
            Array.from(tableEl.querySelectorAll('tr') || []).forEach((row) => {
                if (row && row.closest && row.closest('table') === tableEl) {
                    rows.push(row);
                }
            });
            return rows;
        };
        const makeRichPasteTableMarkdown = (tableEl) => {
            const caption = tableEl && tableEl.querySelector ? tableEl.querySelector('caption') : null;
            const captionText = caption ? normalizeRichPastePdfText(collectRichPasteInlineText(caption)) : '';
            const rows = collectRichPasteTableRows(tableEl).map((row) => {
                const cells = Array.from(row.children || []).filter((cell) => {
                    const tag = String(cell.tagName || '').toLowerCase();
                    return tag === 'th' || tag === 'td';
                }).map((cell) => normalizeRichPastePdfText(collectRichPasteInlineText(cell)) || ' ');
                return cells;
            }).filter((row) => row.length);
            if (!rows.length) {
                return captionText || '[table]';
            }
            const colCount = rows.reduce((max, row) => Math.max(max, row.length), 0);
            const paddedRows = rows.map((row) => {
                const copy = row.slice(0, colCount);
                while (copy.length < colCount) copy.push(' ');
                return copy;
            });
            const separator = `| ${Array(colCount).fill('---').join(' | ')} |`;
            const lines = [];
            if (captionText) {
                lines.push(`Table: ${captionText}`);
                lines.push('');
            }
            lines.push(`| ${paddedRows[0].join(' | ')} |`);
            lines.push(separator);
            for (let i = 1; i < paddedRows.length; i += 1) {
                lines.push(`| ${paddedRows[i].join(' | ')} |`);
            }
            return lines.join('\n');
        };
        const collectRichPasteListBlocks = (listEl, ordered = false, depth = 0) => {
            const blocks = [];
            const items = Array.from(listEl.children || []).filter((child) => {
                const tag = String(child.tagName || '').toLowerCase();
                return tag === 'li';
            });
            let itemIndex = 1;
            items.forEach((li) => {
                const clone = li.cloneNode(true);
                Array.from(clone.querySelectorAll('ul,ol') || []).forEach((nested) => {
                    try { nested.remove(); } catch (e) {}
                });
                const segments = collectRichPasteInlineSegments(clone);
                if (segments.length > 0) {
                    blocks.push({
                        type: 'list_item',
                        ordered,
                        depth,
                        index: itemIndex,
                        segments: segments
                    });
                }
                Array.from(li.children || []).forEach((child) => {
                    const tag = String(child.tagName || '').toLowerCase();
                    if (tag === 'ul' || tag === 'ol') {
                        blocks.push(...collectRichPasteListBlocks(child, tag === 'ol', depth + 1));
                    }
                });
                itemIndex += 1;
            });
            return blocks;
        };
        const collectRichPastePdfBlocks = (root, depth = 0) => {
            const blocks = [];
            if (!root) return blocks;
            let buffer = [];
            const flushBuffer = () => {
                if (buffer.length === 0) return;
                blocks.push({ type: 'paragraph', segments: [...buffer] });
                buffer = [];
            };
            Array.from(root.childNodes || []).forEach((node) => {
                if (!node) return;
                if (node.nodeType === Node.TEXT_NODE) {
                    const text = (node.textContent || '').replace(/\u00a0/g, ' ');
                    if (text) buffer.push({ text });
                    return;
                }
                if (node.nodeType !== Node.ELEMENT_NODE) return;
                const tag = String(node.tagName || '').toLowerCase();
                if (RICH_PASTE_NOISE_TAGS.has(tag)) return;
                if (tag === 'br') {
                    buffer.push({ text: '\n' });
                    return;
                }
                if (/^h[1-6]$/.test(tag)) {
                    flushBuffer();
                    const segments = collectRichPasteInlineSegments(node);
                    if (segments.length > 0) {
                        blocks.push({ type: 'heading', level: Number(tag.slice(1)) || 1, segments });
                    }
                    return;
                }
                if (tag === 'p') {
                    flushBuffer();
                    const segments = collectRichPasteInlineSegments(node);
                    if (segments.length > 0) {
                        blocks.push({ type: 'paragraph', segments });
                    }
                    return;
                }
                if (tag === 'blockquote') {
                    flushBuffer();
                    const segments = collectRichPasteInlineSegments(node, { italic: true });
                    if (segments.length > 0) {
                        blocks.push({ type: 'blockquote', segments });
                    }
                    return;
                }
                if (tag === 'pre') {
                    flushBuffer();
                    const text = normalizeRichPastePdfCodeText(node.innerText || node.textContent || '');
                    if (text.trim()) {
                        blocks.push({ type: 'code', text });
                    }
                    return;
                }
                if (tag === 'table') {
                    flushBuffer();
                    const text = makeRichPasteTableMarkdown(node);
                    if (text) {
                        blocks.push({ type: 'table', text });
                    }
                    return;
                }
                if (tag === 'ul' || tag === 'ol') {
                    flushBuffer();
                    blocks.push(...collectRichPasteListBlocks(node, tag === 'ol', depth));
                    return;
                }
                if (tag === 'hr') {
                    flushBuffer();
                    blocks.push({ type: 'hr' });
                    return;
                }
                if (tag === 'figure') {
                    flushBuffer();
                    const img = node.querySelector('img');
                    if (img) {
                        blocks.push({
                            type: 'image',
                            src: String(img.getAttribute('src') || '').trim(),
                            alt: String(img.getAttribute('alt') || img.getAttribute('title') || '').trim(),
                            title: String(img.getAttribute('title') || '').trim()
                        });
                    }
                    const figcaption = node.querySelector('figcaption');
                    if (figcaption) {
                        const segments = collectRichPasteInlineSegments(figcaption);
                        if (segments.length > 0) {
                            blocks.push({ type: 'paragraph', segments });
                        }
                    }
                    return;
                }
                if (tag === 'img') {
                    flushBuffer();
                    blocks.push({
                        type: 'image',
                        src: String(node.getAttribute('src') || '').trim(),
                        alt: String(node.getAttribute('alt') || node.getAttribute('title') || '').trim(),
                        title: String(node.getAttribute('title') || '').trim()
                    });
                    return;
                }
                if (tag === 'li') {
                    flushBuffer();
                    blocks.push(...collectRichPasteListBlocks(node, false, depth));
                    return;
                }
                const hasBlockChild = Array.from(node.children || []).some((child) => {
                    const childTag = String(child.tagName || '').toLowerCase();
                    return /^h[1-6]$/.test(childTag) || ['p', 'div', 'section', 'article', 'main', 'blockquote', 'pre', 'table', 'ul', 'ol', 'hr', 'figure', 'img', 'li'].includes(childTag);
                });
                if (hasBlockChild && ['div', 'section', 'article', 'main', 'figure'].includes(tag)) {
                    flushBuffer();
                    blocks.push(...collectRichPastePdfBlocks(node, depth + 1));
                    return;
                }
                const inlineSegments = collectRichPasteInlineSegments(node);
                if (inlineSegments.length > 0) {
                    buffer.push(...inlineSegments);
                }
            });
            flushBuffer();
            return blocks;
        };
        const detectImageMimeType = (dataUrl) => {
            const match = String(dataUrl || '').match(/^data:(image\/[a-z0-9.+-]+);/i);
            return match ? match[1].toLowerCase() : 'image/png';
        };
        const loadRichPasteImageData = async (src, timeoutMs = 3000) => {
            const rawSrc = String(src || '').trim();
            if (!rawSrc) return null;
            if (rawSrc.startsWith('data:image/')) {
                return { dataUrl: rawSrc, mimeType: detectImageMimeType(rawSrc) };
            }
            let resolvedUrl = null;
            try {
                resolvedUrl = new URL(rawSrc, window.location.href);
            } catch (e) {
                return null;
            }
            const allowedSameOrigin = resolvedUrl.origin === window.location.origin;
            if (!allowedSameOrigin) {
                return null;
            }
            const fetchPromise = (async () => {
                try {
                    const response = await fetch(resolvedUrl.toString(), { credentials: 'same-origin', cache: 'force-cache' });
                    if (!response.ok) return null;
                    const blob = await response.blob();
                    const dataUrl = await blobToDataUrl(blob);
                    return {
                        dataUrl,
                        mimeType: blob.type || detectImageMimeType(dataUrl)
                    };
                } catch (e) {
                    return null;
                }
            })();
            return await Promise.race([
                fetchPromise,
                new Promise((resolve) => setTimeout(() => resolve(null), Math.max(250, timeoutMs)))
            ]);
        };
        const buildRichPastePreviewHtml = (mode = 'preview') => {
            const editor = getRichPasteEditor();
            if (!editor) return '';
            const title = inferRichPasteTitle();
            const createdAt = new Date().toLocaleString('ja-JP');
            const contentHtml = sanitizeRichPasteHtml(editor.innerHTML || '');
            const isPdf = mode === 'pdf';
            return `<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${escapeHtml(title)} - Preview</title>
  <style>
    body { margin: 0; background: ${isPdf ? '#ffffff' : '#0b1220'}; color: ${isPdf ? '#111827' : '#e5e7eb'}; font-family: "Noto Sans JP", system-ui, sans-serif; }
    .page { max-width: ${isPdf ? '794px' : '920px'}; margin: 0 auto; padding: ${isPdf ? '28px 30px 36px' : '24px'}; }
    .card { background: ${isPdf ? '#ffffff' : '#111827'}; border: 1px solid ${isPdf ? '#dbe3ee' : '#243244'}; border-radius: 18px; padding: 20px; box-shadow: ${isPdf ? 'none' : '0 18px 45px rgba(0,0,0,0.35)'}; }
    .title { margin: 0; font-size: ${isPdf ? '22px' : '24px'}; line-height: 1.35; color: ${isPdf ? '#0f172a' : '#f8fafc'}; }
    .meta { margin-top: 8px; color: ${isPdf ? '#64748b' : '#94a3b8'}; font-size: 12px; }
    .content { margin-top: 18px; color: ${isPdf ? '#111827' : '#e5e7eb'}; font-size: ${isPdf ? '15px' : '15px'}; line-height: 1.85; word-break: break-word; overflow-wrap: anywhere; }
    .content img, .content video, .content iframe, .content table, .content pre, .content blockquote { max-width: 100%; }
    .content table { display: block; overflow-x: auto; border-collapse: collapse; }
    .content th, .content td { border: 1px solid ${isPdf ? '#cbd5e1' : '#334155'}; padding: 8px 10px; }
    .content pre { padding: 14px 16px; border-radius: 14px; background: ${isPdf ? '#f8fafc' : '#020617'}; overflow: auto; color: ${isPdf ? '#111827' : 'inherit'}; }
    .content blockquote { margin: 1em 0; padding: 12px 16px; border-left: 4px solid ${isPdf ? '#f59e0b' : '#f59e0b'}; background: ${isPdf ? '#fff9eb' : 'rgba(245,158,11,0.08)'}; border-radius: 12px; color: ${isPdf ? '#334155' : 'inherit'}; }
    .content a { color: ${isPdf ? '#0f766e' : '#7dd3fc'}; }
    .toolbar { display:${isPdf ? 'none' : 'flex'}; gap:10px; margin-top: 16px; flex-wrap: wrap; }
    .toolbar button { border: 1px solid #334155; background: #0f172a; color: #e2e8f0; border-radius: 999px; padding: 8px 12px; cursor: pointer; }
    ${isPdf ? '.card { border-radius: 0; } .page { max-width: none; padding: 0; }' : ''}
  </style>
</head>
<body>
  <div class="page">
    <div class="card">
      <h1 class="title">${escapeHtml(title)}</h1>
      <div class="meta">Clipboard import | ${escapeHtml(createdAt)} | 本文確認用プレビュー</div>
      <div class="toolbar">
        <button onclick="window.close()">閉じる</button>
      </div>
      <div class="content">${contentHtml || '<p>内容がありません。</p>'}</div>
    </div>
  </div>
</body>
</html>`;
        };
        const openSandboxedHtmlTab = (html) => {
            const frameHtml = JSON.stringify(String(html || ''))
                .replace(/</g, '\\u003c')
                .replace(/\u2028/g, '\\u2028')
                .replace(/\u2029/g, '\\u2029');
            const shell = `<!doctype html><html><head><meta charset="utf-8"><meta name="referrer" content="no-referrer"><style>html,body,iframe{width:100%;height:100%;margin:0;border:0;background:#fff}body{overflow:hidden}</style></head><body><iframe id="preview" sandbox="allow-scripts allow-forms allow-modals allow-popups" referrerpolicy="no-referrer"></iframe><script>document.getElementById('preview').srcdoc=${frameHtml};<\/script></body></html>`;
            const blob = new Blob([shell], { type: 'text/html;charset=utf-8' });
            const url = URL.createObjectURL(blob);
            const tab = window.open(url, '_blank', 'noopener,noreferrer');
            if (!tab) {
                URL.revokeObjectURL(url);
                return false;
            }
            setTimeout(() => URL.revokeObjectURL(url), 60000);
            return true;
        };
        const openRichPastePreviewTab = () => {
            const html = buildRichPastePreviewHtml('preview');
            if (!html) {
                showToast('確認する内容がありません', 'warning', true);
                return;
            }
            if (!openSandboxedHtmlTab(html)) {
                showToast('別タブの表示に失敗しました', 'error', true);
            }
        };
        const renderRichPastePdfBlob = async () => {
            const progressContainer = get('rich-paste-progress-container');
            const progressBar = get('rich-paste-progress-bar');
            const progressText = get('rich-paste-progress-text');
            const updateProgress = (pct) => {
                const safePct = Math.max(0, Math.min(100, Number(pct) || 0));
                if (progressBar) {
                    progressBar.style.width = '100%';
                    progressBar.style.transformOrigin = 'left center';
                    // Ensure smooth transition
                    if (!progressBar.style.transition || progressBar.style.transition.indexOf('transform') === -1) {
                        progressBar.style.transition = 'transform 0.45s cubic-bezier(0.22, 1, 0.36, 1)';
                    }
                    progressBar.style.transform = `scaleX(${safePct / 100})`;
                    progressBar.style.willChange = 'transform';
                }
                if (progressText) progressText.innerText = `${Math.round(safePct)}%`;
            };

            // Show progress bar immediately
            if (progressContainer) {
                progressContainer.classList.remove('hidden');
                progressContainer.style.setProperty('display', 'block', 'important');
            }
            
            if (progressBar) {
                // Initialize without transition to prevent initial jump
                progressBar.style.transition = 'none';
                progressBar.style.width = '100%';
                progressBar.style.transformOrigin = 'left center';
                progressBar.style.transform = 'scaleX(0)';
                // Force reflow
                void progressBar.offsetHeight;
                // Set the smooth transition
                progressBar.style.transition = 'transform 0.45s cubic-bezier(0.22, 1, 0.36, 1)';
            }
            
            updateProgress(0);

            // Give browser a moment to paint the progress bar
            await new Promise(res => requestAnimationFrame(() => setTimeout(res, 150)));

            const editor = getRichPasteEditor();
            if (!editor) throw new Error('PDF化する内容がありません');
            const title = inferRichPasteTitle();
            const contentHtml = sanitizeRichPasteHtml(editor.innerHTML || '');
            await ensurePdfLibraries();
            const JsPdfCtor = window.jspdf && window.jspdf.jsPDF ? window.jspdf.jsPDF : null;
            if (!JsPdfCtor) throw new Error('jsPDF ライブラリが読み込まれていません');
            const html2canvasFn = window.html2canvas;
            if (typeof html2canvasFn !== 'function') throw new Error('html2canvas ライブラリが読み込まれていません');

            updateProgress(5);

            const wrapper = document.createElement('div');
            wrapper.style.position = 'absolute';
            wrapper.style.left = '-10000px';
            wrapper.style.top = '-20000px';
            wrapper.style.width = '794px';
            wrapper.style.background = '#ffffff';
            wrapper.style.boxSizing = 'border-box';
            wrapper.style.fontFamily = '"Noto Sans JP", "Segoe UI", "Helvetica Neue", Arial, sans-serif';
            wrapper.innerHTML = `
                <style>
                    /* Proper Light Mode Reset */
                    .pdf-root-wrapper {
                        background-color: #ffffff !important;
                        color: #000000 !important;
                        padding: 40px;
                        width: 794px;
                        min-height: 1123px;
                        color-scheme: light !important;
                        line-height: 1.5 !important;
                        text-align: left !important;
                        font-size: 15px !important;
                    }
                    .pdf-root-wrapper * {
                        box-sizing: border-box !important;
                        text-shadow: none !important;
                        box-shadow: none !important;
                        letter-spacing: normal !important;
                        word-spacing: normal !important;
                        vertical-align: baseline !important;
                    }
                    .pdf-root-wrapper h1,
                    .pdf-root-wrapper h2,
                    .pdf-root-wrapper h3,
                    .pdf-root-wrapper h4,
                    .pdf-root-wrapper h5,
                    .pdf-root-wrapper h6 {
                        line-height: 1.25 !important;
                        margin: 1.1em 0 0.55em 0 !important;
                    }
                    .pdf-title { 
                        font-size: 26px !important; 
                        font-weight: bold !important; 
                        margin: 0 0 15px 0 !important; 
                        border-bottom: 2px solid #000 !important; 
                        padding-bottom: 10px !important; 
                        line-height: 1.2 !important;
                    }
                    .pdf-meta { 
                        font-size: 12px !important; 
                        color: #555 !important; 
                        margin-bottom: 30px !important; 
                    }
                    .pdf-content { 
                        font-size: 15px !important; 
                        line-height: 1.6 !important; 
                        color: inherit !important;
                    }
                    .pdf-content p { margin: 0 0 1em 0 !important; }
                    .pdf-content img { max-width: 100% !important; height: auto !important; display: block !important; margin: 20px auto !important; }
                    .pdf-content video,
                    .pdf-content iframe {
                        max-width: 100% !important;
                        display: block !important;
                        margin: 20px auto !important;
                    }
                    .pdf-content table { width: 100% !important; border-collapse: collapse !important; margin: 20px 0 !important; table-layout: fixed !important; border: 1px solid #666 !important; }
                    .pdf-content th, .pdf-content td { border: 1px solid #666 !important; padding: 10px !important; text-align: left !important; word-break: break-word !important; vertical-align: top !important; }
                    .pdf-content th { background-color: #f0f0f0 !important; font-weight: bold !important; }
                    .pdf-content pre { 
                        background-color: #f5f5f5 !important; 
                        border: 1px solid #ccc !important; 
                        padding: 15px !important; 
                        border-radius: 5px !important; 
                        white-space: pre-wrap !important; 
                        word-break: break-word !important;
                        font-family: "Noto Sans Mono", monospace !important; 
                        font-size: 13px !important; 
                        margin: 1.2em 0 !important; 
                        line-height: 1.4 !important;
                        display: block !important;
                        width: 100% !important;
                        overflow-wrap: anywhere !important;
                    }
                    .pdf-content code { 
                        font-family: "Noto Sans Mono", monospace !important; 
                        background-color: #eeeeee !important; 
                        padding: 1px 4px !important; 
                        border-radius: 3px !important; 
                        font-size: 0.9em !important; 
                        display: inline-block !important;
                        line-height: 1.2 !important;
                        margin: 1px 0 !important;
                        vertical-align: middle !important;
                    }
                    .pdf-content pre code {
                        display: block !important;
                        padding: 0 !important;
                        margin: 0 !important;
                        border-radius: 0 !important;
                        background: transparent !important;
                        color: inherit !important;
                        font-size: inherit !important;
                        line-height: inherit !important;
                        white-space: pre-wrap !important;
                    }
                    .pdf-content pre code * {
                        background: transparent !important;
                        color: inherit !important;
                    }
                    .pdf-content blockquote { 
                        border-left: 5px solid #ddd !important; 
                        padding: 5px 0 5px 20px !important; 
                        margin: 1em 0 !important; 
                        font-style: italic !important; 
                        color: #333 !important;
                    }
                    .pdf-content a {
                        color: #0f766e !important;
                        text-decoration: underline !important;
                    }
                    .pdf-content ul,
                    .pdf-content ol {
                        margin: 0 0 1em 0 !important;
                        padding-left: 1.5em !important;
                    }
                    .pdf-content li { margin-bottom: 0.4em !important; }
                </style>
                <div class="pdf-root-wrapper">
                    <div class="pdf-title">${escapeHtml(title)}</div>
                    <div class="pdf-meta">Created at: ${new Date().toLocaleString('ja-JP')}</div>
                    <div class="pdf-content">${contentHtml}</div>
                </div>
            `;
            document.body.appendChild(wrapper);
            await waitForRichPasteMedia(wrapper, 4000);
            updateProgress(15);

            try {
                const pdf = new JsPdfCtor({ unit: 'mm', format: 'a4', orientation: 'portrait', compress: true });
                const pdfWidth = pdf.internal.pageSize.getWidth();
                const pageHeightMm = pdf.internal.pageSize.getHeight();
                
                // Get total height of the wrapper
                const totalHeight = wrapper.scrollHeight || wrapper.offsetHeight;
                const chunkHeight = 1500; // Chunk size in pixels for rendering
                let currentY = 0;
                let isFirstPage = true;
                
                const totalChunks = Math.ceil(totalHeight / chunkHeight);
                let chunkIndex = 0;

                while (currentY < totalHeight) {
                    if (richPasteAbortController && richPasteAbortController.signal.aborted) {
                        throw new DOMException('Aborted', 'AbortError');
                    }
                    const h = Math.min(chunkHeight, totalHeight - currentY);
                    const canvas = await new Promise((resolve, reject) => {
                        const timer = setTimeout(() => reject(new Error('PDF chunk rendering timed out')), 120000);
                        html2canvasFn(wrapper, {
                            scale: 1, // Use scale 1 for large documents to avoid canvas limits
                            useCORS: true,
                            allowTaint: true,
                            backgroundColor: '#ffffff',
                            logging: false,
                            x: 0,
                            y: currentY,
                            width: 794,
                            height: h,
                            windowWidth: 794,
                            onclone: (clonedDoc) => {
                                const root = clonedDoc.querySelector('.pdf-root-wrapper');
                                if (root) {
                                    root.style.position = 'relative';
                                    root.style.left = '0';
                                    root.style.top = '0';
                                }
                            }
                        }).then(c => { clearTimeout(timer); resolve(c); }).catch(e => { clearTimeout(timer); reject(e); });
                    });

                    const imgData = canvas.toDataURL('image/jpeg', 0.9);
                    const imgProps = pdf.getImageProperties(imgData);
                    const chunkHeightMm = (imgProps.height * pdfWidth) / imgProps.width;
                    
                    if (!isFirstPage) {
                        pdf.addPage();
                    }

                    pdf.addImage(imgData, 'JPEG', 0, 0, pdfWidth, chunkHeightMm);
                    isFirstPage = false;
                    currentY += chunkHeight;
                    chunkIndex++;
                    
                    // Update progress after each chunk
                    const progress = Math.min(100, 15 + Math.round((chunkIndex / totalChunks) * 85));
                    updateProgress(progress);
                    
                    // Small delay to prevent UI freezing and allow progress update
                    await new Promise(res => setTimeout(res, 100));
                }

                updateProgress(100);

                const blob = pdf.output('blob');
                return { blob, fileName: buildRichPastePdfFilename() };
            } finally {
                if (progressContainer) {
                    progressContainer.classList.add('hidden');
                    progressContainer.style.display = 'none';
                }
                if (wrapper && wrapper.parentNode) document.body.removeChild(wrapper);
            }
        };
        const createRichPastePdfBlob = async () => {
            return await renderRichPastePdfBlob();
        };
        const buildRichPasteServerPayload = () => {
            const editor = getRichPasteEditor();
            if (!editor) throw new Error('PDF化する内容がありません');
            const rawHtml = String(editor.innerHTML || '').trim();
            const rawText = String(editor.textContent || '').trim();
            return {
                title: inferRichPasteTitle(),
                html: rawHtml || (rawText ? `<p>${escapeHtml(rawText).replace(/\n/g, '<br/>')}</p>` : ''),
                created_at: new Date().toLocaleString('ja-JP')
            };
        };
        const attachRichPastePdfAndSend = async (pdfBlob, fileName, promptText, previousPrompt) => {
            const beforePaths = new Set(collectAttachmentItemsForSend().map((it) => it.path));
            const pdfFile = new File([pdfBlob], fileName, { type: 'application/pdf', lastModified: Date.now() });
            const promptInput = get('prompt-input');
            if (promptInput) promptInput.value = promptText;
            await handleFiles([pdfFile], { openModal: false });
            const afterPaths = collectAttachmentItemsForSend().map((it) => it.path);
            const attached = afterPaths.some((path) => !beforePaths.has(path));
            if (!attached) {
                if (promptInput) promptInput.value = previousPrompt;
                throw new Error('PDFの添付に失敗しました');
            }
            const sendPromise = sendMessage();
            clearRichPasteEditor(true);
            window.closeRichPasteModal();
            showToast('PDFを添付して送信を開始しました', 'success');
            if (sendPromise && typeof sendPromise.catch === 'function') {
                sendPromise.catch(() => {});
            }
        };
        const openRichPasteModal = async () => {
            await ensureUserSettingsSnapshot();
            showModal('rich-paste-modal');
            if (location.pathname !== '/paste') {
                history.pushState({ modal: 'paste' }, '', '/paste');
            }
            const prompt = getRichPastePrompt();
            if (prompt) {
                richPastePromptPreferenceSyncing = true;
                prompt.value = getRichPasteEffectivePrompt(userSettingsSnapshot);
                richPastePromptPreferenceSyncing = false;
            }
            updateRichPasteStatus();
            setTimeout(() => focusRichPasteEditor(), 80);
        };
        window.closeRichPasteModal = (skipHistory = false) => {
            hideModal('rich-paste-modal');
            if (!skipHistory && location.pathname === '/paste') {
                history.back();
            }
        };
        const sendRichPasteToModel = async (options = {}) => {
            const serverSide = !!(options && options.serverSide);
            if (abortController || richPasteAbortController) {
                showToast('回答生成中またはPDF変換中です。完了までお待ちいただくか、停止してください。', 'warning', true);
                return;
            }
            const editor = getRichPasteEditor();
            const promptEl = getRichPastePrompt();
            const sendBtn = serverSide ? get('rich-paste-send-server-btn') : get('rich-paste-send-btn');
            const cancelBtn = get('rich-paste-cancel-btn');

            if (!editor || !editor.innerText || !editor.innerText.trim()) {
                showToast('貼り付ける内容を入力してください', 'warning', true);
                return;
            }

            richPasteAbortController = new AbortController();
            if (cancelBtn) {
                cancelBtn.onclick = () => {
                    if (richPasteAbortController) {
                        richPasteAbortController.abort();
                        showToast('PDF変換をキャンセルしました', 'info');
                    }
                };
            }

            const promptText = (promptEl && promptEl.value && promptEl.value.trim()) ? promptEl.value.trim() : RICH_PASTE_DEFAULT_PROMPT;
            const previousPrompt = get('prompt-input') ? get('prompt-input').value : '';
            if (sendBtn) sendBtn.disabled = true;
            try {
                // Clear any existing generation toast
                const stack = get('toast-stack');
                if (stack) {
                    stack.querySelectorAll('.toast').forEach(t => {
                        if (t.innerText.includes('PDFを生成しています') || t.innerText.includes('サーバー側でPDFを生成しています')) t.remove();
                    });
                }
                if (serverSide) {
                    showToast('サーバー側でPDFを生成しています...', 'info', true);
                    setRichPasteProgress(2, 'サーバー側でPDFを生成しています...');
                } else {
                    showToast('PDFを生成しています...', 'info', true);
                }
                if (serverSide) {
                    if (!RICH_PASTE_PDF_SERVER_ROUTE) {
                        throw new Error('サーバー側PDF生成のURLが見つかりません');
                    }
                    const payload = buildRichPasteServerPayload();
                    setRichPasteProgress(10, 'サーバーへ送信中...');
                    const response = await apiFetch(RICH_PASTE_PDF_SERVER_ROUTE, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload),
                        signal: richPasteAbortController.signal
                    });
                    setRichPasteProgress(60, 'PDFを受信中...');
                    if (!response.ok) {
                        let message = '';
                        try {
                            const data = await response.json();
                            message = data && (data.message || data.error) ? String(data.message || data.error) : '';
                        } catch (e) {
                            try {
                                message = await response.text();
                            } catch (err) {
                                message = '';
                            }
                        }
                        if (message === 'missing_html') {
                            throw new Error('サーバーへ送るHTMLが空です。クリップボード内容の取り込みを先に行ってください');
                        }
                        throw new Error(message ? `サーバーPDF生成に失敗しました: ${message}` : 'サーバーPDF生成に失敗しました');
                    }
                    setRichPasteProgress(75, 'PDFを添付中...');
                    const blob = await response.blob();
                    const fileName = response.headers.get('X-Rich-Paste-Filename') || buildRichPastePdfFilename();
                    const downloadOnly = !!(get('rich-paste-download-only') && get('rich-paste-download-only').checked);
                    
                    if (downloadOnly) {
                        setRichPasteProgress(90, 'ダウンロード中...');
                        downloadBlob(blob, fileName);
                        showToast('PDFをダウンロードしました', 'success');
                        hideModal('rich-paste-modal', { skipConfirm: true });
                    } else {
                        await attachRichPastePdfAndSend(blob, fileName, promptText, previousPrompt);
                    }
                    setRichPasteProgress(100, '完了');
                    setTimeout(() => hideRichPasteProgress(), 400);
                } else {
                    const generated = await createRichPastePdfBlob();
                    const downloadOnly = !!(get('rich-paste-download-only') && get('rich-paste-download-only').checked);
                    if (downloadOnly) {
                        downloadBlob(generated.blob, generated.fileName);
                        showToast('PDFをダウンロードしました', 'success');
                        hideModal('rich-paste-modal', { skipConfirm: true });
                    } else {
                        await attachRichPastePdfAndSend(generated.blob, generated.fileName, promptText, previousPrompt);
                    }
                }
            } catch (e) {
                if (e.name === 'AbortError') {
                    console.log('PDF generation aborted by user');
                    if (serverSide) {
                        setRichPasteProgress(0, 'キャンセルされました');
                        setTimeout(() => hideRichPasteProgress(), 800);
                    }
                    return; // Early return for manual cancel
                }
                if (get('prompt-input')) get('prompt-input').value = previousPrompt;
                const msg = (e && e.message) ? e.message : 'PDF化して送信できませんでした';
                showToast(msg, 'error', true);
                if (serverSide) {
                    setRichPasteProgress(0, '失敗しました');
                    setTimeout(() => hideRichPasteProgress(), 1200);
                }
            } finally {
                if (sendBtn) sendBtn.disabled = false;
                richPasteAbortController = null;
            }
        };
        const csrfToken = document.querySelector('meta[name="csrf-token"]').content;
        const apiFetch = (url, opts = {}) => {
            const method = (opts.method || 'GET').toUpperCase();
            const headers = Object.assign({}, opts.headers || {});
            if (!['GET', 'HEAD', 'OPTIONS'].includes(method)) {
                headers['X-CSRF-Token'] = csrfToken;
            }
            const credentials = opts.credentials || 'include';
            return fetch(url, Object.assign({}, opts, { headers, credentials }));
        };

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
        let lastClientDebugEnabled = null;
        const isClientDebugLogEnabled = () => {
            const settingEl = get('set-client-debug-log');
            return !!(settingEl && settingEl.checked);
        };
        const sendClientDebugLog = (level, message) => {
            if (!isClientDebugLogEnabled()) return;
            const payload = {
                level: String(level || 'info'),
                message: String(message || '')
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
        const ATTACHMENT_MAX_FILES = Number(CHAT_CONFIG.attachmentMaxFiles) || 30;
        const UPLOAD_CONCURRENCY = Math.max(1, Number(CHAT_CONFIG.uploadConcurrency) || 3);
        const TEMP_CHAT_TIMEOUT_MIN_SECONDS = 10;
        const TEMP_CHAT_TIMEOUT_MAX_SECONDS = 3600;
        const TEMP_CHAT_DEFAULT_TIMEOUT_SECONDS = 90;
        const TEMP_CHAT_HEARTBEAT_MIN_MS = 4000;
        const TEMP_CHAT_HEARTBEAT_MAX_MS = 15000;
        const CONNECTION_CHECK_INTERVAL_MS = 15000;
        const CONNECTION_CHECK_FAST_INTERVAL_MS = 4000;
        const CONNECTION_CHECK_TIMEOUT_MS = 5000;
        const CONNECTION_UNSTABLE_LATENCY_MS = 1800;
        const CONNECTION_FAIL_TO_OFFLINE = 2;
        const CONNECTION_UNSTABLE_HOLD_MS = 45000;
        const CONNECTION_RECOVERED_BANNER_MS = 5000;
        let activeGem = null, editingGemUuid = null, currentImageUrls = [], currentMaskImage = null, abortController = null, richPasteAbortController = null, userAutoScroll = true, searchTimeout;
        
        // Prompt History
        let promptHistory = [];
        let historyIndex = -1;
        let tempPrompt = "";
         
        const markerAppliedUploads = new Set();
        const attachmentSourceByPath = new Map();
        const attachmentNameByPath = new Map();
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
        const threadGemMap = {};
        let pendingGemForNewThread = null;
        let loadedGems = [];
        let currentJobId = null; 
        let currentThreadPending = null;
        let currentVisionModel = null;
        let activeStreamingBubbleId = null;
        let manualStopContext = null;
        let manualStopSeq = 0;
        let isStopMode = false;
        const suppressedPendingJobIds = new Set();
        let editingMessageId = null; // Track message being edited
        const messageStore = {}, lib = { modal: get('lib-modal'), grid: get('lib-grid'), files: [], selected: new Set(), attachMode: false, searchQuery: '' };
        const LIB_SORT_KEY = 'lib_sort_order';
        let threadPage = 1, threadLoading = false, hasMoreThreads = true;
        let threadObserver = null;
        let currentQuote = "";
        let currentThreadTitle = null;
        let temporaryChatEnabled = false;
        let temporaryChatTimeoutSeconds = TEMP_CHAT_DEFAULT_TIMEOUT_SECONDS;
        let tempChatExpiresAtMs = null;
        let tempChatHeartbeatTimer = null;
        let tempChatHeartbeatIntervalMs = 0;
        let tempChatHeartbeatInFlight = false;
        let tempChatHeaderTicker = null;
        let connectionCheckTimer = null;
        let connectionCheckIntervalMs = 0;
        let connectionCheckInFlight = false;
        let connectionConsecutiveFail = 0;
        let connectionUnstableUntil = 0;
        let connectionStatus = 'unknown';
        let connectionRecoveredHideTimer = null;
        let enterToSend = CHAT_CONFIG.enterToSend;
        let autoSearchOnLinks = CHAT_CONFIG.autoSearchOnLinks;
        let useSwCache = CHAT_CONFIG.useSwCache;
        let compactPromptMode = CHAT_CONFIG.compactPromptMode;
        const CANVAS_MODE_STORAGE_KEY = 'canvas_mode_enabled_v1';
        let canvasModeEnabled = false;
        const canvasPreviewState = {
            blocks: [],
            rawText: '',
            renderText: '',
            selectedIndex: -1,
            selectedKey: '',
            mobileView: 'preview',
            panelAnimationToken: 0,
            panelHideTimer: null,
            lastCanvasData: null
        };
        try {
            canvasModeEnabled = localStorage.getItem(CANVAS_MODE_STORAGE_KEY) === 'true';
        } catch (e) {
            canvasModeEnabled = false;
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
                        inlineMath: [['\\(', '\\)']],
                        displayMath: [['$$', '$$']],
                        processEscapes: true
                    },
                    options: {
                        ignoreHtmlClass: 'tex2jax_ignore',
                        processHtmlClass: 'tex2jax_process'
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
            return t.includes('$$') || t.includes('\\(') || t.includes('\\[');
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
                    if (window.MathJax && typeof window.MathJax.typesetPromise === 'function') {
                        return window.MathJax.typesetPromise([container]).catch(() => {});
                    }
                })
                .catch(() => {});
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
                        ? '<i class="fas fa-chevron-down"></i> Expand'
                        : '<i class="fas fa-chevron-up"></i> Collapse';
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
                            ? '<i class="fas fa-chevron-down"></i> Expand'
                            : '<i class="fas fa-chevron-up"></i> Collapse';
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
        function applyPromptControlMode() {
            const details = get('prompt-details-controls');
            const toggleBtn = get('prompt-controls-toggle-btn');
            const toggleText = get('prompt-controls-toggle-text');
            const toggleIcon = get('prompt-controls-toggle-icon');
            const row = get('prompt-controls-row');
            if (!details || !toggleBtn) return;
            const showDetails = !compactPromptMode || promptControlsExpanded;
            if (row) row.classList.toggle('compact-collapsed', compactPromptMode && !showDetails);
            
            if (compactPromptMode) {
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

            if (compactPromptMode) {
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
            if (!compactPromptMode) {
                promptControlsExpanded = true;
            } else if (!keepExpanded) {
                promptControlsExpanded = false;
            }
            applyPromptControlMode();
        }
        function togglePromptControlDetails() {
            if (!compactPromptMode) return;
            promptControlsExpanded = !promptControlsExpanded;
            applyPromptControlMode();
        }
        function applyChatDefaults(d) {
            if (!d) return;
            applyTemporaryChatTimeoutSeconds(d.temp_chat_timeout_seconds);
            if (chatDefaultsLoaded) return;
            const useLast = !!d.use_last_chat_settings;
            const src = useLast ? {
                model: d.last_model,
                enable_search: d.last_enable_search,
                enable_url_context: d.last_enable_url_context,
                enable_maps: d.last_enable_maps,
                enable_python: d.last_enable_python,
                enable_thinking: d.last_enable_thinking,
                thinking_level: d.last_thinking_level,
                thinking_budget: d.last_thinking_budget,
                reasoning_effort: d.last_reasoning_effort,
                enable_system_prompt: d.last_enable_system_prompt,
                safety_setting: d.last_safety_setting
            } : {
                model: d.default_model,
                enable_search: d.default_enable_search,
                enable_url_context: d.default_enable_url_context,
                enable_maps: d.default_enable_maps,
                enable_python: d.default_enable_python,
                enable_thinking: d.default_enable_thinking,
                thinking_level: d.default_thinking_level,
                thinking_budget: d.default_thinking_budget,
                reasoning_effort: d.default_reasoning_effort,
                enable_system_prompt: d.default_enable_system_prompt,
                safety_setting: d.default_safety_setting
            };
            const s = (v, fallback) => (v === undefined || v === null || v === "") ? fallback : v;
            if (src.model) selectModelById(src.model);
            if (get('enable-search')) get('enable-search').checked = !!s(src.enable_search, get('enable-search').checked);
            if (get('enable-url-context')) get('enable-url-context').checked = !!s(src.enable_url_context, get('enable-url-context').checked);
            if (get('enable-maps')) get('enable-maps').checked = !!s(src.enable_maps, get('enable-maps').checked);
            if (get('enable-python')) get('enable-python').checked = !!s(src.enable_python, get('enable-python').checked);
            if (get('enable-thinking')) get('enable-thinking').checked = !!s(src.enable_thinking, get('enable-thinking').checked);
            if (get('thinking-level')) get('thinking-level').value = s(src.thinking_level, get('thinking-level').value || "high");
            if (get('thinking-budget')) get('thinking-budget').value = s(src.thinking_budget, get('thinking-budget').value || 4096);
            if (get('reasoning-effort')) get('reasoning-effort').value = s(src.reasoning_effort, get('reasoning-effort').value || "medium");
            if (get('enable-sys-prompt')) get('enable-sys-prompt').checked = !!s(src.enable_system_prompt, get('enable-sys-prompt').checked);
            if (get('safety-setting')) get('safety-setting').value = s(src.safety_setting, get('safety-setting').value || "default");
            chatDefaultsLoaded = true;
            toggleOptions();
        }
        function setEditUi(active) {
            const bar = get('edit-bar');
            if (!bar) return;
            if (active) {
                bar.classList.remove('hidden');
                bar.classList.add('flex');
            } else {
                bar.classList.add('hidden');
                bar.classList.remove('flex');
            }
            updatePromptPlaceholder();
        }
        function cancelEdit() {
            editingMessageId = null;
            currentParentId = currentLeafId || null;
            const input = get('prompt-input');
            if (input) {
                input.value = '';
                input.style.height = 'auto';
            }
            currentImageUrls = [];
            get('file-preview').classList.add('hidden');
            get('file-input').value = '';
            clearQuote();
            setEditUi(false);
        }
        function beginEditMessage(id, autoSend = false) {
            const text = messageStore[id];
            if (text === undefined || text === null) return;
            const input = get('prompt-input');
            input.value = text || '';
            input.focus();
            input.style.height = 'auto';
            input.style.height = input.scrollHeight + 'px';
            
            const msg = allMessages.find(m => m.id == id);
            const meta = messageMeta[id] || {};
            
            // Set parent_id for branching
            if (msg) {
                currentParentId = (msg.parent_id === undefined ? null : msg.parent_id);
            } else if (meta.parent_id !== undefined) {
                currentParentId = meta.parent_id;
            }

            editingMessageId = id;
            setEditUi(true);

            const imageUrl = msg ? msg.image_url : meta.image_url;
            if (imageUrl) {
                try {
                    const imgs = JSON.parse(imageUrl);
                    if (Array.isArray(imgs) && imgs.length) {
                        currentImageUrls = imgs.map((u) => {
                            let src = 'unknown';
                            let path = u;
                            if (u && typeof u === 'object') {
                                src = normalizeAttachmentSource(u.source);
                                path = u.filepath || u.path || u.url || u.file || '';
                            }
                            const norm = normalizeAttachmentPath(path);
                            if (norm) setAttachmentSourceForPath(norm, src);
                            return norm;
                        }).filter(Boolean);
                        get('file-preview').classList.remove('hidden');
                        get('file-name').innerText = `${currentImageUrls.length} files ready`;
                    } else {
                        currentImageUrls = [];
                        get('file-preview').classList.add('hidden');
                        get('file-input').value = '';
                    }
                } catch (e) {
                    currentImageUrls = [];
                    get('file-preview').classList.add('hidden');
                    get('file-input').value = '';
                }
            } else {
                currentImageUrls = [];
                get('file-preview').classList.add('hidden');
                get('file-input').value = '';
            }

            const quoteText = msg ? msg.quote_text : meta.quote_text;
            if (quoteText) {
                currentQuote = quoteText;
                get('quote-text-display').innerText = currentQuote;
                get('quote-bar').classList.add('visible');
            } else {
                clearQuote();
            }
            schedulePromptTokenEstimate(true);
            if (autoSend) sendMessage();
        }
        function playSendAnimation() {
            const btn = get('send-btn');
            if (!btn) return;
            btn.classList.remove('fly');
            void btn.offsetWidth;
            btn.classList.add('fly');
        }
        function setSendBtnToStopMode() {
            const btn = get('send-btn');
            if (!btn) return;
            btn.onclick = stopGeneration;
            isStopMode = true;
            btn.disabled = false;
            const applyStopUi = () => {
                if (!btn || !isStopMode) return;
                btn.classList.add('stop-mode');
                btn.innerHTML = '<span style="font-size:20px;line-height:1;color:#fff;">■</span>';
                btn.classList.add('btn-swap');
                setTimeout(() => btn.classList.remove('btn-swap'), 300);
            };
            if (btn.classList.contains('fly')) {
                const onEnd = (e) => {
                    if (e.animationName !== 'sendBtnPop') return;
                    btn.removeEventListener('animationend', onEnd);
                    applyStopUi();
                };
                btn.addEventListener('animationend', onEnd);
                setTimeout(applyStopUi, 700);
            } else {
                applyStopUi();
            }
        }
        function setSendBtnToSendMode() {
            const btn = get('send-btn');
            if (!btn) return;
            btn.classList.remove('stop-mode', 'fly', 'btn-swap');
            btn.innerHTML = '<i class="fas fa-paper-plane"></i>';
            btn.classList.add('btn-swap');
            setTimeout(() => btn.classList.remove('btn-swap'), 300);
            btn.onclick = sendMessage;
            isStopMode = false;
        }
        async function stopGeneration() {
            const stopThreadId = (currentThreadId !== null && currentThreadId !== undefined && currentThreadId !== '') ? String(currentThreadId) : null;
            const stopJobId = normalizeJobIdForUi(currentJobId);
            const stopSeq = ++manualStopSeq;
            const partialSnapshot = captureStoppedPartialBubbleSnapshot(getActiveStreamingBubbleElement());
            manualStopContext = { seq: stopSeq, threadId: stopThreadId, jobId: stopJobId, partialSnapshot };
            if (stopJobId) suppressPendingJob(stopJobId);
            if(abortController) abortController.abort();
            try {
                if(stopJobId || stopThreadId) {
                    const stopPayload = {};
                    if (stopJobId) stopPayload.job_id = stopJobId;
                    if (stopThreadId) stopPayload.thread_id = stopThreadId;
                    const stopRes = await apiFetch("/api/stop_chat", {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(stopPayload)});
                    const stopData = await stopRes.json().catch(() => ({}));
                    const resolvedStopJobId = normalizeJobIdForUi(stopData && stopData.job_id);
                    if (resolvedStopJobId) {
                        suppressPendingJob(resolvedStopJobId);
                        if (manualStopContext && manualStopContext.seq === stopSeq) {
                            manualStopContext.jobId = resolvedStopJobId;
                        }
                    }
                }
                if (manualStopContext && manualStopContext.seq === stopSeq) {
                    const synced = await syncThreadAfterAbortedStream(stopThreadId, { retries: 2, retryDelayMs: 180, notifyOnFailure: true });
                    if (synced && manualStopContext.partialSnapshot) {
                        appendStoppedPartialBubbleSnapshot(manualStopContext.partialSnapshot, stopThreadId);
                    }
                }
            } finally {
                if (manualStopContext && manualStopContext.seq === stopSeq) {
                    manualStopContext = null;
                }
                setSendBtnToSendMode();
                updateFilePreview();
            }
        }
        async function purgeCaches() {
            if ('caches' in window) {
                const keys = await caches.keys();
                await Promise.all(keys.map(k => caches.delete(k)));
            }
            if (navigator.serviceWorker) {
                const regs = await navigator.serviceWorker.getRegistrations();
                await Promise.all(regs.map(r => r.unregister()));
            }
        }
        async function applyCacheMode(enable) {
            if (!('serviceWorker' in navigator)) return;
            if (enable) {
                try {
                    await navigator.serviceWorker.register(`/sw.js?v=${encodeURIComponent(appVersion)}`);
                } catch (e) {}
            } else {
                await purgeCaches();
            }
        }
        function checkAndNotifyVersion(latest) {
            if (!latest || !appVersion) return;
            if (latest === appVersion) return;
            const notified = localStorage.getItem("version_notified") || "";
            if (notified === latest) return;
            localStorage.setItem("app_version", latest);
            syncVersionUpdateCachePreferenceUi();
            showModal("version-update-modal");
        }
        async function checkVersion() {
            try {
                const res = await fetch("/api/version", { cache: "no-store" });
                if (!res.ok) return;
                const data = await res.json();
                const latest = data.version || "";
                const stored = localStorage.getItem("app_version") || "";
                if (latest && !stored) localStorage.setItem("app_version", latest);
                if (latest && stored && latest !== stored) {
                    await purgeCaches();
                    checkAndNotifyVersion(latest);
                }
            } catch (e) {}
        }
        function setConnectionBanner(mode, message = '') {
            const b = get('offline-banner');
            const icon = get('offline-banner-icon');
            const text = get('offline-banner-text');
            if (!b || !icon || !text) return;
            if (mode !== 'online' && connectionRecoveredHideTimer) {
                window.clearTimeout(connectionRecoveredHideTimer);
                connectionRecoveredHideTimer = null;
            }
            if (mode === 'hidden') {
                b.classList.remove('visible', 'offline', 'unstable', 'online');
                document.body.classList.remove('network-banner-visible');
                return;
            }
            b.classList.add('visible');
            document.body.classList.add('network-banner-visible');
            if (mode === 'offline') {
                b.classList.add('offline');
                b.classList.remove('unstable', 'online');
                icon.className = 'fas fa-wifi';
                text.textContent = message || 'サーバーとの通信が切断されています';
                return;
            }
            if (mode === 'online') {
                b.classList.add('online');
                b.classList.remove('offline', 'unstable');
                icon.className = 'fas fa-check-circle';
                text.textContent = message || 'サーバーとの通信が復帰しました';
                return;
            }
            b.classList.add('unstable');
            b.classList.remove('offline', 'online');
            icon.className = 'fas fa-exclamation-triangle';
            text.textContent = message || 'サーバーとの通信が不安定です';
        }
        function showConnectionRecoveredBanner(message = 'サーバーとの通信が復帰しました') {
            setConnectionBanner('online', message);
            connectionRecoveredHideTimer = window.setTimeout(() => {
                connectionRecoveredHideTimer = null;
                if (connectionStatus === 'online') {
                    setConnectionBanner('hidden');
                }
            }, CONNECTION_RECOVERED_BANNER_MS);
        }
        function getConnectionCheckIntervalMs() {
            if (connectionStatus === 'offline' || connectionStatus === 'unstable') return CONNECTION_CHECK_FAST_INTERVAL_MS;
            if (connectionConsecutiveFail > 0) return CONNECTION_CHECK_FAST_INTERVAL_MS;
            if (connectionUnstableUntil && Date.now() < connectionUnstableUntil) return CONNECTION_CHECK_FAST_INTERVAL_MS;
            return CONNECTION_CHECK_INTERVAL_MS;
        }
        function refreshConnectionMonitorTimer(force = false) {
            const nextIntervalMs = getConnectionCheckIntervalMs();
            if (!force && connectionCheckTimer && connectionCheckIntervalMs === nextIntervalMs) return;
            if (connectionCheckTimer) {
                window.clearInterval(connectionCheckTimer);
                connectionCheckTimer = null;
            }
            connectionCheckIntervalMs = nextIntervalMs;
            connectionCheckTimer = window.setInterval(probeServerConnection, nextIntervalMs);
        }
        async function probeServerConnection() {
            if (!navigator.onLine) {
                connectionConsecutiveFail = CONNECTION_FAIL_TO_OFFLINE;
                connectionUnstableUntil = 0;
                connectionStatus = 'offline';
                setConnectionBanner('offline');
                refreshConnectionMonitorTimer();
                return;
            }
            if (connectionCheckInFlight) return;
            connectionCheckInFlight = true;
            const startedAt = performance.now();
            const ctrl = new AbortController();
            const timeoutId = window.setTimeout(() => ctrl.abort(), CONNECTION_CHECK_TIMEOUT_MS);
            try {
                const heartbeatRes = await fetch(`/api/version?heartbeat=${Date.now()}`, {
                    method: 'GET',
                    cache: 'no-store',
                    credentials: 'include',
                    signal: ctrl.signal,
                    headers: { 'Accept': 'application/json', 'Cache-Control': 'no-cache' }
                });
                if (!heartbeatRes.ok) throw new Error(`heartbeat ${heartbeatRes.status}`);
                const hbData = await heartbeatRes.json();
                const hbVersion = hbData.version || "";
                if (hbVersion && hbVersion !== appVersion) {
                    const hbNotified = localStorage.getItem("version_notified") || "";
                    if (hbNotified !== hbVersion) {
                        localStorage.setItem("app_version", hbVersion);
                        await purgeCaches();
                        checkAndNotifyVersion(hbVersion);
                    }
                }
                const latencyMs = Math.round(performance.now() - startedAt);
                const now = Date.now();
                const hadFailure = connectionConsecutiveFail > 0;
                const wasDisconnected = connectionStatus === 'offline' || connectionStatus === 'unstable';
                connectionConsecutiveFail = 0;
                if (latencyMs >= CONNECTION_UNSTABLE_LATENCY_MS || hadFailure) {
                    connectionUnstableUntil = now + CONNECTION_UNSTABLE_HOLD_MS;
                    connectionStatus = 'unstable';
                    setConnectionBanner('unstable', `サーバーとの通信が不安定です（遅延 ${latencyMs}ms）`);
                } else if (now < connectionUnstableUntil) {
                    connectionStatus = 'unstable';
                    setConnectionBanner('unstable', 'サーバーとの通信が不安定です（回復確認中）');
                } else {
                    connectionUnstableUntil = 0;
                    connectionStatus = 'online';
                    if (wasDisconnected) {
                        showConnectionRecoveredBanner();
                    } else {
                        setConnectionBanner('hidden');
                    }
                }
                refreshConnectionMonitorTimer();
            } catch (e) {
                const now = Date.now();
                connectionConsecutiveFail += 1;
                connectionUnstableUntil = now + CONNECTION_UNSTABLE_HOLD_MS;
                if (!navigator.onLine || connectionConsecutiveFail >= CONNECTION_FAIL_TO_OFFLINE) {
                    connectionStatus = 'offline';
                    setConnectionBanner('offline');
                } else {
                    connectionStatus = 'unstable';
                    setConnectionBanner('unstable', 'サーバーとの通信が不安定です（再接続を試行中）');
                }
                refreshConnectionMonitorTimer();
            } finally {
                window.clearTimeout(timeoutId);
                connectionCheckInFlight = false;
            }
        }
        function startConnectionMonitor() {
            refreshConnectionMonitorTimer(true);
            probeServerConnection();
        }
        function stopConnectionMonitor() {
            if (!connectionCheckTimer) return;
            window.clearInterval(connectionCheckTimer);
            connectionCheckTimer = null;
            connectionCheckIntervalMs = 0;
            if (connectionRecoveredHideTimer) {
                window.clearTimeout(connectionRecoveredHideTimer);
                connectionRecoveredHideTimer = null;
            }
        }
        window.initTurnstileWidget = () => {
            if (!botConfig || !botConfig.turnstileSiteKey || !window.turnstile) return;
            if (turnstileWidgetId !== null) return;
            const container = document.getElementById('turnstile-container');
            if (!container) return;
            turnstileWidgetId = window.turnstile.render(container, {
                sitekey: botConfig.turnstileSiteKey,
                size: 'compact',
                callback: (token) => { turnstileToken = token; turnstilePending = false; container.classList.add('hidden'); },
                'expired-callback': () => { turnstileToken = null; turnstilePending = false; container.classList.add('hidden'); },
                'error-callback': () => { turnstileToken = null; turnstilePending = false; container.classList.add('hidden'); }
            });
        };
        async function getTurnstileToken() {
            if (!botConfig || !botConfig.turnstileSiteKey) return null;
            if (turnstileToken) return turnstileToken;
            if (!window.turnstile || turnstileWidgetId === null) return null;
            const container = document.getElementById('turnstile-container');
            if (container) container.classList.remove('hidden');
            turnstilePending = true;
            return await new Promise((resolve) => {
                const prevToken = turnstileToken;
                const timeout = setTimeout(() => resolve(null), 1500);
                try {
                    window.turnstile.execute(turnstileWidgetId);
                } catch (e) {
                    clearTimeout(timeout);
                    resolve(null);
                    return;
                }
                const interval = setInterval(() => {
                    if (turnstileToken && turnstileToken !== prevToken) {
                        clearTimeout(timeout);
                        clearInterval(interval);
                        resolve(turnstileToken);
                    }
                }, 50);
            });
        }
        function resetTurnstileToken() {
            turnstileToken = null;
            turnstilePending = false;
            if (window.turnstile && turnstileWidgetId !== null) {
                try { window.turnstile.reset(turnstileWidgetId); } catch (e) {}
            }
        }
        const botTelemetry = (() => {
            const state = {
                enabled: false,
                windowStart: performance.now(),
                lastSend: 0,
                clicks: 0,
                keys: 0,
                moves: 0,
                fastClicks: 0,
                fastKeys: 0,
                clickTimes: [],
                keyTimes: [],
                clickIntervals: [],
                lastClickTs: 0,
                lastKeyTs: 0,
                lastMove: null,
                speedMax: 0,
                speedSum: 0,
                speedSamples: 0,
                lastMoveSample: 0
            };
            const refreshEnabled = () => {
                state.enabled = !!(botConfig && botConfig.globalEnabled && botConfig.accountEnabled && !isAdminUser);
            };
            const resetWindow = () => {
                state.windowStart = performance.now();
                state.clicks = 0;
                state.keys = 0;
                state.moves = 0;
                state.fastClicks = 0;
                state.fastKeys = 0;
                state.clickTimes = [];
                state.keyTimes = [];
                state.clickIntervals = [];
                state.speedMax = 0;
                state.speedSum = 0;
                state.speedSamples = 0;
            };
            const recordClick = () => {
                const now = performance.now();
                state.clicks += 1;
                if (state.lastClickTs) {
                    const delta = now - state.lastClickTs;
                    state.clickIntervals.push(delta);
                    if (state.clickIntervals.length > 10) state.clickIntervals.shift();
                    if (delta < 120) state.fastClicks += 1;
                }
                state.lastClickTs = now;
                state.clickTimes.push(now);
                state.clickTimes = state.clickTimes.filter(t => now - t <= 2000);
                if (state.fastClicks >= 4) send(true);
            };
            const recordKey = () => {
                const now = performance.now();
                state.keys += 1;
                if (state.lastKeyTs) {
                    const delta = now - state.lastKeyTs;
                    if (delta < 50) state.fastKeys += 1;
                }
                state.lastKeyTs = now;
                state.keyTimes.push(now);
                state.keyTimes = state.keyTimes.filter(t => now - t <= 2000);
            };
            const recordMove = (e) => {
                const now = performance.now();
                if (now - state.lastMoveSample < 80) return;
                state.lastMoveSample = now;
                state.moves += 1;
                if (state.lastMove) {
                    const dx = e.clientX - state.lastMove.x;
                    const dy = e.clientY - state.lastMove.y;
                    const dt = now - state.lastMove.t;
                    if (dt > 0) {
                        const speed = Math.sqrt(dx * dx + dy * dy) / (dt / 1000);
                        state.speedMax = Math.max(state.speedMax, speed);
                        state.speedSum += speed;
                        state.speedSamples += 1;
                    }
                }
                state.lastMove = { x: e.clientX, y: e.clientY, t: now };
            };
            const computeStats = () => {
                const windowMs = Math.max(1, performance.now() - state.windowStart);
                const clickBurst = state.clickTimes.length;
                const keyBurst = state.keyTimes.length;
                const avgSpeed = state.speedSamples ? (state.speedSum / state.speedSamples) : 0;
                let avgClick = 0;
                let clickCv = 1.0;
                if (state.clickIntervals.length >= 3) {
                    const mean = state.clickIntervals.reduce((a, b) => a + b, 0) / state.clickIntervals.length;
                    const variance = state.clickIntervals.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / state.clickIntervals.length;
                    avgClick = mean;
                    clickCv = mean > 0 ? Math.sqrt(variance) / mean : 1.0;
                }
                return {
                    window_ms: Math.round(windowMs),
                    clicks: state.clicks,
                    keys: state.keys,
                    moves: state.moves,
                    fast_clicks: state.fastClicks,
                    fast_keys: state.fastKeys,
                    click_burst: clickBurst,
                    key_burst: keyBurst,
                    avg_click_ms: avgClick,
                    click_cv: clickCv,
                    event_rate: (state.clicks + state.keys + state.moves) / (windowMs / 1000),
                    pointer_speed_max: state.speedMax,
                    pointer_speed_avg: avgSpeed
                };
            };
            const isSuspicious = (payload) => {
                if (payload.fast_clicks >= 4) return true;
                if (payload.fast_keys >= 8) return true;
                if (payload.click_burst >= 8) return true;
                if (payload.key_burst >= 14) return true;
                if (payload.event_rate >= 20) return true;
                if (payload.avg_click_ms > 0 && payload.avg_click_ms < 160 && payload.click_cv < 0.08) return true;
                return false;
            };
            const send = async (force = false) => {
                if (!state.enabled) return;
                const now = performance.now();
                if (!force && now - state.lastSend < 3000) return;
                state.lastSend = now;
                const payload = computeStats();
                if ((payload.clicks + payload.keys + payload.moves) === 0) return;
                if (!force && !isSuspicious(payload)) return;
                payload.turnstile_token = await getTurnstileToken();
                if (botConfig && botConfig.turnstileSiteKey && !payload.turnstile_token) return;
                try {
                    const res = await apiFetch('/api/bot-telemetry', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload)
                    });
                    if (res.status === 403) {
                        let data = null;
                        try { data = await res.json(); } catch (e) {}
                        if (data && data.error === 'banned') {
                            showToast('ボット判定によりBANされました。', 'error', true);
                            setTimeout(() => { location.href = '/banned'; }, 800);
                            return;
                        }
                    }
                } catch (e) {}
                resetTurnstileToken();
                resetWindow();
            };
            const start = () => {
                refreshEnabled();
                if (!state.enabled) return;
                document.addEventListener('click', recordClick, true);
                document.addEventListener('pointerdown', recordClick, true);
                document.addEventListener('touchstart', recordClick, true);
                document.addEventListener('keydown', recordKey, true);
                document.addEventListener('wheel', () => { state.moves += 1; }, { passive: true });
                document.addEventListener('mousemove', recordMove, true);
                setInterval(() => send(false), 4000);
            };
            return { start, refreshEnabled, send };
        })();
        function openImageViewer(url) {
            const viewer = get('image-viewer');
            const img = get('image-viewer-img');
            if (!viewer || !img) return;
            img.src = url;
            viewer.classList.add('visible');
        }
        function closeImageViewer() {
            const viewer = get('image-viewer');
            const img = get('image-viewer-img');
            if (!viewer || !img) return;
            img.src = '';
            viewer.classList.remove('visible');
        }
        function openFileViewer(url, filename = '') {
            if (!url) return;
            const ext = (filename || url).split('.').pop().toLowerCase();
            const imageExt = ['png','jpg','jpeg','webp','gif'];
            const videoExt = ['mp4','mov','mkv','avi','m4v','webm'];
            const audioExt = ['mp3','wav','m4a','ogg','flac'];
            const docExt = ['pdf','txt','md','csv','log','json','docx'];
            if (imageExt.includes(ext)) {
                openImageViewer(url);
                return;
            }
            const viewer = get('file-viewer');
            const body = get('file-viewer-body');
            const title = get('file-viewer-title');
            if (!viewer || !body || !title) return;
            title.textContent = filename || 'File Preview';
            body.replaceChildren();
            if (videoExt.includes(ext)) {
                const video = document.createElement('video');
                video.src = String(url);
                video.controls = true;
                video.playsInline = true;
                video.preload = 'metadata';
                body.appendChild(video);
            } else if (audioExt.includes(ext)) {
                const audio = document.createElement('audio');
                audio.src = String(url);
                audio.controls = true;
                body.appendChild(audio);
            } else if (docExt.includes(ext)) {
                const frame = document.createElement('iframe');
                frame.src = String(url);
                frame.setAttribute('sandbox', '');
                frame.referrerPolicy = 'no-referrer';
                body.appendChild(frame);
            } else {
                const fallback = document.createElement('div');
                fallback.className = 'fallback';
                fallback.appendChild(document.createTextNode('この形式はプレビューできません。'));
                const actions = document.createElement('div');
                actions.className = 'mt-3 flex justify-center gap-2';
                const download = document.createElement('a');
                download.href = String(url);
                download.download = '';
                download.className = 'px-3 py-1 bg-gray-800 text-white rounded text-xs border border-gray-700';
                download.textContent = 'ダウンロード';
                const open = document.createElement('a');
                open.href = String(url);
                open.target = '_blank';
                open.rel = 'noopener noreferrer';
                open.className = download.className;
                open.textContent = '新しいタブで開く';
                actions.append(download, open);
                fallback.appendChild(actions);
                body.appendChild(fallback);
            }
            viewer.classList.add('visible');
        }
        function closeFileViewer() {
            const viewer = get('file-viewer');
            const body = get('file-viewer-body');
            if (!viewer || !body) return;
            body.innerHTML = '';
            viewer.classList.remove('visible');
        }
        function showToast(msg, type = "error", sticky = false) {
            const stack = get('toast-stack');
            if (!stack) return;
            while (stack.children.length >= 3) {
                stack.removeChild(stack.firstChild);
            }
            const el = document.createElement('div');
            el.className = `toast ${type}`;
            el.innerHTML = `<i class="fas ${type==='error' ? 'fa-triangle-exclamation' : 'fa-circle-info'}"></i><span class="flex-1">${escapeHtml(msg)}</span><button aria-label="close"><i class="fas fa-times"></i></button>`;
            el.querySelector('button').onclick = () => el.remove();
            stack.appendChild(el);
            if (!sticky) setTimeout(() => { if (el.parentNode) el.remove(); }, 7000);
            return el;
        }
        function showProgressToast(msg, type = "info") {
            const stack = get('toast-stack');
            if (!stack) return null;
            while (stack.children.length >= 3) {
                stack.removeChild(stack.firstChild);
            }
            const el = document.createElement('div');
            el.className = `toast ${type} flex-col !items-start min-w-[240px]`;
            el.innerHTML = `
                <div class="flex items-center gap-2 w-full">
                    <i class="fas ${type==='error' ? 'fa-triangle-exclamation' : 'fa-circle-info'}"></i>
                    <span class="flex-1 font-bold">${escapeHtml(msg)}</span>
                    <button aria-label="close" class="ml-auto opacity-50 hover:opacity-100"><i class="fas fa-times"></i></button>
                </div>
                <div class="w-full bg-white/10 h-1.5 rounded-full mt-2.5 overflow-hidden">
                    <div class="progress-bar h-full bg-blue-500 transition-all duration-300 shadow-[0_0_8px_rgba(59,130,246,0.5)]" style="width: 0%"></div>
                </div>
                <div class="w-full text-[10px] text-right mt-1.5 opacity-70 font-mono progress-text">0%</div>
            `;
            el.querySelector('button').onclick = () => el.remove();
            stack.appendChild(el);
            return {
                update: (pct) => {
                    const bar = el.querySelector('.progress-bar');
                    const text = el.querySelector('.progress-text');
                    if (bar) bar.style.width = `${Math.min(100, Math.max(0, pct))}%`;
                    if (text) text.innerText = `${Math.round(pct)}%`;
                },
                remove: () => { if (el.parentNode) el.remove(); }
            };
        }
        let activeSettingsTab = 'general';
        const TAB_LABELS = { general: '一般', security: 'セキュリティ', '2fa': '2要素認証', feedback: 'フィードバック' };
        const ALL_TABS = ['general', 'security', '2fa', 'feedback'];
        function getSectionHeading(el) {
            const h3 = el.querySelector('h3');
            if (h3) return h3.textContent.trim();
            const bold = el.querySelector('.font-bold');
            if (bold && !bold.querySelector('input') && !bold.querySelector('select')) return bold.textContent.trim();
            const firstLabel = el.querySelector('label');
            if (firstLabel) {
                const t = firstLabel.textContent.trim().replace(/[：:].*$/, '').trim();
                if (t) return t;
            }
            return '';
        }
        function getSectionSnippet(el, query) {
            const text = el.textContent;
            const lower = text.toLowerCase();
            const idx = lower.indexOf(query.toLowerCase());
            if (idx === -1) return '';
            const start = Math.max(0, idx - 25);
            const end = Math.min(text.length, idx + query.length + 35);
            let snippet = text.substring(start, end).replace(/\s+/g, ' ').trim();
            if (start > 0) snippet = '…' + snippet;
            if (end < text.length) snippet = snippet + '…';
            return snippet;
        }
        function removeSearchOverlays() {
            ALL_TABS.forEach(tabId => {
                const tab = get('tab-' + tabId);
                if (!tab) return;
                const ov = tab.querySelector('.settings-search-overlay');
                if (ov) ov.remove();
                Array.from(tab.children).forEach(child => {
                    if (child.classList.contains('settings-no-results')) return;
                    child.style.display = '';
                });
            });
        }
        function filterSettings() {
            const input = get('settings-search');
            if (!input) return;
            const q = input.value.trim().toLowerCase();
            const clearBtn = get('settings-search-clear');
            if (clearBtn) clearBtn.classList.toggle('hidden', !q);
            removeSearchOverlays();
            if (!q) {
                ALL_TABS.forEach(tabId => {
                    const btn = get('btn-tab-' + tabId);
                    if (btn) { const b = btn.querySelector('.settings-search-badge'); if (b) b.remove(); }
                    const tab = get('tab-' + tabId);
                    if (!tab) return;
                    tab.classList.toggle('hidden', tabId !== activeSettingsTab);
                });
                return;
            }
            let results = [];
            ALL_TABS.forEach(tabId => {
                const tab = get('tab-' + tabId);
                if (!tab) return;
                tab.classList.add('hidden');
                Array.from(tab.children).forEach(child => {
                    if (child.classList.contains('settings-no-results') || child.classList.contains('settings-search-overlay')) return;
                    if (child.textContent.toLowerCase().includes(q)) {
                        const title = getSectionHeading(child) || tabId;
                        const snippet = getSectionSnippet(child, q);
                        results.push({ tabId, title, snippet, element: child });
                    }
                });
            });
            let targetTabId = activeSettingsTab;
            if (!results.some(r => r.tabId === targetTabId)) {
                const fr = results.find(r => r.tabId);
                if (fr) targetTabId = fr.tabId;
            }
            const targetTab = get('tab-' + targetTabId);
            if (!targetTab) return;
            targetTab.classList.remove('hidden');
            Array.from(targetTab.children).forEach(child => {
                if (child.classList.contains('settings-no-results') || child.classList.contains('settings-search-overlay')) return;
                child.style.display = 'none';
            });
            const overlay = document.createElement('div');
            overlay.className = 'settings-search-overlay';
            if (results.length === 0) {
                const empty = document.createElement('div');
                empty.className = 'text-center py-12 text-gray-500 text-sm';
                empty.textContent = '「' + q.replace(/</g, '&lt;') + '」に一致する設定項目はありません。';
                overlay.appendChild(empty);
            } else {
                const hdr = document.createElement('div');
                hdr.className = 'text-[11px] text-gray-500 px-1 pb-2';
                hdr.textContent = results.length + '件の一致';
                overlay.appendChild(hdr);
                let prevTabId = null;
                results.forEach((r, i) => {
                    if (r.tabId !== prevTabId) {
                        if (prevTabId !== null) {
                            const sep = document.createElement('div');
                            sep.className = 'border-t border-gray-700/50 my-1.5';
                            overlay.appendChild(sep);
                        }
                        if (r.tabId !== targetTabId) {
                            const lbl = document.createElement('div');
                            lbl.className = 'text-[10px] text-gray-500 px-1 pb-1 font-bold';
                            lbl.textContent = '▼ ' + (TAB_LABELS[r.tabId] || r.tabId);
                            overlay.appendChild(lbl);
                        }
                        prevTabId = r.tabId;
                    }
                    const item = document.createElement('div');
                    item.className = 'settings-search-result-item flex items-start gap-2.5 px-3 py-2.5 rounded-lg cursor-pointer transition-all duration-150';
                    item.style.animation = 'fadeIn 0.28s cubic-bezier(0.22, 1, 0.36, 1) both';
                    item.style.animationDelay = (i * 30) + 'ms';
                    const badge = document.createElement('span');
                    badge.className = 'settings-result-tab-badge shrink-0 mt-0.5';
                    badge.textContent = TAB_LABELS[r.tabId] || r.tabId;
                    const inner = document.createElement('div');
                    inner.className = 'min-w-0 flex-1';
                    const heading = document.createElement('div');
                    heading.className = 'text-sm font-bold text-white truncate';
                    heading.textContent = r.title;
                    const snippet = document.createElement('div');
                    snippet.className = 'text-[11px] text-gray-400 truncate mt-0.5';
                    snippet.textContent = r.snippet;
                    inner.appendChild(heading);
                    inner.appendChild(snippet);
                    item.appendChild(badge);
                    item.appendChild(inner);
                    item.addEventListener('click', () => jumpToSetting(r.tabId, r.element));
                    overlay.appendChild(item);
                });
            }
            targetTab.insertBefore(overlay, targetTab.firstChild);
        }
        function jumpToSetting(tabId, element) {
            const ss = get('settings-search');
            if (ss) ss.value = '';
            removeSearchOverlays();
            filterSettings();
            if (tabId !== activeSettingsTab) switchTab(tabId);
            setTimeout(() => {
                element.scrollIntoView({ behavior: 'smooth', block: 'center' });
                element.classList.add('settings-jump-highlight');
                setTimeout(() => element.classList.remove('settings-jump-highlight'), 2000);
            }, 260);
        }
        function clickTab(t) {
            const ss = get('settings-search');
            if (ss) ss.value = '';
            switchTab(t);
        }
        function switchTab(t) { 
            if (t === activeSettingsTab) return;
            const prev = get('tab-' + activeSettingsTab);
            if (prev) {
                prev.classList.remove('tab-enter');
                prev.classList.add('tab-exit');
                setTimeout(() => {
                    prev.classList.add('hidden');
                    prev.classList.remove('tab-exit');
                }, 170);
            }
            ALL_TABS.forEach(x => {
                const btn = get('btn-tab-'+x);
                if(x === t) {
                    const panel = get('tab-'+x);
                    panel.classList.remove('hidden');
                    panel.classList.remove('tab-exit');
                    panel.classList.remove('tab-enter');
                    void panel.offsetWidth;
                    panel.classList.add('tab-enter');
                    btn.classList.add('text-blue-400','border-blue-400','font-bold');
                    btn.classList.remove('text-gray-400','hover:text-white','border-transparent');
                } else {
                    btn.classList.remove('text-blue-400','border-blue-400','font-bold');
                    btn.classList.add('text-gray-400','hover:text-white','border-transparent');
                }
            });
            activeSettingsTab = t;
            filterSettings();
        }
        get('chat-container').addEventListener('scroll', function() { userAutoScroll = (this.scrollHeight - this.scrollTop - this.clientHeight) < 50; }, { passive: true });
        function scrollToBottom() { if(userAutoScroll) { const c = get('chat-container'); c.scrollTop = c.scrollHeight; } }
        
        // Image Viewer Logic
        let viewerImages = [];
        let viewerIndex = 0;

        function openImageViewer(startUrl, groupSelector = '.chat-image') {
            const allImgs = Array.from(document.querySelectorAll(groupSelector));
            // Filter out duplicate sources if needed, but keep DOM order
            viewerImages = allImgs.map(img => ({
                url: img.dataset.viewerSrc || img.currentSrc || img.src,
                filename: img.dataset.viewerFilename || img.title || (img.dataset.viewerSrc || img.currentSrc || img.src).split('/').pop(),
                element: img
            }));
            
            // Find index of startUrl
            viewerIndex = viewerImages.findIndex(item => item.url === startUrl);
            if (viewerIndex === -1) {
                // Fallback if not found in DOM list (single view)
                viewerImages = [{ url: startUrl, filename: startUrl.split('/').pop() }];
                viewerIndex = 0;
            }
            
            updateViewerState();
            get('image-viewer').classList.add('visible');
            document.addEventListener('keydown', handleViewerKeydown);
        }

        function closeImageViewer() {
            get('image-viewer').classList.remove('visible');
            document.removeEventListener('keydown', handleViewerKeydown);
            viewerImages = [];
        }

        function updateViewerState() {
            if (!viewerImages.length) return;
            const item = viewerImages[viewerIndex];
            const img = get('image-viewer-img');
            const meta = get('image-viewer-meta');
            const prevBtn = document.querySelector('.viewer-nav.prev');
            const nextBtn = document.querySelector('.viewer-nav.next');

            // Preload next image
            if (viewerIndex < viewerImages.length - 1) {
                const preload = new Image();
                preload.src = viewerImages[viewerIndex + 1].url;
            }

            img.style.opacity = '0.5';
            img.style.transform = 'scale(0.95)';
            
            setTimeout(() => {
                img.src = item.url;
                img.onload = () => {
                    img.style.opacity = '1';
                    img.style.transform = 'scale(1)';
                };
                meta.innerText = `${viewerIndex + 1} / ${viewerImages.length} • ${item.filename}`;
            }, 150);

            prevBtn.style.display = viewerImages.length > 1 ? 'flex' : 'none';
            nextBtn.style.display = viewerImages.length > 1 ? 'flex' : 'none';
            prevBtn.style.opacity = viewerIndex > 0 ? '1' : '0.3';
            nextBtn.style.opacity = viewerIndex < viewerImages.length - 1 ? '1' : '0.3';
            prevBtn.style.pointerEvents = viewerIndex > 0 ? 'auto' : 'none';
            nextBtn.style.pointerEvents = viewerIndex < viewerImages.length - 1 ? 'auto' : 'none';
        }

        function navImage(dir) {
            const newIndex = viewerIndex + dir;
            if (newIndex >= 0 && newIndex < viewerImages.length) {
                viewerIndex = newIndex;
                updateViewerState();
            }
        }

        function handleViewerKeydown(e) {
            if (e.key === 'ArrowLeft') navImage(-1);
            if (e.key === 'ArrowRight') navImage(1);
            if (e.key === 'Escape') closeImageViewer();
        }

        function downloadCurrentImage() {
            if (!viewerImages.length) return;
            const item = viewerImages[viewerIndex];
            const a = document.createElement('a');
            a.href = item.url;
            a.download = item.filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
        }

        function copyCurrentImageUrl() {
            if (!viewerImages.length) return;
            const url = viewerImages[viewerIndex].url;
            // Try to resolve full URL if relative
            const fullUrl = new URL(url, window.location.origin).href;
            copyToClipboard(fullUrl, () => showToast("画像URLをコピーしました", "success"), () => showToast("コピーに失敗しました"));
        }

        function reuseCurrentImage() {
            if (!viewerImages.length) return;
            const item = viewerImages[viewerIndex];
            let path = item.url;
            try {
                // Extract relative path from URL: /files/user/filename -> user/filename
                const urlObj = new URL(path, window.location.origin);
                if (urlObj.pathname.startsWith('/files/')) {
                    path = decodeURIComponent(urlObj.pathname.replace('/files/', ''));
                }
            } catch(e) {}
            
            if (path) {
                if (!currentImageUrls.includes(path)) {
                    currentImageUrls.push(path);
                    setAttachmentNameForPath(path, item.filename || '');
                    updateFilePreview();
                    showToast("画像を添付ファイルに追加しました", "success");
                    closeImageViewer();
                } else {
                    showToast("この画像は既に添付されています", "info");
                }
            }
        }

        // Robust Copy Function with Fallback
        async function copyToClipboard(text, successCallback, errorCallback) {
            try {
                if (navigator.clipboard && navigator.clipboard.writeText) {
                    await navigator.clipboard.writeText(text);
                    if(successCallback) successCallback();
                } else {
                    throw new Error("Clipboard API unavailable");
                }
            } catch (err) {
                try {
                    const textArea = document.createElement("textarea");
                    textArea.value = text;
                    textArea.style.position = "fixed";
                    textArea.style.left = "-9999px";
                    document.body.appendChild(textArea);
                    textArea.focus();
                    textArea.select();
                    const successful = document.execCommand('copy');
                    document.body.removeChild(textArea);
                    if (successful) {
                        if(successCallback) successCallback();
                    } else {
                        if(errorCallback) errorCallback(err);
                    }
                } catch (e) {
                    if(errorCallback) errorCallback(e);
                }
            }
        }

        // Quote Reply Logic
        function handleQuotePopover() {
            const sel = window.getSelection();
            const btn = get('quote-popover');
            if (!btn) return;
            if (!sel || sel.rangeCount === 0) {
                btn.style.display = 'none';
                btn.classList.remove('show');
                return;
            }
            const text = sel.toString().trim();
            if (text.length > 0 && get('chat-container').contains(sel.anchorNode)) {
                const range = sel.getRangeAt(0);
                const rect = range.getBoundingClientRect();
                const wasHidden = btn.style.display === 'none' || !btn.style.display || getComputedStyle(btn).display === 'none';
                btn.style.display = 'block';
                btn.style.top = (rect.top - 40) + 'px';
                btn.style.left = rect.left + 'px';
                if (wasHidden) {
                    btn.classList.remove('show');
                    void btn.offsetWidth;
                    btn.classList.add('show');
                }
            } else {
                btn.style.display = 'none';
                btn.classList.remove('show');
            }
        }
        document.addEventListener('mouseup', handleQuotePopover);
        document.addEventListener('touchend', () => setTimeout(handleQuotePopover, 0), { passive: true });
        document.addEventListener('selectionchange', () => {
            // Avoid excessive updates during selection drag
            if (window.getSelection && window.getSelection().type === 'Range') {
                handleQuotePopover();
            }
        });
        
        get('quote-popover').onclick = () => {
            currentQuote = window.getSelection().toString().trim();
            if(currentQuote) {
                get('quote-text-display').innerText = currentQuote;
                get('quote-bar').classList.add('visible');
                get('prompt-input').focus();
            }
            schedulePromptTokenEstimate();
            const qBtn = get('quote-popover');
            if (qBtn) {
                qBtn.style.display = 'none';
                qBtn.classList.remove('show');
            }
        };

        window.clearQuote = () => {
            currentQuote = "";
            get('quote-bar').classList.remove('visible');
            get('quote-text-display').innerText = "";
            schedulePromptTokenEstimate();
        };

        const MODELS = [
            {
                category: "Gemini 3.5",
                icon: "fas fa-star text-yellow-400",
                description: "Google's latest multimodal models",
                items: [
                    { id: "gemini-3.5-flash", name: "Gemini 3.5 Flash", desc: "Most intelligent Gemini 3.5 model built for speed.", price: "In $1.50/1M, Out $9.00/1M" },
                ]
            },
            {
                category: "Gemini 3.0",
                icon: "fas fa-star text-yellow-400",
                description: "Previous Gemini 3.x generation models",
                items: [
                    { id: "gemini-3.1-pro-preview", name: "Gemini 3.1 Pro", desc: "Next-gen native multimodal model.", price: "In $2.00/1M, Out $12.00/1M (≤200k)" },
                    { id: "gemini-3.1-flash-lite-preview", name: "Gemini 3.1 Flash-Lite", desc: "Fastest and most cost-efficient Gemini 3.1 model.", price: "In $0.25/1M, Out $1.50/1M" },
                    { id: "gemini-3-flash-preview", name: "Gemini 3.0 Flash", desc: "Fastest and most cost-efficient.", price: "In $0.50/1M, Out $3.00/1M" },
                    { id: "gemini-3-pro-preview", name: "Gemini 3.0 Pro", desc: "Best performing model for complex tasks.", price: "In $2.00/1M, Out $12.00/1M (≤200k)" }
                ]
            },
            {
                category: "Gemini 2.5",
                icon: "fas fa-history text-gray-400",
                description: "Gemini 2.5 generation models",
                items: [
                    { id: "gemini-2.5-flash-lite", name: "Gemini 2.5 Flash-Lite", desc: "Fastest and most cost-efficient Gemini 2.5 model.", price: "In $0.10/1M, Out $0.40/1M" },
                    { id: "gemini-2.5-flash", name: "Gemini 2.5 Flash", desc: "Balanced performance.", price: "In $0.30/1M, Out $2.50/1M" }
                ]
            },
            {
                category: "Gemini Image (Banana)",
                icon: "fas fa-image text-pink-400",
                description: "Gemini image generation models",
                items: [
                    { id: "gemini-2.5-flash-image", name: "Nano Banana", desc: "Fast image generation.", price: "In $0.30/1M, Out $0.039/image" },
                    { id: "gemini-3.1-flash-image-preview", name: "Nano Banana 2", desc: "Fast image generation with Gemini 3.1 Flash Image.", price: "In $0.50/1M, Out $0.067/1K image ($60/1M img tokens)" },
                    { id: "gemini-3.1-flash-lite-image", name: "Nano Banana 2 Lite", desc: "Fastest and cheapest Gemini Image model from Nano Banana family.", price: "In $0.25/1M, Out $0.0336/1K image ($30/1M img)" },
                    { id: "gemini-3-pro-image-preview", name: "Nano Banana Pro", desc: "High quality image generation.", price: "In $2.00/1M, Out $0.134 (1K/2K) or $0.24 (4K)" }
                ]
            },
            {
                category: "OpenAI Image Gen",
                icon: "fas fa-paint-brush text-purple-400",
                description: "GPT Image models",
                items: [
                    { id: "gpt-image-2", name: "GPT Image 2", desc: "State-of-the-art image generation and editing.", price: "Text In $5/1M; Image In $8/1M; Image Out $30/1M" },
                    { id: "gpt-image-1.5", name: "GPT Image 1.5", desc: "Previous-generation flagship image model.", price: "Text In $5/1M, Text Out $10/1M; Image Out $32/1M" },
                    { id: "gpt-image-1", name: "GPT Image 1", desc: "Standard quality.", price: "Text In $5/1M; Image Out $40/1M" },
                    { id: "gpt-image-1-mini", name: "GPT Image 1 Mini", desc: "Faster, lower resolution.", price: "Text In $2/1M; Image In $2.50/1M; Image Out $8/1M" }
                ]
            },
            {
                category: "OpenAI GPT",
                icon: "fas fa-brain text-green-400",
                description: "OpenAI's flagship models",
                items: [
                    { id: "gpt-4o", name: "GPT-4o", desc: "Multimodal flagship model.", price: "In $2.50/1M, Out $10.00/1M" },
                    { id: "gpt-4o-mini", name: "GPT-4o mini", desc: "Fast, low-cost model.", price: "In $0.15/1M, Out $0.60/1M" },
                    { id: "gpt-5.5", name: "GPT-5.5", desc: "Experimental OpenAI model ID for accounts with access.", price: "In $5.00/1M, Out $30.00/1M" },
                    { id: "gpt-5.5-mini", name: "GPT-5.5 mini", desc: "Smaller and more cost-efficient GPT-5.5 tier.", price: "Pricing not publicly listed" },
                    { id: "gpt-5.5-nano", name: "GPT-5.5 nano", desc: "Smallest and fastest GPT-5.5 tier.", price: "Pricing not publicly listed" },
                    { id: "gpt-5.5-pro", name: "GPT-5.5 Pro", desc: "Higher-capacity GPT-5.5 tier for accounts with access.", price: "In $30.00/1M, Out $180.00/1M" },
                    { id: "gpt-5.4", name: "GPT-5.4", desc: "Experimental OpenAI model ID for accounts with access.", price: "In $2.50/1M, Out $15.00/1M" },
                    { id: "gpt-5.4-mini", name: "GPT-5.4 mini", desc: "Smaller and more cost-efficient GPT-5.4 tier.", price: "In $0.75/1M, Out $4.50/1M" },
                    { id: "gpt-5.4-nano", name: "GPT-5.4 nano", desc: "Smallest and fastest GPT-5.4 tier.", price: "In $0.20/1M, Out $1.25/1M" },
                    { id: "gpt-5.4-pro", name: "GPT-5.4 Pro", desc: "Higher-capacity GPT-5.4 tier for accounts with access.", price: "In $30.00/1M, Out $180.00/1M" },
                    { id: "gpt-5.2", name: "GPT-5.2 (Responses API)", desc: "Most capable reasoning model.", price: "In $1.75/1M, Out $14.00/1M" },
                    { id: "gpt-5-search-api", name: "GPT-5 Search (API)", desc: "Search-optimized model (Chat Completions).", price: "Model rates + Web search $10/1k calls" },
                    { id: "gpt-5.1", name: "GPT-5.1", desc: "High intelligence.", price: "In $1.25/1M, Out $10.00/1M" },
                    { id: "gpt-5-mini", name: "GPT-5 mini", desc: "Small and efficient.", price: "In $0.25/1M, Out $2.00/1M" }
                ]
            },
            {
                category: "DeepSeek V4",
                icon: "fas fa-bolt text-cyan-400",
                description: "DeepSeek's OpenAI-compatible text models",
                items: [
                    { id: "deepseek-v4-flash", name: "DeepSeek V4 Flash", desc: "Fast DeepSeek V4 model with thinking / non-thinking modes.", price: "In $0.14/1M (miss), Out $0.28/1M" },
                    { id: "deepseek-v4-pro", name: "DeepSeek V4 Pro", desc: "Higher-capacity DeepSeek V4 model with thinking / non-thinking modes.", price: "In $0.435/1M (miss), Out $0.87/1M" }
                ]
            },
            {
                category: "Anthropic Claude",
                icon: "fas fa-brain text-orange-400",
                description: "Anthropic's latest deep reasoning models",
                items: [
                    { id: "claude-opus-4-6", name: "Claude Opus 4.6", desc: "Most capable model for deep reasoning and complex tasks.", price: "In $5.00/1M, Out $25.00/1M" },
                    { id: "claude-sonnet-4-6", name: "Claude Sonnet 4.6", desc: "Excellent balance of speed and intelligence with adaptive thinking.", price: "In $3.00/1M, Out $15.00/1M" }
                ]
            },
            {
                category: "Audio (TTS)",
                icon: "fas fa-microphone text-red-400",
                description: "Text-to-Speech models",
                items: [
                    { id: "gemini-3.1-flash-tts-preview", name: "Gemini 3.1 Flash TTS", desc: "Google TTS (Preview).", price: "Text In $1.00/1M, Audio Out $20.00/1M" },
                    { id: "gpt-4o-mini-tts", name: "GPT-4o Mini TTS", desc: "OpenAI TTS.", price: "Text In $0.60/1M, Audio Out $12.00/1M" },
                    { id: "gemini-2.5-flash-preview-tts", name: "Gemini 2.5 Flash TTS", desc: "Google TTS (Preview).", price: "Text In $0.50/1M, Audio Out $10.00/1M" },
                    { id: "gemini-2.5-pro-preview-tts", name: "Gemini 2.5 Pro TTS", desc: "Google TTS Pro (Preview).", price: "Text In $1.00/1M, Audio Out $20.00/1M" },
                    { id: "google-tts-studio", name: "Google TTS (Studio)", desc: "High fidelity studio voices.", price: "$160 / 1M chars" },
                    { id: "google-tts-neural", name: "Google TTS (Neural2)", desc: "Standard neural voices.", price: "$16 / 1M chars" },
                    { id: "grok-tts", name: "Grok TTS", desc: "xAI Text-to-Speech with expressive voices.", price: "$15.00 / 1M chars" }
                ]
            },
            {
                category: "Realtime Audio (STS)",
                icon: "fas fa-headset text-cyan-400",
                description: "Realtime voice models (audio in / audio out)",
                items: [
                    { id: "gpt-realtime-2", name: "OpenAI Realtime 2", desc: "Most capable speech-to-speech reasoning model.", price: "Audio In $32/1M, Audio Out $64/1M" },
                    { id: "gpt-realtime-translate", name: "OpenAI Realtime Translate", desc: "Streaming speech-to-speech translation.", price: "$0.034 / minute" },
                    { id: "gpt-realtime-whisper", name: "OpenAI Realtime Whisper", desc: "Streaming speech-to-text (transcription).", price: "$0.017 / minute" },
                    { id: "gpt-realtime-1.5", name: "OpenAI Realtime 1.5", desc: "Latest OpenAI speech-to-speech flagship model.", price: "Audio In $32/1M, Audio Out $64/1M" },
                    { id: "gpt-realtime", name: "OpenAI Realtime", desc: "OpenAI realtime speech-to-speech model.", price: "Audio In $32/1M, Audio Out $64/1M" },
                    { id: "gpt-realtime-mini", name: "OpenAI Realtime Mini", desc: "Lower-latency, smaller realtime model.", price: "Audio In $10/1M, Audio Out $20/1M" },
                    { id: "gemini-2.5-flash-native-audio-preview-12-2025", name: "Gemini 2.5 Flash Native Audio (Live)", desc: "Google Live native audio model.", price: "Audio In $3.00/1M, Audio Out $12.00/1M" },
                    { id: "gemini-3.1-flash-live-preview", name: "Gemini 3.1 Flash Live", desc: "Google Live native audio model.", price: "Audio In $3.00/1M (~$0.005/min), Out $12.00/1M" },
                    { id: "grok-voice-latest", name: "Grok Voice Latest", desc: "Recommended alias. Always points to the newest flagship voice model.", price: "$0.05 / min ($3.00 / hr)" },
                    { id: "grok-voice-think-fast-1.0", name: "Grok Voice Think Fast 1.0", desc: "Current flagship voice model with advanced reasoning.", price: "$0.05 / min ($3.00 / hr)" },
                    { id: "grok-voice-fast-1.0", name: "Grok Voice Fast 1.0", desc: "Legacy xAI realtime voice model (deprecated).", price: "$0.05 / min ($3.00 / hr)" },
                    { id: "grok-voice-agent", name: "Grok Voice Agent", desc: "xAI realtime voice agent API.", price: "$0.05 / min (Realtime)", deprecated: true }
                ]
            },
            {
                category: "Grok Imagine",
                icon: "fas fa-magic text-blue-400",
                description: "Grok generation models",
                items: [
                    { id: "grok-imagine-image-quality", name: "Grok Imagine Image Quality", desc: "Next-gen Grok image generation with 1K/2K support.", price: "$0.05 / image" },
                    { id: "grok-imagine-image", name: "Grok Imagine Image", desc: "Latest Grok image generation.", price: "$0.02 / image" },
                    { id: "grok-imagine-image-pro", name: "Grok Imagine Image Pro", desc: "High quality Grok image generation.", price: "$0.07 / image" },
                    { id: "grok-imagine-video", name: "Grok Imagine Video", desc: "Latest Grok video generation.", price: "$0.05 / second" }
                ]
            },
            {
                category: "xAI Grok",
                icon: "fas fa-rocket text-white",
                description: "Models by xAI",
                items: [
                    { id: "grok-4.3", name: "Grok 4.3", desc: "Most intelligent and fastest flagship model.", price: "In $1.25/1M, Out $2.50/1M" },
                    { id: "grok-build-0.1", name: "Grok Build 0.1 (Coding)", desc: "Fast agentic coding model with vision and reasoning support.", price: "In $1.00/1M, Out $2.00/1M" },
                    { id: "grok-4.20-reasoning", name: "Grok 4.20 (Reasoning)", desc: "Flagship reasoning model.", price: "In $1.25/1M, Out $2.50/1M" },
                    { id: "grok-4.20-non-reasoning", name: "Grok 4.20 (Non-Reasoning)", desc: "Flagship standard model.", price: "In $1.25/1M, Out $2.50/1M" },
                    { id: "grok-4.20-multi-agent", name: "Grok 4.20 Multi-Agent", desc: "Agentic flagship model.", price: "In $1.25/1M, Out $2.50/1M" },
                    { id: "grok-4-1-fast-reasoning", name: "Grok 4.1 Fast (Reasoning)", desc: "Fast with reasoning capabilities.", price: "In $0.20/1M, Out $0.50/1M", deprecated: true },
                    { id: "grok-4-1-fast-non-reasoning", name: "Grok 4.1 Fast (Non-Reasoning)", desc: "Fast standard model.", price: "In $0.20/1M, Out $0.50/1M", deprecated: true },
                    { id: "grok-4-fast-reasoning", name: "Grok 4 Fast (Reasoning)", desc: "Previous gen reasoning.", price: "In $0.20/1M, Out $0.50/1M", deprecated: true },
                    { id: "grok-4-fast-non-reasoning", name: "Grok 4 Fast (Non-Reasoning)", desc: "Previous gen standard.", price: "In $0.20/1M, Out $0.50/1M", deprecated: true }
                ]
            }
        ];

        const normalizeModelApiKeyMap = (raw) => {
            if (!raw || typeof raw !== 'object') return {};
            const out = {};
            Object.entries(raw).forEach(([modelId, apiKey]) => {
                const mk = String(modelId || '').trim();
                const kv = String(apiKey || '').trim();
                if (!mk || !kv) return;
                out[mk] = kv;
            });
            return out;
        };
        const MODEL_NAME_BY_ID = (() => {
            const map = new Map();
            MODELS.forEach((group) => {
                (group.items || []).forEach((item) => {
                    const id = String(item.id || '').trim();
                    if (!id || map.has(id)) return;
                    map.set(id, String(item.name || id));
                });
            });
            return map;
        })();
        const getModelNameById = (modelId) => {
            const mk = String(modelId || '').trim();
            if (!mk) return '';
            return MODEL_NAME_BY_ID.get(mk) || mk;
        };
        const maskApiKeyPreview = (key) => {
            const txt = String(key || '');
            if (!txt) return '';
            if (txt.length <= 8) return '********';
            return `${txt.slice(0, 4)}...${txt.slice(-4)}`;
        };
        const setModelApiKeyPanelOpen = (open) => {
            const panel = get('model-api-keys-panel');
            const btn = get('toggle-model-api-keys-btn');
            if (!panel || !btn) return;
            const show = !!open;
            panel.classList.toggle('hidden', !show);
            btn.innerText = show ? 'モデル別APIキー設定を閉じる' : 'モデル別のAPIキーを設定する';
        };
        const syncModelApiKeyModelOptions = () => {
            const select = get('model-api-key-model');
            if (!select) return;
            const prev = select.value || '';
            select.innerHTML = '';
            const first = document.createElement('option');
            first.value = '';
            first.textContent = 'モデルを選択';
            select.appendChild(first);
            MODELS.forEach((group) => {
                const items = Array.isArray(group.items) ? group.items.filter(m => !m.deprecated) : [];
                if (!items.length) return;
                const optgroup = document.createElement('optgroup');
                optgroup.label = String(group.category || 'Models');
                items.forEach((item) => {
                    const id = String(item.id || '').trim();
                    if (!id) return;
                    const op = document.createElement('option');
                    op.value = id;
                    op.textContent = `${String(item.name || id)} (${id})`;
                    optgroup.appendChild(op);
                });
                if (optgroup.children.length > 0) select.appendChild(optgroup);
            });
            if (prev) {
                const hasPrev = Array.from(select.options).some((op) => op.value === prev);
                if (hasPrev) select.value = prev;
            }
        };
        const renderModelApiKeyList = () => {
            const list = get('model-api-key-list');
            if (!list) return;
            modelApiKeyMap = normalizeModelApiKeyMap(modelApiKeyMap);
            const entries = Object.entries(modelApiKeyMap).sort((a, b) => a[0].localeCompare(b[0]));
            list.innerHTML = '';
            if (!entries.length) {
                const empty = document.createElement('div');
                empty.className = 'text-[11px] text-gray-500';
                empty.textContent = 'モデル別キーは未設定です。';
                list.appendChild(empty);
                return;
            }
            entries.forEach(([modelId, key]) => {
                const row = document.createElement('div');
                row.className = 'flex items-center justify-between gap-3 rounded border border-gray-700 bg-gray-900/70 px-3 py-2';
                const left = document.createElement('div');
                left.className = 'min-w-0';
                const title = document.createElement('div');
                title.className = 'text-[11px] text-gray-200 truncate';
                title.textContent = `${getModelNameById(modelId)} (${modelId})`;
                const keyView = document.createElement('div');
                keyView.className = 'text-[10px] text-cyan-300 font-mono';
                keyView.textContent = maskApiKeyPreview(key);
                left.appendChild(title);
                left.appendChild(keyView);
                const delBtn = document.createElement('button');
                delBtn.type = 'button';
                delBtn.className = 'text-[10px] bg-red-700/80 hover:bg-red-600 text-white px-2 py-1 rounded font-bold btn-hover shrink-0';
                delBtn.textContent = '削除';
                delBtn.onclick = () => {
                    delete modelApiKeyMap[modelId];
                    renderModelApiKeyList();
                    showToast(`モデル別APIキーを削除: ${modelId}`, 'success');
                };
                row.appendChild(left);
                row.appendChild(delBtn);
                list.appendChild(row);
            });
        };
        const bindModelApiKeySettingsControls = () => {
            const toggleBtn = get('toggle-model-api-keys-btn');
            if (toggleBtn && !toggleBtn.dataset.bound) {
                toggleBtn.dataset.bound = '1';
                toggleBtn.addEventListener('click', () => {
                    const panel = get('model-api-keys-panel');
                    setModelApiKeyPanelOpen(panel ? panel.classList.contains('hidden') : true);
                });
            }
            const addBtn = get('model-api-key-apply-btn');
            if (addBtn && !addBtn.dataset.bound) {
                addBtn.dataset.bound = '1';
                addBtn.addEventListener('click', () => {
                    const modelSel = get('model-api-key-model');
                    const keyInput = get('model-api-key-input');
                    const modelId = modelSel ? String(modelSel.value || '').trim() : '';
                    const keyVal = keyInput ? String(keyInput.value || '').trim() : '';
                    if (!modelId) {
                        showToast('モデルを選択してください', 'error', true);
                        return;
                    }
                    if (!keyVal) {
                        showToast('APIキーを入力してください', 'error', true);
                        return;
                    }
                    modelApiKeyMap = normalizeModelApiKeyMap(modelApiKeyMap);
                    modelApiKeyMap[modelId] = keyVal;
                    if (keyInput) keyInput.value = '';
                    renderModelApiKeyList();
                    showToast(`モデル別APIキーを設定: ${modelId}`, 'success');
                });
            }
            const keyInput = get('model-api-key-input');
            if (keyInput && !keyInput.dataset.bound) {
                keyInput.dataset.bound = '1';
                keyInput.addEventListener('keydown', (e) => {
                    if (e.key === 'Enter') {
                        e.preventDefault();
                        const addBtnEl = get('model-api-key-apply-btn');
                        if (addBtnEl) addBtnEl.click();
                    }
                });
            }
            syncModelApiKeyModelOptions();
            renderModelApiKeyList();
            setModelApiKeyPanelOpen(false);
        };

        let activeModelTag = 'all';
        const MODEL_TAGS = ['all','openai','gemini','deepseek','xai','image','audio','reasoning','fast'];

        // Slash command system (extensible command palette triggered by / in prompt bar)
        const SLASH_COMMANDS = [
            {
                id: 'settings',
                label: '/settings',
                description: 'AIで自然言語を使って設定を変更（現在選択中のモデルを使用）',
                icon: 'fa-cog',
                example: 'デフォルトモデルを gemini-2.5-flash に変更して thinking をオンに'
            }
            // 将来コマンドをここに追加予定
        ];
        let slashSuggestionsVisible = false;
        let slashSelectedIndex = 0;
        let pendingSlashCommand = null; // 'settings' など。コマンド選択後に残る引数テキストで発動

        // Gem suggestion system (triggered by @ in prompt bar)
        let gemSuggestionsVisible = false;
        let gemSelectedIndex = 0;

        const STS_MODELS = new Set([
            'gpt-realtime-2',
            'gpt-realtime-translate',
            'gpt-realtime-whisper',
            'gpt-realtime-1.5',
            'gpt-realtime',
            'gpt-realtime-mini',
            'gemini-2.5-flash-native-audio-preview-12-2025',
            'gemini-3.1-flash-live-preview',
            'grok-voice-latest',
            'grok-voice-think-fast-1.0',
            'grok-voice-fast-1.0',
            'grok-voice-agent'
        ]);
        const FILE_BASE_URL = CHAT_CONFIG.urls.serveFileBase;
        const FILE_THUMB_BASE_URL = CHAT_CONFIG.urls.serveFileThumbBase;
        const RICH_PASTE_PDF_SERVER_ROUTE = CHAT_CONFIG.urls.richPastePdfServer;
        const IMAGE_EXTS = ['png','jpg','jpeg','webp','gif','bmp','avif','heic','heif'];
        const AUDIO_EXTS = ['mp3','wav','aac','ogg','flac','aiff','aif','m4a','opus','oga','weba','webm'];
        const VIDEO_EXTS = ['mp4','mov','avi','mkv','m4v','webm','mpg','mpeg','wmv','3gp','3gpp','flv'];
        const getFileExt = (name) => {
            const text = typeof name === 'string' ? name : (name == null ? '' : String(name));
            if (!text) return '';
            const idx = text.lastIndexOf('.');
            if (idx === -1) return '';
            return text.slice(idx + 1).toLowerCase();
        };
        const normalizeAttachmentPath = (value) => {
            if (!value) return '';
            let v = '';
            if (typeof value === 'string') {
                v = value;
            } else if (typeof value === 'object') {
                v = String(value.path || value.url || value.name || value.filename || value.filepath || '');
            }
            if (!v) return '';
            
            try {
                if (v.includes('://')) {
                    v = new URL(v, window.location.origin).pathname || '';
                }
            } catch (e) {}
            if (v.includes('?')) v = v.split('?', 1)[0];
            if (v.includes('#')) v = v.split('#', 1)[0];
            v = v.replace(/^\/+/, '');
            if (v.startsWith('files/')) v = v.slice(6);
            try { v = decodeURIComponent(v); } catch (e) {}
            return v;
        };
        const isGeminiImageModelKey = (model) => {
            const m = (model || '').toLowerCase();
            return m.includes('gemini') && (m.includes('image') || m.includes('nano'));
        };
        const isClaudeModelKey = (model) => {
            const m = (model || '').toLowerCase();
            return m.includes('claude');
        };
        /** API provider for a model (prompt-cache lock key). */
        const getModelApiProvider = (modelId) => {
            const m = String(modelId || '').toLowerCase().trim();
            if (!m) return null;
            if (m.includes('claude')) return 'anthropic';
            if (m.includes('deepseek')) return 'deepseek';
            if (m.includes('grok') && !m.includes('gpt')) return 'xai';
            if (m.includes('google-tts')) return 'google';
            if (m.includes('gemini')) return 'gemini';
            return 'openai';
        };
        const PROVIDER_LABELS = {
            openai: 'OpenAI',
            gemini: 'Gemini',
            anthropic: 'Anthropic (Claude)',
            xai: 'xAI (Grok)',
            deepseek: 'DeepSeek',
            google: 'Google Cloud'
        };
        const isPromptCacheEnabled = () => {
            const el = get('enable-prompt-cache');
            return !!(el && el.checked);
        };
        const getPromptCacheLockedProvider = () => {
            if (!isPromptCacheEnabled()) return null;
            const modelEl = get('model-select');
            return getModelApiProvider(modelEl ? modelEl.value : '');
        };
        const updatePromptCacheUi = () => {
            const cacheCont = get('prompt-cache-container');
            const cacheChk = get('enable-prompt-cache');
            const modelBtn = get('model-selector-btn');
            if (!cacheChk) return;
            const enabled = !!cacheChk.checked;
            if (cacheCont) {
                cacheCont.classList.toggle('ring-1', enabled);
                cacheCont.classList.toggle('ring-teal-500/50', enabled);
                cacheCont.classList.toggle('rounded', enabled);
                cacheCont.classList.toggle('px-1', enabled);
            }
            if (modelBtn) {
                if (enabled) {
                    modelBtn.title = 'PromptCache有効: 同一APIプロバイダのモデルのみ選択可能';
                    modelBtn.classList.add('border-teal-500/60');
                } else {
                    modelBtn.title = '';
                    modelBtn.classList.remove('border-teal-500/60');
                }
            }
        };
        const bindPromptCacheControls = () => {
            const cacheChk = get('enable-prompt-cache');
            if (!cacheChk || cacheChk.dataset.bound === '1') return;
            cacheChk.dataset.bound = '1';
            cacheChk.addEventListener('change', () => {
                updatePromptCacheUi();
                if (cacheChk.checked) {
                    const prov = getModelApiProvider(get('model-select') ? get('model-select').value : '');
                    const label = PROVIDER_LABELS[prov] || prov || '現在のAPI';
                    showToast(`PromptCache を有効化しました。以降は ${label} 以外のモデルに変更できません。`, 'info', true);
                }
            });
        };
        const getModelMediaSupport = (model) => {
            const m = (model || '').toLowerCase();
            if (!m.includes('gemini')) return { audio: false, video: false };
            if (m.includes('image') || m.includes('nano') || m.includes('tts') || m.includes('native-audio') || m.includes('live')) {
                return { audio: false, video: false };
            }
            return { audio: true, video: true };
        };
        const supportsAudioInputModel = () => getModelMediaSupport(get('model-select').value).audio;
        const supportsVideoInputModel = () => getModelMediaSupport(get('model-select').value).video;
        const isImagePath = (path) => IMAGE_EXTS.includes(getFileExt(path || ''));
        const isAudioPath = (path) => AUDIO_EXTS.includes(getFileExt(path || ''));
        const isVideoPath = (path) => VIDEO_EXTS.includes(getFileExt(path || ''));
        const OPENAI_TTS_VOICES = ['alloy','ash','ballad','coral','echo','fable','nova','onyx','sage','shimmer','verse','marin','cedar'];
        const GEMINI_TTS_VOICES = [
            'Zephyr','Puck','Charon','Kore','Fenrir','Leda','Orus','Aoede','Callirrhoe','Autonoe',
            'Enceladus','Iapetus','Umbriel','Algieba','Despina','Erinome','Algenib','Rasalgethi','Laomedeia','Achernar',
            'Alnilam','Schedar','Gacrux','Pulcherrima','Achird','Zubenelgenubi','Vindemiatrix','Sadachbia','Sadaltager','Sulafat'
        ];
        const OPENAI_STS_VOICES = ['alloy','ash','ballad','coral','echo','sage','shimmer','verse','marin','cedar'];
        const GROK_STS_VOICES = ['Ara','Rex','Sal','Eve','Leo'];
        const GROK_TTS_VOICES = ['Eve','Ara','Rex','Sal','Leo'];
        const GEMINI_STS_VOICES = [
            'Zephyr','Puck','Charon','Kore','Fenrir','Leda','Orus','Aoede','Callirrhoe','Autonoe',
            'Enceladus','Iapetus','Umbriel','Algieba','Despina','Erinome','Algenib','Rasalgethi','Laomedeia','Achernar',
            'Alnilam','Schedar','Gacrux','Pulcherrima','Achird','Zubenelgenubi','Vindemiatrix','Sadachbia','Sadaltager','Sulafat'
        ];
        const GROK_PCM_RATES = [8000,16000,21050,24000,32000,44100,48000];
        const isTtsModel = () => get('model-select').value.includes('tts');
        const isGptImageModel = () => (get('model-select').value || '').includes('gpt-image');
        const isGeminiImageModel = () => isGeminiImageModelKey(get('model-select').value);
        const isLlmModel = () => {
            const m = (get('model-select').value || '').toLowerCase();
            if (
                m.includes('tts') ||
                m.includes('realtime') ||
                m.includes('voice-agent') ||
                m.includes('native-audio') ||
                m.includes('live') ||
                m.includes('image') ||
                m.includes('video')
            ) return false;
            if (m.includes('gemini') && (m.includes('image') || m.includes('nano'))) return false;
            return m.includes('gpt') || m.includes('gemini') || m.includes('grok') || m.includes('deepseek');
        };
        const isGrokImageModel = () => {
            const m = (get('model-select').value || '').toLowerCase();
            return m.includes('grok') && (m.includes('imagine') || m.includes('image')) && !m.includes('video');
        };
        const isGrokVideoModel = () => {
            const m = (get('model-select').value || '').toLowerCase();
            return m.includes('grok') && m.includes('video');
        };

        const isStsModel = () => STS_MODELS.has(get('model-select').value);
        const isGeminiLiveModel = () => get('model-select').value === 'gemini-3.1-flash-live-preview';
        const getStsProvider = (model) => {
            const m = (model || '').toLowerCase();
            if (m.includes('gpt-realtime')) return 'openai';
            if (m.includes('grok-voice')) return 'xai';
            if (m.includes('gemini') && (m.includes('native-audio') || m.includes('live'))) return 'gemini';
            return null;
        };
        function setStsStatus(text, recording = false) {
            const s = get('sts-status');
            const b = get('sts-mic-btn');
            if (s && text) s.innerText = text;
            if (b) {
                if (recording) {
                    b.classList.add('bg-red-600', 'animate-pulse');
                    b.classList.remove('bg-cyan-600');
                } else {
                    b.classList.remove('bg-red-600', 'animate-pulse');
                    b.classList.add('bg-cyan-600');
                }
            }
        }
        function updateStsUi() {
            const sts = isStsModel();
            const inputRow = get('input-row');
            const stsPanel = get('sts-panel');
            const filePreview = get('file-preview');
            if (sts) {
                if (inputRow) inputRow.classList.add('hidden');
                if (stsPanel) stsPanel.classList.remove('hidden');
                if (filePreview) filePreview.classList.add('hidden');
                setStsStatus('Tap to speak', false);
            } else {
                if (inputRow) inputRow.classList.remove('hidden');
                if (stsPanel) stsPanel.classList.add('hidden');
            }
        }
        function updateStsOptions() {
            if (!isStsModel()) return;
            const model = get('model-select').value || '';
            const provider = getStsProvider(model);
            const voiceSel = get('sts-voice');
            const speedWrap = get('sts-speed-wrap');
            const speedInput = get('sts-speed');
            const speedLabel = get('sts-speed-label');
            const rateWrap = get('sts-rate-wrap');
            const rateIn = get('sts-rate-in');
            const rateOut = get('sts-rate-out');
            const thinkingWrap = get('sts-thinking-wrap');
            const note = get('sts-note');

            if (provider === 'openai') {
                setSelectOptions(voiceSel, OPENAI_STS_VOICES, voiceSel.value || 'alloy');
                if (speedWrap) speedWrap.classList.remove('hidden');
                if (speedInput) {
                    speedInput.min = 0.25; speedInput.max = 1.5; speedInput.step = 0.05;
                    if (!speedInput.value) speedInput.value = 1;
                    if (Number(speedInput.value) < 0.25) speedInput.value = 0.25;
                    if (Number(speedInput.value) > 1.5) speedInput.value = 1.5;
                }
                if (rateWrap) rateWrap.classList.add('hidden');
                if (thinkingWrap) thinkingWrap.classList.add('hidden');
                if (note) note.textContent = 'OpenAI Realtimeは24kHz PCM固定';
            } else if (provider === 'xai') {
                setSelectOptions(voiceSel, GROK_STS_VOICES, voiceSel.value || 'Ara');
                if (speedWrap) speedWrap.classList.add('hidden');
                if (rateWrap) rateWrap.classList.remove('hidden');
                if (thinkingWrap) thinkingWrap.classList.add('hidden');
                setSelectOptions(rateIn, GROK_PCM_RATES, Number(rateIn.value || 24000));
                setSelectOptions(rateOut, GROK_PCM_RATES, Number(rateOut.value || 24000));
                if (note) note.textContent = 'xAIはPCMサンプルレート変更可';
            } else if (provider === 'gemini') {
                setSelectOptions(voiceSel, GEMINI_STS_VOICES, voiceSel.value || 'Kore');
                if (speedWrap) speedWrap.classList.add('hidden');
                if (rateWrap) rateWrap.classList.add('hidden');
                if (thinkingWrap) thinkingWrap.classList.remove('hidden');
                if (note) note.textContent = 'Gemini Liveは音声速度変更非対応';
            }
            if (speedWrap && speedLabel && speedInput && !speedWrap.classList.contains('hidden')) {
                speedLabel.textContent = `${Number(speedInput.value || 1).toFixed(2)}x`;
            }
        }
        function stsOpt(id) {
            const el = get(id);
            if (id === 'sts-auto-play' || id === 'sts-auto-restart') {
                return el ? !!el.checked : true;
            }
            return el ? !!el.checked : false;
        }
        function getStsSilenceMs() {
            const el = get('sts-silence-sec');
            let v = el ? parseFloat(el.value) : 1.5;
            if (isNaN(v) || v < 0.5) v = 0.5;
            if (v > 10) v = 10;
            return Math.round(v * 1000);
        }
        function getTtsProvider(model) {
            if (!model) return null;
            const m = model.toLowerCase();
            if (m.includes('google-tts')) return 'google';
            if (m.includes('gemini') && m.includes('tts')) return 'gemini';
            if (m.includes('grok-tts') || m.includes('xai-tts')) return 'xai';
            if (m.includes('tts')) return 'openai';
            return null;
        }
        function setSelectOptions(el, list, selected) {
            if (!el) return;
            el.innerHTML = '';
            list.forEach(v => {
                const opt = document.createElement('option');
                opt.value = v.value || v;
                opt.textContent = v.label || v;
                if ((v.value || v) === selected) opt.selected = true;
                el.appendChild(opt);
            });
        }
        function updateTtsUi() {
            const model = get('model-select').value || '';
            const provider = getTtsProvider(model);
            const wrap = get('audio-gen-options');
            if (!wrap) return;
            if (!provider) {
                wrap.classList.add('hidden');
                return;
            }
            wrap.classList.remove('hidden');
            const voiceSel = get('tts-voice');
            const voiceCustomWrap = get('tts-voice-custom-wrap');
            const voiceCustom = get('tts-voice-custom');
            const langWrap = get('tts-language-wrap');
            const langInput = get('tts-language');
            const speedWrap = get('tts-speed-wrap');
            const speedInput = get('tts-speed');
            const speedLabel = get('tts-speed-label');
            const speedNote = get('tts-speed-note');

            if (provider === 'openai') {
                setSelectOptions(voiceSel, OPENAI_TTS_VOICES, voiceSel.value || 'alloy');
                voiceCustomWrap.classList.add('hidden');
                langWrap.classList.add('hidden');
                if (speedInput) { 
                    speedInput.min = 0.25; speedInput.max = 4; speedInput.step = 0.05; 
                    if (!speedInput.value) speedInput.value = 1;
                    if (Number(speedInput.value) < 0.25) speedInput.value = 0.25;
                    if (Number(speedInput.value) > 4) speedInput.value = 4;
                    speedInput.disabled = false; 
                }
                if (speedNote) speedNote.textContent = '';
            } else if (provider === 'gemini') {
                setSelectOptions(voiceSel, GEMINI_TTS_VOICES, voiceSel.value || 'Kore');
                voiceCustomWrap.classList.add('hidden');
                langWrap.classList.add('hidden');
                if (speedInput) { speedInput.disabled = true; }
                if (speedNote) speedNote.textContent = '(Gemini TTSは速度変更非対応)';
            } else if (provider === 'google') {
                setSelectOptions(voiceSel, [
                    { value: 'auto', label: 'Auto (Studio/Neural2)' },
                    { value: 'custom', label: 'Custom Voice Name' }
                ], voiceSel.value || 'auto');
                if (voiceSel.value === 'custom') {
                    voiceCustomWrap.classList.remove('hidden');
                } else {
                    voiceCustomWrap.classList.add('hidden');
                    if (voiceCustom) voiceCustom.value = '';
                }
                langWrap.classList.remove('hidden');
                if (langInput && !langInput.value) langInput.value = 'ja-JP';
                if (speedInput) { 
                    speedInput.min = 0.25; speedInput.max = 2; speedInput.step = 0.05; 
                    if (!speedInput.value) speedInput.value = 1;
                    if (Number(speedInput.value) < 0.25) speedInput.value = 0.25;
                    if (Number(speedInput.value) > 2) speedInput.value = 2;
                    speedInput.disabled = false; 
                }
                if (speedNote) speedNote.textContent = '';
            } else if (provider === 'xai') {
                setSelectOptions(voiceSel, GROK_TTS_VOICES, voiceSel.value || 'Eve');
                voiceCustomWrap.classList.remove('hidden');
                langWrap.classList.remove('hidden');
                if (langInput && !langInput.value) langInput.value = 'ja';
                if (speedInput) { 
                    speedInput.min = 0.7; speedInput.max = 1.5; speedInput.step = 0.05; 
                    if (!speedInput.value) speedInput.value = 1;
                    if (Number(speedInput.value) < 0.7) speedInput.value = 0.7;
                    if (Number(speedInput.value) > 1.5) speedInput.value = 1.5;
                    speedInput.disabled = false; 
                }
                if (speedNote) speedNote.textContent = 'xAI TTS supports speed 0.7–1.5 and speech tags';
            }
            if (speedInput && speedLabel) {
                speedLabel.textContent = `${Number(speedInput.value || 1).toFixed(2)}x`;
            }
        }

        function getModelTags(m, group) {
            const tags = [];
            const id = (m.id || '').toLowerCase();
            const name = (m.name || '').toLowerCase();
            const desc = (m.desc || '').toLowerCase();
            const cat = (group.category || '').toLowerCase();
            if (
                cat.includes('gemini') ||
                id.includes('gemini') ||
                name.includes('gemini') ||
                desc.includes('gemini') ||
                cat.includes('banana') ||
                name.includes('banana')
            ) tags.push('gemini');
            if (
                cat.includes('deepseek') ||
                id.includes('deepseek') ||
                name.includes('deepseek') ||
                desc.includes('deepseek')
            ) tags.push('deepseek');
            if (
                cat.includes('gpt') ||
                cat.includes('openai') ||
                id.includes('gpt') ||
                name.includes('gpt') ||
                desc.includes('openai')
            ) tags.push('openai');
            if (
                cat.includes('xai') ||
                cat.includes('grok') ||
                id.includes('grok') ||
                name.includes('grok') ||
                desc.includes('xai')
            ) tags.push('xai');
            if (cat.includes('image') || id.includes('image') || name.includes('image') || desc.includes('image')) tags.push('image');
            if (
                cat.includes('audio') ||
                cat.includes('speech') ||
                id.includes('tts') ||
                name.includes('tts') ||
                desc.includes('tts') ||
                id.includes('realtime') ||
                id.includes('live') ||
                id.includes('voice-agent') ||
                id.includes('native-audio') ||
                name.includes('audio') ||
                desc.includes('audio')
            ) tags.push('audio');
            if (id.includes('reasoning') || name.includes('reasoning') || desc.includes('reasoning')) tags.push('reasoning');
            if ((cat.includes('deepseek') || id.includes('deepseek') || name.includes('deepseek')) && !tags.includes('reasoning')) tags.push('reasoning');
            if (id.includes('fast') || name.includes('fast') || desc.includes('fast') || cat.includes('fast')) tags.push('fast');
            if ((id.includes('deepseek-v4-flash') || (cat.includes('deepseek') && name.includes('flash'))) && !tags.includes('fast')) tags.push('fast');
            return tags;
        }

        function updateModelTagUi() {
            const bar = get('model-tag-bar');
            if (!bar) return;
            const btns = bar.querySelectorAll('.model-tag-btn');
            btns.forEach(b => {
                const t = b.innerText.trim().toLowerCase();
                const active = (t === 'all' ? 'all' : t) === activeModelTag;
                b.className = `model-tag-btn px-2 py-1 text-[10px] rounded border transition ${active ? 'bg-blue-600/20 border-blue-500 text-blue-300' : 'bg-gray-800 border-gray-700 text-gray-300 hover:border-gray-500'}`;
            });
        }

        function renderModelList(filter = "") {
            const container = get('model-list-container');
            if (!container) return;
            container.classList.remove('model-list-animate');
            container.innerHTML = '';
            const f = filter.toLowerCase();
            const lockedProvider = window._visionPickerActive ? null : getPromptCacheLockedProvider();
            const lockedLabel = lockedProvider ? (PROVIDER_LABELS[lockedProvider] || lockedProvider) : '';
            
            if (lockedProvider) {
                const banner = document.createElement('div');
                banner.className = 'mb-4 px-3 py-2 rounded-lg border border-teal-500/40 bg-teal-900/20 text-[11px] text-teal-200';
                banner.innerHTML = `<i class="fas fa-database mr-1.5"></i>PromptCache 有効中: <strong>${lockedLabel}</strong> のモデルのみ選択できます（他APIへの切替は不可）`;
                container.appendChild(banner);
            }

            MODELS.forEach(group => {
                const matches = group.items.filter(m => {
                    if (m.deprecated) return false;
                    const hit = m.name.toLowerCase().includes(f) || m.id.toLowerCase().includes(f);
                    if (!hit) return false;
                    if (lockedProvider && getModelApiProvider(m.id) !== lockedProvider) return false;
                    if (activeModelTag === 'all') return true;
                    return getModelTags(m, group).includes(activeModelTag);
                });
                if (matches.length > 0) {
                    const groupEl = document.createElement('div');
                    groupEl.innerHTML = `
                        <div class="flex items-center gap-2 mb-3 px-2">
                            <i class="${group.icon}"></i>
                            <div>
                                <h3 class="font-bold text-gray-200 text-sm">${group.category}</h3>
                                <p class="text-[10px] text-gray-500">${group.description}</p>
                            </div>
                        </div>
                        <div class="grid grid-cols-1 md:grid-cols-2 gap-2 mb-6"></div>
                    `;
                    const grid = groupEl.querySelector('.grid');
                    
                    matches.forEach(m => {
                        const item = document.createElement('button');
                        const selectedForHighlight = get('model-select') ? get('model-select').value : '';
                        const isSelected = selectedForHighlight === m.id;
                        const priceHtml = m.price ? `<div class="text-[10px] text-amber-400/90 mt-1.5 font-mono flex items-start gap-1"><i class="fas fa-tag text-[9px] mt-0.5 opacity-70 shrink-0"></i><span>${m.price}</span></div>` : '';
                        item.className = `flex flex-col text-left p-3 rounded-lg border transition ${isSelected ? 'bg-blue-600/20 border-blue-500 ring-1 ring-blue-500' : 'bg-gray-800 border-gray-700 hover:border-gray-500 hover:bg-gray-750'}`;
                        item.onclick = () => selectModel(m.id, m.name);
                        item.innerHTML = `
                            <div class="flex justify-between items-center w-full mb-1">
                                <span class="font-bold text-sm text-gray-200">${m.name}</span>
                                ${isSelected ? '<i class="fas fa-check-circle text-blue-400"></i>' : ''}
                            </div>
                            <span class="text-[10px] text-gray-400">${m.desc}</span>
                            ${priceHtml}
                        `;
                        grid.appendChild(item);
                    });
                    container.appendChild(groupEl);
                }
            });
            
            if (container.children.length === 0 || (lockedProvider && container.children.length === 1 && !container.querySelector('.grid'))) {
                const empty = document.createElement('div');
                empty.className = 'text-center text-gray-500 py-8';
                empty.textContent = lockedProvider
                    ? `No ${lockedLabel} models found.`
                    : 'No models found.';
                container.appendChild(empty);
            }
            requestAnimationFrame(() => container.classList.add('model-list-animate'));
        }

        function openModelModal() {
            showModal('model-modal');
            if (location.pathname !== '/model') {
                history.pushState({ modal: 'model' }, '', '/model');
            }
            get('model-search').value = '';
            // Prevent auto-focus on mobile to avoid keyboard popup
            if (window.innerWidth > 768) {
                get('model-search').focus();
            }
            updateModelTagUi();
            renderModelList();
        }
        window.closeModelModal = (skipHistory = false) => {
            hideModal('model-modal');
            if (!skipHistory && location.pathname === '/model') {
                history.back();
            }
        };

        if (get('token-detail-modal')) {
            get('token-detail-modal').addEventListener('click', (e) => {
                if (e.target.id === 'token-detail-modal') closeTokenDetail();
            });
        }
        if (get('encryption-status-modal')) {
            get('encryption-status-modal').addEventListener('click', (e) => {
                if (e.target.id === 'encryption-status-modal') closeEncryptionModal();
            });
        }

        function selectModel(id, name) {
            if (window._visionPickerActive) {
                currentVisionModel = id;
                window._visionPickerActive = false;
                window.closeModelModal();
                _syncVisionModelDisplay();
                return;
            }
            if (isPromptCacheEnabled()) {
                const currentProvider = getModelApiProvider(get('model-select') ? get('model-select').value : '');
                const nextProvider = getModelApiProvider(id);
                if (currentProvider && nextProvider && currentProvider !== nextProvider) {
                    const curLabel = PROVIDER_LABELS[currentProvider] || currentProvider;
                    const nextLabel = PROVIDER_LABELS[nextProvider] || nextProvider;
                    showToast(`PromptCache 有効中は他API（${nextLabel}）のモデルに変更できません。現在: ${curLabel}`, 'warning', true);
                    return;
                }
            }
            const el = get('model-select');
            el.value = id;
            get('model-selector-text').innerText = name;
            window.closeModelModal();
            const event = new Event('change');
            el.dispatchEvent(event);
        }
        function selectModelById(id) {
            let name = id;
            for (const g of MODELS) {
                const found = g.items.find(i => i.id === id);
                if (found) { name = found.name; break; }
            }
            selectModel(id, name);
        }

        function populateAiSafeFormFields(d) {
            // Mirror a subset of the population in openSettingsModal for live update after AI apply
            if (!d) return;
            try {
                if (get('set-default-model')) get('set-default-model').value = d.default_model || get('set-default-model').value;
                if (get('set-default-vision-model')) get('set-default-vision-model').value = d.default_vision_model || 'gemini-3-flash-preview';
                if (get('set-default-search')) get('set-default-search').checked = !!d.default_enable_search;
                if (get('set-default-url-context')) get('set-default-url-context').checked = !!d.default_enable_url_context;
                if (get('set-default-maps')) get('set-default-maps').checked = !!d.default_enable_maps;
                if (get('set-default-python')) get('set-default-python').checked = !!d.default_enable_python;
                if (get('set-default-thinking')) get('set-default-thinking').checked = !!d.default_enable_thinking;
                if (get('set-default-sys-prompt')) get('set-default-sys-prompt').checked = !!d.default_enable_system_prompt;
                if (get('set-default-thinking-level')) get('set-default-thinking-level').value = d.default_thinking_level || 'high';
                if (get('set-default-thinking-budget')) get('set-default-thinking-budget').value = d.default_thinking_budget || 4096;
                if (get('set-default-reasoning-effort')) get('set-default-reasoning-effort').value = d.default_reasoning_effort || 'medium';
                if (get('set-default-safety')) get('set-default-safety').value = d.default_safety_setting || 'default';
                if (get('sys-prompt-text')) get('sys-prompt-text').value = d.system_prompt || '';
                if (get('set-global-sys-prompt-enabled')) get('set-global-sys-prompt-enabled').checked = d.system_prompt_enabled !== false;
                if (get('set-apply-global-sys-prompt')) get('set-apply-global-sys-prompt').checked = d.apply_global_system_prompt !== false;
                if (get('set-apply-auto-sys-prompt-notices')) get('set-apply-auto-sys-prompt-notices').checked = d.apply_auto_system_prompt_notices !== false;
                if (get('set-mic-transcribe-mode')) get('set-mic-transcribe-mode').value = d.mic_transcribe_mode || 'stt_api';
                if (get('set-stt-model')) get('set-stt-model').value = d.stt_model || 'gpt-4o-mini-transcribe';
                if (get('set-llm-transcribe-prompt')) get('set-llm-transcribe-prompt').value = d.llm_transcribe_prompt || '';
                if (get('set-enter-to-send')) get('set-enter-to-send').checked = !!d.enter_to_send;
                if (get('set-compact-prompt-mode')) get('set-compact-prompt-mode').checked = !!d.compact_prompt_mode;
                if (get('set-use-sw-cache')) get('set-use-sw-cache').checked = !!d.use_sw_cache;
                if (get('set-liquid-glass')) get('set-liquid-glass').checked = !!d.liquid_glass_enabled;
                applyLiquidGlassMode(!!d.liquid_glass_enabled);
                if (get('set-auto-search-links')) get('set-auto-search-links').checked = d.auto_search_on_links !== false;
                if (get('set-use-last-settings')) get('set-use-last-settings').checked = !!d.use_last_chat_settings;
                if (get('set-latency-metrics')) get('set-latency-metrics').checked = !!d.enable_latency_metrics;
                if (get('set-client-debug-log')) syncClientDebugLogToggle(!!d.enable_client_debug_log, 'ai-settings');
                if (get('set-bot-detect')) get('set-bot-detect').checked = d.bot_detection_enabled !== false;
                if (get('set-skip-2fa-google')) get('set-skip-2fa-google').checked = !!d.skip_2fa_on_google_login;
                if (get('set-default-2fa-method')) get('set-default-2fa-method').value = d.default_2fa_method || 'totp';
                // theme etc handled elsewhere if needed
            } catch (e) { /* element missing ok */ }
        }

        get('model-search').addEventListener('input', (e) => renderModelList(e.target.value));
        if (get('model-tag-bar')) {
            get('model-tag-bar').addEventListener('click', (e) => {
                const btn = e.target.closest('.model-tag-btn');
                if (!btn) return;
                const t = btn.innerText.trim().toLowerCase();
                activeModelTag = MODEL_TAGS.includes(t) ? t : 'all';
                updateModelTagUi();
                renderModelList(get('model-search').value);
            });
            updateModelTagUi();
        }

        window.quickStart = (m) => {
            selectModelById(m);
            get('welcome-screen').classList.add('hidden');
        };

        // Critical UI initializations - independent listener to survive errors in main init
        document.addEventListener('DOMContentLoaded', () => {
            if (get('menu-btn')) {
                get('menu-btn').onclick = () => { get('sidebar').classList.toggle('open'); get('overlay').classList.toggle('active'); };
            }
            if (get('overlay')) {
                get('overlay').onclick = () => { get('sidebar').classList.remove('open'); get('overlay').classList.remove('active'); };
            }
        });

        document.addEventListener('DOMContentLoaded', () => {
            initThemeFromServer();
            applyLiquidGlassMode(INITIAL_LIQUID_GLASS_ENABLED);
            updateCurrentChatHeaderUi();
            ensureCurrentChatHeaderTicker();
            const bar = document.getElementById('alpha-bar'); setTimeout(() => { if(bar) { const target = document.getElementById('version-display'); if(target) { const barRect = bar.getBoundingClientRect(); const targetRect = target.getBoundingClientRect(); const tx = targetRect.left + (targetRect.width/2) - (barRect.left + barRect.width/2); const ty = targetRect.top + (targetRect.height/2) - (barRect.top + barRect.height/2); bar.style.transform = `translate(${tx}px, ${ty}px) scale(0.1)`; bar.style.opacity = '0'; setTimeout(() => { target.classList.add('pulse-target'); setTimeout(() => target.classList.remove('pulse-target'), 2000); bar.remove(); }, 800); } else { bar.style.opacity = '0'; setTimeout(() => bar.remove(), 1000); } } }, 3000);
            function updateGptImageUi() {
                const wrap = get('gpt-image-options');
                if (!wrap) return;
                if (isGptImageModel()) {
                    wrap.classList.remove('hidden');
                } else {
                    wrap.classList.add('hidden');
                }
                const fmt = get('gpt-image-format');
                const compWrap = get('gpt-image-compression-wrap');
                if (fmt && compWrap) {
                    if (fmt.value === 'png') {
                        compWrap.classList.add('hidden');
                    } else {
                        compWrap.classList.remove('hidden');
                    }
                }
            }
            function updateGeminiImageUi() {
                const wrap = get('gemini-image-options');
                if (!wrap) return;
                if (isGeminiImageModel()) {
                    wrap.classList.remove('hidden');
                } else {
                    wrap.classList.add('hidden');
                }
                const model = (get('model-select').value || '').toLowerCase();
                const sizeEl = get('gemini-image-size');
                if (!sizeEl) return;
                const isFlashImage = model.includes('gemini') && model.includes('flash-image');
                Array.from(sizeEl.options).forEach(opt => {
                    if (opt.value !== '1K') {
                        opt.disabled = isFlashImage;
                    }
                });
                if (isFlashImage && sizeEl.value !== '1K') {
                    sizeEl.value = '1K';
                }
            }
        function updateGrokImageUi() {
            const wrap = get('grok-image-options');
            if (!wrap) return;
            const model = (get('model-select').value || '').toLowerCase();
            const isGrokImg = isGrokImageModel();
            if (isGrokImg) {
                wrap.classList.remove('hidden');
                const resWrap = get('grok-image-resolution') ? get('grok-image-resolution').parentElement : null;
                if (resWrap) {
                    resWrap.classList.toggle('hidden', model !== 'grok-imagine-image-quality');
                }
            } else {
                wrap.classList.add('hidden');
            }
            
            // Modal part
            const modalWrap = get('modal-grok-image-options');
            if (modalWrap) {
                const resWrapModal = get('modal-grok-image-resolution') ? get('modal-grok-image-resolution').parentElement : null;
                if (resWrapModal) {
                    resWrapModal.classList.toggle('hidden', model !== 'grok-imagine-image-quality');
                }
            }
        }
        function updateGrokVideoUi() {
            const wrap = get('grok-video-options');
            if (!wrap) return;
            if (isGrokVideoModel()) {
                wrap.classList.remove('hidden');
            } else {
                wrap.classList.add('hidden');
            }
        }
        function updateImageInputLimits() {
            const el = get('image-input-limits');
            if (!el) return;
            const model = (get('model-select').value || '').toLowerCase();
            let html = '';
            let show = false;
            if (model.includes('gpt-image')) {
                show = true;
                html = [
                    '<div class="font-bold text-gray-300 mb-1">GPT-Image 入力制限</div>',
                    '<div>最大 16 枚 / 画像1枚あたり 50MB 未満 / PNG・JPG・WEBP</div>',
                    '<div>マスク使用時: PNGのみ、4MB未満、元画像と同サイズ</div>'
                ].join('');
            } else if (model.includes('deepseek')) {
                show = true;
                html = [
                    '<div class="font-bold text-gray-300 mb-1">DeepSeek V4 入力制限</div>',
                    '<div>テキスト専用。画像・音声・動画入力は非対応です。</div>'
                ].join('');
            } else if (isGeminiImageModelKey(model)) {
                show = true;
                if (model.includes('gemini-3.1-flash-lite-image')) {
                    html = [
                        '<div class="font-bold text-gray-300 mb-1">Nano Banana 2 Lite 入力目安</div>',
                        '<div>画像入力は最大1枚を推奨（Gemini 3.1 Flash Lite Image）</div>',
                        '<div>複数参照入力やマルチターン編集には最適化されていません</div>'
                    ].join('');
                } else if (model.includes('gemini-3.1-flash-image')) {
                    html = [
                        '<div class="font-bold text-gray-300 mb-1">Nano Banana 2 入力目安</div>',
                        '<div>画像入力は最大3枚程度を推奨（Gemini 3.1 Flash Image）</div>'
                    ].join('');
                } else if (model.includes('gemini-2.5') && model.includes('image')) {
                    html = [
                        '<div class="font-bold text-gray-300 mb-1">Nano Banana 入力目安</div>',
                        '<div>画像入力は最大3枚までが推奨</div>'
                    ].join('');
                } else {
                    html = [
                        '<div class="font-bold text-gray-300 mb-1">Nano Banana Pro 入力目安</div>',
                        '<div>高精度は最大5枚 / 合計14枚まで対応</div>'
                    ].join('');
                }
            } else if (model.includes('grok')) {
                show = true;
                html = [
                    '<div class="font-bold text-gray-300 mb-1">Grok 画像入力制限</div>',
                    '<div>最大 20MiB / PNG・JPG のみ / 枚数制限なし</div>'
                ].join('');
            } else if (model.includes('grok') && model.includes('video')) {
                show = true;
                html = [
                    '<div class="font-bold text-gray-300 mb-1">Grok 動画生成制限</div>',
                    '<div>Duration: 1-15s / Resolution: 720p, 480p</div>',
                    '<div>画像からの動画生成に対応 (PNG・JPG)</div>'
                ].join('');
            }
            if (show) {
                el.innerHTML = html;
                el.classList.remove('hidden');
            } else {
                el.classList.add('hidden');
                el.innerHTML = '';
            }
        }
            function toggleOptions() { 
                const modelEl = get('model-select');
                if (!modelEl) return;
                const model = modelEl.value; 
                const modelLower = String(model || '').toLowerCase();
                const isDeepSeek = modelLower.includes('deepseek');
                const thinkOpts = get('thinking-options'); 
                const reasonOpts = get('reasoning-effort-container'); 
                const thinkChk = get('enable-thinking'); 
                const thinkLvl = get('thinking-level');
                const thinkBudget = get('thinking-budget');
                const searchChk = get('enable-search'); 
                const searchCont = get('search-container'); 
                const urlCont = get('url-context-container');
                const mapsChk = get('enable-maps');
                const mapsCont = get('maps-grounding-container');
                const sysChk = get('enable-sys-prompt'); const sysLbl = get('sys-prompt-option'); 
                const pyChk = get('enable-python'); const pyCont = get('python-container');
                const cacheCont = get('prompt-cache-container');
                const cacheChk = get('enable-prompt-cache');
                const isSearchModel = model === 'gpt-5-search-api';
                const isTts = model.includes('tts');
                const isNanoBanana2 = modelLower.includes('gemini-3.1-flash-image') && !modelLower.includes('lite');
                const isClaude = isClaudeModelKey(model);
                const promptCacheSupported = isLlmModel() && !isTts && !modelLower.includes('realtime') && !modelLower.includes('native-audio') && !modelLower.includes('live');
                if (cacheCont) {
                    if (promptCacheSupported) {
                        cacheCont.classList.remove('hidden', 'opacity-50', 'pointer-events-none');
                        if (cacheChk) cacheChk.disabled = false;
                    } else {
                        if (cacheChk) { cacheChk.checked = false; cacheChk.disabled = true; }
                        cacheCont.classList.add('opacity-50', 'pointer-events-none');
                    }
                }
                updatePromptCacheUi();
                
                if (thinkOpts) thinkOpts.classList.add('hidden'); 
                if (reasonOpts) reasonOpts.classList.add('hidden'); 
                const vmi = get('vision-model-info');
                if (vmi) vmi.classList.add('hidden');
                if (reasonOpts) {
                    const effortSel = get('reasoning-effort');
                    if (effortSel) {
                        Array.from(effortSel.options).forEach(opt => {
                            if (opt.value === 'xhigh') {
                                opt.classList.toggle('hidden', !modelLower.includes('multi-agent'));
                            } else if (opt.value === 'medium') {
                                opt.classList.toggle('hidden', !(modelLower.includes('grok-4.3') || modelLower.includes('grok-build') || modelLower.includes('multi-agent') || modelLower.includes('gpt-5') || modelLower.includes('o1') || modelLower.includes('o3')));
                            } else if (opt.value === 'none') {
                                opt.classList.toggle('hidden', !modelLower.includes('grok-4.3') && !modelLower.includes('grok-build') && !modelLower.includes('gpt-5'));
                            }
                        });
                    }
                }
                if(urlCont) urlCont.classList.add('hidden');
                if(mapsCont) mapsCont.classList.add('hidden');
                if(thinkChk) thinkChk.disabled = false; 
                
                if(thinkBudget) {
                    thinkBudget.disabled = true;
                    thinkBudget.classList.add('opacity-50');
                }
                const isGeminiImage = isGeminiImageModelKey(model);
                if(isTts) {
                    if(searchCont) { get('enable-search').checked = false; searchCont.classList.add('opacity-50', 'pointer-events-none'); }
                    if(urlCont) { get('enable-url-context').checked = false; urlCont.classList.add('opacity-50', 'pointer-events-none'); }
                    if(mapsCont && mapsChk) { mapsChk.checked = false; mapsCont.classList.add('opacity-50', 'pointer-events-none'); }
                    if(pyCont) { pyChk.checked = false; pyCont.classList.add('opacity-50', 'pointer-events-none'); }
                    sysChk.checked = false; sysChk.disabled = true; sysLbl.classList.add('opacity-50');
                } else if (isNanoBanana2) {
                    if (mapsCont && mapsChk) {
                        mapsChk.checked = false;
                        mapsCont.classList.add('hidden', 'opacity-50', 'pointer-events-none');
                    }
                    thinkOpts.classList.remove('hidden');
                    Array.from(thinkLvl.options).forEach(opt => {
                        if (['low', 'medium'].includes(opt.value)) opt.disabled = true;
                        if (['minimal', 'high'].includes(opt.value)) opt.disabled = false;
                    });
                    if (!['minimal', 'high'].includes(thinkLvl.value)) {
                        thinkLvl.value = 'high';
                    }
                    if (thinkChk) thinkChk.disabled = false;
                } else if (isGeminiImage) {
                    if (mapsCont && mapsChk) {
                        mapsChk.checked = false;
                        mapsCont.classList.add('hidden', 'opacity-50', 'pointer-events-none');
                    }
                } else if (isClaude) {
                    thinkOpts.classList.remove('hidden');
                    if (thinkBudget) {
                        thinkBudget.disabled = false;
                        thinkBudget.classList.remove('opacity-50');
                    }
                    Array.from(thinkLvl.options).forEach(opt => {
                        opt.disabled = true; // Claude thinking only uses budget and enabled flag
                    });
                    if (pyCont) { pyChk.checked = false; pyCont.classList.add('opacity-50', 'pointer-events-none'); }
                } else if(model.includes('gemini') && !isGeminiImage) { 
                    thinkOpts.classList.remove('hidden'); 
                    if(urlCont) {
                        urlCont.classList.remove('hidden', 'opacity-50', 'pointer-events-none');
                    }
                    const isGemini3 = model.includes('gemini-3') || model.includes('gemini-3.1');
                    if (mapsCont) {
                        if (isGemini3) {
                            mapsCont.classList.remove('hidden', 'opacity-50', 'pointer-events-none');
                        } else {
                            if (mapsChk) mapsChk.checked = false;
                            mapsCont.classList.add('hidden', 'opacity-50', 'pointer-events-none');
                        }
                    }
                    const isFlash = model.includes('flash');
                    Array.from(thinkLvl.options).forEach(opt => {
                        if(['minimal', 'medium'].includes(opt.value)) { opt.disabled = !isFlash; }
                    });
                    if(!isFlash && ['minimal', 'medium'].includes(thinkLvl.value)) { thinkLvl.value = 'high'; }
                    if(isGemini3) { 
                        if(thinkChk) { thinkChk.checked = true; thinkChk.disabled = true; } 
                    } else if(thinkChk) {
                        thinkChk.disabled = false;
                    }
                    if(thinkBudget && model.includes('gemini-2.5')) {
                        thinkBudget.disabled = false;
                        thinkBudget.classList.remove('opacity-50');
                    }
                    if(thinkBudget && !model.includes('gemini-2.5')) {
                        thinkBudget.disabled = true;
                        thinkBudget.classList.add('opacity-50');
                    }
                } 
                else if((modelLower.includes('gpt-5') || modelLower.includes('o1') || modelLower.includes('o3') || modelLower.includes('grok-4.3') || modelLower.includes('grok-build') || modelLower.includes('multi-agent') || (modelLower.includes('gpt') && !modelLower.includes('tts')))) { 
                    reasonOpts.classList.remove('hidden'); 
                    if(searchCont) searchCont.classList.remove('opacity-50', 'pointer-events-none'); 
                } else if (isDeepSeek) {
                    reasonOpts.classList.remove('hidden');
                    const vmi = get('vision-model-info');
                    if (vmi) vmi.classList.remove('hidden');
                    if (searchChk) {
                        searchChk.checked = false;
                        searchChk.disabled = true;
                    }
                    if (searchCont) searchCont.classList.add('opacity-50', 'pointer-events-none');
                    if (urlCont) {
                        const urlChk = get('enable-url-context');
                        if (urlChk) urlChk.checked = false;
                        urlCont.classList.add('opacity-50', 'pointer-events-none');
                    }
                    if (mapsCont && mapsChk) {
                        mapsChk.checked = false;
                        mapsCont.classList.add('opacity-50', 'pointer-events-none');
                    }
                    if (pyCont) {
                        pyChk.checked = false;
                        pyCont.classList.add('hidden', 'opacity-50', 'pointer-events-none');
                        pyChk.disabled = true;
                    }
                } 
                else { 
                    if(searchCont) searchCont.classList.remove('opacity-50', 'pointer-events-none'); 
                    if(mapsCont && mapsChk) {
                        mapsChk.checked = false;
                        mapsCont.classList.add('hidden', 'opacity-50', 'pointer-events-none');
                    }
                } 
                
                // TTS Special Handling
                if(isTts) {
                    if(pyCont) pyCont.classList.add('opacity-50', 'pointer-events-none');
                } else {
                    if(pyCont) pyCont.classList.remove('opacity-50', 'pointer-events-none');
                    if((!isGeminiImage || isNanoBanana2) && !model.includes('gpt-image')) {
                         sysChk.disabled = false; sysLbl.classList.remove('opacity-50');
                    }
                }

                if(((isGeminiImage && !isNanoBanana2) || model.includes('gpt-image') || isGrokImageModel() || isGrokVideoModel())) { sysChk.checked = false; sysChk.disabled = true; sysLbl.classList.add('opacity-50'); }
                if (pyCont) {
                    if (isLlmModel() && !isDeepSeek) {
                        pyCont.classList.remove('hidden');
                        pyChk.disabled = false;
                    } else {
                        pyChk.checked = false;
                        pyChk.disabled = true;
                        pyCont.classList.add('hidden');
                    }
                }
                if (isSearchModel) {
                    if (searchChk) {
                        searchChk.checked = true;
                        searchChk.disabled = true;
                    }
                    if (searchCont) searchCont.classList.add('opacity-50', 'pointer-events-none');
                    if (pyCont) {
                        pyChk.checked = false;
                        pyChk.disabled = true;
                        pyCont.classList.add('opacity-50', 'pointer-events-none');
                    }
                } else if (searchChk && !model.includes('tts') && !isDeepSeek) {
                    searchChk.disabled = false;
                }
                const maskBtn = get('mask-btn');
                if (maskBtn) {
                    if (isGptImageModel()) {
                        maskBtn.classList.remove('hidden');
                    } else {
                        maskBtn.classList.add('hidden');
                        currentMaskImage = null;
                        updateMaskPreview();
                    }
                }
                updateTtsUi();
                updateStsUi();
                updateStsOptions();
                updateGptImageUi();
                updateGeminiImageUi();
                updateGrokImageUi();
                updateGrokVideoUi();
                updateImageInputLimits();
                purgeUnsupportedAttachments(true);
            }
            if (get('model-select')) {
                get('model-select').addEventListener('change', toggleOptions);
                get('model-select').addEventListener('change', () => schedulePromptTokenEstimate(true));
            }
            bindPromptCacheControls();
            toggleOptions();
            setCompactPromptMode(compactPromptMode, true);
            const canvasModeCheckbox = get('enable-canvas-mode');
            if (canvasModeCheckbox) {
                canvasModeCheckbox.checked = canvasModeEnabled;
                canvasModeCheckbox.addEventListener('change', () => syncCanvasModeUi(canvasModeCheckbox.checked));
            }
            syncCanvasModeUi(canvasModeEnabled, { persist: false, skipReset: false });
            if (get('canvas-panel-close-btn')) {
                get('canvas-panel-close-btn').addEventListener('click', () => syncCanvasModeUi(false));
            }
            if (get('canvas-panel-clear-btn')) {
                get('canvas-panel-clear-btn').addEventListener('click', () => {
                    if (!canvasModeEnabled) return;
                    resetCanvasPreviewPanel();
                    showToast('Canvasプレビューをクリアしました', 'info', false);
                });
            }
            if (get('canvas-block-list')) {
                get('canvas-block-list').addEventListener('click', (e) => {
                    const btn = e.target.closest('[data-canvas-block-index]');
                    if (!btn) return;
                    const index = Number(btn.getAttribute('data-canvas-block-index'));
                    applyCanvasSelection(index);
                });
            }
            if (get('canvas-panel-tabs')) {
                get('canvas-panel-tabs').addEventListener('click', (e) => {
                    const btn = e.target.closest('[data-canvas-panel-view]');
                    if (!btn) return;
                    const view = btn.getAttribute('data-canvas-panel-view');
                    syncCanvasPanelViewUi(view, { focus: false });
                });
            }
            if (get('canvas-panel-copy-btn')) {
                get('canvas-panel-copy-btn').addEventListener('click', () => {
                    const els = getCanvasModeElements();
                    const text = els && els.code ? (els.code.textContent || '') : '';
                    if (!text.trim()) {
                        showToast('コピーするコードがありません', 'info', false);
                        return;
                    }
                    copyToClipboard(text, () => showToast('Canvasコードをコピーしました', 'success'), () => showToast('コピーに失敗しました', 'error', true));
                });
            }
            const promptControlsToggleBtn = get('prompt-controls-toggle-btn');
            if (promptControlsToggleBtn) {
                promptControlsToggleBtn.onclick = () => togglePromptControlDetails();
            }
            if (get('tts-voice')) get('tts-voice').addEventListener('change', updateTtsUi);
            if (get('gpt-image-format')) get('gpt-image-format').addEventListener('change', () => updateGptImageUi());
            if (get('gemini-image-size')) get('gemini-image-size').addEventListener('change', () => updateGeminiImageUi());
            if (get('tts-speed') && get('tts-speed-label')) {
                get('tts-speed').addEventListener('input', () => {
                    get('tts-speed-label').textContent = `${Number(get('tts-speed').value || 1).toFixed(2)}x`;
                });
            }
            if (get('sts-speed') && get('sts-speed-label')) {
                get('sts-speed').addEventListener('input', () => {
                    get('sts-speed-label').textContent = `${Number(get('sts-speed').value || 1).toFixed(2)}x`;
                });
            }
            marked.use({
                renderer: {
                    code(c, i, e) {
                        const l = (i || '').match(/\S*/)[0];
                        if (l === 'pyexec') {
                            try {
                                const obj = JSON.parse(c);
                                const codeRaw = obj.code == null ? '' : String(obj.code);
                                const outputRaw = obj.output == null ? '' : String(obj.output);
                                let codeHtml = '';
                                try {
                                    codeHtml = hljs.highlight(codeRaw, { language: 'python' }).value;
                                } catch (e3) {
                                    codeHtml = escapeHtml(codeRaw);
                                }
                                const outputHtml = escapeHtml(outputRaw);
                                const encCode = encodeURIComponent(codeRaw).replace(/'/g, "%27");
                                const encOut = encodeURIComponent(outputRaw).replace(/'/g, "%27");
                                const codeKey = hashString(`pyexec\n${codeRaw}\n${outputRaw}`);
                                const downloadBtn = `<button class="download-btn" data-code="${encCode}" data-lang="python"><i class="fas fa-download"></i> DL Code</button>`;
                                return `<div class="code-wrapper python-box collapsed" data-collapsed="true" data-code-key="${codeKey}"><div class="code-header"><span class="code-lang"><i class="fas fa-terminal"></i> Python Execution</span><div class="code-actions"><button class="code-toggle" aria-expanded="false"><i class="fas fa-chevron-down"></i> Expand</button>${downloadBtn}<button class="copy-btn" data-copy="code" data-code="${encCode}"><i class="fas fa-copy"></i> Copy Code</button><button class="copy-btn" data-copy="output" data-code="${encOut}"><i class="fas fa-copy"></i> Copy Output</button></div></div><div class="code-body"><div class="python-section"><div class="python-label">Code</div><pre><code class="hljs language-python python-code">${codeHtml}</code></pre></div><div class="python-section"><div class="python-label">Output</div><pre><code class="hljs language-plaintext python-output">${outputHtml}</code></pre></div></div></div>`;
                            } catch (e2) {}
                        }
                        const raw = c || '';
                        const lowerLang = (l || '').toLowerCase();
                        let h = '';
                        try {
                            const lang = hljs.getLanguage(l) ? l : 'plaintext';
                            // Optimization: Skip heavy highlighting for very large blocks during streaming
                            if (activeStreamingBubbleId && raw.length > 20000) {
                                h = escapeHtml(raw);
                            } else {
                                h = hljs.highlight(raw, { language: lang }).value;
                            }
                        } catch (err) {
                            h = escapeHtml(raw);
                        }
                        const enc = encodeURIComponent(raw).replace(/'/g, "%27");
                        const isSuspicious = detectBlockedScriptsInCode(raw);
                        const codeKey = hashString(`${l || 'TEXT'}\n${raw || ''}`);
                        let previewBtn = '';
                        if (canvasModeEnabled) {
                            const isActiveCanvasBlock = String(canvasPreviewState.selectedKey || '') === codeKey;
                            previewBtn = `<button class="canvas-preview-btn${isActiveCanvasBlock ? ' canvas-active' : ''}" data-code="${enc}" data-code-key="${codeKey}" data-canvas-lang="${escapeHtml(l || 'txt')}" title="${isActiveCanvasBlock ? 'Canvasで表示中' : 'Canvasでプレビューする'}" aria-pressed="${isActiveCanvasBlock ? 'true' : 'false'}"><i class="fas ${isActiveCanvasBlock ? 'fa-layer-group' : 'fa-window-restore'}"></i> ${isActiveCanvasBlock ? 'Canvasで表示中' : 'Canvasに表示'}</button>`;
                        } else if (isHtmlPreviewCandidate(lowerLang, raw)) {
                            const label = isSuspicious ? 'Safe Preview' : 'Preview';
                            const icon = isSuspicious ? 'fa-shield-halved' : 'fa-up-right-from-square';
                            previewBtn = `<button class="html-preview-btn" data-code="${enc}" ${isSuspicious ? 'data-suspicious="1"' : ''}><i class="fas ${icon}"></i> ${label}</button>`;
                        }
                        const downloadBtn = `<button class="download-btn" data-code="${enc}" data-lang="${l || 'txt'}"><i class="fas fa-download"></i> Download</button>`;
                        const langLabel = (l || 'TEXT') + (isSuspicious ? ' <span class="suspicious-badge" title="polyfill.io などの危険スクリプトURLを検出しました">⚠</span>' : '');
                        return `<div class="code-wrapper collapsed" data-collapsed="true" data-code-key="${codeKey}"><div class="code-header"><span class="code-lang">${langLabel}</span><div class="code-actions"><button class="code-toggle" aria-expanded="false"><i class="fas fa-chevron-down"></i> Expand</button>${previewBtn}${downloadBtn}<button class="copy-btn" data-code="${enc}"><i class="fas fa-copy"></i> Copy</button></div></div><div class="code-body"><pre><code class="hljs language-${l}">${h}</code></pre></div></div>`;
                    },
                    link(h, t, x) { return `<a href="${h}" title="${t || ''}" target="_blank">${x}</a>`; },
                    image(h, t, x) { const alt = escapeHtml(x || ''); const title = t ? ` title="${escapeHtml(t)}"` : ''; const viewerSrc = escapeHtml(h || ''); return `<img src="${h}" data-viewer-src="${viewerSrc}" alt="${alt}"${title} class="chat-image" loading="lazy" width="320" height="320">`; }
                },
                breaks: true,
                gfm: true
            });
            
            // Infinite Scroll Observer
            threadObserver = new IntersectionObserver((entries) => {
                if(entries[0].isIntersecting && hasMoreThreads) loadThreads(true);
            }, { root: get('thread-list'), threshold: 0.1 });
            threadObserver.observe(get('scroll-sentinel'));
            
            initLowBandwidthMode();
            checkVersion();
            get('version-update-dismiss')?.addEventListener('click', () => {
                const latest = localStorage.getItem("app_version") || "";
                if (latest) localStorage.setItem("version_notified", latest);
                hideModal('version-update-modal');
            });
            const versionUpdateClearCacheSetting = get('version-update-clear-cache');
            if (versionUpdateClearCacheSetting) {
                versionUpdateClearCacheSetting.checked = !!(window.CHAT_CONFIG && window.CHAT_CONFIG.clearCacheOnVersionUpdate);
                versionUpdateClearCacheSetting.addEventListener('change', () => {
                    versionUpdateCachePreferenceSavePromise = saveVersionUpdateCachePreference(versionUpdateClearCacheSetting.checked);
                });
            }
            get('version-update-reload')?.addEventListener('click', async () => {
                await versionUpdateCachePreferenceSavePromise.catch(() => {});
                const clearCacheEnabled = !!get('version-update-clear-cache')?.checked;
                if (clearCacheEnabled) {
                    await clearSiteCacheAndReload(get('version-update-reload'), { scanFirst: true });
                } else {
                    location.reload();
                }
            });
            startConnectionMonitor();
            window.addEventListener('online', probeServerConnection);
            window.addEventListener('offline', () => {
                connectionStatus = 'offline';
                setConnectionBanner('offline');
                refreshConnectionMonitorTimer();
            });
            window.addEventListener('pagehide', stopConnectionMonitor);
            applyCacheMode(useSwCache);
            if (window.__turnstileApiLoaded && window.initTurnstileWidget) window.initTurnstileWidget();
            if (botConfig && botConfig.globalEnabled && botConfig.accountEnabled && !isAdminUser) {
                try { botTelemetry.start(); } catch (e) { console.error(e); }
            } else {
                const container = get('turnstile-container');
                if (container) container.classList.add('hidden');
            }
            const formatSessionTime = (val) => {
                if (!val) return '不明';
                const d = new Date(val);
                if (Number.isNaN(d.getTime())) return val;
                return d.toLocaleString();
            };
            const renderPasskeyList = (items) => {
                const list = Array.isArray(items) ? items : [];
                const wrap = get('passkey-list');
                const countEl = get('passkey-count');
                if (countEl) countEl.innerText = String(list.length);
                if (!wrap) return;
                if (!list.length) {
                    wrap.innerHTML = '<div class="text-[11px] text-gray-500">登録済みのパスキーはありません。</div>';
                    return;
                }
                wrap.innerHTML = '';
                list.forEach((item, idx) => {
                    const credId = item && item.id ? String(item.id) : '';
                    const row = document.createElement('div');
                    row.className = 'bg-gray-800/60 border border-gray-700 rounded p-2 flex items-center justify-between gap-2';
                    const left = document.createElement('div');
                    left.className = 'min-w-0';
                    const nameEl = document.createElement('div');
                    nameEl.className = 'text-xs text-gray-200 truncate';
                    nameEl.innerText = (item && item.name) ? String(item.name) : `Security Key ${idx + 1}`;
                    const metaEl = document.createElement('div');
                    metaEl.className = 'text-[10px] text-gray-500 mt-1';
                    metaEl.innerText = item && item.created_at ? `登録日時: ${formatSessionTime(item.created_at)}` : '登録日時: 不明';
                    left.appendChild(nameEl);
                    left.appendChild(metaEl);
                    row.appendChild(left);
                    const btn = document.createElement('button');
                    btn.type = 'button';
                    btn.className = 'bg-red-700 hover:bg-red-600 text-white px-2 py-1 rounded text-[10px] font-bold btn-hover shrink-0';
                    btn.innerText = '削除';
                    btn.disabled = !credId;
                    if (credId) {
                        btn.onclick = () => window.removeWebAuthnCredential(credId);
                    }
                    row.appendChild(btn);
                    wrap.appendChild(row);
                });
            };
            const renderSessions = (sessions) => {
                const list = get('session-list');
                if (!list) return;
                if (!sessions || !sessions.length) {
                    list.innerHTML = '<div class="text-xs text-gray-500">アクティブなセッションはありません。</div>';
                    return;
                }
                list.innerHTML = sessions.map((s) => {
                    const currentBadge = s.is_current ? '<span class="text-[10px] bg-blue-600 text-white px-1.5 py-0.5 rounded">現在</span>' : '';
                    const revokedBadge = s.is_revoked ? '<span class="text-[10px] bg-gray-700 text-gray-300 px-1.5 py-0.5 rounded">失効</span>' : '';
                    const actionBtn = (!s.is_current && !s.is_revoked) ? `<button data-session-id="${escapeHtml(s.id)}" class="session-revoke-btn bg-gray-700 hover:bg-gray-600 text-white px-3 py-1 rounded text-[11px] font-bold btn-hover">ログアウト</button>` : '';
                    const ua = (s.user_agent || 'Unknown').slice(0, 120);
                    const ip = s.ip_address || 'Unknown';
                    return `<div class="ui-enter-item bg-gray-800/60 border border-gray-700 rounded p-3 flex items-center justify-between gap-3"><div class="min-w-0"><div class="flex items-center gap-2 mb-1">${currentBadge}${revokedBadge}<div class="text-xs text-gray-200">${escapeHtml(ip)}</div></div><div class="text-[11px] text-gray-400 truncate">${escapeHtml(ua)}</div><div class="text-[10px] text-gray-500 mt-1">最終アクセス: ${escapeHtml(formatSessionTime(s.last_seen_at))} / 作成: ${escapeHtml(formatSessionTime(s.created_at))}</div></div>${actionBtn}</div>`;
                }).join('');
                list.querySelectorAll('.session-revoke-btn').forEach((btn) => {
                    btn.onclick = async () => {
                        const id = btn.getAttribute('data-session-id');
                        if (!id) return;
                        if (!confirm('このセッションをログアウトしますか？')) return;
                        const res = await apiFetch('/api/sessions/revoke', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({id})});
                        let data = {};
                        try { data = await res.json(); } catch (e) {}
                        if (res.ok) {
                            if (data.logged_out) {
                                location.href = '/login';
                                return;
                            }
                            await loadSessions();
                        } else {
                            showToast((data && data.error) || 'ログアウトに失敗しました', 'error', true);
                        }
                    };
                });
            };
            const loadSessions = async () => {
                const list = get('session-list');
                if (list) list.innerHTML = '<div class="text-xs text-gray-500">読み込み中...</div>';
                const res = await apiFetch('/api/sessions');
                let data = {};
                try { data = await res.json(); } catch (e) {}
                if (!res.ok) {
                    if (data && data.error === 'session_revoked') {
                        location.href = '/login';
                        return;
                    }
                    if (list) list.innerHTML = '<div class="text-xs text-red-400">セッションの取得に失敗しました。</div>';
                    return;
                }
                const sessions = (data.sessions || []).filter(s => !s.is_revoked);
                renderSessions(sessions);
            };
            const bindSessionButtons = () => {
                const refreshBtn = get('session-refresh-btn');
                if (refreshBtn) refreshBtn.onclick = () => loadSessions();
                const revokeOthersBtn = get('session-revoke-others-btn');
                if (revokeOthersBtn) revokeOthersBtn.onclick = async () => {
                    if (!confirm('現在の端末以外をログアウトしますか？')) return;
                    const res = await apiFetch('/api/sessions/revoke_others', {method:'POST'});
                    if (res.ok) {
                        await loadSessions();
                    } else {
                        showToast('操作に失敗しました', 'error', true);
                    }
                };
                const revokeAllBtn = get('session-revoke-all-btn');
                if (revokeAllBtn) revokeAllBtn.onclick = async () => {
                    if (!confirm('全セッションを強制ログアウトします。よろしいですか？')) return;
                    const res = await apiFetch('/api/sessions/revoke_all', {method:'POST'});
                    if (res.ok) {
                        location.href = '/login';
                    } else {
                        showToast('操作に失敗しました', 'error', true);
                    }
                };
            };
            apiFetch(CHAT_CONFIG.urls.handleSettingsQuery).then(r => r.json()).then(d => {
                cacheUserSettings(d);
                if (d) {
                    currentVisionModel = d.default_vision_model || 'gemini-3-flash-preview';
                }
                applyChatDefaults(d);
                if (d && d.theme_color) {
                    applyThemeColor(d.theme_color, true);
                }
                if (d && Object.prototype.hasOwnProperty.call(d, 'compact_prompt_mode')) {
                    setCompactPromptMode(!!d.compact_prompt_mode);
                }
                if (get('set-client-debug-log')) {
                    syncClientDebugLogToggle(d.enable_client_debug_log === true, 'settings sync');
                }
                const sysChk = get('enable-sys-prompt');
                if (sysChk && d && d.system_prompt && String(d.system_prompt).trim()) {
                    if (!sysChk.disabled && !d.default_enable_system_prompt && !d.use_last_chat_settings) sysChk.checked = true;
                    toggleOptions();
                }
            }).catch(() => {});
            loadThreads(); loadGems();
 
            get('send-btn').onclick = () => { if (isStopMode) stopGeneration(); else sendMessage(); }; 
            get('new-chat-btn').onclick = () => startNewChat(); 
            get('upload-btn').onclick = () => openUploadModal(); 
            const vmBtn = get('vision-model-change-btn');
            if (vmBtn) vmBtn.onclick = () => _openVisionModelSelector();
            const onlyEl = get('compression-format-only');
            if (onlyEl) {
                onlyEl.onchange = () => {
                    const disabled = onlyEl.checked;
                    const sizeEl = get('compression-max-size');
                    const dimEl = get('compression-max-dim');
                    if (sizeEl) sizeEl.disabled = disabled;
                    if (dimEl) dimEl.disabled = disabled;
                    const sizeWrap = get('compression-size-wrap');
                    const dimWrap = get('compression-dim-wrap');
                    if (sizeWrap) sizeWrap.style.opacity = disabled ? '0.4' : '1';
                    if (dimWrap) dimWrap.style.opacity = disabled ? '0.4' : '1';
                };
            }
            const bindTemporaryChatToggle = () => {
                const temporaryChatChk = get('enable-temporary-chat');
                if (!temporaryChatChk || temporaryChatChk.dataset.bound === '1') return;
                temporaryChatChk.dataset.bound = '1';
                temporaryChatChk.checked = !!temporaryChatEnabled;
                temporaryChatChk.onchange = async () => {
                    const prev = temporaryChatEnabled;
                    const ok = await applyTemporaryChatSetting(temporaryChatChk.checked);
                    if (!ok) {
                        setTemporaryChatUiState(prev);
                        ensureTemporaryChatHeartbeat(false);
                    }
                };
            };
            bindTemporaryChatToggle();
            document.addEventListener('visibilitychange', () => {
                if (document.visibilityState === 'visible') {
                    ensureTemporaryChatHeartbeat(true);
                }
            });
            window.addEventListener('focus', () => {
                ensureTemporaryChatHeartbeat(true);
            });
            window.addEventListener('beforeunload', () => {
                stopTemporaryChatHeartbeat();
                stopCameraCaptureStream();
            });
            const storageRefreshBtn = get('storage-usage-refresh');
            if (storageRefreshBtn) storageRefreshBtn.onclick = () => loadStorageUsage();
            const siteCacheRefreshBtn = get('site-cache-usage-refresh');
            if (siteCacheRefreshBtn) siteCacheRefreshBtn.onclick = () => loadSiteCacheUsage();
            const clearSiteCacheBtn = get('clear-site-cache-btn');
            if (clearSiteCacheBtn) {
                clearSiteCacheBtn.onclick = async () => {
                    if (!confirm('サイトキャッシュを削除しますか？\nCookie は削除されません。')) return;
                    await clearSiteCacheAndReload(clearSiteCacheBtn);
                };
            }
            const encScanResult = get('enc-scan-result');
            const runEncScan = async (threadId = null) => {
                if (encScanResult) encScanResult.textContent = 'スキャン中...';
                let url = '/api/encryption_scan';
                if (threadId) url += `?thread_id=${encodeURIComponent(threadId)}`;
                try {
                    const res = await apiFetch(url, { cache: 'no-store' });
                    const data = await res.json();
                    if (!res.ok) {
                        if (encScanResult) encScanResult.textContent = data.error || '失敗しました';
                        return;
                    }
                    const total = data.total || 0;
                    const enc = data.encrypted || 0;
                    const unenc = data.unencrypted || 0;
                    let html = `Total: ${total} / Encrypted: ${enc} / Plain: ${unenc}`;
                    if (data.samples && data.samples.length) {
                        const items = data.samples.slice(0, 8).map(s => {
                            const t = s.timestamp ? new Date(s.timestamp).toLocaleString() : '';
                            return `#${s.id} (${s.role || ''}) ${t}`;
                        }).join(' / ');
                        html += `<div class="text-[10px] text-gray-400 mt-1">例: ${items}</div>`;
                    }
                    if (encScanResult) encScanResult.innerHTML = html;
                } catch (e) {
                    if (encScanResult) encScanResult.textContent = '失敗しました';
                }
            };
            const encScanAllBtn = get('enc-scan-all');
            if (encScanAllBtn) encScanAllBtn.onclick = () => runEncScan(null);
            const encScanThreadBtn = get('enc-scan-thread');
            if (encScanThreadBtn) encScanThreadBtn.onclick = () => currentThreadId ? runEncScan(currentThreadId) : showToast('スレッドがありません', 'error', true);
            get('file-input').onchange = (e) => { 
                const files = Array.from(e.target.files || []);
                // Clear after copying to avoid losing selections during async uploads.
                e.target.value = '';
                if (files.length) handleFiles(files); 
            }; 
            if (get('photo-input')) {
                get('photo-input').onchange = (e) => {
                    const files = Array.from(e.target.files || []);
                    e.target.value = '';
                    if (files.length) handleFiles(files);
                };
            }
            const renderBanAppeals = (items) => {
                const box = get('ban-appeal-list');
                if (!box) return;
                if (!items || !items.length) {
                    box.innerHTML = '<div class="text-[11px] text-gray-500">現在、申し立てはありません。</div>';
                    return;
                }
                box.innerHTML = items.map(a => {
                    const statusLabel = a.status || 'new';
                    const readTag = a.admin_read_at ? '<span class="text-[10px] text-gray-500 ml-2">既読</span>' : '<span class="text-[10px] text-yellow-300 ml-2">未読</span>';
                    const time = a.created_at ? new Date(a.created_at).toLocaleString() : '';
                    const repliedAt = a.replied_at ? new Date(a.replied_at).toLocaleString() : '';
                    const replyValue = a.admin_reply || '';
                    return `
                        <div class="border border-gray-700/70 rounded p-2 bg-gray-900/60" data-appeal-id="${a.id}">
                            <div class="flex items-center justify-between">
                                <div class="text-xs text-blue-200 font-bold">${escapeHtml(a.username || '')}${readTag}</div>
                                <div class="text-[10px] text-gray-500">${escapeHtml(time)}</div>
                            </div>
                            <div class="text-[11px] text-gray-400 mt-1">Status: ${escapeHtml(statusLabel)}</div>
                            <div class="text-xs text-gray-200 mt-2 whitespace-pre-wrap">${escapeHtml(a.message || '')}</div>
                            <div class="text-[10px] text-gray-500 mt-2">BAN理由: ${escapeHtml(a.ban_reason || 'N/A')}</div>
                            <div class="mt-3">
                                <label class="text-[10px] text-gray-400">管理者返信</label>
                                <textarea class="ban-appeal-reply w-full mt-1 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-[11px] text-gray-100" rows="3" placeholder="返信内容">${escapeHtml(replyValue)}</textarea>
                                ${replyValue ? `<div class=\"text-[10px] text-gray-500 mt-1\">返信日時: ${escapeHtml(repliedAt)}</div>` : ''}
                            </div>
                            <div class="mt-2 flex flex-wrap gap-2">
                                <button class="ban-appeal-mark text-[10px] px-2 py-1 bg-gray-700 hover:bg-gray-600 rounded" data-id="${a.id}">既読</button>
                                <button class="ban-appeal-status text-[10px] px-2 py-1 bg-blue-700 hover:bg-blue-600 rounded" data-id="${a.id}" data-status="in_review">対応中</button>
                                <button class="ban-appeal-status text-[10px] px-2 py-1 bg-green-700 hover:bg-green-600 rounded" data-id="${a.id}" data-status="resolved">完了</button>
                                <button class="ban-appeal-status text-[10px] px-2 py-1 bg-red-700 hover:bg-red-600 rounded" data-id="${a.id}" data-status="rejected">却下</button>
                                <button class="ban-appeal-reply-send text-[10px] px-2 py-1 bg-sky-700 hover:bg-sky-600 rounded" data-id="${a.id}">返信送信</button>
                                <button class="ban-appeal-block text-[10px] px-2 py-1 bg-rose-700 hover:bg-rose-600 rounded" data-id="${a.id}">申し立てブロック</button>
                            </div>
                        </div>
                    `;
                }).join('');
            };
            const refreshBanAppealSummary = async (notify = false) => {
                if (!isAdminUser) return;
                const countBox = get('ban-appeal-count');
                if (!countBox) return;
                try {
                    const res = await apiFetch('/api/ban/appeals/summary', { cache: 'no-store' });
                    if (!res.ok) return;
                    const data = await res.json();
                    const count = data.unread_count || 0;
                    countBox.textContent = String(count);
                    if (notify && count > 0) {
                        showToast(`BAN異議申し立てが${count}件あります。`, 'success');
                    }
                } catch (e) {}
            };
            const loadBanAppeals = async () => {
                if (!isAdminUser) return;
                const listBox = get('ban-appeal-list');
                if (!listBox) return;
                listBox.innerHTML = '<div class="text-[11px] text-gray-500">読み込み中...</div>';
                try {
                    const res = await apiFetch('/api/ban/appeals?limit=80', { cache: 'no-store' });
                    if (!res.ok) return;
                    const data = await res.json();
                    renderBanAppeals(data.items || []);
                    await refreshBanAppealSummary(false);
                } catch (e) {}
            };
            const markBanAppealsRead = async (ids = null) => {
                if (!isAdminUser) return;
                const payload = ids ? { ids } : { all: true };
                try {
                    const res = await apiFetch('/api/ban/appeals/mark_read', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload) });
                    if (res.ok) {
                        await loadBanAppeals();
                    }
                } catch (e) {}
            };
            const updateBanAppealStatus = async (payload) => {
                if (!isAdminUser) return;
                try {
                    const res = await apiFetch('/api/ban/appeals/update', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload) });
                    if (res.ok) {
                        await loadBanAppeals();
                    }
                } catch (e) {}
            };
            const ensureTemporaryChatSettingsCard = () => {
                const tab = get('tab-general');
                if (!tab || get('temp-chat-settings-card')) return;
                const card = document.createElement('div');
                card.id = 'temp-chat-settings-card';
                card.className = 'bg-gray-900 p-4 rounded border border-gray-700';
                card.innerHTML = `
                    <h3 class="text-sm font-bold text-amber-300 mb-2">一時チャット</h3>
                    <div class="space-y-3 text-xs text-gray-300">
                        <label class="text-xs text-gray-500 block">切断タイムアウト（秒）</label>
                        <input id="set-temp-chat-timeout-seconds" type="number" min="${TEMP_CHAT_TIMEOUT_MIN_SECONDS}" max="${TEMP_CHAT_TIMEOUT_MAX_SECONDS}" step="1" class="w-28 bg-gray-800 border border-gray-600 rounded px-2 py-1 text-xs text-white">
                        <div class="text-[10px] text-gray-500">一時チャットでページの表示/接続が途切れた状態がこの秒数を超えると、自動削除されます。</div>
                    </div>
                `;
                tab.appendChild(card);
            };
            const ensureLlmTranscribePromptSettingsUi = () => {
                const sttModelEl = get('set-stt-model');
                if (!sttModelEl || get('set-llm-transcribe-prompt')) return;
                const host = sttModelEl.closest('.space-y-2');
                if (!host) return;
                const wrap = document.createElement('div');
                wrap.className = 'pt-2 border-t border-gray-700/60';
                wrap.innerHTML = `
                    <label class="text-xs text-gray-500 block">LLM文字起こしプロンプト（LLM方式）</label>
                    <textarea id="set-llm-transcribe-prompt" class="w-full h-24 bg-gray-800 border border-gray-600 rounded px-2 py-2 text-xs text-white mt-1" placeholder=""></textarea>
                    <div class="flex items-center gap-2 mt-2">
                        <button type="button" id="reset-llm-transcribe-prompt" class="bg-gray-700 hover:bg-gray-600 text-white px-2 py-1 rounded text-[10px] font-bold btn-hover">既定に戻す</button>
                        <div class="text-[10px] text-gray-500">LLM方式のマイク文字起こし時のみ使用。空欄で保存すると既定文面を使います（無音時の安全ガードは別途自動付与）。</div>
                    </div>
                `;
                host.appendChild(wrap);
                const resetBtn = get('reset-llm-transcribe-prompt');
                if (resetBtn) {
                    resetBtn.onclick = () => {
                        const ta = get('set-llm-transcribe-prompt');
                        if (ta) ta.value = '';
                        showToast('LLM文字起こしプロンプトを既定値に戻しました（保存してください）', 'success');
                    };
                }
            };
            const AUTO_SYS_PROMPT_ITEMS = [
                { key: 'python', label: 'Python 実行案内' },
                { key: 'gemini_local_python', label: 'Gemini 音声/動画 + Python（ローカル実行）' },
                { key: 'grok_search', label: 'Search補助（Grok）' },
                { key: 'openai_search', label: 'Search補助（OpenAI/xAI Responses）' },
                { key: 'marker', label: 'Marker編集時' },
                { key: 'attachment_names', label: '添付ファイル名（LLM入力時）', hint: '利用可能変数: {{attachment_names}} / {{attachment_count}}' },
                { key: 'mathjax', label: 'MathJax（LaTeX数式）' },
                { key: 'image_analysis', label: '画像解析（Vision Model指示文）' }
            ];
            window.buildAutoSystemPromptRows = (prefix, compact = false) => {
                const textClass = compact
                    ? 'w-full h-14 bg-gray-950 border border-gray-700 rounded p-2 text-[11px] text-gray-200'
                    : 'w-full h-20 bg-gray-950 border border-gray-700 rounded p-2 text-xs text-gray-200';
                return AUTO_SYS_PROMPT_ITEMS.map((item) => `
                    <div class="rounded border border-gray-700 p-2 bg-gray-950/40">
                        <div class="flex items-center justify-between mb-1">
                            <div class="text-[11px] text-gray-300">${item.label}</div>
                            <label class="flex items-center gap-1 text-[10px] text-gray-500">
                                <input type="checkbox" id="${prefix}-auto-sys-${item.key}-enabled" class="accent-yellow-500 w-3 h-3">
                                <span>適用</span>
                            </label>
                        </div>
                        <textarea id="${prefix}-auto-sys-${item.key}-text" class="${textClass}" placeholder="自動注入文言"></textarea>
                        ${item.hint ? `<div class="text-[10px] text-gray-500 mt-1">${item.hint}</div>` : ''}
                    </div>
                `).join('');
            };
            window.applyAutoSystemPromptConfigToForm = (prefix, cfg = {}) => {
                AUTO_SYS_PROMPT_ITEMS.forEach((item) => {
                    const row = cfg && typeof cfg === 'object' ? (cfg[item.key] || {}) : {};
                    const enabledEl = get(`${prefix}-auto-sys-${item.key}-enabled`);
                    const textEl = get(`${prefix}-auto-sys-${item.key}-text`);
                    if (enabledEl) enabledEl.checked = row.enabled !== false;
                    if (textEl) {
                        textEl.value = row.text || '';
                        textEl.placeholder = row.default_text || '自動注入文言';
                    }
                });
            };
            const resetAutoSystemPromptConfigToCodeDefaults = (prefix, applyToggleId = null) => {
                if (applyToggleId) {
                    const applyToggleEl = get(applyToggleId);
                    if (applyToggleEl) applyToggleEl.checked = true;
                }
                AUTO_SYS_PROMPT_ITEMS.forEach((item) => {
                    const enabledEl = get(`${prefix}-auto-sys-${item.key}-enabled`);
                    const textEl = get(`${prefix}-auto-sys-${item.key}-text`);
                    if (enabledEl) enabledEl.checked = true;
                    if (textEl) {
                        const defaultText = textEl.placeholder || '';
                        textEl.value = defaultText;
                    }
                });
            };
            const collectAutoSystemPromptConfigFromForm = (prefix) => {
                const cfg = {};
                AUTO_SYS_PROMPT_ITEMS.forEach((item) => {
                    const enabledEl = get(`${prefix}-auto-sys-${item.key}-enabled`);
                    const textEl = get(`${prefix}-auto-sys-${item.key}-text`);
                    cfg[item.key] = {
                        enabled: enabledEl ? enabledEl.checked : true,
                        text: textEl ? textEl.value : ''
                    };
                });
                return cfg;
            };
            window.ensureAutoSystemPromptSettingsCard = () => {
                const promptToggle = get('set-global-sys-prompt-enabled');
                const wrapHost = promptToggle ? promptToggle.closest('.space-y-4') : null;
                if (!wrapHost || get('auto-sys-prompt-settings')) return;
                const box = document.createElement('div');
                box.id = 'auto-sys-prompt-settings';
                box.className = 'border-t border-gray-700 pt-3';
                box.innerHTML = `
                    <div class="flex items-center justify-between mb-2">
                        <label class="text-xs text-gray-500 block">自動注入システムプロンプト（ユーザー単位）</label>
                        <div class="flex items-center gap-2">
                            <button type="button" id="reset-set-auto-sys-prompt-defaults" class="bg-gray-700 hover:bg-gray-600 text-white px-2 py-1 rounded text-[10px] font-bold btn-hover">既定に戻す</button>
                            <label class="flex items-center gap-1 text-[10px] text-gray-400">
                                <input type="checkbox" id="set-apply-auto-sys-prompt-notices" class="accent-yellow-500 w-3 h-3">
                                <span>全体適用</span>
                            </label>
                        </div>
                    </div>
                    <div id="set-auto-sys-prompt-items" class="space-y-2">${window.buildAutoSystemPromptRows('set', false)}</div>
                    <div class="text-[10px] text-gray-500 mt-2">各文面はユーザー単位で編集されます。空欄で保存すると既定文面に戻ります。</div>
                `;
                wrapHost.appendChild(box);
            };
            window.ensureThreadAutoSystemPromptCard = () => {
                const threadPrompt = get('thread-global-sys-prompt');
                const wrapHost = threadPrompt ? threadPrompt.closest('.space-y-3') : null;
                if (!wrapHost || get('thread-auto-sys-prompt-settings')) return;
                const box = document.createElement('div');
                box.id = 'thread-auto-sys-prompt-settings';
                box.className = 'border-t border-gray-700 pt-3';
                box.innerHTML = `
                    <div class="flex items-center justify-between mb-2">
                        <div class="text-xs text-gray-400">自動注入システムプロンプト（ユーザー単位）</div>
                        <div class="flex items-center gap-2">
                            <button type="button" id="reset-thread-auto-sys-prompt-defaults" class="bg-gray-700 hover:bg-gray-600 text-white px-2 py-1 rounded text-[10px] font-bold btn-hover">既定に戻す</button>
                            <label class="flex items-center gap-1 text-[10px] text-gray-500">
                                <input type="checkbox" id="thread-apply-auto-sys-prompt-notices" class="accent-yellow-500 w-3 h-3">
                                <span>全体適用</span>
                            </label>
                        </div>
                    </div>
                    <div id="thread-auto-sys-prompt-items" class="space-y-2">${window.buildAutoSystemPromptRows('thread', true)}</div>
                `;
                wrapHost.appendChild(box);
            };
            ensureTemporaryChatSettingsCard();
            ensureLlmTranscribePromptSettingsUi();
            bindTemporaryChatToggle();
            const populateDefaultModelOptions = () => {
                const sel = get('set-default-model');
                if (!sel) return;
                const current = sel.value;
                sel.innerHTML = '';
                MODELS.forEach(group => {
                    const optgroup = document.createElement('optgroup');
                    optgroup.label = group.category;
                    (group.items || []).forEach(item => {
                        const opt = document.createElement('option');
                        opt.value = item.id;
                        opt.textContent = item.name;
                        optgroup.appendChild(opt);
                    });
                    sel.appendChild(optgroup);
                });
                if (current) sel.value = current;
            };
            const populateDefaultVisionModelOptions = () => {
                const sel = get('set-default-vision-model');
                if (!sel) return;
                const current = sel.value;
                sel.innerHTML = '';
                MODELS.forEach(group => {
                    const visionItems = (group.items || []).filter(m => {
                        const id = (m.id || '').toLowerCase();
                        return id.startsWith('gemini-') || id.startsWith('gpt-4o') || id.startsWith('claude-') || id.startsWith('grok-3');
                    });
                    if (visionItems.length === 0) return;
                    const optgroup = document.createElement('optgroup');
                    optgroup.label = group.category;
                    visionItems.forEach(item => {
                        const opt = document.createElement('option');
                        opt.value = item.id;
                        opt.textContent = item.name + ' ★';
                        optgroup.appendChild(opt);
                    });
                    sel.appendChild(optgroup);
                });
                if (current) sel.value = current;
            };

            window.openSettingsModal = () => {
                const ss = get('settings-search');
                if (ss) { ss.value = ''; }
                filterSettings();
                populateDefaultModelOptions();
                populateDefaultVisionModelOptions();
                showModal('settings-modal');
                loadStorageUsage();
                loadSiteCacheUsage();
                ensureLlmTranscribePromptSettingsUi();
                if (location.pathname !== '/settings') {
                    history.pushState({ modal: 'settings', from: location.pathname }, '', '/settings');
                }
                    refreshBanAppealSummary(true);
                    loadBanAppeals();
                    apiFetch(CHAT_CONFIG.urls.handleSettingsQuery).then(r=>r.json()).then(d=>{ 
                        cacheUserSettings(d);
                        const globalPreview = get('app-global-sys-prompt-preview');
                        if (globalPreview) {
                            globalPreview.value = d.global_system_prompt_effective || '';
                        }
                        const globalPreviewStatus = get('app-global-sys-prompt-preview-status');
                        if (globalPreviewStatus) {
                            if (d.global_system_prompt_enabled === false) {
                                globalPreviewStatus.textContent = '現在は無効化されています。';
                            } else if (d.global_system_prompt_uses_time_fallback) {
                                globalPreviewStatus.textContent = '管理者設定が空欄のため、時刻の既定プロンプトが適用されています。';
                            } else {
                                globalPreviewStatus.textContent = '管理者が設定した全体システムプロンプトが適用されています。';
                            }
                        }
                        if(get('sys-prompt-text')) get('sys-prompt-text').value = d.system_prompt || ''; 
                        if(get('set-global-sys-prompt-enabled')) get('set-global-sys-prompt-enabled').checked = d.system_prompt_enabled !== false;

                        window.ensureAutoSystemPromptSettingsCard();
                        if(get('set-apply-global-sys-prompt')) get('set-apply-global-sys-prompt').checked = d.apply_global_system_prompt !== false;
                        if(get('set-apply-auto-sys-prompt-notices')) get('set-apply-auto-sys-prompt-notices').checked = d.apply_auto_system_prompt_notices !== false;
                        window.applyAutoSystemPromptConfigToForm('set', d.auto_system_prompt_notices_config || {});
                        
                        if(get('set-latency-metrics')) get('set-latency-metrics').checked = d.enable_latency_metrics === true;
                        if(get('set-client-debug-log')) syncClientDebugLogToggle(d.enable_client_debug_log === true, 'settings modal sync');
                        if(get('set-openai')) get('set-openai').value = d.openai_key || ''; 
                        if(get('set-gemini')) get('set-gemini').value = d.gemini_key || ''; 
                        if(get('set-deepseek')) get('set-deepseek').value = d.deepseek_key || '';
                        if(get('set-anthropic')) get('set-anthropic').value = d.anthropic_key || '';
                        if(get('set-gemini-backend')) get('set-gemini-backend').value = normalizeGeminiBackend(d.gemini_backend || 'gemini_api');
                        if(get('set-gemini-vertex-project')) get('set-gemini-vertex-project').value = d.gemini_vertex_project || '';
                        if(get('set-gemini-vertex-location')) get('set-gemini-vertex-location').value = d.gemini_vertex_location || 'global';
                        ensureGeminiVertexCredentialsField();
                        if(get('set-gemini-vertex-credentials-json')) get('set-gemini-vertex-credentials-json').value = d.gemini_vertex_credentials_json || '';
                        syncGeminiBackendUi();
                        if(get('set-admin-api-key-mode')) get('set-admin-api-key-mode').value = normalizeAdminApiKeyMode(d.admin_api_key_mode || 'env_fallback');
                        syncAdminApiKeyModeUi();
                        if(get('set-xai')) get('set-xai').value = d.xai_key || ''; 
                    if(get('set-google-key')) get('set-google-key').value = d.google_key || ''; 
                    if(get('set-google-project')) get('set-google-project').value = d.google_project || ''; 
                    modelApiKeyMap = normalizeModelApiKeyMap(d.model_api_keys || {});
                    syncModelApiKeyModelOptions();
                    renderModelApiKeyList();
                    setModelApiKeyPanelOpen(false);
                    if(get('set-mic-transcribe-mode')) get('set-mic-transcribe-mode').value = d.mic_transcribe_mode || 'stt_api';
                    if(get('set-stt-model')) get('set-stt-model').value = d.stt_model || 'gpt-4o-mini-transcribe';
                    if(get('set-llm-transcribe-prompt')) {
                        get('set-llm-transcribe-prompt').value = d.llm_transcribe_prompt || '';
                        get('set-llm-transcribe-prompt').placeholder = d.llm_transcribe_prompt_default || '';
                    }
                    syncRichPastePromptPreferencesUi(d);
                    updateGoogleLinkUI(d);
                    if(get('set-enter-to-send')) get('set-enter-to-send').checked = !!d.enter_to_send;
                    if(get('set-compact-prompt-mode')) get('set-compact-prompt-mode').checked = !!d.compact_prompt_mode;
                        if(get('set-use-sw-cache')) get('set-use-sw-cache').checked = !!d.use_sw_cache;
                        if(get('set-clear-cache-on-version-update')) get('set-clear-cache-on-version-update').checked = !!d.clear_cache_on_version_update;
                    if(get('set-liquid-glass')) get('set-liquid-glass').checked = !!d.liquid_glass_enabled;
                    if(get('set-auto-search-links')) get('set-auto-search-links').checked = d.auto_search_on_links !== false;
                    if(get('set-use-last-settings')) get('set-use-last-settings').checked = !!d.use_last_chat_settings;
                    if(get('set-default-model')) get('set-default-model').value = d.default_model || 'gemini-3.1-flash-lite-preview';
                    if(get('set-default-vision-model')) get('set-default-vision-model').value = d.default_vision_model || 'gemini-3-flash-preview';
                    applyTemporaryChatTimeoutSeconds(d.temp_chat_timeout_seconds);
                    if(get('set-default-search')) get('set-default-search').checked = !!d.default_enable_search;
                    if(get('set-default-url-context')) get('set-default-url-context').checked = !!d.default_enable_url_context;
                    if(get('set-default-maps')) get('set-default-maps').checked = !!d.default_enable_maps;
                    if(get('set-default-python')) get('set-default-python').checked = !!d.default_enable_python;
                    if(get('set-default-thinking')) get('set-default-thinking').checked = !!d.default_enable_thinking;
                    if(get('set-default-sys-prompt')) get('set-default-sys-prompt').checked = !!d.default_enable_system_prompt;
                    if(get('set-default-thinking-level')) get('set-default-thinking-level').value = d.default_thinking_level || 'high';
                    if(get('set-default-thinking-budget')) get('set-default-thinking-budget').value = d.default_thinking_budget || 4096;
                    if(get('set-default-reasoning-effort')) get('set-default-reasoning-effort').value = d.default_reasoning_effort || 'medium';
                    if(get('set-default-safety')) get('set-default-safety').value = d.default_safety_setting || 'default';
                    get('set-e2ee').checked = d.enable_e2ee; 
                    if(get('set-bot-detect')) get('set-bot-detect').checked = d.bot_detection_enabled !== false;
                    if(get('set-bot-detect-global')) get('set-bot-detect-global').checked = d.bot_detection_global_enabled !== false;
                    const botStatus = get('bot-status');
                    if (botStatus) {
                        if (d.is_bot_banned) {
                            botStatus.textContent = `BAN中: ${d.bot_ban_reason || 'Bot detection'}`;
                            botStatus.classList.remove('hidden');
                            botStatus.classList.add('text-red-400');
                        } else {
                            botStatus.classList.add('hidden');
                        }
                    }
                    if (d && d.theme_color) {
                        applyThemeColor(d.theme_color, true);
                        syncThemeInputs(d.theme_color);
                    } else {
                        syncThemeInputs(localStorage.getItem(THEME_STORAGE_KEY) || INITIAL_THEME_COLOR || THEME_DEFAULT);
                    }
                    syncGeminiLocalPyDialogSetting();
                    syncCompressionSettingsUi();
                    if(get('set-username')) get('set-username').value = d.username; 
                    
                    // 2FA UI Update
                    const badge = get('2fa-badge');
                    const disBtn = get('disable-2fa-btn');
                    if(d.is_2fa_enabled) {
                        badge.innerText = "ENABLED";
                        badge.classList.replace('bg-gray-700', 'bg-green-600');
                        badge.classList.replace('text-gray-400', 'text-white');
                        disBtn.classList.remove('hidden');
                    } else {
                        badge.innerText = "DISABLED";
                        badge.classList.replace('bg-green-600', 'bg-gray-700');
                        badge.classList.replace('text-white', 'text-gray-400');
                        disBtn.classList.add('hidden');
                    }
                    if(get('set-skip-2fa-google')) get('set-skip-2fa-google').checked = !!d.skip_2fa_on_google_login;
                    if(get('set-default-2fa-method')) get('set-default-2fa-method').value = d.default_2fa_method || 'totp';

                    const pkOnly = get('set-passkey-only-login');
                    const pkNote = get('passkey-only-note');
                    const passkeys = Array.isArray(d.passkey_credentials) ? d.passkey_credentials : [];
                    renderPasskeyList(passkeys);
                    if (pkOnly) {
                        pkOnly.checked = !!d.passkey_only_login;
                        const hasKey = passkeys.length > 0 || !!d.has_webauthn;
                        pkOnly.disabled = !hasKey;
                        if (!hasKey) pkOnly.checked = false;
                        if (pkNote) {
                            if (hasKey) pkNote.classList.add('hidden');
                            else pkNote.classList.remove('hidden');
                        }
                    }

                    const migBox = get('mig-status-box');
                    const migText = get('mig-progress-text');
                    const migBar = get('mig-progress-bar');
                    const status = d.migration_status || 'idle';
                    if (status === 'processing') {
                        migBox.classList.remove('hidden');
                        const prog = (d.migration_progress || '').split('/');
                        if (prog.length === 2) {
                            const done = parseInt(prog[0] || '0', 10);
                            const total = parseInt(prog[1] || '0', 10);
                            if (migText) migText.innerText = `${done} / ${total}`;
                            if (migBar && total > 0) migBar.style.width = `${Math.min(100, Math.floor((done/total)*100))}%`;
                        }
                    } else {
                        migBox.classList.add('hidden');
                        if (migBar) migBar.style.width = '0%';
                        if (migText) migText.innerText = '';
                    }
                }); 
                loadFeedback();
                bindSessionButtons();
                loadSessions();
            };
            const closeSettingsModal = (skipHistory = false) => {
                hideModal('settings-modal');
                if (!skipHistory && location.pathname === '/settings') {
                    history.back();
                }
            };
            const bindThemeControls = () => {
                const colorInput = get('set-theme-color');
                const textInput = get('set-theme-color-text');
                const resetBtn = get('theme-reset-btn');
                const swatches = document.querySelectorAll('#theme-presets .theme-swatch');
                const applyFromValue = (value, persist = true) => {
                    const hex = normalizeHex(value);
                    if (!hex) return;
                    applyThemeColor(hex, persist);
                    syncThemeInputs(hex);
                };
                if (colorInput) {
                    colorInput.addEventListener('input', () => applyFromValue(colorInput.value, true));
                }
                if (textInput) {
                    textInput.addEventListener('change', () => {
                        const hex = normalizeHex(textInput.value);
                        if (!hex) {
                            syncThemeInputs(localStorage.getItem(THEME_STORAGE_KEY) || THEME_DEFAULT);
                            return;
                        }
                        applyFromValue(hex, true);
                    });
                    textInput.addEventListener('keydown', (e) => {
                        if (e.key === 'Enter') {
                            e.preventDefault();
                            textInput.blur();
                        }
                    });
                }
                if (resetBtn) resetBtn.onclick = () => applyFromValue(THEME_DEFAULT, true);
                swatches.forEach((btn) => {
                    btn.addEventListener('click', () => applyFromValue(btn.getAttribute('data-color'), true));
                });
            };
            const bindSystemPromptControls = () => {
                const resetGlobal = get('reset-global-sys-prompt');
                if (resetGlobal) {
                    resetGlobal.onclick = () => {
                        if (get('sys-prompt-text')) get('sys-prompt-text').value = '';
                        if (get('set-global-sys-prompt-enabled')) get('set-global-sys-prompt-enabled').checked = false;
                        showToast('ユーザーシステムプロンプトをリセットしました（保存してください）', 'success');
                    };
                }
                const resetAutoSet = get('reset-set-auto-sys-prompt-defaults');
                if (resetAutoSet) {
                    resetAutoSet.onclick = () => {
                        resetAutoSystemPromptConfigToCodeDefaults('set', 'set-apply-auto-sys-prompt-notices');
                        showToast('自動注入システムプロンプトを既定値に戻しました（保存してください）', 'success');
                    };
                }
                const resetAutoThread = get('reset-thread-auto-sys-prompt-defaults');
                if (resetAutoThread) {
                    resetAutoThread.onclick = () => {
                        resetAutoSystemPromptConfigToCodeDefaults('thread', 'thread-apply-auto-sys-prompt-notices');
                        showToast('自動注入システムプロンプトを既定値に戻しました（保存してください）', 'success');
                    };
                }
            };
            get('settings-btn').onclick = () => { 
                openSettingsModal();
            }; 
            get('close-settings-btn').onclick = () => closeSettingsModal(); 
            const settingsSearchInput = get('settings-search');
            if (settingsSearchInput) {
                settingsSearchInput.addEventListener('input', filterSettings);
                settingsSearchInput.addEventListener('keydown', (e) => {
                    if (e.key === 'Enter') {
                        const tab = get('tab-' + activeSettingsTab);
                        if (!tab) return;
                        const first = tab.querySelector(':scope > .settings-match');
                        if (first) first.scrollIntoView({ behavior: 'smooth', block: 'start' });
                    }
                });
            }
            const settingsSearchClear = get('settings-search-clear');
            if (settingsSearchClear) {
                settingsSearchClear.addEventListener('click', () => {
                    if (settingsSearchInput) {
                        settingsSearchInput.value = '';
                        filterSettings();
                        settingsSearchInput.focus();
                    }
                });
            }
            bindThemeControls();
            bindSystemPromptControls();
            bindModelApiKeySettingsControls();
            syncGeminiLocalPyDialogSetting();
            syncCompressionSettingsUi();
            const localPyDialogSetting = get('set-gemini-local-python-dialog');
            if (localPyDialogSetting) {
                localPyDialogSetting.onchange = () => setGeminiLocalPyDialogEnabled(localPyDialogSetting.checked);
            }
            const geminiBackendSetting = get('set-gemini-backend');
            if (geminiBackendSetting) {
                geminiBackendSetting.onchange = () => syncGeminiBackendUi();
            }
            const adminApiModeSetting = get('set-admin-api-key-mode');
            if (adminApiModeSetting) {
                adminApiModeSetting.onchange = () => syncAdminApiKeyModeUi();
            }
            const tempChatTimeoutSetting = get('set-temp-chat-timeout-seconds');
            if (tempChatTimeoutSetting) {
                tempChatTimeoutSetting.onchange = () => {
                    applyTemporaryChatTimeoutSeconds(tempChatTimeoutSetting.value);
                };
            }

            // Wire slash command cancel button (for pending command mode)
            const slashCancelBtn = get('slash-command-cancel-btn');
            if (slashCancelBtn) {
                slashCancelBtn.onclick = () => {
                    hidePendingSlashCommandIndicator();
                    const input = get('prompt-input');
                    if (input) input.focus();
                };
            }
            syncGeminiBackendUi();
            syncAdminApiKeyModeUi();
            get('save-settings-btn').onclick = async () => {
                const uEl = get('set-username');
                const pEl = get('set-password');
                const b = {
                    system_prompt: get('sys-prompt-text') ? get('sys-prompt-text').value : '',
                    system_prompt_enabled: get('set-global-sys-prompt-enabled') ? get('set-global-sys-prompt-enabled').checked : true,
                    apply_global_system_prompt: get('set-apply-global-sys-prompt') ? get('set-apply-global-sys-prompt').checked : true,
                    apply_auto_system_prompt_notices: get('set-apply-auto-sys-prompt-notices') ? get('set-apply-auto-sys-prompt-notices').checked : true,
                    auto_system_prompt_notices_config: collectAutoSystemPromptConfigFromForm('set'),
                    theme_color: normalizeHex(get('set-theme-color-text') ? get('set-theme-color-text').value : '') || THEME_DEFAULT,
                    mic_transcribe_mode: get('set-mic-transcribe-mode') ? get('set-mic-transcribe-mode').value : 'stt_api',
                    stt_model: get('set-stt-model') ? get('set-stt-model').value : null,
                    llm_transcribe_prompt: get('set-llm-transcribe-prompt') ? get('set-llm-transcribe-prompt').value : '',
                    enter_to_send: get('set-enter-to-send') ? get('set-enter-to-send').checked : false,
                    compact_prompt_mode: get('set-compact-prompt-mode') ? get('set-compact-prompt-mode').checked : false,
                    use_sw_cache: get('set-use-sw-cache') ? get('set-use-sw-cache').checked : false,
                    clear_cache_on_version_update: get('set-clear-cache-on-version-update') ? get('set-clear-cache-on-version-update').checked : false,
                    liquid_glass_enabled: get('set-liquid-glass') ? get('set-liquid-glass').checked : false,
                    auto_search_on_links: get('set-auto-search-links') ? get('set-auto-search-links').checked : true,
                    use_last_chat_settings: get('set-use-last-settings') ? get('set-use-last-settings').checked : false,
                    default_model: get('set-default-model') ? get('set-default-model').value : null,
                    default_vision_model: get('set-default-vision-model') ? get('set-default-vision-model').value : null,
                    temp_chat_timeout_seconds: normalizeTemporaryChatTimeoutSeconds(
                        get('set-temp-chat-timeout-seconds') ? get('set-temp-chat-timeout-seconds').value : temporaryChatTimeoutSeconds
                    ),
                    default_enable_search: get('set-default-search') ? get('set-default-search').checked : false,
                    default_enable_url_context: get('set-default-url-context') ? get('set-default-url-context').checked : false,
                    default_enable_maps: get('set-default-maps') ? get('set-default-maps').checked : false,
                    default_enable_python: get('set-default-python') ? get('set-default-python').checked : false,
                    default_enable_thinking: get('set-default-thinking') ? get('set-default-thinking').checked : false,
                    default_thinking_level: get('set-default-thinking-level') ? get('set-default-thinking-level').value : null,
                    default_thinking_budget: get('set-default-thinking-budget') ? get('set-default-thinking-budget').value : null,
                    default_reasoning_effort: get('set-default-reasoning-effort') ? get('set-default-reasoning-effort').value : null,
                    default_enable_system_prompt: get('set-default-sys-prompt') ? get('set-default-sys-prompt').checked : false,
                    default_safety_setting: get('set-default-safety') ? get('set-default-safety').value : null,
                    enable_latency_metrics: get('set-latency-metrics') ? get('set-latency-metrics').checked : false,
                    enable_client_debug_log: get('set-client-debug-log') ? get('set-client-debug-log').checked : false,
                    enable_e2ee: get('set-e2ee') ? get('set-e2ee').checked : false,
                    passkey_only_login: get('set-passkey-only-login') ? get('set-passkey-only-login').checked : false,
                    skip_2fa_on_google_login: get('set-skip-2fa-google') ? get('set-skip-2fa-google').checked : false,
                    default_2fa_method: get('set-default-2fa-method') ? get('set-default-2fa-method').value : 'totp',
                    new_username: uEl ? uEl.value : null,
                    new_password: pEl ? pEl.value : null
                };
                if (get('set-openai')) b.openai_key = get('set-openai').value;
                if (get('set-gemini')) b.gemini_key = get('set-gemini').value;
                if (get('set-deepseek')) b.deepseek_key = get('set-deepseek').value;
                if (get('set-anthropic')) b.anthropic_key = get('set-anthropic').value;
                b.model_api_keys = normalizeModelApiKeyMap(modelApiKeyMap);
                if (get('set-gemini-backend')) b.gemini_backend = normalizeGeminiBackend(get('set-gemini-backend').value);
                if (get('set-gemini-vertex-project')) b.gemini_vertex_project = get('set-gemini-vertex-project').value;
                if (get('set-gemini-vertex-location')) b.gemini_vertex_location = get('set-gemini-vertex-location').value;
                if (get('set-gemini-vertex-credentials-json')) b.gemini_vertex_credentials_json = get('set-gemini-vertex-credentials-json').value;
                if (get('set-xai')) b.xai_key = get('set-xai').value;
                if (get('set-google-key')) b.google_key = get('set-google-key').value;
                if (get('set-google-project')) b.google_project = get('set-google-project').value;
                if (get('set-admin-api-key-mode')) b.admin_api_key_mode = normalizeAdminApiKeyMode(get('set-admin-api-key-mode').value);
                if (get('set-bot-detect')) b.bot_detection_enabled = get('set-bot-detect').checked;
                if (get('set-bot-detect-global')) b.bot_detection_global_enabled = get('set-bot-detect-global').checked;
                const res = await apiFetch(CHAT_CONFIG.urls.handleSettings, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(b)});
                if (res.ok) { 
                    closeSettingsModal();
                    
                    const oldUsername = currentUsername;
                    const oldE2EE = CHAT_CONFIG.enableE2EE;
                    
                    // Update client-side variables
                    enterToSend = b.enter_to_send;
                    autoSearchOnLinks = b.auto_search_on_links;
                    useSwCache = b.use_sw_cache;
                    if (window.CHAT_CONFIG) {
                        window.CHAT_CONFIG.clearCacheOnVersionUpdate = !!b.clear_cache_on_version_update;
                    }
                    compactPromptMode = b.compact_prompt_mode;
                    temporaryChatTimeoutSeconds = b.temp_chat_timeout_seconds;
                    
                    // Apply theme color
                    applyThemeColor(b.theme_color, true);
                    syncThemeInputs(b.theme_color);
                    applyLiquidGlassMode(b.liquid_glass_enabled);
                    
                    // Update UI components
                    setCompactPromptMode(compactPromptMode);
                    
                    showToast("設定を保存しました", "success");
                    syncClientDebugLogToggle(b.enable_client_debug_log, 'settings saved');

                    // Critical changes that still might benefit from a reload for full consistency
                    if (b.new_username && b.new_username !== oldUsername) {
                        setTimeout(() => location.reload(), 1000);
                    } else if (b.new_password) {
                        showToast("パスワードを変更しました。次回ログイン時から有効です。", "info");
                    }
                }
                else {
                    let d = {};
                    try { d = await res.json(); } catch (e) {}
                    showToast(d.error || "設定の保存に失敗しました", "error", true);
                }
            };
            get('disable-2fa-btn').onclick = async () => { 
                if(confirm("Disable 2FA?")) { 
                    const res = await apiFetch(CHAT_CONFIG.urls.handleSettings, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({disable_2fa: true})}); 
                    if (res.ok) {
                        showToast("2FAを無効化しました", "success");
                        get('disable-2fa-btn').classList.add('hidden');
                        const badge = get('2fa-badge');
                        if (badge) {
                            badge.innerText = 'DISABLED';
                            badge.className = 'px-2 py-0.5 rounded text-xs font-bold bg-gray-700 text-gray-400';
                        }
                    } else {
                        showToast("2FAの無効化に失敗しました", "error", true);
                    }
                } 
            };
            if (get('bot-unban-btn')) {
                get('bot-unban-btn').onclick = async () => {
                    const u = get('bot-unban-username');
                    const username = u ? u.value.trim() : '';
                    if (!username) { showToast('ユーザー名を入力してください', 'error', true); return; }
                    if (!confirm(`ユーザー ${username} のBANを解除しますか？`)) return;
                    const res = await apiFetch('/api/bot/unban', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({username, mode:'single'})});
                    const data = await res.json();
                    const box = get('bot-unban-result');
                    if (res.ok && data && data.status === 'ok') {
                        if (box) { box.textContent = `${username} のBANを単独解除しました`; box.classList.remove('hidden'); }
                        if (u) u.value = '';
                    } else {
                        const msg = data && data.error ? data.error : '解除に失敗しました';
                        showToast(msg, 'error', true);
                    }
                };
            }
            if (get('bot-unban-linked-btn')) {
                get('bot-unban-linked-btn').onclick = async () => {
                    const u = get('bot-unban-username');
                    const username = u ? u.value.trim() : '';
                    if (!username) { showToast('ユーザー名を入力してください', 'error', true); return; }
                    if (!confirm(`ユーザー ${username} の連鎖BANを解除しますか？`)) return;
                    const res = await apiFetch('/api/bot/unban', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({username, mode:'linked'})});
                    const data = await res.json();
                    const box = get('bot-unban-result');
                    if (res.ok && data && data.status === 'ok') {
                        if (box) { box.textContent = `${username} の連鎖BANを解除しました`; box.classList.remove('hidden'); }
                        if (u) u.value = '';
                    } else {
                        const msg = data && data.error ? data.error : '解除に失敗しました';
                        showToast(msg, 'error', true);
                    }
                };
            }
            if (get('bot-speed-test-btn')) {
                get('bot-speed-test-btn').onclick = async () => {
                    const btn = get('bot-speed-test-btn');
                    const box = get('bot-speed-test-result');
                    if (btn) btn.disabled = true;
                    if (btn) btn.classList.add('opacity-60', 'cursor-not-allowed');
                    if (box) {
                        box.classList.remove('hidden');
                        box.textContent = '実行中...';
                    }
                    try {
                        const setBox = (text) => { if (box) box.textContent = text; };
                        const cacheBust = () => `${Date.now()}_${Math.random().toString(36).slice(2)}`;
                        const toMbps = (bytes, ms) => {
                            if (!bytes || !ms || ms <= 0) return 0;
                            return (bytes * 8) / (ms / 1000) / 1000 / 1000;
                        };
                        const fmtMs = (v) => Number.isFinite(v) ? `${v.toFixed(0)} ms` : '-';
                        const fmtMbps = (v) => Number.isFinite(v) ? `${v.toFixed(v >= 100 ? 0 : 1)} Mbps` : '-';
                        const parseErr = async (res, fallback) => {
                            const data = await res.json().catch(() => ({}));
                            return (data && data.error) ? data.error : fallback;
                        };

                        const pingSamples = [];
                        setBox('測定中... ping');
                        for (let i = 0; i < 4; i++) {
                            const t0 = performance.now();
                            const res = await apiFetch(`/api/speedtest/ping?_=${cacheBust()}`, { cache: 'no-store' });
                            const t1 = performance.now();
                            if (!res.ok) throw new Error(await parseErr(res, 'ping_failed'));
                            await res.json().catch(() => ({}));
                            pingSamples.push(t1 - t0);
                        }
                        const pingAvg = pingSamples.reduce((a, b) => a + b, 0) / Math.max(1, pingSamples.length);
                        const pingMin = Math.min(...pingSamples);

                        const runDownload = async (bytes) => {
                            const t0 = performance.now();
                            const res = await apiFetch(`/api/speedtest/download?bytes=${bytes}&_=${cacheBust()}`, { cache: 'no-store' });
                            if (!res.ok) throw new Error(await parseErr(res, 'download_failed'));
                            const buf = await res.arrayBuffer();
                            const t1 = performance.now();
                            return { bytes: buf.byteLength || bytes, ms: t1 - t0, mbps: toMbps(buf.byteLength || bytes, t1 - t0) };
                        };

                        setBox(`測定中... ping ${fmtMs(pingAvg)}\n測定中... download`);
                        const dlRuns = [];
                        for (const bytes of [2 * 1024 * 1024, 8 * 1024 * 1024]) {
                            dlRuns.push(await runDownload(bytes));
                            setBox(`測定中... ping ${fmtMs(pingAvg)}\ndownload ${fmtMbps(Math.max(...dlRuns.map(x => x.mbps)))}\n測定中... upload`);
                        }
                        const downloadBest = Math.max(...dlRuns.map(x => x.mbps));

                        const runUpload = async (bytes) => {
                            const payload = new Uint8Array(bytes);
                            const t0 = performance.now();
                            const res = await apiFetch(`/api/speedtest/upload?_=${cacheBust()}`, {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/octet-stream' },
                                body: payload,
                                cache: 'no-store'
                            });
                            const t1 = performance.now();
                            if (!res.ok) throw new Error(await parseErr(res, 'upload_failed'));
                            const data = await res.json().catch(() => ({}));
                            const actualBytes = Number(data.bytes_received || bytes) || bytes;
                            return { bytes: actualBytes, ms: t1 - t0, mbps: toMbps(actualBytes, t1 - t0), serverMs: Number(data.server_elapsed_ms || 0) || 0 };
                        };

                        const ulRuns = [];
                        for (const bytes of [1 * 1024 * 1024, 4 * 1024 * 1024]) {
                            ulRuns.push(await runUpload(bytes));
                        }
                        const uploadBest = Math.max(...ulRuns.map(x => x.mbps));

                        const lines = [
                            '結果 (ブラウザ⇔このサーバー)',
                            `Ping (avg/min): ${fmtMs(pingAvg)} / ${fmtMs(pingMin)}`,
                            `Download (best): ${fmtMbps(downloadBest)}`,
                            `Upload (best): ${fmtMbps(uploadBest)}`,
                            `Download runs: ${dlRuns.map(x => `${Math.round(x.bytes / 1024 / 1024)}MB=${fmtMbps(x.mbps)}`).join(', ')}`,
                            `Upload runs: ${ulRuns.map(x => `${Math.round(x.bytes / 1024 / 1024)}MB=${fmtMbps(x.mbps)}`).join(', ')}`,
                            '注記: fast.com のようなインターネット全体の速度ではなく、このアプリサーバーまでの回線速度の目安です。'
                        ];
                        setBox(lines.join('\n'));
                        showToast('回線速度テストを実行しました', 'success');
                    } catch (e) {
                        if (box) box.textContent = `エラー: ${e && e.message ? e.message : '回線速度テストに失敗しました'}`;
                        showToast('回線速度テストに失敗しました', 'error', true);
                    } finally {
                        if (btn) {
                            btn.disabled = false;
                            btn.classList.remove('opacity-60', 'cursor-not-allowed');
                        }
                    }
                };
            }
            if (get('ban-appeal-refresh')) {
                get('ban-appeal-refresh').onclick = () => loadBanAppeals();
            }
            if (get('ban-appeal-mark-read')) {
                get('ban-appeal-mark-read').onclick = () => markBanAppealsRead();
            }
            if (get('ban-appeal-list')) {
                get('ban-appeal-list').addEventListener('click', async (e) => {
                    const btn = e.target.closest('button');
                    if (!btn) return;
                    const id = btn.getAttribute('data-id');
                    if (btn.classList.contains('ban-appeal-mark')) {
                        if (id) await markBanAppealsRead([Number(id)]);
                        return;
                    }
                    if (btn.classList.contains('ban-appeal-status')) {
                        const status = btn.getAttribute('data-status');
                        if (id && status) await updateBanAppealStatus({ id: Number(id), status });
                        return;
                    }
                    if (btn.classList.contains('ban-appeal-reply-send')) {
                        const card = btn.closest('[data-appeal-id]');
                        const textarea = card ? card.querySelector('.ban-appeal-reply') : null;
                        const reply = textarea ? textarea.value : '';
                        if (id) await updateBanAppealStatus({ id: Number(id), admin_reply: reply });
                        return;
                    }
                    if (btn.classList.contains('ban-appeal-block')) {
                        if (!confirm('このユーザーの異議申し立てをブロックしますか？')) return;
                        const reason = prompt('ブロック理由 (任意)') || '';
                        if (id) await updateBanAppealStatus({ id: Number(id), block_user: true, block_reason: reason });
                        return;
                    }
                });
            }
            if (get('upload-modal-close')) get('upload-modal-close').onclick = () => closeUploadModal();
            if (get('upload-select-btn')) get('upload-select-btn').onclick = () => get('file-input').click();
            if (get('upload-camera-btn')) get('upload-camera-btn').onclick = () => openCameraCaptureModal();
            if (get('upload-photo-btn')) get('upload-photo-btn').onclick = () => get('photo-input').click();
            if (get('camera-modal-close')) get('camera-modal-close').onclick = () => closeCameraCaptureModal();
            if (get('camera-capture-btn')) get('camera-capture-btn').onclick = () => captureCameraShot();
            if (get('camera-attach-btn')) get('camera-attach-btn').onclick = () => attachCameraCapturedFiles();
            if (get('camera-switch-btn')) get('camera-switch-btn').onclick = () => toggleCameraCaptureFacing();
            if (get('camera-clear-btn')) get('camera-clear-btn').onclick = () => resetCameraCapturePending();
            if (get('camera-fallback-btn')) {
                get('camera-fallback-btn').onclick = () => {
                    closeCameraCaptureModal();
                    const photoInput = get('photo-input');
                    if (photoInput) photoInput.click();
                };
            }
            if (get('upload-clear-btn')) {
                get('upload-clear-btn').onclick = () => {
                    resetUploadState();
                };
            }
            if (get('marker-modal-close')) {
                get('marker-modal-close').onclick = () => {
                    closeMarkerModal();
                    markerState.row = null;
                };
            }
            if (get('marker-tool-draw')) get('marker-tool-draw').onclick = () => setMarkerMode('draw');
            if (get('marker-tool-mosaic')) get('marker-tool-mosaic').onclick = () => setMarkerMode('mosaic');
            if (get('marker-tool-crop')) get('marker-tool-crop').onclick = () => setMarkerMode('crop');
            const markerColorPicker = get('marker-color-picker');
            if (markerColorPicker) {
                markerColorPicker.oninput = (e) => setMarkerColor(e.target.value);
                markerColorPicker.onchange = (e) => setMarkerColor(e.target.value);
            }
            const markerOpacity = get('marker-opacity');
            if (markerOpacity) {
                markerOpacity.oninput = (e) => setMarkerOpacity(e.target.value);
                markerOpacity.onchange = (e) => setMarkerOpacity(e.target.value);
            }
            const markerOpacityNumber = get('marker-opacity-number');
            if (markerOpacityNumber) {
                markerOpacityNumber.onchange = (e) => setMarkerOpacity(e.target.value);
                markerOpacityNumber.onblur = (e) => setMarkerOpacity(e.target.value);
                markerOpacityNumber.onkeydown = (e) => {
                    if (e.key === 'Enter') {
                        setMarkerOpacity(e.target.value);
                        e.target.blur();
                    }
                };
            }
            document.querySelectorAll('#marker-toolbar .marker-color-chip[data-marker-color]').forEach((chip) => {
                chip.onclick = () => setMarkerColor(chip.getAttribute('data-marker-color'));
            });
            if (get('marker-view-reset')) get('marker-view-reset').onclick = () => resetMarkerTransform();
            if (get('marker-crop-reset')) get('marker-crop-reset').onclick = () => clearCropRect();
            if (get('marker-undo')) get('marker-undo').onclick = () => undoMarkerCanvas();
            if (get('marker-clear')) get('marker-clear').onclick = () => clearMarkerCanvas();
            if (get('marker-save')) get('marker-save').onclick = () => saveMarkerToRow();
            syncMarkerColorControls();
            initMarkerCanvas();
            initCropCanvas();
            window.addEventListener('resize', () => {
                const markerModal = get('marker-modal');
                if (!markerModal || markerModal.classList.contains('hidden')) return;
                applyMarkerTransform();
                renderCropOverlay();
            });
            const dropzone = get('upload-dropzone');
            if (dropzone) {
                dropzone.addEventListener('dragover', (e) => {
                    e.preventDefault();
                    dropzone.classList.add('dragover');
                });
                dropzone.addEventListener('dragleave', () => {
                    dropzone.classList.remove('dragover');
                });
                dropzone.addEventListener('drop', (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    dropzone.classList.remove('dragover');
                    const files = e.dataTransfer ? e.dataTransfer.files : null;
                    if (files && files.length) handleFiles(files);
                });
            }
            const dropOverlay = get('drop-overlay');
            let dragCounter = 0;
            const showDropOverlay = () => {
                if (dropOverlay) {
                    dropOverlay.classList.remove('hidden');
                    dropOverlay.classList.add('flex');
                }
            };
            const hideDropOverlay = () => {
                if (dropOverlay) {
                    dropOverlay.classList.add('hidden');
                    dropOverlay.classList.remove('flex');
                }
            };
            window.addEventListener('dragenter', (e) => {
                if (!e.dataTransfer || !e.dataTransfer.types || !e.dataTransfer.types.includes('Files')) return;
                dragCounter += 1;
                showDropOverlay();
            });
            window.addEventListener('dragover', (e) => {
                if (!e.dataTransfer || !e.dataTransfer.types || !e.dataTransfer.types.includes('Files')) return;
                e.preventDefault();
            });
            window.addEventListener('dragleave', (e) => {
                if (!e.dataTransfer || !e.dataTransfer.types || !e.dataTransfer.types.includes('Files')) return;
                dragCounter = Math.max(0, dragCounter - 1);
                if (dragCounter === 0) hideDropOverlay();
            });
            window.addEventListener('drop', (e) => {
                if (!e.dataTransfer || !e.dataTransfer.files || e.dataTransfer.files.length === 0) return;
                e.preventDefault();
                if (dropzone && dropzone.contains(e.target)) return;
                dragCounter = 0;
                hideDropOverlay();
                handleFiles(e.dataTransfer.files);
            });
            const botAdminModal = get('bot-admin-modal');
            const renderBotUsers = (users) => {
                const list = get('bot-admin-list');
                if (!list) return;
                list.innerHTML = '';
                if (!users || !users.length) {
                    list.innerHTML = '<div class="text-xs text-gray-400">該当ユーザーがいません。</div>';
                    return;
                }
                users.forEach(u => {
                    const isBanned = !!u.is_bot_banned;
                    const detOn = u.bot_detection_enabled !== false;
                    const row = document.createElement('div');
                    row.className = 'flex items-center gap-2 bg-gray-900 border border-gray-700 rounded p-2 text-xs';
                    row.innerHTML = `
                        <div class="flex-1">
                            <div class="text-gray-200 font-bold">${escapeHtml(u.username)}</div>
                            <div class="text-[10px] text-gray-500">${isBanned ? 'BAN中' : '正常'} ${u.bot_ban_reason ? ' / ' + escapeHtml(u.bot_ban_reason) : ''}</div>
                        </div>
                        <button class="bot-toggle-detect bg-gray-700 hover:bg-gray-600 text-white px-2 py-1 rounded" data-user="${escapeHtml(u.username)}" data-enabled="${detOn ? '1' : '0'}">${detOn ? '検出ON' : '検出OFF'}</button>
                        <button class="bot-toggle-ban ${isBanned ? 'bg-green-600 hover:bg-green-500' : 'bg-red-600 hover:bg-red-500'} text-white px-2 py-1 rounded" data-user="${escapeHtml(u.username)}" data-banned="${isBanned ? '1' : '0'}">${isBanned ? '単独解除' : 'BAN'}</button>                        ${isBanned ? `<button class=\"bot-toggle-unban-linked bg-rose-600 hover:bg-rose-500 text-white px-2 py-1 rounded\" data-user=\"${escapeHtml(u.username)}\">連鎖解除</button>` : ''}
                        <button class="bot-delete-account bg-red-800 hover:bg-red-700 text-white px-2 py-1 rounded" data-progress-expected-slow="true" data-user="${escapeHtml(u.username)}">削除</button>
                    `;
                    list.appendChild(row);
                });
            };
            const loadBotUsers = async (q = '') => {
                const res = await apiFetch(`/api/bot/users?q=${encodeURIComponent(q)}`);
                const data = await res.json();
                if (res.ok && data && data.users) renderBotUsers(data.users);
                else showToast('ユーザー一覧取得に失敗しました', 'error', true);
            };
            const openBotAdminModal = async () => {
                if (botAdminModal) showModal('bot-admin-modal');
                if (location.pathname !== '/admin-bots') {
                    history.pushState({ modal: 'admin-bots' }, '', '/admin-bots');
                }
                await loadBotUsers(get('bot-admin-search') ? get('bot-admin-search').value.trim() : '');
            };
            window.closeBotAdminModal = (skipHistory = false) => {
                if (botAdminModal) hideModal('bot-admin-modal');
                if (!skipHistory && location.pathname === '/admin-bots') {
                    history.back();
                }
            };
            if (get('bot-admin-open')) {
                get('bot-admin-open').onclick = openBotAdminModal;
            }
            if (get('bot-admin-close')) {
                get('bot-admin-close').onclick = () => closeBotAdminModal();
            }
            if (get('bot-admin-search-btn')) {
                get('bot-admin-search-btn').onclick = async () => {
                    await loadBotUsers(get('bot-admin-search') ? get('bot-admin-search').value.trim() : '');
                };
            }
            if (get('bot-admin-refresh-btn')) {
                get('bot-admin-refresh-btn').onclick = async () => { await loadBotUsers(''); };
            }
            if (get('bot-admin-search')) {
                get('bot-admin-search').addEventListener('keydown', async (e) => {
                    if (e.key === 'Enter') await loadBotUsers(get('bot-admin-search').value.trim());
                });
            }
            if (get('bot-admin-list')) {
                get('bot-admin-list').onclick = async (e) => {
                    const btn = e.target.closest('button');
                    if (!btn) return;
                    const username = btn.getAttribute('data-user');
                    if (!username) return;

                    let res;
                    if (btn.classList.contains('bot-toggle-detect')) {
                        const enabled = btn.getAttribute('data-enabled') !== '1';
                        res = await apiFetch('/api/bot/update', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({username, action:'toggle_detection', enabled})});
                    } else if (btn.classList.contains('bot-toggle-ban')) {
                        const banned = btn.getAttribute('data-banned') === '1';
                        if (banned) {
                            res = await apiFetch('/api/bot/update', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({username, action:'unban'})});
                        } else {
                            if (!confirm(`ユーザー ${username} をBANしますか？`)) return;
                            res = await apiFetch('/api/bot/update', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({username, action:'ban', reason:'Admin ban'})});
                        }
                    } else if (btn.classList.contains('bot-toggle-unban-linked')) {
                        if (!confirm(`ユーザー ${username} の連鎖BANを解除しますか？`)) return;
                        res = await apiFetch('/api/bot/update', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({username, action:'unban_linked'})});
                    } else if (btn.classList.contains('bot-delete-account')) {
                        if (!confirm(`ユーザー ${username} のアカウントを完全削除しますか？\n関連データも即時削除され、この操作は取り消せません。`)) return;
                        res = await apiFetch('/api/bot/update', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({username, action:'delete_account'})});
                    }

                    if (res) {
                        if (res.status === 404) {
                            showToast(`ユーザー ${username} は既に見つかりません（削除された可能性があります）`, 'error', true);
                        } else if (!res.ok) {
                            const d = await res.json();
                            showToast(d.error || 'エラーが発生しました', 'error', true);
                        } else if (btn.classList.contains('bot-delete-account')) {
                            showToast(`ユーザー ${username} を削除しました`, 'success');
                            if (username === currentUsername) {
                                location.href = '/';
                                return;
                            }
                        }
                        await loadBotUsers(get('bot-admin-search') ? get('bot-admin-search').value.trim() : '');
                    }
                };
            }
            const MODAL_CONFIG = {
                '/settings': { id: 'settings-modal', open: () => window.openSettingsModal() },
                '/upload': { id: 'upload-modal', open: () => openUploadModal() },
                '/library': { id: 'lib-modal', open: () => { setLibAttachMode(false); showModal('lib-modal'); loadLibraryFiles(); } },
                '/history': { id: 'history-modal', open: () => window.showHistoryModal() },
                '/branch': { id: 'branch-modal', open: () => window.showBranchModal() },
                '/paste': { id: 'rich-paste-modal', open: () => openRichPasteModal() },
                '/camera': { id: 'camera-capture-modal', open: () => openCameraCaptureModal() },
                '/edit-image': { id: 'marker-modal', open: () => { /* Marker modal usually needs context */ } },
                '/chat-settings': { id: 'thread-modal', open: () => window.openThreadModal() },
                '/model': { id: 'model-modal', open: () => openModelModal() },
                '/gem': { id: 'gem-modal', open: () => { editingGemUuid = null; get('gem-modal-title').innerHTML = `<i class="fas fa-gem text-blue-500 mr-2"></i>Create New Gem`; showModal('gem-modal'); } },
                '/compression': { id: 'compression-modal', open: () => window.openCompressionModal() },
                '/admin-bots': { id: 'bot-admin-modal', open: () => { if(CHAT_CONFIG.botConfig && CHAT_CONFIG.botConfig.isAdmin) showModal('bot-admin-modal'); } }
            };

            const closeModalById = (id, skipHistory = false) => {
                switch (id) {
                    case 'settings-modal': closeSettingsModal(skipHistory); break;
                    case 'upload-modal': closeUploadModal(skipHistory); break;
                    case 'camera-capture-modal': closeCameraCaptureModal(skipHistory ? {skipHistory: true} : {}); break;
                    case 'history-modal': if (window.closeHistoryModal) window.closeHistoryModal(skipHistory); break;
                    case 'lib-modal': if (window.closeLibModal) window.closeLibModal(skipHistory); break;
                    case 'branch-modal': if (window.closeBranchModal) window.closeBranchModal(skipHistory); break;
                    case 'rich-paste-modal': if (window.closeRichPasteModal) window.closeRichPasteModal(skipHistory); break;
                    case 'marker-modal': if (window.closeMarkerModal) window.closeMarkerModal(skipHistory); break;
                    case 'thread-modal': if (window.closeThreadModal) window.closeThreadModal(skipHistory); break;
                    case 'model-modal': if (window.closeModelModal) window.closeModelModal(skipHistory); break;
                    case 'gem-modal': if (window.closeGemModal) window.closeGemModal(skipHistory); break;
                    case 'compression-modal': if (window.closeCompressionModal) window.closeCompressionModal(skipHistory); break;
                    case 'bot-admin-modal': if (window.closeBotAdminModal) window.closeBotAdminModal(skipHistory); break;
                    case 'version-update-modal':
                        const latest = localStorage.getItem("app_version") || "";
                        if (latest) localStorage.setItem("version_notified", latest);
                        hideModal(id);
                        break;
                    default: hideModal(id); break;
                }
            };

            window.addEventListener('popstate', (e) => {
                let modalClosed = false;
                Object.values(MODAL_CONFIG).forEach(cfg => {
                    const el = get(cfg.id);
                    if (el && el.classList.contains('modal-open')) {
                        if (location.pathname !== Object.keys(MODAL_CONFIG).find(k => MODAL_CONFIG[k].id === cfg.id)) {
                            closeModalById(cfg.id, true);
                            modalClosed = true;
                        }
                    }
                });
                
                // If we navigated TO a modal path, open it
                const config = MODAL_CONFIG[location.pathname];
                if (config) {
                    const el = get(config.id);
                    if (el && !el.classList.contains('modal-open')) {
                        config.open();
                    }
                }
            });

            const initialPath = location.pathname;
            if (MODAL_CONFIG[initialPath]) {
                history.replaceState({}, '', '/');
                setTimeout(() => MODAL_CONFIG[initialPath].open(), 500); // Small delay to ensure everything is initialized
            }
            if (get('easy-login-generate')) {
                get('easy-login-generate').onclick = async () => {
                    const minsEl = get('easy-login-mins');
                    const mins = minsEl ? parseInt(minsEl.value || '5', 10) : 5;
                    if (!confirm(`簡易ログインを${mins}分間有効にしますか？`)) return;
                    const res = await apiFetch("/api/easy_login", {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({minutes: mins})});
                    const data = await res.json();
                    if (data && data.temp_password) {
                        get('easy-login-code').textContent = data.temp_password;
                        get('easy-login-exp').textContent = data.expires_at || '';
                        get('easy-login-result').classList.remove('hidden');
                    } else {
                        showToast("簡易ログインの発行に失敗しました", "error", true);
                    }
                };
            }
            if (get('easy-login-cancel')) {
                get('easy-login-cancel').onclick = async () => {
                    if (!confirm("現在の一時パスワード発行をキャンセルしますか？")) return;
                    const res = await apiFetch("/api/easy_login", {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({cancel: true})});
                    const data = await res.json();
                    if (data && data.cancelled) {
                        const box = get('easy-login-result');
                        if (box) box.classList.add('hidden');
                        showToast("簡易ログインをキャンセルしました", "success");
                    } else {
                        showToast("キャンセルに失敗しました", "error", true);
                    }
                };
            }
            get('fb-submit').onclick = async () => {
                const title = get('fb-title').value.trim();
                const message = get('fb-message').value.trim();
                if(!message) { showToast("フィードバック内容を入力してください", "error", true); return; }
                await apiFetch("/api/feedback", {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({title, message})});
                get('fb-title').value = '';
                get('fb-message').value = '';
                loadFeedback();
            };

            async function loadFeedback() {
                const res = await apiFetch("/api/feedback?all=1");
                const data = await res.json();
                const list = get('fb-list');
                list.innerHTML = '';
                (data.items || []).filter(i => !data.is_admin || i.user_id === undefined || i.user_id === null || true).forEach(item => {
                    if(data.is_admin) return;
                    const el = document.createElement('div');
                    el.className = 'p-2 rounded border border-gray-700 bg-gray-800/50';
                    el.innerHTML = `<div class="text-[11px] text-gray-400">${item.created_at}</div><div class="font-bold text-sm">${escapeHtml(item.title||'No Title')}</div><div class="text-sm whitespace-pre-wrap">${escapeHtml(item.message)}</div><div class="text-[11px] text-gray-400 mt-1">Status: ${escapeHtml(item.status)}</div>${item.admin_reply ? `<div class="text-[11px] text-green-300 mt-1">Reply: ${escapeHtml(item.admin_reply)}</div>` : ''}`;
                    list.appendChild(el);
                });

                const adminPanel = get('fb-admin-panel');
                const adminList = get('fb-admin-list');
                if(data.is_admin) {
                    adminPanel.classList.remove('hidden');
                    adminList.innerHTML = '';
                    (data.items || []).forEach(item => {
                        const el = document.createElement('div');
                        el.className = 'p-2 rounded border border-gray-700 bg-gray-800/50 space-y-2';
                        el.innerHTML = `
                            <div class="text-[11px] text-gray-400">#${item.id} / user:${item.user_id} / ${item.created_at}</div>
                            <div class="font-bold text-sm">${escapeHtml(item.title||'No Title')}</div>
                            <div class="text-sm whitespace-pre-wrap">${escapeHtml(item.message)}</div>
                            <div class="flex items-center gap-2">
                                <select class="fb-status bg-gray-900 border border-gray-700 rounded px-2 py-1 text-xs text-white">
                                    <option value="new">new</option>
                                    <option value="in_review">in_review</option>
                                    <option value="replied">replied</option>
                                    <option value="rejected">rejected</option>
                                    <option value="resolved">resolved</option>
                                </select>
                                <button class="fb-save bg-blue-600 hover:bg-blue-500 text-white px-3 py-1 rounded text-xs">保存</button>
                            </div>
                            <textarea class="fb-reply w-full bg-gray-900 border border-gray-700 rounded px-2 py-1 text-xs text-white" rows="3" placeholder="返信内容">${escapeHtml(item.admin_reply||'')}</textarea>
                        `;
                        el.querySelector('.fb-status').value = item.status || 'new';
                        el.querySelector('.fb-save').onclick = async () => {
                            const status = el.querySelector('.fb-status').value;
                            const admin_reply = el.querySelector('.fb-reply').value;
                            await apiFetch(`/api/feedback/${item.id}/update`, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({status, admin_reply})});
                            loadFeedback();
                        };
                        adminList.appendChild(el);
                    });
                } else {
                    adminPanel.classList.add('hidden');
                }
            }
            
            window.setupTOTP = async () => {
                const r = await apiFetch("/api/2fa/totp/setup", {method:'POST'});
                const d = await r.json();
                get('totp-qr').src = d.qr_image;
                get('totp-secret-disp').innerText = d.secret;
                get('totp-setup-area').classList.remove('hidden');
            };
            
            window.enableTOTP = async () => {
                const c = get('totp-verify-code').value;
                if(!c) return;
                const r = await apiFetch("/api/2fa/totp/enable", {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({code:c})});
                if(r.ok) { 
                    showToast("TOTPが有効になりました", "success");
                    get('totp-setup-area').classList.add('hidden');
                    get('totp-verify-code').value = '';
                    openSettingsModal(); // Refresh UI
                }
                else showToast("認証コードが正しくありません", "error", true);
            };
            
            window.registerWebAuthn = async () => {
                const btn = get('register-webauthn-btn');
                const nameInput = get('webauthn-name');
                const passkeyName = nameInput ? String(nameInput.value || '').trim() : '';
                try {
                    if (btn) btn.disabled = true;
                    const optsRes = await apiFetch('/api/2fa/webauthn/register/options', {method:'POST'});
                    const options = await optsRes.json();
                    if (!optsRes.ok) {
                        showToast(options.error || 'パスキー登録の準備に失敗しました', 'error', true);
                        return;
                    }
                    const webauthnJSON = await ensureWebAuthnJson();
                    const credential = await webauthnJSON.create({ publicKey: options });
                    const verifyRes = await apiFetch('/api/2fa/webauthn/register/verify', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(Object.assign({}, credential, { name: passkeyName }))
                    });
                    const d = await verifyRes.json().catch(() => ({}));
                    if (verifyRes.ok) {
                        if (nameInput) nameInput.value = '';
                        showToast('パスキーを登録しました', 'success');
                        openSettingsModal();
                    } else {
                        showToast(d.error || 'パスキー登録に失敗しました', 'error', true);
                    }
                } catch(e) {
                    showToast(`WebAuthn Error: ${e}`, 'error', true);
                } finally {
                    if (btn) btn.disabled = false;
                }
            };
            window.removeWebAuthnCredential = async (credId) => {
                if (!credId) return;
                if (!confirm('このパスキーを削除しますか？')) return;
                const res = await apiFetch('/api/2fa/webauthn/remove', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ id: credId })
                });
                const d = await res.json().catch(() => ({}));
                if (res.ok) {
                    showToast('パスキーを削除しました', 'success');
                    openSettingsModal();
                    return;
                }
                showToast(d.error || 'パスキー削除に失敗しました', 'error', true);
            };

            if (get('delete-account-btn')) {
                get('delete-account-btn').onclick = async () => {
                    if(confirm("本当にアカウントを削除しますか？\nこの操作は取り消せません。")) {
                        await apiFetch(CHAT_CONFIG.urls.deleteAccount, {method:'POST'});
                        location.href = "/";
                    }
                };
            }
            get('prompt-input').onkeydown = (e) => {
                if (e.isComposing) return;

                const input = get('prompt-input');

                // Slash command keyboard navigation
                if (slashSuggestionsVisible) {
                    const box = get('slash-command-suggestions');
                    if (e.key === 'ArrowDown') {
                        e.preventDefault();
                        slashSelectedIndex = Math.min(slashSelectedIndex + 1, SLASH_COMMANDS.length - 1);
                        showSlashCommandSuggestions(input.value.trim().substring(1));
                        return;
                    }
                    if (e.key === 'ArrowUp') {
                        e.preventDefault();
                        slashSelectedIndex = Math.max(slashSelectedIndex - 1, 0);
                        showSlashCommandSuggestions(input.value.trim().substring(1));
                        return;
                    }
                    if (e.key === 'Enter') {
                        e.preventDefault();
                        // Pick the currently highlighted (or first)
                        const filtered = SLASH_COMMANDS.filter(c => 
                            c.label.toLowerCase().includes(input.value.trim().substring(1).toLowerCase())
                        );
                        if (filtered[slashSelectedIndex]) {
                            selectSlashCommand(filtered[slashSelectedIndex].id);
                        } else if (filtered.length > 0) {
                            selectSlashCommand(filtered[0].id);
                        }
                        return;
                    }
                    if (e.key === 'Escape') {
                        e.preventDefault();
                        hideSlashCommandSuggestions();
                        return;
                    }
                }

                // Gem suggestion keyboard navigation
                if (gemSuggestionsVisible) {
                    const val = input.value.trim();
                    if (e.key === 'ArrowDown') {
                        e.preventDefault();
                        gemSelectedIndex = gemSelectedIndex + 1;
                        showGemSuggestions(val.substring(1));
                        return;
                    }
                    if (e.key === 'ArrowUp') {
                        e.preventDefault();
                        gemSelectedIndex = Math.max(gemSelectedIndex - 1, 0);
                        showGemSuggestions(val.substring(1));
                        return;
                    }
                    if (e.key === 'Enter') {
                        e.preventDefault();
                        const filter = val.substring(1).toLowerCase();
                        const filtered = loadedGems.filter(g =>
                            g.name.toLowerCase().includes(filter) || (g.description && g.description.toLowerCase().includes(filter))
                        );
                        if (filtered[gemSelectedIndex]) {
                            selectGemSuggestion(filtered[gemSelectedIndex]);
                        } else if (filtered.length > 0) {
                            selectGemSuggestion(filtered[0]);
                        }
                        return;
                    }
                    if (e.key === 'Escape') {
                        e.preventDefault();
                        hideGemSuggestions();
                        return;
                    }
                }

                // Cancel pending slash command mode with Escape
                if (e.key === 'Escape' && pendingSlashCommand) {
                    e.preventDefault();
                    hidePendingSlashCommandIndicator();
                    // Optionally clear the input if user wants a fresh start
                    // input.value = '';
                    return;
                }

                if (e.key === 'ArrowUp' && (input.selectionStart === 0 || e.ctrlKey)) {
                    if (promptHistory.length > 0) {
                        if (historyIndex === -1) {
                            tempPrompt = input.value;
                        }
                        if (historyIndex < promptHistory.length - 1) {
                            e.preventDefault();
                            historyIndex++;
                            input.value = promptHistory[historyIndex];
                            input.dispatchEvent(new Event('input'));
                        }
                    }
                } else if (e.key === 'ArrowDown' && (input.selectionEnd === input.value.length || e.ctrlKey)) {
                    if (historyIndex > -1) {
                        e.preventDefault();
                        historyIndex--;
                        if (historyIndex === -1) {
                            input.value = tempPrompt;
                        } else {
                            input.value = promptHistory[historyIndex];
                        }
                        input.dispatchEvent(new Event('input'));
                    }
                }

                if (enterToSend) {
                    if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        sendMessage();
                    }
                } else if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
                    e.preventDefault();
                    sendMessage();
                }
            };
            if (get('prompt-input')) {
                get('prompt-input').addEventListener('input', function() {
                    this.style.height = 'auto';
                    this.style.height = (this.scrollHeight) + 'px';
                    schedulePromptTokenEstimate();

                    // Slash command / gem suggestion triggers
                    const val = this.value.trim();
                    if (val.startsWith('@')) {
                        const filter = val.substring(1);
                        showGemSuggestions(filter);
                        if (slashSuggestionsVisible) hideSlashCommandSuggestions();
                    } else if (val.startsWith('/')) {
                        const filter = val.substring(1);
                        showSlashCommandSuggestions(filter);
                        if (gemSuggestionsVisible) hideGemSuggestions();
                    } else {
                        if (gemSuggestionsVisible) hideGemSuggestions();
                        if (slashSuggestionsVisible) hideSlashCommandSuggestions();
                    }
                });

                get('prompt-input').addEventListener('blur', () => {
                    // Delay to allow click on suggestion items
                    setTimeout(() => {
                        if (slashSuggestionsVisible) hideSlashCommandSuggestions();
                        if (gemSuggestionsVisible) hideGemSuggestions();
                    }, 150);
                });
            }
            if (get('cancel-edit-btn')) get('cancel-edit-btn').onclick = cancelEdit;
            updatePromptPlaceholder();
            if (get('search-box')) {
                get('search-box').addEventListener('input', () => { 
                    clearTimeout(searchTimeout); 
                    searchTimeout = setTimeout(() => { threadPage=1; get('thread-list').innerHTML='<div id="scroll-sentinel"></div>'; loadThreads(); }, 300); 
                }); 
            }
            if (get('mobile-new-chat-btn')) get('mobile-new-chat-btn').onclick = () => startNewChat();
            if (get('sts-mic-btn')) get('sts-mic-btn').onclick = () => { if (isStsModel()) get('mic-btn').click(); };
            if (get('sts-cancel-btn')) get('sts-cancel-btn').onclick = () => { if (isStsModel()) cancelRecording(); };
            if (get('prompt-input')) {
                get('prompt-input').addEventListener('paste', async (e) => {
                    const items = (e.clipboardData || window.clipboardData).items;
                    const files = [];
                    for (let i = 0; i < items.length; i++) {
                        if (items[i].kind === 'file') {
                            const file = items[i].getAsFile();
                            if (file) files.push(file);
                        }
                    }
                    if (files.length > 0) {
                        e.preventDefault();
                        await handleFiles(files, { openModal: false });
                    }
                });
            }
            if (get('rich-paste-btn')) get('rich-paste-btn').onclick = () => openRichPasteModal();
            if (get('rich-paste-modal-close')) get('rich-paste-modal-close').onclick = () => closeRichPasteModal();
            if (get('rich-paste-close-btn')) get('rich-paste-close-btn').onclick = () => closeRichPasteModal();
            if (get('rich-paste-focus-btn')) get('rich-paste-focus-btn').onclick = () => focusRichPasteEditor();
            if (get('rich-paste-clear-btn')) get('rich-paste-clear-btn').onclick = () => clearRichPasteEditor(true);
            if (get('rich-paste-preview-btn')) get('rich-paste-preview-btn').onclick = () => openRichPastePreviewTab();
            if (get('rich-paste-send-btn')) get('rich-paste-send-btn').onclick = () => sendRichPasteToModel();
            if (get('rich-paste-send-server-btn')) get('rich-paste-send-server-btn').onclick = () => sendRichPasteToModel({ serverSide: true });
            if (get('rich-paste-import-btn')) get('rich-paste-import-btn').onclick = async () => {
                try {
                    const inserted = await readClipboardRichContent();
                    if (!inserted) {
                        showToast('クリップボードにリッチテキストが見つかりませんでした。Ctrl+V で貼り付けてください。', 'warning', true);
                    }
                } catch (e) {
                    const msg = (e && e.message) ? e.message : 'クリップボードの取り込みに失敗しました';
                    showToast(msg, 'error', true);
                }
            };
            if (get('rich-paste-prompt')) {
                const promptEl = get('rich-paste-prompt');
                promptEl.addEventListener('input', () => {
                    if (!richPastePromptPreferenceSyncing) {
                        queueRichPastePromptPreferenceSave();
                    }
                });
            }
            if (get('rich-paste-use-default')) {
                const checkbox = get('rich-paste-use-default');
                checkbox.addEventListener('change', () => {
                    if (!richPastePromptPreferenceSyncing) {
                        queueRichPastePromptPreferenceSave();
                    }
                });
            }
            if (get('rich-paste-capture')) {
                const capture = get('rich-paste-capture');
                capture.addEventListener('paste', async (e) => {
                    const clipboard = e.clipboardData || window.clipboardData;
                    if (!clipboard) return;
                    e.preventDefault();
                    try {
                        const inserted = await ingestRichPasteClipboardData(clipboard);
                        if (!inserted) {
                            showToast('クリップボードに貼り付け可能な内容がありませんでした', 'warning', true);
                        }
                        updateRichPasteStatus();
                    } catch (err) {
                        showToast('貼り付けの取り込みに失敗しました', 'error', true);
                    }
                });
                capture.addEventListener('input', () => {
                    capture.value = '';
                });
            }
            get('chat-container').addEventListener('click', (e) => {
                const img = e.target.closest('img.chat-image');
                const viewerSrc = img ? (img.dataset.viewerSrc || img.currentSrc || img.src) : '';
                if (img && viewerSrc) {
                    e.preventDefault();
                    openImageViewer(viewerSrc);
                }
            });
            get('image-viewer').addEventListener('click', (e) => {
                // Close if clicking the background area, but not the image or UI controls
                if (e.target.id === 'image-viewer' || e.target.classList.contains('viewer-content')) {
                    closeImageViewer();
                }
            });
            get('file-viewer').addEventListener('click', (e) => {
                if (e.target.id === 'file-viewer') closeFileViewer();
            });
            document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeImageViewer(); });
            
            // Microphone / STT or STS
            let mediaRecorder;
            let currentGeminiLive = null;
            let audioChunks = [];
            let stsCancelPending = false;
            let stsSilenceInterval = null;
            let stsAudioCtx = null;
            let stsAnalyser = null;
            let stsSource = null;
            let stsLastSoundTs = 0;
            let stsHasSound = false;
            let stsPlaybackAudio = null;
            let micVizAudioCtx = null;
            let micVizAnalyser = null;
            let micVizSource = null;
            let micVizAnimationFrame = null;
            let micVizData = null;
            let micVizBars = null;
            let micIndicatorHideTimer = null;
            function ensureMicWaveformBars() {
                const wave = get('mic-waveform');
                if (!wave) return [];
                if (Array.isArray(micVizBars) && micVizBars.length) return micVizBars;
                wave.innerHTML = '';
                const bars = [];
                for (let i = 0; i < 24; i++) {
                    const bar = document.createElement('span');
                    bar.className = 'block rounded-full';
                    bar.style.background = 'rgba(252, 165, 165, 0.92)';
                    bar.style.width = '2px';
                    bar.style.transition = 'height 75ms linear, opacity 75ms linear';
                    bar.style.height = '2px';
                    bar.style.opacity = '0.4';
                    bars.push(bar);
                    wave.appendChild(bar);
                }
                micVizBars = bars;
                return bars;
            }
            function setMicRecordingIndicator(text, mode = 'hidden') {
                const box = get('mic-recording-indicator');
                const label = get('mic-recording-text');
                if (!box) return;
                if (micIndicatorHideTimer) {
                    clearTimeout(micIndicatorHideTimer);
                    micIndicatorHideTimer = null;
                }
                if (mode === 'hidden') {
                    box.classList.add('hidden');
                    return;
                }
                if (label && text) label.innerText = text;
                box.classList.remove('hidden');
                if (mode === 'recording') box.style.color = 'rgb(252 165 165)';
                else if (mode === 'processing') box.style.color = 'rgb(253 224 71)';
                else box.style.color = 'rgb(209 213 219)';
            }
            function resetMicWaveformBars() {
                const bars = ensureMicWaveformBars();
                bars.forEach((bar) => {
                    bar.style.height = '2px';
                    bar.style.opacity = '0.35';
                });
            }
            function stopMicWaveform() {
                if (micVizAnimationFrame) {
                    cancelAnimationFrame(micVizAnimationFrame);
                    micVizAnimationFrame = null;
                }
                if (micVizSource) { try { micVizSource.disconnect(); } catch (e) {} micVizSource = null; }
                if (micVizAudioCtx) { try { micVizAudioCtx.close(); } catch (e) {} micVizAudioCtx = null; }
                micVizAnalyser = null;
                micVizData = null;
                resetMicWaveformBars();
            }
            function startMicWaveform(stream) {
                stopMicWaveform();
                const bars = ensureMicWaveformBars();
                if (!bars.length) return;
                const AC = window.AudioContext || window.webkitAudioContext;
                if (!AC) return;
                try {
                    micVizAudioCtx = new AC();
                    micVizAnalyser = micVizAudioCtx.createAnalyser();
                    micVizAnalyser.fftSize = 256;
                    micVizAnalyser.smoothingTimeConstant = 0;
                    micVizSource = micVizAudioCtx.createMediaStreamSource(stream);
                    micVizSource.connect(micVizAnalyser);
                    micVizData = new Uint8Array(micVizAnalyser.frequencyBinCount);
                } catch (e) {
                    stopMicWaveform();
                    return;
                }
                const render = () => {
                    if (!micVizAnalyser || !micVizData) return;
                    micVizAnalyser.getByteFrequencyData(micVizData);
                    const step = Math.max(1, Math.floor(micVizData.length / bars.length));
                    for (let i = 0; i < bars.length; i++) {
                        const raw = micVizData[Math.min(micVizData.length - 1, i * step)] || 0;
                        const level = raw / 255;
                        const px = Math.max(2, Math.round(2 + (level * 10)));
                        bars[i].style.height = `${px}px`;
                        bars[i].style.opacity = `${0.35 + level * 0.65}`;
                    }
                    micVizAnimationFrame = requestAnimationFrame(render);
                };
                render();
            }
            function stopSilenceMonitor() {
                if (stsSilenceInterval) { clearInterval(stsSilenceInterval); stsSilenceInterval = null; }
                if (stsSource) { try { stsSource.disconnect(); } catch(e) {} stsSource = null; }
                if (stsAudioCtx) { try { stsAudioCtx.close(); } catch(e) {} stsAudioCtx = null; }
                stsAnalyser = null;
            }
            function startSilenceMonitor(stream) {
                if (!isStsModel() || !stsOpt('sts-auto-send')) return;
                stopSilenceMonitor();
                const AC = window.AudioContext || window.webkitAudioContext;
                if (!AC) return;
                stsAudioCtx = new AC();
                stsAnalyser = stsAudioCtx.createAnalyser();
                stsAnalyser.fftSize = 2048;
                stsSource = stsAudioCtx.createMediaStreamSource(stream);
                stsSource.connect(stsAnalyser);
                const data = new Uint8Array(stsAnalyser.fftSize);
                const silenceMs = getStsSilenceMs();
                const threshold = 0.02;
                stsLastSoundTs = 0;
                stsHasSound = false;
                stsSilenceInterval = setInterval(() => {
                    if (!stsAnalyser) return;
                    stsAnalyser.getByteTimeDomainData(data);
                    let sum = 0;
                    for (let i = 0; i < data.length; i++) {
                        const v = (data[i] - 128) / 128;
                        sum += v * v;
                    }
                    const rms = Math.sqrt(sum / data.length);
                    if (rms > threshold) {
                        if (!stsHasSound) stsHasSound = true;
                        stsLastSoundTs = Date.now();
                        return;
                    }
                    if (stsHasSound && Date.now() - stsLastSoundTs > silenceMs) {
                        if (mediaRecorder && mediaRecorder.state === "recording") {
                            mediaRecorder.stop();
                        }
                    }
                }, 200);
            }
            class GeminiLiveClient {
                constructor() {
                    this.ws = null;
                    this.audioContext = null;
                    this.processor = null;
                    this.stream = null;
                    this.rtPlayer = null;
                    this.assistantText = '';
                    this.assistantThought = '';
                    this.inputTranscript = '';
                    this.assistantAudioChunks = [];
                    this.userAudioChunks = [];
                    this.onMessage = null;
                    this.onClose = null;
                    this.onError = null;
                    this.setupComplete = false;
                    this.model = null;
                }
                async start(token, url, model, config = {}) {
                    this.model = model;
                    this.ws = new WebSocket(`${url}?access_token=${token}`);
                    this.ws.binaryType = 'arraybuffer';
                    
                    this.ws.onopen = () => {
                        console.log("Gemini Live WebSocket opened. Sending setup...");
                        // Refined setup message based on docs and SDK patterns
                        const setupMsg = {
                            setup: {
                                model: `models/${model}`,
                                generationConfig: {
                                    responseModalities: ["AUDIO"]
                                },
                                // Enable transcription at setup level (as per docs)
                                inputAudioTranscription: {},
                                outputAudioTranscription: {}
                            }
                        };
                        
                        // speechConfig is inside generationConfig
                        if (config.speechConfig) {
                            setupMsg.setup.generationConfig.speechConfig = config.speechConfig;
                        }
                        
                        // thinkingConfig is also inside generationConfig in the Live API schema
                        if (config.thinkingConfig) {
                            setupMsg.setup.generationConfig.thinkingConfig = config.thinkingConfig;
                        }

                        console.log("Sending setup:", JSON.stringify(setupMsg));
                        this.ws.send(JSON.stringify(setupMsg));
                    };

                    this.ws.onmessage = (e) => this._handleMessage(e);
                    this.ws.onerror = (e) => { 
                        console.error("Gemini Live WebSocket error:", e);
                        if (this.onError) this.onError(e); 
                    };
                    this.ws.onclose = (e) => { 
                        console.log("Gemini Live WebSocket closed:", e.code, e.reason);
                        if (this.onClose) this.onClose(e); 
                    };
                    
                    this.audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
                    this.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    const source = this.audioContext.createMediaStreamSource(this.stream);
                    this.processor = this.audioContext.createScriptProcessor(4096, 1, 1);
                    
                    this.userAudioChunks = [];
                    const backupRecorder = new MediaRecorder(this.stream);
                    backupRecorder.ondataavailable = (e) => {
                        if (e.data.size > 0) this.userAudioChunks.push(e.data);
                    };
                    backupRecorder.start(500); 
                    this.backupRecorder = backupRecorder;

                    this.processor.onaudioprocess = (e) => {
                        if (!this.ws || this.ws.readyState !== WebSocket.OPEN || !this.setupComplete) return;
                        const inputData = e.inputBuffer.getChannelData(0);
                        const pcmData = new Int16Array(inputData.length);
                        for (let i = 0; i < inputData.length; i++) {
                            pcmData[i] = Math.max(-1, Math.min(1, inputData[i])) * 0x7FFF;
                        }
                        this.ws.send(JSON.stringify({
                            realtimeInput: {
                                audio: {
                                    data: btoa(String.fromCharCode.apply(null, new Uint8Array(pcmData.buffer))),
                                    mimeType: "audio/pcm;rate=16000"
                                }
                            }
                        }));
                    };
                    source.connect(this.processor);
                    this.processor.connect(this.audioContext.destination);
                }
                _handleMessage(e) {
                    const data = JSON.parse(e.data);
                    console.log("Gemini Live raw message received:", data);
                    
                    if (data.setupComplete) {
                        console.log("Gemini Live setup complete confirmed");
                        this.setupComplete = true;
                    }
                    if (data.serverContent) {
                        const sc = data.serverContent;
                        if (sc.modelTurn) {
                            sc.modelTurn.parts.forEach(p => {
                                if (p.text) {
                                    if (p.thought) {
                                        console.log("Gemini thought delta:", p.text);
                                        this.assistantThought += p.text;
                                    } else {
                                        console.log("Gemini transcript delta (parts):", p.text);
                                        this.assistantText += p.text;
                                    }
                                }
                                if (p.inlineData && p.inlineData.data) {
                                    const audioBase64 = p.inlineData.data;
                                    console.log("Gemini audio chunk received, size:", audioBase64.length);
                                    if (this.rtPlayer) this.rtPlayer.addChunk(audioBase64);
                                    const binary = atob(audioBase64);
                                    const bytes = new Uint8Array(binary.length);
                                    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
                                    this.assistantAudioChunks.push(bytes);
                                }
                            });
                        }
                        if (sc.outputTranscription) {
                            console.log("Gemini output transcription delta:", sc.outputTranscription.text);
                            // Some versions send transcript via outputTranscription instead of modelTurn parts
                            if (!this.assistantText.includes(sc.outputTranscription.text)) {
                                this.assistantText += sc.outputTranscription.text;
                            }
                        }
                        if (sc.inputTranscription) {
                            console.log("User input transcription delta:", sc.inputTranscription.text);
                            this.inputTranscript += sc.inputTranscription.text;
                        }
                    }
                    if (this.onMessage) this.onMessage(data);
                }
                stop() {
                    if (this.ws) this.ws.close();
                    if (this.processor) this.processor.disconnect();
                    if (this.audioContext) this.audioContext.close();
                    if (this.stream) this.stream.getTracks().forEach(t => t.stop());
                    if (this.backupRecorder) this.backupRecorder.stop();
                }
                async getFinalData() {
                    const assistantAudioBlob = new Blob(this.assistantAudioChunks);
                    const assistantAudioBase64 = await this._blobToBase64(assistantAudioBlob);
                    const userAudioBlob = new Blob(this.userAudioChunks);
                    const userAudioBase64 = await this._blobToBase64(userAudioBlob);
                    return {
                        user_text: this.inputTranscript,
                        assistant_text: this.assistantText,
                        assistant_thought: this.assistantThought,
                        audio_base64: assistantAudioBase64,
                        user_audio_base64: userAudioBase64
                    };
                }
                _blobToBase64(blob) {
                    return new Promise(resolve => {
                        const reader = new FileReader();
                        reader.onloadend = () => resolve(reader.result.split(',')[1]);
                        reader.readAsDataURL(blob);
                    });
                }
            }
            class RealTimeAudioPlayer {
                constructor(sampleRate = 24000) {
                    const AC = window.AudioContext || window.webkitAudioContext;
                    this.ctx = new AC({ sampleRate });
                    this.nextStartTime = 0;
                    this.bufferDelay = 0.1; // 100ms jitter buffer
                    this.started = false;
                }
                async addChunk(base64Data) {
                    if (!this.ctx) return;
                    const binary = atob(base64Data);
                    const bytes = new Uint8Array(binary.length);
                    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
                    const pcm = new Int16Array(bytes.buffer);
                    const float32 = new Float32Array(pcm.length);
                    for (let i = 0; i < pcm.length; i++) float32[i] = pcm[i] / 32768.0;
                    
                    const buffer = this.ctx.createBuffer(1, float32.length, this.ctx.sampleRate);
                    buffer.getChannelData(0).set(float32);
                    
                    if (this.ctx.state === 'suspended') await this.ctx.resume();
                    
                    const source = this.ctx.createBufferSource();
                    source.buffer = buffer;
                    source.connect(this.ctx.destination);
                    
                    if (!this.started) {
                        this.nextStartTime = this.ctx.currentTime + this.bufferDelay;
                        this.started = true;
                    }
                    const startTime = Math.max(this.ctx.currentTime, this.nextStartTime);
                    source.start(startTime);
                    this.nextStartTime = startTime + buffer.duration;
                }
                stop() {
                    if (this.ctx) {
                        this.ctx.close();
                        this.ctx = null;
                    }
                }
            }

            let currentRtPlayer = null;

            function stopStsPlayback() {
                if (currentRtPlayer) {
                    currentRtPlayer.stop();
                    currentRtPlayer = null;
                }
                if (stsPlaybackAudio) {
                    try { stsPlaybackAudio.pause(); } catch (e) {}
                    try { stsPlaybackAudio.src = ''; } catch (e) {}
                    stsPlaybackAudio = null;
                }
            }
            async function playStsAudio(url) {
                stopStsPlayback();
                const audio = new Audio();
                audio.src = url;
                audio.preload = 'auto';
                audio.autoplay = true;
                audio.playsInline = true;
                stsPlaybackAudio = audio;
                await audio.play();
                return new Promise(resolve => {
                    audio.onended = () => resolve('ended');
                    audio.onerror = () => resolve('error');
                });
            }
            function cancelRecording() {
                if (currentGeminiLive) {
                    currentGeminiLive.stop();
                    currentGeminiLive = null;
                    stopStsPlayback();
                    get('mic-btn').classList.remove('bg-red-600', 'animate-pulse');
                    get('mic-btn').classList.add('bg-gray-700');
                    setStsStatus('Canceled', false);
                    setTimeout(() => setStsStatus('Tap to speak', false), 800);
                    stopMicWaveform();
                    return;
                }
                if (mediaRecorder && mediaRecorder.state === "recording") {
                    stsCancelPending = true;
                    mediaRecorder.stop();
                }
            }
            function getMicCaptureConstraints() {
                const stsMode = isStsModel();
                if (stsMode) return { audio: true };
                const supported = (navigator.mediaDevices && navigator.mediaDevices.getSupportedConstraints)
                    ? navigator.mediaDevices.getSupportedConstraints()
                    : {};
                const audio = { channelCount: 1 };
                if (supported.echoCancellation) audio.echoCancellation = false;
                if (supported.noiseSuppression) audio.noiseSuppression = false;
                if (supported.autoGainControl) audio.autoGainControl = false;
                return { audio };
            }
            get('mic-btn').onclick = async () => {
                if (abortController) {
                    showToast("回答生成中です。完了までお待ちいただくか、停止してください。", "warning", true);
                    return;
                }
                if (uploadProgressState.active > 0) {
                    showToast("ファイルの送信・処理中です。しばらくお待ちください。", "warning", true);
                    return;
                }
                if (currentGeminiLive) {
                    setStsStatus('Processing...', true);
                    const client = currentGeminiLive;
                    currentGeminiLive = null;
                    client.stop();
                    
                    get('mic-btn').classList.remove('bg-red-600', 'animate-pulse');
                    get('mic-btn').classList.add('bg-gray-700');

                    try {
                        const finalData = await client.getFinalData();
                        
                        if (!currentThreadId) {
                            const r = await apiFetch(CHAT_CONFIG.urls.handleThreads, {
                                method:'POST',
                                headers:{'Content-Type':'application/json'},
                                body: JSON.stringify({ is_temporary: temporaryChatEnabled })
                            });
                            const d = await r.json();
                            currentThreadId = String(d.id);
                            history.pushState({}, '', '/c/' + d.id);
                            get('welcome-screen').classList.add('hidden');
                        }

                        finalData.thread_id = currentThreadId;
                        finalData.model = get('model-select').value;
                        
                        await apiFetch('/api/gemini/save_sts', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify(finalData)
                        });
                        
                        setStsStatus('Saved', false);
                        setTimeout(() => setStsStatus('Tap to speak', false), 1000);
                        await loadMessages(currentThreadId);
                    } catch (e) {
                        console.error("Failed to save Gemini Live session:", e);
                        setStsStatus('Error saving', false);
                    }
                    return;
                }
                if (mediaRecorder && mediaRecorder.state === "recording") {
                    mediaRecorder.stop();
                    get('mic-btn').classList.remove('bg-red-600', 'animate-pulse');
                    get('mic-btn').classList.add('bg-gray-700');
                    if (!isStsModel()) setMicRecordingIndicator('録音を処理中…', 'processing');
                    if (isStsModel()) setStsStatus('Processing...', true);
                    return;
                }
                try {
                    // Pre-activate audio for STS to bypass browser autoplay restrictions
                    if (isStsModel()) {
                        try {
                            const dummyAudio = new Audio();
                            dummyAudio.src = 'data:audio/wav;base64,UklGRiQAAABXQVZFRm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA=';
                            dummyAudio.play().catch(() => {});
                        } catch (e) {}
                    }

                    if (isGeminiLiveModel()) {
                        setStsStatus('Connecting...', true);
                        try {
                            const res = await apiFetch('/api/gemini/session', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({
                                    model: get('model-select').value,
                                    voice: get('sts-voice') ? get('sts-voice').value : 'Kore',
                                    thinking_level: get('sts-thinking-level') ? get('sts-thinking-level').value : 'minimal',
                                    include_thoughts: get('sts-include-thoughts') ? get('sts-include-thoughts').checked : false
                                })
                            });
                            if (!res.ok) throw new Error("Failed to get session token");
                            const { token, url } = await res.json();
                            
                            const modelKey = get('model-select').value;
                            const voice = get('sts-voice') ? get('sts-voice').value : 'Kore';
                            const thinking_level = get('sts-thinking-level') ? get('sts-thinking-level').value : 'minimal';
                            const include_thoughts = get('sts-include-thoughts') ? get('sts-include-thoughts').checked : false;

                            currentGeminiLive = new GeminiLiveClient();
                            if (stsOpt('sts-auto-play')) {
                                currentGeminiLive.rtPlayer = new RealTimeAudioPlayer();
                            }
                            
                            await currentGeminiLive.start(token, url, modelKey, {
                                speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: voice } } },
                                thinkingConfig: { thinkingLevel: thinking_level, includeThoughts: include_thoughts }
                            });
                            mediaRecorder = currentGeminiLive.backupRecorder; // For silence monitor
                            
                            // Bind onstop to trigger the save logic (clicking mic-btn again)
                            mediaRecorder.onstop = () => {
                                if (currentGeminiLive) get('mic-btn').click();
                            };

                            let firstAudio = true;
                            let liveMsgId = 'live-sts-' + Date.now();
                            currentGeminiLive.onMessage = (data) => {
                                if (data.serverContent) {
                                    if (data.serverContent.modelTurn) {
                                        if (firstAudio) {
                                            setStsStatus('Gemini is speaking...', false);
                                            firstAudio = false;
                                        }
                                        // Update UI with transcript so far
                                        const container = get('chat-messages');
                                        let msgEl = document.getElementById(liveMsgId);
                                        if (!msgEl) {
                                            msgEl = document.createElement('div');
                                            msgEl.id = liveMsgId;
                                            msgEl.className = 'flex flex-col gap-2 mb-4 assistant-message bg-slate-800/40 p-3 rounded-lg border border-slate-700/50';
                                            msgEl.innerHTML = `
                                                <div class="text-[10px] text-cyan-400 font-bold uppercase tracking-wider flex items-center gap-2">
                                                    <i class="fas fa-robot"></i> Gemini Live (Streaming)
                                                </div>
                                                <div class="thought-container hidden italic text-slate-400 text-xs border-l-2 border-slate-600 pl-2 my-1"></div>
                                                <div class="message-content text-sm text-slate-100 leading-relaxed"></div>
                                            `;
                                            container.appendChild(msgEl);
                                            container.scrollTop = container.scrollHeight;
                                        }
                                        
                                        const thoughtEl = msgEl.querySelector('.thought-container');
                                        const contentEl = msgEl.querySelector('.message-content');
                                        
                                        if (currentGeminiLive.assistantThought) {
                                            thoughtEl.classList.remove('hidden');
                                            thoughtEl.innerText = currentGeminiLive.assistantThought;
                                        }
                                        contentEl.innerText = currentGeminiLive.assistantText;
                                        container.scrollTop = container.scrollHeight;
                                    }
                                }
                            };
                            
                            setStsStatus('Listening...', true);
                            get('mic-btn').classList.remove('bg-gray-700');
                            get('mic-btn').classList.add('bg-red-600', 'animate-pulse');
                            startMicWaveform(currentGeminiLive.stream);
                            startSilenceMonitor(currentGeminiLive.stream);
                            return;
                        } catch (e) {
                            showToast("Gemini Live connection failed: " + e.message, "error", true);
                            setStsStatus('Error', false);
                            return;
                        }
                    }

                    if (!isStsModel()) {
                        resetMicWaveformBars();
                        setMicRecordingIndicator('録音準備中…', 'processing');
                    }
                    const stream = await navigator.mediaDevices.getUserMedia(getMicCaptureConstraints());
                    mediaRecorder = new MediaRecorder(stream);
                    audioChunks = [];
                    stsCancelPending = false;
                    const recordingStartedInStsMode = isStsModel();
                    mediaRecorder.ondataavailable = (e) => audioChunks.push(e.data);
                    mediaRecorder.onstop = async () => {
                        if (stsCancelPending) {
                            audioChunks = [];
                            get('file-preview').classList.add('hidden');
                            stream.getTracks().forEach(track => track.stop());
                            stopSilenceMonitor();
                            stopMicWaveform();
                            if (!recordingStartedInStsMode) {
                                setMicRecordingIndicator('録音をキャンセルしました', 'idle');
                                micIndicatorHideTimer = setTimeout(() => setMicRecordingIndicator('', 'hidden'), 900);
                            }
                            if (isStsModel()) setStsStatus('Canceled', false);
                            setTimeout(() => { if (isStsModel()) setStsStatus('Tap to speak', false); }, 800);
                            return;
                        }
                        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                        const file = new File([audioBlob], "recording.webm", { type: 'audio/webm' });
                        const fd = new FormData();
                        fd.append('file', file);
                        get('file-preview').classList.remove('hidden');
                        const stsMode = recordingStartedInStsMode;
                        get('file-name').innerText = stsMode ? "Processing voice..." : "Transcribing...";
                        try {
                            if (stsMode) {
                                if (!currentThreadId) {
                                    const r = await apiFetch(CHAT_CONFIG.urls.handleThreads, {
                                        method:'POST',
                                        headers:{'Content-Type':'application/json'},
                                        body: JSON.stringify({ is_temporary: temporaryChatEnabled })
                                    });
                                    const d = await r.json();
                                    currentThreadId = d.id !== null && d.id !== undefined ? String(d.id) : d.id;
                                    setTemporaryChatUiState(!!(d && d.is_temporary));
                                    setCurrentChatHeaderTitle(d && d.title);
                                    applyTemporaryChatRuntimeMeta(d || {});
                                    ensureTemporaryChatHeartbeat(true);
                                    history.pushState({}, '', '/c/' + d.id);
                                    get('welcome-screen').classList.add('hidden');
                                }
                                if (currentThreadId && activeGem) {
                                    threadGemMap[currentThreadId] = activeGem;
                                    pendingGemForNewThread = null;
                                }
                                fd.append('model', get('model-select').value);
                                fd.append('thread_id', currentThreadId);
                                if (get('sts-voice')) fd.append('sts_voice', get('sts-voice').value || '');
                                if (get('sts-speed')) fd.append('sts_speed', get('sts-speed').value || '');
                                if (get('sts-rate-in')) fd.append('sts_rate_in', get('sts-rate-in').value || '');
                                if (get('sts-rate-out')) fd.append('sts_rate_out', get('sts-rate-out').value || '');
                                if (get('sts-thinking-level')) fd.append('sts_thinking_level', get('sts-thinking-level').value || '');
                                if (get('sts-include-thoughts')) fd.append('sts_include_thoughts', get('sts-include-thoughts').checked ? 'true' : '');
                                setStsStatus('Sending audio...', true);
                                const stsRes = await apiFetch("/sts", { method: 'POST', body: fd });
                                if (!stsRes.ok) {
                                    const errData = await stsRes.json().catch(() => ({}));
                                    throw new Error(errData.error || "Speech-to-speech failed");
                                }

                                const reader = stsRes.body.getReader();
                                const decoder = new TextDecoder();
                                let buffer = '';
                                let stsData = null;
                                let rtPlayer = null;
                                if (stsOpt('sts-auto-play')) {
                                    rtPlayer = new RealTimeAudioPlayer();
                                    currentRtPlayer = rtPlayer;
                                }

                                setStsStatus('Gemini is thinking...', true);

                                let firstChunk = true;
                                while (true) {
                                    const { done, value } = await reader.read();
                                    if (done) break;
                                    
                                    buffer += decoder.decode(value, { stream: true });
                                    const lines = buffer.split('\n');
                                    buffer = lines.pop();

                                    for (const line of lines) {
                                        if (!line.trim()) continue;
                                        const chunk = JSON.parse(line);
                                        if (chunk.error) throw new Error(chunk.error);
                                        
                                        if (chunk.audio_delta && rtPlayer) {
                                            if (firstChunk) {
                                                setStsStatus('Gemini is speaking...', false);
                                                firstChunk = false;
                                            }
                                            await rtPlayer.addChunk(chunk.audio_delta);
                                        }
                                        if (chunk.transcript_delta) {
                                            // Optional: Update UI with transcript deltas in real-time
                                        }
                                        if (chunk.final) {
                                            stsData = chunk;
                                        }
                                    }
                                }

                                if (stsData && stsData.audio_url) {
                                    if (stsOpt('sts-auto-restart') && isStsModel()) {
                                        // Wait a bit for rtPlayer to finish if it's still playing
                                        setTimeout(() => {
                                            setStsStatus('Listening...', true);
                                            get('mic-btn').click();
                                        }, 500); 
                                    } else {
                                        setStsStatus('Tap to speak', false);
                                    }
                                    await loadMessages(currentThreadId);
                                } else if (!stsData && !rtPlayer) {
                                     // Fallback for sync responses if any
                                }
                            } else {
                                const micModeEl = get('set-mic-transcribe-mode');
                                const llmMicMode = !!(micModeEl && micModeEl.value === 'llm');
                                if (llmMicMode && !supportsAudioInputModel()) {
                                    showToast("現在のモデルはLLM音声文字起こし（音声入力）に対応していません", "error", true);
                                    return;
                                }
                                fd.append('llm_model', get('model-select') ? (get('model-select').value || '') : '');
                                const transRes = await apiFetch(CHAT_CONFIG.urls.transcribe, {
                                    method: 'POST',
                                    body: fd
                                });
                                const transData = await transRes.json();
                                if (transData.transcript) {
                                    const input = get('prompt-input');
                                    input.value += (input.value ? " " : "") + transData.transcript;
                                    input.style.height = 'auto';
                                    input.style.height = input.scrollHeight + 'px';
                                } else {
                                    showToast(transData.error || "Transcription failed", "error", true);
                                }
                            }
                        } catch (err) {
                            showToast("Audio processing error: " + err.message, "error", true);
                        } finally {
                            get('file-preview').classList.add('hidden');
                            stream.getTracks().forEach(track => track.stop());
                            stopSilenceMonitor();
                            stopMicWaveform();
                            if (!stsMode) setMicRecordingIndicator('', 'hidden');
                            if (stsMode) setStsStatus('Tap to speak', false);
                        }
                    };
                    mediaRecorder.start();
                    get('mic-btn').classList.remove('bg-gray-700');
                    get('mic-btn').classList.add('bg-red-600', 'animate-pulse');
                    if (!isStsModel()) {
                        setMicRecordingIndicator('録音中…', 'recording');
                        startMicWaveform(stream);
                    }
                    startSilenceMonitor(stream);
                    if (isStsModel()) setStsStatus('Recording... Tap to stop', true);
                } catch (err) {
                    stopMicWaveform();
                    if (!isStsModel()) setMicRecordingIndicator('', 'hidden');
                    alert("Microphone access denied or not available.");
                }
            };

        window.updateLibSelectionUi = function () {
            if (!lib.selected) lib.selected = new Set();
            const count = lib.selected.size;
            const delBtn = get('lib-del-btn');
            const downloadBtn = get('lib-download-btn');
            const attachBtn = get('lib-attach-btn');
            const renameBtn = get('lib-rename-btn');
            if (delBtn) {
                delBtn.disabled = count === 0;
                delBtn.innerText = count ? `削除 (${count})` : "削除";
            }
            if (downloadBtn) {
                downloadBtn.disabled = count === 0;
                downloadBtn.innerText = count ? `ダウンロード (${count})` : "ダウンロード";
            }
            if (attachBtn) {
                attachBtn.disabled = count === 0;
                attachBtn.innerText = count ? `添付 (${count})` : "添付";
            }
            if (renameBtn) {
                renameBtn.disabled = count !== 1;
                renameBtn.innerText = "名前変更";
            }
            if (lib.modal) {
                const isMobile = window.matchMedia('(max-width: 768px)').matches;
                lib.modal.classList.toggle('lib-selecting', isMobile && count > 0);
            }
        };
        function setLibAttachMode(flag) {
            lib.attachMode = !!flag;
        }
        const openLibModal = (attachMode = false) => {
            setLibAttachMode(attachMode);
            showModal('lib-modal');
            loadLibraryFiles();
            if (location.pathname !== '/library') {
                history.pushState({ modal: 'library' }, '', '/library');
            }
        };
        window.closeLibModal = (skipHistory = false) => {
            hideModal('lib-modal');
            if (!skipHistory && location.pathname === '/library') {
                history.back();
            }
        };
        get('lib-btn').onclick = () => openLibModal(false);
        get('lib-del-btn').onclick = deleteSelectedFiles;
        if (get('lib-download-btn')) get('lib-download-btn').onclick = () => downloadSelectedLibraryFiles();
        if (get('lib-attach-btn')) get('lib-attach-btn').onclick = () => attachSelectedLibraryFiles();
        if (get('lib-rename-btn')) get('lib-rename-btn').onclick = () => renameSelectedLibraryFile();
        if (get('upload-lib-btn')) get('upload-lib-btn').onclick = () => openLibModal(true);
        if (get('lib-search')) {
            get('lib-search').oninput = () => {
                lib.searchQuery = (get('lib-search').value || '').trim();
                renderLibraryGrid();
            };
        }
        if (get('lib-sort')) {
            const storedSort = localStorage.getItem(LIB_SORT_KEY) || 'newest';
            get('lib-sort').value = storedSort;
            get('lib-sort').onchange = () => {
                const v = get('lib-sort').value || 'newest';
                localStorage.setItem(LIB_SORT_KEY, v);
                renderLibraryGrid();
            };
        }
            if (get('add-gem-fixed-prompt-row')) {
                get('add-gem-fixed-prompt-row').onclick = () => addGemFixedPromptRow();
            }

        const openGemModal = () => {
            editingGemUuid = null;
            get('gem-modal-title').innerHTML = `<i class="fas fa-gem text-blue-500 mr-2"></i>Create New Gem`;
            get('save-gem-btn').innerText = "Create Gem";
            showModal('gem-modal');
            get('gem-name').value=''; get('gem-desc').value=''; get('gem-inst').value=''; get('gem-default-model').value='';
            if (get('gem-fixed-prompts-container')) get('gem-fixed-prompts-container').innerHTML = '';
            if (location.pathname !== '/gem') {
                history.pushState({ modal: 'gem' }, '', '/gem');
            }
        };
        window.closeGemModal = (skipHistory = false) => {
            hideModal('gem-modal');
            if (!skipHistory && location.pathname === '/gem') {
                history.back();
            }
        };
        get('add-gem-btn').onclick = () => openGemModal();
            get('save-gem-btn').onclick = async () => {
                const name = get('gem-name').value;
                const desc = get('gem-desc').value;
                const inst = get('gem-inst').value;
                const fixed_prompts = collectGemFixedPrompts();

                if(name && inst) {
                    const method = editingGemUuid ? 'PUT' : 'POST';
                    const url = editingGemUuid ? `/api/gems/${editingGemUuid}` : CHAT_CONFIG.urls.handleGems;
                    await apiFetch(url, {
                        method: method,
                        headers:{'Content-Type':'application/json'},
                        body:JSON.stringify({name, description:desc, instruction:inst, fixed_prompts, default_model: get('gem-default-model').value || null})
                    }); 
                    window.closeGemModal(); 
                    loadGems(); 
                    if (editingGemUuid && activeGem && activeGem.uuid === editingGemUuid) {
                       activeGem.name = name;
                       activeGem.instruction = inst;
                       activeGem.fixed_prompts = fixed_prompts;
                       applyActiveGem(activeGem);
                    }
                } else alert("Name and Instruction are required.");
            };
            document.addEventListener('click', function(e) { 
                if (e.target.closest('.edit-btn')) { 
                    const btn = e.target.closest('.edit-btn'); 
                    const id = btn.getAttribute('data-id'); 
                    beginEditMessage(id);
                } 
                if (e.target.closest('.code-toggle')) {
                    const btn = e.target.closest('.code-toggle');
                    const wrapper = btn.closest('.code-wrapper');
                    if (!wrapper) return;
                    const isCollapsed = wrapper.classList.toggle('collapsed');
                    wrapper.setAttribute('data-collapsed', isCollapsed ? 'true' : 'false');
                    btn.setAttribute('aria-expanded', isCollapsed ? 'false' : 'true');
                    btn.innerHTML = isCollapsed
                        ? '<i class="fas fa-chevron-down"></i> Expand'
                        : '<i class="fas fa-chevron-up"></i> Collapse';
                }
                if (e.target.closest('.download-btn')) {
                    const btn = e.target.closest('.download-btn');
                    const codeEnc = btn.getAttribute('data-code');
                    const lang = (btn.getAttribute('data-lang') || 'txt').toLowerCase();
                    if (codeEnc) {
                        try {
                            const code = decodeURIComponent(codeEnc);
                            const blob = new Blob([code], { type: 'text/plain' });
                            const url = URL.createObjectURL(blob);
                            const a = document.createElement('a');
                            a.href = url;
                            const map = {
                                python: 'py', javascript: 'js', typescript: 'ts', markdown: 'md',
                                html: 'html', css: 'css', json: 'json', xml: 'xml', sql: 'sql',
                                bash: 'sh', sh: 'sh', shell: 'sh', zsh: 'sh',
                                c: 'c', cpp: 'cpp', csharp: 'cs', cs: 'cs',
                                java: 'java', kotlin: 'kt', swift: 'swift',
                                go: 'go', rust: 'rs', ruby: 'rb', php: 'php',
                                perl: 'pl', lua: 'lua', r: 'r', matlab: 'm',
                                yaml: 'yaml', yml: 'yaml', toml: 'toml', ini: 'ini',
                                plaintext: 'txt', text: 'txt'
                            };
                            let ext = map[lang] || lang;
                            if (lang.length > 8 || /[^a-z0-9]/.test(lang)) ext = 'txt';
                            let filename = `code.${ext}`;
                            if (lang === 'dockerfile') filename = 'Dockerfile';
                            if (lang === 'makefile') filename = 'Makefile';
                            a.download = filename;
                            document.body.appendChild(a);
                            a.click();
                            document.body.removeChild(a);
                            URL.revokeObjectURL(url);
                        } catch (err) {
                            console.error('Download failed', err);
                        }
                    }
                }
                if (e.target.closest('.copy-btn')) {
                    const btn = e.target.closest('.copy-btn');
                    const code = btn.getAttribute('data-code');
                    if (code) window.copyCode(btn, code);
                }
                if (e.target.closest('.html-preview-btn')) {
                    const btn = e.target.closest('.html-preview-btn');
                    const code = btn.getAttribute('data-code');
                    if (code) openHtmlCodePreview(code);
                }
                if (e.target.closest('.canvas-preview-btn')) {
                    const btn = e.target.closest('.canvas-preview-btn');
                    previewCanvasCodeFromButton(btn);
                }
            });
            document.querySelectorAll('.modal-overlay').forEach((overlay) => {
                overlay.addEventListener('click', (e) => {
                    if (e.target === overlay) closeModalById(overlay.id);
                });
            });

            // Initial Load Logic for Permalink
            if(currentThreadId) { loadMessages(currentThreadId); }
            else { schedulePromptTokenEstimate(true); }
        });
        
        function updateFilePreview() {
            const preview = get('file-preview');
            const nameEl = get('file-name');
            const progressContainer = get('upload-total-progress');
            const progressBar = get('upload-total-progress-bar');
            const thumbContainer = get('file-preview-thumbs');

            const modalStatusText = get('upload-modal-status-text');
            const modalProgressContainer = get('upload-modal-total-progress');
            const modalProgressBar = get('upload-modal-total-progress-bar');

            if (!preview || !nameEl) return;
            
            // Sync Thumbnails from the upload list (modal)
            if (thumbContainer) {
                const uploadRows = document.querySelectorAll('#upload-list .upload-row');
                
                // Rebuild thumbnail list
                thumbContainer.innerHTML = '';
                uploadRows.forEach((row, index) => {
                    const localUrl = row.getAttribute('data-local-url');
                    const filename = row.getAttribute('data-filename');
                    // Check if it's meant to be an image
                    const isImage = row.querySelector('img.upload-preview') !== null;
                    
                    let el;
                    if (isImage) {
                        let src = localUrl;
                        if (!src && filename) {
                            const displayPath = filename.replace(/^\d+\//, '');
                            src = buildAttachmentPreviewUrl(displayPath);
                        }
                        
                        if (src) {
                            el = document.createElement('img');
                            el.src = src;
                            el.className = 'thumb-item shadow-sm';
                            el.dataset.viewerSrc = src;
                            el.dataset.viewerFilename = filename || src.split('/').pop();
                            el.onclick = function(e) {
                                e.preventDefault();
                                openImageViewer(this.dataset.viewerSrc, '.thumb-item');
                            };
                            el.onerror = function() {
                                // Fallback if image fails to load (e.g. revoked URL)
                                this.parentElement.replaceChild(createFileThumb('ERR'), this);
                            };
                        }
                    }
                    
                    if (!el) {
                        el = createFileThumb('FILE');
                    }
                    
                    el.style.animationDelay = `${index * 32}ms`;
                    thumbContainer.appendChild(el);
                });
                
                if (uploadRows.length > 0) thumbContainer.classList.remove('hidden');
                else thumbContainer.classList.add('hidden');
            }

            function createFileThumb(text) {
                const el = document.createElement('div');
                el.className = 'thumb-item bg-gray-800 flex items-center justify-center text-gray-500 text-[9px] shadow-sm font-bold';
                el.innerText = text;
                return el;
            }
            
            const pendingUrls = collectImageUrlsForSend();
            const total = uploadProgressState.total;
            const completed = uploadProgressState.completed;
            const active = uploadProgressState.active;

            if (total === 0) {
                preview.classList.add('hidden');
                if (progressContainer) progressContainer.classList.add('hidden');
                if (modalProgressContainer) modalProgressContainer.classList.add('hidden');
                if (thumbContainer) thumbContainer.classList.add('hidden');
            }

            const sendBtn = get('send-btn');
            const micBtn = get('mic-btn');
            const maskBtn = get('mask-btn');
            const isAiThinking = isStopMode;
            if (active > 0) {
                if (sendBtn) sendBtn.disabled = true;
                if (micBtn) micBtn.disabled = true;
                if (maskBtn) maskBtn.disabled = true;
            } else if (!isAiThinking) {
                if (sendBtn) sendBtn.disabled = false;
                if (micBtn) micBtn.disabled = false;
                if (maskBtn) maskBtn.disabled = false;
            }

            if (active > 0) {
                const statusStr = `Preparing... (${completed}/${total})`;
                preview.classList.remove('hidden');
                nameEl.innerText = statusStr;
                if (modalStatusText) modalStatusText.innerText = `(${completed}/${total})`;

                let sumPct = completed * 100;
                let countActive = 0;
                for (let id in uploadProgressState.perFilePct) {
                    sumPct += uploadProgressState.perFilePct[id];
                    countActive++;
                }
                const pct = total > 0 ? (sumPct / (total * 100)) * 100 : 0;
                const pctStr = `${Math.min(100, pct)}%`;

                if (progressContainer && progressBar) {
                    progressContainer.classList.remove('hidden');
                    progressBar.style.width = pctStr;
                }
                if (modalProgressContainer && modalProgressBar) {
                    modalProgressContainer.classList.remove('hidden');
                    modalProgressBar.style.width = pctStr;
                }
            } else {
                if (modalStatusText) modalStatusText.innerText = '';
                if (modalProgressContainer) modalProgressContainer.classList.add('hidden');

                if (pendingUrls.length > 0) {
                    preview.classList.remove('hidden');
                    nameEl.innerText = `${pendingUrls.length} files ready`;
                    if (progressContainer) progressContainer.classList.add('hidden');
                } else {
                    preview.classList.add('hidden');
                    nameEl.innerText = '';
                    if (progressContainer) progressContainer.classList.add('hidden');
                }
            }
            schedulePromptTokenEstimate();
        }
        function updateMaskPreview() {
            const preview = get('mask-preview');
            const nameEl = get('mask-name');
            if (!preview || !nameEl) return;
            if (currentMaskImage) {
                preview.classList.remove('hidden');
                nameEl.innerText = `Mask: ${currentMaskImage.split('/').pop()}`;
            } else {
                preview.classList.add('hidden');
                nameEl.innerText = '';
            }
        }
        const markerToolHints = {
            draw: 'マーカー（色・透明度変更可） / 二本指で拡大',
            mosaic: 'ドラッグで範囲モザイク（複数追加可） / 二本指で拡大',
            crop: '外側をドラッグして切り取り / 二本指で拡大'
        };
        function normalizeMarkerHexColor(color) {
            const v = String(color || '').trim().toLowerCase();
            if (/^#[0-9a-f]{6}$/.test(v)) return v;
            if (/^#[0-9a-f]{3}$/.test(v)) {
                const r = v[1], g = v[2], b = v[3];
                return `#${r}${r}${g}${g}${b}${b}`;
            }
            return '#facc15';
        }
        function markerHexToRgb(hex) {
            const normalized = normalizeMarkerHexColor(hex);
            return {
                r: parseInt(normalized.slice(1, 3), 16),
                g: parseInt(normalized.slice(3, 5), 16),
                b: parseInt(normalized.slice(5, 7), 16)
            };
        }
        function clampMarkerOpacityPct(value, fallbackPct = 60) {
            const parsed = Number(value);
            const pct = Number.isFinite(parsed) ? parsed : fallbackPct;
            return Math.max(MARKER_OPACITY_MIN_PCT, Math.min(MARKER_OPACITY_MAX_PCT, pct));
        }
        function formatMarkerOpacityPct(pctValue) {
            const rounded = Math.round(clampMarkerOpacityPct(pctValue) * 10) / 10;
            return Number.isInteger(rounded) ? String(rounded) : String(rounded).replace(/\.0$/, '');
        }
        function getMarkerStrokeStyle() {
            const rgb = markerHexToRgb(markerState.colorHex);
            const alpha = Math.max(MARKER_OPACITY_MIN_ALPHA, Math.min(1, Number(markerState.opacity) || 0.6));
            return `rgba(${rgb.r},${rgb.g},${rgb.b},${alpha})`;
        }
        function syncMarkerColorControls() {
            const colorHex = normalizeMarkerHexColor(markerState.colorHex);
            markerState.colorHex = colorHex;
            const alpha = Math.max(MARKER_OPACITY_MIN_ALPHA, Math.min(1, Number(markerState.opacity) || 0.6));
            markerState.opacity = alpha;
            const opacityPct = alpha * 100;
            const opacityPctText = formatMarkerOpacityPct(opacityPct);
            const colorPicker = get('marker-color-picker');
            if (colorPicker && colorPicker.value !== colorHex) colorPicker.value = colorHex;
            const opacityInput = get('marker-opacity');
            if (opacityInput && opacityInput.value !== opacityPctText) opacityInput.value = opacityPctText;
            const opacityNumber = get('marker-opacity-number');
            if (opacityNumber && opacityNumber.value !== opacityPctText) opacityNumber.value = opacityPctText;
            const opacityText = get('marker-opacity-value');
            if (opacityText) opacityText.textContent = `${opacityPctText}%`;
            const chips = document.querySelectorAll('#marker-toolbar .marker-color-chip[data-marker-color]');
            chips.forEach((chip) => {
                const v = normalizeMarkerHexColor(chip.getAttribute('data-marker-color'));
                chip.classList.toggle('active', v === colorHex);
            });
        }
        function setMarkerColor(colorHex) {
            markerState.colorHex = normalizeMarkerHexColor(colorHex);
            syncMarkerColorControls();
        }
        function setMarkerOpacity(pctValue) {
            const pct = clampMarkerOpacityPct(pctValue, 60);
            markerState.opacity = pct / 100;
            syncMarkerColorControls();
        }
        function setMarkerMode(mode) {
            markerState.mode = mode;
            if (mode !== 'mosaic') markerState.mosaicPreviewRect = null;
            const btnDraw = get('marker-tool-draw');
            const btnMosaic = get('marker-tool-mosaic');
            const btnCrop = get('marker-tool-crop');
            if (btnDraw) btnDraw.classList.toggle('active', mode === 'draw');
            if (btnMosaic) btnMosaic.classList.toggle('active', mode === 'mosaic');
            if (btnCrop) btnCrop.classList.toggle('active', mode === 'crop');
            const hint = get('marker-tool-hint');
            if (hint) hint.textContent = markerToolHints[mode] || '';
            const cropReset = get('marker-crop-reset');
            if (cropReset) cropReset.classList.toggle('hidden', mode !== 'crop');
            const drawCanvas = get('marker-canvas');
            if (drawCanvas) drawCanvas.style.pointerEvents = (mode === 'crop') ? 'none' : 'auto';
            const cropCanvas = get('marker-crop-canvas');
            if (cropCanvas) cropCanvas.style.pointerEvents = (mode === 'crop') ? 'auto' : 'none';
            if (mode === 'crop' && (!markerState.cropRect || markerState.cropRect.w <= 1 || markerState.cropRect.h <= 1)) {
                resetCropRectToFull();
            }
            renderCropOverlay();
        }
        function clearCropRect() {
            resetCropRectToFull();
            renderCropOverlay();
        }
        function resetCropRectToFull() {
            const cropCanvas = get('marker-crop-canvas');
            if (!cropCanvas) return;
            const w = Math.max(1, cropCanvas.width || 0);
            const h = Math.max(1, cropCanvas.height || 0);
            if (w <= 1 || h <= 1) return;
            markerState.cropRect = { x: 0, y: 0, w, h };
        }
        function clampMarkerViewOffset() {
            markerView.scale = Math.min(markerView.maxScale, Math.max(markerView.minScale, Number(markerView.scale) || 1));
            if (markerView.scale <= markerView.minScale + 0.0001) {
                markerView.offsetX = 0;
                markerView.offsetY = 0;
                return;
            }
            const stage = get('marker-stage');
            const viewport = get('marker-viewport');
            if (!stage || !viewport) return;
            const stageW = Math.max(1, stage.clientWidth || 0);
            const stageH = Math.max(1, stage.clientHeight || 0);
            const baseW = Math.max(1, viewport.offsetWidth || viewport.clientWidth || 0);
            const baseH = Math.max(1, viewport.offsetHeight || viewport.clientHeight || 0);
            if (stageW <= 1 || stageH <= 1 || baseW <= 1 || baseH <= 1) return;
            const baseLeft = (stageW - baseW) / 2;
            const baseTop = (stageH - baseH) / 2;
            const scaledW = baseW * markerView.scale;
            const scaledH = baseH * markerView.scale;
            const minVisibleX = Math.min(stageW * 0.45, Math.max(24, stageW * 0.12));
            const minVisibleY = Math.min(stageH * 0.45, Math.max(24, stageH * 0.12));
            const minOffsetX = minVisibleX - baseLeft - scaledW;
            const maxOffsetX = stageW - minVisibleX - baseLeft;
            const minOffsetY = minVisibleY - baseTop - scaledH;
            const maxOffsetY = stageH - minVisibleY - baseTop;
            const clampOffset = (value, min, max) => {
                if (!Number.isFinite(value)) return 0;
                if (min > max) return (min + max) / 2;
                return Math.min(max, Math.max(min, value));
            };
            markerView.offsetX = clampOffset(markerView.offsetX, minOffsetX, maxOffsetX);
            markerView.offsetY = clampOffset(markerView.offsetY, minOffsetY, maxOffsetY);
        }
        function applyMarkerTransform() {
            const viewport = get('marker-viewport');
            if (!viewport) return;
            clampMarkerViewOffset();
            viewport.style.transform = `translate(${markerView.offsetX}px, ${markerView.offsetY}px) scale(${markerView.scale})`;
        }
        function resetMarkerTransform() {
            markerView.scale = 1;
            markerView.offsetX = 0;
            markerView.offsetY = 0;
            applyMarkerTransform();
        }
        function getRowMarkerKey(row) {
            if (!row) return null;
            return row.dataset.uploadId || row.getAttribute('data-filename') || null;
        }
        function setRowMarkerState(row, applied) {
            const key = getRowMarkerKey(row);
            if (key) {
                if (applied) markerAppliedUploads.add(key);
                else markerAppliedUploads.delete(key);
            }
            const tag = row ? row.querySelector('.upload-marker-tag') : null;
            if (tag) tag.classList.toggle('hidden', !applied);
        }
        function hasMarkerHint() {
            return markerAppliedUploads.size > 0;
        }
        function normalizeAttachmentSource(source) {
            const raw = String(source || '').trim().toLowerCase();
            if (raw === 'library' || raw === 'lib') return 'library';
            if (raw === 'upload' || raw === 'uploaded') return 'upload';
            return 'unknown';
        }
        function normalizeAttachmentDisplayName(name) {
            if (name === null || name === undefined) return '';
            let v = String(name).replace(/\u0000/g, '');
            v = v.replace(/\r/g, ' ').replace(/\n/g, ' ').replace(/\t/g, ' ');
            v = v.trim();
            if (!v) return '';
            v = v.split('/').pop().split('\\').pop().trim();
            v = v.replace(/\s{2,}/g, ' ');
            v = v.replace(/[<>:"/\\|?*]+/g, '_');
            if (!v || v === '.' || v === '..') return '';
            if (v.length > 180) v = v.slice(0, 180).trim();
            return v;
        }
        function defaultAttachmentDisplayName(path) {
            const norm = normalizeAttachmentPath(path);
            if (!norm) return '';
            return norm.split('/').pop() || norm;
        }
        function setAttachmentNameForPath(path, displayName) {
            const norm = normalizeAttachmentPath(path);
            if (!norm) return;
            const next = normalizeAttachmentDisplayName(displayName) || defaultAttachmentDisplayName(norm);
            if (!next) return;
            attachmentNameByPath.set(norm, next);
        }
        function getAttachmentNameForPath(path) {
            const norm = normalizeAttachmentPath(path);
            if (!norm) return '';
            const named = normalizeAttachmentDisplayName(attachmentNameByPath.get(norm));
            if (named) return named;
            return defaultAttachmentDisplayName(norm);
        }
        function setRowAttachmentName(row, displayName) {
            if (!row) return;
            const next = normalizeAttachmentDisplayName(displayName) || getAttachmentNameForPath(row.getAttribute('data-filename')) || 'file';
            row.dataset.displayName = next;
            const nameEl = row.querySelector('.truncate');
            if (nameEl) nameEl.textContent = next;
            const path = row.getAttribute('data-filename');
            if (path) setAttachmentNameForPath(path, next);
        }
        function isRowAttachmentNameCustomized(row) {
            return !!(row && row.dataset.sendNameCustomized === '1');
        }
        function setRowAttachmentNameCustomized(row, customized) {
            if (!row) return;
            row.dataset.sendNameCustomized = customized ? '1' : '';
        }
        function getRowDefaultAttachmentName(row) {
            if (!row) return 'file';
            const path = row.getAttribute('data-filename');
            if (path) return defaultAttachmentDisplayName(path) || 'file';
            const localDefault = normalizeAttachmentDisplayName(row.dataset.defaultDisplayName);
            if (localDefault) return localDefault;
            return normalizeAttachmentDisplayName(row.dataset.displayName) || 'file';
        }
        function promptRowAttachmentName(row) {
            if (!row) return;
            const currentName = getRowAttachmentName(row) || getRowDefaultAttachmentName(row) || 'file';
            const input = prompt('送信時のファイル名を入力してください（空欄でデフォルトに戻す）', currentName);
            if (input === null) return;
            const next = normalizeAttachmentDisplayName(input);
            if (!next) {
                const fallback = getRowDefaultAttachmentName(row);
                setRowAttachmentName(row, fallback);
                setRowAttachmentNameCustomized(row, false);
                showToast('送信名をデフォルトに戻しました', 'success');
                return;
            }
            setRowAttachmentName(row, next);
            setRowAttachmentNameCustomized(row, true);
            showToast('送信名を更新しました', 'success');
        }
        function getRowAttachmentName(row) {
            if (!row) return '';
            const path = row.getAttribute('data-filename');
            const fromPath = getAttachmentNameForPath(path);
            if (fromPath) return fromPath;
            const fromRow = normalizeAttachmentDisplayName(row.dataset.displayName);
            if (fromRow) return fromRow;
            const nameEl = row.querySelector('.truncate');
            const fromText = normalizeAttachmentDisplayName(nameEl ? nameEl.textContent : '');
            if (fromText) return fromText;
            return getAttachmentNameForPath(path);
        }
        function setAttachmentSourceForPath(path, source) {
            const norm = normalizeAttachmentPath(path);
            if (!norm) return;
            const src = normalizeAttachmentSource(source);
            if (src === 'unknown') return;
            attachmentSourceByPath.set(norm, src);
        }
        function getAttachmentSourceForPath(path) {
            const norm = normalizeAttachmentPath(path);
            if (!norm) return 'unknown';
            return normalizeAttachmentSource(attachmentSourceByPath.get(norm));
        }
        function setRowAttachmentSource(row, source) {
            if (!row) return;
            const src = normalizeAttachmentSource(source);
            row.dataset.fileSource = src;
            const path = row.getAttribute('data-filename');
            if (path) setAttachmentSourceForPath(path, src);
        }
        function getRowAttachmentSource(row) {
            if (!row) return 'unknown';
            const fromRow = normalizeAttachmentSource(row.dataset.fileSource);
            if (fromRow !== 'unknown') return fromRow;
            const path = row.getAttribute('data-filename');
            return getAttachmentSourceForPath(path);
        }
        function getRowOriginalAttachmentSource(row) {
            if (!row) return 'unknown';
            const fromRow = normalizeAttachmentSource(row.dataset.originalSource);
            if (fromRow !== 'unknown') return fromRow;
            const path = row.getAttribute('data-original-filename');
            return getAttachmentSourceForPath(path);
        }
        function prepareMarkerBaseCanvas(img, width, height) {
            const base = document.createElement('canvas');
            base.width = width;
            base.height = height;
            const bctx = base.getContext('2d');
            if (bctx) {
                bctx.drawImage(img, 0, 0, width, height);
                markerState.baseImageData = bctx.getImageData(0, 0, width, height);
                markerState.baseCanvas = base;
            } else {
                markerState.baseImageData = null;
                markerState.baseCanvas = null;
            }
        }
        function renderCropOverlay() {
            const cropCanvas = get('marker-crop-canvas');
            if (!cropCanvas) return;
            const ctx = cropCanvas.getContext('2d');
            if (!ctx) return;
            ctx.clearRect(0, 0, cropCanvas.width, cropCanvas.height);
            const drawRect = (rect, stroke, fill = null, dashed = false) => {
                if (!rect) return;
                const x = Math.max(0, rect.x);
                const y = Math.max(0, rect.y);
                const w = Math.max(1, rect.w);
                const h = Math.max(1, rect.h);
                if (fill) {
                    ctx.fillStyle = fill;
                    ctx.fillRect(x, y, w, h);
                }
                ctx.save();
                if (dashed) ctx.setLineDash([6, 4]);
                ctx.strokeStyle = stroke;
                ctx.lineWidth = 2;
                ctx.strokeRect(x + 0.5, y + 0.5, Math.max(1, w - 1), Math.max(1, h - 1));
                ctx.restore();
            };

            const rect = markerState.cropRect;
            const isFull = rect && rect.x === 0 && rect.y === 0 && Math.abs(rect.w - cropCanvas.width) < 1 && Math.abs(rect.h - cropCanvas.height) < 1;

            if (rect && (markerState.mode === 'crop' || !isFull)) {
                ctx.fillStyle = 'rgba(0,0,0,0.35)';
                ctx.fillRect(0, 0, cropCanvas.width, cropCanvas.height);
                const x = Math.max(0, rect.x);
                const y = Math.max(0, rect.y);
                const w = Math.max(1, rect.w);
                const h = Math.max(1, rect.h);
                ctx.clearRect(x, y, w, h);
                if (markerState.mode === 'crop') {
                    drawRect(rect, 'rgba(250,204,21,0.9)');
                } else {
                    drawRect(rect, 'rgba(250,204,21,0.4)');
                }
            }

            if (markerState.mode === 'crop') {
                return;
            }
            if (markerState.mode !== 'mosaic') return;
            const rects = Array.isArray(markerState.mosaicRects) ? markerState.mosaicRects : [];
            rects.forEach((rect) => drawRect(rect, 'rgba(250,204,21,0.9)', 'rgba(250,204,21,0.10)'));
            if (markerState.mosaicPreviewRect) {
                drawRect(markerState.mosaicPreviewRect, 'rgba(56,189,248,0.95)', 'rgba(56,189,248,0.14)', true);
            }
        }
        function collectImageUrlsForSend() {
            return collectAttachmentItemsForSend().map((it) => it.path);
        }
        function collectAttachmentItemsForSend() {
            const items = [];
            const indexByPath = new Map();
            const pushItem = (path, source, displayName) => {
                const norm = normalizeAttachmentPath(path);
                if (!norm) return;
                const src = normalizeAttachmentSource(source);
                const name = normalizeAttachmentDisplayName(displayName) || getAttachmentNameForPath(norm);
                const idx = indexByPath.get(norm);
                if (idx === undefined) {
                    const nextIndex = items.length;
                    indexByPath.set(norm, nextIndex);
                    items.push({ path: norm, source: src, name });
                    return;
                }
                const prev = items[idx];
                if (!prev) return;
                const prevSrc = normalizeAttachmentSource(prev.source);
                if (prevSrc === 'unknown' && src !== 'unknown') {
                    prev.source = src;
                } else if (prevSrc === 'library' && src === 'upload') {
                    prev.source = src;
                }
                if (!normalizeAttachmentDisplayName(prev.name) && name) {
                    prev.name = name;
                }
            };

            const list = get('upload-list');
            if (list) {
                list.querySelectorAll('[data-filename]').forEach((row) => {
                    const main = row.getAttribute('data-filename');
                    pushItem(main, getRowAttachmentSource(row), getRowAttachmentName(row));
                    const original = row.getAttribute('data-original-filename');
                    const attachOriginal = row.dataset.attachOriginal === '1';
                    if (attachOriginal) {
                        pushItem(original, getRowOriginalAttachmentSource(row), getAttachmentNameForPath(original));
                    }
                });
            }
            if (currentImageUrls && currentImageUrls.length) {
                currentImageUrls.forEach((u) => {
                    pushItem(u, getAttachmentSourceForPath(u), getAttachmentNameForPath(u));
                });
            }
            return items;
        }
        function collectUploadedImageUrlsForSend() {
            return collectAttachmentItemsForSend()
                .filter((it) => normalizeAttachmentSource(it.source) === 'upload')
                .map((it) => it.path);
        }
        function purgeUnsupportedAttachments(notify = true) {
            const support = getModelMediaSupport(get('model-select').value);
            let removedAudio = 0;
            let removedVideo = 0;
            if (Array.isArray(currentImageUrls) && currentImageUrls.length) {
                const filtered = [];
                currentImageUrls.forEach((fp) => {
                    const normalized = normalizeAttachmentPath(fp);
                    if (!normalized) return;
                    const isAudio = isAudioPath(normalized);
                    const isVideo = isVideoPath(normalized);
                    if ((isAudio && !support.audio) || (isVideo && !support.video)) {
                        if (isAudio) removedAudio += 1;
                        if (isVideo) removedVideo += 1;
                        return;
                    }
                    filtered.push(normalized);
                });
                if (filtered.length !== currentImageUrls.length) {
                    currentImageUrls = filtered;
                }
            }
            const list = get('upload-list');
            if (list) {
                list.querySelectorAll('[data-filename]').forEach((row) => {
                    const fp = row.getAttribute('data-filename');
                    if (!fp) return;
                    if (!currentImageUrls.includes(fp) && (isAudioPath(fp) || isVideoPath(fp))) {
                        setRowMarkerState(row, false);
                        row.remove();
                    }
                });
                if (list.children.length === 0) {
                    list.innerHTML = '<div class="text-xs text-gray-500">まだアップロードがありません。</div>';
                }
            }
            updateFilePreview();
            if (notify && (removedAudio || removedVideo)) {
                const parts = [];
                if (removedAudio) parts.push(`${removedAudio}件の音声`);
                if (removedVideo) parts.push(`${removedVideo}件の動画`);
                showToast(`このモデルは${parts.join('・')}入力に非対応のため削除しました`, "error", true);
            }
        }
        function getRowImageSource(row) {
            if (!row) return '';
            const localUrl = row.getAttribute('data-local-url');
            if (localUrl) return localUrl;
            const filepath = row.getAttribute('data-filename');
            if (filepath) return buildFileUrl(filepath);
            return '';
        }
        function buildFileUrl(path) {
            const norm = normalizeAttachmentPath(path);
            if (!norm) return '';
            return FILE_BASE_URL + norm;
        }
        function buildAttachmentPreviewUrl(path) {
            const norm = normalizeAttachmentPath(path);
            if (!norm) return '';
            if (isImagePath(norm)) return FILE_THUMB_BASE_URL + norm;
            return FILE_BASE_URL + norm;
        }
        window.closeMarkerModal = (skipHistory = false) => {
            hideModal('marker-modal');
            if (!skipHistory && location.pathname === '/edit-image') {
                history.back();
            }
        };
        function openMarkerModalForRow(row) {
            const src = getRowImageSource(row);
            if (!src) {
                showToast("画像が読み込めませんでした", "error", true);
                return;
            }
            markerState.row = row;
            const nameEl = row ? row.querySelector('.truncate') : null;
            markerState.filename = nameEl ? nameEl.textContent.trim() : 'image.png';
            markerState.hasStroke = false;
            markerState.history = [];
            markerState.naturalWidth = 0;
            markerState.naturalHeight = 0;
            markerState.cropRect = null;
            markerState.mosaicRects = [];
            markerState.mosaicPreviewRect = null;
            markerState.baseCanvas = null;
            markerState.baseImageData = null;
            setMarkerMode('draw');
            const attachOriginal = get('marker-attach-original');
            if (attachOriginal) {
                attachOriginal.checked = row.dataset.attachOriginal === '1';
            }
            const img = get('marker-image');
            const canvas = get('marker-canvas');
            const cropCanvas = get('marker-crop-canvas');
            if (canvas) {
                const ctx = canvas.getContext('2d');
                if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
            }
            if (cropCanvas) {
                const cctx = cropCanvas.getContext('2d');
                if (cctx) cctx.clearRect(0, 0, cropCanvas.width, cropCanvas.height);
            }
            resetMarkerTransform();
            showModal('marker-modal');
            if (location.pathname !== '/edit-image') {
                history.pushState({ modal: 'marker' }, '', '/edit-image');
            }
            if (img) {
                img.onload = () => {
                    const stage = get('marker-stage');
                    if (!stage || !canvas) return;
                    const width = Math.max(1, Math.floor(img.clientWidth));
                    const height = Math.max(1, Math.floor(img.clientHeight));
                    canvas.width = width;
                    canvas.height = height;
                    canvas.style.width = `${width}px`;
                    canvas.style.height = `${height}px`;
                    canvas.style.left = '0px';
                    canvas.style.top = '0px';
                    if (cropCanvas) {
                        cropCanvas.width = width;
                        cropCanvas.height = height;
                        cropCanvas.style.width = `${width}px`;
                        cropCanvas.style.height = `${height}px`;
                        cropCanvas.style.left = '0px';
                        cropCanvas.style.top = '0px';
                    }
                    markerState.naturalWidth = img.naturalWidth || width;
                    markerState.naturalHeight = img.naturalHeight || height;
                    const ctx = canvas.getContext('2d');
                    if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
                    prepareMarkerBaseCanvas(img, width, height);
                    saveMarkerHistory();
                    if (markerState.mode === 'crop' && !markerState.cropRect) {
                        resetCropRectToFull();
                    }
                    renderCropOverlay();
                    resetMarkerTransform();
                };
                img.src = src;
            }
        }
        let uploadProgressState = { total: 0, completed: 0, active: 0, perFilePct: {} };
        const uploadCancelTokens = new Set();

        function updateGlobalUploadProgress(uploadId, pct) {
            if (uploadProgressState.perFilePct.hasOwnProperty(uploadId)) {
                uploadProgressState.perFilePct[uploadId] = pct;
                updateFilePreview();
            }
        }
        function resetUploadState() {
            currentImageUrls = [];
            currentMaskImage = null;
            uploadProgressState = { total: 0, completed: 0, active: 0, perFilePct: {} };
            uploadCancelTokens.clear();
            markerAppliedUploads.clear();
            const previewBar = get('file-preview');
            if (previewBar) previewBar.classList.add('hidden');
            const thumbContainer = get('file-preview-thumbs');
            if (thumbContainer) {
                thumbContainer.innerHTML = '';
                thumbContainer.classList.add('hidden');
            }
            updateFilePreview();
            updateMaskPreview();
            const list = get('upload-list');
            if (list) list.innerHTML = '<div class="text-xs text-gray-500">まだアップロードがありません。</div>';
            
            const input = get('file-input');
            if (input) input.value = '';
            const photoInput = get('photo-input');
            if (photoInput) photoInput.value = '';
            const maskInput = get('mask-input');
            if (maskInput) maskInput.value = '';
        }

        async function uploadMaskFile(file) {
            if (!file) return;
            const fd = new FormData();
            fd.append('file', file);
            try {
                const r = await fetch(CHAT_CONFIG.urls.upload, { method: 'POST', body: fd });
                const d = await r.json();
                if (r.ok && d.filename) {
                    currentMaskImage = d.filename;
                    updateMaskPreview();
                } else {
                    showToast(d.error || "Mask upload failed", "error", true);
                }
            } catch (e) {
                showToast("Mask upload failed", "error", true);
            }
        }
        function setCameraCaptureStatus(text, isError = false) {
            const el = get('camera-status');
            if (!el) return;
            el.textContent = text || '';
            el.classList.toggle('text-red-300', !!isError);
            el.classList.toggle('text-gray-400', !isError);
        }
        function updateCameraCapturePendingUi() {
            const count = cameraCapturePendingFiles.length;
            const attachBtn = get('camera-attach-btn');
            if (attachBtn) {
                attachBtn.disabled = count === 0 || cameraCaptureBusy;
                attachBtn.textContent = count ? `添付 (${count})` : '添付 (0)';
            }
            const clearBtn = get('camera-clear-btn');
            if (clearBtn) clearBtn.disabled = count === 0 || cameraCaptureBusy;
            const previewList = get('camera-capture-preview-list');
            if (previewList) {
                previewList.innerHTML = '';
                cameraCapturePendingPreviewUrls.forEach((url, index) => {
                    const item = document.createElement('div');
                    item.className = 'relative rounded overflow-hidden border border-gray-700 bg-black aspect-square';
                    item.innerHTML = `
                        <img src="${url}" alt="capture ${index + 1}" class="w-full h-full object-cover block">
                        <div class="absolute bottom-0 right-0 text-[10px] px-1 py-0.5 bg-black/70 text-white">${index + 1}</div>
                    `;
                    previewList.appendChild(item);
                });
                previewList.classList.toggle('hidden', count === 0);
            }
        }
        function resetCameraCapturePending(opts = {}) {
            while (cameraCapturePendingPreviewUrls.length) {
                const url = cameraCapturePendingPreviewUrls.pop();
                try { URL.revokeObjectURL(url); } catch (e) {}
            }
            cameraCapturePendingFiles.length = 0;
            updateCameraCapturePendingUi();
            if (!opts.keepStatus) {
                if (cameraCaptureStream) setCameraCaptureStatus('撮影して追加できます。最後に「添付」を押してください。');
                else setCameraCaptureStatus('カメラを起動中...');
            }
        }
        function stopCameraCaptureStream() {
            const video = get('camera-video');
            if (video && video.srcObject) {
                try { video.pause(); } catch (e) {}
                video.srcObject = null;
            }
            if (cameraCaptureStream) {
                try {
                    cameraCaptureStream.getTracks().forEach((track) => {
                        try { track.stop(); } catch (e) {}
                    });
                } catch (e) {}
            }
            cameraCaptureStream = null;
            cameraCaptureBusy = false;
            const captureBtn = get('camera-capture-btn');
            if (captureBtn) captureBtn.disabled = true;
            const switchBtn = get('camera-switch-btn');
            if (switchBtn) switchBtn.disabled = true;
        }
        async function startCameraCaptureStream(preferredFacingMode = 'environment') {
            const video = get('camera-video');
            if (!video) throw new Error('camera video element not found');
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                throw new Error('このブラウザはカメラAPIに対応していません');
            }
            stopCameraCaptureStream();
            setCameraCaptureStatus('カメラを起動中...');
            const switchBtn = get('camera-switch-btn');
            if (switchBtn) switchBtn.disabled = true;
            const candidates = [
                { video: { facingMode: { ideal: preferredFacingMode }, width: { ideal: 1920 }, height: { ideal: 1080 } }, audio: false },
                { video: { facingMode: preferredFacingMode }, audio: false },
                { video: true, audio: false }
            ];
            let lastErr = null;
            for (const constraints of candidates) {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia(constraints);
                    cameraCaptureStream = stream;
                    video.srcObject = stream;
                    try { await video.play(); } catch (e) {}
                    const track = stream.getVideoTracks && stream.getVideoTracks()[0];
                    const settings = track && track.getSettings ? track.getSettings() : {};
                    const actualFacing = String(settings.facingMode || '').toLowerCase();
                    if (actualFacing === 'user' || actualFacing === 'environment') {
                        cameraCaptureFacingMode = actualFacing;
                    } else {
                        cameraCaptureFacingMode = preferredFacingMode;
                    }
                    const captureBtn = get('camera-capture-btn');
                    if (captureBtn) captureBtn.disabled = false;
                    if (switchBtn) switchBtn.disabled = false;
                    setCameraCaptureStatus(cameraCapturePendingFiles.length > 0
                        ? `${cameraCapturePendingFiles.length}枚撮影済み。続けて撮影するか「添付」を押してください。`
                        : '撮影して追加できます。最後に「添付」を押してください。');
                    updateCameraCapturePendingUi();
                    return stream;
                } catch (e) {
                    lastErr = e;
                }
            }
            throw lastErr || new Error('カメラを起動できませんでした');
        }
        async function openCameraCaptureModal() {
            if (!window.isSecureContext && location.hostname !== 'localhost' && location.hostname !== '127.0.0.1') {
                showToast('カメラ起動は HTTPS / localhost 環境で利用できます。写真選択に切り替えます。', 'warning', true);
                const photoInput = get('photo-input');
                if (photoInput) photoInput.click();
                return;
            }
            resetCameraCapturePending({ keepStatus: true });
            updateCameraCapturePendingUi();
            showModal('camera-capture-modal');
            if (location.pathname !== '/camera') {
                history.pushState({ modal: 'camera' }, '', '/camera');
            }
            try {
                await startCameraCaptureStream(cameraCaptureFacingMode || 'environment');
            } catch (e) {
                const msg = (e && e.message) ? e.message : 'カメラを起動できませんでした';
                setCameraCaptureStatus(msg, true);
                showToast(msg, 'error', true);
                const captureBtn = get('camera-capture-btn');
                if (captureBtn) captureBtn.disabled = true;
                const attachBtn = get('camera-attach-btn');
                if (attachBtn) attachBtn.disabled = true;
            }
        }
        function closeCameraCaptureModal(options = {}) {
            const skipHistory = options.skipHistory || false;
            hideModal('camera-capture-modal', options);
            if (!skipHistory && location.pathname === '/camera') {
                history.back();
            }
        }
        async function toggleCameraCaptureFacing() {
            if (cameraCaptureBusy) return;
            const switchBtn = get('camera-switch-btn');
            if (switchBtn) switchBtn.disabled = true;
            const next = String(cameraCaptureFacingMode || '').toLowerCase() === 'user' ? 'environment' : 'user';
            cameraCaptureFacingMode = next;
            try {
                await startCameraCaptureStream(next);
            } catch (e) {
                const msg = (e && e.message) ? e.message : 'カメラ切替に失敗しました';
                setCameraCaptureStatus(msg, true);
                showToast(msg, 'error', true);
            } finally {
                if (switchBtn && get('camera-capture-modal') && !get('camera-capture-modal').classList.contains('hidden')) {
                    switchBtn.disabled = false;
                }
            }
        }
        function buildCameraCaptureFilename() {
            const now = new Date();
            const pad = (n) => String(n).padStart(2, '0');
            const ms = String(now.getMilliseconds()).padStart(3, '0');
            cameraCaptureSequence = (cameraCaptureSequence + 1) % 1000;
            const seq = String(cameraCaptureSequence).padStart(3, '0');
            return `camera_${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}_${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}_${ms}_${seq}.jpg`;
        }
        async function captureCameraShot() {
            if (cameraCaptureBusy) return;
            const video = get('camera-video');
            const canvas = get('camera-canvas');
            const modal = get('camera-capture-modal');
            if (!video || !canvas || !modal) return;
            if (!video.videoWidth || !video.videoHeight) {
                showToast('カメラ映像の準備中です。少し待ってから再度お試しください。', 'warning', true);
                return;
            }
            cameraCaptureBusy = true;
            const captureBtn = get('camera-capture-btn');
            if (captureBtn) captureBtn.disabled = true;
            const attachBtn = get('camera-attach-btn');
            if (attachBtn) attachBtn.disabled = true;
            setCameraCaptureStatus('撮影中...');
            try {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                const ctx = canvas.getContext('2d');
                if (!ctx) throw new Error('撮影処理に失敗しました');
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                const blob = await new Promise((resolve, reject) => {
                    canvas.toBlob((b) => {
                        if (b) resolve(b);
                        else reject(new Error('画像の生成に失敗しました'));
                    }, 'image/jpeg', 0.92);
                });
                const file = new File([blob], buildCameraCaptureFilename(), { type: 'image/jpeg', lastModified: Date.now() });
                cameraCapturePendingFiles.push(file);
                cameraCapturePendingPreviewUrls.push(URL.createObjectURL(blob));
                updateCameraCapturePendingUi();
                setCameraCaptureStatus(`${cameraCapturePendingFiles.length}枚撮影済み。続けて撮影するか「添付」を押してください。`);
            } catch (e) {
                const msg = (e && e.message) ? e.message : '撮影に失敗しました';
                setCameraCaptureStatus(msg, true);
                showToast(msg, 'error', true);
            } finally {
                cameraCaptureBusy = false;
                if (captureBtn && modal && !modal.classList.contains('hidden')) {
                    captureBtn.disabled = false;
                }
                updateCameraCapturePendingUi();
            }
        }
        async function attachCameraCapturedFiles() {
            if (cameraCaptureBusy) return;
            if (!cameraCapturePendingFiles.length) {
                showToast('先に撮影してください', 'warning', true);
                return;
            }
            const modal = get('camera-capture-modal');
            cameraCaptureBusy = true;
            const captureBtn = get('camera-capture-btn');
            const switchBtn = get('camera-switch-btn');
            const attachBtn = get('camera-attach-btn');
            const clearBtn = get('camera-clear-btn');
            if (captureBtn) captureBtn.disabled = true;
            if (switchBtn) switchBtn.disabled = true;
            if (attachBtn) attachBtn.disabled = true;
            if (clearBtn) clearBtn.disabled = true;
            const filesToUpload = Array.from(cameraCapturePendingFiles).reverse();
            
            // モーダルを即座に閉じて他の操作を可能にする
            // skipReset: true を指定して、アップロード中にファイルが消されないようにする
            closeCameraCaptureModal({ skipReset: true });
            // closeUploadModal(); // Prevent history conflict
            
            // stopCameraCaptureStream が busy = false にしてしまうため、戻す
            cameraCaptureBusy = true;
            
            setCameraCaptureStatus(`${filesToUpload.length}枚を添付中...`);
            try {
                await handleFiles(filesToUpload, { openModal: false });
                showToast(`${filesToUpload.length}枚の画像を添付しました`, 'success');
            } catch (e) {
                const msg = (e && e.message) ? e.message : '撮影画像の添付に失敗しました';
                showToast(msg, 'error', true);
            } finally {
                cameraCaptureBusy = false;
                // 成功・失敗に関わらず最後は撮影キューを空にする
                resetCameraCapturePending({ keepStatus: true });
                if (modal && !modal.classList.contains('hidden')) {
                    if (captureBtn) captureBtn.disabled = false;
                    if (switchBtn) switchBtn.disabled = false;
                    updateCameraCapturePendingUi();
                }
            }
        }
        function openUploadModal() {
            syncUploadRowsFromCurrent();
            showModal('upload-modal');
            if (location.pathname !== '/upload') {
                history.pushState({ modal: 'upload' }, '', '/upload');
            }
            const vmi = get('vision-model-info');
            if (vmi) {
                const model = get('model-select') ? get('model-select').value : '';
                vmi.classList.toggle('hidden', !model.toLowerCase().includes('deepseek'));
            }
            _syncVisionModelDisplay();
        }
        function _syncVisionModelDisplay() {
            const display = get('vision-model-display');
            if (!display) return;
            const vm = currentVisionModel;
            if (vm) {
                let name = vm;
                MODELS.forEach(g => (g.items || []).forEach(m => { if (m.id === vm) name = m.name; }));
                display.textContent = name;
            } else {
                display.textContent = '設定から選択';
            }
        }
        function _openVisionModelSelector() {
            window._visionPickerActive = true;
            openModelModal();
            setTimeout(() => {
                const searchEl = get('model-search');
                if (searchEl) searchEl.value = '';
                renderModelList('');
            }, 50);
        }
        function closeUploadModal(skipHistory = false) {
            hideModal('upload-modal');
            if (!skipHistory && location.pathname === '/upload') {
                history.back();
            }
        }
        function syncUploadRowsFromCurrent() {
            const list = get('upload-list');
            if (!list) return;
            const existing = new Set();
            list.querySelectorAll('[data-filename]').forEach((el) => {
                const fp = el.getAttribute('data-filename');
                if (fp) existing.add(fp);
            });
            currentImageUrls.forEach((fp) => {
                if (!existing.has(fp)) {
                    addStoredUploadRow(fp, {
                        source: getAttachmentSourceForPath(fp),
                        displayName: getAttachmentNameForPath(fp)
                    });
                }
            });
            if (list.children.length === 0) {
                list.innerHTML = '<div class="text-xs text-gray-500">まだアップロードがありません。</div>';
            }
        }
        function decrementUploadTotal(uploadId) {
            if (uploadProgressState.total > 0) uploadProgressState.total--;
            if (uploadProgressState.perFilePct.hasOwnProperty(uploadId)) {
                delete uploadProgressState.perFilePct[uploadId];
                if (uploadProgressState.active > 0) uploadProgressState.active--;
            }
            if (uploadProgressState.active <= 0) {
                uploadProgressState.total = 0;
                uploadProgressState.completed = 0;
                uploadProgressState.active = 0;
                uploadProgressState.perFilePct = {};
            }
            updateFilePreview();
        }
        function addStoredUploadRow(filepath, opts = {}) {
            if (!filepath) return null;
            filepath = normalizeAttachmentPath(filepath);
            if (!filepath) return null;
            const source = normalizeAttachmentSource(opts.source);
            const list = get('upload-list');
            if (!list) return null;
            if (list.children.length === 1 && list.children[0].classList.contains('text-gray-500')) {
                list.innerHTML = '';
            }
            const filename = filepath.split('/').pop() || filepath;
            const displayName = normalizeAttachmentDisplayName(opts.displayName) || getAttachmentNameForPath(filepath) || filename;
            const ext = (filename.split('.').pop() || '').toLowerCase();
            const isImage = ['png','jpg','jpeg','webp','gif'].includes(ext);
            const fileUrl = buildFileUrl(filepath);
            const previewUrl = isImage ? buildAttachmentPreviewUrl(filepath) : fileUrl;
            const uploadId = `lib_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
            const row = document.createElement('div');
            row.className = 'upload-row ui-enter bg-gray-900/60 rounded p-2';
            row.dataset.uploadId = uploadId;
            row.setAttribute('data-filename', filepath);
            row.dataset.fileSource = source;
            row.dataset.displayName = displayName;
            row.dataset.defaultDisplayName = displayName;
            row.dataset.sendNameCustomized = '';
            const safeName = escapeHtml(displayName);
            const markerBtnHtml = isImage
                ? `<button class="upload-marker text-[10px] border rounded px-2 py-1">画像編集</button>`
                : '';
            const previewHtml = isImage
                ? `<img src="${previewUrl}" loading="lazy" decoding="async" class="upload-preview w-12 h-12 object-cover rounded border border-gray-700 cursor-pointer" alt="${safeName}">`
                : `<div class="upload-preview w-12 h-12 bg-gray-800 rounded border border-gray-700 flex items-center justify-center text-gray-400 text-sm cursor-pointer">FILE</div>`;
            row.innerHTML = `
                <div class="flex items-center gap-3">
                    ${previewHtml}
                    <div class="flex-1 min-w-0">
                        <div class="truncate text-xs text-gray-200">${safeName}</div>
                        <div class="flex items-center gap-2">
                            <div class="upload-status text-[10px] text-gray-400">ready</div>
                            <span class="upload-marker-tag hidden">編集済み</span>
                        </div>
                    </div>
                    <div class="flex items-center gap-1">
                        ${markerBtnHtml}
                        <button class="upload-send-name text-[10px] text-gray-300 hover:text-white border border-gray-700 rounded px-2 py-1">送信名</button>
                        <button class="upload-remove text-[10px] text-gray-400 hover:text-red-400 border border-gray-700 rounded px-2 py-1">削除</button>
                    </div>
                </div>
                <div class="upload-progress h-2 rounded mt-2 overflow-hidden">
                    <div style="width:100%"></div>
                </div>
            `;
            const previewEl = row.querySelector('.upload-preview');
            if (previewEl) {
                previewEl.onclick = () => openFileViewer(fileUrl, getRowAttachmentName(row) || displayName);
            }
            const sendNameBtn = row.querySelector('.upload-send-name');
            if (sendNameBtn) {
                sendNameBtn.onclick = () => promptRowAttachmentName(row);
            }
            const removeBtn = row.querySelector('.upload-remove');
            if (removeBtn) {
                removeBtn.onclick = () => {
                    uploadCancelTokens.add(uploadId);
                    decrementUploadTotal(uploadId);
                    const stored = row.getAttribute('data-filename');
                    if (stored) currentImageUrls = currentImageUrls.filter(x => x !== stored);
                    setRowMarkerState(row, false);
                    row.remove();
                    updateFilePreview();
                    if (list.children.length === 0) {
                        list.innerHTML = '<div class="text-xs text-gray-500">まだアップロードがありません。</div>';
                    }
                };
            }
            const markerBtn = row.querySelector('.upload-marker');
            if (markerBtn) {
                markerBtn.onclick = () => openMarkerModalForRow(row);
            }
            setAttachmentSourceForPath(filepath, source);
            setAttachmentNameForPath(filepath, displayName);
            list.prepend(row);
            return {
                row: row,
                bar: row.querySelector('.upload-progress > div'),
                status: row.querySelector('.upload-status'),
                uploadId: uploadId
            };
        }
        function addUploadRow(file) {
            const list = get('upload-list');
            if (!list) return null;
            if (list.children.length === 1 && list.children[0].classList.contains('text-gray-500')) {
                list.innerHTML = '';
            }
            const uploadId = `up_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
            const row = document.createElement('div');
            row.className = 'upload-row ui-enter bg-gray-900/60 rounded p-2';
            row.dataset.uploadId = uploadId;
            row.dataset.fileSource = 'upload';
            const displayName = normalizeAttachmentDisplayName(file.name || 'file') || 'file';
            row.dataset.displayName = displayName;
            row.dataset.defaultDisplayName = displayName;
            row.dataset.sendNameCustomized = '';
            const safeName = escapeHtml(displayName);
            const isImage = file && file.type && file.type.startsWith('image/');
            let previewHtml = '<div class="upload-preview w-12 h-12 bg-gray-800 rounded border border-gray-700 flex items-center justify-center text-gray-400 text-sm">FILE</div>';
            const markerBtnHtml = isImage
                ? `<button class="upload-marker text-[10px] border rounded px-2 py-1">画像編集</button>`
                : '';
            let previewUrl = '';
            if (isImage) {
                previewUrl = URL.createObjectURL(file);
                previewHtml = `<img src="${previewUrl}" class="upload-preview w-12 h-12 object-cover rounded border border-gray-700 cursor-pointer" alt="${safeName}">`;
            } else {
                previewUrl = URL.createObjectURL(file);
                previewHtml = `<div class="upload-preview w-12 h-12 bg-gray-800 rounded border border-gray-700 flex items-center justify-center text-gray-400 text-sm cursor-pointer">FILE</div>`;
            }
            row.innerHTML = `
                <div class="flex items-center gap-3">
                    ${previewHtml}
                    <div class="flex-1 min-w-0">
                        <div class="truncate text-xs text-gray-200">${safeName}</div>
                        <div class="flex items-center gap-2">
                            <div class="upload-status text-[10px] text-gray-400">待機中</div>
                            <span class="upload-marker-tag hidden">編集済み</span>
                        </div>
                    </div>
                    <div class="flex items-center gap-1">
                        ${markerBtnHtml}
                        <button class="upload-send-name text-[10px] text-gray-300 hover:text-white border border-gray-700 rounded px-2 py-1">送信名</button>
                        <button class="upload-remove text-[10px] text-gray-400 hover:text-red-400 border border-gray-700 rounded px-2 py-1">削除</button>
                    </div>
                </div>
                <div class="upload-progress h-2 rounded mt-2 overflow-hidden">
                    <div style="width:0%"></div>
                </div>
            `;
            if (previewUrl) row.setAttribute('data-local-url', previewUrl);
            const previewEl = row.querySelector('.upload-preview');
            if (previewEl) {
                previewEl.onclick = () => {
                    const filepath = row.getAttribute('data-filename');
                    const url = filepath ? buildFileUrl(filepath) : row.getAttribute('data-local-url');
                    const openName = normalizeAttachmentDisplayName(row.dataset.displayName) || file.name || filepath || '';
                    openFileViewer(url, openName);
                };
            }
            const removeBtn = row.querySelector('.upload-remove');
            if (removeBtn) {
                removeBtn.onclick = () => {
                    uploadCancelTokens.add(uploadId);
                    decrementUploadTotal(uploadId);
                    const localUrl = row.getAttribute('data-local-url');
                    if (localUrl) URL.revokeObjectURL(localUrl);
                    const stored = row.getAttribute('data-filename');
                    if (stored) currentImageUrls = currentImageUrls.filter(x => x !== stored);
                    setRowMarkerState(row, false);
                    row.remove();
                    updateFilePreview();
                    if (list.children.length === 0) {
                        list.innerHTML = '<div class="text-xs text-gray-500">まだアップロードがありません。</div>';
                    }
                };
            }
            const markerBtn = row.querySelector('.upload-marker');
            if (markerBtn) {
                markerBtn.onclick = () => openMarkerModalForRow(row);
            }
            const sendNameBtn = row.querySelector('.upload-send-name');
            if (sendNameBtn) {
                sendNameBtn.onclick = () => promptRowAttachmentName(row);
            }
            list.prepend(row);
            return {
                uploadId,
                row,
                status: row.querySelector('.upload-status'),
                bar: row.querySelector('.upload-progress > div')
            };
        }
        // Use chunked upload for medium/large files to avoid proxy/body size limits.
        const CHUNK_THRESHOLD_BYTES = 20 * 1024 * 1024;
        async function uploadFileChunked(file, row) {
            if (!file) return false;
            try {
                const initRes = await apiFetch("/upload/init", {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json', 'X-CSRF-Token': csrfToken},
                    body: JSON.stringify({ filename: file.name, size: file.size })
                });
                const initData = await initRes.json();
                if (!initRes.ok) {
                    const msg = initData && initData.error ? initData.error : "アップロードに失敗しました";
                    if (row && row.status) row.status.textContent = '失敗';
                    showToast(msg, "error", true);
                    return false;
                }
                const uploadId = initData.upload_id;
                const chunkSize = initData.chunk_size || (10 * 1024 * 1024);
                const totalChunks = Math.ceil(file.size / chunkSize);
                for (let i = 0; i < totalChunks; i++) {
                    const start = i * chunkSize;
                    const end = Math.min(file.size, start + chunkSize);
                    const chunk = file.slice(start, end);
                    const ok = await new Promise((resolve) => {
                        const xhr = new XMLHttpRequest();
                        xhr.open('POST', "/upload/chunk", true);
                        xhr.setRequestHeader('X-CSRF-Token', csrfToken);
                        xhr.upload.onprogress = (e) => {
                            if (e.lengthComputable && row && row.bar) {
                                const done = start + e.loaded;
                                const pct = Math.min(100, Math.floor((done / file.size) * 100));
                                row.bar.style.width = `${pct}%`;
                                if (row.status) row.status.textContent = `${pct}%`;
                                if (row.uploadId) updateGlobalUploadProgress(row.uploadId, pct);
                            }
                        };
                        xhr.onload = () => {
                            if (xhr.status >= 200 && xhr.status < 300) resolve(true);
                            else resolve(false);
                        };
                        xhr.onerror = () => resolve(false);
                        const fd = new FormData();
                        fd.append('upload_id', uploadId);
                        fd.append('index', String(i));
                        fd.append('total', String(totalChunks));
                        fd.append('chunk', chunk, file.name);
                        xhr.send(fd);
                    });
                    if (!ok) {
                        if (row && row.status) row.status.textContent = '失敗';
                        showToast("アップロードに失敗しました", "error", true);
                        return false;
                    }
                }
                if (row && row.status) row.status.textContent = '処理中...';
                const doneRes = await apiFetch("/upload/complete", {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json', 'X-CSRF-Token': csrfToken},
                    body: JSON.stringify({ upload_id: uploadId })
                });
                const doneData = await doneRes.json();
                if (doneRes.ok && doneData && doneData.filename) {
                    if (row && row.row && row.uploadId && uploadCancelTokens.has(row.uploadId)) {
                        if (row.row && row.row.parentNode) row.row.remove();
                        return false;
                    }
                    if (row && row.row) {
                        const localUrl = row.row.getAttribute('data-local-url');
                        if (localUrl) URL.revokeObjectURL(localUrl);
                        row.row.removeAttribute('data-local-url');
                        
                        // Update source to server URL
                        const img = row.row.querySelector('img.upload-preview');
                        if (img) {
                            const displayPath = doneData.filename.replace(/^\d+\//, '');
                            img.src = buildAttachmentPreviewUrl(displayPath);
                        }
                    }
                    const normPath = normalizeAttachmentPath(doneData.filename);
                    if (normPath) currentImageUrls.push(normPath);
                    if (row && row.row) {
                        row.row.setAttribute('data-filename', normPath || doneData.filename);
                        setRowAttachmentSource(row.row, 'upload');
                        if (normPath) {
                            const hasCustom = isRowAttachmentNameCustomized(row.row);
                            const fallbackName = defaultAttachmentDisplayName(normPath);
                            const nextName = hasCustom ? (normalizeAttachmentDisplayName(row.row.dataset.displayName) || fallbackName) : fallbackName;
                            row.row.dataset.defaultDisplayName = fallbackName;
                            setRowAttachmentName(row.row, nextName);
                        }
                    }
                    if (normPath) setAttachmentSourceForPath(normPath, 'upload');
                    if (row && row.status) row.status.textContent = '完了';
                    updateFilePreview();
                    const filenames = Array.isArray(doneData.filenames) && doneData.filenames.length ? doneData.filenames : [doneData.filename];
                    filenames.forEach((fp) => addLibraryFileFromPath(fp));
                    return true;
                }
                const msg = (doneData && doneData.error) ? doneData.error : "アップロードに失敗しました";
                if (row && row.status) row.status.textContent = '失敗';
                showToast(msg, "error", true);
                return false;
            } catch (e) {
                if (row && row.status) row.status.textContent = '失敗';
                showToast("アップロード中にエラーが発生しました", "error", true);
                return false;
            }
        }
        function uploadFileWithProgress(file, row) {
            return new Promise((resolve) => {
                if (file && file.size > CHUNK_THRESHOLD_BYTES) {
                    uploadFileChunked(file, row).then(resolve);
                    return;
                }
                const xhr = new XMLHttpRequest();
                xhr.open('POST', CHAT_CONFIG.urls.upload, true);
                xhr.setRequestHeader('X-CSRF-Token', csrfToken);
                xhr.upload.onprogress = (e) => {
                    if (e.lengthComputable && row && row.bar) {
                        const pct = Math.min(100, Math.floor((e.loaded / e.total) * 100));
                        row.bar.style.width = `${pct}%`;
                        if (row.status) row.status.textContent = `${pct}%`;
                        if (row.uploadId) updateGlobalUploadProgress(row.uploadId, pct);
                    }
                };
                xhr.onload = () => {
                    let d = {};
                    try { d = JSON.parse(xhr.responseText || '{}'); } catch (e) {}
                    if (xhr.status >= 200 && xhr.status < 300 && d && d.filename) {
                        if (row && row.row && row.uploadId && uploadCancelTokens.has(row.uploadId)) {
                            if (row.row && row.row.parentNode) row.row.remove();
                            resolve(false);
                            return;
                        }
                        if (row && row.row) {
                            const localUrl = row.row.getAttribute('data-local-url');
                            if (localUrl) URL.revokeObjectURL(localUrl);
                            row.row.removeAttribute('data-local-url');

                            // Update source to server URL
                            const img = row.row.querySelector('img.upload-preview');
                            if (img) {
                                const displayPath = d.filename.replace(/^\d+\//, '');
                                img.src = buildAttachmentPreviewUrl(displayPath);
                            }
                        }
                        const normPath = normalizeAttachmentPath(d.filename);
                        if (normPath) currentImageUrls.push(normPath);
                        if (row && row.row) {
                            row.row.setAttribute('data-filename', normPath || d.filename);
                            setRowAttachmentSource(row.row, 'upload');
                            if (normPath) {
                                const hasCustom = isRowAttachmentNameCustomized(row.row);
                                const fallbackName = defaultAttachmentDisplayName(normPath);
                                const nextName = hasCustom ? (normalizeAttachmentDisplayName(row.row.dataset.displayName) || fallbackName) : fallbackName;
                                row.row.dataset.defaultDisplayName = fallbackName;
                                setRowAttachmentName(row.row, nextName);
                            }
                        }
                        if (normPath) setAttachmentSourceForPath(normPath, 'upload');
                        if (row && row.status) row.status.textContent = '完了';
                        updateFilePreview();
                        const filenames = Array.isArray(d.filenames) && d.filenames.length ? d.filenames : [d.filename];
                        filenames.forEach((fp) => addLibraryFileFromPath(fp));
                        resolve(true);
                    } else {
                        const msg = (d && d.error) ? d.error : "アップロードに失敗しました";
                        if (row && row.status) row.status.textContent = '失敗';
                        showToast(msg, "error", true);
                        resolve(false);
                    }
                };
                xhr.onerror = () => {
                    if (row && row.status) row.status.textContent = '失敗';
                    showToast("アップロード中にエラーが発生しました", "error", true);
                    resolve(false);
                };
                const fd = new FormData();
                fd.append('file', file);
                xhr.send(fd);
            });
        }
        function isVideoFile(file) {
            if (!file) return false;
            if (file.type && file.type.startsWith('video/')) return true;
            return VIDEO_EXTS.includes(getFileExt(file.name || ''));
        }
        function isAudioFile(file) {
            if (!file) return false;
            if (file.type && file.type.startsWith('audio/')) return true;
            return AUDIO_EXTS.includes(getFileExt(file.name || ''));
        }
        function encodeWav(buffers, sampleRate) {
            let length = 0;
            buffers.forEach(b => { length += b.length; });
            const pcm = new Float32Array(length);
            let offset = 0;
            buffers.forEach(b => {
                pcm.set(b, offset);
                offset += b.length;
            });
            const bytes = new ArrayBuffer(44 + pcm.length * 2);
            const view = new DataView(bytes);
            const writeString = (o, s) => { for (let i = 0; i < s.length; i++) view.setUint8(o + i, s.charCodeAt(i)); };
            writeString(0, 'RIFF');
            view.setUint32(4, 36 + pcm.length * 2, true);
            writeString(8, 'WAVE');
            writeString(12, 'fmt ');
            view.setUint32(16, 16, true);
            view.setUint16(20, 1, true);
            view.setUint16(22, 1, true);
            view.setUint32(24, sampleRate, true);
            view.setUint32(28, sampleRate * 2, true);
            view.setUint16(32, 2, true);
            view.setUint16(34, 16, true);
            writeString(36, 'data');
            view.setUint32(40, pcm.length * 2, true);
            let idx = 44;
            for (let i = 0; i < pcm.length; i++) {
                const s = Math.max(-1, Math.min(1, pcm[i]));
                view.setInt16(idx, s < 0 ? s * 0x8000 : s * 0x7fff, true);
                idx += 2;
            }
            return new Blob([view], { type: 'audio/wav' });
        }
        function pickAudioRecorderType() {
            if (typeof MediaRecorder === 'undefined') return '';
            const candidates = [
                'audio/webm;codecs=opus',
                'audio/webm',
                'audio/ogg;codecs=opus',
                'audio/ogg'
            ];
            for (const c of candidates) {
                if (MediaRecorder.isTypeSupported(c)) return c;
            }
            return '';
        }
        function updateUploadRowFile(row, file) {
            if (!row || !row.row || !file) return;
            const nameEl = row.row.querySelector('.truncate');
            const hasCustomName = isRowAttachmentNameCustomized(row.row);
            const displayName = hasCustomName
                ? (normalizeAttachmentDisplayName(row.row.dataset.displayName) || 'file')
                : (normalizeAttachmentDisplayName(file.name || 'file') || 'file');
            if (nameEl) nameEl.textContent = displayName;
            row.row.dataset.displayName = displayName;
            if (!hasCustomName) row.row.dataset.defaultDisplayName = displayName;
            const localUrl = row.row.getAttribute('data-local-url');
            if (localUrl) URL.revokeObjectURL(localUrl);
            const newUrl = URL.createObjectURL(file);
            row.row.setAttribute('data-local-url', newUrl);
            const isImage = file.type && file.type.startsWith('image/');
            const safeName = escapeHtml(displayName);
            const previewHtml = isImage
                ? `<img src="${newUrl}" class="upload-preview w-12 h-12 object-cover rounded border border-gray-700 cursor-pointer" alt="${safeName}">`
                : `<div class="upload-preview w-12 h-12 bg-gray-800 rounded border border-gray-700 flex items-center justify-center text-gray-400 text-sm cursor-pointer">FILE</div>`;
            const previewOld = row.row.querySelector('.upload-preview');
            if (previewOld) previewOld.outerHTML = previewHtml;
            const previewEl = row.row.querySelector('.upload-preview');
            if (previewEl) {
                previewEl.onclick = () => {
                    const filepath = row.row.getAttribute('data-filename');
                    const url = filepath ? buildFileUrl(filepath) : row.row.getAttribute('data-local-url');
                    openFileViewer(url, getRowAttachmentName(row.row) || displayName || filepath || '');
                };
            }
            const markerBtn = row.row.querySelector('.upload-marker');
            if (markerBtn) {
                markerBtn.classList.toggle('hidden', !isImage);
            }
            if (!isImage) {
                setRowMarkerState(row.row, false);
                row.row.dataset.originalFilename = '';
                row.row.dataset.originalSource = '';
                row.row.dataset.attachOriginal = '';
            }
        }
        function saveMarkerHistory() {
            const canvas = get('marker-canvas');
            if (!canvas) return;
            const ctx = canvas.getContext('2d');
            if (!ctx) return;
            const mosaicRects = Array.isArray(markerState.mosaicRects)
                ? markerState.mosaicRects.map((r) => ({ x: r.x, y: r.y, w: r.w, h: r.h }))
                : [];
            markerState.history.push({
                imageData: ctx.getImageData(0, 0, canvas.width, canvas.height),
                mosaicRects
            });
            if (markerState.history.length > 40) markerState.history.shift();
        }
        function undoMarkerCanvas() {
            if (markerState.history.length <= 1) return; // Only initial state left or empty
            markerState.history.pop(); // Remove current
            const canvas = get('marker-canvas');
            if (!canvas) return;
            const ctx = canvas.getContext('2d');
            if (!ctx) return;
            const last = markerState.history[markerState.history.length - 1];
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            if (last && last.imageData) {
                ctx.putImageData(last.imageData, 0, 0);
                markerState.mosaicRects = Array.isArray(last.mosaicRects)
                    ? last.mosaicRects.map((r) => ({ x: r.x, y: r.y, w: r.w, h: r.h }))
                    : [];
            } else if (last) {
                // Backward compatible fallback for old history snapshots.
                ctx.putImageData(last, 0, 0);
                markerState.mosaicRects = [];
            } else {
                markerState.mosaicRects = [];
            }
            markerState.mosaicPreviewRect = null;
            markerState.hasStroke = markerState.history.length > 1;
            renderCropOverlay();
        }
        function clearMarkerCanvas() {
            const canvas = get('marker-canvas');
            if (!canvas) return;
            const ctx = canvas.getContext('2d');
            if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
            markerState.hasStroke = false;
            markerState.mosaicRects = [];
            markerState.mosaicPreviewRect = null;
            renderCropOverlay();
            saveMarkerHistory();
        }
        function initMarkerCanvas() {
            const canvas = get('marker-canvas');
            if (!canvas) return;
            const ctx = canvas.getContext('2d');
            const sizeInput = get('marker-size');
            const activePointers = new Map();
            let pinching = false;
            let pinchStartDist = 0;
            let pinchStartScale = markerView.scale;
            let pinchStartOffset = { x: 0, y: 0 };
            let pinchStartMid = { x: 0, y: 0 };
            let strokePoints = [];
            let strokeSize = 16;
            let strokeStyle = '';
            let drawBaseCanvas = null;
            let drawBaseCtx = null;
            let drawLayerCanvas = null;
            let drawLayerCtx = null;
            let mosaicSelecting = false;
            let mosaicStartPoint = null;
            const getPoint = (e) => {
                const rect = canvas.getBoundingClientRect();
                const x = (e.clientX - rect.left) * (canvas.width / rect.width);
                const y = (e.clientY - rect.top) * (canvas.height / rect.height);
                return { x, y };
            };
            const getMid = (a, b) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });
            const getDist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
            let drawing = false;
            const ensureDrawBuffers = () => {
                if (!drawBaseCanvas) {
                    drawBaseCanvas = document.createElement('canvas');
                    drawBaseCtx = drawBaseCanvas.getContext('2d');
                }
                if (!drawLayerCanvas) {
                    drawLayerCanvas = document.createElement('canvas');
                    drawLayerCtx = drawLayerCanvas.getContext('2d');
                }
                if (drawBaseCanvas.width !== canvas.width || drawBaseCanvas.height !== canvas.height) {
                    drawBaseCanvas.width = canvas.width;
                    drawBaseCanvas.height = canvas.height;
                }
                if (drawLayerCanvas.width !== canvas.width || drawLayerCanvas.height !== canvas.height) {
                    drawLayerCanvas.width = canvas.width;
                    drawLayerCanvas.height = canvas.height;
                }
            };
            const renderDrawPreview = () => {
                if (!ctx || !drawBaseCanvas || !drawLayerCanvas) return;
                const alpha = Math.max(MARKER_OPACITY_MIN_ALPHA, Math.min(1, Number(markerState.opacity) || 0.6));
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(drawBaseCanvas, 0, 0);
                ctx.save();
                ctx.globalAlpha = alpha;
                ctx.drawImage(drawLayerCanvas, 0, 0);
                ctx.restore();
            };
            const applyMarkerBrush = () => {
                if (!drawLayerCtx) return;
                drawLayerCtx.strokeStyle = strokeStyle;
                drawLayerCtx.fillStyle = strokeStyle;
                drawLayerCtx.lineWidth = strokeSize;
                drawLayerCtx.lineCap = 'round';
                drawLayerCtx.lineJoin = 'round';
            };
            const appendStrokePoint = (point) => {
                if (!point) return false;
                if (strokePoints.length === 0) {
                    strokePoints.push(point);
                    return true;
                }
                const from = strokePoints[strokePoints.length - 1];
                const dx = point.x - from.x;
                const dy = point.y - from.y;
                const dist = Math.hypot(dx, dy);
                const minStep = Math.max(0.35, strokeSize * 0.04);
                if (dist < minStep) return false;
                const maxStep = Math.max(1, strokeSize * 0.25);
                const steps = Math.max(1, Math.ceil(dist / maxStep));
                for (let i = 1; i <= steps; i++) {
                    const t = i / steps;
                    strokePoints.push({
                        x: from.x + dx * t,
                        y: from.y + dy * t
                    });
                }
                return true;
            };
            const renderStrokeLayer = () => {
                if (!drawLayerCtx) return;
                drawLayerCtx.clearRect(0, 0, drawLayerCanvas.width, drawLayerCanvas.height);
                if (strokePoints.length === 0) return;
                applyMarkerBrush();
                if (strokePoints.length === 1) {
                    const p = strokePoints[0];
                    drawLayerCtx.beginPath();
                    drawLayerCtx.arc(p.x, p.y, strokeSize / 2, 0, Math.PI * 2);
                    drawLayerCtx.fill();
                    return;
                }
                drawLayerCtx.beginPath();
                drawLayerCtx.moveTo(strokePoints[0].x, strokePoints[0].y);
                if (strokePoints.length === 2) {
                    drawLayerCtx.lineTo(strokePoints[1].x, strokePoints[1].y);
                } else {
                    for (let i = 1; i < strokePoints.length - 2; i++) {
                        const p = strokePoints[i];
                        const n = strokePoints[i + 1];
                        const mid = getMid(p, n);
                        drawLayerCtx.quadraticCurveTo(p.x, p.y, mid.x, mid.y);
                    }
                    const secondLast = strokePoints[strokePoints.length - 2];
                    const last = strokePoints[strokePoints.length - 1];
                    drawLayerCtx.quadraticCurveTo(secondLast.x, secondLast.y, last.x, last.y);
                }
                drawLayerCtx.stroke();
            };
            const normalizeMosaicRect = (a, b) => {
                if (!a || !b) return null;
                const x = Math.min(a.x, b.x);
                const y = Math.min(a.y, b.y);
                const w = Math.abs(a.x - b.x);
                const h = Math.abs(a.y - b.y);
                return { x, y, w, h };
            };
            const buildMosaicRectFromPoint = (p) => {
                const size = sizeInput ? Number(sizeInput.value || 16) : 16;
                const side = Math.max(6, Math.floor(size));
                const half = Math.floor(side / 2);
                return { x: p.x - half, y: p.y - half, w: side, h: side };
            };
            const getMosaicSourceImageData = () => {
                const sourceCanvas = document.createElement('canvas');
                sourceCanvas.width = canvas.width;
                sourceCanvas.height = canvas.height;
                const sourceCtx = sourceCanvas.getContext('2d');
                if (!sourceCtx) return null;
                if (markerState.baseCanvas) {
                    sourceCtx.drawImage(markerState.baseCanvas, 0, 0);
                }
                sourceCtx.drawImage(canvas, 0, 0);
                try {
                    return sourceCtx.getImageData(0, 0, canvas.width, canvas.height);
                } catch (e) {
                    return null;
                }
            };
            const applyMosaicRect = (rect) => {
                if (!ctx || !rect) return false;
                const source = getMosaicSourceImageData();
                if (!source) return false;
                const size = sizeInput ? Number(sizeInput.value || 16) : 16;
                const block = Math.max(4, Math.floor(size / 2));
                const x1 = Math.max(0, Math.floor(rect.x));
                const y1 = Math.max(0, Math.floor(rect.y));
                const x2 = Math.min(canvas.width, Math.ceil(rect.x + rect.w));
                const y2 = Math.min(canvas.height, Math.ceil(rect.y + rect.h));
                if (x2 <= x1 || y2 <= y1) return false;
                for (let y = y1; y < y2; y += block) {
                    for (let x = x1; x < x2; x += block) {
                        const sw = Math.min(block, x2 - x);
                        const sh = Math.min(block, y2 - y);
                        const cx = Math.min(canvas.width - 1, Math.max(0, x + Math.floor(sw / 2)));
                        const cy = Math.min(canvas.height - 1, Math.max(0, y + Math.floor(sh / 2)));
                        const idx = (cy * canvas.width + cx) * 4;
                        const r = source.data[idx];
                        const g = source.data[idx + 1];
                        const b = source.data[idx + 2];
                        ctx.fillStyle = `rgb(${r},${g},${b})`;
                        ctx.fillRect(x, y, sw, sh);
                    }
                }
                return true;
            };
            const start = (e) => {
                if (!ctx) return;
                activePointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
                if (activePointers.size >= 2) {
                    const pts = Array.from(activePointers.values());
                    const p1 = pts[0];
                    const p2 = pts[1];
                    pinching = true;
                    drawing = false;
                    strokePoints = [];
                    mosaicSelecting = false;
                    mosaicStartPoint = null;
                    markerState.mosaicPreviewRect = null;
                    pinchStartDist = getDist(p1, p2) || 1;
                    pinchStartScale = markerView.scale;
                    pinchStartOffset = { x: markerView.offsetX, y: markerView.offsetY };
                    pinchStartMid = getMid(p1, p2);
                    renderCropOverlay();
                    if (canvas.setPointerCapture) canvas.setPointerCapture(e.pointerId);
                    e.preventDefault();
                    return;
                }
                if (pinching) return;
                if (markerState.mode === 'crop') return;
                drawing = true;
                const p = getPoint(e);
                if (markerState.mode === 'mosaic') {
                    mosaicSelecting = true;
                    mosaicStartPoint = p;
                    markerState.mosaicPreviewRect = buildMosaicRectFromPoint(p);
                    renderCropOverlay();
                } else {
                    ensureDrawBuffers();
                    if (!drawBaseCtx || !drawLayerCtx) return;
                    drawBaseCtx.clearRect(0, 0, drawBaseCanvas.width, drawBaseCanvas.height);
                    drawBaseCtx.drawImage(canvas, 0, 0);
                    drawLayerCtx.clearRect(0, 0, drawLayerCanvas.width, drawLayerCanvas.height);
                    strokeSize = sizeInput ? Number(sizeInput.value || 16) : 16;
                    strokeStyle = normalizeMarkerHexColor(markerState.colorHex);
                    strokePoints = [];
                    appendStrokePoint(p);
                    renderStrokeLayer();
                    markerState.hasStroke = true;
                    renderDrawPreview();
                }
                if (canvas.setPointerCapture) canvas.setPointerCapture(e.pointerId);
                e.preventDefault();
            };
            const move = (e) => {
                if (activePointers.has(e.pointerId)) {
                    activePointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
                }
                if (pinching && activePointers.size >= 2) {
                    const pts = Array.from(activePointers.values());
                    const p1 = pts[0];
                    const p2 = pts[1];
                    const mid = getMid(p1, p2);
                    const dist = getDist(p1, p2) || 1;
                    const nextScale = pinchStartScale * (dist / pinchStartDist);
                    markerView.scale = Math.min(markerView.maxScale, Math.max(markerView.minScale, nextScale));
                    markerView.offsetX = pinchStartOffset.x + (mid.x - pinchStartMid.x);
                    markerView.offsetY = pinchStartOffset.y + (mid.y - pinchStartMid.y);
                    applyMarkerTransform();
                    e.preventDefault();
                    return;
                }
                if (!drawing || !ctx) return;
                const p = getPoint(e);
                if (markerState.mode === 'mosaic') {
                    if (!mosaicSelecting || !mosaicStartPoint) return;
                    markerState.mosaicPreviewRect = normalizeMosaicRect(mosaicStartPoint, p) || buildMosaicRectFromPoint(p);
                    renderCropOverlay();
                } else {
                    const changed = appendStrokePoint(p);
                    if (changed) {
                        renderStrokeLayer();
                        renderDrawPreview();
                    }
                }
                e.preventDefault();
            };
            const end = (e) => {
                const wasDrawing = drawing;
                activePointers.delete(e.pointerId);
                if (activePointers.size < 2) {
                    pinching = false;
                }
                if (activePointers.size === 0) {
                    drawing = false;
                    if (
                        wasDrawing &&
                        ctx &&
                        markerState.mode === 'draw' &&
                        strokePoints.length > 0
                    ) {
                        renderStrokeLayer();
                        renderDrawPreview();
                    }
                    if (wasDrawing && markerState.mode === 'mosaic' && mosaicStartPoint) {
                        const p = getPoint(e);
                        let rect = normalizeMosaicRect(mosaicStartPoint, p);
                        if (!rect || rect.w < 2 || rect.h < 2) {
                            rect = buildMosaicRectFromPoint(mosaicStartPoint);
                        }
                        if (applyMosaicRect(rect)) {
                            markerState.hasStroke = true;
                            markerState.mosaicRects.push(rect);
                        }
                    }
                    strokePoints = [];
                    mosaicSelecting = false;
                    mosaicStartPoint = null;
                    markerState.mosaicPreviewRect = null;
                    renderCropOverlay();
                    if (wasDrawing) {
                        saveMarkerHistory();
                    }
                }
                if (canvas.releasePointerCapture) canvas.releasePointerCapture(e.pointerId);
                e.preventDefault();
            };
            canvas.addEventListener('pointerdown', start);
            canvas.addEventListener('pointermove', move);
            canvas.addEventListener('pointerup', end);
            canvas.addEventListener('pointercancel', end);
        }
        function initCropCanvas() {
            const cropCanvas = get('marker-crop-canvas');
            if (!cropCanvas) return;
            const ctx = cropCanvas.getContext('2d');
            const activePointers = new Map();
            let dragging = false;
            let dragStart = null;
            let dragMode = null;
            let startRect = null;
            let pinching = false;
            let pinchStartDist = 0;
            let pinchStartScale = markerView.scale;
            let pinchStartOffset = { x: 0, y: 0 };
            let pinchStartMid = { x: 0, y: 0 };
            const MIN_CROP_SIZE = 8;
            const HANDLE_RADIUS = 14;
            const clamp = (val, min, max) => Math.min(max, Math.max(min, val));
            const getPoint = (e) => {
                const rect = cropCanvas.getBoundingClientRect();
                const x = (e.clientX - rect.left) * (cropCanvas.width / rect.width);
                const y = (e.clientY - rect.top) * (cropCanvas.height / rect.height);
                return { x, y };
            };
            const getMid = (a, b) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });
            const getDist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
            const ensureCropRect = () => {
                if (!markerState.cropRect) {
                    resetCropRectToFull();
                }
                return markerState.cropRect;
            };
            const hitTest = (p, rect) => {
                if (!rect) return 'move';
                const x1 = rect.x;
                const y1 = rect.y;
                const x2 = rect.x + rect.w;
                const y2 = rect.y + rect.h;
                const nearLeft = Math.abs(p.x - x1) <= HANDLE_RADIUS;
                const nearRight = Math.abs(p.x - x2) <= HANDLE_RADIUS;
                const nearTop = Math.abs(p.y - y1) <= HANDLE_RADIUS;
                const nearBottom = Math.abs(p.y - y2) <= HANDLE_RADIUS;
                if (nearLeft && nearTop) return 'nw';
                if (nearRight && nearTop) return 'ne';
                if (nearLeft && nearBottom) return 'sw';
                if (nearRight && nearBottom) return 'se';
                if (nearTop) return 'n';
                if (nearBottom) return 's';
                if (nearLeft) return 'w';
                if (nearRight) return 'e';
                const inside = (p.x > x1 + HANDLE_RADIUS && p.x < x2 - HANDLE_RADIUS && p.y > y1 + HANDLE_RADIUS && p.y < y2 - HANDLE_RADIUS);
                if (inside) return 'move';
                const outsideX = p.x < x1 ? 'left' : (p.x > x2 ? 'right' : null);
                const outsideY = p.y < y1 ? 'top' : (p.y > y2 ? 'bottom' : null);
                if (outsideX && outsideY) {
                    if (outsideX === 'left' && outsideY === 'top') return 'nw';
                    if (outsideX === 'right' && outsideY === 'top') return 'ne';
                    if (outsideX === 'left' && outsideY === 'bottom') return 'sw';
                    if (outsideX === 'right' && outsideY === 'bottom') return 'se';
                }
                if (outsideX) return outsideX === 'left' ? 'w' : 'e';
                if (outsideY) return outsideY === 'top' ? 'n' : 's';
                return 'move';
            };
            const start = (e) => {
                if (markerState.mode !== 'crop') return;
                activePointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
                if (activePointers.size >= 2) {
                    const pts = Array.from(activePointers.values());
                    const p1 = pts[0];
                    const p2 = pts[1];
                    pinching = true;
                    dragging = false;
                    pinchStartDist = getDist(p1, p2) || 1;
                    pinchStartScale = markerView.scale;
                    pinchStartOffset = { x: markerView.offsetX, y: markerView.offsetY };
                    pinchStartMid = getMid(p1, p2);
                    if (cropCanvas.setPointerCapture) cropCanvas.setPointerCapture(e.pointerId);
                    e.preventDefault();
                    return;
                }
                if (pinching) return;
                dragging = true;
                const p = getPoint(e);
                const rect = ensureCropRect();
                dragMode = hitTest(p, rect);
                dragStart = p;
                startRect = rect ? { x: rect.x, y: rect.y, w: rect.w, h: rect.h } : null;
                renderCropOverlay();
                if (cropCanvas.setPointerCapture) cropCanvas.setPointerCapture(e.pointerId);
                e.preventDefault();
            };
            const move = (e) => {
                if (markerState.mode !== 'crop') return;
                if (activePointers.has(e.pointerId)) {
                    activePointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
                }
                if (pinching && activePointers.size >= 2) {
                    const pts = Array.from(activePointers.values());
                    const p1 = pts[0];
                    const p2 = pts[1];
                    const mid = getMid(p1, p2);
                    const dist = getDist(p1, p2) || 1;
                    const nextScale = pinchStartScale * (dist / pinchStartDist);
                    markerView.scale = Math.min(markerView.maxScale, Math.max(markerView.minScale, nextScale));
                    markerView.offsetX = pinchStartOffset.x + (mid.x - pinchStartMid.x);
                    markerView.offsetY = pinchStartOffset.y + (mid.y - pinchStartMid.y);
                    applyMarkerTransform();
                    renderCropOverlay();
                    e.preventDefault();
                    return;
                }
                if (!dragging || !dragStart || !startRect) return;
                const p = getPoint(e);
                const maxW = cropCanvas.width;
                const maxH = cropCanvas.height;
                const rect = { x: startRect.x, y: startRect.y, w: startRect.w, h: startRect.h };
                const right = startRect.x + startRect.w;
                const bottom = startRect.y + startRect.h;
                const applyW = () => {
                    const nx = clamp(p.x, 0, right - MIN_CROP_SIZE);
                    rect.x = nx;
                    rect.w = right - nx;
                };
                const applyE = () => {
                    rect.w = clamp(p.x - startRect.x, MIN_CROP_SIZE, maxW - startRect.x);
                };
                const applyN = () => {
                    const ny = clamp(p.y, 0, bottom - MIN_CROP_SIZE);
                    rect.y = ny;
                    rect.h = bottom - ny;
                };
                const applyS = () => {
                    rect.h = clamp(p.y - startRect.y, MIN_CROP_SIZE, maxH - startRect.y);
                };
                switch (dragMode) {
                    case 'move': {
                        const dx = p.x - dragStart.x;
                        const dy = p.y - dragStart.y;
                        rect.x = clamp(startRect.x + dx, 0, maxW - startRect.w);
                        rect.y = clamp(startRect.y + dy, 0, maxH - startRect.h);
                        break;
                    }
                    case 'w':
                        applyW();
                        break;
                    case 'e':
                        applyE();
                        break;
                    case 'n':
                        applyN();
                        break;
                    case 's':
                        applyS();
                        break;
                    case 'nw':
                        applyN();
                        applyW();
                        break;
                    case 'ne':
                        applyN();
                        applyE();
                        break;
                    case 'sw':
                        applyS();
                        applyW();
                        break;
                    case 'se':
                        applyS();
                        applyE();
                        break;
                    default:
                        break;
                }
                rect.x = clamp(rect.x, 0, maxW - rect.w);
                rect.y = clamp(rect.y, 0, maxH - rect.h);
                markerState.cropRect = rect;
                renderCropOverlay();
                e.preventDefault();
            };
            const end = (e) => {
                activePointers.delete(e.pointerId);
                if (activePointers.size < 2) {
                    pinching = false;
                }
                if (activePointers.size === 0) {
                    renderCropOverlay();
                    dragging = false;
                    dragStart = null;
                    dragMode = null;
                    startRect = null;
                }
                if (cropCanvas.releasePointerCapture) cropCanvas.releasePointerCapture(e.pointerId);
                e.preventDefault();
            };
            cropCanvas.addEventListener('pointerdown', start);
            cropCanvas.addEventListener('pointermove', move);
            cropCanvas.addEventListener('pointerup', end);
            cropCanvas.addEventListener('pointercancel', end);
            cropCanvas.addEventListener('pointerleave', end);
        }
        async function saveMarkerToRow() {
            const row = markerState.row;
            const img = get('marker-image');
            const canvas = get('marker-canvas');
            if (!row || !img || !canvas) return;
            const attachOriginal = get('marker-attach-original');
            if (attachOriginal) {
                row.dataset.attachOriginal = attachOriginal.checked ? '1' : '';
            }
            let out = document.createElement('canvas');
            const w = markerState.naturalWidth || img.naturalWidth || canvas.width;
            const h = markerState.naturalHeight || img.naturalHeight || canvas.height;
            out.width = w;
            out.height = h;
            const octx = out.getContext('2d');
            if (!octx) return;
            octx.drawImage(img, 0, 0, w, h);
            octx.drawImage(canvas, 0, 0, w, h);
            if (markerState.cropRect) {
                const scaleX = w / canvas.width;
                const scaleY = h / canvas.height;
                const cx = Math.max(0, Math.floor(markerState.cropRect.x * scaleX));
                const cy = Math.max(0, Math.floor(markerState.cropRect.y * scaleY));
                const cw = Math.min(w, Math.max(1, Math.floor(markerState.cropRect.w * scaleX)));
                const ch = Math.min(h, Math.max(1, Math.floor(markerState.cropRect.h * scaleY)));
                const cropped = document.createElement('canvas');
                cropped.width = cw;
                cropped.height = ch;
                const cctx = cropped.getContext('2d');
                if (cctx) {
                    cctx.drawImage(out, cx, cy, cw, ch, 0, 0, cw, ch);
                    out = cropped;
                }
            }
            const blob = await new Promise((resolve) => out.toBlob(resolve, 'image/png', 0.92));
            if (!blob) {
                showToast("編集画像の生成に失敗しました", "error", true);
                return;
            }
            const originalName = markerState.filename || 'marked.png';
            const base = originalName.replace(/\.[^/.]+$/, '');
            const file = new File([blob], `${base}_marked.png`, { type: 'image/png' });
            const rowObj = {
                row,
                uploadId: row.dataset.uploadId,
                status: row.querySelector('.upload-status'),
                bar: row.querySelector('.upload-progress > div')
            };
            if (rowObj.status) rowObj.status.textContent = '編集反映中...';
            updateUploadRowFile(rowObj, file);
            const prevFilename = row.getAttribute('data-filename');
            const prevSource = getRowAttachmentSource(row);
            if (prevFilename && !row.dataset.originalFilename) {
                row.dataset.originalFilename = prevFilename;
                row.dataset.originalSource = prevSource;
                setAttachmentSourceForPath(prevFilename, prevSource);
            }
            const success = await uploadFileWithProgress(file, rowObj);
            if (success) {
                if (prevFilename) currentImageUrls = currentImageUrls.filter(x => x !== prevFilename);
                setRowAttachmentSource(row, 'upload');
                setRowMarkerState(row, true);
            } else {
                showToast("編集画像のアップロードに失敗しました", "error", true);
            }
            updateFilePreview();
            window.closeMarkerModal();
            markerState.row = null;
        }
        async function extractAudioFromVideo(file, row) {
            if (!isVideoFile(file)) return null;
            if (!HTMLMediaElement.prototype.captureStream) {
                return null;
            }
            if (row && row.status) row.status.textContent = '音声抽出中...';
            return new Promise((resolve) => {
                const video = document.createElement('video');
                video.preload = 'auto';
                video.muted = true;
                video.playsInline = true;
                video.src = URL.createObjectURL(file);
                let stream = null;
                let audioCtx = null;
                let processor = null;
                let source = null;
                let buffers = [];
                let timeoutId = null;

                const cleanup = () => {
                    if (timeoutId) clearTimeout(timeoutId);
                    try { URL.revokeObjectURL(video.src); } catch (e) {}
                    try { video.remove(); } catch (e) {}
                    if (stream) {
                        stream.getTracks().forEach(t => t.stop());
                    }
                    if (processor) {
                        try { processor.disconnect(); } catch (e) {}
                    }
                    if (source) {
                        try { source.disconnect(); } catch (e) {}
                    }
                    if (audioCtx) {
                        try { audioCtx.close(); } catch (e) {}
                    }
                };
                const fail = () => {
                    cleanup();
                    resolve(null);
                };

                video.onloadedmetadata = async () => {
                    try {
                        stream = video.captureStream();
                        const audioTracks = stream.getAudioTracks();
                        if (!audioTracks || !audioTracks.length) return fail();
                        audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
                        source = audioCtx.createMediaStreamSource(new MediaStream(audioTracks));
                        processor = audioCtx.createScriptProcessor(4096, 1, 1);
                        processor.onaudioprocess = (e) => {
                            const input = e.inputBuffer.getChannelData(0);
                            buffers.push(new Float32Array(input));
                        };
                        source.connect(processor);
                        processor.connect(audioCtx.destination);
                        const durationMs = isFinite(video.duration) ? Math.max(1, Math.ceil(video.duration * 1000)) : 0;
                        if (durationMs > 0) {
                            timeoutId = setTimeout(() => {
                                const baseName = (file.name || 'video').replace(/\.[^/.]+$/, '');
                                const wavBlob = encodeWav(buffers, audioCtx.sampleRate);
                                const audioFile = new File([wavBlob], `${baseName}.audio.wav`, { type: 'audio/wav' });
                                cleanup();
                                resolve(audioFile);
                            }, durationMs + 250);
                        }
                        await video.play();
                        video.onended = () => {
                            const baseName = (file.name || 'video').replace(/\.[^/.]+$/, '');
                            const wavBlob = encodeWav(buffers, audioCtx.sampleRate);
                            const audioFile = new File([wavBlob], `${baseName}.audio.wav`, { type: 'audio/wav' });
                            cleanup();
                            resolve(audioFile);
                        };
                    } catch (e) {
                        fail();
                    }
                };
                video.onerror = () => fail();
            });
        }
        async function handleFiles(fs, opts = {}) {
            if(!fs || !fs.length) return;
            const incoming = Array.from(fs).filter(Boolean);
            if (!incoming.length) return;
            const existingAttachments = collectImageUrlsForSend().length + Math.max(0, Number(uploadProgressState.active) || 0);
            let allowedIncoming = incoming;
            if (existingAttachments + incoming.length > ATTACHMENT_MAX_FILES) {
                const remain = Math.max(0, ATTACHMENT_MAX_FILES - existingAttachments);
                if (remain <= 0) {
                    showToast(`添付は最大${ATTACHMENT_MAX_FILES}件です`, "error", true);
                    return;
                }
                allowedIncoming = incoming.slice(0, remain);
                showToast(`添付は最大${ATTACHMENT_MAX_FILES}件です。先頭${remain}件のみ追加します。`, "warning", true);
            }
            if (opts.openModal !== false) {
                openUploadModal();
            } else {
                syncUploadRowsFromCurrent();
            }
            
            uploadProgressState.total += allowedIncoming.length;
            uploadProgressState.active += allowedIncoming.length;
            updateFilePreview();

            const audioOnly = !!(get('upload-audio-only') && get('upload-audio-only').checked);
            const support = getModelMediaSupport(get('model-select').value);
            const processOne = async (f) => {
                let currentUploadId = null;
                try {
                    if (isAudioFile(f) && !support.audio) {
                        showToast("このモデルは音声入力に対応していません", "error", true);
                        if (uploadProgressState.total > 0) uploadProgressState.total--;
                        if (uploadProgressState.active > 0) uploadProgressState.active--;
                        return false;
                    }
                    if (isVideoFile(f) && !support.video) {
                        showToast("このモデルは動画入力に対応していません", "error", true);
                        if (uploadProgressState.total > 0) uploadProgressState.total--;
                        if (uploadProgressState.active > 0) uploadProgressState.active--;
                        return false;
                    }
                    const rowObj = addUploadRow(f);
                    updateFilePreview(); // Immediate feedback
                    currentUploadId = rowObj.uploadId;
                    uploadProgressState.perFilePct[currentUploadId] = 0;

                    let t = f;
                    if (audioOnly && isVideoFile(f)) {
                        const audioFile = await extractAudioFromVideo(f, rowObj);
                        if (audioFile) {
                            t = audioFile;
                            updateUploadRowFile(rowObj, audioFile);
                            if (rowObj && rowObj.status) rowObj.status.textContent = '音声のみ';
                        } else {
                            if (rowObj && rowObj.status) rowObj.status.textContent = '抽出失敗: 動画送信';
                            showToast("音声抽出に失敗しました。動画のまま送信します。", "error", true);
                        }
                    }
                    if (get('enable-compression').checked && f.type.startsWith('image/')) {
                        try {
                            const outputType = getCompressionOutputType();
                            const formatOnly = getCompressionFormatOnly();
                            const o = {
                                maxSizeMB: formatOnly ? 50 : getCompressionMaxSizeMB(),
                                maxWidthOrHeight: formatOnly ? 16384 : getCompressionMaxDim(),
                                useWebWorker: true
                            };
                            if (outputType && outputType !== 'original') {
                                o.fileType = outputType;
                            } else if (formatOnly) {
                                // If formatOnly is true but outputType is original, do nothing.
                                throw new Error('formatOnly with original type');
                            }
                            await ensureImageCompression();
                            const c = await window.imageCompression(f, o);
                            const compressedFile = new File([c], f.name, { type: c.type });
                            if (compressedFile.size > f.size) {
                                showToast(`圧縮後にサイズが増加しました: ${formatBytes(f.size)} -> ${formatBytes(compressedFile.size)}（元ファイルを使用）`, "warning", true);
                                t = f;
                            } else {
                                t = compressedFile;
                            }
                        } catch(e){}
                    }
                    return await uploadFileWithProgress(t, rowObj);
                } finally {
                    if (currentUploadId) {
                        if (uploadProgressState.perFilePct.hasOwnProperty(currentUploadId)) {
                            delete uploadProgressState.perFilePct[currentUploadId];
                            uploadProgressState.completed++;
                            uploadProgressState.active--;
                        }
                    }
                    if (uploadProgressState.active <= 0) {
                        uploadProgressState.total = 0;
                        uploadProgressState.completed = 0;
                        uploadProgressState.active = 0;
                        uploadProgressState.perFilePct = {};
                    }
                    updateFilePreview();
                }
            };
            let cursor = 0;
            const workerCount = Math.min(UPLOAD_CONCURRENCY, allowedIncoming.length);
            const workers = Array.from({ length: workerCount }).map(async () => {
                while (true) {
                    const idx = cursor++;
                    if (idx >= allowedIncoming.length) break;
                    await processOne(allowedIncoming[idx]);
                }
            });
            await Promise.all(workers);
        }
        get('clear-file-btn').onclick = () => { resetUploadState(); };
        if (get('clear-mask-btn')) {
            get('clear-mask-btn').onclick = () => {
                currentMaskImage = null;
                updateMaskPreview();
            };
        }
        if (get('mask-btn') && get('mask-input')) {
            get('mask-btn').onclick = () => {
                get('mask-input').click();
            };
            get('mask-input').addEventListener('change', async (e) => {
                const f = e.target.files && e.target.files[0];
                if (!f) return;
                await uploadMaskFile(f);
                e.target.value = '';
            });
        }
        
        const messageMeta = {};
        function sanitizeMarkdownHtml(text) {
            return DOMPurify.sanitize(marked.parse(text || ''));
        }
        function getCanvasModeElements() {
            const panel = get('canvas-panel');
            if (!panel) return null;
            return {
                panel,
                stage: get('conversation-stage'),
                title: get('canvas-panel-title'),
                status: get('canvas-panel-status'),
                blockCount: get('canvas-block-count'),
                blockList: get('canvas-block-list'),
                panelTabs: get('canvas-panel-tabs'),
                previewLang: get('canvas-preview-lang'),
                sourceLang: get('canvas-source-lang'),
                frame: get('canvas-preview-frame'),
                empty: get('canvas-preview-empty'),
                sourceScroll: get('canvas-source-scroll'),
                code: get('canvas-code-text'),
                copyBtn: get('canvas-panel-copy-btn'),
                clearBtn: get('canvas-panel-clear-btn'),
                closeBtn: get('canvas-panel-close-btn')
            };
        }
        function isCanvasHtmlPreviewCandidate(lang, code) {
            const token = String(lang || '').trim().toLowerCase();
            if (token === 'html' || token === 'htm' || token === 'xhtml') return true;
            if (token) return false;
            const raw = String(code || '');
            return /<!doctype\s+html/i.test(raw) || /<html[\s>]/i.test(raw);
        }
        function normalizeCanvasBlock(block, index) {
            const lang = String(block && block.lang ? block.lang : '').trim();
            const code = String(block && block.code !== undefined && block.code !== null ? block.code : '');
            const open = !!(block && block.open);
            return {
                ...block,
                index,
                lang,
                code,
                open,
                key: hashString(`${lang || 'TEXT'}\n${code || ''}`)
            };
        }
        function parseCanvasMarkdown(text) {
            const rawText = String(text || '');
            const lines = rawText.split(/\r?\n/);
            const blocks = [];
            const output = [];
            const fenceStartRe = /^(\s*)(```|~~~)(.*)$/;
            let activeFence = null;
            let activeLang = '';
            let activeBuffer = [];
            for (const line of lines) {
                if (!activeFence) {
                    const match = line.match(fenceStartRe);
                    if (match) {
                        activeFence = match[2];
                        activeLang = String(match[3] || '').trim();
                        activeBuffer = [];
                        blocks.push({ lang: activeLang, code: '', open: true });
                        output.push('<div class="canvas-code-placeholder">Canvasで表示中</div>');
                        continue;
                    }
                    output.push(line);
                    continue;
                }
                const trimmed = String(line || '').trim();
                if (trimmed && trimmed.replace(/\s+/g, '') === activeFence) {
                    const block = blocks[blocks.length - 1];
                    if (block) {
                        block.code = activeBuffer.join('\n');
                        block.open = false;
                    }
                    activeFence = null;
                    activeLang = '';
                    activeBuffer = [];
                    continue;
                }
                activeBuffer.push(line);
                const block = blocks[blocks.length - 1];
                if (block) block.code = activeBuffer.join('\n');
            }
            if (activeFence && blocks.length) {
                const block = blocks[blocks.length - 1];
                if (block) {
                    block.code = activeBuffer.join('\n');
                    block.open = true;
                }
            }
            const normalizedBlocks = blocks.map((block, index) => normalizeCanvasBlock(block, index));
            const primarySelection = selectCanvasPreviewBlock(normalizedBlocks, rawText);
            return {
                renderText: output.join('\n'),
                blocks: normalizedBlocks,
                primaryBlock: primarySelection ? primarySelection.block : null,
                primaryIndex: primarySelection ? primarySelection.index : -1,
                rawText
            };
        }
        function selectCanvasPreviewBlock(blocks, rawText = '', preferredIndex = -1) {
            const list = Array.isArray(blocks) ? blocks : [];
            if (Number.isInteger(preferredIndex) && preferredIndex >= 0 && preferredIndex < list.length) {
                const preferred = list[preferredIndex];
                return {
                    block: preferred,
                    index: preferredIndex,
                    previewType: isCanvasHtmlPreviewCandidate(preferred.lang, preferred.code) ? 'html' : 'code'
                };
            }
            for (let i = list.length - 1; i >= 0; i--) {
                const block = list[i];
                if (!block) continue;
                if (isCanvasHtmlPreviewCandidate(block.lang, block.code)) {
                    return { block, index: i, previewType: 'html' };
                }
            }
            if (list.length > 0) {
                return { block: list[list.length - 1], index: list.length - 1, previewType: 'code' };
            }
            const raw = String(rawText || '');
            if (isCanvasHtmlPreviewCandidate('', raw)) {
                const fallback = normalizeCanvasBlock({ lang: 'html', code: raw, open: true, fallback: true }, 0);
                return { block: fallback, index: -1, previewType: 'html' };
            }
            return null;
        }
        function getCanvasSelectedBlock() {
            const blocks = Array.isArray(canvasPreviewState.blocks) ? canvasPreviewState.blocks : [];
            if (!blocks.length) {
                const raw = String(canvasPreviewState.rawText || '');
                if (isCanvasHtmlPreviewCandidate('', raw)) {
                    return { block: normalizeCanvasBlock({ lang: 'html', code: raw, open: true, fallback: true }, 0), index: -1 };
                }
                return null;
            }
            const preferredIndex = Number.isInteger(canvasPreviewState.selectedIndex) ? canvasPreviewState.selectedIndex : -1;
            const selection = selectCanvasPreviewBlock(blocks, canvasPreviewState.rawText, preferredIndex);
            if (!selection || !selection.block) return null;
            return selection;
        }
        function syncCanvasPreviewButtons(root = document) {
            if (!root || typeof root.querySelectorAll !== 'function') return;
            const selectedKey = String(canvasPreviewState.selectedKey || '');
            root.querySelectorAll('.canvas-preview-btn').forEach((btn) => {
                const codeKey = String(btn.getAttribute('data-code-key') || '');
                const isActive = !!selectedKey && selectedKey === codeKey;
                btn.classList.toggle('canvas-active', isActive);
                btn.setAttribute('aria-pressed', isActive ? 'true' : 'false');
                btn.setAttribute('data-canvas-active', isActive ? '1' : '0');
                btn.innerHTML = isActive
                    ? '<i class="fas fa-layer-group"></i> Canvasで表示中'
                    : '<i class="fas fa-window-restore"></i> Canvasに表示';
                btn.title = isActive ? 'Canvasで表示中' : 'Canvasでプレビューする';
            });
        }
        function isCanvasMobileLayout() {
            try {
                return window.matchMedia('(max-width: 1023px)').matches;
            } catch (e) {
                return false;
            }
        }
        function syncCanvasPanelViewUi(view = canvasPreviewState.mobileView, options = {}) {
            const els = getCanvasModeElements();
            if (!els || !els.panel) return;
            const nextView = ['preview', 'blocks', 'source'].includes(view) ? view : 'preview';
            canvasPreviewState.mobileView = nextView;
            els.panel.dataset.canvasMobileView = nextView;
            const tabs = els.panelTabs ? Array.from(els.panelTabs.querySelectorAll('[data-canvas-panel-view]')) : [];
            tabs.forEach((btn) => {
                const active = btn.getAttribute('data-canvas-panel-view') === nextView;
                btn.classList.toggle('active', active);
                btn.setAttribute('aria-pressed', active ? 'true' : 'false');
            });
            if (options.focus !== false && isCanvasMobileLayout()) {
                if (nextView === 'preview' && els.frame && !els.frame.classList.contains('hidden')) {
                    els.frame.focus({ preventScroll: true });
                } else if (nextView === 'source' && els.sourceScroll) {
                    els.sourceScroll.focus({ preventScroll: true });
                } else if (nextView === 'blocks' && els.blockList) {
                    els.blockList.focus?.({ preventScroll: true });
                }
            }
        }
        function renderCanvasBlockChips() {
            const els = getCanvasModeElements();
            if (!els || !els.blockList) return;
            const blocks = Array.isArray(canvasPreviewState.blocks) ? canvasPreviewState.blocks : [];
            if (els.blockCount) els.blockCount.textContent = String(blocks.length);
            if (!blocks.length) {
                els.blockList.innerHTML = '<div class="px-2 py-3 text-xs text-gray-500">コードブロックを待機中</div>';
                return;
            }
            const selectedIndex = Number.isInteger(canvasPreviewState.selectedIndex) ? canvasPreviewState.selectedIndex : -1;
            els.blockList.innerHTML = blocks.map((block, index) => {
                const lang = String(block && block.lang ? block.lang : 'text').trim() || 'text';
                const selected = index === selectedIndex;
                const stateLabel = block && block.open ? '生成中' : '表示';
                const title = `${selected ? '現在表示中' : '切り替え'}: ${lang}`;
                return `<button type="button" class="canvas-block-chip${selected ? ' active' : ''}" data-canvas-block-index="${index}" title="${escapeHtml(title)}" aria-pressed="${selected ? 'true' : 'false'}"><span class="canvas-block-chip-index">#${index + 1}</span><span class="canvas-block-chip-lang">${escapeHtml(lang)}</span><span class="canvas-block-chip-state">${selected ? '表示中' : stateLabel}</span></button>`;
            }).join('');
        }
        function showCanvasPreviewPanel() {
            const els = getCanvasModeElements();
            if (!els) return;
            canvasPreviewState.panelAnimationToken += 1;
            const token = canvasPreviewState.panelAnimationToken;
            if (canvasPreviewState.panelHideTimer) {
                clearTimeout(canvasPreviewState.panelHideTimer);
                canvasPreviewState.panelHideTimer = null;
            }
            els.panel.classList.remove('hidden', 'canvas-closing');
            if (els.stage) els.stage.classList.add('canvas-enabled');
            requestAnimationFrame(() => {
                if (token !== canvasPreviewState.panelAnimationToken) return;
                els.panel.classList.add('canvas-panel-open');
            });
        }
        function hideCanvasPreviewPanel(animate = true) {
            const els = getCanvasModeElements();
            if (!els) return;
            canvasPreviewState.panelAnimationToken += 1;
            if (canvasPreviewState.panelHideTimer) {
                clearTimeout(canvasPreviewState.panelHideTimer);
                canvasPreviewState.panelHideTimer = null;
            }
            if (!animate) {
                els.panel.classList.add('hidden');
                els.panel.classList.remove('canvas-panel-open', 'canvas-closing');
                if (els.stage) els.stage.classList.remove('canvas-enabled');
                return;
            }
            els.panel.classList.remove('canvas-panel-open');
            els.panel.classList.add('canvas-closing');
            canvasPreviewState.panelHideTimer = window.setTimeout(() => {
                els.panel.classList.add('hidden');
                els.panel.classList.remove('canvas-closing');
                if (els.stage) els.stage.classList.remove('canvas-enabled');
                canvasPreviewState.panelHideTimer = null;
            }, 220);
        }
        function resetCanvasPreviewPanel(message = 'Canvasで表示中') {
            const els = getCanvasModeElements();
            if (!els) return;
            canvasPreviewState.blocks = [];
            canvasPreviewState.rawText = '';
            canvasPreviewState.renderText = '';
            canvasPreviewState.selectedIndex = -1;
            canvasPreviewState.selectedKey = '';
            canvasPreviewState.mobileView = 'preview';
            canvasPreviewState.lastCanvasData = null;
            showCanvasPreviewPanel();
            syncCanvasPanelViewUi('preview', { focus: false });
            if (els.title) els.title.textContent = message;
            if (els.status) els.status.textContent = 'コードブロックを待機中';
            if (els.previewLang) els.previewLang.textContent = 'idle';
            if (els.sourceLang) els.sourceLang.textContent = '-';
            if (els.code) els.code.textContent = '';
            if (els.blockCount) els.blockCount.textContent = '0';
            if (els.blockList) els.blockList.innerHTML = '<div class="px-2 py-3 text-xs text-gray-500">コードブロックを待機中</div>';
            if (els.sourceScroll) els.sourceScroll.scrollTop = 0;
            if (els.frame) {
                els.frame.removeAttribute('srcdoc');
                els.frame.classList.add('hidden');
            }
            if (els.empty) els.empty.classList.remove('hidden');
            syncCanvasPreviewButtons();
        }
        function updateCanvasPreviewState(canvasData = null) {
            const data = canvasData || canvasPreviewState.lastCanvasData;
            if (!data) return null;
            canvasPreviewState.lastCanvasData = data;
            canvasPreviewState.blocks = Array.isArray(data.blocks) ? data.blocks.slice() : [];
            canvasPreviewState.rawText = String(data.rawText || '');
            canvasPreviewState.renderText = String(data.renderText || '');
            const blocks = canvasPreviewState.blocks;
            if (!blocks.length) {
                const fallbackSelection = selectCanvasPreviewBlock([], canvasPreviewState.rawText);
                if (fallbackSelection && fallbackSelection.block) {
                    canvasPreviewState.selectedIndex = -1;
                    canvasPreviewState.selectedKey = fallbackSelection.block.key || '';
                    return fallbackSelection.block;
                }
                canvasPreviewState.selectedIndex = -1;
                canvasPreviewState.selectedKey = '';
                return null;
            }
            const currentIndex = Number.isInteger(canvasPreviewState.selectedIndex) ? canvasPreviewState.selectedIndex : -1;
            let nextIndex = currentIndex;
            if (nextIndex < 0 || nextIndex >= blocks.length) {
                nextIndex = Number.isInteger(data.primaryIndex) && data.primaryIndex >= 0 && data.primaryIndex < blocks.length ? data.primaryIndex : blocks.length - 1;
            }
            const nextBlock = blocks[nextIndex] || null;
            canvasPreviewState.selectedIndex = nextBlock ? nextIndex : -1;
            canvasPreviewState.selectedKey = nextBlock && nextBlock.key ? nextBlock.key : '';
            return nextBlock;
        }
        function refreshCanvasPreviewPanel() {
            const els = getCanvasModeElements();
            if (!els || !canvasModeEnabled) return;
            showCanvasPreviewPanel();
            syncCanvasPanelViewUi(canvasPreviewState.mobileView || 'preview', { focus: false });
            const blocks = Array.isArray(canvasPreviewState.blocks) ? canvasPreviewState.blocks : [];
            const selected = getCanvasSelectedBlock();
            const block = selected && selected.block ? selected.block : null;
            const index = selected && Number.isInteger(selected.index) ? selected.index : -1;
            const hasBlock = !!block;
            const blockLang = String(block && block.lang ? block.lang : '').trim();
            const code = String(block && block.code !== undefined && block.code !== null ? block.code : '');
            const isHtml = hasBlock ? isCanvasHtmlPreviewCandidate(blockLang, code) : false;
            const statusText = !hasBlock
                ? 'コードブロックを待機中'
                : (isHtml ? 'HTML をリアルタイムでプレビューしています' : (block && block.open ? 'コードブロックを生成中' : 'コードブロックをプレビューしています'));
            const titleText = !hasBlock
                ? 'Canvasで表示中'
                : (isHtml ? `HTML Canvas Preview${blocks.length > 1 && index >= 0 ? ` #${index + 1}/${blocks.length}` : ''}` : `Canvas Preview: ${blockLang || 'text'}${blocks.length > 1 && index >= 0 ? ` #${index + 1}/${blocks.length}` : ''}`);
            if (els.title) els.title.textContent = titleText;
            if (els.status) els.status.textContent = statusText;
            if (els.previewLang) els.previewLang.textContent = hasBlock ? (blockLang || 'text') : 'idle';
            if (els.sourceLang) els.sourceLang.textContent = hasBlock ? (blockLang || 'text') : '-';
            if (els.code) els.code.textContent = code;
            if (els.sourceScroll) els.sourceScroll.scrollTop = 0;
            if (els.blockCount) els.blockCount.textContent = String(blocks.length);
            renderCanvasBlockChips();

            if (hasBlock) {
                const previewDoc = buildCanvasPreviewDocument(block);
                if (els.frame) {
                    els.frame.srcdoc = previewDoc;
                    els.frame.classList.remove('hidden');
                }
                if (els.empty) els.empty.classList.add('hidden');
            } else {
                if (els.frame) {
                    els.frame.removeAttribute('srcdoc');
                    els.frame.classList.add('hidden');
                }
                if (els.empty) els.empty.classList.remove('hidden');
            }
            syncCanvasPreviewButtons();
        }
        function applyCanvasSelection(index) {
            const blocks = Array.isArray(canvasPreviewState.blocks) ? canvasPreviewState.blocks : [];
            if (!blocks.length) return false;
            const nextIndex = Number(index);
            if (!Number.isInteger(nextIndex) || nextIndex < 0 || nextIndex >= blocks.length) return false;
            canvasPreviewState.selectedIndex = nextIndex;
            canvasPreviewState.selectedKey = blocks[nextIndex] && blocks[nextIndex].key ? blocks[nextIndex].key : '';
            syncCanvasPanelViewUi('preview', { focus: false });
            renderCanvasBlockChips();
            syncCanvasPreviewButtons();
            refreshCanvasPreviewPanel();
            return true;
        }
        function applyCanvasSelectionByKey(codeKey) {
            const blocks = Array.isArray(canvasPreviewState.blocks) ? canvasPreviewState.blocks : [];
            if (!blocks.length) return false;
            const targetKey = String(codeKey || '');
            if (!targetKey) return false;
            const index = blocks.findIndex((block) => block && block.key === targetKey);
            if (index === -1) return false;
            return applyCanvasSelection(index);
        }
        function decodeCanvasPreviewButtonCode(btn) {
            if (!btn) return null;
            const codeEnc = btn.getAttribute('data-code') || '';
            if (!codeEnc) return null;
            let code = '';
            try {
                code = decodeURIComponent(codeEnc);
            } catch (e) {
                code = codeEnc;
            }
            const lang = String(btn.getAttribute('data-canvas-lang') || btn.getAttribute('data-lang') || '').trim();
            const codeKey = String(btn.getAttribute('data-code-key') || hashString(`${lang || 'TEXT'}\n${code || ''}`));
            return { code, lang, codeKey };
        }
        function collectCanvasBlocksFromButton(btn) {
            const payload = decodeCanvasPreviewButtonCode(btn);
            if (!payload) return null;
            const group = btn && typeof btn.closest === 'function' ? btn.closest('.message-group') : null;
            const sourceBtns = group ? Array.from(group.querySelectorAll('.canvas-preview-btn')) : [];
            if (!sourceBtns.length) {
                const block = normalizeCanvasBlock({ lang: payload.lang, code: payload.code, open: false }, 0);
                return { blocks: [block], selectedIndex: 0, selectedKey: block.key || payload.codeKey || '' };
            }
            const blocks = [];
            let selectedIndex = -1;
            sourceBtns.forEach((sourceBtn, index) => {
                const sourcePayload = decodeCanvasPreviewButtonCode(sourceBtn);
                if (!sourcePayload) return;
                const block = normalizeCanvasBlock({ lang: sourcePayload.lang, code: sourcePayload.code, open: false }, index);
                blocks.push(block);
                if (selectedIndex === -1 && sourcePayload.codeKey === payload.codeKey) {
                    selectedIndex = blocks.length - 1;
                }
            });
            if (!blocks.length) return null;
            if (selectedIndex === -1) selectedIndex = 0;
            const selected = blocks[selectedIndex] || blocks[0] || null;
            return {
                blocks,
                selectedIndex,
                selectedKey: selected && selected.key ? selected.key : payload.codeKey || ''
            };
        }
        function previewCanvasCodeFromButton(btn) {
            if (!btn) return false;
            const payload = collectCanvasBlocksFromButton(btn);
            if (!payload || !payload.blocks || !payload.blocks.length) return false;
            const currentBlocks = Array.isArray(canvasPreviewState.blocks) ? canvasPreviewState.blocks : [];
            const matchedIndex = currentBlocks.findIndex((block) => block && block.key === payload.selectedKey);
            if (matchedIndex !== -1 && currentBlocks.length > 1) {
                return applyCanvasSelection(matchedIndex);
            }
            const selectedBlock = payload.blocks[payload.selectedIndex] || payload.blocks[0] || null;
            canvasPreviewState.blocks = payload.blocks;
            canvasPreviewState.rawText = selectedBlock && selectedBlock.code !== undefined && selectedBlock.code !== null ? String(selectedBlock.code) : '';
            canvasPreviewState.renderText = canvasPreviewState.rawText;
            canvasPreviewState.selectedIndex = Number.isInteger(payload.selectedIndex) ? payload.selectedIndex : 0;
            canvasPreviewState.selectedKey = payload.selectedKey || (selectedBlock && selectedBlock.key) || '';
            canvasPreviewState.lastCanvasData = {
                renderText: canvasPreviewState.renderText,
                blocks: payload.blocks,
                primaryBlock: selectedBlock,
                primaryIndex: canvasPreviewState.selectedIndex,
                rawText: canvasPreviewState.rawText
            };
            canvasPreviewState.mobileView = 'preview';
            syncCanvasPanelViewUi('preview', { focus: false });
            refreshCanvasPreviewPanel();
            return true;
        }
        function buildCanvasPreviewDocument(block) {
            const code = String(block && block.code !== undefined && block.code !== null ? block.code : '');
            const lang = String(block && block.lang ? block.lang : '').trim().toLowerCase();
            const isHtml = isCanvasHtmlPreviewCandidate(lang, code);
            if (isHtml) {
                return sanitizeHtmlForPreview(code);
            }
            const title = lang ? `Canvas Preview: ${lang}` : 'Canvas Preview';
            const safeCode = escapeHtml(code || '');
            return `<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>${escapeHtml(title)}</title><style>
                :root { color-scheme: dark; }
                html, body { margin: 0; min-height: 100%; background: #0b1220; color: #e5e7eb; font-family: "Noto Sans JP", system-ui, -apple-system, "Segoe UI", sans-serif; }
                body { box-sizing: border-box; padding: 16px; }
                .frame {
                    background: linear-gradient(180deg, rgba(15, 23, 42, 0.92), rgba(2, 6, 23, 0.94));
                    border: 1px solid rgba(148, 163, 184, 0.18);
                    border-radius: 14px;
                    padding: 14px;
                    box-shadow: 0 20px 48px rgba(0, 0, 0, 0.34);
                }
                .label {
                    font-size: 11px;
                    text-transform: uppercase;
                    letter-spacing: 0.14em;
                    color: #67e8f9;
                    margin-bottom: 10px;
                }
                pre {
                    margin: 0;
                    white-space: pre-wrap;
                    word-break: break-word;
                    overflow-wrap: anywhere;
                    font-family: "JetBrains Mono", "Noto Sans Mono", ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
                    font-size: 13px;
                    line-height: 1.6;
                    color: #e2e8f0;
                }
                .muted { color: #94a3b8; }
            </style></head><body><div class="frame"><div class="label">${escapeHtml(title)}</div><pre>${safeCode || '<span class="muted">Canvasで表示中</span>'}</pre></div></body></html>`;
        }
        function syncCanvasModeUi(enabled = canvasModeEnabled, options = {}) {
            const persist = options.persist !== false;
            canvasModeEnabled = !!enabled;
            if (persist) {
                try {
                    localStorage.setItem(CANVAS_MODE_STORAGE_KEY, canvasModeEnabled ? 'true' : 'false');
                } catch (e) {}
            }
            const checkbox = get('enable-canvas-mode');
            if (checkbox && checkbox.checked !== canvasModeEnabled) checkbox.checked = canvasModeEnabled;
            if (!canvasModeEnabled) {
                hideCanvasPreviewPanel(options.animate !== false);
                if (!activeStreamingBubbleId && currentThreadId) {
                    try {
                        renderThreadTree({ silent: true, keepScroll: true });
                    } catch (e) {}
                }
                return;
            }
            showCanvasPreviewPanel();
            if (isCanvasMobileLayout()) {
                syncCanvasPanelViewUi('preview', { focus: false });
            }
            syncCanvasPanelViewUi(canvasPreviewState.mobileView || 'preview', { focus: false });
            if (options.skipReset) return;
            if (!activeStreamingBubbleId) {
                resetCanvasPreviewPanel();
                if (currentThreadId) {
                    try {
                        renderThreadTree({ silent: true, keepScroll: true });
                    } catch (e) {}
                }
            } else {
                refreshCanvasPreviewPanel();
            }
        }
        function buildAiMarkdownHtml(text) {
            const canvasData = canvasModeEnabled ? parseCanvasMarkdown(text) : { renderText: text || '', blocks: [], primaryBlock: null, rawText: String(text || '') };
            if (canvasModeEnabled) {
                updateCanvasPreviewState(canvasData);
                refreshCanvasPreviewPanel();
            }
            const wrap = document.createElement('div');
            wrap.className = 'prose prose-invert text-sm break-words';
            wrap.innerHTML = sanitizeMarkdownHtml(canvasData.renderText);
            wrapRenderedSvgBoxes(wrap);
            if (!lowBandwidthMode) {
                if (maybeNeedsHighlight(canvasData.renderText, wrap)) ensureHighlightLoaded().catch(() => {});
                if (maybeNeedsMathJax(canvasData.renderText)) ensureMathJaxLoaded().catch(() => {});
            }
            return wrap.outerHTML;
        }
        function renderAiMarkdownInto(container, text) {
            if (!container) return;
            const canvasData = canvasModeEnabled ? parseCanvasMarkdown(text) : { renderText: text || '', blocks: [], primaryBlock: null, rawText: String(text || '') };
            if (canvasModeEnabled) {
                updateCanvasPreviewState(canvasData);
                refreshCanvasPreviewPanel();
            }
            container.innerHTML = sanitizeMarkdownHtml(canvasData.renderText);
            wrapRenderedSvgBoxes(container);
            queueMessageDecorations(container, canvasData.renderText);
        }
        function wrapRenderedSvgBoxes(root) {
            if (!root || typeof root.querySelectorAll !== 'function') return;
            root.querySelectorAll('svg').forEach((svg) => {
                if (!svg || !svg.parentNode) return;
                if (svg.closest('.svg-render-box')) return;
                if (svg.closest('pre, code, .code-wrapper, .thought-container')) return;
                const frame = document.createElement('span');
                frame.className = 'svg-render-box';
                svg.parentNode.insertBefore(frame, svg);
                frame.appendChild(svg);
            });
        }
        function renderMessage(id, role, text, imgUrl, thoughtData, modelName, versionInfo = null, animate = true, quoteText = null, tokenCount = null, tokenIn = null, tokenOut = null, isEncrypted = null, tokensContent = null, tokensThought = null, target = null, doScroll = true, parentId = null, gemName = null) { 
            const isUser = role === 'user'; 
            const bg = isUser ? 'bg-blue-600' : 'bg-gray-700'; 
            const align = isUser ? 'justify-end' : 'justify-start'; 
            messageStore[id] = text; 
            let totalTokens = tokenCount;
            if (totalTokens === null || totalTokens === undefined) {
                const inVal = (tokenIn !== null && tokenIn !== undefined) ? Number(tokenIn) : 0;
                const outVal = (tokenOut !== null && tokenOut !== undefined) ? Number(tokenOut) : 0;
                if ((tokenIn !== null && tokenIn !== undefined) || (tokenOut !== null && tokenOut !== undefined)) {
                    totalTokens = inVal + outVal;
                }
            }
            messageMeta[id] = {
                tokens_in: tokenIn,
                tokens_out: tokenOut,
                tokens_total: totalTokens,
                tokens_content: tokensContent,
                tokens_thought: tokensThought,
                is_encrypted: isEncrypted,
                role: role,
                model: modelName,
                parent_id: parentId,
                quote_text: quoteText,
                image_url: imgUrl,
                gem_name: gemName
            };
            
            let qh = '';
            if (quoteText) {
                qh = `<div class="mb-2 p-2 bg-black/20 rounded border-l-4 border-blue-400 text-xs text-gray-300 italic truncate max-w-full"><i class="fas fa-quote-left mr-1 opacity-50"></i>${escapeHtml(quoteText)}</div>`;
            }

            let th = ''; 
            if (thoughtData && !isUser) { 
                let tx = ""; 
                try { const o = JSON.parse(thoughtData); tx = o.text || ""; } catch(e) { tx = thoughtData; } 
                if (tx) th = `<div class="thought-container"><div class="thought-header" onclick="toggleThinking(this)"><i class="fas fa-brain text-purple-400"></i> Thinking Process</div><div class="thought-content collapsed">${escapeHtml(tx)}</div></div>`; 
            } 
            
            let at = ''; 
            if(imgUrl) { 
                try { 
                    const imgs = JSON.parse(imgUrl); 
                    if(imgs.length) { 
                        const items = [];
                        imgs.forEach(u => { 
                            let path = u;
                            let source = 'unknown';
                            if (path && typeof path === 'object') {
                                source = normalizeAttachmentSource(path.source);
                                path = path.filepath || path.path || path.url || path.file || '';
                            }
                            path = normalizeAttachmentPath(path) || path;
                            if (!path) return;
                            setAttachmentSourceForPath(path, source);
                            const displayPath = path.replace(/^\d+\//, '');
                            const url = buildFileUrl(displayPath);
                            const previewUrl = buildAttachmentPreviewUrl(displayPath);
                            const fn = path.split('/').pop(); 
                            const ext = fn.split('.').pop().toLowerCase(); 
                            if(['jpg','jpeg','png','webp','gif'].includes(ext)) {
                                items.push(`<img src="${previewUrl}" data-viewer-src="${url}" data-viewer-filename="${escapeHtml(fn)}" class="chat-image" loading="lazy" onclick="openImageViewer('${url}')" title="${fn}">`);
                            } else {
                                items.push(`<div class="file-thumb bg-gray-800 border border-gray-600 rounded flex flex-col items-center justify-center cursor-pointer hover:bg-gray-700" onclick="window.open('${url}')" title="${fn}"><i class="fas fa-file text-2xl text-gray-400 mb-1"></i><span class="text-[9px] truncate w-20 text-center">${fn}</span></div>`);
                            }
                        }); 
                        
                        if (items.length > 0) {
                            let gridClass = 'grid-multi';
                            if (items.length === 1) gridClass = 'grid-1';
                            else if (items.length === 2) gridClass = 'grid-2';
                            else if (items.length === 3) gridClass = 'grid-3';
                            else if (items.length === 4) gridClass = 'grid-4';
                            
                            at = `<div class="image-grid ${gridClass}">${items.join('')}</div>`;
                        }
                    } 
                } catch(e){} 
            } 
            
            const regenBtn = !isUser ? `<button class="ctrl-btn" onclick="regenerateMessage('${id}')"><i class="fas fa-rotate-right"></i></button>` : '';
            const ctrl = `<div class="msg-controls absolute -top-5 right-0 hidden group-hover:flex gap-1 z-10"><button class="ctrl-btn" onclick="window.copyMessage('${id}', this)"><i class="fas fa-copy"></i></button>${isUser?`<button class="ctrl-btn edit-btn" data-id="${id}"><i class="fas fa-pen"></i></button>`:''}${regenBtn}<button class="ctrl-btn" onclick="deleteMessage('${id}')"><i class="fas fa-trash"></i></button></div>`; 
            const footerParts = [];
            if (!isUser && modelName) footerParts.push(escapeHtml(modelName));
            if (gemName) {
                if (isUser) {
                    footerParts.push(`<span class="text-purple-300/90"><i class="fas fa-gem mr-0.5"></i>${escapeHtml(gemName)}</span>`);
                } else {
                    footerParts.push(`<span class="text-purple-300/90"><i class="fas fa-gem mr-0.5"></i>${escapeHtml(gemName)}</span>`);
                }
            }
            const tokenParts = [];
            if (tokenIn !== null && tokenIn !== undefined) tokenParts.push(`In ${tokenIn}`);
            if (tokenOut !== null && tokenOut !== undefined) {
                let outLabel = `Out ${tokenOut}`;
                if (tokensThought !== null && tokensThought !== undefined && Number(tokensThought) > 0) {
                    outLabel += ` (Thought ${tokensThought})`;
                }
                tokenParts.push(outLabel);
            }
            if (tokenParts.length || (tokenCount !== null && tokenCount !== undefined)) {
                const tLabel = tokenParts.length ? tokenParts.join(' / ') : `${tokenCount} tokens`;
                footerParts.push(`<button class="underline decoration-dotted hover:text-white token-detail-btn" onclick="openTokenDetail('${id}')">${tLabel}</button>`);
            }
            if (isEncrypted !== null && isEncrypted !== undefined) {
                const lockIcon = isEncrypted ? 'fa-lock' : 'fa-lock-open';
                const lockTitle = isEncrypted ? 'Encrypted' : 'Plain';
                footerParts.push(`<button class="text-slate-300/80 hover:text-white" title="${lockTitle}" onclick="openEncryptionSettings('${id}')"><i class="fas ${lockIcon}"></i></button>`);
            }
            const mHtml = footerParts.length ? `<div class="text-[10px] text-slate-300/90 mt-2 text-right font-mono">${footerParts.join(' • ')}</div>` : ''; 
            
            let contentHtml;
            if (isUser) {
                // User message: RAW TEXT DISPLAY (Preserve whitespace, no markdown)
                contentHtml = `<div class="content-area whitespace-pre-wrap font-sans text-sm break-words">${escapeHtml(text||'')}</div>`;
            } else {
                // AI message: Markdown Rendered
                contentHtml = buildAiMarkdownHtml(text);
                // Ensure content-area class is present if not already in buildAiMarkdownHtml
                if (!contentHtml.includes('content-area')) {
                    contentHtml = contentHtml.replace('prose ', 'content-area prose ');
                }
            }

            // Version Switcher UI
            let versionSwitcher = '';
            if (versionInfo) {
                const prevId = versionInfo.siblings[versionInfo.current - 2];
                const nextId = versionInfo.siblings[versionInfo.current];
                versionSwitcher = `
                    <div class="flex items-center gap-2 text-[10px] text-gray-400 mt-1 select-none">
                        <button class="hover:text-white disabled:opacity-30" onclick="switchVersion(${prevId})" ${!prevId ? 'disabled' : ''}><i class="fas fa-chevron-left"></i></button>
                        <span>${versionInfo.current} / ${versionInfo.total}</span>
                        <button class="hover:text-white disabled:opacity-30" onclick="switchVersion(${nextId})" ${!nextId ? 'disabled' : ''}><i class="fas fa-chevron-right"></i></button>
                    </div>
                `;
            }
            
            const fadeClass = animate ? 'fade-in' : ''; 
            const msgEl = document.createElement('div');
            msgEl.className = `flex ${align} mb-4 ${fadeClass} relative message-group group`;
            msgEl.id = `msg-${id}`;
            msgEl.innerHTML = `<div class="message-bubble ${bg} text-white p-4 rounded-2xl shadow-md relative">${ctrl}${qh}${th}${contentHtml}${at}${versionSwitcher}${mHtml}</div>`; 

            const container = target || get('chat-container');
            if (container) {
                container.appendChild(msgEl);
                if (doScroll) scrollToBottom(); 
                if (!isUser) {
                    queueMessageDecorations(msgEl, text);
                }
            }
            return msgEl;
        }

        function openTokenDetail(id) {
            const meta = messageMeta[id];
            if (!meta) return;
            const modal = get('token-detail-modal');
            if (!modal) return;
            const total = meta.tokens_total !== null && meta.tokens_total !== undefined ? meta.tokens_total : '-';
            const tIn = meta.tokens_in !== null && meta.tokens_in !== undefined ? meta.tokens_in : '-';
            const tOut = meta.tokens_out !== null && meta.tokens_out !== undefined ? meta.tokens_out : '-';
            const tContent = meta.tokens_content !== null && meta.tokens_content !== undefined ? meta.tokens_content : '-';
            const tThought = meta.tokens_thought !== null && meta.tokens_thought !== undefined ? meta.tokens_thought : '-';
            const enc = meta.is_encrypted === null || meta.is_encrypted === undefined ? '-' : (meta.is_encrypted ? 'Encrypted' : 'Plain');
            get('token-detail-total').innerText = total;
            get('token-detail-in').innerText = tIn;
            get('token-detail-out').innerText = tOut;
            get('token-detail-content').innerText = tContent;
            get('token-detail-thought').innerText = tThought;
            get('token-detail-encrypted').innerText = enc;
            const title = meta.model ? `${meta.model} (${meta.role})` : `${meta.role}`;
            get('token-detail-title').innerText = title;
            modal.classList.remove('hidden');
            modal.classList.add('flex');
        }

        function closeTokenDetail() {
            const modal = get('token-detail-modal');
            if (!modal) return;
            modal.classList.add('hidden');
            modal.classList.remove('flex');
        }

        function openEncryptionSettings(id) {
            const meta = messageMeta[id];
            if (!meta) return;
            openEncryptionModal(meta.is_encrypted);
        }

        function openEncryptionModal(isEncrypted) {
            const modal = get('encryption-status-modal');
            if (!modal) return;
            const title = get('encryption-status-title');
            const body = get('encryption-status-body');
            if (isEncrypted) {
                if (title) title.innerText = '暗号化されています';
                if (body) body.innerText = 'このメッセージはE2EEで暗号化されています。';
            } else {
                if (title) title.innerText = '暗号化されていません';
                if (body) body.innerText = 'このメッセージは暗号化されていません。';
            }
            modal.classList.remove('hidden');
            modal.classList.add('flex');
        }

        function closeEncryptionModal() {
            const modal = get('encryption-status-modal');
            if (!modal) return;
            modal.classList.add('hidden');
            modal.classList.remove('flex');
        }

        function goToEncryptionSettings() {
            closeEncryptionModal();
            if (typeof openSettingsModal === 'function') {
                openSettingsModal();
                switchTab('security');
                setTimeout(() => {
                    const card = get('e2ee-card');
                    if (card) card.scrollIntoView({ behavior: 'smooth', block: 'center' });
                }, 150);
            }
        }
        function openTemporaryChatSettings() {
            if (typeof openSettingsModal !== 'function') return;
            openSettingsModal();
            switchTab('general');
            setTimeout(() => {
                const card = get('temp-chat-settings-card');
                if (!card) return;
                card.scrollIntoView({ behavior: 'smooth', block: 'center' });
                card.classList.add('ring-1', 'ring-amber-400/70');
                setTimeout(() => card.classList.remove('ring-1', 'ring-amber-400/70'), 1400);
            }, 150);
        }
        
        const isGeminiLocalPythonMode = (modelId, hasAudio, hasVideo, pyEnabled) => {
            const m = (modelId || '').toLowerCase();
            if (!m.includes('gemini')) return false;
            if (m.includes('image') || m.includes('nano') || m.includes('tts') || m.includes('native-audio')) return false;
            return !!pyEnabled && (hasAudio || hasVideo);
        };
        const confirmGeminiLocalPythonSwitch = async () => {
            if (!isGeminiLocalPyDialogEnabled()) return true;
            const modal = get('gemini-local-python-modal');
            if (!modal) return true;
            const remember = get('gemini-local-python-dont-show');
            const okBtn = get('gemini-local-python-continue');
            const cancelBtn = get('gemini-local-python-cancel');
            const closeBtn = get('gemini-local-python-close');
            if (remember) remember.checked = false;
            showModal('gemini-local-python-modal');
            return await new Promise((resolve) => {
                let done = false;
                function cleanup() {
                    if (okBtn) okBtn.removeEventListener('click', onOk);
                    if (cancelBtn) cancelBtn.removeEventListener('click', onCancel);
                    if (closeBtn) closeBtn.removeEventListener('click', onCancel);
                    modal.removeEventListener('click', onOverlay, true);
                }
                function finalize(proceed) {
                    if (done) return;
                    done = true;
                    const skip = remember && remember.checked;
                    if (skip) {
                        setGeminiLocalPyDialogEnabled(false);
                        syncGeminiLocalPyDialogSetting();
                    }
                    cleanup();
                    hideModal('gemini-local-python-modal');
                    resolve(proceed);
                }
                function onOk() { finalize(true); }
                function onCancel() { finalize(false); }
                function onOverlay(e) {
                    if (e.target === modal) {
                        e.preventDefault();
                        e.stopImmediatePropagation();
                        onCancel();
                    }
                }
                if (okBtn) okBtn.addEventListener('click', onOk);
                if (cancelBtn) cancelBtn.addEventListener('click', onCancel);
                if (closeBtn) closeBtn.addEventListener('click', onCancel);
                modal.addEventListener('click', onOverlay, true);
            });
        };

        function renderPendingMessage(target = null, animate = true, doScroll = true, pendingId = null, modelId = null) {
            const fadeClass = animate ? 'fade-in' : '';
            const idAttr = pendingId ? ` id="${pendingId}"` : '';
            const skeletonHtml = buildPendingSkeletonHtml(modelId, '回答を生成中...');
            const html = `<div class="flex justify-start mb-4 ${fadeClass}"><div${idAttr} class="message-bubble ai-pending-bubble bg-gray-700 text-white p-4 rounded-2xl rounded-tl-none shadow-md relative">${skeletonHtml}</div></div>`;
            const container = target || get('chat-container');
            container.insertAdjacentHTML('beforeend', html);
            if (doScroll) scrollToBottom();
        }

        function beginPendingToStreamTransition(bubbleEl) {
            if (!bubbleEl) return;
            if (bubbleEl.getAttribute('data-stream-transition') === '1') return;
            const contentArea = bubbleEl.querySelector('.content-area');
            if (contentArea) {
                contentArea.classList.remove('pending-shimmer', 'skeleton-pending');
                contentArea.removeAttribute('data-skeleton-kind');
            }
            bubbleEl.setAttribute('data-stream-transition', '1');
            bubbleEl.classList.remove('ai-pending-bubble');
            bubbleEl.classList.add('ai-stream-transition');
            if (contentArea) {
                contentArea.classList.add('ai-stream-content-transition');
                setTimeout(() => {
                    if (contentArea) contentArea.classList.remove('ai-stream-content-transition');
                }, 300);
            }
            setTimeout(() => {
                if (bubbleEl) bubbleEl.classList.remove('ai-stream-transition');
            }, 320);
        }

        function normalizeJobIdForUi(jobId) {
            if (jobId === null || jobId === undefined || jobId === '') return null;
            return String(jobId);
        }

        function getActiveStreamingBubbleElement() {
            if (!activeStreamingBubbleId) return null;
            return get(activeStreamingBubbleId);
        }

        function captureStoppedPartialBubbleSnapshot(bubbleEl) {
            if (!bubbleEl) return null;
            const hasRenderedContent = Array.from(bubbleEl.querySelectorAll('.prose')).some((el) => String(el.textContent || '').trim());
            const hasPythonBox = !!bubbleEl.querySelector('.python-box');
            const hasThoughtContent = Array.from(bubbleEl.querySelectorAll('.thought-content')).some((el) => {
                const txt = String(el.textContent || '').trim();
                return !!txt && el.getAttribute('data-placeholder') !== '1';
            });
            if (!hasRenderedContent && !hasPythonBox && !hasThoughtContent) return null;
            const wrapper = bubbleEl.parentElement;
            if (!wrapper) return null;
            const clone = wrapper.cloneNode(true);
            clone.setAttribute('data-local-stopped-partial', '1');
            clone.classList.remove('fade-in');
            const cloneBubble = clone.querySelector('.message-bubble');
            if (cloneBubble) {
                cloneBubble.classList.remove('ai-pending-bubble', 'ai-stream-transition');
                cloneBubble.removeAttribute('data-stream-transition');
                cloneBubble.removeAttribute('id');
                if (!clone.querySelector('[data-stopped-partial-note="1"]')) {
                    const note = document.createElement('div');
                    note.setAttribute('data-stopped-partial-note', '1');
                    note.className = 'text-[10px] text-amber-200/90 mt-2 text-right';
                    note.textContent = '停止済み（途中まで）';
                    cloneBubble.appendChild(note);
                }
            }
            return {
                html: clone.outerHTML,
                threadId: (currentThreadId !== null && currentThreadId !== undefined && currentThreadId !== '')
                    ? String(currentThreadId)
                    : null
            };
        }

        function appendStoppedPartialBubbleSnapshot(snapshot, expectedThreadId = null) {
            if (!snapshot || !snapshot.html) return false;
            const current = (currentThreadId !== null && currentThreadId !== undefined && currentThreadId !== '')
                ? String(currentThreadId)
                : null;
            const expected = (expectedThreadId !== null && expectedThreadId !== undefined && expectedThreadId !== '')
                ? String(expectedThreadId)
                : (snapshot.threadId ? String(snapshot.threadId) : null);
            if (expected && current && expected !== current) return false;
            const container = get('chat-container');
            if (!container) return false;
            container.querySelectorAll('[data-local-stopped-partial="1"]').forEach((el) => el.remove());
            container.insertAdjacentHTML('beforeend', snapshot.html);
            scrollToBottom();
            return true;
        }

        function suppressPendingJob(jobId) {
            const id = normalizeJobIdForUi(jobId);
            if (!id) return;
            suppressedPendingJobIds.add(id);
        }

        function isPendingJobSuppressed(jobId) {
            const id = normalizeJobIdForUi(jobId);
            return !!(id && suppressedPendingJobIds.has(id));
        }

        function isManualStopAbortForThread(startedThreadId = null) {
            if (!manualStopContext) return false;
            const stopThreadId = manualStopContext.threadId ? String(manualStopContext.threadId) : null;
            const started = (startedThreadId !== null && startedThreadId !== undefined && startedThreadId !== '')
                ? String(startedThreadId)
                : null;
            const current = (currentThreadId !== null && currentThreadId !== undefined && currentThreadId !== '')
                ? String(currentThreadId)
                : null;
            if (stopThreadId && started && stopThreadId !== started) return false;
            if (stopThreadId && current && stopThreadId !== current) return false;
            return true;
        }

        async function syncThreadAfterAbortedStream(startedThreadId = null, opts = {}) {
            const retries = Math.max(0, Number(opts.retries ?? 1) || 0);
            const retryDelayMs = Math.max(0, Number(opts.retryDelayMs ?? 180) || 0);
            const notifyOnFailure = !!opts.notifyOnFailure;
            const started = (startedThreadId !== null && startedThreadId !== undefined && startedThreadId !== '')
                ? String(startedThreadId)
                : null;
            const current = (currentThreadId !== null && currentThreadId !== undefined && currentThreadId !== '')
                ? String(currentThreadId)
                : null;
            if (!current) return false;
            if (started && current !== started) return false;
            for (let attempt = 0; attempt <= retries; attempt++) {
                try {
                    if ((currentThreadId !== null && currentThreadId !== undefined && currentThreadId !== '') && String(currentThreadId) !== current) {
                        return false;
                    }
                    await loadMessages(current, { preserveDraft: true, silent: true });
                    return true;
                } catch (e) {
                    if (attempt < retries && retryDelayMs > 0) {
                        await new Promise((resolve) => setTimeout(resolve, retryDelayMs));
                    }
                }
            }
            if (notifyOnFailure) {
                const nowThread = (currentThreadId !== null && currentThreadId !== undefined && currentThreadId !== '')
                    ? String(currentThreadId)
                    : null;
                if (nowThread === current) {
                    showToast("停止後の履歴同期に失敗しました。画面を再読み込みすると確実です。", "warning", true);
                }
            }
            return false;
        }
        
        function vibrateHelper(pattern) {
            try {
                if (typeof navigator !== 'undefined' && navigator.vibrate) {
                    navigator.vibrate(pattern);
                }
            } catch (e) {
                console.warn('Vibration failed:', e);
            }
        }

        // === Slash command palette helpers ===
        function hideSlashCommandSuggestions() {
            const box = get('slash-command-suggestions');
            if (box) box.classList.add('hidden');
            slashSuggestionsVisible = false;
            slashSelectedIndex = 0;
        }

        function showPendingSlashCommandIndicator(cmdId) {
            const indicator = get('slash-command-indicator');
            const nameEl = get('slash-command-name');
            if (!indicator || !nameEl) return;

            const cmd = SLASH_COMMANDS.find(c => c.id === cmdId);
            nameEl.textContent = cmd ? cmd.label : `/${cmdId}`;

            indicator.classList.remove('hidden');
            indicator.classList.add('flex');

            // Give contextual guidance via placeholder
            const input = get('prompt-input');
            if (input) {
                if (cmdId === 'settings') {
                    input.dataset.originalPlaceholder = input.placeholder;
                    input.placeholder = '設定変更の指示を入力（例: デフォルトモデルをgemini-2.5-flashに変更）...';
                }
            }
        }

        function hidePendingSlashCommandIndicator() {
            const indicator = get('slash-command-indicator');
            if (indicator) {
                indicator.classList.remove('flex');
                indicator.classList.add('hidden');
            }

            // Restore placeholder
            const input = get('prompt-input');
            if (input && input.dataset.originalPlaceholder) {
                input.placeholder = input.dataset.originalPlaceholder;
                delete input.dataset.originalPlaceholder;
            }

            pendingSlashCommand = null;
        }

        function showSlashCommandSuggestions(filter = '') {
            const box = get('slash-command-suggestions');
            const listEl = get('slash-command-list');
            const inputRow = get('input-row');
            if (!box || !listEl || !inputRow) return;

            const f = filter.toLowerCase();
            const filtered = SLASH_COMMANDS.filter(c =>
                c.label.toLowerCase().includes(f) || c.description.toLowerCase().includes(f)
            );

            if (filtered.length === 0) {
                hideSlashCommandSuggestions();
                return;
            }

            listEl.innerHTML = '';
            filtered.forEach((cmd, idx) => {
                const item = document.createElement('div');
                item.className = `px-3 py-2 flex items-center gap-3 cursor-pointer text-sm hover:bg-gray-700 ${idx === slashSelectedIndex ? 'bg-gray-700' : ''}`;
                item.innerHTML = `
                    <i class="fas ${cmd.icon || 'fa-terminal'} w-4 text-blue-400"></i>
                    <div class="flex-1 min-w-0">
                        <div class="font-mono text-blue-300">${cmd.label}</div>
                        <div class="text-[11px] text-gray-400 truncate">${cmd.description}</div>
                    </div>
                `;
                item.onclick = () => selectSlashCommand(cmd.id);
                item.onmouseenter = () => {
                    slashSelectedIndex = idx;
                    showSlashCommandSuggestions(filter); // re-render highlight
                };
                listEl.appendChild(item);
            });

            // Smart vertical positioning to avoid going off-screen
            const rect = inputRow.getBoundingClientRect();
            const viewportHeight = window.innerHeight;
            const spaceBelow = viewportHeight - rect.bottom;
            const spaceAbove = rect.top;

            const desiredMaxHeight = 260; // px
            const padding = 8;

            box.style.position = 'fixed';
            box.style.left = `${Math.max(8, rect.left)}px`;
            box.style.zIndex = '80';
            box.style.maxHeight = 'none'; // reset

            // Decide whether to show below or above
            const showAbove = spaceBelow < 180 && spaceAbove > spaceBelow;

            if (showAbove) {
                // Place above the input row
                const popupHeight = Math.min(desiredMaxHeight, spaceAbove - padding);
                box.style.top = 'auto';
                box.style.bottom = `${viewportHeight - rect.top + 4}px`;
                listEl.style.maxHeight = `${popupHeight}px`;
            } else {
                // Place below (preferred)
                const popupHeight = Math.min(desiredMaxHeight, spaceBelow - padding);
                box.style.top = `${rect.bottom + 4}px`;
                box.style.bottom = 'auto';
                listEl.style.maxHeight = `${popupHeight}px`;
            }

            box.classList.remove('hidden');
            slashSuggestionsVisible = true;
        }

        function selectSlashCommand(cmdId) {
            const input = get('prompt-input');
            if (!input) return;

            const val = input.value;
            // Find the last '/' before cursor or end, and remove from there to the command
            const lastSlash = val.lastIndexOf('/');
            if (lastSlash !== -1) {
                // Keep everything before the command start
                input.value = val.substring(0, lastSlash).trimEnd();
            } else {
                input.value = '';
            }

            hideSlashCommandSuggestions();
            pendingSlashCommand = cmdId;

            // Show visual feedback that we are now in command argument mode
            showPendingSlashCommandIndicator(cmdId);

            // Optional: give a small hint in placeholder or just let user start typing the instruction
            input.focus();
            // Dispatch input to update any token estimate etc.
            input.dispatchEvent(new Event('input', { bubbles: true }));
        }

        // === Gem suggestion helpers (triggered by @ in prompt bar) ===
        function hideGemSuggestions() {
            const box = get('gem-suggestions');
            if (box) box.classList.add('hidden');
            gemSuggestionsVisible = false;
            gemSelectedIndex = 0;
        }

        function showGemSuggestions(filter = '') {
            const box = get('gem-suggestions');
            const listEl = get('gem-suggestions-list');
            const inputRow = get('input-row');
            if (!box || !listEl || !inputRow) return;

            if (!loadedGems || loadedGems.length === 0) {
                hideGemSuggestions();
                return;
            }

            const f = filter.toLowerCase();
            const filtered = loadedGems.filter(g =>
                g.name.toLowerCase().includes(f) || (g.description && g.description.toLowerCase().includes(f))
            );

            if (filtered.length === 0) {
                hideGemSuggestions();
                return;
            }

            // Clamp selected index
            if (gemSelectedIndex >= filtered.length) gemSelectedIndex = 0;

            listEl.innerHTML = '';
            filtered.forEach((gem, idx) => {
                const item = document.createElement('div');
                item.className = `px-3 py-2 flex items-center gap-3 cursor-pointer text-sm hover:bg-gray-700 ${idx === gemSelectedIndex ? 'bg-gray-700' : ''}`;
                item.innerHTML = `
                    <i class="fas fa-gem w-4 text-blue-400"></i>
                    <div class="flex-1 min-w-0">
                        <div class="text-blue-300 truncate font-medium">${gem.name}</div>
                        ${gem.description ? `<div class="text-[11px] text-gray-400 truncate">${gem.description}</div>` : ''}
                    </div>
                `;
                item.onclick = () => selectGemSuggestion(gem);
                item.onmouseenter = () => {
                    gemSelectedIndex = idx;
                    showGemSuggestions(filter);
                };
                listEl.appendChild(item);
            });

            // Smart vertical positioning (same logic as slash commands)
            const rect = inputRow.getBoundingClientRect();
            const viewportHeight = window.innerHeight;
            const spaceBelow = viewportHeight - rect.bottom;
            const spaceAbove = rect.top;

            const desiredMaxHeight = 260;
            const padding = 8;

            box.style.position = 'fixed';
            box.style.left = `${Math.max(8, rect.left)}px`;
            box.style.zIndex = '80';
            box.style.maxHeight = 'none';

            const showAbove = spaceBelow < 180 && spaceAbove > spaceBelow;

            if (showAbove) {
                const popupHeight = Math.min(desiredMaxHeight, spaceAbove - padding);
                box.style.top = 'auto';
                box.style.bottom = `${viewportHeight - rect.top + 4}px`;
                listEl.style.maxHeight = `${popupHeight}px`;
            } else {
                const popupHeight = Math.min(desiredMaxHeight, spaceBelow - padding);
                box.style.top = `${rect.bottom + 4}px`;
                box.style.bottom = 'auto';
                listEl.style.maxHeight = `${popupHeight}px`;
            }

            box.classList.remove('hidden');
            gemSuggestionsVisible = true;
        }

        function selectGemSuggestion(gem) {
            const input = get('prompt-input');
            if (!input) return;

            const val = input.value;
            const lastAt = val.lastIndexOf('@');
            if (lastAt !== -1) {
                input.value = val.substring(0, lastAt).trimEnd();
            } else {
                input.value = '';
            }

            hideGemSuggestions();
            activateGem(gem);
            input.focus();
            input.dispatchEvent(new Event('input', { bubbles: true }));
        }

        async function sendMessage() { 
            vibrateHelper(50);
            if (abortController) {
                showToast("回答生成中です。完了までお待ちいただくか、停止してください。", "warning", true);
                return;
            }
            if (uploadProgressState.active > 0) {
                showToast("ファイルの送信・処理中です。しばらくお待ちください。", "warning", true);
                return;
            }
            const rawText = get('prompt-input').value; // RAW INPUT (No trim)

            // Handle pending slash command (e.g. after selecting /settings via the palette)
            if (pendingSlashCommand) {
                const cmd = pendingSlashCommand;
                hidePendingSlashCommandIndicator(); // clears state + hides indicator

                const instruction = rawText.trim();
                const modelForCmd = get('model-select') ? get('model-select').value : null;

                get('prompt-input').value = '';
                get('prompt-input').style.height = 'auto';

                if (cmd === 'settings') {
                    if (!instruction) {
                        showToast('設定変更の指示を入力してください（例: デフォルトモデルをgemini-2.5-flashに）', 'info');
                        return;
                    }
                    if (!modelForCmd) {
                        showToast('モデルを選択してください', 'error', true);
                        return;
                    }

                    // Call the existing AI settings endpoint (reuses all previous backend logic + fallbacks)
                    (async () => {
                        try {
                            const res = await apiFetch('/api/settings/apply-ai-prompt', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({ prompt: instruction, model: modelForCmd })
                            });
                            const data = await res.json().catch(() => ({}));

                            if (data && data.status === 'ok' && data.applied) {
                                showToast(`設定を更新しました（${Object.keys(data.applied).length}項目）`, 'success');
                                try {
                                    const fresh = await apiFetch(CHAT_CONFIG.urls.handleSettingsQuery).then(r => r.json());
                                    if (typeof populateAiSafeFormFields === 'function') populateAiSafeFormFields(fresh);
                                    if (typeof cacheUserSettings === 'function') cacheUserSettings(fresh);
                                } catch (e) {}
                                // Visual feedback in chat
                                const summary = Object.entries(data.applied).slice(0,3).map(([k,v])=>`${k}→${v}`).join(', ');
                                renderMessage(Date.now(), 'assistant', `✅ 設定をAIで適用しました\n${summary}${Object.keys(data.applied).length>3?' ...':''}`, null, null, null, null, true, null, null, null, null, null, null, null, null, true);
                            } else {
                                const msg = data.message || data.error || '設定変更に失敗しました';
                                showToast(msg, 'error', true);
                            }
                        } catch (err) {
                            showToast('設定変更の通信に失敗しました', 'error', true);
                        }
                    })();
                }
                return; // Do not treat as normal message
            }

            // Save to prompt history
            if (rawText.trim()) {
                if (promptHistory.length === 0 || promptHistory[0] !== rawText) {
                    promptHistory.unshift(rawText);
                    if (promptHistory.length > 100) promptHistory.pop();
                }
            }
            historyIndex = -1;
            tempPrompt = "";

            const attachmentItemsToSend = collectAttachmentItemsForSend();
            const imageUrlsToSend = attachmentItemsToSend.map((it) => it.path);
            const uploadedImageUrlsToSend = attachmentItemsToSend
                .filter((it) => normalizeAttachmentSource(it.source) === 'upload')
                .map((it) => it.path);
            if (imageUrlsToSend.length > ATTACHMENT_MAX_FILES) {
                showToast(`添付は最大${ATTACHMENT_MAX_FILES}件です。添付を減らして再送してください。`, "error", true);
                return;
            }
            const support = getModelMediaSupport(get('model-select').value);
            const hasAudio = imageUrlsToSend.some((fp) => isAudioPath(fp));
            const hasVideo = imageUrlsToSend.some((fp) => isVideoPath(fp));
            const modelId = (get('model-select').value || '').toLowerCase();
            const pyChk = get('enable-python');
            const pyEnabled = !!(pyChk && pyChk.checked);
            if ((hasAudio && !support.audio) || (hasVideo && !support.video)) {
                showToast("このモデルは音声/動画入力に対応していません", "error", true);
                purgeUnsupportedAttachments(true);
                return;
            }
            if(!rawText.trim() && imageUrlsToSend.length === 0) return; // Check trim only for empty check

            // === /settings command: natural language settings change via AI (reuses prompt bar model + toggles) ===
            const trimmedRaw = rawText.trim();
            if (trimmedRaw.toLowerCase().startsWith('/settings')) {
                const instruction = trimmedRaw.replace(/^\/settings\s*/i, '').trim();
                if (!instruction) {
                    showToast('使い方: /settings デフォルトモデルを gemini-2.5-flash に変更して thinking をオンに', 'info');
                    get('prompt-input').value = '/settings ';
                    return;
                }
                const settingsModel = get('model-select') ? get('model-select').value : null;
                if (!settingsModel) {
                    showToast('モデルが選択されていません', 'error', true);
                    return;
                }
                // Optimistic UI: clear input immediately
                get('prompt-input').value = '';
                get('prompt-input').style.height = 'auto';

                try {
                    const res = await apiFetch('/api/settings/apply-ai-prompt', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ prompt: instruction, model: settingsModel })
                    });
                    const data = await res.json().catch(() => ({}));

                    if (data && data.status === 'ok' && data.applied) {
                        showToast(`設定をAIで更新しました (${Object.keys(data.applied).length}項目)`, 'success');
                        try {
                            const fresh = await apiFetch(CHAT_CONFIG.urls.handleSettingsQuery).then(r => r.json());
                            if (typeof populateAiSafeFormFields === 'function') populateAiSafeFormFields(fresh);
                            if (typeof cacheUserSettings === 'function') cacheUserSettings(fresh);
                        } catch (e) {}
                        // Show a nice feedback message in the current chat (non-persisted visual only)
                        const appliedSummary = Object.entries(data.applied).slice(0, 5).map(([k, v]) => `${k}: ${v}`).join(', ');
                        renderMessage(Date.now(), 'assistant', `設定を更新しました。\n${appliedSummary}${Object.keys(data.applied).length > 5 ? ' ...' : ''}`, null, null, null, null, true, null, null, null, null, null, null, null, null, true);
                    } else {
                        const msg = (data && (data.message || data.error)) || 'AI設定の適用に失敗しました';
                        showToast(msg, 'error', true);
                        // Admin debug (temporary)
                        const adminDbg = get('ai-settings-admin-debug');
                        if (adminDbg && (data.raw_error || data.message)) {
                            const dbgContent = get('ai-settings-debug-content');
                            if (dbgContent) {
                                dbgContent.textContent = `モデル: ${data.original_model || settingsModel}\nエラー: ${msg}\n${data.raw_error ? '詳細: ' + data.raw_error : ''}`;
                            }
                            adminDbg.classList.remove('hidden');
                            const copyBtn = get('ai-settings-debug-copy');
                            if (copyBtn) copyBtn.onclick = () => navigator.clipboard.writeText(dbgContent.textContent || '');
                        }
                    }
                } catch (e) {
                    showToast('AI設定変更の通信に失敗しました', 'error', true);
                }
                return; // Do not proceed to normal chat send
            }

            if (isGeminiLocalPythonMode(modelId, hasAudio, hasVideo, pyEnabled)) {
                const proceed = await confirmGeminiLocalPythonSwitch();
                if (!proceed) return;
            }
            sendClientDebugLog(
                'info',
                `Prompt send start: model=${get('model-select').value} thread=${currentThreadId || '-'} text_len=${rawText.length} attachments=${imageUrlsToSend.length} search=${get('enable-search').checked}`
            );
            const t = rawText;
            const markerSysPrompt = hasMarkerHint() ? MARKER_HINT_TEXT : null;
            if (isGptImageModel() && currentMaskImage && imageUrlsToSend.length === 0) {
                showToast("Mask は画像入力が必要です", "error", true);
                return;
            }
            const editingId = editingMessageId;
            const capturedParentId = currentParentId;
            const parentIdExplicit = (editingId !== null && editingId !== undefined);
            if (editingId) {
                editingMessageId = null;
                setEditUi(false);
            }
            playSendAnimation();
            get('welcome-screen').classList.add('hidden'); 

            const hiddenBranch = [];
            const hideRenderedBranchFrom = (startId) => {
                if (startId === null || startId === undefined) return;
                let node = document.getElementById(`msg-${startId}`);
                while (node) {
                    if (node.classList && node.classList.contains('message-group')) {
                        hiddenBranch.push({ node, prevDisplay: node.style.display });
                        node.style.display = 'none';
                    }
                    node = node.nextElementSibling;
                }
            };
            const restoreHiddenBranch = () => {
                hiddenBranch.forEach(({ node, prevDisplay }) => {
                    if (!node) return;
                    node.style.display = prevDisplay || '';
                });
                hiddenBranch.length = 0;
            };

            // If editing/regenerating, hide the current branch before optimistic render.
            if (editingId) {
                hideRenderedBranchFrom(editingId);
            }

            // Render UI immediately to reduce perceived latency.
            renderMessage(Date.now(), 'user', t, JSON.stringify(imageUrlsToSend), null, null, null, true, currentQuote, null, null, null, null, null, null, null, true, capturedParentId, activeGem ? activeGem.name : null); 
            get('prompt-input').value = ''; get('prompt-input').style.height = 'auto'; 
            schedulePromptTokenEstimate(true);

            let disableAutoSearch = false;
            const xLinkPattern = /(https?:\/\/)?(x\.com|twitter\.com)\//i;
            const hasXLink = xLinkPattern.test(t || '') || xLinkPattern.test(currentQuote || '');
            const grokXModel = "grok-4-fast-reasoning";
            const applyXLinkAuto = () => {
                get('enable-search').checked = true;
                if (get('model-select').value !== grokXModel) {
                    selectModelById(grokXModel);
                }
            };
            if (hasXLink && !get('enable-search').checked) {
                if (autoSearchOnLinks) {
                    applyXLinkAuto();
                } else {
                    const banner = get('auto-search-banner');
                    const onBtn = get('auto-search-on-btn');
                    const offBtn = get('auto-search-off-btn');
                    const remember = get('auto-search-remember');
                    if (banner && onBtn && offBtn) {
                        if (remember) remember.checked = false;
                        await new Promise((resolve) => {
                            banner.classList.remove('hidden');
                            const cleanup = (choice) => {
                                banner.classList.add('hidden');
                                onBtn.onclick = null;
                                offBtn.onclick = null;
                                resolve(choice);
                            };
                            onBtn.onclick = () => cleanup('enable');
                            offBtn.onclick = () => cleanup('disable');
                        }).then(async (choice) => {
                            if (choice === 'enable') {
                                applyXLinkAuto();
                                if (remember && remember.checked) {
                                    autoSearchOnLinks = true;
                                    await apiFetch(CHAT_CONFIG.urls.handleSettings, {
                                        method: 'POST',
                                        headers: {'Content-Type': 'application/json'},
                                        body: JSON.stringify({ auto_search_on_links: true })
                                    });
                                }
                            } else {
                                disableAutoSearch = true;
                            }
                        });
                    }
                }
            }
            
            const p = { 
                thread_id: currentThreadId, 
                message: t, 
                model: get('model-select').value, 
                image_urls: imageUrlsToSend, 
                image_items: attachmentItemsToSend,
                uploaded_image_urls: uploadedImageUrlsToSend,
                temporary_chat: temporaryChatEnabled,
                enable_search: get('enable-search').checked, 
                enable_url_context: get('enable-url-context') ? get('enable-url-context').checked : false,
                enable_maps: get('enable-maps') ? get('enable-maps').checked : false,
                enable_python: get('enable-python').checked, 
                enable_thinking: get('enable-thinking').checked, 
                thinking_level: get('thinking-level').value, 
                thinking_budget: get('thinking-budget') ? get('thinking-budget').value : null,
                reasoning_effort: get('reasoning-effort').value, 
                enable_system_prompt: get('enable-sys-prompt').checked,
                enable_prompt_caching: get('enable-prompt-cache') ? get('enable-prompt-cache').checked : false,
                marker_system_prompt: markerSysPrompt,
                safety_setting: get('safety-setting').value, 
                tts_voice: isTtsModel() && get('tts-voice') ? get('tts-voice').value : null,
                tts_voice_custom: isTtsModel() && get('tts-voice-custom') ? get('tts-voice-custom').value : null,
                tts_language: isTtsModel() && get('tts-language') ? get('tts-language').value : null,
                tts_speed: isTtsModel() && get('tts-speed') ? get('tts-speed').value : null,
                image_size: isGptImageModel() && get('gpt-image-size') ? get('gpt-image-size').value : null,
                image_quality: isGptImageModel() && get('gpt-image-quality') ? get('gpt-image-quality').value : null,
                image_format: isGptImageModel() && get('gpt-image-format') ? get('gpt-image-format').value : null,
                image_compression: isGptImageModel() && get('gpt-image-compression') ? get('gpt-image-compression').value : null,
                image_mask: isGptImageModel() ? currentMaskImage : null,
                gemini_image_aspect: isGeminiImageModel() && get('gemini-image-aspect') ? get('gemini-image-aspect').value : null,
                gemini_image_size: isGeminiImageModel() && get('gemini-image-size') ? get('gemini-image-size').value : null,
                grok_image_aspect: isGrokImageModel() && get('grok-image-aspect') ? get('grok-image-aspect').value : null,
                grok_image_resolution: isGrokImageModel() && get('grok-image-resolution') ? get('grok-image-resolution').value : null,
                grok_video_duration: isGrokVideoModel() && get('grok-video-duration') ? get('grok-video-duration').value : null,
                grok_video_aspect: isGrokVideoModel() && get('grok-video-aspect') ? get('grok-video-aspect').value : null,
                grok_video_resolution: isGrokVideoModel() && get('grok-video-resolution') ? get('grok-video-resolution').value : null,
                quote_text: currentQuote,
                parent_id: capturedParentId,
                parent_id_explicit: parentIdExplicit,
                disable_auto_search: disableAutoSearch,
                image_vision_model: currentVisionModel || null
            }; 
            const threadCustomInstructionEl = get('thread-custom-instruction');
            if (threadCustomInstructionEl) {
                p.thread_custom_instruction = threadCustomInstructionEl.value || '';
            }
            if (activeGem) { p.system_prompt = activeGem.instruction; p.enable_system_prompt = true; p.gem_uuid = activeGem.uuid; } else { p.gem_uuid = null; }
            resetUploadState(); setSendBtnToStopMode(); userAutoScroll = true; const aid = 'ai-' + Date.now(); clearQuote(); 
            const modelLower = String(p.model || '').toLowerCase();
            const effortLower = String(p.reasoning_effort || '').toLowerCase();
            const reasoningRequested = !!p.enable_thinking || (!!effortLower && effortLower !== 'none');
            const reasoningCapableModel =
                modelLower.includes('gemini') ||
                modelLower.includes('o1') ||
                modelLower.includes('o3') ||
                modelLower.includes('gpt-5') ||
                (modelLower.includes('reasoning') && !modelLower.includes('non-reasoning'));
            const shouldShowReasoningProgress = reasoningRequested && reasoningCapableModel;
        
            let initialHtml = buildPendingSkeletonHtml(p.model, 'APIに送信中...');
            get('chat-container').insertAdjacentHTML('beforeend', `<div class="flex justify-start mb-4 fade-in"><div id="${aid}" class="message-bubble ai-pending-bubble bg-gray-700 text-white p-4 rounded-2xl rounded-tl-none shadow-md relative">${initialHtml}</div></div>`); 
            scrollToBottom(); 
            const adiv = get(aid);
            activeStreamingBubbleId = aid;
            if (canvasModeEnabled) {
                resetCanvasPreviewPanel();
            }
            let thoughtPlaceholderEl = null;
            const ensureThoughtPlaceholder = (text) => {
                if (!shouldShowReasoningProgress || !adiv) return null;
                if (!thoughtPlaceholderEl || !adiv.contains(thoughtPlaceholderEl)) {
                    thoughtPlaceholderEl = adiv.querySelector('.thought-content');
                }
                if (!thoughtPlaceholderEl) {
                    const tHtml = `<div class="thought-container"><div class="thought-header thinking-shimmer" onclick="toggleThinking(this)"><i class="fas fa-brain text-purple-400"></i> Thinking Process</div><div class="thought-content collapsed" data-placeholder="1"></div></div>`;
                    adiv.insertAdjacentHTML('afterbegin', tHtml);
                    thoughtPlaceholderEl = adiv.querySelector('.thought-content');
                }
                if (thoughtPlaceholderEl) {
                    thoughtPlaceholderEl.setAttribute('data-placeholder', '1');
                    thoughtPlaceholderEl.textContent = text || '推論プロセスを準備中...';
                }
                return thoughtPlaceholderEl;
            };
            if (shouldShowReasoningProgress) {
                ensureThoughtPlaceholder('推論プロセスを準備中...');
            }
            
            abortController = new AbortController(); 
            const streamStartedThreadId = currentThreadId;
            const sendStartPerfMs = nowPerfMs();
            const sendStartEpochMs = Date.now();
            let firstStatusLatencySent = false;
            let firstThoughtLatencySent = false;
            let firstContentLatencySent = false;
            let firstStatusLatencyMs = null;
            let firstThoughtLatencyMs = null;
            let firstContentLatencyMs = null;

            let streamThreadIdForMetric = (currentThreadId !== null && currentThreadId !== undefined && currentThreadId !== '')
                ? String(currentThreadId)
                : null;
            const maybeReportFirstEventLatency = (eventType, shouldReport) => {
                if (!shouldReport) return;
                if (eventType === 'status' && firstStatusLatencySent) return;
                if (eventType === 'thought' && firstThoughtLatencySent) return;
                if (eventType === 'content' && firstContentLatencySent) return;
                const elapsedMs = Math.max(0, nowPerfMs() - sendStartPerfMs);
                
                if (eventType === 'status') firstStatusLatencyMs = elapsedMs;
                else if (eventType === 'thought') firstThoughtLatencyMs = elapsedMs;
                else if (eventType === 'content') firstContentLatencyMs = elapsedMs;

                reportFirstTokenLatency({
                    latency_seconds: elapsedMs / 1000,
                    latency_ms: elapsedMs,
                    thread_id: streamThreadIdForMetric || currentThreadId,
                    job_id: currentJobId,
                    model: p.model,
                    first_event_type: eventType,
                    client_sent_at_ms: sendStartEpochMs
                });
                if (eventType === 'status') firstStatusLatencySent = true;
                else if (eventType === 'thought') firstThoughtLatencySent = true;
                else if (eventType === 'content') firstContentLatencySent = true;
            };
            try { 
                if (p.thread_id && activeGem) {
                    threadGemMap[p.thread_id] = activeGem;
                    pendingGemForNewThread = null;
                }
                const r = await apiFetch(CHAT_CONFIG.urls.chatStream, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(p), signal:abortController.signal}); 
                sendClientDebugLog('info', `Prompt stream response status: ${r.status}`);
                const markApiAccepted = () => {
                    if (!adiv) return;
                    const ca = adiv.querySelector('.content-area');
                    if (!ca) return;
                    if (ca.getAttribute('data-api-accepted') === '1') return;
                    ca.setAttribute('data-api-accepted', '1');
                    if (!updatePendingSkeletonStatus(adiv, '接続完了。モデル応答を待機中...', 'キュー待機や初期化中の可能性があります')) {
                        ca.outerHTML = buildPendingSkeletonHtml(p.model, '接続完了。モデル応答を待機中...');
                        const newCa = adiv.querySelector('.content-area');
                        if (newCa) newCa.setAttribute('data-api-accepted', '1');
                        updatePendingSkeletonStatus(adiv, '接続完了。モデル応答を待機中...', 'キュー待機や初期化中の可能性があります');
                    }
                };
                markApiAccepted();
                const reader = r.body.getReader(); 
                const dec = new TextDecoder(); 
                let buf="", acc="", tht="", first=true, thEl=null, cEl=null, searchBox=null, hadError=false; 
                const pyBoxes = {}; 
                let lastRenderTime = 0;
            
                while(true) { 
                    const {done, value} = await reader.read(); 
                    if(done) break; 
                    buf += dec.decode(value, {stream:true}); 
                    let ls = buf.split("\n"); 
                    buf = ls.pop(); 
                    let contentChanged = false;
                    let thoughtChanged = false;
                    for(let l of ls) { 
                        if(!l.trim()) continue; 
                        try { 
                            const j = JSON.parse(l); 
                            if (j.type === 'thread_id') {
                                markApiAccepted();
                                const streamThreadId = j.content !== null && j.content !== undefined ? String(j.content) : j.content;
                                if (streamThreadId) {
                                    streamThreadIdForMetric = streamThreadId;
                                    if (currentThreadId !== streamThreadId) {
                                        currentThreadId = streamThreadId;
                                        history.pushState({}, '', '/c/' + streamThreadId);
                                    }
                                    if (activeGem) {
                                        threadGemMap[streamThreadId] = activeGem;
                                        pendingGemForNewThread = null;
                                    }
                                    ensureTemporaryChatHeartbeat(true);
                                }
                                continue;
                            }
                            if (j.type === 'job_id') { markApiAccepted(); currentJobId = j.content; continue; }
                            if (j.type === 'search_status') {
                                if (j.content === 'searching' && !searchBox) {
                                     adiv.insertAdjacentHTML('afterbegin', `<div class="search-box visible animate-pulse mb-2"><i class="fas fa-globe"></i> Searching web...</div>`);
                                     searchBox = adiv.querySelector('.search-box');
                                } else if (j.content === 'done' && searchBox) {
                                     searchBox.classList.remove('animate-pulse');
                                     searchBox.innerHTML = '<i class="fas fa-check-circle text-green-400"></i> Search complete';
                                     setTimeout(() => { if(searchBox) searchBox.remove(); searchBox=null; }, 2000);
                                }
                                continue;
                            }
                            if (j.type === 'status') {
                                markApiAccepted();
                                const statusText = (j.content === null || j.content === undefined) ? '' : String(j.content);
                                maybeReportFirstEventLatency('status', !!statusText);
                                if (first && adiv) {
                                    const headline = statusText || 'モデル処理中...';
                                    if (!updatePendingSkeletonStatus(adiv, headline, '応答開始までの進捗を表示しています')) {
                                        const ca = adiv.querySelector('.content-area');
                                        if (ca) {
                                            ca.outerHTML = buildPendingSkeletonHtml(p.model, headline);
                                            updatePendingSkeletonStatus(adiv, headline, '応答開始までの進捗を表示しています');
                                        }
                                    }
                                }
                                if (shouldShowReasoningProgress) {
                                    ensureThoughtPlaceholder(statusText || '推論プロセスを準備中...');
                                }
                                continue;
                            }
                            if(first){ 
                                 beginPendingToStreamTransition(adiv);
                                 const ca = adiv.querySelector('.content-area');
                                 if(ca) ca.innerHTML = '';
                                 first=false; 
                            }
                            if(j.type==='thought'){ 
                                if(!thEl){
                                    thEl = adiv.querySelector('.thought-content');
                                }
                                tht+=j.content; 
                                maybeReportFirstEventLatency('thought', !!j.content);
                                if(!thEl){ 
                                    const tHtml = `<div class="thought-container"><div class="thought-header" onclick="toggleThinking(this)"><i class="fas fa-brain text-purple-400"></i> Thinking Process</div><div class="thought-content"></div></div>`; 
                                    if(searchBox) searchBox.insertAdjacentHTML('afterend', tHtml); 
                                    else adiv.insertAdjacentHTML('afterbegin', tHtml); 
                                    thEl=adiv.querySelector('.thought-content'); 
                                } 
                                if (thEl && thEl.getAttribute('data-placeholder') === '1') {
                                    thEl.textContent = '';
                                    thEl.removeAttribute('data-placeholder');
                                    if (thEl) {
                                        const thoughtHeader = thEl.parentElement.querySelector('.thought-header');
                                        if (thoughtHeader) thoughtHeader.classList.remove('thinking-shimmer');
                                    }
                                    tht = j.content;
                                }
                                thEl.classList.remove('collapsed');
                                thoughtChanged = true;
                            } else if (j.type === 'image_analysis') {
                                const iaText = (j.content === null || j.content === undefined) ? '' : String(j.content);
                                if (!adiv) continue;
                                let iaEl = adiv.querySelector('.image-analysis-box');
                                if (!iaEl) {
                                    const iaHtml = `<div class="image-analysis-box mb-2 p-2 bg-blue-900/20 border border-blue-500/30 rounded"><div class="text-[10px] text-blue-300 font-medium mb-1"><i class="fas fa-image mr-1"></i>Image Analysis</div><div class="image-analysis-text text-[11px] text-gray-300"></div></div>`;
                                    if(searchBox) searchBox.insertAdjacentHTML('afterend', iaHtml);
                                    else adiv.insertAdjacentHTML('afterbegin', iaHtml);
                                    iaEl = adiv.querySelector('.image-analysis-box');
                                }
                                const iaTxt = iaEl.querySelector('.image-analysis-text');
                                if (iaTxt) iaTxt.textContent = iaText;
                            } else if(j.type==='python'){ 
                                const py = j.content || {}; 
                                const pyId = py.id || `py_${Date.now()}`; 
                                if(!pyBoxes[pyId]){ 
                                    const boxHtml = `<div class="code-wrapper python-box collapsed" data-py-id="${pyId}" data-collapsed="true" data-code-key="${pyId}"><div class="code-header"><span class="code-lang"><i class="fas fa-terminal"></i> Python Execution</span><div class="code-actions"><button class="code-toggle" aria-expanded="false"><i class="fas fa-chevron-down"></i> Expand</button><button class="copy-btn" data-copy="code" data-code=""><i class="fas fa-copy"></i> Copy Code</button><button class="copy-btn" data-copy="output" data-code=""><i class="fas fa-copy"></i> Copy Output</button></div></div><div class="code-body"><div class="python-section"><div class="python-label">Code</div><pre><code class="hljs language-python python-code"></code></pre></div><div class="python-section"><div class="python-label">Output</div><pre><code class="hljs language-plaintext python-output"></code></pre></div></div></div>`; 
                                    if(searchBox) searchBox.insertAdjacentHTML('afterend', boxHtml); 
                                    else adiv.insertAdjacentHTML('afterbegin', boxHtml); 
                                    pyBoxes[pyId] = adiv.querySelector(`[data-py-id="${pyId}"]`); 
                                } 
                                const box = pyBoxes[pyId]; 
                                if(box){ 
                                    if(py.code !== undefined){ 
                                        const codeText = py.code == null ? '' : String(py.code);
                                        const codeEl = box.querySelector('.python-code');
                                        if (codeEl) {
                                            codeEl.textContent = codeText;
                                            codeEl.removeAttribute('data-highlighted');
                                            queueHighlight(box, codeText);
                                        }
                                        const codeBtn = box.querySelector('.copy-btn[data-copy="code"]');
                                        if (codeBtn) codeBtn.setAttribute('data-code', encodeURIComponent(codeText).replace(/'/g, "%27"));
                                    } 
                                    if(py.output !== undefined){ 
                                        const outText = py.output == null ? '' : String(py.output);
                                        const outEl = box.querySelector('.python-output');
                                        if (outEl) outEl.textContent = outText;
                                        const outBtn = box.querySelector('.copy-btn[data-copy="output"]');
                                        if (outBtn) outBtn.setAttribute('data-code', encodeURIComponent(outText).replace(/'/g, "%27"));
                                    } 
                                } 
                            } else if(j.type==='content'){ 
                                const contentDelta = (j.content === null || j.content === undefined) ? '' : String(j.content);
                                acc += contentDelta; 
                                if(!cEl){ 
                                    cEl = adiv.querySelector('.content-area') || document.createElement('div'); 
                                    cEl.className='prose prose-invert text-sm break-words'; 
                                    if(!adiv.contains(cEl)) adiv.appendChild(cEl); 
                                } 
                                contentChanged = true;
                                maybeReportFirstEventLatency('content', !!contentDelta);
                            } else if(j.type==='error'){ 
                                hadError = true;
                                adiv.insertAdjacentHTML('beforeend', `<div class="text-red-400 text-xs mt-2 border border-red-500 p-2 rounded">Error: ${escapeHtml(j.content)}</div>`);
                                showToast(j.content || "Unknown error", "error", true);
                            } 
                        } catch(e){} 
                    } 
                    if (thoughtChanged && thEl) {
                        thEl.textContent = tht;
                        if (userAutoScroll) thEl.scrollTop = thEl.scrollHeight;
                    }
                    if (contentChanged && cEl) {
                        const now = Date.now();
                        if (now - lastRenderTime > 100) {
                            const collapseState = snapshotCodeCollapse(cEl);
                            renderAiMarkdownInto(cEl, acc);
                            applyCodeCollapse(cEl, collapseState, true);
                            lastRenderTime = now;
                        }
                    }
                    scrollToBottom(); 
                } 
                // Final render to catch any remaining content
                if (cEl) {
                    const collapseState = snapshotCodeCollapse(cEl);
                    renderAiMarkdownInto(cEl, acc);
                    applyCodeCollapse(cEl, collapseState, true);
                }
                scrollToBottom();

                vibrateHelper([100, 50, 100]);

                if (adiv) {
                    queueHighlight(adiv, acc);
                    queueMathTypeset(adiv, acc);

                    if (enableLatencyMetrics) {
                        const totalLatencyMs = nowPerfMs() - sendStartPerfMs;
                        
                        // Send total latency to server
                        reportFirstTokenLatency({
                            is_total: true,
                            latency_seconds: totalLatencyMs / 1000,
                            latency_ms: totalLatencyMs,
                            thread_id: streamThreadIdForMetric || currentThreadId,
                            job_id: currentJobId,
                            model: p.model,
                            client_sent_at_ms: sendStartEpochMs,
                            client_done_at_ms: Date.now()
                        });

                        let latencyHtml = `<div class="mt-2 pt-2 border-t border-gray-700/30 flex flex-col gap-1 items-end opacity-70 text-[10px] font-mono text-gray-400">`;
                        
                        // First Token
                        let firstAnyMs = null;
                        if (firstStatusLatencyMs !== null) firstAnyMs = firstStatusLatencyMs;
                        if (firstThoughtLatencyMs !== null && (firstAnyMs === null || firstThoughtLatencyMs < firstAnyMs)) firstAnyMs = firstThoughtLatencyMs;
                        if (firstContentLatencyMs !== null && (firstAnyMs === null || firstContentLatencyMs < firstAnyMs)) firstAnyMs = firstContentLatencyMs;
                        
                        if (firstAnyMs !== null) {
                            latencyHtml += `<div>Initial: ${(firstAnyMs/1000).toFixed(2)}s</div>`;
                        }
                        
                        // Breakdown if available
                        if (firstContentLatencyMs !== null && firstContentLatencyMs !== firstAnyMs) {
                            latencyHtml += `<div>Content: ${(firstContentLatencyMs/1000).toFixed(2)}s</div>`;
                        }

                        latencyHtml += `<div class="font-bold text-gray-300">Total: ${(totalLatencyMs/1000).toFixed(2)}s</div>`;
                        if (currentJobId) {
                            latencyHtml += `<div class="text-[9px] opacity-50">Job ID: ${escapeHtml(currentJobId)}</div>`;
                        }
                        latencyHtml += `<div class="text-[10px] mt-1">${escapeHtml(get('model-select').value)}</div>`;
                        latencyHtml += `</div>`;
                        adiv.insertAdjacentHTML('beforeend', latencyHtml);
                    } else {
                        adiv.insertAdjacentHTML('beforeend', `<div class="text-[10px] text-gray-500/50 mt-2 text-right font-mono">${escapeHtml(get('model-select').value)}</div>`);
                    }
                }

                // Reset editing state
                editingMessageId = null;
                setEditUi(false);

                if (!hadError && adiv) {
                    const thoughts = adiv.querySelectorAll('.thought-content');
                    thoughts.forEach(t => t.classList.add('collapsed'));
                }
                // Full reload to establish new tree structure (skip on error to keep the error visible)
                if (!hadError) {
                    await loadMessages(currentThreadId, { preserveDraft: true, silent: true });
                }
                
                // Only auto-scroll if user was already at bottom or auto-scroll is active
                if(userAutoScroll) scrollToBottom();

                if (document.querySelectorAll('.message-group').length <= 2 || !currentThreadTitle || currentThreadTitle === 'New Chat' || currentThreadTitle === 'No Title') {
                     apiFetch("/api/generate_title", {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({thread_id: currentThreadId, model_id: get('model-select').value})}).then(r=>r.json()).then(d=>{ if(d.title) { document.title = d.title + " - AI Chat"; setCurrentChatHeaderTitle(d.title); loadThreads(); } });
                } else loadThreads(false); 

            } catch(e){ 
                let syncedAfterAbort = false;
                if (e.name === 'AbortError' && !isManualStopAbortForThread(streamStartedThreadId)) {
                    syncedAfterAbort = await syncThreadAfterAbortedStream(streamStartedThreadId, { retries: 2, retryDelayMs: 180, notifyOnFailure: true });
                }
                sendClientDebugLog('error', `Prompt send error: ${e.message}`);
                if(e.name!=='AbortError') { 
                    const msg = "Connection Error: " + e.message;
                    showToast(msg, "error", true);
                }
                // Restore old message if error occurred during edit
                if (editingId && !syncedAfterAbort) restoreHiddenBranch();
            } finally { 
                setSendBtnToSendMode();
                updateFilePreview();
                if (activeStreamingBubbleId === aid) activeStreamingBubbleId = null;
                abortController=null; currentJobId=null; editingMessageId=null; setEditUi(false); 
            } 
        }

        async function resumePendingStream(pending) {
            if (abortController) return;
            if (!pending || !pending.job_id || !currentThreadId) return;
            if (isPendingJobSuppressed(pending.job_id)) return;
            const jobId = pending.job_id;
            const bubbleId = `pending-${jobId}`;
            const pendingModelRaw = (pending && pending.model) ? String(pending.model) : '';
            if (!get(bubbleId)) {
                renderPendingMessage(get('chat-container'), true, true, bubbleId, pendingModelRaw);
            }
            const adiv = get(bubbleId);
            if (!adiv) return;
            activeStreamingBubbleId = bubbleId;
            adiv.classList.add('ai-pending-bubble');
            // Ensure skeleton matches pending model even if bubble already existed
            if (!adiv.querySelector('.content-area.skeleton-pending')) {
                const ca = adiv.querySelector('.content-area');
                if (ca) ca.outerHTML = buildPendingSkeletonHtml(pendingModelRaw, '回答を生成中...');
                else adiv.insertAdjacentHTML('afterbegin', buildPendingSkeletonHtml(pendingModelRaw, '回答を生成中...'));
            }
            currentJobId = jobId;
            setSendBtnToStopMode();
            if (canvasModeEnabled) {
                resetCanvasPreviewPanel();
            }
            abortController = new AbortController();
            const resumeStartedThreadId = currentThreadId;
            const pendingModel = pendingModelRaw.toLowerCase();
            const pendingReasoningModel =
                pendingModel.includes('gemini') ||
                pendingModel.includes('o1') ||
                pendingModel.includes('o3') ||
                pendingModel.includes('gpt-5') ||
                (pendingModel.includes('reasoning') && !pendingModel.includes('non-reasoning'));
            let thoughtPlaceholderEl = null;
            const ensureThoughtPlaceholder = (text) => {
                if (!pendingReasoningModel || !adiv) return null;
                if (!thoughtPlaceholderEl || !adiv.contains(thoughtPlaceholderEl)) {
                    thoughtPlaceholderEl = adiv.querySelector('.thought-content');
                }
                if (!thoughtPlaceholderEl) {
                    const tHtml = `<div class="thought-container"><div class="thought-header thinking-shimmer" onclick="toggleThinking(this)"><i class="fas fa-brain text-purple-400"></i> Thinking Process</div><div class="thought-content collapsed" data-placeholder="1"></div></div>`;
                    adiv.insertAdjacentHTML('afterbegin', tHtml);
                    thoughtPlaceholderEl = adiv.querySelector('.thought-content');
                }
                if (thoughtPlaceholderEl) {
                    thoughtPlaceholderEl.setAttribute('data-placeholder', '1');
                    thoughtPlaceholderEl.textContent = text || '推論プロセスを準備中...';
                }
                return thoughtPlaceholderEl;
            };
            if (pendingReasoningModel) {
                ensureThoughtPlaceholder('推論プロセスを準備中...');
            }
            let buf="", acc="", tht="", first=true, thEl=null, cEl=null, searchBox=null, hadError=false;
            const pyBoxes = {};
            let lastRenderTime = 0;
            try {
                const r = await apiFetch("/chat_stream_resume", {
                    method: 'POST',
                    headers: {'Content-Type':'application/json'},
                    body: JSON.stringify({ thread_id: currentThreadId, job_id: jobId }),
                    signal: abortController.signal
                });
                if (!r.ok) {
                    throw new Error(`Resume failed (${r.status})`);
                }
                const reader = r.body.getReader();
                const dec = new TextDecoder();
                while (true) {
                    const {done, value} = await reader.read();
                    if (done) break;
                    buf += dec.decode(value, {stream:true});
                    let ls = buf.split("\n");
                    buf = ls.pop();
                    let contentChanged = false;
                    let thoughtChanged = false;
                    for (let l of ls) {
                        if (!l.trim()) continue;
                        try {
                            const j = JSON.parse(l);
                            if (j.type === 'job_id') { currentJobId = j.content || jobId; continue; }
                            if (j.type === 'search_status') {
                                if (j.content === 'searching' && !searchBox) {
                                     adiv.insertAdjacentHTML('afterbegin', `<div class="search-box visible animate-pulse mb-2"><i class="fas fa-globe"></i> Searching web...</div>`);
                                     searchBox = adiv.querySelector('.search-box');
                                } else if (j.content === 'done' && searchBox) {
                                     searchBox.classList.remove('animate-pulse');
                                     searchBox.innerHTML = '<i class="fas fa-check-circle text-green-400"></i> Search complete';
                                     setTimeout(() => { if(searchBox) searchBox.remove(); searchBox=null; }, 2000);
                                }
                                continue;
                            }
                            if (j.type === 'status') {
                                const statusText = (j.content === null || j.content === undefined) ? '' : String(j.content);
                                if (first && adiv) {
                                    const headline = statusText || 'モデル処理中...';
                                    if (!updatePendingSkeletonStatus(adiv, headline, '応答開始までの進捗を表示しています')) {
                                        const ca = adiv.querySelector('.content-area');
                                        if (ca) {
                                            ca.outerHTML = buildPendingSkeletonHtml(pendingModelRaw, headline);
                                            updatePendingSkeletonStatus(adiv, headline, '応答開始までの進捗を表示しています');
                                        }
                                    }
                                }
                                if (pendingReasoningModel) {
                                    ensureThoughtPlaceholder(statusText || '推論プロセスを準備中...');
                                }
                                continue;
                            }
                            if (first) {
                                beginPendingToStreamTransition(adiv);
                                const ca = adiv.querySelector('.content-area');
                                if (ca) ca.innerHTML = '';
                                first = false;
                            }
                            if (j.type === 'thought') {
                                if (!thEl) {
                                    thEl = adiv.querySelector('.thought-content');
                                }
                                tht += j.content;
                                if (!thEl) {
                                    const tHtml = `<div class="thought-container"><div class="thought-header" onclick="toggleThinking(this)"><i class="fas fa-brain text-purple-400"></i> Thinking Process</div><div class="thought-content"></div></div>`;
                                    if (searchBox) searchBox.insertAdjacentHTML('afterend', tHtml);
                                    else adiv.insertAdjacentHTML('afterbegin', tHtml);
                                    thEl = adiv.querySelector('.thought-content');
                                }
                                if (thEl && thEl.getAttribute('data-placeholder') === '1') {
                                    thEl.textContent = '';
                                    thEl.removeAttribute('data-placeholder');
                                    if (thEl) {
                                        const thoughtHeader = thEl.parentElement.querySelector('.thought-header');
                                        if (thoughtHeader) thoughtHeader.classList.remove('thinking-shimmer');
                                    }
                                    tht = j.content;
                                }
                                thEl.classList.remove('collapsed');
                                thoughtChanged = true;
                            } else if (j.type === 'image_analysis') {
                                const iaText = (j.content === null || j.content === undefined) ? '' : String(j.content);
                                if (!adiv) continue;
                                let iaEl = adiv.querySelector('.image-analysis-box');
                                if (!iaEl) {
                                    const iaHtml = `<div class="image-analysis-box mb-2 p-2 bg-blue-900/20 border border-blue-500/30 rounded"><div class="text-[10px] text-blue-300 font-medium mb-1"><i class="fas fa-image mr-1"></i>Image Analysis</div><div class="image-analysis-text text-[11px] text-gray-300"></div></div>`;
                                    if(searchBox) searchBox.insertAdjacentHTML('afterend', iaHtml);
                                    else adiv.insertAdjacentHTML('afterbegin', iaHtml);
                                    iaEl = adiv.querySelector('.image-analysis-box');
                                }
                                const iaTxt = iaEl.querySelector('.image-analysis-text');
                                if (iaTxt) iaTxt.textContent = iaText;
                            } else if (j.type === 'python') {
                                const py = j.content || {};
                                const pyId = py.id || `py_${Date.now()}`;
                                if (!pyBoxes[pyId]) {
                                    const boxHtml = `<div class="code-wrapper python-box collapsed" data-py-id="${pyId}" data-collapsed="true" data-code-key="${pyId}"><div class="code-header"><span class="code-lang"><i class="fas fa-terminal"></i> Python Execution</span><div class="code-actions"><button class="code-toggle" aria-expanded="false"><i class="fas fa-chevron-down"></i> Expand</button><button class="copy-btn" data-copy="code" data-code=""><i class="fas fa-copy"></i> Copy Code</button><button class="copy-btn" data-copy="output" data-code=""><i class="fas fa-copy"></i> Copy Output</button></div></div><div class="code-body"><div class="python-section"><div class="python-label">Code</div><pre><code class="hljs language-python python-code"></code></pre></div><div class="python-section"><div class="python-label">Output</div><pre><code class="hljs language-plaintext python-output"></code></pre></div></div></div>`;
                                    if (searchBox) searchBox.insertAdjacentHTML('afterend', boxHtml);
                                    else adiv.insertAdjacentHTML('afterbegin', boxHtml);
                                    pyBoxes[pyId] = adiv.querySelector(`[data-py-id="${pyId}"]`);
                                }
                                const box = pyBoxes[pyId];
                                if (box) {
                                    if (py.code !== undefined) {
                                        const codeText = py.code == null ? '' : String(py.code);
                                        const codeEl = box.querySelector('.python-code');
                                        if (codeEl) {
                                            codeEl.textContent = codeText;
                                            codeEl.removeAttribute('data-highlighted');
                                            queueHighlight(box, codeText);
                                        }
                                        const codeBtn = box.querySelector('.copy-btn[data-copy="code"]');
                                        if (codeBtn) codeBtn.setAttribute('data-code', encodeURIComponent(codeText).replace(/'/g, "%27"));
                                    }
                                    if (py.output !== undefined) {
                                        const outText = py.output == null ? '' : String(py.output);
                                        const outEl = box.querySelector('.python-output');
                                        if (outEl) outEl.textContent = outText;
                                        const outBtn = box.querySelector('.copy-btn[data-copy="output"]');
                                        if (outBtn) outBtn.setAttribute('data-code', encodeURIComponent(outText).replace(/'/g, "%27"));
                                    }
                                }
                            } else if (j.type === 'content') {
                                const contentDelta = (j.content === null || j.content === undefined) ? '' : String(j.content);
                                acc += contentDelta;
                                if (!cEl) {
                                    cEl = adiv.querySelector('.content-area') || document.createElement('div');
                                    cEl.className = 'prose prose-invert text-sm break-words';
                                    if (!adiv.contains(cEl)) adiv.appendChild(cEl);
                                }
                                contentChanged = true;
                            } else if (j.type === 'error') {
                                hadError = true;
                                adiv.insertAdjacentHTML('beforeend', `<div class="text-red-400 text-xs mt-2 border border-red-500 p-2 rounded">Error: ${escapeHtml(j.content)}</div>`);
                                showToast(j.content || "Unknown error", "error", true);
                            }
                        } catch (e) {}
                    }
                    if (thoughtChanged && thEl) {
                        thEl.textContent = tht;
                        if (userAutoScroll) thEl.scrollTop = thEl.scrollHeight;
                    }
                    if (contentChanged && cEl) {
                        const now = Date.now();
                        if (now - lastRenderTime > 100) {
                            const collapseState = snapshotCodeCollapse(cEl);
                            renderAiMarkdownInto(cEl, acc);
                            applyCodeCollapse(cEl, collapseState, true);
                            lastRenderTime = now;
                        }
                    }
                    scrollToBottom();
                }
                // Final render to catch any remaining content
                if (cEl) {
                    const collapseState = snapshotCodeCollapse(cEl);
                    renderAiMarkdownInto(cEl, acc);
                    applyCodeCollapse(cEl, collapseState, true);
                }

                vibrateHelper([100, 50, 100]);

                if (adiv) {
                    queueHighlight(adiv, acc);
                    queueMathTypeset(adiv, acc);
                }

                if (!hadError) {
                    const thoughts = adiv.querySelectorAll('.thought-content');
                    thoughts.forEach(t => t.classList.add('collapsed'));
                    await loadMessages(currentThreadId, { preserveDraft: true, silent: true });
                    loadThreads(false);
                }
            } catch (e) {
                if (e.name === 'AbortError' && !isManualStopAbortForThread(resumeStartedThreadId)) {
                    await syncThreadAfterAbortedStream(resumeStartedThreadId, { retries: 2, retryDelayMs: 180, notifyOnFailure: true });
                }
                if (e.name !== 'AbortError') {
                    const msg = "Connection Error: " + e.message;
                    showToast(msg, "error", true);
                }
            } finally {
                setSendBtnToSendMode();
                updateFilePreview();
                if (activeStreamingBubbleId === bubbleId) activeStreamingBubbleId = null;
                abortController = null;
                currentJobId = null;
                currentThreadPending = null;
            }
        }

        function updateThreadHighlighting() {
            const list = get('thread-list');
            if (!list) return;
            const threads = list.querySelectorAll('[data-thread-id]');
            threads.forEach(el => {
                const tid = el.dataset.threadId;
                if (tid === String(currentThreadId)) {
                    el.classList.add('bg-gray-700/60', 'border-l-2', 'border-blue-500');
                } else {
                    el.classList.remove('bg-gray-700/60', 'border-l-2', 'border-blue-500');
                }
            });
        }

        async function loadThreads(append=false) { 
            if(threadLoading) return; 
            threadLoading = true;
            if(!append) { 
                threadPage = 1; 
                hasMoreThreads = true; 
                get('thread-list').innerHTML = '<div id="scroll-sentinel"></div>'; 
                if (threadObserver) {
                    threadObserver.disconnect();
                    threadObserver.observe(get('scroll-sentinel'));
                }
            }
            
            const r = await apiFetch(`${CHAT_CONFIG.urls.handleThreads}?q=${encodeURIComponent(get('search-box').value)}&page=${threadPage}`); 
            const d = await r.json(); 
            const l = get('thread-list');
            const sentinel = get('scroll-sentinel');
            
            d.threads.forEach((t, i) => { 
                const tid = String(t.id);
                const d = document.createElement('div'); 
                const star = t.is_bookmarked ? 'text-yellow-400' : 'text-gray-500';
                const tempBadge = t.is_temporary ? '<span class="text-[9px] text-amber-300 border border-amber-500/50 rounded px-1 py-0">一時</span>' : '';
                
                // Active highlighting
                const isActive = (tid === String(currentThreadId));
                const activeClass = isActive ? 'bg-gray-700/60 border-l-2 border-blue-500' : '';
                
                d.className = `p-2 rounded hover:bg-gray-700 cursor-pointer text-sm text-gray-300 truncate flex justify-between items-center group model-list-animate opacity-0 ${activeClass}`; 
                d.dataset.threadId = tid;
                d.style.animationDelay = `${i * 0.035}s`;
                d.innerHTML = `<div class="flex items-center gap-1 truncate flex-1"><button class="${star} hover:text-yellow-400 px-1" onclick="toggleBookmark(event, '${tid}')"><i class="fas fa-star text-[10px]"></i></button><span class="truncate">${escapeHtml(t.title || "No Title")}</span>${tempBadge}</div><div class="flex gap-1 opacity-0 group-hover:opacity-100" data-thread-actions="1"><button class="text-gray-500 hover:text-white px-1" onclick="renameThread(event, '${tid}')"><i class="fas fa-pen text-xs"></i></button><button class="text-gray-500 hover:text-red-400 px-1" onclick="deleteThread(event, '${tid}')"><i class="fas fa-trash text-xs"></i></button></div>`;
                d.onclick = (e) => {
                    if (e.target.closest('button') || e.target.closest('[data-thread-actions]')) return;
                    loadMessages(tid);
                };
                l.insertBefore(d, sentinel); 
            });
            
            hasMoreThreads = d.has_next;
            if(hasMoreThreads) threadPage++;
            threadLoading = false;
            updateThreadHighlighting();
        }

        async function toggleBookmark(e, tid) {
            if (e) e.stopPropagation();
            await apiFetch(`/api/threads/${tid}/bookmark`, {method:'POST'});
            loadThreads();
        }

        async function loadMessages(tid, opts = {}) { 
            const loadSequence = ++threadLoadSequence;
            if (window.closeHistoryModal) window.closeHistoryModal();
            const preserveDraft = !!opts.preserveDraft;
            const silent = !!opts.silent;
            const codeState = silent ? snapshotCodeCollapseByMessage(get('chat-container')) : null;
            let draftText = '';
            let draftHeight = '';
            let draftImages = [];
            if (preserveDraft) {
                const input = get('prompt-input');
                draftText = input ? input.value : '';
                draftHeight = input ? input.style.height : '';
                draftImages = currentImageUrls ? currentImageUrls.slice() : [];
                editingMessageId = null;
                setEditUi(false);
            } else {
                cancelEdit();
            }
            currentThreadId = tid !== null && tid !== undefined ? String(tid) : tid; 
            history.pushState({}, '', '/c/' + tid); 
            updateThreadHighlighting();
            syncActiveGemForThread(currentThreadId); 
            get('welcome-screen').classList.add('hidden'); 
            if (!silent) {
                // Show shimmer skeleton while history loads (open from list / page reload)
                get('chat-container').innerHTML = buildChatLoadingSkeletonHtml();
            }
            try {
                const threadUrl = new URL(CHAT_CONFIG.urls.handleThreadItem.replace('0', tid), window.location.origin);
                threadUrl.searchParams.set('limit', String(getEffectiveThreadInitialMessageLimit()));
                const r = await apiFetch(threadUrl.toString());
                if (!r.ok) throw new Error(`thread request failed (${r.status})`);
                const threadData = await r.json();
                if (!threadData || !Array.isArray(threadData.messages)) throw new Error('invalid thread response');
                if (loadSequence !== threadLoadSequence) return;
            setCurrentChatHeaderTitle(threadData && threadData.title);
            allMessages = threadData.messages;
            threadHasOlderMessages = !!threadData.has_older_messages;
            oldestLoadedMessageId = threadData.oldest_loaded_id || (allMessages.length ? allMessages[0].id : null);

            // Load prompt history from thread messages
            const threadUserPrompts = (allMessages || [])
                .filter(m => m.role === 'user' && m.content)
                .map(m => m.content);
            
            // Deduplicate (latest sent should be first in promptHistory for ArrowUp)
            promptHistory = [...new Set(threadUserPrompts.slice().reverse())];
            historyIndex = -1;
            tempPrompt = "";

            currentThreadPending = threadData.pending_job || null;
            setTemporaryChatUiState(!!(threadData && threadData.is_temporary));
            applyTemporaryChatRuntimeMeta(threadData || {});
            ensureTemporaryChatHeartbeat(true);
            
            // Update thread-specific settings UI
            if (get('thread-custom-instruction')) {
                get('thread-custom-instruction').value = threadData.custom_instruction || '';
            }
            if (threadData.last_model) {
                selectModelById(threadData.last_model);
            }
            if (get('enable-prompt-cache')) {
                get('enable-prompt-cache').checked = !!threadData.enable_prompt_caching;
                updatePromptCacheUi();
            }
            if (threadData.last_gem_uuid && loadedGems.length > 0) {
                const gem = loadedGems.find(g => g.uuid === threadData.last_gem_uuid);
                if (gem) {
                    threadGemMap[currentThreadId] = gem;
                    applyActiveGem(gem);
                }
            }
            
            // Set default leaf (latest message or fixed branch)
            const storedFixedId = localStorage.getItem(`fixed_branch_${currentThreadId}`);
            if (storedFixedId && allMessages.find(m => String(m.id) === String(storedFixedId))) {
                currentLeafId = storedFixedId;
            } else if (allMessages.length > 0) {
                currentLeafId = allMessages[allMessages.length - 1].id;
            } else {
                currentLeafId = null;
            }
            
            renderThreadTree({ silent });
            if (silent && codeState) {
                applyCodeCollapseByMessage(get('chat-container'), codeState, true);
            } else if (!silent) {
                applyCodeCollapseByMessage(get('chat-container'), null, true);
            }
            if (currentThreadPending && !silent && !isPendingJobSuppressed(currentThreadPending.job_id)) {
                resumePendingStream(currentThreadPending);
            }
            if (preserveDraft) {
                const input = get('prompt-input');
                if (input) {
                    input.value = draftText || '';
                    if (draftHeight) input.style.height = draftHeight;
                    else input.style.height = 'auto';
                }
                currentImageUrls = draftImages;
                if (currentImageUrls && currentImageUrls.length) {
                    get('file-preview').classList.remove('hidden');
                    get('file-name').innerText = `${currentImageUrls.length} files ready`;
                } else {
                    get('file-preview').classList.add('hidden');
                }
                schedulePromptTokenEstimate(true);
            }
            if (!preserveDraft) schedulePromptTokenEstimate(true);
            if(window.innerWidth < 768) get('overlay').click(); 
            } catch (err) {
                if (loadSequence !== threadLoadSequence) return;
                console.error('Failed to load chat thread:', err);
                if (!silent) showChatLoadError(tid);
                showToast('チャットの読み込みに失敗しました', 'error', true);
            }
        }

        async function loadOlderMessages() {
            if (loadingOlderMessages || !currentThreadId || !threadHasOlderMessages || !oldestLoadedMessageId) return;
            loadingOlderMessages = true;
            const container = get('chat-container');
            const oldHeight = container ? container.scrollHeight : 0;
            const oldTop = container ? container.scrollTop : 0;
            try {
                const url = new URL(CHAT_CONFIG.urls.handleThreadItem.replace('0', currentThreadId), window.location.origin);
                url.searchParams.set('before_id', String(oldestLoadedMessageId));
                url.searchParams.set('limit', String(getEffectiveThreadOlderPageSize()));
                url.searchParams.set('include_meta', '0');
                const r = await apiFetch(url.toString());
                const data = await r.json();
                const older = Array.isArray(data.messages) ? data.messages : [];
                if (older.length) {
                    const existing = new Set(allMessages.map(m => m.id));
                    const merged = older.filter(m => !existing.has(m.id));
                    if (merged.length) {
                        allMessages = merged.concat(allMessages);
                    }
                }
                threadHasOlderMessages = !!data.has_older_messages;
                oldestLoadedMessageId = data.oldest_loaded_id || (allMessages.length ? allMessages[0].id : null);
                renderThreadTree({ silent: true, keepScroll: true });
                if (container) {
                    const newHeight = container.scrollHeight;
                    container.scrollTop = Math.max(0, oldTop + (newHeight - oldHeight));
                }
            } catch (e) {
                showToast('過去メッセージの読み込みに失敗しました', 'error', true);
            } finally {
                loadingOlderMessages = false;
                const btn = get('load-older-messages-btn');
                if (btn && threadHasOlderMessages) {
                    btn.disabled = false;
                    btn.innerHTML = '<i class="fas fa-clock-rotate-left mr-1"></i>過去メッセージを読み込む';
                }
            }
        }

        function renderThreadTree(opts = {}) {
            const silent = !!opts.silent;
            const animateMessages = !!opts.animate && !silent;
            const keepScroll = !!opts.keepScroll;
            const container = get('chat-container');
            if (!container) return;
            
            // Always clear and rebuild to ensure the UI reflects the current state (allMessages/currentLeafId).
            // Silent mode in loadMessages handles skipping the spinner, but we still need to swap the content here.
            container.innerHTML = '';
            
            if (allMessages.length === 0) {
                currentParentId = null;
                updateTotalTokenBar(0);
                return;
            }

            // Build map and find children
            const msgMap = {};
            allMessages.forEach(m => {
                msgMap[m.id] = m;
                m.childrenIds = [];
            });
            allMessages.forEach(m => {
                if (m.parent_id && msgMap[m.parent_id]) {
                    msgMap[m.parent_id].childrenIds.push(m.id);
                }
            });

            // If currentLeafId is not set or not in messages, pick the latest
            if (!currentLeafId || !msgMap[currentLeafId]) {
                currentLeafId = allMessages.length > 0 ? allMessages[allMessages.length - 1].id : null;
            }

            // Trace path from leaf to root
            const path = [];
            let curr = msgMap[currentLeafId];
            while (curr) {
                path.unshift(curr);
                curr = msgMap[curr.parent_id];
            }

            // Render path
            const pathTotals = buildTokenTotals(path);
            const allBranchTotals = buildTokenTotals(allMessages);
            const fragment = document.createDocumentFragment();
            if (threadHasOlderMessages) {
                const countText = loadingOlderMessages ? '読み込み中...' : '過去メッセージを読み込む';
                const disabledAttr = loadingOlderMessages ? 'disabled' : '';
                const olderDiv = document.createElement('div');
                olderDiv.className = 'mb-3 text-center';
                olderDiv.innerHTML = `<button id="load-older-messages-btn" class="px-3 py-1.5 text-xs rounded border border-gray-600 text-gray-200 hover:bg-gray-800 disabled:opacity-50 disabled:cursor-not-allowed" onclick="loadOlderMessages()" ${disabledAttr}><i class="fas fa-clock-rotate-left mr-1"></i>${countText}</button>`;
                fragment.appendChild(olderDiv);
            }
            path.forEach(m => {
                const parent = m.parent_id ? msgMap[m.parent_id] : null;
                const siblings = parent ? parent.childrenIds : allMessages.filter(x => !x.parent_id).map(x => x.id);
                const versionInfo = siblings.length > 1 ? {
                    current: siblings.indexOf(m.id) + 1,
                    total: siblings.length,
                    siblings: siblings
                } : null;
                
                renderMessage(
                    m.id,
                    m.role,
                    m.content,
                    m.image_url,
                    m.thought_data,
                    m.model,
                    versionInfo,
                    animateMessages,
                    m.quote_text,
                    m.tokens,
                    m.tokens_in,
                    m.tokens_out,
                    m.is_encrypted,
                    m.tokens_content,
                    m.tokens_thought,
                    fragment,
                    false,
                    m.parent_id,
                    m.gem_name
                );
            });
            const pending = currentThreadPending;
            if (pending && !isPendingJobSuppressed(pending.job_id)) {
                const pendingId = pending.message_id;
                const pathIds = new Set(path.map(p => p.id));
                const lastMsg = path.length ? path[path.length - 1] : null;
                const shouldRender = (pendingId && pathIds.has(pendingId) && currentLeafId === pendingId)
                    || (!pendingId && lastMsg && lastMsg.role === 'user');
                if (shouldRender) {
                    const bubbleId = pending.job_id ? `pending-${pending.job_id}` : null;
                    renderPendingMessage(fragment, animateMessages, false, bubbleId, pending.model || null);
                }
            }
            
            container.appendChild(fragment);

            updateTotalTokenBar(pathTotals.tokens_total, pathTotals, allBranchTotals);
            currentParentId = currentLeafId;
            if (!keepScroll) scrollToBottom();
            if (lowBandwidthMode) {
                queueMessageDecorations(container, container ? (container.textContent || '') : '');
            } else {
                queueHighlight(container);
                if (path.length) {
                    const latestText = path[path.length - 1] && path[path.length - 1].content;
                    queueMathTypeset(container, latestText);
                }
            }
        }

        function switchVersion(targetId) {
            currentLeafId = targetId;
            // When switching, we might want to find the "latest" leaf of the new branch
            const msgMap = {};
            allMessages.forEach(m => {
                msgMap[m.id] = m;
                m.childrenIds = [];
            });
            allMessages.forEach(m => {
                if (m.parent_id && msgMap[m.parent_id]) {
                    msgMap[m.parent_id].childrenIds.push(m.id);
                }
            });

            let currId = targetId;
            if (!msgMap[currId]) {
                currentLeafId = allMessages.length > 0 ? allMessages[allMessages.length - 1].id : null;
                renderThreadTree({ animate: true });
                return;
            }
            while (msgMap[currId] && msgMap[currId].childrenIds.length > 0) {
                // Pick the first child (or latest? let's pick latest child by ID)
                const children = msgMap[currId].childrenIds;
                currId = Math.max(...children);
            }
            currentLeafId = currId;
            renderThreadTree({ animate: true });
        }
        async function loadGems() { 
            const r = await apiFetch(CHAT_CONFIG.urls.handleGems); 
            const gs = await r.json(); 
            loadedGems = gs;
            const l = get('gem-list'); 
            l.innerHTML = ''; 
            gs.forEach((g, i) => { 
                const d = document.createElement('div'); 
                d.className = 'gem-item p-2 rounded hover:bg-gray-700 cursor-pointer text-sm text-gray-300 flex justify-between items-center group model-list-animate opacity-0'; 
                d.style.animationDelay = `${i * 0.05}s`;
                d.innerHTML = `<div class="flex items-center gap-2 overflow-hidden"><i class="fas fa-gem text-blue-500"></i><span class="truncate">${escapeHtml(g.name)}</span></div><div class="flex items-center gap-1"><button class="text-gray-400 hover:text-blue-400 opacity-100 md:opacity-0 md:group-hover:opacity-100 px-2 transition" onclick="openEditGemModal(event,'${g.uuid}')"><i class="fas fa-pencil-alt text-[10px]"></i></button><button class="text-gray-400 hover:text-red-400 opacity-100 md:opacity-0 md:group-hover:opacity-100 px-2 transition" onclick="deleteGem(event,'${g.uuid}')"><i class="fas fa-trash text-[10px]"></i></button></div>`;
                d.onclick = (e) => { if(!e.target.closest('button')) activateGem(g); }; 
                l.appendChild(d); 
            }); 
        }
        async function openEditGemModal(e, id) {
            e.stopPropagation();
            editingGemUuid = id;
            try {
                const r = await apiFetch(`/api/gems/${id}`);
                const g = await r.json();
                get('gem-name').value = g.name;
                get('gem-desc').value = g.description || '';
                get('gem-inst').value = g.instruction;
                get('gem-default-model').value = g.default_model || '';
                renderGemFixedPromptsForEdit(g.fixed_prompts);
                get('gem-modal-title').innerHTML = `<i class="fas fa-gem text-blue-500 mr-2"></i>Edit Gem`;
                get('save-gem-btn').innerText = "Save Changes";
                showModal('gem-modal');
                if (location.pathname !== '/gem') {
                    history.pushState({ modal: 'gem' }, '', '/gem');
                }
            } catch (err) {
                showToast("Gemの取得に失敗しました", "error", true);
            }
        }
        async function createGem(name, inst) { await apiFetch(CHAT_CONFIG.urls.handleGems, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({name, instruction:inst})}); loadGems(); }

        function applyActiveGem(g) {
            activeGem = g || null;
            const bar = get('fixed-prompts-bar');
            if (activeGem) {
                if (activeGem.default_model) {
                    selectModelById(activeGem.default_model);
                }
                get('active-gem-name').innerText = activeGem.name;
                get('gem-active-indicator').classList.remove('hidden');
                
                if (bar) {
                    bar.innerHTML = '';
                    let prompts = [];
                    try {
                        if (activeGem.fixed_prompts) prompts = JSON.parse(activeGem.fixed_prompts);
                    } catch(e) {}
                    
                    if (prompts.length > 0) {
                        bar.classList.remove('hidden');
                        prompts.forEach((p, i) => {
                            const btn = document.createElement('button');
                            btn.className = 'fixed-prompt-chip whitespace-nowrap px-4 py-1.5 text-[11px] font-bold bg-gray-700 hover:bg-gray-600 text-gray-100 rounded-full transition-all shadow-md border border-gray-600/50 flex items-center';
                            btn.style.animationDelay = `${i * 40}ms`;
                            btn.textContent = String(p.name || '');
                            btn.onclick = () => {
                                const input = get('prompt-input');
                                if (input) {
                                    input.value = p.content;
                                    input.dispatchEvent(new Event('input'));
                                    sendMessage();
                                }
                            };
                            bar.appendChild(btn);
                        });
                    } else {
                        bar.classList.add('hidden');
                    }
                }
            } else {
                get('gem-active-indicator').classList.add('hidden');
                if (bar) {
                    bar.innerHTML = '';
                    bar.classList.add('hidden');
                }
            }
            get('sys-prompt-option').style.opacity = '1';
        }
        function syncActiveGemForThread(tid) {
            const g = tid && threadGemMap[tid] ? threadGemMap[tid] : null;
            applyActiveGem(g);
        }
        async function saveThreadGemUuid(threadId, gemUuid) {
            try {
                await apiFetch(CHAT_CONFIG.urls.handleSettings, {
                    method:'POST',
                    headers:{'Content-Type':'application/json'},
                    body:JSON.stringify({last_gem_uuid: gemUuid, thread_id: threadId})
                });
            } catch(e) {}
        }
        function activateGem(g, skipSave) {
            if (currentThreadId) {
                threadGemMap[currentThreadId] = g;
                applyActiveGem(g);
                showToast(`Gem "${g.name}" をこのチャットに適用しました`, "success");
                if (!skipSave) saveThreadGemUuid(currentThreadId, g ? g.uuid : null);
            } else {
                pendingGemForNewThread = g;
                applyActiveGem(g);
                if (allMessages && allMessages.length > 0) {
                    startNewChat({ preserveGem: true });
                }
            }
        }
        function clearActiveGem() { if (currentThreadId) { delete threadGemMap[currentThreadId]; saveThreadGemUuid(currentThreadId, null); } pendingGemForNewThread = null; applyActiveGem(null); }

        // Gem Fixed Prompts UI helper
        function addGemFixedPromptRow(name = '', content = '') {
            const container = get('gem-fixed-prompts-container');
            if (!container) return;
            const div = document.createElement('div');
            div.className = 'flex gap-2 items-start gem-fixed-prompt-row ui-enter';
            div.innerHTML = `
                <input type="text" class="gem-fp-name bg-gray-900 border border-gray-600 rounded p-1.5 text-white text-[10px] w-24" placeholder="名前" value="${escapeHtml(name)}" autocomplete="off" spellcheck="false">
                <textarea class="gem-fp-content flex-1 bg-gray-900 border border-gray-600 rounded p-1.5 text-white text-[10px] h-9 resize-none" placeholder="プロンプト内容" spellcheck="false">${escapeHtml(content)}</textarea>
                <button type="button" class="text-gray-500 hover:text-red-400 p-1.5" onclick="this.parentElement.remove()"><i class="fas fa-times"></i></button>
            `;
            container.appendChild(div);
        }

        function collectGemFixedPrompts() {
            const rows = document.querySelectorAll('.gem-fixed-prompt-row');
            const prompts = [];
            rows.forEach(row => {
                const name = row.querySelector('.gem-fp-name').value.trim();
                const content = row.querySelector('.gem-fp-content').value.trim();
                if (name && content) {
                    prompts.push({ name, content });
                }
            });
            return prompts.length > 0 ? JSON.stringify(prompts) : null;
        }

        function renderGemFixedPromptsForEdit(fixedPromptsJson) {
            const container = get('gem-fixed-prompts-container');
            if (!container) return;
            container.innerHTML = '';
            try {
                if (fixedPromptsJson) {
                    const prompts = JSON.parse(fixedPromptsJson);
                    prompts.forEach(p => addGemFixedPromptRow(p.name, p.content));
                }
            } catch(e) {}
        }
        function getCurrentChatHeaderTitleText() {
            if (typeof currentThreadTitle === 'string' && currentThreadTitle.trim()) return currentThreadTitle.trim();
            if (currentThreadId) return 'No Title';
            return 'AI Chat';
        }
        function getTemporaryChatTimeoutLabel() {
            if (!temporaryChatEnabled) return '';
            const sec = normalizeTemporaryChatTimeoutSeconds(temporaryChatTimeoutSeconds);
            return `${sec}秒`;
        }
        function updateCurrentChatHeaderUi() {
            const titleText = getCurrentChatHeaderTitleText();
            const timeoutLabel = getTemporaryChatTimeoutLabel();
            const showTempLabel = !!temporaryChatEnabled;
            const titleTargets = ['sidebar-chat-title', 'mobile-chat-title'];
            const tempLabelTargets = ['sidebar-chat-temporary-label', 'mobile-chat-temporary-label'];
            const ttlTargets = ['sidebar-chat-ttl', 'mobile-chat-ttl'];
            titleTargets.forEach((id) => {
                const el = get(id);
                if (el) el.textContent = titleText;
            });
            tempLabelTargets.forEach((id) => {
                const el = get(id);
                if (el) el.classList.toggle('hidden', !showTempLabel);
            });
            ttlTargets.forEach((id) => {
                const el = get(id);
                if (!el) return;
                if (showTempLabel && timeoutLabel) {
                    el.textContent = timeoutLabel;
                    el.classList.remove('hidden');
                } else {
                    el.textContent = '';
                    el.classList.add('hidden');
                }
            });
        }
        function setCurrentChatHeaderTitle(title) {
            currentThreadTitle = typeof title === 'string' ? title : null;
            updateCurrentChatHeaderUi();
        }
        function resetTemporaryChatExpiresAt() {
            tempChatExpiresAtMs = null;
            updateCurrentChatHeaderUi();
        }
        function applyTemporaryChatRuntimeMeta(data) {
            if (!data || typeof data !== 'object') return;
            if (Object.prototype.hasOwnProperty.call(data, 'timeout_seconds')) {
                applyTemporaryChatTimeoutSeconds(data.timeout_seconds);
            }
            let nextExpiresAtMs = null;
            const exp = Number(data.temp_chat_expires_at);
            if (Number.isFinite(exp) && exp > 0) {
                nextExpiresAtMs = Math.floor(exp * 1000);
            } else {
                const remaining = Number(data.temp_chat_remaining_seconds);
                if (Number.isFinite(remaining) && remaining >= 0) {
                    nextExpiresAtMs = Date.now() + Math.floor(remaining * 1000);
                }
            }
            if (nextExpiresAtMs !== null) {
                tempChatExpiresAtMs = nextExpiresAtMs;
            } else if (data.is_temporary === false || !temporaryChatEnabled) {
                tempChatExpiresAtMs = null;
            }
            updateCurrentChatHeaderUi();
        }
        function ensureCurrentChatHeaderTicker() {}
        function normalizeTemporaryChatTimeoutSeconds(value, fallback = TEMP_CHAT_DEFAULT_TIMEOUT_SECONDS) {
            let sec = Number(value);
            if (!Number.isFinite(sec)) sec = Number(fallback);
            if (!Number.isFinite(sec)) sec = TEMP_CHAT_DEFAULT_TIMEOUT_SECONDS;
            sec = Math.trunc(sec);
            if (sec < TEMP_CHAT_TIMEOUT_MIN_SECONDS) sec = TEMP_CHAT_TIMEOUT_MIN_SECONDS;
            if (sec > TEMP_CHAT_TIMEOUT_MAX_SECONDS) sec = TEMP_CHAT_TIMEOUT_MAX_SECONDS;
            return sec;
        }
        function updateTemporaryChatDescriptionText() {
            const sec = normalizeTemporaryChatTimeoutSeconds(temporaryChatTimeoutSeconds);
            const desc = `このページが非表示/切断の状態で ${sec} 秒経過すると、この一時チャットとこのチャットでアップロードした添付を自動削除します（ライブラリ添付は除外）。`;
            const welcome = get('temporary-chat-welcome-desc');
            if (welcome) welcome.textContent = desc;
            const tempToggle = get('temporary-chat-container');
            if (tempToggle) tempToggle.title = `切断後 ${sec} 秒で、このチャットとアップロード添付を自動削除`;
        }
        function applyTemporaryChatTimeoutSeconds(value) {
            temporaryChatTimeoutSeconds = normalizeTemporaryChatTimeoutSeconds(value, temporaryChatTimeoutSeconds);
            const input = get('set-temp-chat-timeout-seconds');
            if (input) input.value = String(temporaryChatTimeoutSeconds);
            updateTemporaryChatDescriptionText();
            updateCurrentChatHeaderUi();
            if (temporaryChatEnabled) ensureTemporaryChatHeartbeat(false);
        }
        function getTemporaryChatHeartbeatIntervalMs() {
            const sec = normalizeTemporaryChatTimeoutSeconds(temporaryChatTimeoutSeconds);
            const byTimeout = Math.floor((sec * 1000) / 3);
            return Math.max(TEMP_CHAT_HEARTBEAT_MIN_MS, Math.min(TEMP_CHAT_HEARTBEAT_MAX_MS, byTimeout));
        }
        function setTemporaryChatUiState(enabled) {
            temporaryChatEnabled = !!enabled;
            const chk = get('enable-temporary-chat');
            if (chk && chk.checked !== temporaryChatEnabled) chk.checked = temporaryChatEnabled;
            const welcomeDefault = get('welcome-default-content');
            if (welcomeDefault) welcomeDefault.classList.toggle('hidden', temporaryChatEnabled);
            const welcomeTemporary = get('welcome-temporary-content');
            if (welcomeTemporary) welcomeTemporary.classList.toggle('hidden', !temporaryChatEnabled);
            if (!temporaryChatEnabled) tempChatExpiresAtMs = null;
            updateTemporaryChatDescriptionText();
            updateCurrentChatHeaderUi();
        }
        function stopTemporaryChatHeartbeat() {
            if (tempChatHeartbeatTimer) {
                clearInterval(tempChatHeartbeatTimer);
                tempChatHeartbeatTimer = null;
            }
            tempChatHeartbeatIntervalMs = 0;
            tempChatHeartbeatInFlight = false;
        }
        function canHeartbeatTemporaryChat() {
            return !!(temporaryChatEnabled && currentThreadId && document.visibilityState === 'visible');
        }
        async function sendTemporaryChatHeartbeat(force = false) {
            if (!canHeartbeatTemporaryChat()) return;
            if (tempChatHeartbeatInFlight && !force) return;
            tempChatHeartbeatInFlight = true;
            try {
                const res = await apiFetch('/api/temporary_chat/heartbeat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ thread_id: currentThreadId, active: true })
                });
                const data = await res.json().catch(() => ({}));
                if (res.ok && data) applyTemporaryChatRuntimeMeta(data);
                if (res.ok && data && data.is_temporary === false) {
                    setTemporaryChatUiState(false);
                    stopTemporaryChatHeartbeat();
                }
            } catch (e) {
            } finally {
                tempChatHeartbeatInFlight = false;
            }
        }
        function ensureTemporaryChatHeartbeat(force = false) {
            if (!temporaryChatEnabled || !currentThreadId) {
                stopTemporaryChatHeartbeat();
                return;
            }
            const nextInterval = getTemporaryChatHeartbeatIntervalMs();
            if (!tempChatHeartbeatTimer || tempChatHeartbeatIntervalMs !== nextInterval) {
                if (tempChatHeartbeatTimer) clearInterval(tempChatHeartbeatTimer);
                tempChatHeartbeatIntervalMs = nextInterval;
                tempChatHeartbeatTimer = setInterval(() => {
                    sendTemporaryChatHeartbeat(false);
                }, tempChatHeartbeatIntervalMs);
            }
            if (force) sendTemporaryChatHeartbeat(true);
        }
        async function applyTemporaryChatSetting(enabled) {
            const next = !!enabled;
            setTemporaryChatUiState(next);
            if (!currentThreadId) {
                ensureTemporaryChatHeartbeat(true);
                return true;
            }
            try {
                const res = await apiFetch(`/api/threads/${currentThreadId}/settings`, {
                    method: 'PUT',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ is_temporary: next })
                });
                const data = await res.json().catch(() => ({}));
                if (!res.ok) throw new Error((data && data.error) || '設定更新に失敗しました');
                setTemporaryChatUiState(!!(data && data.is_temporary));
                applyTemporaryChatRuntimeMeta(data || {});
                ensureTemporaryChatHeartbeat(true);
                return true;
            } catch (err) {
                showToast("一時チャット設定の更新に失敗しました", "error", true);
                return false;
            }
        }
        function startNewChat(opts = {}) { 
            threadLoadSequence++;
            if(abortController) abortController.abort(); 
            cancelEdit(); 
            resetUploadState(); 
            stopTemporaryChatHeartbeat();
            setTemporaryChatUiState(false);
            currentThreadTitle = null;
            tempChatExpiresAtMs = null;
            currentThreadId = null; 
            allMessages = []; 
            promptHistory = [];
            historyIndex = -1;
            tempPrompt = "";
            threadHasOlderMessages = false;
            oldestLoadedMessageId = null;
            loadingOlderMessages = false;
            currentLeafId = null; 
            currentParentId = null; 
            currentThreadPending = null;
            updateTotalTokenBar(0); 
            history.pushState({}, '', '/'); 
            get('chat-container').innerHTML = ''; 
            get('welcome-screen').classList.remove('hidden'); 
            updateCurrentChatHeaderUi();
            if (get('thread-custom-instruction')) get('thread-custom-instruction').value = '';
            if (get('enable-prompt-cache')) {
                get('enable-prompt-cache').checked = false;
                updatePromptCacheUi();
            }
            if (!opts.preserveGem) {
                applyActiveGem(null);
            } else if (activeGem) {
                applyActiveGem(activeGem);
            }
            loadThreads();
 
            if(window.innerWidth < 768) get('overlay').click(); 
        }

        let threadModalLoadSeq = 0;
        window.openThreadModal = async () => {
            if (!currentThreadId) {
                try {
                    const r = await apiFetch(CHAT_CONFIG.urls.handleThreads, {
                        method:'POST',
                        headers:{'Content-Type':'application/json'},
                        body: JSON.stringify({ is_temporary: temporaryChatEnabled })
                    }); 
                    const d = await r.json(); 
                    currentThreadId = d.id !== null && d.id !== undefined ? String(d.id) : d.id; 
                    setTemporaryChatUiState(!!(d && d.is_temporary));
                    setCurrentChatHeaderTitle(d && d.title);
                    applyTemporaryChatRuntimeMeta(d || {});
                    ensureTemporaryChatHeartbeat(true);
                    history.pushState({}, '', '/c/' + d.id);
                    loadThreads();
                } catch (err) {
                    showToast("チャットの作成に失敗しました", "error", true);
                    return;
                }
            }
            const modalLoadSeq = ++threadModalLoadSeq;
            const targetThreadId = String(currentThreadId);
            modalThreadId = targetThreadId;
            showModal('thread-modal');
            if (location.pathname !== '/chat-settings') {
                history.pushState({ modal: 'thread' }, '', '/chat-settings');
            }
            try {
                const [settingsRes, threadSettingsRes] = await Promise.all([
                    apiFetch(CHAT_CONFIG.urls.handleSettingsQuery),
                    apiFetch(`/api/threads/${targetThreadId}/settings`)
                ]);
                if (modalLoadSeq !== threadModalLoadSeq) return;
                if (modalThreadId !== targetThreadId) return;
                if (settingsRes.ok) {
                    const d = await settingsRes.json();
                    
                    const threadGlobalPreview = get('thread-app-global-sys-prompt-preview');
                    if (threadGlobalPreview) {
                        threadGlobalPreview.value = d.global_system_prompt_effective || '';
                    }
                    const threadGlobalPreviewStatus = get('thread-app-global-sys-prompt-preview-status');
                    if (threadGlobalPreviewStatus) {
                        if (d.global_system_prompt_enabled === false) {
                            threadGlobalPreviewStatus.textContent = '現在は無効化されています。';
                        } else if (d.global_system_prompt_uses_time_fallback) {
                            threadGlobalPreviewStatus.textContent = '管理者設定が空欄のため、時刻の既定プロンプトが適用されています。';
                        } else {
                            threadGlobalPreviewStatus.textContent = '管理者が設定した全体システムプロンプトが適用されています。';
                        }
                    }
                    if (get('thread-global-sys-prompt')) get('thread-global-sys-prompt').value = d.system_prompt || '';
                    if (get('thread-global-sys-prompt-enabled')) get('thread-global-sys-prompt-enabled').checked = d.system_prompt_enabled !== false;
                    
                    window.ensureThreadAutoSystemPromptCard();
                    if (get('thread-apply-auto-sys-prompt-notices')) get('thread-apply-auto-sys-prompt-notices').checked = d.apply_auto_system_prompt_notices !== false;
                    window.applyAutoSystemPromptConfigToForm('thread', d.auto_system_prompt_notices_config || {});
                }
                if (threadSettingsRes.ok) {
                    const threadSettings = await threadSettingsRes.json();
                    if (modalLoadSeq !== threadModalLoadSeq) return;
                    if (modalThreadId !== targetThreadId) return;
                    const customInstructionEl = get('thread-custom-instruction');
                    if (customInstructionEl) customInstructionEl.value = threadSettings.custom_instruction || '';
                    const includeGlobalEl = get('thread-include-global-instruction');
                    if (includeGlobalEl) includeGlobalEl.checked = threadSettings.include_global_instruction !== false;
                }
            } catch (err) {
                showToast("チャット設定の読み込みに失敗しました", "error", true);
            }
            };
            window.closeThreadModal = (skipHistory = false) => {
            hideModal('thread-modal');
            if (!skipHistory && location.pathname === '/chat-settings') {
                history.back();
            }
            };

        get('save-thread-settings-btn').onclick = async () => {
            const targetId = modalThreadId;
            sendClientDebugLog('info', "Save clicked for thread: " + targetId);
            if (!targetId) return;
            const saveBtn = get('save-thread-settings-btn');
            const originalLabel = saveBtn ? saveBtn.textContent : '';
            if (saveBtn) {
                saveBtn.disabled = true;
                saveBtn.textContent = '保存中...';
            }
            const customInstructionEl = get('thread-custom-instruction');
            const custom_instruction = customInstructionEl ? customInstructionEl.value : '';
            const includeGlobalEl = get('thread-include-global-instruction');
            const include_global_instruction = includeGlobalEl ? includeGlobalEl.checked : true;

            const globalPromptEl = get('thread-global-sys-prompt');
            const globalEnabledEl = get('thread-global-sys-prompt-enabled');
            let userPromptPayload = null;
            try {
                userPromptPayload = (globalPromptEl || globalEnabledEl) ? {
                    system_prompt: globalPromptEl ? globalPromptEl.value : '',
                    system_prompt_enabled: globalEnabledEl ? globalEnabledEl.checked : true,
                    apply_auto_system_prompt_notices: get('thread-apply-auto-sys-prompt-notices') ? get('thread-apply-auto-sys-prompt-notices').checked : true,
                    auto_system_prompt_notices_config: collectAutoSystemPromptConfigFromForm('thread')
                } : null;
            } catch (payloadErr) {
                sendClientDebugLog('error', "Payload construction failed: " + payloadErr.message);
            }
            
            try {
                sendClientDebugLog('info', "Starting PUT request for thread: " + targetId);
                const res = await apiFetch(`/api/threads/${targetId}/settings`, {
                    method: 'PUT',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ custom_instruction, include_global_instruction })
                });
                sendClientDebugLog('info', "PUT request finished, status: " + res.status);
                let userResOk = true;
                if (userPromptPayload) {
                    sendClientDebugLog('info', "Starting POST request for user settings");
                    const userRes = await apiFetch(CHAT_CONFIG.urls.handleSettings, {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(userPromptPayload)
                    });
                    userResOk = userRes.ok;
                    sendClientDebugLog('info', "POST request finished, status: " + userRes.status);
                }
                if (res.ok && userResOk) {
                    window.closeThreadModal();
                    showToast("保存されました", "success");
                } else {
                    showToast("保存に失敗しました", "error", true);
                }
            } catch (err) {
                sendClientDebugLog('error', "Save failed with error: " + err.message);
                showToast("エラー: " + err.message, "error", true);
            } finally {
                if (saveBtn) {
                    saveBtn.disabled = false;
                    saveBtn.textContent = originalLabel || '保存';
                }
            }
        };

        window.openCompressionModal = () => {
            syncCompressionSettingsUi();
            showModal('compression-modal');
            if (location.pathname !== '/compression') {
                history.pushState({ modal: 'compression' }, '', '/compression');
            }
        };
        window.closeCompressionModal = (skipHistory = false) => {
            hideModal('compression-modal');
            if (!skipHistory && location.pathname === '/compression') {
                history.back();
            }
        };

        get('save-compression-settings-btn').onclick = () => {
            const size = get('compression-max-size').value;
            const dim = get('compression-max-dim').value;
            const type = get('compression-output-type').value;
            const formatOnly = get('compression-format-only').checked;
            setCompressionSettings(size, dim, type, formatOnly);

            // Sync back to the prompt bar inputs
            const syncBack = (modalId, targetId) => { if(get(modalId) && get(targetId)) get(targetId).value = get(modalId).value; };
            syncBack('modal-gpt-image-size', 'gpt-image-size');
            syncBack('modal-gpt-image-quality', 'gpt-image-quality');
            syncBack('modal-gpt-image-format', 'gpt-image-format');
            syncBack('modal-gpt-image-compression', 'gpt-image-compression');
            syncBack('modal-gemini-image-aspect', 'gemini-image-aspect');
            syncBack('modal-gemini-image-size', 'gemini-image-size');
            syncBack('modal-grok-image-aspect', 'grok-image-aspect');
            syncBack('modal-grok-image-resolution', 'grok-image-resolution');

            window.closeCompressionModal();
            showToast('設定を保存しました', 'success');
        };
        async function deleteGem(e, id) { e.stopPropagation(); if(!confirm("Delete?")) return; await apiFetch(CHAT_CONFIG.urls.handleGemItem.replace('0', id), {method: 'DELETE'}); loadGems(); }
        async function renameThread(e, id) { e.stopPropagation(); const n = prompt("Title:"); if(n) { const res = await apiFetch(CHAT_CONFIG.urls.updateTitle.replace('0', id), { method: 'PUT', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({title: n}) }); const d = await res.json().catch(() => ({})); if (res.ok && currentThreadId === String(id)) setCurrentChatHeaderTitle((d && d.title) || n); loadThreads(); } }
        async function deleteThread(e, id) { e.stopPropagation(); if(!confirm("Delete?")) return; await apiFetch(CHAT_CONFIG.urls.handleThreadItem.replace('0', id), {method:'DELETE'}); if(currentThreadId === id) startNewChat(); else loadThreads(); }
        async function deleteMessage(id) { if(!confirm("Delete this message and subsequent history?")) return; await apiFetch(CHAT_CONFIG.urls.deleteMessage.replace('0', id), {method:'DELETE'}); loadMessages(currentThreadId); }
        let activePdfPrintFrame = null;
        const PDF_IMAGE_EXTS = new Set(['jpg', 'jpeg', 'png', 'webp', 'gif', 'bmp', 'avif', 'svg']);
        const PDF_PRINT_ROUTE = CHAT_CONFIG.urls.exportThreadPdf;
        const pdfEscapeAttr = (value) => escapeHtml(value == null ? '' : String(value));
        const pdfFormatTimestamp = (value) => {
            if (!value) return '';
            try {
                const date = new Date(value);
                if (Number.isNaN(date.getTime())) return String(value);
                return new Intl.DateTimeFormat('ja-JP', {
                    year: 'numeric',
                    month: '2-digit',
                    day: '2-digit',
                    hour: '2-digit',
                    minute: '2-digit',
                    second: '2-digit'
                }).format(date);
            } catch (e) {
                return String(value);
            }
        };
        const pdfNormalizeAttachmentPath = (path) => {
            if (!path) return '';
            let v = String(path).trim();
            if (!v) return '';
            try {
                if (v.includes('://')) {
                    v = new URL(v, window.location.origin).pathname || '';
                }
            } catch (e) {}
            if (v.includes('?')) v = v.split('?', 1)[0];
            if (v.includes('#')) v = v.split('#', 1)[0];
            v = v.replace(/^\/+/, '');
            if (v.startsWith('files/')) v = v.slice(6);
            try { v = decodeURIComponent(v); } catch (e) {}
            return v;
        };
        const buildPdfAttachmentUrl = (path) => {
            const norm = pdfNormalizeAttachmentPath(path);
            return norm ? `${window.location.origin}/files/${encodeURI(norm)}` : '';
        };
        const buildPdfAttachmentPreviewUrl = (path) => {
            const norm = pdfNormalizeAttachmentPath(path);
            if (!norm) return '';
            return `${window.location.origin}/${PDF_IMAGE_EXTS.has((norm.split('.').pop() || '').toLowerCase()) ? 'files/thumb/' : 'files/'}${encodeURI(norm)}`;
        };
        const buildPdfMessageAttachments = (message) => {
            const attachments = Array.isArray(message && message.attachments) ? message.attachments : [];
            return attachments.map((attachment) => {
                const path = pdfNormalizeAttachmentPath(attachment && attachment.path ? attachment.path : attachment);
                if (!path) return null;
                const filename = attachment && attachment.filename ? attachment.filename : path.split('/').pop();
                const source = attachment && attachment.source ? String(attachment.source) : 'attachment';
                const isImage = !!(attachment && attachment.is_image);
                const url = attachment && attachment.url ? attachment.url : buildPdfAttachmentUrl(path);
                const previewUrl = attachment && attachment.preview_url ? attachment.preview_url : buildPdfAttachmentPreviewUrl(path);
                return { path, filename, source, isImage, url, previewUrl };
            }).filter(Boolean);
        };
        const buildPdfDocumentHtml = (data) => {
            const thread = data && data.thread ? data.thread : {};
            const messages = Array.isArray(data && data.messages) ? data.messages : [];
            const needsMathJax = messages.some(m => maybeNeedsMathJax(m.content) || maybeNeedsMathJax(m.thought_text));
            // Remove inline configuration script to avoid CSP violation
            const mathJaxScript = needsMathJax ? `
        <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js" id="MathJax-script" async data-cfasync="false"></script>` : '';
            const coverTitle = thread.title || 'AI Chat';
            const coverMeta = [
                { label: 'Exported At', value: pdfFormatTimestamp(data && data.generated_at) },
                { label: 'Leaf Message', value: data && data.leaf_id ? `#${data.leaf_id}` : 'none' },
                { label: 'Messages', value: String(messages.length) },
                { label: 'Version', value: `AI Playground ${appVersion}` }
            ];
            const messageHtml = messages.map((message) => {
                const isUser = message.role === 'user';
                const quoteHtml = message.quote_text
                    ? `<div class="quote"><strong>Quote</strong><br>${escapeHtml(message.quote_text)}</div>`
                    : '';
                const thoughtHtml = message.thought_text
                    ? `<div class="thought">${escapeHtml(message.thought_text)}</div>`
                    : '';
                const contentHtml = isUser
                    ? `<div class="content" style="white-space: pre-wrap;">${escapeHtml(message.content || '')}</div>`
                    : `<div class="content">${sanitizeMarkdownHtml(message.content || '')}</div>`;
                const attachments = buildPdfMessageAttachments(message);
                const attachmentsHtml = attachments.length
                    ? `<div class="attachments">${attachments.map((attachment) => {
                        if (attachment.isImage) {
                            return `<div class="attachment"><img src="${pdfEscapeAttr(attachment.previewUrl)}" alt="${pdfEscapeAttr(attachment.filename)}"><div class="file-caption">${pdfEscapeAttr(attachment.filename)}</div></div>`;
                        }
                        return `<div class="attachment"><a class="file" href="${pdfEscapeAttr(attachment.url)}" target="_blank" rel="noreferrer noopener"><span class="file-icon">📄</span><span><span class="file-name">${pdfEscapeAttr(attachment.filename)}</span><span class="file-source">${pdfEscapeAttr(attachment.source)}</span></span></a></div>`;
                    }).join('')}</div>`
                    : '';
                const metaBits = [];
                if (message.model && !isUser) metaBits.push(message.model);
                if (message.tokens !== null && message.tokens !== undefined) metaBits.push(`tokens:${message.tokens}`);
                if (message.tokens_in !== null && message.tokens_in !== undefined) metaBits.push(`in:${message.tokens_in}`);
                if (message.tokens_out !== null && message.tokens_out !== undefined) metaBits.push(`out:${message.tokens_out}`);
                if (message.tokens_thought !== null && message.tokens_thought !== undefined) metaBits.push(`thought:${message.tokens_thought}`);
                if (message.is_encrypted) metaBits.push('encrypted');
                if (message.parent_id !== null && message.parent_id !== undefined) metaBits.push(`parent:#${message.parent_id}`);
                const metaHtml = metaBits.length ? `<div class="message-meta">${pdfEscapeAttr(metaBits.join(' • '))}</div>` : '';
                return `
                    <article class="message ${isUser ? 'user' : 'ai'}">
                        <div class="message-head">
                            <div class="message-role" style="color:${isUser ? 'var(--user)' : 'var(--ai)'}"><span class="dot"></span><span>${isUser ? 'User' : 'Assistant'}</span></div>
                            <div class="message-time">${pdfEscapeAttr(pdfFormatTimestamp(message.timestamp))}</div>
                        </div>
                        <div class="message-body">
                            ${quoteHtml}
                            ${contentHtml}
                            ${thoughtHtml}
                            ${attachmentsHtml}
                            ${metaHtml}
                        </div>
                    </article>
                `;
            }).join('');
            return `
        <!DOCTYPE html>
        <html lang="ja">
        <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>${pdfEscapeAttr(coverTitle)} - PDF Export</title>
        ${mathJaxScript}
        <style>
        :root { --ink:#0f172a; --muted:#475569; --line:#dbe3ee; --panel:#fff; --panel-soft:#f8fafc; --user:#0ea5e9; --ai:#10b981; --accent:#0f766e; }
        * { box-sizing: border-box; }
        html, body { margin: 0; padding: 0; }
        body { font-family: "Noto Sans JP", system-ui, sans-serif; color: var(--ink); background: linear-gradient(180deg, #eef4fb 0%, #f8fbff 45%, #eef2f7 100%); }
        .page { max-width: 980px; margin: 0 auto; padding: 24px 18px 48px; }
        .cover { position: relative; overflow: hidden; border-radius: 26px; padding: 26px 24px; color: #eff6ff; background: linear-gradient(135deg, #0f172a 0%, #0b3b57 56%, #0f766e 100%); box-shadow: 0 24px 48px rgba(15, 23, 42, 0.18); }
        .cover h1 { margin: 0 0 8px; font-size: 40px; line-height: 1.1; }
        .cover p { margin: 0; max-width: 72ch; color: rgba(226, 232, 240, 0.88); font-size: 14px; line-height: 1.8; }
        .meta-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; margin-top: 18px; }
        .meta-card { padding: 12px 14px; border-radius: 16px; background: rgba(15, 23, 42, 0.2); border: 1px solid rgba(255,255,255,0.15); }
        .meta-label { font-size: 11px; letter-spacing: 0.08em; color: rgba(226,232,240,0.68); margin-bottom: 6px; text-transform: uppercase; }
        .meta-value { font-size: 14px; font-weight: 700; word-break: break-word; }
        .message-list { margin-top: 20px; display: flex; flex-direction: column; gap: 16px; }
        .message { border-radius: 22px; border: 1px solid var(--line); background: var(--panel); box-shadow: 0 14px 30px rgba(15, 23, 42, 0.06); overflow: hidden; break-inside: avoid; page-break-inside: avoid; }
        .message.user { border-left: 6px solid var(--user); }
        .message.ai { border-left: 6px solid var(--ai); }
        .message-head { display:flex; gap:10px; justify-content:space-between; align-items:flex-start; padding:14px 18px 0; }
        .message-role { display:inline-flex; align-items:center; gap:8px; font-weight:900; font-size:13px; }
        .message-role .dot { width:10px; height:10px; border-radius:50%; background: currentColor; }
        .message-time { color: var(--muted); font-size: 11px; white-space: nowrap; }
        .message-body { padding: 12px 18px 18px; }
        .quote { margin:0 0 12px; padding:10px 12px; border-left:4px solid rgba(14,165,233,0.7); background: var(--panel-soft); color: var(--muted); border-radius:12px; font-size:12px; line-height:1.7; }
        .thought { margin: 12px 0 0; padding: 12px 14px; border-radius: 14px; background: rgba(139, 92, 246, 0.06); border: 1px solid rgba(139, 92, 246, 0.18); color: #4c1d95; font-size: 12px; line-height: 1.8; white-space: pre-wrap; }
        .content { font-size: 14px; line-height: 1.85; word-break: break-word; }
        .content pre { overflow:auto; padding:12px 14px; border-radius:14px; background:#0b1020; color:#e2e8f0; border:1px solid rgba(15,23,42,0.18); }
        .content code { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }
        .content blockquote { margin:12px 0; padding:8px 12px; border-left:4px solid rgba(14,165,233,0.65); background: rgba(14,165,233,0.06); border-radius:10px; color:#334155; }
        .content img { max-width:100%; height:auto; border-radius:14px; border:1px solid rgba(148,163,184,0.28); margin:10px 0; }
        .attachments { margin-top: 14px; display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }
        .attachment { border-radius: 16px; border: 1px solid var(--line); background: #f8fafc; overflow: hidden; break-inside: avoid; }
        .attachment img { width: 100%; height: auto; display: block; }
        .attachment .file { display:flex; gap:10px; align-items:center; padding:12px 14px; color: var(--ink); text-decoration:none; }
        .file-icon { font-size: 18px; color: var(--accent); }
        .file-name { display:block; font-weight:700; font-size:13px; word-break:break-word; }
        .file-source { display:block; color: var(--muted); font-size: 11px; margin-top: 3px; }
        .file-caption { padding:10px 12px; font-size:11px; color: var(--muted); }
        .message-meta { margin-top: 14px; text-align: right; color: var(--muted); font-size: 11px; line-height: 1.7; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }
        @media print { body { background:#fff; } .page { padding:0; max-width:none; } .cover, .message { box-shadow:none; } .message, .attachment, .meta-card { break-inside: avoid; page-break-inside: avoid; } a { color: inherit; text-decoration: none; } }
        </style>
        </head>
        <body>
        <div class="page">
        <section class="cover">
        <h1>${pdfEscapeAttr(coverTitle)}</h1>
        <p>スレッド ID: ${pdfEscapeAttr(thread.public_id || '')}。表示中の履歴をそのまま印刷できるように、画面キャプチャではなく全メッセージを再構成して出力しています。</p>
        <div class="meta-grid">
        ${coverMeta.map((item) => `<div class="meta-card"><div class="meta-label">${pdfEscapeAttr(item.label)}</div><div class="meta-value">${pdfEscapeAttr(item.value)}</div></div>`).join('')}
        </div>
        </section>
        <main id="pdf-message-list" class="message-list">${messageHtml || '<div class="meta-card" style="margin-top:20px;background:#fff;color:var(--muted);text-align:center;border:1px dashed rgba(148,163,184,0.5);padding:28px;border-radius:20px;">このスレッドにはメッセージがありません。</div>'}</main>
        </div>
        </body>
        </html>`;
        };
        async function openThreadPdfPrintDialog() {
            if (!currentThreadId) {
                showToast("PDF化するスレッドを開いてください", "warning", true);
                return;
            }
            if (activePdfPrintFrame) {
                showToast("PDF出力の準備中です。しばらくお待ちください。", "warning", true);
                return;
            }

            // Set temporary lock to prevent double clicks during fetch
            const lockPlaceholder = { isLock: true };
            activePdfPrintFrame = lockPlaceholder;

            const progress = showProgressToast("PDF出力の準備中です", "info");
            progress.update(5);

            try {
                const url = new URL(PDF_PRINT_ROUTE.replace('0', currentThreadId), window.location.origin);
                if (currentLeafId !== null && currentLeafId !== undefined && String(currentLeafId).trim()) {
                    url.searchParams.set('leaf_id', String(currentLeafId));
                }
                const res = await apiFetch(url.toString(), { headers: { 'Accept': 'application/json' } });
                progress.update(20);
                if (!res.ok) {
                    activePdfPrintFrame = null;
                    if (progress) progress.remove();
                    showToast("PDFデータの取得に失敗しました", "error", true);
                    return;
                }
                const data = await res.json().catch(() => null);
                progress.update(30);
                if (!data) {
                    activePdfPrintFrame = null;
                    if (progress) progress.remove();
                    showToast("PDFデータの解析に失敗しました", "error", true);
                    return;
                }

                const frame = document.createElement('iframe');
                activePdfPrintFrame = frame;
                frame.setAttribute('aria-hidden', 'true');
                frame.style.position = 'fixed';
                frame.style.right = '0';
                frame.style.bottom = '0';
                frame.style.width = '1px';
                frame.style.height = '1px';
                frame.style.opacity = '0';
                frame.style.pointerEvents = 'none';
                frame.style.border = '0';

                let cleanupTimeout = null;
                const cleanup = () => {
                    if (cleanupTimeout) {
                        clearTimeout(cleanupTimeout);
                        cleanupTimeout = null;
                    }
                    if (progress) progress.remove();
                    if (activePdfPrintFrame === frame || activePdfPrintFrame === lockPlaceholder) activePdfPrintFrame = null;
                    try {
                        if (frame.parentNode) frame.parentNode.removeChild(frame);
                    } catch (e) {}
                };

                // Fallback cleanup
                cleanupTimeout = setTimeout(() => {
                    if (activePdfPrintFrame === frame) {
                        console.log("PDF print cleanup fallback triggered");
                        cleanup();
                    }
                }, 60000);

                frame.onload = async () => {
                    try {
                        const doc = frame.contentDocument;
                        const win = frame.contentWindow;
                        if (!doc || !win) {
                            cleanup();
                            showToast("PDF印刷モーダルの準備に失敗しました", "error", true);
                            return;
                        }
                        progress.update(40);

                        // Setup MathJax configuration from parent context to avoid CSP violation
                        const messages = Array.isArray(data && data.messages) ? data.messages : [];
                        const needsMathJax = messages.some(m => maybeNeedsMathJax(m.content) || maybeNeedsMathJax(m.thought_text));
                        if (needsMathJax) {
                            win.MathJax = {
                                tex: {
                                    inlineMath: [['\\(', '\\)']],
                                    displayMath: [['$$', '$$']],
                                    processEscapes: true
                                },
                                options: {
                                    ignoreHtmlClass: 'tex2jax_ignore',
                                    processHtmlClass: 'tex2jax_process'
                                }
                            };
                        }
                        progress.update(50);

                        if (doc.fonts && doc.fonts.ready) {
                            try { await doc.fonts.ready; } catch (e) {}
                        }
                        progress.update(60);

                        const imgs = Array.from(doc.images || []);
                        const imageLoadPromise = Promise.all(imgs.map((img) => {
                            if (img.complete) return Promise.resolve();
                            return new Promise((resolve) => {
                                img.addEventListener('load', resolve, { once: true });
                                img.addEventListener('error', resolve, { once: true });
                            });
                        }));

                        // Wait for images with 5s timeout to avoid hanging
                        await Promise.race([
                            imageLoadPromise,
                            new Promise(resolve => setTimeout(resolve, 5000))
                        ]);
                        progress.update(80);

                        const mathJaxScript = doc.getElementById('MathJax-script');
                        if (mathJaxScript) {
                            let retryCount = 0;
                            while (retryCount < 100 && (!win.MathJax || typeof win.MathJax.typesetPromise !== 'function')) {
                                await new Promise(r => setTimeout(r, 50));
                                retryCount++;
                            }
                            if (win.MathJax && typeof win.MathJax.typesetPromise === 'function') {
                                try {
                                    await win.MathJax.typesetPromise();
                                } catch (e) {
                                    console.error("PDF MathJax typeset failed", e);
                                }
                            }
                        }
                        progress.update(95);

                        setTimeout(() => {
                            try {
                                win.focus();
                                win.addEventListener('afterprint', () => { cleanup(); }, { once: true });
                                progress.update(100);
                                setTimeout(() => { if (progress) progress.remove(); }, 1000);
                                win.print();
                            } catch (e) {
                                cleanup();
                                showToast("PDF印刷モーダルを開けませんでした", "error", true);
                            }
                        }, 100);
                    } catch (e) {
                        cleanup();
                        showToast("PDF印刷モーダルの準備に失敗しました", "error", true);
                    }
                };

                const html = buildPdfDocumentHtml(data);
                const blob = new Blob([html], { type: 'text/html' });
                frame.src = URL.createObjectURL(blob);
                document.body.appendChild(frame);
            } catch (e) {
                if (progress) progress.remove();
                activePdfPrintFrame = null;
                showToast("PDF出力中にエラーが発生しました", "error", true);
            }
        }
        function exportCurrentThreadPdf() {
            openThreadPdfPrintDialog().catch(() => {
                showToast("PDF出力に失敗しました", "error", true);
            });
        }
        window.regenerateMessage = (id) => {
            const msg = allMessages.find(m => m.id == id);
            if (!msg || !msg.parent_id) { showToast("再生成できるメッセージが見つかりません", "error", true); return; }
            beginEditMessage(msg.parent_id, true);
        };
        function getLibSortOrder() {
            const sel = get('lib-sort');
            let v = sel ? sel.value : '';
            if (!v) v = localStorage.getItem(LIB_SORT_KEY) || 'newest';
            if (sel && sel.value !== v) sel.value = v;
            return v || 'newest';
        }
        function sortLibraryFiles(list) {
            const order = getLibSortOrder();
            const files = Array.isArray(list) ? list.slice() : [];
            const collator = new Intl.Collator('ja', { numeric: true, sensitivity: 'base' });
            const nameAsc = (a, b) => collator.compare(a.filename || '', b.filename || '');
            const nameDesc = (a, b) => collator.compare(b.filename || '', a.filename || '');
            const tsDesc = (a, b) => (Number(b.ts) || 0) - (Number(a.ts) || 0);
            const tsAsc = (a, b) => (Number(a.ts) || 0) - (Number(b.ts) || 0);
            if (order === 'name_asc') files.sort((a, b) => nameAsc(a, b) || tsDesc(a, b));
            else if (order === 'name_desc') files.sort((a, b) => nameDesc(a, b) || tsDesc(a, b));
            else if (order === 'oldest') files.sort((a, b) => tsAsc(a, b) || nameAsc(a, b));
            else files.sort((a, b) => tsDesc(a, b) || nameAsc(a, b));
            return files;
        }
        function getLibSearchQuery() {
            const q = lib.searchQuery || (get('lib-search') ? get('lib-search').value : '') || '';
            return String(q).trim().toLocaleLowerCase();
        }
        function fileNameForSearch(item) {
            return String((item && item.filename) || '').toLocaleLowerCase();
        }
        function renderLibraryGrid() {
            const grid = get('lib-grid');
            if (!grid) return;
            grid.innerHTML = '';
            if (!lib.files || !lib.files.length) {
                grid.innerHTML = '<div class="text-xs text-gray-500">ファイルがありません。</div>';
                const countEl = get('lib-total-count');
                if (countEl) countEl.innerText = "0 files";
                return;
            }
            const ordered = sortLibraryFiles(lib.files);
            const q = getLibSearchQuery();
            const filtered = q ? ordered.filter((f) => fileNameForSearch(f).includes(q)) : ordered;
            const countEl = get('lib-total-count');
            if (countEl) {
                if (q) countEl.innerText = `${filtered.length} / ${lib.files.length} files`;
                else countEl.innerText = `${lib.files.length} files`;
            }
            if (!filtered.length) {
                grid.innerHTML = '<div class="text-xs text-gray-500">一致するファイルがありません。</div>';
                return;
            }
            let idx = 0;
            filtered.forEach((f) => {
                try {
                    const el = renderLibraryItem(f, idx++);
                    grid.appendChild(el);
                } catch (e) {
                    // Skip broken entries instead of failing entire render.
                }
            });
        }
        function renderLibraryItem(f, i = 0) {
            const el = document.createElement('div'); 
            el.className = "relative group bg-gray-700 library-thumb-card rounded flex items-center justify-center border border-gray-600 cursor-pointer transition hover:border-gray-400 model-list-animate overflow-hidden"; 
            if (i !== null && i !== undefined) el.style.animationDelay = `${i * 0.03}s`;
            const thumbSrc = f.thumbnail_url || f.thumb_url || f.url;
            const content = f.type==='image' ? `<img src="${escapeHtml(thumbSrc)}" loading="lazy" decoding="async" class="library-thumb-media">` : `<div class="library-thumb-file flex flex-col items-center"><i class="fas fa-file text-2xl mb-1"></i><span class="text-[10px] truncate w-full text-center">${escapeHtml(f.filename)}</span></div>`;
            const overlay = `<div class="lib-overlay absolute inset-0 bg-black/60 hidden group-hover:flex items-center justify-center gap-2 transition rounded z-10"><a href="${escapeHtml(f.url)}" download="${escapeHtml(f.filename)}" class="p-2 bg-gray-700 hover:bg-gray-600 rounded-full text-white" onclick="event.stopPropagation()"><i class="fas fa-download"></i></a><button class="lib-open-btn p-2 bg-gray-700 hover:bg-gray-600 rounded-full text-white" onclick="event.stopPropagation()"><i class="fas fa-eye"></i></button></div>`;
            const actions = `<div class="absolute top-1 right-1 flex gap-1 z-20"><button class="lib-open-btn w-7 h-7 rounded-full bg-gray-900/70 border border-gray-600 text-gray-200 text-[10px]" title="開く"><i class="fas fa-eye"></i></button><button class="lib-del-btn w-7 h-7 rounded-full bg-gray-900/70 border border-gray-600 text-red-300 text-[10px]" title="削除"><i class="fas fa-trash"></i></button></div>`;
            el.innerHTML = content + overlay + actions; 
            el.onclick = () => {
                if (lib.selected.has(f.filepath)) {
                    lib.selected.delete(f.filepath);
                    el.classList.remove('ring-2', 'ring-blue-500', 'border-blue-500');
                    el.classList.add('border-gray-600');
                } else {
                    lib.selected.add(f.filepath);
                    el.classList.add('ring-2', 'ring-blue-500', 'border-blue-500');
                    el.classList.remove('border-gray-600');
                }
                window.updateLibSelectionUi();
            }; 
            if (lib.selected && lib.selected.has(f.filepath)) {
                el.classList.add('ring-2', 'ring-blue-500', 'border-blue-500');
                el.classList.remove('border-gray-600');
            }
            const openBtns = el.querySelectorAll('.lib-open-btn');
            openBtns.forEach((btn) => {
                btn.onclick = (e) => {
                    e.stopPropagation();
                    openFileViewer(f.url, f.filename);
                };
            });
            const delBtn = el.querySelector('.lib-del-btn');
            if (delBtn) {
                delBtn.onclick = async (e) => {
                    e.stopPropagation();
                    await deleteSingleLibraryFile(f.filepath, el);
                };
            }
            return el;
        }
        function addLibraryFileFromPath(filepath) {
            if (!filepath) return;
            if (!lib.fileSet) lib.fileSet = new Set();
            if (lib.fileSet.has(filepath)) return;
            const filename = filepath.split('/').pop() || filepath;
            const ext = (filename.split('.').pop() || '').toLowerCase();
            const type = ['png','jpg','jpeg','webp','gif'].includes(ext) ? 'image' : 'file';
            const url = FILE_BASE_URL + filepath;
            const thumbnail_url = type === 'image' ? (FILE_THUMB_BASE_URL + filepath) : null;
            const f = { filename, original_filename: filename, filepath, url, thumbnail_url, type, ext, ts: Math.floor(Date.now() / 1000) };
            setAttachmentNameForPath(filepath, filename);
            lib.fileSet.add(filepath);
            if (!lib.files) lib.files = [];
            lib.files.unshift(f);
            const grid = get('lib-grid');
            if (grid && lib.modal && lib.modal.classList.contains('modal-open')) {
                renderLibraryGrid();
            }
        }
        async function renameSelectedLibraryFile() {
            if (!lib.selected || lib.selected.size !== 1) return;
            const filepath = Array.from(lib.selected)[0];
            const item = (lib.files || []).find((f) => f.filepath === filepath);
            const currentName = (item && item.filename) || (filepath.split('/').pop() || filepath);
            const nextNameRaw = prompt('新しいファイル名を入力してください', currentName);
            if (nextNameRaw === null) return;
            const nextName = (nextNameRaw || '').trim();
            if (!nextName) {
                showToast('ファイル名を入力してください', 'error', true);
                return;
            }
            try {
                const r = await apiFetch(CHAT_CONFIG.urls.renameLibraryFile, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ filepath, filename: nextName })
                });
                const d = await r.json().catch(() => ({}));
                if (!r.ok) {
                    showToast(d.error || '名前変更に失敗しました', 'error', true);
                    return;
                }
                if (item) {
                    item.filename = d.filename || nextName;
                    setAttachmentNameForPath(filepath, item.filename);
                }
                const uploadList = get('upload-list');
                if (uploadList) {
                    uploadList.querySelectorAll('[data-filename]').forEach((row) => {
                        if (row.getAttribute('data-filename') === filepath) {
                            setRowAttachmentName(row, item ? item.filename : (d.filename || nextName));
                        }
                    });
                }
                renderLibraryGrid();
                window.updateLibSelectionUi();
                showToast('ファイル名を変更しました', 'success');
            } catch (e) {
                showToast('名前変更に失敗しました', 'error', true);
            }
        }
        async function deleteSingleLibraryFile(filepath, el) {
            if (!filepath) return;
            if (!confirm('削除しますか？')) return;
            try {
                await apiFetch(CHAT_CONFIG.urls.deleteFilesBatch, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({filenames:[filepath]})});
                if (el && el.parentNode) el.remove();
                if (lib.files) lib.files = lib.files.filter(f => f.filepath !== filepath);
                if (lib.fileSet) lib.fileSet.delete(filepath);
                lib.selected.delete(filepath);
                renderLibraryGrid();
                window.updateLibSelectionUi();
            } catch(e) {
                showToast("削除に失敗しました", "error", true);
            }
        }
        async function loadLibraryFiles() { 
            const grid = get('lib-grid');
            if (grid) grid.innerHTML = '<div class="text-xs text-gray-500">読み込み中...</div>';
            let files = null;
            let lastErr = null;
            const baseUrl = CHAT_CONFIG.urls.getFilesLib;
            for (let i = 0; i < 2; i++) {
                try {
                    const url = i === 0 ? baseUrl : (baseUrl + (baseUrl.includes('?') ? '&' : '?') + 't=' + Date.now());
                    const r = await apiFetch(url, { cache: 'no-store', headers: { 'Accept': 'application/json' } }); 
                    if (!r.ok) throw new Error('HTTP ' + r.status);
                    const raw = await r.text();
                    let parsed = [];
                    try { parsed = JSON.parse(raw); } catch (e) { parsed = []; }
                    if (Array.isArray(parsed)) {
                        files = parsed;
                        lastErr = null;
                        break;
                    }
                } catch (e) {
                    lastErr = e;
                }
            }
            if (!Array.isArray(files)) files = [];
            try {
                const base = FILE_BASE_URL;
                const thumbBase = FILE_THUMB_BASE_URL;
                const seenPaths = new Set(files.map(f => f && f.filepath).filter(Boolean));
                const extra = Array.isArray(currentImageUrls) ? currentImageUrls : [];
                extra.forEach((fp) => {
                    if (!fp || seenPaths.has(fp)) return;
                    const filename = getAttachmentNameForPath(fp) || (fp.split('/').pop() || fp);
                    const ext = (filename.split('.').pop() || '').toLowerCase();
                    const type = ['png','jpg','jpeg','webp','gif'].includes(ext) ? 'image' : 'file';
                    const thumbUrl = type === 'image' ? (thumbBase + fp) : null;
                    files.unshift({ filename, original_filename: filename, filepath: fp, url: base + fp, thumbnail_url: thumbUrl, type, ext, ts: Math.floor(Date.now() / 1000) });
                    seenPaths.add(fp);
                });
            } catch (e) {}
            try {
                if (grid) grid.innerHTML = '';
                if (!lib.selected) lib.selected = new Set();
                lib.selected.clear(); 
                lib.files = files.filter(f => f && f.filepath && f.url);
                lib.files.forEach((f) => {
                    if (f && f.filepath) setAttachmentNameForPath(f.filepath, f.filename || f.original_filename || '');
                });
                lib.fileSet = new Set(lib.files.map(f => f.filepath));
                window.updateLibSelectionUi();
                renderLibraryGrid();
            } catch (e) {
                lastErr = lastErr || e;
            }
            if (lastErr && grid) {
                console.error('Library load failed:', lastErr);
                grid.innerHTML = '<div class="text-xs text-red-400">ライブラリの読み込みに失敗しました。</div>';
            }
        }
        async function deleteSelectedFiles() {
            if(!confirm('削除しますか？')) return;
            try{
                await apiFetch(CHAT_CONFIG.urls.deleteFilesBatch, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({filenames:Array.from(lib.selected)})});
                loadLibraryFiles();
            } catch(e){
                alert("削除エラー");
            }
        }
        function attachSelectedLibraryFiles() {
            if (!lib.selected.size) return;
            const support = getModelMediaSupport(get('model-select').value);
            let skippedAudio = 0;
            let skippedVideo = 0;
            const selected = Array.from(lib.selected);
            selected.forEach((fp) => {
                const isAudio = isAudioPath(fp);
                const isVideo = isVideoPath(fp);
                if ((isAudio && !support.audio) || (isVideo && !support.video)) {
                    if (isAudio) skippedAudio += 1;
                    if (isVideo) skippedVideo += 1;
                    return;
                }
                const norm = normalizeAttachmentPath(fp);
                if (!norm) return;
                const item = (lib.files || []).find((f) => f && f.filepath === fp);
                if (item && item.filename) {
                    setAttachmentNameForPath(norm, item.filename);
                }
                if (!currentImageUrls.includes(norm)) currentImageUrls.push(norm);
                setAttachmentSourceForPath(norm, 'library');
            });
            syncUploadRowsFromCurrent();
            updateFilePreview();
            lib.selected.clear();
            window.updateLibSelectionUi();
            window.closeLibModal();
            if (skippedAudio || skippedVideo) {
                const parts = [];
                if (skippedAudio) parts.push(`${skippedAudio}件の音声`);
                if (skippedVideo) parts.push(`${skippedVideo}件の動画`);
                showToast(`このモデルは${parts.join('・')}入力に非対応のため除外しました`, "error", true);
            } else {
                showToast("ライブラリから添付しました", "success");
            }
        }
        function downloadSelectedLibraryFiles() {
            if (!lib.selected || !lib.selected.size) return;
            const selected = Array.from(lib.selected);
            selected.forEach((fp) => {
                const item = (lib.files || []).find((f) => f && f.filepath === fp);
                if (item && item.url) {
                    const a = document.createElement('a');
                    a.href = item.url;
                    a.download = item.filename || item.original_filename || (fp.split('/').pop() || 'file');
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                }
            });
            showToast(`${selected.length}件のファイルをダウンロードしました`, "success");
        }
        window.showLegal = async (t) => { 
            const title = t === 'terms' ? '利用規約' : 'プライバシーポリシー';
            get('legal-title').innerText = title;
            showModal('legal-modal'); 
            const res = await apiFetch("/static/legal/" + t + ".md?t=" + Date.now()); 
            if(!res.ok) return; 
            const text = await res.text(); 
            get('legal-content').innerHTML = DOMPurify.sanitize(marked.parse(text)); 
        }
        window.showAlphaInfo = () => {
            if (typeof showModal === 'function') {
                showModal('alpha-info-modal');
                return;
            }
            const el = get('alpha-info-modal');
            if (el) {
                el.classList.remove('hidden');
                el.style.display = 'flex';
            }
        };
        window.refreshLogs = async () => {
            const panel = get('debug-console');
            const content = get('debug-content');
            if (!panel || !content) return;
            content.textContent = '読み込み中...';
            try {
                if (!isAdminUser) {
                    content.textContent = '管理者のみ閲覧できます。';
                    return;
                }
                const res = await apiFetch('/api/debug/log', { cache: 'no-store' });
                if (!res.ok) {
                    content.textContent = `取得に失敗しました (HTTP ${res.status})`;
                    return;
                }
                const text = await res.text();
                content.textContent = text || 'ログがありません。';
                panel.scrollTop = panel.scrollHeight;
            } catch (e) {
                content.textContent = '取得に失敗しました。';
            }
        };
        window.toggleDebug = () => {
            const panel = get('debug-console');
            if (!panel) return;
            if (!isAdminUser) {
                showToast('管理者のみ利用できます', 'error', true);
                return;
            }
            const isVisible = panel.style.display !== 'none' && getComputedStyle(panel).display !== 'none';
            panel.style.display = isVisible ? 'none' : 'block';
            if (!isVisible) refreshLogs();
        };
        window.copyCode = (btn, code) => { 
            const text = decodeURIComponent(code);
            copyToClipboard(text, 
                () => { btn.innerHTML = '<i class="fas fa-check"></i> Copied'; setTimeout(() => btn.innerHTML = '<i class="fas fa-copy"></i> Copy', 2000); },
                (err) => { console.error(err); btn.innerHTML = '<i class="fas fa-times"></i>'; setTimeout(() => btn.innerHTML = '<i class="fas fa-copy"></i> Copy', 2000); }
            );
        };
        window.copyMessage = (id, btn) => { 
            const txt = messageStore[id] || ""; 
            copyToClipboard(txt, 
                () => { btn.innerHTML = '<i class="fas fa-check"></i>'; setTimeout(() => btn.innerHTML = '<i class="fas fa-copy"></i>', 2000); },
                (err) => { console.error(err); btn.innerHTML = '<i class="fas fa-times"></i>'; setTimeout(() => btn.innerHTML = '<i class="fas fa-copy"></i>', 2000); }
            );
        };
        window.toggleThinking = (el) => { const c = el.nextElementSibling; if(c.classList.contains('collapsed')) { c.classList.remove('collapsed'); } else { c.classList.add('collapsed'); } };

        // --- Branch Management System ---
        let selectedBranchNodeId = null;
        let branchLabelNames = {};
        let threadFixedBranchId = null;

        function loadBranchData() {
            if (!currentThreadId) return;
            const names = localStorage.getItem(`branch_names_${currentThreadId}`);
            branchLabelNames = names ? JSON.parse(names) : {};
            threadFixedBranchId = localStorage.getItem(`fixed_branch_${currentThreadId}`);
        }

        function saveBranchData() {
            if (!currentThreadId) return;
            localStorage.setItem(`branch_names_${currentThreadId}`, JSON.stringify(branchLabelNames));
            if (threadFixedBranchId) {
                localStorage.setItem(`fixed_branch_${currentThreadId}`, threadFixedBranchId);
            } else {
                localStorage.removeItem(`fixed_branch_${currentThreadId}`);
            }
        }

        function getCumulativeTokensForNode(nodeId) {
            let total = 0;
            let curr = nodeId;
            const msgMap = {};
            (allMessages || []).forEach(m => msgMap[m.id] = m);
            while (curr && msgMap[curr]) {
                const m = msgMap[curr];
                total += (m.tokens || (Number(m.tokens_in || 0) + Number(m.tokens_out || 0)));
                curr = m.parent_id;
            }
            return total;
        }

        function getPerModelTokensForPath(nodeId) {
            const modelStats = {}; // { modelName: { total, in, out, thought } }
            let curr = nodeId;
            const msgMap = {};
            (allMessages || []).forEach(m => msgMap[m.id] = m);
            
            while (curr && msgMap[curr]) {
                const m = msgMap[curr];
                const model = m.model || 'Unknown';
                if (!modelStats[model]) {
                    modelStats[model] = { total: 0, in: 0, out: 0, thought: 0 };
                }
                const rowTotal = (m.tokens || (Number(m.tokens_in || 0) + Number(m.tokens_out || 0)));
                modelStats[model].total += rowTotal;
                modelStats[model].in += Number(m.tokens_in || 0);
                modelStats[model].out += Number(m.tokens_out || 0);
                modelStats[model].thought += Number(m.tokens_thought || 0);
                curr = m.parent_id;
            }
            return modelStats;
        }

        window.showBranchModal = () => {
            if (!currentThreadId) {
                showToast('チャットを選択してください', 'error');
                return;
            }
            loadBranchData();
            selectedBranchNodeId = null;
            renderBranchTreeVisualization();
            updateBranchDetailPane();
            showModal('branch-modal');
            if (location.pathname !== '/branch') {
                history.pushState({ modal: 'branch' }, '', '/branch');
            }
            const allTotals = buildTokenTotals(allMessages);
            get('branch-total-tokens').innerText = allTotals.tokens_total || 0;
        };
        window.closeBranchModal = (skipHistory = false) => {
            hideModal('branch-modal');
            if (!skipHistory && location.pathname === '/branch') {
                history.back();
            }
        };

        function renderBranchTreeVisualization() {
            const container = get('branch-tree-canvas');
            container.innerHTML = '';
            if (!allMessages || allMessages.length === 0) return;
            const nodes = {};
            const roots = [];
            allMessages.forEach(msg => nodes[msg.id] = { ...msg, children: [] });
            allMessages.forEach(msg => {
                if (msg.parent_id && nodes[msg.parent_id]) {
                    nodes[msg.parent_id].children.push(nodes[msg.id]);
                } else if (!msg.parent_id) {
                    roots.push(nodes[msg.id]);
                }
            });

            function renderNodeRecursive(node) {
                const nodeEl = document.createElement('div');
                nodeEl.className = 'flex flex-col items-center mt-4';
                const item = document.createElement('div');
                const isCurrent = (String(node.id) === String(currentLeafId));
                const isFixed = (node.id === threadFixedBranchId);
                const name = branchLabelNames[node.id] || (node.role === 'user' ? 'User' : 'AI');
                const pathTokens = getCumulativeTokensForNode(node.id);
                
                item.className = `ui-enter-scale px-3 py-2 rounded-lg border cursor-pointer transition-all text-[10px] min-w-[120px] max-w-[180px] text-center relative ${
                    selectedBranchNodeId === node.id ? 'ring-2 ring-purple-500 border-purple-400' : 'border-gray-700 hover:border-gray-500'
                } ${isCurrent ? 'bg-blue-900/40 border-blue-500/50' : 'bg-gray-800'}`;
                
                item.innerHTML = `
                    <div class="font-bold truncate">${name}</div>
                    <div class="text-[9px] text-gray-500 flex justify-between mt-1 gap-2">
                        <span class="truncate">${node.model || '-'}</span>
                        <span class="text-blue-400 font-mono font-bold" title="Cumulative tokens for this path">${pathTokens}</span>
                    </div>
                    ${isFixed ? '<div class="absolute -top-1 -right-1 w-3 h-3 bg-amber-500 rounded-full border border-gray-900 shadow-sm" title="Fixed Branch"></div>' : ''}
                    ${isCurrent ? '<div class="absolute -top-1 -left-1 w-3 h-3 bg-blue-500 rounded-full border border-gray-900 shadow-sm" title="Current Branch"></div>' : ''}
                `;
                item.onclick = (e) => {
                    e.stopPropagation();
                    selectedBranchNodeId = node.id;
                    renderBranchTreeVisualization();
                    updateBranchDetailPane();
                };
                nodeEl.appendChild(item);
                if (node.children.length > 0) {
                    const connector = document.createElement('div');
                    connector.className = 'w-px h-4 bg-gray-700';
                    nodeEl.appendChild(connector);
                    const childrenContainer = document.createElement('div');
                    childrenContainer.className = 'flex gap-4 items-start';
                    node.children.forEach(child => childrenContainer.appendChild(renderNodeRecursive(child)));
                    nodeEl.appendChild(childrenContainer);
                }
                return nodeEl;
            }
            roots.forEach(root => container.appendChild(renderNodeRecursive(root)));
        }

        function updateBranchDetailPane() {
            const detailPanel = get('branch-detail-panel');
            const emptyPanel = get('branch-empty-panel');
            if (!selectedBranchNodeId || !allMessages) {
                detailPanel.classList.add('hidden');
                emptyPanel.classList.remove('hidden');
                return;
            }
            const node = allMessages.find(m => m.id === selectedBranchNodeId);
            if (!node) return;
            detailPanel.classList.remove('hidden');
            emptyPanel.classList.add('hidden');
            get('br-id').innerText = node.id;
            get('br-date').innerText = node.created_at || '-';
            get('br-model').innerText = node.model || '-';
            const nodeTokens = (node.tokens || (Number(node.tokens_in || 0) + Number(node.tokens_out || 0)));
            const pathTokens = getCumulativeTokensForNode(node.id);
            get('br-tokens').innerHTML = `<span title="Current message tokens">${nodeTokens}</span> <span class="text-gray-500">/</span> <span class="text-purple-400 font-bold" title="Path total tokens">${pathTokens} total</span>`;
            
            // Render Per-Model Breakdown
            const breakdownContainer = get('branch-model-breakdown');
            const modelStats = getPerModelTokensForPath(node.id);
            breakdownContainer.innerHTML = '';
            
            Object.entries(modelStats).sort((a, b) => b[1].total - a[1].total).forEach(([model, stats]) => {
                const div = document.createElement('div');
                div.className = 'bg-gray-800/50 p-2 rounded border border-gray-700/50';
                div.innerHTML = `
                    <div class="flex justify-between font-bold text-gray-300 mb-1">
                        <span class="truncate pr-2">${model}</span>
                        <span class="text-blue-400 shrink-0">${stats.total}</span>
                    </div>
                    <div class="grid grid-cols-3 gap-1 text-[9px] text-gray-500 font-mono">
                        <div title="Input tokens">In: ${stats.in}</div>
                        <div title="Output tokens">Out: ${stats.out}</div>
                        <div title="Thought/Reasoning tokens">${stats.thought > 0 ? `Th: ${stats.thought}` : ''}</div>
                    </div>
                `;
                breakdownContainer.appendChild(div);
            });

            get('br-name-input').value = branchLabelNames[node.id] || '';
            const fixBtn = get('br-fix-btn');
            if (selectedBranchNodeId === threadFixedBranchId) {
                fixBtn.innerText = '固定を解除';
                fixBtn.classList.replace('bg-amber-600', 'bg-gray-600');
            } else {
                fixBtn.innerText = 'メインルートに固定';
                fixBtn.classList.replace('bg-gray-600', 'bg-amber-600');
            }
        }

        // Branch UI Handlers
        if (get('branch-manage-btn')) get('branch-manage-btn').onclick = showBranchModal;
        get('br-save-name-btn').onclick = () => {
            if (!selectedBranchNodeId) return;
            const name = get('br-name-input').value.trim();
            if (name) branchLabelNames[selectedBranchNodeId] = name; else delete branchLabelNames[selectedBranchNodeId];
            saveBranchData(); renderBranchTreeVisualization(); showToast('名前を保存しました');
        };
        get('br-switch-btn').onclick = () => {
            if (!selectedBranchNodeId) return;
            switchVersion(selectedBranchNodeId); window.closeBranchModal(); showToast('ブランチを切り替えました');
        };
        get('br-fix-btn').onclick = () => {
            if (!selectedBranchNodeId) return;
            if (threadFixedBranchId === selectedBranchNodeId) { threadFixedBranchId = null; showToast('固定を解除しました'); } 
            else { threadFixedBranchId = selectedBranchNodeId; showToast('メインルートに固定しました'); }
            saveBranchData(); renderBranchTreeVisualization(); updateBranchDetailPane();
        };
        get('br-delete-btn').onclick = () => {
            if (!selectedBranchNodeId) return;
            if (!confirm('このブランチを削除してもよろしいですか？（その後の全てのメッセージも削除されます）')) return;
            deleteMessage(selectedBranchNodeId);
            selectedBranchNodeId = null;
            setTimeout(() => { renderBranchTreeVisualization(); updateBranchDetailPane(); }, 500);
        };

        // --- Extended Client-Side Debug Logging System ---
        (function() {
            const originalLog = console.log;
            const originalError = console.error;
            const originalWarn = console.warn;
            const originalInfo = console.info;
            let isSending = false;

            async function sendToServer(level, args) {
                if (isSending) return;
                if (!isClientDebugLogEnabled()) return;

                isSending = true;
                const message = args.map(arg => {
                    try {
                        if (arg instanceof Error) return arg.stack || arg.message;
                        return typeof arg === 'object' ? JSON.stringify(arg) : String(arg);
                    } catch (e) {
                        return "[Unserializable Object]";
                    }
                }).join(' ');

                try {
                    sendClientDebugLog(level, message);
                } catch (e) {
                    // fallthrough
                } finally {
                    isSending = false;
                }
            }

            console.log = function(...args) {
                originalLog.apply(console, args);
                sendToServer('log', args);
            };
            console.error = function(...args) {
                originalError.apply(console, args);
                sendToServer('error', args);
            };
            console.warn = function(...args) {
                originalWarn.apply(console, args);
                sendToServer('warn', args);
            };
            console.info = function(...args) {
                originalInfo.apply(console, args);
                sendToServer('info', args);
            };

            window.addEventListener('error', function(event) {
                sendToServer('exception', [event.message, event.filename, event.lineno, event.colno, event.error]);
            });
            window.addEventListener('unhandledrejection', function(event) {
                sendToServer('promise-rejection', [event.reason]);
            });
            
            // Trigger initial log to confirm system is active
            setTimeout(() => {
                console.log("Extended debug logging system active. Version: v4.8.506");
            }, 3000);
        })();
