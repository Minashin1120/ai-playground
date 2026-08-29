        const get = (id) => document.getElementById(id);
        const nativeConsoleLog = (typeof console.log === 'function') ? console.log.bind(console) : function () {};
        const nativeConsoleInfo = (typeof console.info === 'function') ? console.info.bind(console) : nativeConsoleLog;
        // Set to true only after the settings modal has fetched the saved values
        // and populated every control.  The save button stays disabled until then
        // so a click during the load window cannot overwrite settings with the
        // form's default/unpopulated values (which used to silently toggle E2EE).
        let settingsModalLoaded = false;
        const setSettingsSaveEnabled = (enabled) => {
            const saveBtn = get('save-settings-btn');
            if (!saveBtn) return;
            saveBtn.disabled = !enabled;
            saveBtn.classList.toggle('opacity-60', !enabled);
            saveBtn.classList.toggle('cursor-not-allowed', !enabled);
            saveBtn.setAttribute('title', enabled ? '' : '設定の読み込み完了後に保存できます');
        };
        // --- File/image load fallback ----------------------------------------
        // Uploaded files are served under /files/ (and /files/thumb/).  When the
        // media bytes exist on disk but the encryption key no longer matches, the
        // server answers HTTP 409 (not 404).  A document-level capture listener
        // turns a failed image into an explicit warning instead of a broken
        // thumbnail.  A status of 404/410/403 means the file is genuinely gone;
        // a status of 200/206/304 means the media is actually served (a transient
        // load failure or a thumbnail the browser cannot decode), so we retry the
        // load (with a cache-buster) and, for a thumbnail URL, fall back to the
        // original full /files/ image that still opens.  Non-/files/ (e.g. blob)
        // images keep their own handlers.
        (function () {
            const MAX_FILE_RETRIES = 2;
            const isFileUrl = (u) => /(\/files\/thumb\/|\/files\/)/.test(String(u || ''));
            const fileUrlStatus = (url) => fetch(url, { method: 'GET', headers: { 'Range': 'bytes=0-0' }, cache: 'no-store' })
                .then((r) => r.status).catch(() => -1);
            const buildWarning = (filename, keyMismatch) => {
                const box = document.createElement('div');
                box.style.cssText = 'display:flex;flex-direction:column;align-items:center;justify-content:center;width:100%;height:100%;min-height:80px;text-align:center;padding:8px;gap:4px;';
                if (keyMismatch) {
                    box.innerHTML = '<i class="fas fa-key" style="font-size:16px;color:#fbbf24"></i><div style="font-size:9px;color:#fcd34d;font-weight:700;line-height:1.3">暗号キーが一致しないため<br>閲覧できません</div>';
                } else {
                    box.innerHTML = '<i class="fas fa-file" style="font-size:16px;color:#6b7280"></i><div style="font-size:9px;color:#9ca3af;font-weight:700">ファイルがありません</div>';
                }
                if (filename) box.setAttribute('data-file-name', String(filename));
                return box;
            };
            const fullFileUrl = (u) => String(u || '').split('?')[0].replace('/files/thumb/', '/files/');
            document.addEventListener('error', (e) => {
                const el = e.target;
                if (!el || el.tagName !== 'IMG') return;
                const src = el.currentSrc || el.src || '';
                if (!isFileUrl(src)) return;
                e.stopImmediatePropagation();
                e.preventDefault();
                const cleanSrc = String(src).split('?')[0];
                const filename = el.getAttribute('data-viewer-filename') || cleanSrc.split('/').pop();
                const showWarning = (keyMismatch) => {
                    const replacement = buildWarning(filename, !!keyMismatch);
                    try { el.replaceWith(replacement); } catch (_) { /* detached */ }
                };
                const retryLoad = (url, retryCount) => {
                    const fresh = el.cloneNode(false);
                    fresh.setAttribute('data-file-retry', String(retryCount));
                    const busted = url + (url.includes('?') ? '&' : '?') + 'retry=' + Date.now() + '_' + retryCount;
                    fresh.setAttribute('src', busted);
                    try { el.replaceWith(fresh); } catch (_) { /* detached */ }
                };
                const handleStatus = (status) => {
                    if (status === 409) { showWarning(true); return; }
                    if (status === 404 || status === 410 || status === 403) { showWarning(false); return; }
                    // The file is still served, or the failure is transient.  Retry
                    // the load before declaring it missing so a viewable file is not
                    // replaced by "ファイルがありません".
                    const retryCount = parseInt((el.getAttribute && el.getAttribute('data-file-retry')) || '0', 10);
                    if (retryCount < MAX_FILE_RETRIES) { retryLoad(src, retryCount + 1); return; }
                    // A thumbnail may fail to decode while the original full file
                    // renders fine; fall back to /files/ once before giving up.
                    if (src.includes('/files/thumb/') && !el.getAttribute('data-file-fallback')) {
                        el.setAttribute('data-file-fallback', '1');
                        retryLoad(fullFileUrl(src), 0);
                        return;
                    }
                    showWarning(false);
                };
                fileUrlStatus(src).then(handleStatus).catch(() => {
                    // Network error: treat as transient, retry.
                    const retryCount = parseInt((el.getAttribute && el.getAttribute('data-file-retry')) || '0', 10);
                    if (retryCount < MAX_FILE_RETRIES) { retryLoad(src, retryCount + 1); return; }
                    if (src.includes('/files/thumb/') && !el.getAttribute('data-file-fallback')) {
                        el.setAttribute('data-file-fallback', '1');
                        retryLoad(fullFileUrl(src), 0);
                        return;
                    }
                    showWarning(false);
                });
            }, true);
        })();
        const isAdminSidebarDebugEnabled = () => {
            try {
                const cfg = window.CHAT_CONFIG || {};
                return !!(cfg.botConfig && cfg.botConfig.isAdmin);
            } catch (error) {
                return false;
            }
        };
        const ADMIN_SIDEBAR_DEBUG_PREFIX = '[admin-sidebar]';
        const adminSidebarDebugEntries = [];
        const snapshotSidebarHistory = (reason) => {
            if (!isAdminSidebarDebugEnabled()) return null;
            const list = get('thread-list');
            const sidebar = get('sidebar');
            const settings = get('settings-modal');
            const historyModal = get('history-modal');
            const listCs = list ? window.getComputedStyle(list) : null;
            const sidebarCs = sidebar ? window.getComputedStyle(sidebar) : null;
            const items = list ? Array.from(list.querySelectorAll('[data-thread-id]')) : [];
            const first = items[0] || null;
            const firstCs = first ? window.getComputedStyle(first) : null;
            let threadLoadingState = null;
            try { threadLoadingState = typeof threadLoading === 'boolean' ? threadLoading : null; } catch (error) { threadLoadingState = null; }
            const snap = {
                t: Date.now(),
                reason: String(reason || ''),
                path: location.pathname,
                vw: window.innerWidth,
                liteHtml: document.documentElement.classList.contains('performance-lite-mode'),
                blurHtml: document.documentElement.classList.contains('performance-blur-disabled'),
                liquidBody: !!(document.body && document.body.classList.contains('liquid-glass-mode')),
                blurMode: adaptiveBlurPreferenceMode,
                liteEnabled: adaptiveBlurLiteEnabled,
                sidebarClass: sidebar ? sidebar.className : null,
                sidebarDisplay: sidebarCs ? sidebarCs.display : null,
                sidebarOpacity: sidebarCs ? sidebarCs.opacity : null,
                sidebarVisibility: sidebarCs ? sidebarCs.visibility : null,
                compact: !!(sidebar && sidebar.classList.contains('compact')),
                sidebarOpen: !!(sidebar && sidebar.classList.contains('open')),
                listExists: !!list,
                listParent: list && list.parentElement ? (list.parentElement.id || list.parentElement.className) : null,
                listClass: list ? list.className : null,
                listChildCount: list ? list.children.length : 0,
                listItemCount: items.length,
                listDisplay: listCs ? listCs.display : null,
                listOpacity: listCs ? listCs.opacity : null,
                listVisibility: listCs ? listCs.visibility : null,
                listHeight: listCs ? listCs.height : null,
                hideCompact: !!(list && list.classList.contains('hide-compact')),
                searchLen: (() => { const el = get('search-box'); return el ? String(el.value || '').length : 0; })(),
                firstItemText: first && first.textContent ? first.textContent.trim().slice(0, 40) : null,
                firstItemOpacity: firstCs ? firstCs.opacity : null,
                firstItemDisplay: firstCs ? firstCs.display : null,
                firstItemVisibility: firstCs ? firstCs.visibility : null,
                firstItemClass: first ? first.className : null,
                settingsHidden: settings ? settings.classList.contains('hidden') : null,
                settingsOpen: settings ? settings.classList.contains('modal-open') : null,
                settingsDisplay: settings ? (settings.style.display || null) : null,
                historyHidden: historyModal ? historyModal.classList.contains('hidden') : null,
                threadLoading: threadLoadingState
            };
            adminSidebarDebugEntries.push(snap);
            if (adminSidebarDebugEntries.length > 80) adminSidebarDebugEntries.shift();
            try { nativeConsoleLog(ADMIN_SIDEBAR_DEBUG_PREFIX, reason, snap); } catch (error) {}
            return snap;
        };
        const installAdminSidebarDebugObserver = () => {
            if (!isAdminSidebarDebugEnabled()) return;
            const list = get('thread-list');
            if (!list || list.dataset.adminSidebarDebugObserved === '1') return;
            list.dataset.adminSidebarDebugObserved = '1';
            try {
                const observer = new MutationObserver((mutations) => {
                    const removedItems = mutations.reduce((sum, mutation) => {
                        return sum + Array.from(mutation.removedNodes || []).filter((node) => {
                            return node && node.nodeType === 1 && node.getAttribute && node.getAttribute('data-thread-id');
                        }).length;
                    }, 0);
                    const addedItems = mutations.reduce((sum, mutation) => {
                        return sum + Array.from(mutation.addedNodes || []).filter((node) => {
                            return node && node.nodeType === 1 && node.getAttribute && node.getAttribute('data-thread-id');
                        }).length;
                    }, 0);
                    snapshotSidebarHistory(`thread-list-mutated added=${addedItems} removed=${removedItems}`);
                });
                observer.observe(list, { childList: true, attributes: true, attributeFilter: ['class', 'style'] });
            } catch (error) {}
        };
        window.__adminSidebarDebugDump = () => {
            if (!isAdminSidebarDebugEnabled()) return [];
            const copy = adminSidebarDebugEntries.slice();
            try { nativeConsoleLog(ADMIN_SIDEBAR_DEBUG_PREFIX, 'dump', copy); } catch (error) {}
            return copy;
        };
        window.copyAdminSidebarDebug = async () => {
            if (!isAdminSidebarDebugEnabled()) return false;
            const text = JSON.stringify(adminSidebarDebugEntries, null, 2);
            try {
                if (navigator.clipboard && navigator.clipboard.writeText) {
                    await navigator.clipboard.writeText(text);
                }
                nativeConsoleLog(ADMIN_SIDEBAR_DEBUG_PREFIX, 'copied', adminSidebarDebugEntries.length, 'entries');
                return true;
            } catch (error) {
                try { nativeConsoleLog(ADMIN_SIDEBAR_DEBUG_PREFIX, 'copy-failed', text); } catch (logError) {}
                return false;
            }
        };
        const ADAPTIVE_BLUR_COOKIE = 'adaptive_blur_disabled';
        const ADAPTIVE_LITE_COOKIE = 'adaptive_lite_mode';
        const ADAPTIVE_BLUR_MODE_COOKIE = 'adaptive_blur_mode';
        const readCookieValue = (cookieName) => {
            try {
                const match = document.cookie.split(';').map((part) => part.trim()).find((part) => part.startsWith(`${cookieName}=`));
                return match ? decodeURIComponent(match.slice(cookieName.length + 1)) : '';
            } catch (error) {
                return '';
            }
        };
        const normalizeAdaptiveBlurMode = (mode) => ['enabled', 'disabled', 'lite'].includes(mode) ? mode : 'auto';
        const writeAdaptiveBlurCookie = (cookieName, value, maxAge = 31536000) => {
            try {
                const secure = window.location.protocol === 'https:' ? '; Secure' : '';
                document.cookie = `${cookieName}=${encodeURIComponent(value)}; Path=/; Max-Age=${maxAge}; SameSite=Lax${secure}`;
            } catch (error) {
                // Cookie access can be blocked by browser policy; the visual mode still applies.
            }
        };
        const adaptiveBlurInteractionCooldownMs = 3000;
        let adaptiveBlurPreferenceMode = normalizeAdaptiveBlurMode(readCookieValue(ADAPTIVE_BLUR_MODE_COOKIE));
        let adaptiveBlurMeasurementActive = false;
        let adaptiveBlurMeasurementLastAt = 0;
        let adaptiveBlurFallbackEnabled = document.documentElement.classList.contains('performance-blur-disabled');
        let adaptiveBlurLiteEnabled = document.documentElement.classList.contains('performance-lite-mode');
        const syncAdaptiveBlurSettingsUi = () => {
            const select = get('set-background-blur-mode');
            const status = get('background-blur-mode-status');
            if (select) select.value = adaptiveBlurPreferenceMode;
            if (!status) return;
            if (adaptiveBlurPreferenceMode === 'lite') {
                status.textContent = '手動設定により、現在は最小負荷の軽量表示を適用しています。';
            } else if (adaptiveBlurPreferenceMode === 'enabled') {
                status.textContent = '手動設定により、背景ぼかしを常に有効にしています。';
            } else if (adaptiveBlurPreferenceMode === 'disabled') {
                status.textContent = '手動設定により、背景ぼかしを無効にしています。';
            } else if (adaptiveBlurLiteEnabled) {
                status.textContent = '自動判定で負荷が非常に高いため、現在は最小負荷の軽量表示を適用しています。';
            } else if (adaptiveBlurFallbackEnabled) {
                status.textContent = '自動判定で描画負荷を検出したため、現在は背景ぼかしを無効にしています。';
            } else {
                status.textContent = '現在は背景ぼかしが有効です。操作時の描画が重い場合は自動で無効化します。';
            }
        };
        const enableAdaptiveBlurFallback = () => {
            if (adaptiveBlurPreferenceMode !== 'auto' || adaptiveBlurFallbackEnabled) return;
            adaptiveBlurFallbackEnabled = true;
            document.documentElement.classList.add('performance-blur-disabled');
            writeAdaptiveBlurCookie(ADAPTIVE_BLUR_COOKIE, '1');
            syncAdaptiveBlurSettingsUi();
        };
        const enableAdaptiveBlurLite = () => {
            if (adaptiveBlurPreferenceMode !== 'auto' || adaptiveBlurLiteEnabled) return;
            adaptiveBlurLiteEnabled = true;
            if (!adaptiveBlurFallbackEnabled) {
                adaptiveBlurFallbackEnabled = true;
                document.documentElement.classList.add('performance-blur-disabled');
                writeAdaptiveBlurCookie(ADAPTIVE_BLUR_COOKIE, '1');
            }
            document.documentElement.classList.add('performance-lite-mode');
            revealPersistentSidebarLists();
            snapshotSidebarHistory('lite-auto-enabled');
            syncAdaptiveBlurSettingsUi();
            showToast('描画負荷が高いため、軽量表示（最小負荷）を自動適用しました。タップで設定を開く', 'info', false, openAdaptiveBlurSettingsFromToast);
            writeAdaptiveBlurCookie(ADAPTIVE_LITE_COOKIE, '1');
        };
        const openAdaptiveBlurSettingsFromToast = () => {
            if (typeof window.openSettingsModal === 'function') {
                window.openSettingsModal();
            }
            const select = get('set-background-blur-mode');
            const tab = get('tab-display') || get('tab-general');
            if (!select || !tab) return;
            for (const child of tab.children) {
                if (child.contains(select)) {
                    jumpToSetting(tab.id === 'tab-display' ? 'display' : 'general', child);
                    return;
                }
            }
        };
        const applyAdaptiveBlurPreference = (mode) => {
            const normalizedMode = normalizeAdaptiveBlurMode(mode);
            if (normalizedMode === adaptiveBlurPreferenceMode) return;
            adaptiveBlurPreferenceMode = normalizedMode;
            adaptiveBlurMeasurementActive = false;
            adaptiveBlurLiteEnabled = false;
            writeAdaptiveBlurCookie(ADAPTIVE_BLUR_COOKIE, '', 0);
            writeAdaptiveBlurCookie(ADAPTIVE_LITE_COOKIE, '', 0);
            if (normalizedMode === 'auto') {
                writeAdaptiveBlurCookie(ADAPTIVE_BLUR_MODE_COOKIE, '', 0);
            } else {
                writeAdaptiveBlurCookie(ADAPTIVE_BLUR_MODE_COOKIE, normalizedMode);
            }
            adaptiveBlurFallbackEnabled = normalizedMode === 'disabled' || normalizedMode === 'lite';
            adaptiveBlurLiteEnabled = normalizedMode === 'lite';
            document.documentElement.classList.toggle('performance-blur-disabled', adaptiveBlurFallbackEnabled);
            document.documentElement.classList.toggle('performance-lite-mode', adaptiveBlurLiteEnabled);
            revealPersistentSidebarLists();
            snapshotSidebarHistory('blur-preference-applied:' + normalizedMode);
            syncAdaptiveBlurSettingsUi();
        };
        const isSettingsModalOpen = () => {
            const el = get('settings-modal');
            if (!el) return false;
            if (el.classList.contains('modal-open') || el.classList.contains('modal-prep')) return true;
            if (el.classList.contains('hidden')) return false;
            return el.style.display && el.style.display !== 'none';
        };
        const restoreThreadSearchValue = (value, reason) => {
            const el = get('search-box');
            if (!el) return;
            if (el.value === value) return;
            el.value = value;
            clearTimeout(searchTimeout);
            snapshotSidebarHistory(reason || 'restored-search-box');
        };
        const THREAD_SEARCH_INPUT_IDS = ['search-box', 'history-search-box'];
        const isUserInitiatedSearchInput = (event) => !!(event && event.inputType);
        const unlockThreadSearchInput = (el) => {
            if (!el) return;
            if (el.hasAttribute('readonly')) el.removeAttribute('readonly');
        };
        const markThreadSearchUserEdited = (el) => {
            if (!el) return;
            el.dataset.userEdited = '1';
        };
        const discardAutofilledThreadSearch = (reason) => {
            const el = get('search-box');
            if (!el || el.dataset.userEdited) return;
            if (!el.value) return;
            restoreThreadSearchValue('', reason || 'cleared-autofill-search-box');
            const historyEl = get('history-search-box');
            if (historyEl && !historyEl.dataset.userEdited) historyEl.value = '';
        };
        const hardenThreadSearchInputs = () => {
            THREAD_SEARCH_INPUT_IDS.forEach((id) => {
                const el = get(id);
                if (!el) return;
                const unlock = () => unlockThreadSearchInput(el);
                el.addEventListener('pointerdown', unlock);
                el.addEventListener('touchstart', unlock, { passive: true });
                el.addEventListener('keydown', unlock);
                el.addEventListener('focus', unlock);
            });
            discardAutofilledThreadSearch('cleared-autofill-search-box-init');
            [0, 50, 250, 1000].forEach((ms) => {
                setTimeout(() => discardAutofilledThreadSearch('cleared-autofill-search-box-' + ms + 'ms'), ms);
            });
        };
        const revealPersistentSidebarLists = () => {
            document.querySelectorAll('#thread-list > [data-thread-id], #gem-list > .gem-item').forEach((el) => {
                el.classList.remove('model-list-animate', 'slide-in-animate', 'fade-in', 'opacity-0');
                el.style.removeProperty('opacity');
                el.style.removeProperty('transform');
                el.style.removeProperty('animation');
                el.style.removeProperty('animation-delay');
                el.style.removeProperty('visibility');
            });
            ['thread-list', 'gem-list'].forEach((id) => {
                const list = get(id);
                if (!list) return;
                list.style.removeProperty('opacity');
                list.style.removeProperty('visibility');
            });
            snapshotSidebarHistory('reveal-sidebar-lists');
        };
        const adaptiveBlurIsBusy = () => {
            if (activeStreamingBubbleId) return true;
            if (document.querySelector('.modal-overlay.modal-open, .modal-overlay.modal-prep, .modal-overlay.modal-close')) return true;
            return false;
        };
        const measureInteractionFrames = (force = false) => {
            if (adaptiveBlurPreferenceMode !== 'auto' || adaptiveBlurLiteEnabled || adaptiveBlurMeasurementActive || document.visibilityState !== 'visible') return;
            if (!force) {
                const now = Date.now();
                if (now - adaptiveBlurMeasurementLastAt < adaptiveBlurInteractionCooldownMs) return;
                if (adaptiveBlurIsBusy()) return;
                adaptiveBlurMeasurementLastAt = now;
            } else {
                adaptiveBlurMeasurementLastAt = Date.now();
            }
            adaptiveBlurMeasurementActive = true;
            const frameIntervals = [];
            let previousTimestamp = 0;
            const sampleFrame = (timestamp) => {
                if (document.visibilityState !== 'visible') {
                    adaptiveBlurMeasurementActive = false;
                    return;
                }
                if (previousTimestamp) {
                    const interval = timestamp - previousTimestamp;
                    // Ignore long pauses caused by debugging, app switching, or OS suspension.
                    if (interval <= 200) frameIntervals.push(interval);
                }
                previousTimestamp = timestamp;
                if (frameIntervals.length < 30) {
                    requestAnimationFrame(sampleFrame);
                    return;
                }
                adaptiveBlurMeasurementActive = false;
                const sorted = [...frameIntervals].sort((a, b) => a - b);
                const baseline = Math.min(17.5, Math.max(7, sorted[Math.floor(sorted.length * 0.2)]));
                const droppedFrameLimit = Math.max(28, baseline * 1.75);
                const severeFrameLimit = Math.max(44, baseline * 2.7);
                const droppedFrames = frameIntervals.filter((interval) => interval >= droppedFrameLimit).length;
                const severeFrames = frameIntervals.filter((interval) => interval >= severeFrameLimit).length;
                if (droppedFrames >= 5 || (droppedFrames >= 4 && severeFrames >= 2)) {
                    // Tier 1: disable the standard backdrop blur. Once that
                    // fallback is already active and frames are still dropped,
                    // the device is severely underpowered, so apply lite mode
                    // and strip every remaining compositor-heavy effect.
                    if (adaptiveBlurFallbackEnabled) {
                        enableAdaptiveBlurLite();
                    } else {
                        enableAdaptiveBlurFallback();
                    }
                }
            };
            requestAnimationFrame(sampleFrame);
        };
        const measureAdaptiveBlurAfterInteraction = () => {
            if (document.readyState !== 'complete' || adaptiveBlurLiteEnabled) return;
            // Wait one frame so click handlers can open a modal first.
            // Measuring during the settings overlay animation would otherwise
            // look like a dropped-frame burst and auto-enable lite mode.
            requestAnimationFrame(() => {
                if (adaptiveBlurLiteEnabled) return;
                measureInteractionFrames();
            });
        };
        document.addEventListener('click', (event) => {
            const target = event.target instanceof Element ? event.target : null;
            if (!target) return;
            if (target.closest('button, a, input, select, textarea, [role="button"], [tabindex]')) measureAdaptiveBlurAfterInteraction();
        }, true);
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
            pending.catch(() => externalScriptLoads.delete(src));
            return pending;
        };
        const ensurePdfLibraries = () => Promise.all([
            loadExternalScript(
                '/static/vendor/html2canvas-pro-2.3.2.min.js',
                () => typeof window.html2canvas === 'function'
            ),
            loadExternalScript(
                '/static/vendor/jspdf-2.5.1.umd.min.js',
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
        const IMAGE_EXTENSION_BY_MIME = {
            'image/jpeg': '.jpg',
            'image/png': '.png',
            'image/webp': '.webp'
        };
        const imageFilenameForMime = (filename, mimeType) => {
            const ext = IMAGE_EXTENSION_BY_MIME[String(mimeType || '').toLowerCase()];
            if (!ext) return filename || 'image';
            const raw = String(filename || 'image');
            const stem = raw.replace(/\.[^./\\]+$/, '') || 'image';
            return `${stem}${ext}`;
        };
        const convertImageFormatOnly = async (file, outputType) => {
            if (!file || !outputType || outputType === 'original' || outputType === file.type) return file;
            await ensureImageCompression();
            const drawn = await window.imageCompression.drawFileInCanvas(file, { fileType: outputType });
            const source = drawn && drawn[0];
            const canvas = drawn && drawn[1];
            if (!canvas) throw new Error('Image conversion canvas is unavailable');
            let blob;
            try {
                if (typeof canvas.convertToBlob === 'function') {
                    blob = await canvas.convertToBlob({ type: outputType, quality: 1 });
                } else {
                    blob = await new Promise((resolve, reject) => {
                        canvas.toBlob((value) => value ? resolve(value) : reject(new Error('Image conversion failed')), outputType, 1);
                    });
                }
            } finally {
                try { window.imageCompression.cleanupCanvasMemory(canvas); } catch (e) {}
                try { if (source && typeof source.close === 'function') source.close(); } catch (e) {}
            }
            return new File(
                [blob],
                imageFilenameForMime(file.name, outputType),
                { type: outputType, lastModified: file.lastModified || Date.now() }
            );
        };
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
            sync('grok-image-quality', 'modal-grok-image-quality');
            sync('ocr-table-format', 'modal-ocr-table-format');
            sync('ocr-pages', 'modal-ocr-pages');
            const syncChk = (srcId, destId) => {
                if (get(srcId) && get(destId)) get(destId).checked = get(srcId).checked;
            };
            syncChk('ocr-extract-header', 'modal-ocr-extract-header');
            syncChk('ocr-extract-footer', 'modal-ocr-extract-footer');
            syncChk('ocr-include-blocks', 'modal-ocr-include-blocks');
            syncChk('ocr-include-images', 'modal-ocr-include-images');

            // Hide/Show sections based on current model support
            const model = get('model-select').value;
            const isGpt = isGptImageModel(model);
            const isGemini = isGeminiImageModel(model);
            const isGrok = isGrokImageModel(model);
            if(get('modal-gpt-image-options')) get('modal-gpt-image-options').classList.toggle('hidden', !isGpt);
            if(get('modal-gemini-image-options')) get('modal-gemini-image-options').classList.toggle('hidden', !isGemini);
            if(get('modal-grok-image-options')) get('modal-grok-image-options').classList.toggle('hidden', !isGrok);
            if(get('modal-mistral-ocr-options')) get('modal-mistral-ocr-options').classList.toggle('hidden', !isMistralOcrModel(model));
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
            const nextVars = [
                ['--theme-500', theme.base],
                ['--theme-600', theme.dark],
                ['--theme-700', theme.darker],
                ['--theme-300', theme.light],
                ['--theme-200', theme.lighter],
                ['--theme-rgb', theme.rgb]
            ];
            nextVars.forEach(([name, nextValue]) => {
                if (root.style.getPropertyValue(name).trim() !== String(nextValue).trim()) {
                    root.style.setProperty(name, nextValue);
                }
            });
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
            '#top-model-bar',
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
