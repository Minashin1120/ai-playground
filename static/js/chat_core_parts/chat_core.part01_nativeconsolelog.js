        const get = (id) => document.getElementById(id);
        const nativeConsoleLog = (typeof console.log === 'function') ? console.log.bind(console) : function () {};
        const nativeConsoleInfo = (typeof console.info === 'function') ? console.info.bind(console) : nativeConsoleLog;
        // Set to true only after the settings modal has fetched the saved values
        // and populated every control.  The save button stays disabled until then
        // so a click during the load window cannot overwrite settings with the
        // form's default/unpopulated values (which used to silently toggle E2EE).
        let settingsModalLoaded = false;
        const setSettingsSaveEnabled = (enabled) => {
            ['set-default-model-picker', 'set-default-vision-model-picker'].forEach(id => {
                const button = get(id);
                if (button) button.disabled = !enabled;
            });
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
            document.addEventListener('load', (e) => {
                const el = e.target;
                if (!el || el.tagName !== 'IMG' || !el.classList.contains('chat-image')) return;
                const frame = el.closest('.chat-image-frame');
                if (!frame) return;
                frame.dataset.chatImageState = 'loaded';
                frame.removeAttribute('aria-busy');
            }, true);
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
            const buildBusyWarning = (filename) => {
                const box = document.createElement('div');
                box.style.cssText = 'display:flex;flex-direction:column;align-items:center;justify-content:center;width:100%;height:100%;min-height:80px;text-align:center;padding:8px;gap:4px;';
                box.innerHTML = '<i class="fas fa-hourglass-half" style="font-size:16px;color:#93c5fd"></i><div style="font-size:9px;color:#bfdbfe;font-weight:700;line-height:1.3">一時的に混雑しています<br>時間をおいて再読み込みしてください</div>';
                if (filename) box.setAttribute('data-file-name', String(filename));
                return box;
            };
            const fullFileUrl = (u) => String(u || '').split('?')[0].replace('/files/thumb/', '/files/');
            document.addEventListener('error', (e) => {
                const el = e.target;
                if (!el || el.tagName !== 'IMG') return;
                const src = el.currentSrc || el.src || '';
                if (!isFileUrl(src)) {
                    const svgFrame = el.closest && el.closest('.svg-code-render');
                    if (svgFrame) {
                        svgFrame.dataset.svgState = 'error';
                        return;
                    }
                    const frame = el.closest && el.closest('.chat-image-frame');
                    if (frame) {
                        frame.dataset.chatImageState = 'error';
                        frame.removeAttribute('aria-busy');
                        const status = frame.querySelector('.chat-image-loading');
                        const label = status && status.querySelector('span');
                        if (label) label.textContent = '画像を読み込めませんでした';
                        const icon = status && status.querySelector('i');
                        if (icon) icon.className = 'fas fa-image';
                    }
                    return;
                }
                e.stopImmediatePropagation();
                e.preventDefault();
                const cleanSrc = String(src).split('?')[0];
                const filename = el.getAttribute('data-viewer-filename') || cleanSrc.split('/').pop();
                const showWarning = (keyMismatch) => {
                    const frame = el.closest && el.closest('.chat-image-frame');
                    if (frame) {
                        frame.dataset.chatImageState = 'failed';
                        frame.removeAttribute('aria-busy');
                    }
                    const replacement = buildWarning(filename, !!keyMismatch);
                    try { el.replaceWith(replacement); } catch (_) { /* detached */ }
                };
                const showBusyWarning = () => {
                    const frame = el.closest && el.closest('.chat-image-frame');
                    if (frame) {
                        frame.dataset.chatImageState = 'busy';
                        frame.removeAttribute('aria-busy');
                    }
                    const replacement = buildBusyWarning(filename);
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
                    if (status === 429 || status === 503) { showBusyWarning(); return; }
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
                    if (src.includes('/files/thumb/')) { showBusyWarning(); return; }
                    showWarning(false);
                });
            }, true);
        })();
        function buildChatImageHtml(src, options = {}) {
            const rawSrc = String(src || '');
            const alt = String(options.alt || '');
            const title = String(options.title || '');
            const viewerSrc = String(options.viewerSrc || rawSrc);
            const filename = String(options.filename || '');
            if (rawSrc.startsWith('sandbox:')) {
                return `<span class="text-xs text-gray-500" title="${escapeHtml(rawSrc)}">${escapeHtml(alt) || '（画像データは取得できませんでした）'}</span>`;
            }
            const titleAttr = title ? ` title="${escapeHtml(title)}"` : '';
            const filenameAttr = filename ? ` data-viewer-filename="${escapeHtml(filename)}"` : '';
            return `<span class="chat-image-frame" data-chat-image-state="loading" aria-busy="true"><span class="chat-image-loading" role="status" aria-label="画像を読み込み中"><i class="fas fa-spinner fa-spin" aria-hidden="true"></i><span>画像を読み込み中…</span></span><img src="${escapeHtml(rawSrc)}" data-viewer-src="${escapeHtml(viewerSrc)}" alt="${escapeHtml(alt)}"${titleAttr}${filenameAttr} class="chat-image" loading="lazy" decoding="async" width="320" height="320"></span>`;
        }
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
            document.querySelectorAll('#thread-list > [data-thread-id], #skill-list > .skill-item').forEach((el) => {
                el.classList.remove('model-list-animate', 'slide-in-animate', 'fade-in', 'opacity-0');
                el.style.removeProperty('opacity');
                el.style.removeProperty('transform');
                el.style.removeProperty('animation');
                el.style.removeProperty('animation-delay');
                el.style.removeProperty('visibility');
            });
            ['thread-list', 'skill-list'].forEach((id) => {
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
        // Own audio player: every <audio controls> that enters the page (answers, file viewer) gets the
        // app's player UI.  The <audio> element stays inside it as the playback engine, so the markup the
        // server stores in messages does not change.
        const AUDIO_PLAYER_SPEEDS = [1, 1.25, 1.5, 2, 0.75];
        const formatAudioPlayerTime = (seconds) => {
            const total = Number.isFinite(seconds) && seconds > 0 ? Math.floor(seconds) : 0;
            return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, '0')}`;
        };
        const upgradeAudioPlayer = (audio) => {
            if (!audio || audio.dataset.aipReady === '1' || !audio.hasAttribute('controls')) return;
            const parent = audio.parentNode;
            if (!parent) return;
            audio.dataset.aipReady = '1';
            audio.removeAttribute('controls');
            if (!audio.getAttribute('preload')) audio.preload = 'metadata';

            const root = document.createElement('div');
            root.className = 'aip';
            root.setAttribute('role', 'group');
            root.setAttribute('aria-label', '音声プレイヤー');
            const toggle = document.createElement('button');
            toggle.type = 'button';
            toggle.className = 'aip-toggle';
            const toggleIcon = document.createElement('i');
            toggle.appendChild(toggleIcon);
            const time = document.createElement('span');
            time.className = 'aip-time';
            const seek = document.createElement('input');
            seek.type = 'range';
            seek.className = 'aip-seek';
            seek.min = '0';
            seek.max = '1000';
            seek.step = '1';
            seek.value = '0';
            seek.setAttribute('aria-label', '再生位置');
            const speed = document.createElement('button');
            speed.type = 'button';
            speed.className = 'aip-speed';
            speed.setAttribute('aria-label', '再生速度');
            const failed = document.createElement('span');
            failed.className = 'aip-failed';
            failed.textContent = '音声を再生できませんでした。';
            root.append(toggle, time, seek, speed, failed);
            parent.insertBefore(root, audio);
            root.appendChild(audio);

            let scrubbing = false;
            const duration = () => (Number.isFinite(audio.duration) && audio.duration > 0 ? audio.duration : 0);
            const refresh = () => {
                const total = duration();
                const position = scrubbing ? (Number(seek.value) / 1000) * total : audio.currentTime;
                time.textContent = `${formatAudioPlayerTime(position)} / ${formatAudioPlayerTime(total)}`;
                if (!scrubbing) seek.value = total ? String(Math.min(1000, (audio.currentTime / total) * 1000)) : '0';
                seek.disabled = !total;
                seek.style.setProperty('--aip-progress', `${Number(seek.value) / 10}%`);
            };
            const syncState = () => {
                const playing = !audio.paused && !audio.ended;
                toggleIcon.className = root.classList.contains('is-loading') ? 'fas fa-spinner fa-spin' : (playing ? 'fas fa-pause' : 'fas fa-play');
                const label = playing ? '一時停止' : '再生';
                toggle.setAttribute('aria-label', label);
                toggle.title = label;
                speed.textContent = `${audio.playbackRate}x`;
            };
            const setLoading = (value) => {
                root.classList.toggle('is-loading', value);
                syncState();
            };
            toggle.addEventListener('click', () => {
                if (root.classList.contains('is-failed')) {
                    root.classList.remove('is-failed');
                    audio.load();
                }
                if (audio.paused || audio.ended) {
                    if (audio.readyState < 3) setLoading(true);
                    const started = audio.play();
                    if (started && typeof started.catch === 'function') {
                        started.catch(() => { setLoading(false); });
                    }
                } else {
                    audio.pause();
                }
            });
            speed.addEventListener('click', () => {
                const index = AUDIO_PLAYER_SPEEDS.indexOf(audio.playbackRate);
                audio.playbackRate = AUDIO_PLAYER_SPEEDS[(index + 1) % AUDIO_PLAYER_SPEEDS.length];
            });
            seek.addEventListener('input', () => {
                scrubbing = true;
                refresh();
            });
            seek.addEventListener('change', () => {
                const total = duration();
                if (total) audio.currentTime = (Number(seek.value) / 1000) * total;
                scrubbing = false;
                refresh();
            });
            ['timeupdate', 'durationchange', 'loadedmetadata', 'seeked', 'emptied'].forEach((name) => audio.addEventListener(name, refresh));
            ['play', 'pause', 'ended', 'ratechange'].forEach((name) => audio.addEventListener(name, syncState));
            audio.addEventListener('waiting', () => setLoading(true));
            ['playing', 'pause', 'canplay', 'ended'].forEach((name) => audio.addEventListener(name, () => setLoading(false)));
            audio.addEventListener('loadeddata', () => root.classList.remove('is-failed'));
            audio.addEventListener('error', () => {
                root.classList.add('is-failed');
                setLoading(false);
            });
            refresh();
            syncState();
        };
        const scanAudioPlayers = (node) => {
            if (!node || node.nodeType !== 1) return;
            if (node.tagName === 'AUDIO') {
                upgradeAudioPlayer(node);
            } else if (node.querySelectorAll) {
                node.querySelectorAll('audio[controls]').forEach(upgradeAudioPlayer);
            }
        };
        const initAudioPlayers = () => {
            try {
                scanAudioPlayers(document.body);
                new MutationObserver((mutations) => {
                    mutations.forEach((mutation) => mutation.addedNodes.forEach(scanAudioPlayers));
                }).observe(document.body, { childList: true, subtree: true });
            } catch (e) {
                // The browser's own audio controls remain usable if the custom player cannot start.
            }
        };
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', initAudioPlayers, { once: true });
        } else {
            initAudioPlayers();
        }
        const THEME_DEFAULT = '#0dd4bf';
        const THEME_STORAGE_KEY = 'theme_color';
        const INITIAL_THEME_COLOR = (window.CHAT_CONFIG && window.CHAT_CONFIG.initialThemeColor) || null;
        const INITIAL_LIGHT_MODE_ENABLED = !!(window.CHAT_CONFIG && window.CHAT_CONFIG.initialLightModeEnabled);
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
        // Both the composer and image splitter use this pipeline, with independent settings.
        const readComposerCompression = () => ({
            enabled: !!get('enable-compression').checked,
            maxSizeMB: getCompressionMaxSizeMB(), maxDimension: getCompressionMaxDim(),
            outputType: getCompressionOutputType(), formatOnly: getCompressionFormatOnly()
        });
        async function applyImageCompression(file, settings) {
            if (!settings.enabled || !file.type.startsWith('image/')) return file;
            if (settings.formatOnly) return convertImageFormatOnly(file, settings.outputType);
            await ensureImageCompression();
            const options = { maxSizeMB: settings.maxSizeMB, maxWidthOrHeight: settings.maxDimension, useWebWorker: true };
            if (settings.outputType !== 'original') options.fileType = settings.outputType;
            const blob = await window.imageCompression(file, options);
            const result = new File([blob], imageFilenameForMime(file.name, blob.type || file.type),
                { type: blob.type || file.type, lastModified: file.lastModified || Date.now() });
            if (result.size > file.size) {
                showToast(`圧縮後にサイズが増加しました: ${formatBytes(file.size)} -> ${formatBytes(result.size)}（元ファイルを使用）`, 'warning', true);
                return file;
            }
            return result;
        }
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
            sync('gpt-image-complexity', 'modal-gpt-image-complexity');
            sync('gpt-image-format', 'modal-gpt-image-format');
            sync('gpt-image-compression', 'modal-gpt-image-compression');
            sync('gpt-image-frame-count', 'modal-gpt-image-frame-count');
            sync('gpt-image-frame-duration', 'modal-gpt-image-frame-duration');
            sync('gemini-image-aspect', 'modal-gemini-image-aspect');
            sync('gemini-image-size', 'modal-gemini-image-size');
            sync('grok-image-aspect', 'modal-grok-image-aspect');
            sync('grok-image-resolution', 'modal-grok-image-resolution');
            sync('grok-image-quality', 'modal-grok-image-quality');
            IDEOGRAM_IMAGE_FIELDS.forEach(f => sync(`ideogram-image-${f}`, `modal-ideogram-image-${f}`));
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
            const isGpt26 = isGpt && String(model).toLowerCase().includes('gpt-image-2.6');
            if(get('modal-gpt-image-complexity-wrap')) get('modal-gpt-image-complexity-wrap').classList.toggle('hidden', !isGpt26);
            if(get('modal-gpt-image-format')) Array.from(get('modal-gpt-image-format').options).forEach(opt => {
                if (opt.hasAttribute('data-gpt-image-26')) opt.hidden = !isGpt26;
            });
            if(get('modal-gpt-image-gif-wrap')) get('modal-gpt-image-gif-wrap').classList.toggle('hidden', !(isGpt26 && get('modal-gpt-image-format') && get('modal-gpt-image-format').value === 'gif'));
            if(get('modal-gpt-image-frame-duration-wrap')) get('modal-gpt-image-frame-duration-wrap').classList.toggle('hidden', !(isGpt26 && get('modal-gpt-image-format') && get('modal-gpt-image-format').value === 'gif'));
            if(get('modal-gemini-image-options')) get('modal-gemini-image-options').classList.toggle('hidden', !isGemini);
            if(get('modal-grok-image-options')) get('modal-grok-image-options').classList.toggle('hidden', !isGrok);
            if(get('modal-ideogram-image-options')) get('modal-ideogram-image-options').classList.toggle('hidden', !isIdeogramModel(model));
            if (typeof updateIdeogramImageUi === 'function') updateIdeogramImageUi();
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
        const applyLightMode = (enabled) => {
            const shouldEnable = !!enabled;
            let link = get('manual-theme-light-css');
            if (shouldEnable && !link) {
                const href = window.CHAT_CONFIG && window.CHAT_CONFIG.urls && window.CHAT_CONFIG.urls.manualLightTheme;
                if (!href) return;
                link = document.createElement('link');
                link.id = 'manual-theme-light-css';
                link.rel = 'stylesheet';
                link.href = href;
                document.head.appendChild(link);
            } else if (!shouldEnable && link) {
                link.remove();
            }
        };
