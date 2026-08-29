
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

                // Restore a chat thread when back/forward lands on /c/<id>
                const threadMatch = location.pathname.match(/^\/c\/(.+)$/);
                if (threadMatch) {
                    const tid = decodeURIComponent(threadMatch[1]);
                    if (String(currentThreadId) !== String(tid)) {
                        loadMessages(tid, { skipHistory: true });
                    }
                } else if (location.pathname === '/') {
                    // Returned to the home/new-chat view
                    if (currentThreadId) {
                        startNewChat({ skipHistory: true });
                    }
                }

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
                    if(!confirm("本当にアカウントを削除しますか？\nこの操作は取り消せません。")) return;
                    let res;
                    try {
                        res = await apiFetch(CHAT_CONFIG.urls.deleteAccount, {method:'POST'});
                    } catch (e) {
                        showToast('通信エラーが発生しました。時間をおいて再度お試しください。', 'error', true);
                        return;
                    }
                    if (res.ok) {
                        location.href = "/";
                        return;
                    }
                    let d = {};
                    try { d = await res.json(); } catch (e) {}
                    if (d && d.error === 'turnstile_required') {
                        showToast('アカウントを削除できませんでした。しばらく待ってから再度お試しください。', 'error', true);
                        return;
                    }
                    showToast(d.error || 'アカウントを削除できませんでした。時間をおいて再度お試しください。', 'error', true);
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
                        showSlashCommandSuggestions(extractSlashCommandToken(input.value));
                        return;
                    }
                    if (e.key === 'ArrowUp') {
                        e.preventDefault();
                        slashSelectedIndex = Math.max(slashSelectedIndex - 1, 0);
                        showSlashCommandSuggestions(extractSlashCommandToken(input.value));
                        return;
                    }
                    if (e.key === 'Enter') {
                        e.preventDefault();
                        // Pick the currently highlighted (or first)
                        const filtered = SLASH_COMMANDS.filter(c =>
                            c.label.toLowerCase().includes((extractSlashCommandToken(input.value) || '').toLowerCase())
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
                    if (codingModeEnabled) {
                        syncCodingModeUi(true, { persist: false });
                    }

                    // Slash command / gem suggestion triggers
                    const val = this.value.trim();
                    if (pendingSlashCommand) {
                        if (gemSuggestionsVisible) hideGemSuggestions();
                        if (slashSuggestionsVisible) hideSlashCommandSuggestions();
                        lastSlashFilter = null;
                    } else if (val.startsWith('@')) {
                        const filter = val.substring(1);
                        showGemSuggestions(filter);
                        if (slashSuggestionsVisible) hideSlashCommandSuggestions();
                        lastSlashFilter = null;
                    } else if (val.startsWith('/')) {
                        const filter = extractSlashCommandToken(val);
                        // The command is only committed by pressing Enter or
                        // clicking a palette item, never while typing. Also,
                        // rebuild the list only when the command name being
                        // typed changes; typing the instruction text after the
                        // command must not re-render the palette each key.
                        if (!slashSuggestionsVisible || filter !== lastSlashFilter) {
                            lastSlashFilter = filter;
                            showSlashCommandSuggestions(filter);
                        }
                        if (gemSuggestionsVisible) hideGemSuggestions();
                    } else {
                        if (gemSuggestionsVisible) hideGemSuggestions();
                        if (slashSuggestionsVisible) hideSlashCommandSuggestions();
                        lastSlashFilter = null;
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
            if (aiSettingsConversation.length > 0) {
                pendingSlashCommand = 'settings';
                showPendingSlashCommandIndicator('settings');
            }
            if (get('search-box')) {
                get('search-box').addEventListener('input', (event) => {
                    const el = get('search-box');
                    if (el && isUserInitiatedSearchInput(event)) {
                        markThreadSearchUserEdited(el);
                    } else if (el && !el.dataset.userEdited) {
                        discardAutofilledThreadSearch('cleared-autofill-search-box-input');
                        return;
                    }
                    if (isSettingsModalOpen()) {
                        snapshotSidebarHistory('ignore-search-input-settings-open');
                        return;
                    }
                    clearTimeout(searchTimeout);
                    searchTimeout = setTimeout(() => { loadThreads(false); }, 300);
                });
                hardenThreadSearchInputs();
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
            const viewerContentEl = document.querySelector('.viewer-content');
            if (viewerContentEl) {
                viewerContentEl.addEventListener('touchstart', onViewerTouchStart, { passive: false });
                viewerContentEl.addEventListener('touchmove', onViewerTouchMove, { passive: false });
                viewerContentEl.addEventListener('touchend', onViewerTouchEnd);
                viewerContentEl.addEventListener('touchcancel', onViewerTouchEnd);
            }
            get('image-viewer').addEventListener('click', (e) => {
                if (suppressViewerCloseClick) { suppressViewerCloseClick = false; return; }
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
                    this.interimInputTranscript = '';
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
                        // Live Transcription: TEXT output + input audio transcription config.
                        // Otherwise: audio-to-audio Live agent with transcription metadata.
                        const isTranscribeMode = !!(config && config.transcriptionConfig);
                        const setupMsg = {
                            setup: {
                                model: `models/${model}`,
                                generationConfig: {
                                    responseModalities: isTranscribeMode ? ["TEXT"] : ["AUDIO"]
                                },
                                // Enable transcription at setup level (as per docs)
                                inputAudioTranscription: isTranscribeMode
                                    ? (config.transcriptionConfig || {})
                                    : {},
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

                        // Live Translate uses a top-level translationConfig in the setup message
                        if (config.translationConfig) {
                            setupMsg.setup.translationConfig = config.translationConfig;
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
                            this.interimInputTranscript = '';
                        }
                        if (sc.interimInputTranscription) {
                            console.log("User interim transcription:", sc.interimInputTranscription.text);
                            this.interimInputTranscript = sc.interimInputTranscription.text;
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

            // ===========================================================================
            // RealtimeVoiceSession — true real-time STS over a server-held session.
            // The server keeps the provider WebSocket open; this controller streams
            // microphone PCM to the server via HTTP, consumes audio/transcripts over
            // SSE, and plays responses while the user keeps talking (server VAD or
            // Gemini's natural turn handling drives turn-taking).
            // ===========================================================================
            class RealtimeVoiceSession {
                constructor() {
                    this.active = false;
                    this.capturing = false;
                    this.sessionId = null;
                    this.abortCtrl = null;
                    this.reader = null;
                    this.audioCtx = null;
                    this.processor = null;
                    this.stream = null;
                    this.rtPlayer = null;
                    this.rateIn = 24000;
                    this.rateOut = 24000;
                    this.userTranscript = '';
                    this.assistantTranscript = '';
                    this.assistantThought = '';
                    this.speechActive = false;
                    this.responseDoneCount = 0;
                    this.lastAudioAt = 0;
                    this.streamError = null;
                    this.saved = false;
                    this.saving = false;
                    this.stopping = false;
                }

                isActive() {
                    return this.active;
                }

                async start() {
                    if (this.active) return;
                    if (this.saving || this.stopping) {
                        showToast('前の会話を処理中です。しばらくお待ちください。', 'warning', true);
                        return;
                    }
                    const model = get('model-select') ? get('model-select').value : '';
                    if (!isRealtimeSessionModel()) {
                        showToast('このモデルはリアルタイム会話に対応していません', 'warning', true);
                        return;
                    }
                    if (!currentThreadId) {
                        try {
                            const r = await apiFetch(CHAT_CONFIG.urls.handleThreads, {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
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
                        } catch (e) {
                            showToast('スレッドの作成に失敗しました: ' + e.message, 'error', true);
                            return;
                        }
                    }
                    const payload = {
                        model,
                        thread_id: currentThreadId,
                        voice: get('sts-voice') ? get('sts-voice').value : '',
                        speed: get('sts-speed') ? get('sts-speed').value : '',
                        rate_in: get('sts-rate-in') ? get('sts-rate-in').value : '',
                        rate_out: get('sts-rate-out') ? get('sts-rate-out').value : '',
                        thinking_level: get('sts-thinking-level') ? get('sts-thinking-level').value : '',
                        include_thoughts: get('sts-include-thoughts') ? get('sts-include-thoughts').checked : false,
                        target_lang: (isGeminiLiveTranslateModel() && get('sts-target-lang')) ? get('sts-target-lang').value : ''
                    };
                    setStsStatus('接続中...', true);
                    try {
                        const resp = await apiFetch('/api/realtime/start', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify(payload)
                        });
                        const data = await resp.json().catch(() => ({}));
                        if (!resp.ok) throw new Error(data.error || 'セッション開始に失敗しました');
                        this.sessionId = data.session_id;
                        this.rateIn = data.rate_in || this.rateIn;
                        this.rateOut = data.rate_out || this.rateOut;
                        this.active = true;
                        this.capturing = true;
                        this.saved = false;
                        this.userTranscript = '';
                        this.assistantTranscript = '';
                        this.assistantThought = '';
                        this.responseDoneCount = 0;
                        this.lastAudioAt = 0;
                        this.streamError = null;
                        this.rtPlayer = null;
                    } catch (e) {
                        setStsStatus('接続エラー', false);
                        showToast('リアルタイムセッションを開始できませんでした: ' + e.message, 'error', true);
                        return;
                    }

                    // Open the SSE stream first so audio events are not missed.
                    this.abortCtrl = new AbortController();
                    this._openStream();

                    // Start mic capture and stream PCM chunks to the server.
                    try {
                        await this._startCapture();
                    } catch (e) {
                        setStsStatus('マイクエラー', false);
                        showToast('マイクを利用できません: ' + e.message, 'error', true);
                        this._cancel();
                        return;
                    }

                    get('mic-btn').classList.remove('bg-gray-700');
                    get('mic-btn').classList.add('bg-red-600', 'animate-pulse');
                    setStsStatus('話してください...', true);
                }

                _openStream() {
                    const url = '/api/realtime/stream?session_id=' + encodeURIComponent(this.sessionId);
                    const opts = window.ProgressSpinner && typeof window.ProgressSpinner.manualRequestOptions === 'function'
                        ? window.ProgressSpinner.manualRequestOptions({ credentials: 'include', signal: this.abortCtrl.signal })
                        : { credentials: 'include', signal: this.abortCtrl.signal };
                    fetch(url, opts).then((resp) => {
                        if (!resp.ok) throw new Error('SSE stream failed (' + resp.status + ')');
                        this.reader = resp.body.getReader();
                        this._readLoop();
                    }).catch((e) => {
                        if (e && e.name === 'AbortError') return;
                        this.streamError = e && e.message ? e.message : 'ストリームエラー';
                        if (this.active) {
                            setStsStatus('ストリームエラー', false);
                            showToast('リアルタイム接続が切断されました', 'error', true);
                        }
                    });
                }

                async _readLoop() {
                    const decoder = new TextDecoder();
                    let buf = '';
                    try {
                        while (this.reader) {
                            const { done, value } = await this.reader.read();
                            if (done) break;
                            buf += decoder.decode(value, { stream: true });
                            let idx;
                            while ((idx = buf.indexOf('\n\n')) >= 0) {
                                const chunk = buf.slice(0, idx);
                                buf = buf.slice(idx + 2);
                                for (const line of chunk.split('\n')) {
                                    if (!line.startsWith('data: ')) continue;
                                    let ev = null;
                                    try { ev = JSON.parse(line.slice(6)); } catch (e) { continue; }
                                    this._handleEvent(ev);
                                }
                            }
                        }
                    } catch (e) {
                        if (e && e.name === 'AbortError') return;
                        if (this.active) {
                            this.streamError = e && e.message ? e.message : 'ストリームエラー';
                        }
                    } finally {
                        this.reader = null;
                    }
                }

                _handleEvent(ev) {
                    if (!ev) return;
                    switch (ev.type) {
                        case 'audio':
                            this.lastAudioAt = Date.now();
                            if (stsOpt('sts-auto-play')) {
                                if (!this.rtPlayer) {
                                    this.rtPlayer = new RealTimeAudioPlayer(this.rateOut || 24000);
                                    currentRtPlayer = this.rtPlayer;
                                }
                                setStsStatus('再生中...', true);
                                this.rtPlayer.addChunk(ev.data);
                            }
                            break;
                        case 'transcript':
                            if (ev.role === 'user') {
                                if (ev.cumulative) {
                                    this.userTranscript = ev.delta;
                                } else {
                                    this.userTranscript += ev.delta;
                                }
                                if (window.VoiceStudio) window.VoiceStudio.log('user', this.userTranscript);
                            } else if (ev.role === 'assistant') {
                                this.assistantTranscript += ev.delta;
                                if (window.VoiceStudio) window.VoiceStudio.log('assistant', this.assistantTranscript);
                            } else if (ev.role === 'thought') {
                                this.assistantThought += ev.delta;
                            }
                            break;
                        case 'speech_started':
                            this.speechActive = true;
                            this._stopPlayback();
                            setStsStatus('聞き取り中...', true);
                            break;
                        case 'speech_stopped':
                            this.speechActive = false;
                            setStsStatus('応答待ち...', true);
                            break;
                        case 'interrupted':
                            this._stopPlayback();
                            break;
                        case 'response_done':
                        case 'turn_complete':
                            this.responseDoneCount += 1;
                            break;
                        case 'status':
                            if (ev.status === 'ready' && this.active) setStsStatus('話してください...', true);
                            break;
                        case 'error':
                            this.streamError = ev.message || 'リアルタイムエラー';
                            setStsStatus('エラー', false);
                            break;
                        case 'final':
                            if (this.active && !this.saved) {
                                // The worker stopped; close out cleanly.
                                this._save();
                            }
                            break;
                    }
                }

                _stopPlayback() {
                    if (this.rtPlayer) {
                        try { this.rtPlayer.stop(); } catch (e) {}
                        this.rtPlayer = null;
                    }
                    currentRtPlayer = null;
                }

                _startCapture() {
                    const AC = window.AudioContext || window.webkitAudioContext;
                    if (!AC) throw new Error('AudioContext not supported');
                    this.audioCtx = new AC({ sampleRate: this.rateIn || 24000 });
                    return navigator.mediaDevices.getUserMedia(getMicCaptureConstraints()).then((stream) => {
                        this.stream = stream;
                        const source = this.audioCtx.createMediaStreamSource(stream);
                        const targetRate = this.rateIn || 24000;
                        const ctxRate = this.audioCtx.sampleRate;
                        const bufSize = 4096;
                        this.processor = this.audioCtx.createScriptProcessor(bufSize, 1, 1);
                        this.processor.onaudioprocess = (e) => {
                            if (!this.active || !this.capturing) return;
                            const input = e.inputBuffer.getChannelData(0);
                            const pcm = pcm16FromFloat32(input, ctxRate, targetRate);
                            if (!pcm || !pcm.byteLength) return;
                            this._sendAudio(pcm);
                        };
                        source.connect(this.processor);
                        this.processor.connect(this.audioCtx.destination);
                    });
                }

                _sendAudio(pcmBytes) {
                    if (!this.sessionId || !this.active) return;
                    const url = '/api/realtime/audio?session_id=' + encodeURIComponent(this.sessionId);
                    const opts = {
                        method: 'POST',
                        credentials: 'include',
                        headers: { 'X-CSRF-Token': csrfToken, 'Content-Type': 'application/octet-stream' },
                        body: pcmBytes
                    };
                    const finalOpts = window.ProgressSpinner && typeof window.ProgressSpinner.manualRequestOptions === 'function'
                        ? window.ProgressSpinner.manualRequestOptions(opts)
                        : opts;
                    fetch(url, finalOpts).catch(() => {});
                }

                _stopCapture() {
                    this.capturing = false;
                    if (this.processor) { try { this.processor.disconnect(); } catch (e) {} this.processor = null; }
                    if (this.stream) { try { this.stream.getTracks().forEach((t) => t.stop()); } catch (e) {} this.stream = null; }
                    if (this.audioCtx) { try { this.audioCtx.close(); } catch (e) {} this.audioCtx = null; }
                    stopSilenceMonitor();
                    stopMicWaveform();
                }

                async stop() {
                    if (!this.active) return;
                    this.active = false;
                    this.stopping = true;
                    this._stopCapture();
                    setStsStatus('応答を待っています...', true);
                    // Finalize any trailing audio so the last turn is included.
                    try {
                        await apiFetch('/api/realtime/commit', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ session_id: this.sessionId })
                        });
                    } catch (e) {}
                    // Wait for the final response (or a quiet timeout) before saving.
                    const startedAt = Date.now();
                    const beforeCount = this.responseDoneCount;
                    let lastActivity = this.lastAudioAt;
                    while (Date.now() - startedAt < 20000) {
                        if (this.responseDoneCount > beforeCount) break;
                        if (this.lastAudioAt > lastActivity) lastActivity = this.lastAudioAt;
                        if (!this.speechActive && Date.now() - startedAt > 2000 && Date.now() - lastActivity > 2500) break;
                        await new Promise((r) => setTimeout(r, 250));
                    }
                    await this._save();
                }

                async _save() {
                    if (this.saved) return;
                    this.saved = true;
                    this.saving = true;
                    try {
                        const resp = await apiFetch('/api/realtime/save', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ session_id: this.sessionId, thread_id: currentThreadId })
                        });
                        const data = await resp.json().catch(() => ({}));
                        if (!resp.ok) throw new Error(data.error || '保存に失敗しました');
                        if (this.streamError) {
                            setStsStatus('エラー', false);
                            showToast('リアルタイム会話でエラーが発生しました: ' + this.streamError, 'error', true);
                        } else {
                            setStsStatus('保存しました', false);
                            setTimeout(() => setStsStatus('Tap to speak', false), 1200);
                            try { await loadMessages(currentThreadId); } catch (e) {}
                        }
                    } catch (e) {
                        setStsStatus('保存エラー', false);
                        showToast('音声会話の保存に失敗しました: ' + (e && e.message ? e.message : e), 'error', true);
                    } finally {
                        this.saving = false;
                        this.stopping = false;
                        this._cleanup();
                    }
                }

                _cancel() {
                    if (this.sessionId) {
                        apiFetch('/api/realtime/cancel', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ session_id: this.sessionId })
                        }).catch(() => {});
                    }
                    this._cleanup();
                    setStsStatus('Canceled', false);
                    setTimeout(() => setStsStatus('Tap to speak', false), 800);
                }

                _cleanup() {
                    this.active = false;
                    this.capturing = false;
                    this.stopping = false;
                    this._stopCapture();
                    this._stopPlayback();
                    if (this.abortCtrl) { try { this.abortCtrl.abort(); } catch (e) {} this.abortCtrl = null; }
                    this.reader = null;
                    this.sessionId = null;
                    const micBtn = get('mic-btn');
                    if (micBtn) {
                        micBtn.classList.remove('bg-red-600', 'animate-pulse');
                        micBtn.classList.add('bg-gray-700');
                    }
                }
            }

            function pcm16FromFloat32(input, ctxRate, targetRate) {
                let data = input;
                if (ctxRate !== targetRate && ctxRate > 0 && targetRate > 0) {
                    const ratio = ctxRate / targetRate;
                    const outLen = Math.floor(data.length / ratio);
                    const down = new Float32Array(outLen);
                    for (let i = 0; i < outLen; i++) {
                        down[i] = data[Math.min(Math.floor(i * ratio), data.length - 1)];
                    }
                    data = down;
                }
                const pcm = new Int16Array(data.length);
                for (let i = 0; i < data.length; i++) {
                    const s = Math.max(-1, Math.min(1, data[i]));
                    pcm[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
                }
                return pcm.buffer;
            }
