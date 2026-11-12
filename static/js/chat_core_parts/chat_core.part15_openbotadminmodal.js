            window.openBotAdminModal = openBotAdminModal;
            // Bot検出ログ（記録の確認・削除）はアカウント管理と別の画面。本体は part17 の BotAdminLog。
            const openBotLogModal = async () => {
                if (!isAdminUser || !get('bot-log-modal') || !window.BotAdminLog) return;
                const settingsEl = get('settings-modal');
                if (settingsEl && (settingsEl.classList.contains('modal-open') || settingsEl.classList.contains('modal-prep'))) {
                    hideModal('settings-modal');
                }
                showModal('bot-log-modal');
                if (location.pathname !== '/admin-bot-logs') {
                    history.pushState({ modal: 'admin-bot-logs' }, '', '/admin-bot-logs');
                }
                window.BotAdminLog.bind();
                await window.BotAdminLog.openList();
            };
            window.openBotLogModal = openBotLogModal;
            window.closeBotLogModal = (skipHistory = false) => {
                if (get('bot-log-modal')) hideModal('bot-log-modal');
                if (!skipHistory && location.pathname === '/admin-bot-logs') {
                    history.back();
                }
            };
            if (get('bot-log-open')) {
                get('bot-log-open').onclick = () => { openBotLogModal(); };
            }
            if (get('bot-log-close')) {
                get('bot-log-close').onclick = () => window.closeBotLogModal();
            }
            window.closeBotAdminModal = (skipHistory = false) => {
                const modal = get('bot-admin-modal') || botAdminModal;
                if (modal) hideModal('bot-admin-modal');
                if (!skipHistory && location.pathname === '/admin-bots') {
                    history.back();
                }
            };
            if (get('bot-admin-open')) {
                get('bot-admin-open').onclick = () => { openBotAdminModal(); };
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
                            let d = {};
                            try { d = await res.json(); } catch (e) {}
                            showToast(d.error === 'admin_account' ? '管理者アカウントは削除できません' : (d.error || 'エラーが発生しました'), 'error', true);
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
                '/batch': { id: 'batch-modal', open: () => window.showBatchModal() },
                '/paste': { id: 'rich-paste-modal', open: () => openRichPasteModal() },
                '/camera': { id: 'camera-capture-modal', open: () => openCameraCaptureModal() },
                '/edit-image': { id: 'marker-modal', open: () => { /* Marker modal usually needs context */ } },
                '/chat-settings': { id: 'thread-modal', open: () => window.openThreadModal() },
                '/model': { id: 'model-modal', open: () => openModelModal() },
                '/token-details': { id: 'token-detail-modal', open: () => {
                    if (history.state && history.state.promptEstimate) restorePromptTokenDetail();
                    else if (history.state && history.state.messageId !== undefined) openTokenDetail(history.state.messageId);
                    else showTokenDetailModal();
                } },
                '/encryption-status': { id: 'encryption-status-modal', open: () => showEncryptionStatusModal() },
                '/python-execution': { id: 'python-exec-modal', open: () => showPythonExecDetailModal() },
                '/skill': { id: 'skill-modal', open: () => { editingSkillUuid = null; get('skill-modal-title').textContent = 'Create New Skill'; showModal('skill-modal'); } },
                '/compression': { id: 'compression-modal', open: () => window.openCompressionModal() },
                '/admin-bots': { id: 'bot-admin-modal', open: () => openBotAdminModal() },
                '/admin-bot-logs': { id: 'bot-log-modal', open: () => openBotLogModal() }
            };
            // Legacy URL alias for bookmarks and old links.
            MODAL_CONFIG['/gem'] = MODAL_CONFIG['/skill'];

            const closeModalById = (id, skipHistory = false) => {
                switch (id) {
                    case 'settings-modal': closeSettingsModal(skipHistory); break;
                    case 'upload-modal': closeUploadModal(skipHistory); break;
                    case 'image-split-modal': closeImageSplit(); break;
                    case 'image-split-compression-modal': hideModal(id); break;
                    case 'camera-capture-modal': closeCameraCaptureModal(skipHistory ? {skipHistory: true} : {}); break;
                    case 'history-modal': if (window.closeHistoryModal) window.closeHistoryModal(skipHistory); break;
                    case 'lib-modal': if (window.closeLibModal) window.closeLibModal(skipHistory); break;
                    case 'branch-modal': if (window.closeBranchModal) window.closeBranchModal(skipHistory); break;
                    case 'batch-modal': if (window.closeBatchModal) window.closeBatchModal(skipHistory); break;
                    case 'rich-paste-modal': if (window.closeRichPasteModal) window.closeRichPasteModal(skipHistory); break;
                    case 'marker-modal': if (window.closeMarkerModal) window.closeMarkerModal(skipHistory); break;
                    case 'thread-modal': if (window.closeThreadModal) window.closeThreadModal(skipHistory); break;
                    case 'model-modal': if (window.closeModelModal) window.closeModelModal(skipHistory); break;
                    case 'token-detail-modal': closeTokenDetail(skipHistory); break;
                    case 'encryption-status-modal': closeEncryptionModal(skipHistory); break;
                    case 'python-exec-modal': closePythonExecDetail(skipHistory); break;
                    case 'skill-modal': if (window.closeSkillModal) window.closeSkillModal(skipHistory); break;
                    case 'compression-modal': if (window.closeCompressionModal) window.closeCompressionModal(skipHistory); break;
                    case 'bot-admin-modal': if (window.closeBotAdminModal) window.closeBotAdminModal(skipHistory); break;
                    case 'bot-log-modal': if (window.closeBotLogModal) window.closeBotLogModal(skipHistory); break;
                    case 'mcp-decision-modal':
                        if (typeof submitMcpDecision === 'function') submitMcpDecision('deny'); else hideModal(id);
                        break;
                    case 'api-key-required-modal': {
                        const cancelBtn = get('api-key-modal-cancel-btn');
                        if (cancelBtn && typeof cancelBtn.onclick === 'function') cancelBtn.click(); else hideModal(id);
                        break;
                    }
                    case 'lyria-studio-modal': if (window.closeLyriaStudio) window.closeLyriaStudio(); else hideModal(id); break;
                    case 'voice-studio-modal': if (window.VoiceStudio) window.VoiceStudio.close(); else hideModal(id); break;
                    case 'version-update-modal':
                        const latest = localStorage.getItem("app_version") || "";
                        if (latest) localStorage.setItem("version_notified", latest);
                        hideModal(id);
                        break;
                    default: hideModal(id); break;
                }
            };

            window.addEventListener('popstate', (e) => {
                // A settings child has its own history entry, even though the URL stays /settings.
                const pickerTarget = e.state?.settingsModelPicker;
                if (location.pathname === '/settings' && settingsModelPickerTargets.includes(pickerTarget)) {
                    const restorePicker = () => openSettingsModelPicker(pickerTarget, true);
                    if (get('settings-modal')?.classList.contains('modal-open')) restorePicker();
                    else window.openSettingsModal().then(() => {
                        if (location.pathname === '/settings' && history.state?.settingsModelPicker === pickerTarget) restorePicker();
                    });
                    return;
                }
                if (settingsModelPickerTarget) window.closeModelModal(true);
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

            document.addEventListener('keydown', (e) => {
                if (e.key === 'Escape' && settingsModelPickerTarget) {
                    e.preventDefault();
                    window.closeModelModal();
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
            // Images attached to a feedback or a reply (server/feedback_images.py): up to four, sent after the text itself.
            const FB_IMAGE_MAX_COUNT = 4;
            const FB_IMAGE_MAX_BYTES = 5 * 1024 * 1024;
            function createFbImagePicker(addButton, input, box) {
                let entries = [];
                const render = () => {
                    if (!box) return;
                    box.innerHTML = '';
                    entries.forEach((entry, index) => {
                        const cell = document.createElement('div');
                        cell.className = 'relative w-16 h-16 rounded border border-gray-600 overflow-hidden';
                        const img = document.createElement('img');
                        img.src = entry.url;
                        img.alt = entry.file.name;
                        img.className = 'w-full h-full object-cover';
                        const remove = document.createElement('button');
                        remove.type = 'button';
                        remove.className = 'absolute top-0 right-0 bg-black/70 text-white text-[10px] leading-none px-1 py-0.5';
                        remove.textContent = '×';
                        remove.onclick = () => {
                            URL.revokeObjectURL(entry.url);
                            entries.splice(index, 1);
                            render();
                        };
                        cell.appendChild(img);
                        cell.appendChild(remove);
                        box.appendChild(cell);
                    });
                };
                if (addButton && input) {
                    addButton.onclick = () => input.click();
                    input.onchange = (event) => {
                        for (const file of Array.from(event.target.files || [])) {
                            if (!/^image\/(png|jpeg|webp|gif)$/.test(file.type)) { showToast("PNG・JPEG・WebP・GIF の画像を選択してください", "error", true); continue; }
                            if (file.size > FB_IMAGE_MAX_BYTES) { showToast("画像は1枚5MBまでです", "error", true); continue; }
                            if (entries.length >= FB_IMAGE_MAX_COUNT) { showToast("添付できる画像は4枚までです", "error", true); break; }
                            entries.push({file, url: URL.createObjectURL(file)});
                        }
                        event.target.value = '';
                        render();
                    };
                }
                return {
                    files: () => entries.map(entry => entry.file),
                    clear: () => {
                        entries.forEach(entry => URL.revokeObjectURL(entry.url));
                        entries = [];
                        render();
                    },
                };
            }
            // Sends [files] to [url] after the feedback or reply was accepted; false when they were not saved.
            async function uploadFbImages(url, files) {
                try {
                    const form = new FormData();
                    files.forEach(file => form.append('images', file, file.name));
                    return (await apiFetch(url, {method:'POST', body: form})).ok;
                } catch (e) { return false; }
            }
            const fbImagePicker = createFbImagePicker(get('fb-images-add'), get('fb-images-input'), get('fb-images-list'));
            get('fb-submit').onclick = async () => {
                const title = get('fb-title').value.trim();
                const message = get('fb-message').value.trim();
                if(!message) { showToast("フィードバック内容を入力してください", "error", true); return; }
                // Attach recorded logs only when selected for this feedback.
                const payload = {title, message, client: 'web', version: (window.CHAT_CONFIG && window.CHAT_CONFIG.appVersion) || ''};
                const attachLogs = get('fb-attach-logs');
                const clientLogs = attachLogs && attachLogs.checked && window.ActivityLog ? window.ActivityLog.feedbackPayload() : null;
                if (clientLogs) payload.client_logs = clientLogs;
                // The server copies everything it has on the open chat, decrypted, next to the logs (server/feedback_chat_copy.py).
                const attachChat = get('fb-attach-chat');
                const chatCopy = !!(attachChat && attachChat.checked);
                if (chatCopy && !currentThreadId) { showToast("コピーを送信するチャットが開かれていません", "error", true); return; }
                if (chatCopy) payload.chat_copy = window.ActivityLog ? window.ActivityLog.chatCopyPayload(currentThreadId) : {client: 'web', thread_id: String(currentThreadId)};
                const res = await apiFetch("/api/feedback", {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
                // The feedback's number comes back with the answer; the images follow it.
                const sent = res.ok ? await res.clone().json().catch(() => ({})) : {};
                const files = fbImagePicker.files();
                const images = {count: files.length, saved: true};
                if (res.ok && images.count) images.saved = await uploadFbImages(`/api/feedback/${sent.public_id}/images`, files);
                if (!(window.ActivityLog ? await window.ActivityLog.reportFeedback(res, clientLogs, chatCopy, images) : res.ok)) return;
                get('fb-title').value = '';
                get('fb-message').value = '';
                fbImagePicker.clear();
                if (attachChat) attachChat.checked = false;
                if (attachLogs) attachLogs.checked = true;
                loadFeedback();
            };

            // The sender (or an administrator) deletes a feedback; the server removes its directory too.
            async function deleteFeedback(id) {
                if (!confirm("このフィードバックを削除しますか？\n一緒に送信した操作ログとチャットのコピーも削除されます。")) return;
                const res = await apiFetch(`/api/feedback/${id}`, {method:'DELETE'});
                if (!res.ok) { showToast("フィードバックを削除できませんでした", "error", true); return; }
                showToast("フィードバックを削除しました", "success");
                loadFeedback();
            }

            // Reply history of a feedback and the reply box under it (the sender on their own feedback, administrators on any).
            // Each 返信 adds to the history. The activity log goes with a reply only when its box is ticked
            // (it needs ログの収集を強化 to be on); images follow the reply like those of a feedback.
            const FB_REPLY_LOGS_LABEL = '保存期間内の操作ログを一緒に送信する';
            let fbReplyPickers = [];
            function feedbackRepliesHtml(item, isAdmin) {
                const replies = item.replies || [];
                if (!replies.length) return '';
                return `<div class="fb-replies mt-2 space-y-2 border-l-2 border-gray-700 pl-2">${replies.map(r => {
                    const who = r.role === 'admin' ? (isAdmin && r.author ? `管理者 (${r.author})` : '管理者') : (isAdmin ? '送信者' : 'あなた');
                    const files = isAdmin ? [r.log_file && `操作ログ: ${r.log_file}`, r.chat_file && `チャットのコピー: ${r.chat_file}`, r.image_dir && `添付画像（${r.image_count}枚）: ${r.image_dir}`] : [r.image_count && `添付画像: ${r.image_count}枚`];
                    return `<div class="text-[11px]"><div class="text-gray-400">${escapeHtml(who)} / ${escapeHtml(r.created_at || '')}</div><div class="text-sm whitespace-pre-wrap ${r.role === 'admin' ? 'text-green-300' : 'text-gray-200'}">${escapeHtml(r.message)}</div>${files.filter(Boolean).map(t => `<div class="${isAdmin ? 'text-amber-300' : 'text-gray-400'}">${escapeHtml(t)}</div>`).join('')}</div>`;
                }).join('')}</div>`;
            }
            function feedbackReplyFormHtml() {
                return `<div class="fb-reply-form mt-2 space-y-2">
                    <textarea class="fb-reply-message w-full bg-gray-900 border border-gray-700 rounded px-2 py-1 text-xs text-white" rows="2" placeholder="返信内容"></textarea>
                    <div>
                        <input type="file" class="fb-reply-images-input hidden" accept="image/png,image/jpeg,image/webp,image/gif" multiple>
                        <button type="button" class="fb-reply-images-add bg-gray-700 hover:bg-gray-600 text-white rounded px-3 py-1 text-xs"><i class="fas fa-image mr-1"></i>画像を添付</button>
                        <span class="text-[10px] text-gray-500 ml-2">PNG・JPEG・WebP・GIF、4枚まで、1枚5MBまで</span>
                        <div class="fb-reply-images-list flex flex-wrap gap-2 mt-2"></div>
                    </div>
                    <label class="flex items-start gap-2 cursor-pointer">
                        <input type="checkbox" class="fb-reply-logs accent-amber-500 mt-0.5 shrink-0">
                        <span>
                            <span class="block text-xs font-bold text-white">${FB_REPLY_LOGS_LABEL}</span>
                            <span class="fb-reply-logs-note block text-[10px] text-gray-500 mt-1"></span>
                        </span>
                    </label>
                    <label class="flex items-start gap-2 cursor-pointer">
                        <input type="checkbox" class="fb-reply-chat accent-amber-500 mt-0.5 shrink-0">
                        <span class="text-xs text-white">現在開いているチャットのコピーを送信する<span class="block text-[10px] text-gray-500">開いているチャットのメッセージ、回答、思考過程、添付ファイル、チャットの設定と、そのチャットに関するエラーログなどの記録を復号して、不具合の調査用に送信します。送信したコピーは、チャットを削除しても残ります。</span></span>
                    </label>
                    <button type="button" class="fb-reply-send bg-blue-600 hover:bg-blue-500 text-white px-3 py-1 rounded text-xs">返信</button>
                </div>`;
            }
            function bindFeedbackReply(item, el) {
                const form = el.querySelector('.fb-reply-form');
                if (!form) return;
                const logs = form.querySelector('.fb-reply-logs');
                const note = form.querySelector('.fb-reply-logs-note');
                const on = !!(window.ActivityLog && window.ActivityLog.canAttach());
                logs.disabled = !on;
                note.textContent = on
                    ? 'チェックしたときだけ、「ログの収集を強化」で記録した保存期間内のログを返信と一緒に送信します。'
                    : '「ログの収集を強化」がオフのため、ログは送信できません。';
                const picker = createFbImagePicker(form.querySelector('.fb-reply-images-add'), form.querySelector('.fb-reply-images-input'), form.querySelector('.fb-reply-images-list'));
                fbReplyPickers.push(picker);
                const send = form.querySelector('.fb-reply-send');
                send.onclick = async () => {
                    const message = form.querySelector('.fb-reply-message').value.trim();
                    if (!message) { showToast("返信内容を入力してください", "error", true); return; }
                    const payload = {message, client: 'web', version: (window.CHAT_CONFIG && window.CHAT_CONFIG.appVersion) || ''};
                    const clientLogs = logs.checked && window.ActivityLog ? window.ActivityLog.feedbackPayload() : null;
                    if (clientLogs) payload.client_logs = clientLogs;
                    const chatCopy = form.querySelector('.fb-reply-chat').checked;
                    if (chatCopy && !currentThreadId) { showToast("コピーを送信するチャットが開かれていません", "error", true); return; }
                    if (chatCopy) payload.chat_copy = window.ActivityLog ? window.ActivityLog.chatCopyPayload(currentThreadId) : {client: 'web', thread_id: String(currentThreadId)};
                    const files = picker.files();
                    send.disabled = true;
                    try {
                        const res = await apiFetch(`/api/feedback/${item.public_id}/replies`, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
                        const data = await res.json().catch(() => ({}));
                        if (!res.ok) { showToast(data.error || "返信を送信できませんでした", "error", true); return; }
                        const imagesSaved = !files.length || await uploadFbImages(`/api/feedback/${item.public_id}/replies/${data.reply_id}/images`, files);
                        const failed = [];
                        if (clientLogs && data.logs_saved === false) failed.push('ログ');
                        if (!imagesSaved) failed.push('画像');
                        if (chatCopy && data.chat_copy_saved !== true) failed.push('チャットのコピー');
                        if (failed.length) {
                            showToast(`返信を送信しました（${failed.join('と')}は保存できませんでした）`, "error", true);
                        } else {
                            const sent = ['返信'];
                            if (chatCopy) sent.push('チャットのコピー');
                            if (clientLogs) sent.push(`保存期間内のログ（${typeof data.logs_count === 'number' ? data.logs_count : clientLogs.entries.length}件）`);
                            if (files.length) sent.push(`画像（${files.length}枚）`);
                            showToast(sent.join(sent.length > 2 ? '、' : 'と') + 'を送信しました', "success");
                        }
                        loadFeedback();
                    } finally {
                        send.disabled = false;
                    }
                };
            }

            function filterAdminFeedback() {
                const status = get('fb-admin-status-filter').value;
                let visible = 0;
                get('fb-admin-list').querySelectorAll('[data-feedback-status]').forEach(row => {
                    const matches = status === 'all' || (status === 'unresolved'
                        ? ['new', 'in_review', 'replied'].includes(row.dataset.feedbackStatus)
                        : row.dataset.feedbackStatus === status);
                    row.hidden = !matches;
                    if (matches) visible++;
                });
                get('fb-admin-empty').classList.toggle('hidden', visible !== 0);
            }
            get('fb-admin-status-filter').onchange = filterAdminFeedback;

            async function loadFeedback() {
                fbReplyPickers.forEach(picker => picker.clear());
                fbReplyPickers = [];
                const res = await apiFetch("/api/feedback?all=1");
                const data = await res.json();
                const list = get('fb-list');
                list.innerHTML = '';
                (data.items || []).filter(i => !data.is_admin || i.user_id === undefined || i.user_id === null || true).forEach(item => {
                    if(data.is_admin) return;
                    const el = document.createElement('div');
                    el.className = 'p-2 rounded border border-gray-700 bg-gray-800/50';
                    el.innerHTML = `<div class="text-[11px] text-gray-400">ID: <span class="select-all font-mono">${escapeHtml(item.public_id||'')}</span> / ${item.created_at}</div><div class="font-bold text-sm">${escapeHtml(item.title||'No Title')}</div><div class="text-sm whitespace-pre-wrap">${escapeHtml(item.message)}</div><div class="text-[11px] text-gray-400 mt-1">Status: ${escapeHtml(item.status === "resolved_unreleased" ? "resolved（non-released）" : item.status)}</div>${item.image_count ? `<div class="text-[11px] text-gray-400 mt-1">添付画像: ${item.image_count}枚</div>` : ''}${feedbackRepliesHtml(item, false)}${feedbackReplyFormHtml()}<div class="mt-2"><button type="button" class="fb-delete bg-red-700 hover:bg-red-600 text-white px-2 py-1 rounded text-[10px]">削除</button></div>`;
                    el.querySelector('.fb-delete').onclick = () => deleteFeedback(item.public_id);
                    bindFeedbackReply(item, el);
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
                        el.dataset.feedbackStatus = item.status || 'new';
                        el.innerHTML = `
                            <div class="text-[11px] text-gray-400">ID: <span class="select-all font-mono">${escapeHtml(item.public_id||'')}</span> / ${item.agent_name ? `agent:${escapeHtml(item.agent_name)}` : `user:${item.user_id}`} / ${item.created_at}${item.handled_by ? ` / 対応: ${escapeHtml(item.handled_by)}` : ''}</div>
                            <div class="font-bold text-sm">${escapeHtml(item.title||'No Title')}</div>
                            <div class="text-sm whitespace-pre-wrap">${escapeHtml(item.message)}</div>
                            ${[item.log_file && `操作ログ: ${item.log_file}`, item.chat_file && `チャットのコピー: ${item.chat_file}`, item.image_dir && `添付画像（${item.image_count}枚）: ${item.image_dir}`].filter(Boolean).map(t => `<div class="text-[11px] text-amber-300">${escapeHtml(t)}</div>`).join('')}
                            <div class="flex items-center gap-2">
                                <select class="fb-status bg-gray-900 border border-gray-700 rounded px-2 py-1 text-xs text-white">
                                    <option value="new">new</option>
                                    <option value="in_review">in_review</option>
                                    <option value="replied">replied</option>
                                    <option value="rejected">rejected</option>
                                    <option value="resolved">resolved</option>
                                    <option value="resolved_unreleased">resolved（non-released）</option>
                                </select>
                                <button class="fb-save bg-blue-600 hover:bg-blue-500 text-white px-3 py-1 rounded text-xs">状態を保存</button>
                                <button type="button" class="fb-delete bg-red-700 hover:bg-red-600 text-white px-3 py-1 rounded text-xs">削除</button>
                            </div>
                            ${feedbackRepliesHtml(item, true)}
                            ${feedbackReplyFormHtml()}
                        `;
                        el.querySelector('.fb-status').value = item.status || 'new';
                        el.querySelector('.fb-save').onclick = async () => {
                            const status = el.querySelector('.fb-status').value;
                            await apiFetch(`/api/feedback/${item.public_id}/update`, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({status})});
                            loadFeedback();
                        };
                        el.querySelector('.fb-delete').onclick = () => deleteFeedback(item.public_id);
                        bindFeedbackReply(item, el);
                        adminList.appendChild(el);
                    });
                    filterAdminFeedback();
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
                        slashSelectedIndex = Math.min(slashSelectedIndex + 1, visibleSlashCommands(lastSlashFilter || '').length - 1);
                        showSlashCommandSuggestions(slashCommandSuggestionFilter(extractSlashCommandToken(input.value), input.value));
                        return;
                    }
                    if (e.key === 'ArrowUp') {
                        e.preventDefault();
                        slashSelectedIndex = Math.max(slashSelectedIndex - 1, 0);
                        showSlashCommandSuggestions(slashCommandSuggestionFilter(extractSlashCommandToken(input.value), input.value));
                        return;
                    }
                    if (e.key === 'Enter') {
                        e.preventDefault();
                        // Pick the currently highlighted (or first)
                        const filtered = visibleSlashCommands(slashCommandSuggestionFilter(extractSlashCommandToken(input.value), input.value));
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

                // Skill suggestion keyboard navigation
                if (skillSuggestionsVisible) {
                    const val = input.value.trim();
                    if (e.key === 'ArrowDown') {
                        e.preventDefault();
                        skillSelectedIndex = skillSelectedIndex + 1;
                        showSkillSuggestions(val.substring(1));
                        return;
                    }
                    if (e.key === 'ArrowUp') {
                        e.preventDefault();
                        skillSelectedIndex = Math.max(skillSelectedIndex - 1, 0);
                        showSkillSuggestions(val.substring(1));
                        return;
                    }
                    if (e.key === 'Enter') {
                        e.preventDefault();
                        const filter = val.substring(1).toLowerCase();
                        const filtered = loadedSkills.filter(g =>
                            g.name.toLowerCase().includes(filter) || (g.description && g.description.toLowerCase().includes(filter))
                        );
                        if (filtered[skillSelectedIndex]) {
                            selectSkillSuggestion(filtered[skillSelectedIndex]);
                        } else if (filtered.length > 0) {
                            selectSkillSuggestion(filtered[0]);
                        }
                        return;
                    }
                    if (e.key === 'Escape') {
                        e.preventDefault();
                        hideSkillSuggestions();
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

                    // Slash command / skill suggestion triggers
                    const val = this.value.trim();
                    if (pendingSlashCommand) {
                        if (skillSuggestionsVisible) hideSkillSuggestions();
                        if (slashSuggestionsVisible) hideSlashCommandSuggestions();
                        lastSlashFilter = null;
                    } else if (val.startsWith('@')) {
                        const filter = val.substring(1);
                        showSkillSuggestions(filter);
                        if (slashSuggestionsVisible) hideSlashCommandSuggestions();
                        lastSlashFilter = null;
                    } else if (val.startsWith('/')) {
                        const filter = slashCommandSuggestionFilter(extractSlashCommandToken(val), this.value);
                        // The command is only committed by pressing Enter or
                        // clicking a palette item, never while typing. Also,
                        // rebuild the list only when the command name being
                        // typed changes; typing the instruction text after the
                        // command must not re-render the palette each key.
                        if (!slashSuggestionsVisible || filter !== lastSlashFilter) {
                            lastSlashFilter = filter;
                            showSlashCommandSuggestions(filter);
                        }
                        if (skillSuggestionsVisible) hideSkillSuggestions();
                    } else {
                        if (skillSuggestionsVisible) hideSkillSuggestions();
                        if (slashSuggestionsVisible) hideSlashCommandSuggestions();
                        lastSlashFilter = null;
                    }
                });

                get('prompt-input').addEventListener('blur', () => {
                    // Delay to allow click on suggestion items
                    setTimeout(() => {
                        if (slashSuggestionsVisible) hideSlashCommandSuggestions();
                        if (skillSuggestionsVisible) hideSkillSuggestions();
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
