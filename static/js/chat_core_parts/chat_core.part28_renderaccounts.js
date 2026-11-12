            const renderAccounts = () => {
                const list = get('bot-log-account-list');
                if (!list) return;
                const accounts = listState.accounts;
                const allChecked = accounts.length && accounts.every(a => listState.selected.has(a.user_id));
                const toolbar = get('bot-log-list-toolbar');
                if (toolbar) {
                    toolbar.innerHTML = `
                        <label class="flex items-center gap-1 text-gray-300"><input type="checkbox" class="bot-log-account-select-all" aria-label="表示中のアカウントをすべて選択" ${allChecked ? 'checked' : ''}>表示中をすべて選択</label>
                        <button class="bot-log-delete-accounts bg-red-600 hover:bg-red-500 text-white px-2 py-1 rounded" ${listState.selected.size ? '' : 'disabled'}>選択したアカウントの記録を一括削除${listState.selected.size ? `（${listState.selected.size}件）` : ''}</button>`;
                }
                if (!accounts.length) {
                    list.innerHTML = '<div class="text-xs text-gray-400">記録のあるアカウントはありません。</div>';
                    return;
                }
                list.innerHTML = accounts.map((a) => {
                    const badges = [];
                    if (!a.exists) badges.push(badge('削除済み', 'bg-gray-600'));
                    if (a.is_bot_banned) badges.push(badge('BAN中', 'bg-red-600'));
                    if (a.lock_remaining_seconds > 0) badges.push(badge(`ロック中（残り${Math.ceil(a.lock_remaining_seconds / 60)}分）`, 'bg-yellow-600'));
                    const summary = a.evidence_count
                        ? `記録 ${a.evidence_count}件・最終 ${escapeHtml(formatTime(a.last_event_at))}`
                        : '記録なし';
                    return `
                        <div class="flex items-center gap-2 bg-gray-900 border border-gray-700 rounded p-2 text-xs">
                            <input type="checkbox" class="bot-log-account-select" data-user-id="${a.user_id}" ${listState.selected.has(a.user_id) ? 'checked' : ''} aria-label="${escapeHtml(a.username || ('ID ' + a.user_id))} を選択">
                            <div class="flex-1 min-w-0">
                                <div class="flex flex-wrap items-center gap-1"><span class="${a.exists ? 'text-gray-200' : 'text-gray-400'} font-bold bot-log-wrap">${escapeHtml(a.username || ('ID ' + a.user_id))}</span>${badges.join('')}</div>
                                <div class="text-[10px] text-gray-500">${summary}</div>
                            </div>
                            <button class="bot-log-open-account shrink-0 whitespace-nowrap bg-gray-700 hover:bg-gray-600 text-white px-2 py-1 rounded" data-user-id="${a.user_id}" data-username="${escapeHtml(a.username || '')}">記録を見る</button>
                        </div>`;
                }).join('');
            };
            const loadAccounts = async () => {
                const list = get('bot-log-account-list');
                if (!list) return;
                list.innerHTML = '<div class="text-xs text-gray-400 py-2"><i class="fas fa-spinner fa-spin mr-1"></i>読み込み中...</div>';
                const search = get('bot-log-search');
                const q = search ? search.value.trim() : '';
                try {
                    const res = await apiFetch(`/api/bot/evidence/accounts?q=${encodeURIComponent(q)}`, { cache: 'no-store' });
                    const data = await res.json().catch(() => ({}));
                    if (!res.ok) throw new Error(data.error || String(res.status));
                    listState.accounts = data.accounts || [];
                    const visible = new Set(listState.accounts.map(a => a.user_id));
                    Array.from(listState.selected).forEach((id) => { if (!visible.has(id)) listState.selected.delete(id); });
                    renderAccounts();
                } catch (err) {
                    list.innerHTML = '<div class="text-xs text-red-400">アカウント一覧を取得できませんでした。</div>';
                    showToast('Bot検出ログの一覧を取得できませんでした', 'error', true);
                }
            };

            // ---- アカウントごとの状態と記録 ----
            const renderState = (st) => {
                if (!st) return '';
                const locks = st.locks || [];
                const idents = st.banned_identifiers || [];
                const badges = [];
                if (!st.exists) badges.push(badge('削除済み', 'bg-gray-600'));
                if (st.exists) {
                    badges.push(badge(st.detection_enabled ? '検出ON' : '検出OFF', st.detection_enabled ? 'bg-gray-600' : 'bg-gray-700'));
                    badges.push(badge(st.is_bot_banned ? 'BAN中' : 'BANなし', st.is_bot_banned ? 'bg-red-600' : 'bg-gray-600'));
                }
                badges.push(badge(locks.length ? 'ロック中' : 'ロックなし', locks.length ? 'bg-yellow-600' : 'bg-gray-600'));
                if (idents.length) badges.push(badge(`IP・端末のBAN ${idents.length}件`, 'bg-red-600'));
                if (st.is_admin) badges.push(badge('管理者（監視の対象外）', 'bg-blue-600'));
                const hasBanRecord = !!(state.counts.ban || state.counts.related_ban);
                const banLine = st.is_bot_banned
                    ? `<div class="text-gray-300">BANの理由: ${escapeHtml(st.bot_ban_reason || '-')}（${escapeHtml(formatTime(st.bot_banned_at))}）</div>${hasBanRecord ? '' : '<div class="text-gray-400">このBANの記録はありません（記録を始める前のBANか、記録が削除されています）。</div>'}`
                    : '';
                const lockLines = locks.map((lock) => {
                    const target = SOURCE_LABELS[lock.source] || lock.source;
                    const ident = lock.identifier ? ` ${escapeHtml(lock.identifier)}${lock.source === 'cookie' ? '…' : ''}` : '';
                    const origin = lock.origin
                        ? `・かけたアカウント: ${escapeHtml(lock.origin.username || String(lock.origin.user_id || '-'))}（${escapeHtml(CLIENT_LABELS[lock.origin.client] || lock.origin.client || '-')}）`
                        : '';
                    return `<div class="text-gray-300">ロック: ${escapeHtml(target)}${ident}・残り${escapeHtml(formatDuration(lock.remaining_seconds))}・${escapeHtml(lock.reason || '')}${origin}</div>`;
                }).join('');
                const identLines = idents.map((row) => `<div class="text-gray-300">IP・端末のBAN: ${escapeHtml(SOURCE_LABELS[row.kind] || row.kind)} ${escapeHtml(row.identifier)}${row.kind === 'cookie' ? '…' : ''}・${escapeHtml(row.reason || '-')}（${escapeHtml(formatTime(row.created_at))}）</div>`).join('');
                const identNote = idents.length ? '<div class="text-gray-400">IP・端末のBANがあると、同じIPアドレスや端末からは新しいアカウントを作れません。</div>' : '';
                const verified = st.turnstile_verified_seconds > 0
                    ? `確認済み（残り${formatDuration(st.turnstile_verified_seconds)}）`
                    : '未確認';
                const appeals = st.appeal_count ? `・異議申し立て ${escapeHtml(String(st.appeal_count))}件` : '';
                const actions = [];
                if (locks.length) actions.push('<button class="bot-log-unlock bg-yellow-600 hover:bg-yellow-500 text-white px-2 py-1 rounded">ロックを解除</button>');
                if (idents.length) actions.push('<button class="bot-log-unblock bg-red-600 hover:bg-red-500 text-white px-2 py-1 rounded">IP・端末のBANを解除</button>');
                return `
                    <div class="bg-gray-900 border border-gray-700 rounded p-2 text-xs space-y-1">
                        <div class="flex flex-wrap items-center gap-1">${badges.join('')}</div>
                        ${banLine}
                        ${lockLines}
                        ${identLines}
                        ${identNote}
                        <div class="text-gray-400">ロック回数（1時間）: ${escapeHtml(String(st.lock_count))} / ${escapeHtml(String(st.lock_count_limit))}・Turnstile: ${escapeHtml(verified)}・失敗 ${escapeHtml(String(st.turnstile_fail_count))} / ${escapeHtml(String(st.turnstile_fail_limit))}・判定スコア ${escapeHtml(String(Math.round((st.score || 0) * 10) / 10))}（操作 ${escapeHtml(String(Math.round((st.behavior_score || 0) * 10) / 10))}）${appeals}</div>
                        ${actions.length ? `<div class="flex flex-wrap gap-2 pt-1">${actions.join('')}</div>` : ''}
                    </div>`;
            };

            const renderFilters = () => {
                const chip = (type, label, count) => {
                    const active = state.type === type;
                    return `<button class="bot-log-filter ${active ? 'bg-blue-600 hover:bg-blue-500' : 'bg-gray-700 hover:bg-gray-600'} text-white px-2 py-1 rounded" data-type="${escapeHtml(type)}">${escapeHtml(label)} (${count})</button>`;
                };
                const chips = [chip('', 'すべて', state.total)];
                Object.keys(state.counts).sort((a, b) => state.counts[b] - state.counts[a]).forEach((type) => {
                    chips.push(chip(type, (EVENT_LABELS[type] || { label: type }).label, state.counts[type]));
                });
                return chips.join('');
            };

            const renderItem = (item) => {
                const ev = EVENT_LABELS[item.event_type] || { label: item.event_type, cls: 'bg-gray-600' };
                const client = item.client ? `<span class="bg-gray-700 text-gray-200 px-1.5 py-0.5 rounded">${escapeHtml(CLIENT_LABELS[item.client] || item.client)}</span>` : '';
                const checked = state.selected.has(item.id) ? 'checked' : '';
                const score = (item.score !== null && item.score !== undefined)
                    ? `<span class="text-gray-400">スコア ${escapeHtml(String(item.score))}${item.behavior_score !== null && item.behavior_score !== undefined ? `（操作 ${escapeHtml(String(item.behavior_score))}）` : ''}</span>`
                    : '';
                const reasons = item.reasons ? `<div class="mt-1 text-gray-300 bot-log-wrap">理由: ${escapeHtml(item.reasons)}</div>` : '';
                const source = (item.ip_address || item.user_agent)
                    ? `<div class="mt-1 text-[10px] text-gray-500 bot-log-wrap">${item.ip_address ? 'IP ' + escapeHtml(item.ip_address) : ''}${item.ip_address && item.user_agent ? '・' : ''}${escapeHtml(item.user_agent || '')}</div>`
                    : '';
                return `
                    <div class="bg-gray-900 border border-gray-700 rounded p-2 text-xs" data-log-id="${item.id}">
                        <div class="flex flex-wrap items-center gap-1">
                            <input type="checkbox" class="bot-log-select" data-log-id="${item.id}" ${checked} aria-label="この記録を選択">
                            <span class="text-gray-400">${escapeHtml(formatTime(item.created_at))}</span>
                            <span class="${ev.cls} text-white px-1.5 py-0.5 rounded">${escapeHtml(ev.label)}</span>
                            ${client}
                            ${score}
                            <button class="bot-log-delete-one ml-auto text-gray-400 hover:text-white px-1" data-log-id="${item.id}" title="この記録を削除" aria-label="この記録を削除"><i class="fas fa-trash"></i></button>
                        </div>
                        ${reasons}
                        ${renderDetails(item.details)}
                        ${source}
                    </div>`;
            };

            const render = () => {
                const el = view();
                if (!el) return;
                const items = state.items.length
                    ? state.items.map(renderItem).join('')
                    : '<div class="text-xs text-gray-400 py-2">記録はありません。</div>';
                const allChecked = state.items.length && state.items.every(item => state.selected.has(item.id));
                el.innerHTML = `
                    <div class="flex flex-wrap items-center gap-2 mb-2">
                        <button class="bot-log-back bg-gray-700 hover:bg-gray-600 text-white px-2 py-1 rounded text-xs"><i class="fas fa-arrow-left mr-1"></i>一覧に戻る</button>
                        <div class="text-sm font-bold text-white bot-log-wrap">${escapeHtml(state.username || ('ID ' + state.userId))}</div>
                        <button class="bot-log-reload ml-auto bg-gray-700 hover:bg-gray-600 text-white px-2 py-1 rounded text-xs">更新</button>
                    </div>
                    <div class="flex-1 overflow-y-auto space-y-2">
                        ${renderState(state.data && state.data.state)}
                        <div class="flex flex-wrap gap-1 text-xs">${renderFilters()}</div>
                        <div class="flex flex-wrap items-center gap-2 text-xs">
                            <label class="flex items-center gap-1 text-gray-300"><input type="checkbox" class="bot-log-select-all" aria-label="表示中の記録をすべて選択" ${allChecked ? 'checked' : ''}>表示中をすべて選択</label>
                            <button class="bot-log-delete-selected bg-red-600 hover:bg-red-500 text-white px-2 py-1 rounded" ${state.selected.size ? '' : 'disabled'}>選択した記録を削除${state.selected.size ? `（${state.selected.size}件）` : ''}</button>
                            <button class="bot-log-delete-all bg-red-800 hover:bg-red-700 text-white px-2 py-1 rounded" ${state.total ? '' : 'disabled'}>すべての記録を削除</button>
                        </div>
                        <div class="space-y-2">${items}</div>
                        ${state.hasMore ? '<button class="bot-log-more w-full bg-gray-700 hover:bg-gray-600 text-white px-2 py-1.5 rounded text-xs">さらに読み込む</button>' : ''}
                    </div>`;
            };

            const load = async (append = false) => {
                const el = view();
                if (!el || !state.userId) return;
                if (!append) {
                    el.innerHTML = '<div class="text-xs text-gray-400 py-2"><i class="fas fa-spinner fa-spin mr-1"></i>読み込み中...</div>';
                }
                const params = new URLSearchParams({ user_id: String(state.userId), limit: String(PAGE_SIZE) });
                if (state.type) params.set('type', state.type);
                if (append && state.items.length) params.set('before_id', String(state.items[state.items.length - 1].id));
                try {
                    const res = await apiFetch(`/api/bot/evidence?${params.toString()}`, { cache: 'no-store' });
                    const data = await res.json().catch(() => ({}));
                    if (!res.ok) throw new Error(data.error || String(res.status));
                    state.counts = data.counts || {};
                    state.total = data.total || 0;
                    state.hasMore = !!data.has_more;
                    if (data.user) {
                        state.username = data.user.username || state.username;
                        state.exists = !!data.user.exists;
                    }
                    if (append) {
                        state.items = state.items.concat(data.items || []);
                    } else {
                        state.items = data.items || [];
                        state.data = data;
                    }
                    render();
                } catch (err) {
                    if (!append) el.innerHTML = '<div class="text-xs text-red-400">記録の取得に失敗しました。</div>';
                    showToast('ボット検出の記録を取得できませんでした', 'error', true);
                }
            };

            const deleteLogs = async (payload, confirmText) => {
                if (!confirm(confirmText)) return;
                try {
                    const data = await postJson('/api/bot/evidence/delete', Object.assign({ user_id: state.userId }, payload));
                    showToast(`${data.deleted || 0}件の記録を削除しました`, 'success');
                    state.selected.clear();
                    await load(false);
                } catch (err) {
                    showToast('記録を削除できませんでした', 'error', true);
                }
            };

            const runAccountAction = async (url, confirmText, doneText) => {
                if (!confirm(confirmText)) return;
                try {
                    await postJson(url, { user_id: state.userId });
                    showToast(doneText, 'success');
                    await load(false);
                } catch (err) {
                    showToast('操作に失敗しました', 'error', true);
                }
            };

            const showList = () => {
                const el = view();
                if (el) { el.classList.add('hidden'); el.classList.remove('flex'); el.innerHTML = ''; }
                const lv = listView();
                if (lv) { lv.classList.remove('hidden'); lv.classList.add('flex'); }
                state.userId = 0;
            };

            const openList = async () => {
                showList();
                await loadAccounts();
            };

            const open = async (userId, username = '') => {
                const el = view();
                if (!el) return;
                Object.assign(state, { userId: Number(userId) || 0, username, exists: false, type: '', items: [], hasMore: false, counts: {}, total: 0, data: null });
                state.selected.clear();
                const lv = listView();
                if (lv) { lv.classList.add('hidden'); lv.classList.remove('flex'); }
                el.classList.remove('hidden');
                el.classList.add('flex');
                await load(false);
            };

            const bind = () => {
                const modal = get('bot-log-modal');
                if (!modal || modal.dataset.bound === '1') return;
                modal.dataset.bound = '1';
                const search = get('bot-log-search');
                if (search) search.addEventListener('keydown', (e) => { if (e.key === 'Enter') loadAccounts(); });
                modal.addEventListener('change', (e) => {
                    const target = e.target;
                    if (!target || !target.classList) return;
                    if (target.classList.contains('bot-log-select')) {
                        const id = Number(target.getAttribute('data-log-id'));
                        if (target.checked) state.selected.add(id); else state.selected.delete(id);
                        render();
                    } else if (target.classList.contains('bot-log-select-all')) {
                        state.items.forEach((item) => { if (target.checked) state.selected.add(item.id); else state.selected.delete(item.id); });
                        render();
                    } else if (target.classList.contains('bot-log-account-select')) {
                        const id = Number(target.getAttribute('data-user-id'));
                        if (target.checked) listState.selected.add(id); else listState.selected.delete(id);
                        renderAccounts();
                    } else if (target.classList.contains('bot-log-account-select-all')) {
                        listState.accounts.forEach((a) => { if (target.checked) listState.selected.add(a.user_id); else listState.selected.delete(a.user_id); });
                        renderAccounts();
                    }
                });
                modal.addEventListener('click', async (e) => {
                    const btn = e.target.closest('button');
                    if (!btn || btn.disabled) return;
                    if (btn.id === 'bot-log-search-btn' || btn.id === 'bot-log-refresh-btn') {
                        if (btn.id === 'bot-log-refresh-btn' && search) search.value = '';
                        await loadAccounts();
                    } else if (btn.classList.contains('bot-log-open-account')) {
                        const id = Number(btn.getAttribute('data-user-id'));
                        if (id) await open(id, btn.getAttribute('data-username') || '');
                    } else if (btn.classList.contains('bot-log-delete-accounts')) {
                        const ids = Array.from(listState.selected);
                        if (!ids.length || !confirm(`選択した${ids.length}件のアカウントの記録をすべて削除しますか？この操作は取り消せません。`)) return;
                        try {
                            const data = await postJson('/api/bot/evidence/delete', { user_ids: ids });
                            showToast(`${data.deleted || 0}件の記録を削除しました`, 'success');
                            listState.selected.clear();
                            await loadAccounts();
                        } catch (err) {
                            showToast('記録を削除できませんでした', 'error', true);
                        }
                    } else if (btn.classList.contains('bot-log-back')) {
                        await openList();
                    } else if (btn.classList.contains('bot-log-reload')) {
                        await load(false);
                    } else if (btn.classList.contains('bot-log-more')) {
                        await load(true);
                    } else if (btn.classList.contains('bot-log-filter')) {
                        state.type = btn.getAttribute('data-type') || '';
                        state.selected.clear();
                        await load(false);
                    } else if (btn.classList.contains('bot-log-delete-one')) {
                        const id = Number(btn.getAttribute('data-log-id'));
                        if (id) await deleteLogs({ ids: [id] }, 'この記録を削除しますか？この操作は取り消せません。');
                    } else if (btn.classList.contains('bot-log-delete-selected')) {
                        if (state.selected.size) await deleteLogs({ ids: Array.from(state.selected) }, `選択した${state.selected.size}件の記録を削除しますか？この操作は取り消せません。`);
                    } else if (btn.classList.contains('bot-log-delete-all')) {
                        await deleteLogs({ all: true }, `${state.username || 'このアカウント'} のボット検出の記録をすべて（${state.total}件）削除しますか？この操作は取り消せません。`);
                    } else if (btn.classList.contains('bot-log-unlock')) {
                        await runAccountAction('/api/bot/account/clear-lock',
                            `${state.username || 'このアカウント'} のロックを解除しますか？\nこのアカウントのIPアドレスと端末にかかっているロックも解除します。`,
                            'ロックを解除しました');
                    } else if (btn.classList.contains('bot-log-unblock')) {
                        await runAccountAction('/api/bot/account/clear-identifiers',
                            `${state.username || 'このアカウント'} が起点のIP・端末のBANを解除しますか？\n解除すると、そのIPアドレスや端末から新しいアカウントを作れるようになります。`,
                            'IP・端末のBANを解除しました');
                    }
                });
            };

            return { open, openList, showList, bind };
        })();
        window.BotAdminLog = BotAdminLog;
        // A branch is a snapshot of every route up to the chosen message.
        let branchCreating = false;
        const branchRequests = new Map();
        const BRANCH_SAVE_WARNING = '分岐先は通常のチャットとして保存され、自動削除されません。コピー対象の添付・生成ファイルも保存対象になります。';

        function confirmBranchSave() {
            return new Promise((resolve) => {
                const dialog = document.createElement('dialog');
                dialog.className = 'rounded-xl bg-gray-800 text-gray-100 p-6 max-w-lg border border-gray-600';
                dialog.setAttribute('aria-label', '新しいチャットに分岐');
                const message = document.createElement('p');
                message.textContent = BRANCH_SAVE_WARNING;
                dialog.appendChild(message);
                const form = document.createElement('form');
                form.method = 'dialog';
                form.className = 'flex justify-end gap-3 mt-4';
                for (const [value, label] of [['cancel', 'キャンセル'], ['save', '保存して分岐']]) {
                    const button = document.createElement('button');
                    button.value = value;
                    button.textContent = label;
                    button.className = 'rounded px-3 py-2 border border-gray-600';
                    form.appendChild(button);
                }
                dialog.appendChild(form);
                dialog.addEventListener('close', () => { const saved = dialog.returnValue === 'save'; dialog.remove(); resolve(saved); }, { once: true });
                document.body.appendChild(dialog);
                dialog.showModal();
            });
        }

        async function branchChatMessage(...args) {
            if (!window.ActivityLog) return branchChatMessageLogged.apply(this, args);
            return window.ActivityLog.operation('branch', () => branchChatMessageLogged.apply(this, args));
        }
        async function branchChatMessageLogged(id, button) {
            if (branchCreating || abortController || !currentThreadId || !/^\d+$/.test(String(id))) return;
            if (temporaryChatEnabled && !await confirmBranchSave()) return;
            const source = String(currentThreadId);
            const requestKey = source + ':' + id;
            if (!branchRequests.has(requestKey)) branchRequests.set(requestKey, crypto.randomUUID().replace(/-/g, ''));
            branchCreating = true;
            if (button) button.disabled = true;
            try {
                const response = await apiFetch(`/api/threads/${encodeURIComponent(source)}/branch`, {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message_id: Number(id), request_id: branchRequests.get(requestKey),
                        confirm_persist_temporary: !!temporaryChatEnabled })
                });
                const data = await response.json();
                if (!response.ok) throw new Error(data.error || '分岐を作成できませんでした');
                branchRequests.delete(requestKey);
                await loadMessages(data.id, { targetMessageId: data.leaf_id });
                await loadThreads();
                showToast('新しいチャットに分岐しました');
            } catch (error) {
                window.ActivityLog?.log('branch.error');
                showToast(error.message || '分岐を作成できませんでした', 'error', true);
            } finally {
                branchCreating = false;
                if (button) button.disabled = false;
            }
        }

        function renderBranchOrigin(origin) {
            let banner = get('branch-origin-banner');
            if (!banner) {
                const chat = get('chat-container');
                if (!chat) return;
                banner = document.createElement('div');
                banner.id = 'branch-origin-banner';
                banner.className = 'px-4 py-2 text-xs text-gray-400 border-b border-gray-700';
                const totals = get('total-token-bar');
                if (totals) totals.before(banner);
                else chat.parentNode.insertBefore(banner, chat);
            }
            banner.replaceChildren();
            banner.hidden = !origin;
            if (!origin) return;
            if (origin.available && origin.thread_id && origin.message_id) {
                const link = document.createElement('a');
                link.textContent = '分岐元のチャットを開く';
                link.className = 'text-blue-400 hover:underline';
                link.href = '/c/' + encodeURIComponent(origin.thread_id) + '?message=' + Number(origin.message_id);
                link.onclick = (event) => {
                    event.preventDefault();
                    loadMessages(origin.thread_id, { targetMessageId: origin.message_id });
                };
                banner.appendChild(link);
            } else {
                banner.textContent = '分岐元のチャットは利用できません';
            }
        }

        function displayMissingMessageFiles(container, messages) {
            const normalize = (value) => {
                try { const url = new URL(value, location.origin); return url.origin === location.origin ? decodeURIComponent(url.pathname).replace(/^\/files\/(thumb\/)?/, '') : ''; }
                catch (_) { return String(value); }
            };
            for (const message of messages) {
                if (!message.missing_files || !message.missing_files.length) continue;
                const element = container.querySelector('#msg-' + Number(message.id));
                if (!element) continue;
                const missing = new Set(message.missing_files.flatMap((ref) => { const value = String(ref).replace(/^\/files\//, ''); return [value, value.replace(/^\d+\//, '')]; }));
                element.querySelectorAll('img[src], a[href], audio[src], video[src], source[src], .file-thumb[onclick]').forEach((node) => {
                    let ref = normalize(node.getAttribute('src') || node.getAttribute('href') || '');
                    if (node.classList.contains('file-thumb')) {
                        const match = (node.getAttribute('onclick') || '').match(/window.open\('([^']+)'/);
                        if (match) ref = normalize(match[1]);
                    }
                    if (!missing.has(ref)) return;
                    const warning = document.createElement('span');
                    warning.className = 'inline-block rounded bg-gray-800 p-3 text-xs text-gray-400';
                    warning.textContent = 'ファイルがありません';
                    warning.title = ref.split('/').pop();
                    const target = node.closest('.chat-image-frame') || (node.tagName === 'SOURCE' ? node.parentNode : node);
                    if (target && element.contains(target)) target.replaceWith(warning);
                });
            }
        }
        // Image splitting uses the Android grid rules. Cropping itself is lossless;
        // the shared compression pipeline is applied once, with its own saved settings.
        const SPLIT_COMPRESSION_KEY = 'image_split_compression';
        const splitState = { files: [], index: 0, image: null, release: null, revision: 0, busy: false, urls: [] };
        function splitClamp(value, min, max, fallback) {
            const n = Number.parseInt(value, 10);
            return Number.isFinite(n) ? Math.max(min, Math.min(max, n)) : fallback;
        }
        function chooseSplitGrid(pieces, width, height) {
            const count = splitClamp(pieces, 2, 30, 4);
            let best = { columns: count, rows: 1 }, bestScore = Infinity;
            for (let columns = 1; columns <= count; columns++) {
                if (count % columns) continue;
                const rows = count / columns;
                const score = Math.abs(Math.log((Math.max(1, width) / columns) / (Math.max(1, height) / rows)));
                if (score < bestScore - 1e-9 || (Math.abs(score - bestScore) <= 1e-9 &&
                    (width >= height ? columns > best.columns : rows > best.rows))) {
                    best = { columns, rows }; bestScore = score;
                }
            }
            return best;
        }
        function splitTiles(width, height, grid, overlapPct) {
            const overlap = splitClamp(overlapPct, 0, 50, 10) / 100;
            const padX = Math.round(width / grid.columns * overlap / 2);
            const padY = Math.round(height / grid.rows * overlap / 2);
            const tiles = [];
            for (let row = 0; row < grid.rows; row++) for (let column = 0; column < grid.columns; column++) {
                const cellLeft = Math.floor(column * width / grid.columns), cellRight = Math.floor((column + 1) * width / grid.columns);
                const cellTop = Math.floor(row * height / grid.rows), cellBottom = Math.floor((row + 1) * height / grid.rows);
                tiles.push({ index: tiles.length, row, column, cellLeft, cellTop, cellRight, cellBottom,
                    left: Math.max(0, cellLeft - padX), top: Math.max(0, cellTop - padY),
                    right: Math.min(width, cellRight + padX), bottom: Math.min(height, cellBottom + padY) });
            }
            return tiles;
        }
        function splitFits(grid, width, height) {
            return Math.floor(width / grid.columns) >= 16 && Math.floor(height / grid.rows) >= 16;
        }
        function splitTileCaption(tile, grid, width, height) {
            return `Tile ${tile.index + 1}/${grid.columns * grid.rows}  row ${tile.row + 1}/${grid.rows}, col ${tile.column + 1}/${grid.columns}  x ${tile.left}-${tile.right}, y ${tile.top}-${tile.bottom} of ${width}x${height}`;
        }
        function splitBaseName(name) {
            return (String(name).split('/').pop().replace(/\.[^.]*$/, '').replace(/[\\/:*?"<>|\s]+/g, '_')
                .replace(/^[_.]+|[_.]+$/g, '').slice(0, 80)) || 'image';
        }
        function splitTileName(base, tile, grid) {
            const count = grid.columns * grid.rows;
            return `${base}_part${String(tile.index + 1).padStart(String(count).length, '0')}of${count}_r${tile.row + 1}c${tile.column + 1}.png`;
        }
        function readSplitCompression() {
            let saved;
            try { saved = JSON.parse(localStorage.getItem(SPLIT_COMPRESSION_KEY) || '{}'); } catch (e) {}
            saved = saved && typeof saved === 'object' ? saved : {};
            return { enabled: saved.enabled !== false, maxSizeMB: Math.max(0.1, Math.min(10, Number(saved.maxSizeMB) || 1)),
                maxDimension: splitClamp(saved.maxDimension, 256, 8192, 1920),
                outputType: ['original', 'image/jpeg', 'image/png', 'image/webp'].includes(saved.outputType) ? saved.outputType : 'original',
                formatOnly: saved.formatOnly === true };
        }
        function splitOptions() {
            return { pieces: splitClamp(get('image-split-pieces').value, 2, 30, 4),
                overlap: splitClamp(get('image-split-overlap').value, 0, 50, 10),
                includeOriginal: get('image-split-original').checked,
                markOriginal: get('image-split-mark-original').checked,
                markPieces: get('image-split-mark-pieces').checked };
        }
        function splitCanvas(width, height) {
            const canvas = document.createElement('canvas'); canvas.width = width; canvas.height = height;
            if (canvas.width !== width || canvas.height !== height || !canvas.getContext('2d')) throw new Error('画像が大きすぎるため処理できません。');
            return canvas;
        }
        async function loadSplitImage(file) {
            // HTMLImageElement applies EXIF orientation. Never request a reduced output size.
            const url = URL.createObjectURL(file), img = new Image();
            try {
                await new Promise((resolve, reject) => { img.onload = resolve; img.onerror = () => reject(new Error(`画像を読み込めません: ${file.name}`)); img.src = url; });
                return { image: img, release: () => { img.src = ''; URL.revokeObjectURL(url); } };
            } catch (e) { URL.revokeObjectURL(url); throw e; }
        }
        function drawSplitOverview(canvas, image, width, height, grid, tiles) {
            const ctx = canvas.getContext('2d'), sx = canvas.width / width, sy = canvas.height / height;
            ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
            const first = tiles[0], padX = first.right - first.cellRight, padY = first.bottom - first.cellBottom;
            ctx.fillStyle = 'rgba(250,204,21,0.18)';
            const lines = [];
            for (let col = 1; col < grid.columns; col++) {
                const x = Math.floor(col * width / grid.columns);
                ctx.fillRect((x - padX) * sx, 0, 2 * padX * sx, canvas.height); lines.push([x * sx, 0, x * sx, canvas.height]);
            }
            for (let row = 1; row < grid.rows; row++) {
                const y = Math.floor(row * height / grid.rows);
                ctx.fillRect(0, (y - padY) * sy, canvas.width, 2 * padY * sy); lines.push([0, y * sy, canvas.width, y * sy]);
            }
            drawSplitLines(ctx, lines, Math.max(2, Math.min(5, Math.max(canvas.width, canvas.height) / 400)), false);
            const size = Math.max(14, Math.min(64, Math.min(canvas.width / grid.columns, canvas.height / grid.rows) * 0.18));
            ctx.font = `bold ${size}px monospace`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
            for (const tile of tiles) {
                const x = (tile.cellLeft + tile.cellRight) / 2 * sx, y = (tile.cellTop + tile.cellBottom) / 2 * sy;
                ctx.lineWidth = Math.max(2, size / 10); ctx.strokeStyle = 'rgba(0,0,0,0.7)'; ctx.strokeText(String(tile.index + 1), x, y);
                ctx.fillStyle = '#facc15'; ctx.fillText(String(tile.index + 1), x, y);
            }
        }
        function drawSplitLines(ctx, lines, stroke, dashed) {
            ctx.save(); ctx.lineWidth = stroke; ctx.strokeStyle = 'rgba(250,204,21,0.6)';
            ctx.shadowColor = 'rgba(0,0,0,0.7)'; ctx.shadowBlur = 2;
            if (dashed) ctx.setLineDash([stroke * 4, stroke * 3]);
            ctx.beginPath(); lines.forEach(([x1,y1,x2,y2]) => { ctx.moveTo(x1,y1); ctx.lineTo(x2,y2); }); ctx.stroke(); ctx.restore();
        }
        async function splitCanvasFile(canvas, name) {
            try {
                const blob = await new Promise((resolve, reject) => canvas.toBlob(b => b ? resolve(b) : reject(new Error('画像を書き出せません。')), 'image/png'));
                return new File([blob], name, { type: 'image/png' });
            } finally { canvas.width = 0; canvas.height = 0; }
        }
        async function renderSplitFiles(file, options, compression) {
            const loaded = await loadSplitImage(file), image = loaded.image;
            try {
                const width = image.naturalWidth, height = image.naturalHeight;
                const grid = chooseSplitGrid(options.pieces, width, height);
                if (!splitFits(grid, width, height)) throw new Error(`画像が小さすぎるため${options.pieces}枚に分割できません: ${file.name}`);
                const tiles = splitTiles(width, height, grid, options.overlap), base = splitBaseName(file.name), outputs = [];
                const append = async output => { outputs.push(await applyImageCompression(output, compression)); };
                if (options.includeOriginal) {
                    if (options.markOriginal) {
                        const canvas = splitCanvas(width, height); drawSplitOverview(canvas, image, width, height, grid, tiles);
                        await append(await splitCanvasFile(canvas, `${base}_split_grid.png`));
                    } else await append(file);
                }
                for (const tile of tiles) {
                    const w = tile.right - tile.left, h = tile.bottom - tile.top;
                    let fontSize = Math.max(12, Math.min(40, Math.max(w, h) / 45)), header = 0;
                    const caption = splitTileCaption(tile, grid, width, height);
                    if (options.markPieces) {
                        const measure = splitCanvas(1,1), ctx = measure.getContext('2d'); ctx.font = `bold ${fontSize}px monospace`;
                        fontSize = Math.max(6, fontSize * Math.min(1, w * 0.96 / ctx.measureText(caption).width));
                        header = Math.max(12, Math.floor(fontSize * 1.8)); measure.width = 0;
                    }
                    const canvas = splitCanvas(w, h + header), ctx = canvas.getContext('2d');
                    if (header) { ctx.fillStyle = '#111827'; ctx.fillRect(0,0,w,header); }
                    ctx.drawImage(image, tile.left, tile.top, w, h, 0, header, w, h);
                    if (options.markPieces) {
                        ctx.fillStyle = '#fff'; ctx.font = `bold ${fontSize}px monospace`; ctx.textBaseline = 'middle'; ctx.fillText(caption, w * 0.02, header / 2);
                        const lines = [];
                        for (const x of [tile.cellLeft, tile.cellRight]) if (x > tile.left && x < tile.right) lines.push([x-tile.left,header,x-tile.left,header+h]);
                        for (const y of [tile.cellTop, tile.cellBottom]) if (y > tile.top && y < tile.bottom) lines.push([0,header+y-tile.top,w,header+y-tile.top]);
                        drawSplitLines(ctx, lines, Math.max(1.5, Math.min(4, Math.max(w,h)/500)), true);
                    }
                    await append(await splitCanvasFile(canvas, splitTileName(base, tile, grid)));
                }
                return outputs;
            } finally { loaded.release(); }
        }
        function clearSplitDownloads() {
            splitState.urls.forEach(url => URL.revokeObjectURL(url)); splitState.urls = [];
            get('image-split-downloads').replaceChildren();
        }
        function closeImageSplit() {
            if (!get('image-split-modal') || splitState.busy) return;
            splitState.revision++; if (splitState.release) splitState.release();
            splitState.release = null; splitState.image = null; splitState.files = [];
            clearSplitDownloads(); get('image-split-preview').width = 0; hideModal('image-split-compression-modal'); hideModal('image-split-modal');
        }
        async function openImageSplit(files) {
            const images = Array.from(files).filter(Boolean);
            if (!images.length) return;
            if (images.length > 10) showToast('画像の分割は一度に10枚までです。先頭10枚のみ使います。', 'warning', true);
            if (splitState.busy) return;
            closeImageSplit(); splitState.files = images.slice(0,10); splitState.index = 0;
            showModal('image-split-modal'); await loadSplitPreview();
        }
        async function loadSplitPreview() {
            const revision = ++splitState.revision;
            if (splitState.release) splitState.release(); splitState.release = null; splitState.image = null;
            get('image-split-status').textContent = '読み込み中...';
            try {
                const loaded = await loadSplitImage(splitState.files[splitState.index]);
                if (revision !== splitState.revision) { loaded.release(); return; }
                splitState.image = loaded.image; splitState.release = loaded.release;
                get('image-split-status').textContent = ''; updateSplitPreview();
            } catch (e) { if (revision === splitState.revision) { get('image-split-status').textContent = e.message; updateSplitPreview(); } }
        }
        function updateSplitPreview() {
            const options = splitOptions(), image = splitState.image, canvas = get('image-split-preview');
            get('image-split-mark-original').disabled = !options.includeOriginal || splitState.busy;
            get('image-split-page').textContent = `${splitState.index + 1} / ${splitState.files.length}（すべて同じ設定で分割します）`;
            get('image-split-prev').disabled = splitState.busy || splitState.index === 0;
            get('image-split-next').disabled = splitState.busy || splitState.index >= splitState.files.length - 1;
            get('image-split-count').textContent = `作成される画像: ${(options.pieces + Number(options.includeOriginal)) * splitState.files.length}枚`;
            let fits = false;
            if (image) {
                const w = image.naturalWidth, h = image.naturalHeight, grid = chooseSplitGrid(options.pieces,w,h); fits = splitFits(grid,w,h);
                const scale = Math.min(1, 800/w, 520/h); canvas.width = Math.round(w*scale); canvas.height = Math.round(h*scale);
                if (fits) drawSplitOverview(canvas,image,w,h,grid,splitTiles(w,h,grid,options.overlap));
                else canvas.getContext('2d').drawImage(image,0,0,canvas.width,canvas.height);
                get('image-split-dimensions').textContent = fits ? `${grid.columns}列 × ${grid.rows}行・1枚あたり約 ${Math.floor(w/grid.columns)}×${Math.floor(h/grid.rows)} px（元画像 ${w}×${h} px）` : `この画像は小さすぎるため${options.pieces}枚に分割できません。分割数を減らしてください。`;
            } else { canvas.width = 0; get('image-split-dimensions').textContent = ''; }
            get('image-split-save').disabled = get('image-split-attach').disabled = splitState.busy || !fits;
        }
        function setSplitBusy(busy) {
            splitState.busy = busy;
            get('image-split-modal').querySelectorAll('input,button').forEach(el => { el.disabled = busy; }); updateSplitPreview();
        }
        async function runImageSplit(...args) {
            if (!window.ActivityLog) return runImageSplitLogged.apply(this, args);
            return window.ActivityLog.operation('image_split', () => runImageSplitLogged.apply(this, args));
        }
        async function runImageSplitLogged(attach) {
            if (splitState.busy) return;
            const options = splitOptions(), compression = readSplitCompression();
            if (attach && browserFastModeEnabled && browserFastLocalFiles.size + splitState.files.length * (options.pieces + Number(options.includeOriginal)) > BROWSER_FAST_MAX_IMAGES) {
                get('image-split-status').textContent = '高速モードの画像は4枚・合計12MBまでです'; return;
            }
            setSplitBusy(true); clearSplitDownloads();
            try {
                const outputs = [];
                // Release the preview before decoding full-size sources, one at a time.
                if (splitState.release) splitState.release(); splitState.release = null; splitState.image = null;
                for (let index = 0; index < splitState.files.length; index++) {
                    get('image-split-status').textContent = `処理中... ${index+1} / ${splitState.files.length}`;
                    outputs.push(...await renderSplitFiles(splitState.files[index], options, compression));
                }
                if (attach) {
                    const existingBytes = Array.from(browserFastLocalFiles.values()).reduce((sum, entry) => sum + entry.file.size, 0);
                    if (browserFastModeEnabled && existingBytes + outputs.reduce((sum,f)=>sum+f.size,0) > BROWSER_FAST_MAX_BYTES) throw new Error('高速モードの画像は4枚・合計12MBまでです');
                    await handleFiles(outputs, { skipCompression: true });
                    setSplitBusy(false); closeImageSplit();
                } else {
                    for (const file of outputs) {
                        const url = URL.createObjectURL(file); splitState.urls.push(url);
                        const link = document.createElement('a'); link.href = url; link.download = file.name; link.textContent = file.name; link.className = 'block underline';
                        get('image-split-downloads').append(link); link.click();
                    }
                    get('image-split-status').textContent = `${outputs.length}枚のダウンロードを開始しました。保存されない画像は下のリンクから保存できます。`;
                }
            } catch (e) { window.ActivityLog?.log('image_split.error'); get('image-split-status').textContent = e.message || '画像を分割できませんでした。'; }
            finally {
                setSplitBusy(false);
                if (splitState.files.length) {
                    const status = get('image-split-status').textContent; await loadSplitPreview(); get('image-split-status').textContent = status;
                }
            }
        }
        function openSplitCompression() {
            const s = readSplitCompression(), prefix = 'image-split-compression';
            get(`${prefix}-enabled`).checked = s.enabled; get(`${prefix}-output-type`).value = s.outputType;
            get(`${prefix}-format-only`).checked = s.formatOnly; get(`${prefix}-max-size`).value = s.maxSizeMB; get(`${prefix}-max-dim`).value = s.maxDimension;
            syncSplitCompressionUi(); showModal(`${prefix}-modal`);
        }
        function syncSplitCompressionUi() {
            const prefix = 'image-split-compression';
            for (const field of ['size','dim']) {
                const off = get(`${prefix}-format-only`).checked;
                get(`${prefix}-max-${field}`).disabled = off; get(`${prefix}-${field}-wrap`).style.opacity = off ? '0.4' : '1';
            }
        }
        window.addEventListener('popstate', () => { if (!splitState.busy) closeImageSplit(); });
        document.addEventListener('keydown', event => {
            if (event.key !== 'Escape' || !get('image-split-modal')) return;
            if (!get('image-split-compression-modal').classList.contains('hidden')) hideModal('image-split-compression-modal');
            else if (!get('image-split-modal').classList.contains('hidden')) closeImageSplit();
        });
        document.addEventListener('DOMContentLoaded', () => {
            if (!get('image-split-modal')) return;
            get('upload-split-btn').onclick = () => get('image-split-input').click();
            get('image-split-input').onchange = e => { openImageSplit(e.target.files); e.target.value = ''; };
            get('image-split-prev').onclick = () => { splitState.index--; loadSplitPreview(); };
            get('image-split-next').onclick = () => { splitState.index++; loadSplitPreview(); };
            for (const field of ['pieces','overlap','original','mark-original','mark-pieces']) get(`image-split-${field}`).oninput = updateSplitPreview;
            get('image-split-cancel').onclick = closeImageSplit;
            get('image-split-save').onclick = () => runImageSplit(false); get('image-split-attach').onclick = () => runImageSplit(true);
            get('image-split-settings').onclick = openSplitCompression;
            get('image-split-compression-enabled').onchange = get('image-split-compression-format-only').onchange = syncSplitCompressionUi;
            get('image-split-compression-close').onclick = () => hideModal('image-split-compression-modal');
            get('image-split-compression-save').onclick = () => {
                const p = 'image-split-compression';
                const size = get(`${p}-max-size`), dim = get(`${p}-max-dim`);
                if (!size.reportValidity() || !dim.reportValidity()) return;
                const settings = { enabled: get(`${p}-enabled`).checked, outputType: get(`${p}-output-type`).value,
                    formatOnly: get(`${p}-format-only`).checked, maxSizeMB: Number(size.value), maxDimension: Number(dim.value) };
                localStorage.setItem(SPLIT_COMPRESSION_KEY, JSON.stringify(settings)); hideModal(`${p}-modal`); showToast('設定を保存しました', 'success');
            };
        });
        // Presentation-only state: never persisted, sent, or replayed by history loading.
        function takePromptForSend(animate = true) {
            const input = get('prompt-input');
            const preview = get('upload-list');
            const dock = document.querySelector('.composer-dock');
            const oldHeight = dock?.getBoundingClientRect().height;
            const bounds = [input, get('file-preview'), get('quote-bar')]
                .filter((el) => el && el.getClientRects().length && getComputedStyle(el).display !== 'none')
                .map((el) => el.getBoundingClientRect());
            const left = Math.min(...bounds.map((r) => r.left)), top = Math.min(...bounds.map((r) => r.top));
            const source = { left, top, width: Math.max(...bounds.map((r) => r.right)) - left,
                height: Math.max(...bounds.map((r) => r.bottom)) - top };
            const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
            const saved = {
                navigation: threadLoadSequence, text: input.value, images: currentImageUrls, mask: currentMaskImage,
                quote: currentQuote, quoteId: currentQuoteMessageId,
                rows: preview ? Array.from(preview.childNodes) : [],
                files: new Map(browserFastLocalFiles), markers: new Set(markerAppliedUploads),
                progress: uploadProgressState,
            };
            // Local-file uploads treat disconnected rows as cancelled. Keep their DOM alive
            // outside the active composer until the fast-mode answer has been saved.
            const heldRows = document.createElement('div');
            heldRows.hidden = true;
            heldRows.className = 'prompt-send-held-files';
            heldRows.setAttribute('aria-hidden', 'true');
            heldRows.append(...saved.rows);
            document.body.appendChild(heldRows);
            let edited = false, settled = false, restored = false;
            const onInput = () => { edited = true; };
            input.addEventListener('input', onInput);
            input.value = '';
            ['file-input', 'photo-input', 'mask-input'].forEach((id) => { const el = get(id); if (el) el.value = ''; });
            input.style.height = 'auto';
            currentImageUrls = [];
            currentMaskImage = null;
            browserFastLocalFiles.clear();
            markerAppliedUploads.clear();
            uploadProgressState = { total: 0, completed: 0, active: 0, perFilePct: {} };
            if (preview) preview.replaceChildren();
            clearQuote();
            updateFilePreview();
            schedulePromptTokenEstimate(true);
            updateMaskPreview();
            let shrink;
            if (dock && animate && !reduced && dock.animate) {
                shrink = dock.animate([{ height: oldHeight + 'px' }, { height: dock.getBoundingClientRect().height + 'px' }],
                    { duration: 350, easing: 'cubic-bezier(0.22, 1, 0.36, 1)' });
            }
            const releaseFiles = () => {
                if (!settled || restored) return;
                saved.files.forEach((entry) => {
                    const url = entry?.rowObj?.row?.getAttribute('data-local-url');
                    if (url) URL.revokeObjectURL(url);
                });
                heldRows.remove();
            };
            const finish = (restore, release = true) => {
                if (settled) return;
                settled = true;
                restore = restore && saved.navigation === threadLoadSequence;
                restored = restore;
                input.removeEventListener('input', onInput);
                if (restore) {
                    if (!edited && !input.value) {
                        input.value = saved.text;
                        input.style.height = 'auto';
                        input.style.height = Math.min(input.scrollHeight, 150) + 'px';
                    }
                    currentImageUrls = Array.from(new Set([...saved.images, ...currentImageUrls]));
                    if (!currentMaskImage) currentMaskImage = saved.mask;
                    saved.files.forEach((value, key) => browserFastLocalFiles.set(key, value));
                    saved.markers.forEach((value) => markerAppliedUploads.add(value));
                    if (preview) preview.prepend(...saved.rows);
                    heldRows.remove();
                    if (!currentQuote && saved.quote) {
                        currentQuote = saved.quote;
                        currentQuoteMessageId = saved.quoteId;
                        renderQuoteBarContent(saved.quote);
                        get('quote-bar').classList.add('visible');
                    }
                    if (!uploadProgressState.total) uploadProgressState = saved.progress;
                    updateFilePreview();
                    updateMaskPreview();
                    schedulePromptTokenEstimate(true);
                } else {
                    if (release) releaseFiles();
                }
            };
            return { source, canRetry: () => !edited && saved.navigation === threadLoadSequence, stopDock: () => shrink?.cancel(), files: saved.files, releaseFiles, accept: (release = true) => finish(false, release), restore: () => finish(true) };
        }

        function animatePromptToMessage(row, transaction, follow) {
            let overlay, frame = 0, closed = false;
            const animations = [];
            const container = get('chat-container');
            const input = get('prompt-input');
            const reduce = window.matchMedia('(prefers-reduced-motion: reduce)');
            const stop = () => {
                if (closed) return;
                closed = true;
                window.ActivityLog?.log('ui.hide');
                transaction.stopDock();
                cancelAnimationFrame(frame);
                animations.forEach((animation) => animation.cancel());
                overlay?.remove();
                row?.classList.remove('prompt-send-hidden');
                window.removeEventListener('resize', stop);
                window.removeEventListener('popstate', stop);
                window.removeEventListener('pagehide', stop);
                document.removeEventListener('visibilitychange', stop);
                container?.removeEventListener('wheel', stop);
                container?.removeEventListener('touchstart', stop);
                input?.removeEventListener('input', stop);
                window.visualViewport?.removeEventListener('resize', stop);
                reduce.removeEventListener('change', stop);
            };
            if (!follow || reduce.matches || !row || !Element.prototype.animate) return stop;
            try {
                row.classList.remove('fade-in');
                row.classList.add('prompt-send-hidden');
                frame = requestAnimationFrame(() => {
                    try {
                        if (!row.isConnected) return stop();
                        const bubble = row.querySelector('.message-bubble');
                        const target = bubble.getBoundingClientRect();
                        const viewport = container.getBoundingClientRect();
                        if (!target.width || !target.height || target.bottom <= viewport.top || target.top >= viewport.bottom) return stop();
                        const source = transaction.source;
                        overlay = bubble.cloneNode(true);
                        overlay.removeAttribute('id');
                        overlay.querySelectorAll('[id]').forEach((el) => el.removeAttribute('id'));
                        overlay.querySelectorAll('.msg-controls, video, audio').forEach((el) => el.remove());
                        overlay.classList.add('prompt-send-overlay');
                        overlay.setAttribute('aria-hidden', 'true');
                        overlay.inert = true;
                        Object.assign(overlay.style, { width: target.width + 'px', height: target.height + 'px', left: target.left + 'px', top: target.top + 'px' });
                        document.body.appendChild(overlay);
                        window.ActivityLog?.log('ui.show');
                        const scale = Math.min(source.width / target.width, Math.max(source.height, 42) / target.height, 1);
                        const dx = source.left + (source.width - target.width * scale) / 2 - target.left;
                        const dy = source.top + (source.height - target.height * scale) / 2 - target.top;
                        const motion = overlay.animate([
                            { transform: `translate(${dx}px, ${dy}px) scale(${scale})`, borderRadius: '12px', backgroundColor: getComputedStyle(get('input-row')).backgroundColor },
                            { transform: 'translate(0, 0) scale(1)', borderRadius: getComputedStyle(bubble).borderRadius, backgroundColor: getComputedStyle(bubble).backgroundColor },
                        ], { duration: 350, easing: 'cubic-bezier(0.22, 1, 0.36, 1)', fill: 'both' });
                        animations.push(motion);
                        animations.push(overlay.animate([{ opacity: 1, offset: 0 }, { opacity: 1, offset: 0.8 }, { opacity: 0 }], { duration: 350, fill: 'both' }));
                        row.classList.remove('prompt-send-hidden');
                        animations.push(bubble.animate([{ opacity: 0, offset: 0 }, { opacity: 0, offset: 0.8 }, { opacity: 1 }], { duration: 350, fill: 'both' }));
                        const watch = () => {
                            if (!row.isConnected || !userAutoScroll || document.hidden) return stop();
                            const now = bubble.getBoundingClientRect();
                            overlay.style.left = now.left + 'px';
                            overlay.style.top = now.top + 'px';
                            frame = requestAnimationFrame(watch);
                        };
                        frame = requestAnimationFrame(watch);
                        motion.finished.then(stop, stop);
                    } catch (_) { console.warn('Prompt send animation failed'); stop(); }
                });
                window.addEventListener('resize', stop);
                window.addEventListener('popstate', stop);
                window.addEventListener('pagehide', stop);
                document.addEventListener('visibilitychange', stop);
                container.addEventListener('wheel', stop, { passive: true });
                container.addEventListener('touchstart', stop, { passive: true });
                input.addEventListener('input', stop);
                window.visualViewport?.addEventListener('resize', stop);
                reduce.addEventListener('change', stop);
            } catch (_) { console.warn('Prompt send animation failed'); stop(); }
            return stop;
        }
        // Costs use each saved answer's model, never the composer selection.
        let costPricing = null;
        let costPricingFailed = false;
        let costBarPath = [];
        let costBarReady = false;
        const COST_ESTIMATE_NOTE = '有料APIの通常料金による概算（USD）。回答ごとの保存済みモデルで計算します。入力使用量がない回答は会話履歴から推定します。モデルAPIの呼び出しごとに入力・出力（Thinkingを含む）を合算します。Ideogramはモデル・品質・速度・処理別の単価と画像枚数で計算します。単価未公表の設定は公表最安値を下限として「以上・一部未算入」で表示します。使用量未取得の呼び出しは一部未算入です。API報告のキャッシュ読込・書込と検索などの追加料金を含みます。Batch APIは事業者ごとの割引率（割引のない事業者を含む）で計算し、割引対象外のキャッシュ読込は通常単価のままです。保存時間や実行時間など取得できない課金情報は一部未算入です。その他の画像・音声・動画生成、税、無料枠・個別契約は含みません。時間帯や入力種別で単価が異なる場合は範囲で表示します。過去の請求額を再現するものではありません。';

        function costTokenNumber(value) {
            if (value === null || value === undefined || value === '') return null;
            const n = Number(value);
            return Number.isFinite(n) && n >= 0 ? n : null;
        }

        // Token-detail summary from saved per-request cache usage; null when no request reported cache fields.
        function costCacheUsage(thoughtData) {
            let metadata = thoughtData;
            if (typeof thoughtData === 'string') {
                if (!thoughtData.includes('cache_')) return null;
                try { metadata = JSON.parse(thoughtData); } catch (_) { return null; }
            }
            if (!metadata || typeof metadata !== 'object' || !Array.isArray(metadata.pricing_calls)) return null;
            let read = 0, write = 0, known = false;
            const ttls = {};
            for (const call of metadata.pricing_calls) {
                if (!call || typeof call !== 'object') continue;
                const readTokens = costTokenNumber(call.cache_read_tokens);
                const writeTokens = costTokenNumber(call.cache_write_tokens);
                let ttlTotal = 0;
                for (const ttl of ['5m', '1h', '30m', '12h']) {
                    const value = costTokenNumber(call[`cache_write_${ttl}_tokens`]);
                    if (value === null) continue;
                    ttls[ttl] = (ttls[ttl] || 0) + value;
                    ttlTotal += value;
                    known = true;
                }
                if (readTokens !== null || writeTokens !== null) known = true;
                read += readTokens || 0;
                write += Math.max(writeTokens || 0, ttlTotal);
            }
            return known ? { read, write, ttls } : null;
        }

        function costRate(pricing, model, date) {
            const key = pricing && (pricing.aliases[model] || model);
            const base = pricing && pricing.models[key];
            if (!base) return null;
            const rate = { ...base };
            for (const change of rate.changes || []) if (date >= change.from) Object.assign(rate, change);
            return rate;
        }

        function addRequestCost(result, request, pricing, date, batch = false) {
            result.requests++;
            if (String(request.model).startsWith('grok-') && Number.isSafeInteger(request.cost_in_usd_ticks) && request.cost_in_usd_ticks >= 0) {
                result.low += request.cost_in_usd_ticks / 1e10; result.high += request.cost_in_usd_ticks / 1e10;
                result.priced++;
                if (request.complete === false) result.unpriced++;
                return;
            }
            const rate = costRate(pricing, request.model, date);
            if (!rate) { result.unpriced++; return; }
            if (rate.unit === 'image') {
                const count = costTokenNumber(request.images);
                const prices = rate.image_rates?.[request.operation];
                const selected = prices?.[rate.image_selector === 'speed' ? request.speed : request.quality];
                const minimum = selected?.minimum ?? prices?.minimum;
                const unit = typeof prices === 'number' ? prices : typeof selected === 'number' ? selected : minimum;
                const lowerBound = Number.isFinite(unit) && typeof prices !== 'number' && typeof selected !== 'number';
                if (count === null || !Number.isFinite(unit)) { result.unpriced++; return; }
                result.low += count * unit;
                result.high += count * unit;
                result.priced++;
                if (lowerBound) result.lowerBound = true;
                if (request.complete === false || lowerBound) result.unpriced++;
                return;
            }
            const input = costTokenNumber(request.input_tokens);
            const output = costTokenNumber(request.output_tokens);
            if (input === null && output === null) { result.unpriced++; return; }
            const long = rate.threshold != null && (rate.threshold_inclusive ? (input || 0) >= rate.threshold : (input || 0) > rate.threshold);
            if (long) result.long++;
            const cost = requestTokenCost(request, rate, long, batch);
            result.low += cost.low; result.high += cost.high;
            if (cost.lowerBound) result.lowerBound = true;
            result.priced++;
            if (input === null || output === null || request.complete === false || cost.incomplete) result.unpriced++;

        }

        function legacyImageRequest(message, rows) {
            let files = message.image_url;
            try { if (typeof files === 'string') files = JSON.parse(files); } catch (_) { files = files ? [files] : []; }
            const count = Array.isArray(files) ? files.length : 0;
            const parent = rows.get(String(message.parent_id));
            const options = parent?.prompt_options || {};
            let source = parent?.image_url;
            let hasSource = Array.isArray(source) ? source.length > 0 : !!source && source !== '[]' && source !== 'null';
            // Image edits can reuse the latest generated image without a new attachment.
            let node = parent;
            const seen = new Set();
            while (!hasSource && node?.parent_id != null && !seen.has(String(node.id))) {
                seen.add(String(node.id)); node = rows.get(String(node.parent_id));
                if (node?.role === 'assistant' && node.image_url && node.image_url !== '[]' && node.image_url !== 'null') hasSource = true;
            }
            const operation = ['remove_object', 'remove_background'].includes(options.ideogram_operation)
                ? options.ideogram_operation : hasSource && message.model === 'ideogram-4.5' ? 'edit' : 'generate';
            let quality = options.ideogram_quality;
            if (!['very_low', 'low', 'medium', 'high'].includes(quality) || (quality === 'very_low' && operation === 'generate')) {
                quality = operation === 'edit' ? 'medium' : 'high';
            }
            const speed = parent ? (['turbo', 'default', 'quality'].includes(options.ideogram_speed) ? options.ideogram_speed : 'default') : null;
            return { model: message.model, images: count || null, operation, quality, speed, complete: !!parent };
        }
