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
