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
        function codingBranchAllowsTarget(target) {
            if (editingMessageId === null || editingMessageId === undefined) return true;
            // The edited prompt and its descendants belong to the old branch.
            // Only ancestors of the new prompt's parent may supply edit targets.
            const byId = new Map(allMessages.map(message => [String(message.id), message]));
            const visited = new Set();
            let id = currentParentId;
            while (id !== null && id !== undefined && !visited.has(String(id))) {
                const key = String(id);
                if (target && String(target.message_id) === key) return true;
                visited.add(key);
                const message = byId.get(key) || messageMeta[key];
                if (!message) break;
                id = message.parent_id;
            }
            return false;
        }
        function findLatestCodingTarget() {
            const root = get('chat-container');
            if (!root) return null;
            const buttons = Array.from(root.querySelectorAll('.message-group .coding-target-btn'));
            for (let i = buttons.length - 1; i >= 0; i--) {
                const target = getCodingTargetFromButton(buttons[i]);
                if (target && codingBranchAllowsTarget(target)) return target;
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
                if ((!selectedThread || !currentThreadId || String(selectedThread) === String(currentThreadId))
                    && codingBranchAllowsTarget(codingTargetSelection)) {
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
                    if (!target || !codingBranchAllowsTarget(target)) return;
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
                if ((!selectedThread || !currentThreadId || String(selectedThread) === String(currentThreadId))
                    && codingBranchAllowsTarget(codingTargetSelection)) {
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
                tokens_thought: 0,
                cache_read_tokens: 0,
                cache_write_tokens: 0,
                cache_write_ttls: {}
            };
            let hasTotal = false;
            let hasIn = false;
            let hasOut = false;
            let hasContent = false;
            let hasThought = false;
            let hasCache = false;
            (messages || []).forEach(m => {
                if (!m) return;
                if (m.__cacheUsageSource !== m.thought_data) {
                    m.__cacheUsage = costCacheUsage(m.thought_data);
                    m.__cacheUsageSource = m.thought_data;
                }
                if (m.__cacheUsage) {
                    totals.cache_read_tokens += m.__cacheUsage.read;
                    totals.cache_write_tokens += m.__cacheUsage.write;
                    for (const [ttl, value] of Object.entries(m.__cacheUsage.ttls || {})) {
                        totals.cache_write_ttls[ttl] = (totals.cache_write_ttls[ttl] || 0) + value;
                    }
                    hasCache = true;
                }
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
                cost_messages: messages || [],
                tokens_total: hasTotal ? totals.tokens_total : 0,
                tokens_in: hasIn ? totals.tokens_in : null,
                tokens_out: hasOut ? totals.tokens_out : null,
                tokens_content: hasContent ? totals.tokens_content : null,
                tokens_thought: hasThought ? totals.tokens_thought : null,
                cache_read_tokens: hasCache ? totals.cache_read_tokens : null,
                cache_write_tokens: hasCache ? totals.cache_write_tokens : null,
                cache_write_ttls: hasCache ? totals.cache_write_ttls : null
            };
        }
        function syncTokenBarVisibility() {
            const bar = get('total-token-bar');
            const estimate = get('prompt-token-estimate');
            if (bar) bar.classList.toggle('hidden', bar.dataset.hasTotals !== 'true' && (!estimate || estimate.classList.contains('hidden')));
        }
        function updateTotalTokenBar(total, details = null) {
            const bar = get('total-token-bar');
            const count = get('total-token-count');
            if (!bar || !count) return;
            const val = Number(total || 0);
            if (val > 0) {
                bar.dataset.hasTotals = 'true';
                count.innerText = `Total: ${val} tokens`;
                if (details) {
                    count.classList.add('cursor-pointer', 'underline', 'decoration-dotted');
                    messageMeta.__total__ = {
                        tokens_total: val,
                        tokens_in: details.tokens_in,
                        tokens_out: details.tokens_out,
                        tokens_content: details.tokens_content,
                        tokens_thought: details.tokens_thought,
                        cache_read_tokens: details.cache_read_tokens,
                        cache_write_tokens: details.cache_write_tokens,
                        cache_write_ttls: details.cache_write_ttls,
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
            } else {
                bar.dataset.hasTotals = 'false';
                count.innerText = 'Total: 0 tokens';
                count.classList.remove('cursor-pointer', 'underline', 'decoration-dotted');
                count.onclick = null;
                delete messageMeta.__total__;
            }
            refreshCostBar(details?.cost_messages || []);
            syncTokenBarVisibility();
        }
        const PROMPT_TOKEN_ESTIMATE_DEBOUNCE_MS = 700;
        let promptTokenEstimateTimer = null;
        let promptTokenEstimateAbort = null;
        let promptTokenEstimateSeq = 0;
        let promptTokenEstimateLastKey = '';
        let promptTokenEstimateLastData = null;
        let promptTokenEstimateVisible = null;
        let promptTokenDetailAbort = null;
        let promptTokenDetailSeq = 0;
        let promptTokenDetailSnapshot = null;
        function cancelPromptTokenDetail() {
            ++promptTokenDetailSeq;
            if (promptTokenDetailAbort) promptTokenDetailAbort.abort();
            promptTokenDetailAbort = null;
        }
        function setPromptTokenEstimateText(text, colorClass = 'text-gray-400') {
            const el = get('prompt-token-estimate');
            if (!el) return;
            el.disabled = true;
            promptTokenEstimateVisible = null;
            if (!text) {
                el.classList.add('hidden');
                el.innerText = '';
                syncTokenBarVisibility();
                return;
            }
            el.className = `text-left text-xs rounded ${colorClass}`;
            el.classList.remove('hidden');
            el.innerText = text;
            syncTokenBarVisibility();
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
            const incomplete = Number(data.files_non_text || 0) + Number(data.files_missing || 0) + Number(data.files_error || 0) > 0;
            setPromptTokenEstimateText(`入力見積: ${incomplete ? '一部のみ ' : ''}${total.toLocaleString('en-US')} tokens · 詳細`, 'text-cyan-300');
            promptTokenEstimateVisible = { data, payload: p };
            const trigger = get('prompt-token-estimate');
            if (trigger) { trigger.disabled = false; trigger.setAttribute('aria-label', `入力トークン見積の詳細: ${incomplete ? '一部のみ ' : ''}${total} tokens`); }
        }
        function renderPromptTokenDetail(data, payload, pending = false, failed = false) {
            const body = get('token-detail-estimate');
            if (!body) return;
            promptTokenDetailSnapshot = { data, payload };
            body.replaceChildren();
            const text = (value, cls = 'text-gray-400') => {
                const p = document.createElement('p'); p.className = cls; p.textContent = value; body.appendChild(p);
            };
            const row = (label, value) => {
                const item = document.createElement('div'); item.className = 'flex justify-between gap-3 items-start';
                const key = document.createElement('span'); key.className = 'min-w-0 break-words'; key.textContent = label;
                const number = document.createElement('span'); number.className = 'font-mono shrink-0';
                number.textContent = value === null || value === undefined ? '未計測' : `${value} tokens`;
                item.append(key, number); body.appendChild(item);
            };
            text(payload.model || '', 'text-blue-300 break-all');
            text('開いた時点の入力');
            row('合計', data.tokens_total);
            row('本文・引用', data.tokens_prompt);
            row('添付ファイル合計', data.tokens_files);
            text(`計測方法: ${data.estimate_method === 'provider' ? 'プロバイダー計測' : '概算'}`);
            text('対象: 入力欄の本文・引用・添付（履歴・SysPrompt・ツール除外）');
            text(`添付 ${data.files_total || 0}件 / 計測済み ${data.files_counted || 0}件 / 未計測 ${data.files_non_text || 0}件 / 未検出 ${data.files_missing || 0}件 / 失敗 ${data.files_error || 0}件`);
            const detail = data.breakdown;
            if (detail) {
                text('内訳', 'font-bold text-gray-200 pt-2');
                row('本文（形式情報を含む場合あり）', detail.message_tokens);
                row('引用（区切りを含む）', detail.quote_tokens);
                const statuses = { missing: '未検出', error: '計測失敗', read_error: '読み込み失敗', unseparated: '分割未計測', non_text: '未計測', no_text: 'テキストなし' };
                (detail.files || []).forEach((file, i) => {
                    row(`添付 ${i + 1}: ${file.name || 'ファイル'}`, file.tokens);
                    text(`${file.method === 'provider' ? 'プロバイダー計測' : '概算'} / ${file.reason === 'ok' ? '計測済み' : (statuses[file.reason] || '未計測')}`);
                });
                row('結合時の差・分割未計測分', detail.adjustment_tokens);
                text('内訳は本文、引用、添付を順に加えたときの増分です。結合時の差や分割できない分は別に表示します。');
            }
            if (pending) text('詳細な内訳を計算中...');
            if (failed) text('詳細な内訳を取得できませんでした。', 'text-red-300');
            if (!pending && !detail && !failed) text('この接続先では詳細な内訳を取得できません。');
            if (failed) {
                const retry = document.createElement('button'); retry.type = 'button'; retry.className = 'text-blue-300 underline';
                retry.textContent = '再試行'; retry.onclick = () => loadPromptTokenDetail(data, payload); body.appendChild(retry);
            }
            text('送信後に返る使用量や請求額の確定値とは異なります。');
        }
        async function loadPromptTokenDetail(snapshot, payload) {
            cancelPromptTokenDetail();
            const seq = promptTokenDetailSeq;
            promptTokenDetailAbort = new AbortController();
            renderPromptTokenDetail(snapshot, payload, true);
            try {
                const response = await apiFetch(CHAT_CONFIG.urls.estimatePromptTokensApi, {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ ...payload, include_details: true }), signal: promptTokenDetailAbort.signal
                });
                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                const data = await response.json();
                if (seq !== promptTokenDetailSeq || get('token-detail-modal').classList.contains('hidden')) return;
                if (!data.countable || (snapshot.estimate_method === 'provider' && data.estimate_method !== 'provider')) throw new Error('Unavailable');
                renderPromptTokenDetail(data, payload);
            } catch (error) {
                if (seq !== promptTokenDetailSeq || error.name === 'AbortError' || get('token-detail-modal').classList.contains('hidden')) return;
                renderPromptTokenDetail(snapshot, payload, false, true);
            }
        }
        function openPromptTokenDetail() {
            const visible = promptTokenEstimateVisible;
            if (!visible || !visible.data.countable) return;
            const payload = JSON.parse(JSON.stringify(visible.payload));
            get('token-detail-title').textContent = '入力トークン見積の詳細';
            get('token-detail-usage').classList.add('hidden');
            get('token-detail-estimate').classList.remove('hidden');
            showTokenDetailModal(null, true);
            get('token-detail-modal').querySelector('button')?.focus();
            loadPromptTokenDetail(visible.data, payload);
        }
        function restorePromptTokenDetail() {
            const snapshot = promptTokenDetailSnapshot;
            if (!snapshot) { showTokenDetailModal(); return; }
            get('token-detail-title').textContent = '入力トークン見積の詳細';
            get('token-detail-usage').classList.add('hidden');
            get('token-detail-estimate').classList.remove('hidden');
            showTokenDetailModal(null, true);
            if (snapshot.data.breakdown) renderPromptTokenDetail(snapshot.data, snapshot.payload);
            else loadPromptTokenDetail(snapshot.data, snapshot.payload);
        }
        function schedulePromptTokenEstimate(immediate = false) {
            const payload = buildPromptTokenEstimatePayload();
            ++promptTokenEstimateSeq;
            if (promptTokenEstimateTimer) { clearTimeout(promptTokenEstimateTimer); promptTokenEstimateTimer = null; }
            if (promptTokenEstimateAbort) { promptTokenEstimateAbort.abort(); promptTokenEstimateAbort = null; }
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
            renderPromptTokenEstimate({ pending: true }, payload);
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
            'gpt-image-options', 'gemini-image-options', 'grok-image-options', 'ideogram-image-options',
            'xai-chat-options', 'grok-video-options', 'mistral-ocr-options', 'image-input-limits', 'audio-gen-options'
        ];
        const THINKING_LEVELS = [
            { value: 'minimal', label: 'Min' },
            { value: 'low', label: 'Low' },
            { value: 'medium', label: 'Mid' },
            { value: 'high', label: 'High' },
            { value: 'xhigh', label: 'XHigh' }
        ];
        const MINIMAL_POPUP_ITEMS = [
            { key: 'attach', icon: 'fa-paperclip', label: 'ファイルを添付', action: 'upload' },
            { key: 'mask', icon: 'fa-mask', label: 'マスクを作成', action: 'button', buttonId: 'mask-btn', requiresMask: true },
            { key: 'voice-input', icon: 'fa-microphone', label: 'Voice Input', action: 'button', buttonId: 'mic-btn' },
            { key: 'rich-paste', icon: 'fa-paste', label: 'リッチ貼り付け', action: 'button', buttonId: 'rich-paste-btn' },
            { key: 'canvas', icon: 'fa-window-restore', label: 'Canvas', checkboxId: 'enable-canvas-mode', containerId: 'canvas-mode-container' },
            { key: 'coding', icon: 'fa-code-branch', label: 'Coding', checkboxId: 'enable-coding-mode', containerId: 'coding-mode-container' },
            { key: 'fast', icon: 'fa-bolt', label: '高速', checkboxId: 'enable-browser-fast-mode', containerId: 'browser-fast-mode-container' },
            { key: 'batch', icon: 'fa-layer-group', label: 'Batch', checkboxId: 'enable-batch-mode', containerId: 'batch-mode-container' },
            { key: 'search', icon: 'fa-search', label: 'Search', checkboxId: 'enable-search', containerId: 'search-container' },
            { key: 'urls', icon: 'fa-link', label: 'URLs', checkboxId: 'enable-url-context', containerId: 'url-context-container' },
            { key: 'maps', icon: 'fa-map-location-dot', label: 'Maps', checkboxId: 'enable-maps', containerId: 'maps-grounding-container' },
            { key: 'python', icon: 'fa-code', label: 'Python', checkboxId: 'enable-python', containerId: 'python-container' },
            { key: 'file', icon: 'fa-file-lines', label: 'File', checkboxId: 'enable-file-creation', containerId: 'file-creation-container' },
            { key: 'mcp', icon: 'fa-plug', label: 'MCP', checkboxId: 'enable-mcp', containerId: 'mcp-container' },
            { key: 'skills', icon: 'fa-wand-magic-sparkles', label: 'Skills', checkboxId: 'enable-skills', containerId: 'skills-container' },
            { key: 'sysprompt', icon: 'fa-terminal', label: 'SysPrompt', checkboxId: 'enable-sys-prompt', containerId: 'sys-prompt-option', gear: true, gearAction: () => { if (window.openThreadModal) window.openThreadModal(); } },
            { key: 'thinking', icon: 'fa-brain', label: 'Thinking', checkboxId: 'enable-thinking', containerId: 'thinking-options', special: 'thinking' },
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
            if (item.requiresMask) {
                const button = get(item.buttonId);
                if (!button || button.classList.contains('hidden')) return false;
            }
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
            if (item.special === 'thinking') {
                const opts = get('thinking-options');
                if (opts && opts.dataset.thinkingMode === 'effort') {
                    const effort = get('reasoning-effort');
                    return !!effort && String(effort.value || '').toLowerCase() !== 'none';
                }
            }
            const chk = get(item.checkboxId);
            return !!chk && chk.checked;
        }
        // effort モデルでは reasoning-effort、それ以外は thinking-level が
        // Thinking ボタンのレベル選択として表示される。
        function activeThinkingSelect() {
            const opts = get('thinking-options');
            if (opts && opts.dataset.thinkingMode === 'effort') return get('reasoning-effort');
            return get('thinking-level');
        }
        function thinkingUsesLevelSlider() {
            if (get('thinking-options')?.dataset.thinkingMode === 'effort') return true;
            return !!get('thinking-budget')?.disabled && thinkingOptionsList().length > 0;
        }
        function thinkingSliderOptions() {
            const opts = thinkingOptionsList();
            const chk = get('enable-thinking');
            if (get('thinking-options')?.dataset.thinkingMode !== 'effort' && chk && !chk.disabled) {
                return [{ value: 'off', textContent: 'Off' }, ...opts];
            }
            return opts;
        }
        function thinkingSliderValue() {
            const chk = get('enable-thinking');
            return get('thinking-options')?.dataset.thinkingMode !== 'effort' && chk && !chk.checked
                ? 'off' : activeThinkingSelect()?.value;
        }
        function thinkingOptionsList() {
            const sel = activeThinkingSelect();
            if (!sel) return [];
            return Array.from(sel.options).filter((opt) => !opt.disabled && !opt.classList.contains('hidden'));
        }
        function currentThinkingLevelLabel() {
            const sel = activeThinkingSelect();
            if (thinkingSliderValue() === 'off') return 'Off';
            if (!sel) return THINKING_LEVELS[3].label;
            if (sel.selectedOptions && sel.selectedOptions[0]) return sel.selectedOptions[0].textContent.trim();
            const level = THINKING_LEVELS.find((item) => item.value === sel.value);
            return level ? level.label : THINKING_LEVELS[3].label;
        }

