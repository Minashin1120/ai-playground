
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
            if(threadLoading) {
                snapshotSidebarHistory('loadThreads-skipped-busy append=' + !!append);
                return;
            }
            threadLoading = true;
            snapshotSidebarHistory('loadThreads-start append=' + !!append);
            try {
                if(!append) {
                    threadPage = 1;
                    hasMoreThreads = true;
                }
                const searchEl = get('search-box');
                const q = searchEl ? searchEl.value : '';
                if (!append && isSettingsModalOpen()) {
                    snapshotSidebarHistory('loadThreads-skipped-settings-open');
                    return;
                }
                const r = await apiFetch(`${CHAT_CONFIG.urls.handleThreads}?q=${encodeURIComponent(q)}&page=${threadPage}`);
                const d = await r.json();
                const l = get('thread-list');
                if (!l) return;

                if(!append) {
                    if (isSettingsModalOpen()) {
                        snapshotSidebarHistory('loadThreads-skip-replace-settings-open');
                        return;
                    }
                    const incomingCount = (d && Array.isArray(d.threads)) ? d.threads.length : -1;
                    const existingCount = l.querySelectorAll('[data-thread-id]').length;
                    if (incomingCount === 0 && existingCount > 0 && String(q || '').trim()) {
                        snapshotSidebarHistory('loadThreads-keep-existing-empty-search');
                        return;
                    }
                    l.innerHTML = '<div id="thread-pull-indicator" class="ptr-pull-indicator" aria-hidden="true"><i class="fas fa-arrow-down ptr-pull-icon"></i><i class="fas fa-spinner fa-spin ptr-pull-spinner"></i><span class="ptr-pull-label"></span></div><div id="scroll-sentinel"></div>';
                    if (threadObserver) {
                        threadObserver.disconnect();
                        const nextSentinel = get('scroll-sentinel');
                        if (nextSentinel) threadObserver.observe(nextSentinel);
                    }
                }

                const sentinel = get('scroll-sentinel');

                if (d && Array.isArray(d.threads)) {
                    d.threads.forEach((t) => {
                        const tid = String(t.id);
                        const row = document.createElement('div');
                        const star = t.is_bookmarked ? 'text-yellow-400' : 'text-gray-500';
                        const tempBadge = t.is_temporary ? '<span class="text-[9px] text-amber-300 border border-amber-500/50 rounded px-1 py-0">一時</span>' : '';

                        // Active highlighting
                        const isActive = (tid === String(currentThreadId));
                        const activeClass = isActive ? 'bg-gray-700/60 border-l-2 border-blue-500' : '';

                        row.className = `p-2 rounded hover:bg-gray-700 cursor-pointer text-sm text-gray-300 truncate flex justify-between items-center group ${activeClass}`;
                        row.dataset.threadId = tid;
                        row.dataset.threadTitle = t.title || '';
                        row.innerHTML = `<div class="flex items-center gap-1 truncate flex-1"><button class="${star} hover:text-yellow-400 px-1" onclick="toggleBookmark(event, '${tid}')"><i class="fas fa-star text-[10px]"></i></button><span class="truncate">${escapeHtml(t.title || "No Title")}</span>${tempBadge}</div><div class="flex items-center gap-1 opacity-100 md:opacity-0 md:group-hover:opacity-100 transition" data-thread-actions="1"><button class="text-gray-500 hover:text-white px-1 transition" onclick="renameThread(event, '${tid}')"><i class="fas fa-pen text-xs"></i></button><button class="text-gray-500 hover:text-red-400 px-1 transition" onclick="deleteThread(event, '${tid}')"><i class="fas fa-trash text-xs"></i></button></div>`;
                        row.onclick = (e) => {
                            if (e.target.closest('button') || e.target.closest('[data-thread-actions]')) return;
                            loadMessages(tid);
                        };
                        if (sentinel) l.insertBefore(row, sentinel);
                        else l.appendChild(row);
                    });

                    hasMoreThreads = !!d.has_next;
                    if(hasMoreThreads) threadPage++;
                    snapshotSidebarHistory('loadThreads-rendered count=' + d.threads.length + ' append=' + !!append);
                } else {
                    snapshotSidebarHistory('loadThreads-empty-or-invalid');
                }
            } catch (err) {
                console.error('Failed to load threads:', err);
                snapshotSidebarHistory('loadThreads-error');
            } finally {
                threadLoading = false;
                updateThreadHighlighting();
                snapshotSidebarHistory('loadThreads-finally');
            }
        }

        // Generic pull-to-refresh for a scrollable list.
        // Works both in the sidebar and inside the history modal (the element is
        // re-parented between the two, so handlers are attached to the list element).
        // refreshFn must return a promise (or undefined if a load is already running).
        function initPullToRefresh(listId, refreshFn) {
            const list = get(listId);
            if (!list) return;
            const indicatorId = `${listId}-pull-indicator`;

            const TRIGGER_DIST = 60;
            const MAX_PULL_DIST = 88;
            const HOLD_DIST = 52;
            const RESISTANCE = 0.5;
            const PULL_DEAD_ZONE = 8;

            let pullStartY = 0;
            let pulling = false;
            let pullDist = 0;
            let pullRefreshPromise = null;

            const indicatorEl = () => get(indicatorId);
            const labelEl = () => {
                const ind = indicatorEl();
                return ind ? ind.querySelector('.ptr-pull-label') : null;
            };

            const applyPullUI = (dist) => {
                const ind = indicatorEl();
                if (!ind) return;
                ind.style.height = Math.min(dist, MAX_PULL_DIST) + 'px';
                ind.classList.toggle('active', dist > 2);
                ind.classList.toggle('pull-ready', dist >= TRIGGER_DIST);
                const lab = labelEl();
                if (lab) lab.textContent = dist >= TRIGGER_DIST ? '離して更新' : '引っ張って更新';
            };

            const resetPullUI = () => {
                const ind = indicatorEl();
                if (!ind) return;
                ind.style.height = '0px';
                ind.classList.remove('active', 'pull-ready', 'refreshing');
                ind.classList.remove('dragging');
            };

            list.addEventListener('touchstart', (e) => {
                if (pullRefreshPromise) { pulling = false; return; }
                if (list.scrollTop > 0) { pulling = false; return; }
                const t = e.touches[0];
                if (!t) return;
                pullStartY = t.clientY;
                pullDist = 0;
                pulling = true;
            }, { passive: true });

            list.addEventListener('touchmove', (e) => {
                if (!pulling || pullRefreshPromise) return;
                if (list.scrollTop > 0) {
                    pulling = false;
                    return;
                }
                const t = e.touches[0];
                if (!t) return;
                const dy = t.clientY - pullStartY;
                if (dy <= 0) {
                    if (pullDist > 0) {
                        pullDist = 0;
                        applyPullUI(0);
                    }
                    pulling = false;
                    return;
                }
                const ind = indicatorEl();
                if (ind && !ind.classList.contains('dragging')) ind.classList.add('dragging');
                pullDist = Math.min(dy * RESISTANCE, MAX_PULL_DIST);
                applyPullUI(pullDist);
                if (dy >= PULL_DEAD_ZONE) {
                    e.preventDefault();
                }
            }, { passive: false });

            list.addEventListener('touchend', () => {
                if (!pulling) return;
                pulling = false;
                if (pullRefreshPromise) return;
                const ind = indicatorEl();
                if (ind) ind.classList.remove('dragging');
                const shouldRefresh = pullDist >= TRIGGER_DIST;
                pullDist = 0;
                if (!shouldRefresh) {
                    resetPullUI();
                    return;
                }
                let p;
                try { p = refreshFn(); } catch (err) { p = null; }
                // If refreshFn rebuilds the list synchronously (loadThreads), grab the
                // freshly re-created indicator; otherwise keep the current one spinning.
                const ind3 = indicatorEl();
                if (ind3) {
                    ind3.classList.add('refreshing');
                    ind3.style.height = HOLD_DIST + 'px';
                    const lab = ind3.querySelector('.ptr-pull-label');
                    if (lab) lab.textContent = '更新中...';
                }
                if (p && typeof p.then === 'function') {
                    pullRefreshPromise = p;
                    p.catch(() => {}).finally(() => {
                        pullRefreshPromise = null;
                        resetPullUI();
                    });
                } else {
                    // Load was already in flight or failed synchronously; snap back shortly.
                    pullRefreshPromise = Promise.resolve();
                    setTimeout(() => {
                        pullRefreshPromise = null;
                        resetPullUI();
                    }, 400);
                }
            });

            list.addEventListener('touchcancel', () => {
                pulling = false;
                pullDist = 0;
                resetPullUI();
            });
        }

        const initThreadPullToRefresh = () => initPullToRefresh('thread-list', () => loadThreads(false));
        const initSkillPullToRefresh = () => initPullToRefresh('skill-list', () => loadSkills());

        const initPullToRefreshAll = () => {
            initThreadPullToRefresh();
            initSkillPullToRefresh();
        };

        /* ================= MCP チャット中イベント（実行カード・確認ダイアログ） ================= */
        let activeMcpDecision = null;
        let mcpDecisionModalBound = false;

        const mcpCardIdSelector = (id) => 'mcp_card_' + String(id).replace(/[^A-Za-z0-9_-]/g, '_');
        const mcpEscHtml = (s) => String(s == null ? '' : s).replace(/[&<>"']/g, (c) => ({
            '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
        }[c]));

        function mcpCardTitle(payload) {
            return `${mcpEscHtml(payload.server_name || 'MCP')} / ${mcpEscHtml(payload.tool_name || payload.internal_name || '')}`;
        }

        function getMcpExecutionList(adiv) {
            if (!adiv) return null;
            let list = adiv.querySelector('.mcp-execution-list');
            if (!list) {
                list = document.createElement('div');
                list.className = 'mcp-execution-list mt-3';
                list.setAttribute('aria-label', 'MCPツール実行');
                // Keep all MCP cards after the answer content, just like the
                // Python execution details are kept out of the prose body.
                adiv.appendChild(list);
            }
            return list;
        }

        function handleMcpStreamEvent(adiv, payload) {
            if (!adiv || !payload || !payload.type) return;
            const needsExecutionList = ['start', 'result', 'error'].includes(payload.type);
            const list = needsExecutionList ? getMcpExecutionList(adiv) : null;
            if (needsExecutionList && !list) return;
            const boxId = mcpCardIdSelector(payload.id || ('mcp_' + Date.now()));
            if (payload.type === 'start') {
                if (list.querySelector('[data-mcp-card="' + boxId + '"]')) return;
                const html = `<div class="mcp-box mcp-running mb-2" data-mcp-card="${boxId}">
    <span class="mcp-spinner"></span>
    <span class="mcp-box-title">${mcpCardTitle(payload)}</span>
    <span class="mcp-box-sub">実行中...</span>
</div>`;
                list.insertAdjacentHTML('beforeend', html);
                return;
            }
            if (payload.type === 'result') {
                let box = list.querySelector('[data-mcp-card="' + boxId + '"]');
                const summary = payload.summary || '';
                if (!box) {
                    const html = `<div class="mcp-box mcp-done mb-2" data-mcp-card="${boxId}">
    <i class="fas fa-check-circle mcp-box-ok"></i>
    <span class="mcp-box-title">${mcpCardTitle(payload)}</span>
    <span class="mcp-box-sub">実行しました</span>
</div>`;
                    list.insertAdjacentHTML('beforeend', html);
                    box = list.querySelector('[data-mcp-card="' + boxId + '"]');
                } else {
                    box.classList.remove('mcp-running');
                    box.classList.add('mcp-done');
                    box.innerHTML = `<i class="fas fa-check-circle mcp-box-ok"></i>
    <span class="mcp-box-title">${mcpCardTitle(payload)}</span>
    <span class="mcp-box-sub">実行しました</span>`;
                }
                if (summary) {
                    const note = document.createElement('div');
                    note.className = 'mcp-box-note';
                    note.textContent = summary.split('\n')[0].slice(0, 220);
                    if (box) box.appendChild(note);
                }
                return;
            }
            if (payload.type === 'error') {
                let box = list.querySelector('[data-mcp-card="' + boxId + '"]');
                const msg = payload.message || 'MCPツールの実行に失敗しました';
                if (!box) {
                    const html = `<div class="mcp-box mcp-error mb-2" data-mcp-card="${boxId}">
    <i class="fas fa-times-circle mcp-box-err"></i>
    <span class="mcp-box-title">${mcpCardTitle(payload)}</span>
    <span class="mcp-box-sub">失敗</span>
</div>`;
                    list.insertAdjacentHTML('beforeend', html);
                    box = list.querySelector('[data-mcp-card="' + boxId + '"]');
                } else {
                    box.classList.remove('mcp-running');
                    box.classList.add('mcp-error');
                    box.innerHTML = `<i class="fas fa-times-circle mcp-box-err"></i>
    <span class="mcp-box-title">${mcpCardTitle(payload)}</span>
    <span class="mcp-box-sub">失敗</span>`;
                }
                const note = document.createElement('div');
                note.className = 'mcp-box-note mcp-box-note-err';
                note.textContent = String(msg).slice(0, 300);
                if (box) box.appendChild(note);
                return;
            }
            if (payload.type === 'decision_resolved') {
                if (activeMcpDecision && activeMcpDecision.id && payload.id && activeMcpDecision.id === payload.id) {
                    const modal = get('mcp-decision-modal');
                    if (modal && !modal.classList.contains('hidden')) {
                        try { hideModal('mcp-decision-modal'); } catch (e) {}
                    }
                    activeMcpDecision = null;
                }
                return;
            }
        }

        function openMcpDecisionModal(payload) {
            const modal = get('mcp-decision-modal');
            if (!modal || !payload) return;
            if (activeMcpDecision && activeMcpDecision.id === payload.id) return;
            activeMcpDecision = {
                id: payload.id || null,
                jobId: currentJobId || null
            };
            const serverEl = get('mcp-decision-server');
            const toolEl = get('mcp-decision-tool');
            const argsEl = get('mcp-decision-args');
            if (serverEl) serverEl.textContent = payload.server_name || '不明なサーバー';
            if (toolEl) toolEl.textContent = payload.tool_name || '';
            if (argsEl) {
                let preview = payload.args_preview || '';
                try {
                    const parsed = JSON.parse(preview);
                    preview = JSON.stringify(parsed, null, 2);
                } catch (e) { /* raw */ }
                argsEl.textContent = preview;
            }
            const allowBtn = get('mcp-decision-allow');
            const denyBtn = get('mcp-decision-deny');
            if (allowBtn) allowBtn.onclick = () => submitMcpDecision('allow');
            if (denyBtn) denyBtn.onclick = () => submitMcpDecision('deny');
            try { showModal('mcp-decision-modal'); } catch (e) {}
        }

        async function submitMcpDecision(decision) {
            const modal = get('mcp-decision-modal');
            try { if (modal) hideModal('mcp-decision-modal'); } catch (e) {}
            const jobId = activeMcpDecision ? activeMcpDecision.jobId : null;
            const id = activeMcpDecision ? activeMcpDecision.id : null;
            activeMcpDecision = null;
            if (!jobId) return;
            try {
                await apiFetch('/api/mcp/chat/' + encodeURIComponent(jobId) + '/decision', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ decision: decision, id: id })
                });
            } catch (e) { /* 失敗してもチャットは継続（タイムアウト=拒否） */ }
        }

        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', initPullToRefreshAll, { once: true });
        } else {
            initPullToRefreshAll();
        }

        let geminiBatchStatusPollBusy = false;

        function showGeminiBatchCompletionBanner(completed) {
            const banner = get('batch-notification-banner');
            const text = get('batch-notification-text');
            const open = get('batch-notification-open');
            if (!banner || !text || !completed || !completed.length) return;
            const first = completed[0];
            const threadId = first.thread_id;
            text.textContent = completed.length === 1
                ? `${first.model} のBatch処理が完了しました。`
                : `${completed.length}件のBatch処理が完了しました。`;
            banner.classList.remove('hidden');
            if (open) {
                open.onclick = async () => {
                    banner.classList.add('hidden');
                    if (threadId) await loadMessages(threadId);
                };
            }
            const close = get('batch-notification-close');
            if (close) close.onclick = () => banner.classList.add('hidden');
        }

        async function refreshGeminiBatchStatus() {
            if (geminiBatchStatusPollBusy) return;
            geminiBatchStatusPollBusy = true;
            try {
                const response = await apiFetch('/api/skillini/batch/status');
                if (!response.ok) return;
                const data = await response.json().catch(() => ({}));
                const hasActiveBatch = (data.active || []).some((j) => (
                    currentThreadId && String(j.thread_id) === String(currentThreadId)
                ));
                if (hasActiveBatch) {
                    await loadMessages(currentThreadId, { preserveDraft: true, silent: true });
                }
                const completed = data.completed || [];
                if (completed.length) {
                    showGeminiBatchCompletionBanner(completed);
                    if (completed.some((job) => (
                        String(job.thread_id) === String(currentThreadId)
                    ))) {
                        await loadMessages(currentThreadId, { preserveDraft: true, silent: true });
                    }
                }
            } catch (error) {
                // Background polling is best-effort; the normal chat UI remains usable.
            } finally {
                geminiBatchStatusPollBusy = false;
            }
        }

        refreshGeminiBatchStatus();
        // Keep the legacy interval value in this source for regression coverage.
        setInterval(refreshGeminiBatchStatus, 2000); // 15000ms was too slow for answer bubbles.

        let chatTransitionSequence = 0;
        let chatTransitionTimer = null;
        const CHAT_TRANSITION_DURATION_MS = 460;

        function playChatTransition(kind) {
            const veil = get('chat-transition-veil');
            if (!veil) return;
            const sequence = ++chatTransitionSequence;
            if (chatTransitionTimer) {
                clearTimeout(chatTransitionTimer);
                chatTransitionTimer = null;
            }
            if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
                veil.classList.remove('is-active');
                veil.removeAttribute('data-transition-kind');
                return;
            }
            veil.dataset.transitionKind = kind || 'history';
            veil.classList.remove('is-active');
            // Restart the CSS animation when the user changes threads rapidly.
            void veil.offsetWidth;
            veil.classList.add('is-active');
            chatTransitionTimer = setTimeout(() => {
                if (sequence !== chatTransitionSequence) return;
                veil.classList.remove('is-active');
                chatTransitionTimer = null;
            }, CHAT_TRANSITION_DURATION_MS);
        }

        async function toggleBookmark(e, tid) {
            if (e) e.stopPropagation();
            await apiFetch(`/api/threads/${tid}/bookmark`, {method:'POST'});
            loadThreads();
        }

        async function loadMessages(tid, opts = {}) {
            const loadSequence = ++threadLoadSequence;
            const targetMessageId = opts.targetMessageId || new URLSearchParams(location.search).get('message');
            if (window.closeHistoryModal) window.closeHistoryModal();
            const preserveDraft = !!opts.preserveDraft;
            const silent = !!opts.silent;
            if (!silent) resumeChatAutoScroll({ scroll: false });
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
                promptSettingsBeforeEdit = null;
                setEditUi(false);
            } else {
                cancelEdit();
            }
            if (!silent) playChatTransition('history');
            const nextThreadId = tid !== null && tid !== undefined ? String(tid) : tid;
            if (nextThreadId !== currentThreadId) modelChosenManually = false;
            currentThreadId = nextThreadId;
            if (!opts.skipHistory) history.pushState({}, '', '/c/' + tid);
            updateThreadHighlighting();
            syncActiveSkillForThread(currentThreadId);
            get('welcome-screen').classList.add('hidden');
            if (!silent) {
                // Show shimmer skeleton while history loads (open from list / page reload)
                get('chat-container').innerHTML = buildChatLoadingSkeletonHtml();
            }
            try {
                const threadUrl = new URL(CHAT_CONFIG.urls.handleThreadItem.replace('0', tid), window.location.origin);
                threadUrl.searchParams.set('limit', String(getEffectiveThreadInitialMessageLimit()));
                const focusId = targetMessageId;
                if (focusId && /^\d+$/.test(String(focusId))) threadUrl.searchParams.set('before_id', String(Number(focusId) + 1));
                const r = await apiFetch(threadUrl.toString());
                if (!r.ok) throw new Error(`thread request failed (${r.status})`);
                const threadData = await r.json();
                if (!threadData || !Array.isArray(threadData.messages)) throw new Error('invalid thread response');
                if (loadSequence !== threadLoadSequence) return false;
            setCurrentChatHeaderTitle(threadData && threadData.title);
            allMessages = threadData.messages;
            renderBranchOrigin(threadData.branch_origin);
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
            // The chat's PromptCache flag comes first, so a lock left by the previous chat
            // never blocks switching to this chat's model.
            if (get('enable-prompt-cache')) {
                get('enable-prompt-cache').checked = !!threadData.enable_prompt_caching;
                updatePromptCacheUi();
            }
            if (threadData.last_model && !modelChosenManually) {
                selectModelById(threadData.last_model);
            }
            if (threadData.last_skill_uuid && loadedSkills.length > 0) {
                const skill = loadedSkills.find(item => item.uuid === threadData.last_skill_uuid);
                if (skill) {
                    threadSkillMap[currentThreadId] = skill;
                    applyActiveSkill(skill);
                }
            }

            // Set default leaf (latest message or fixed branch). A reload right after an
            // edit/regenerate send (forceLatestLeaf) must show the branch just created,
            // not snap back to a branch the user pinned earlier in this thread.
            const storedFixedId = opts.forceLatestLeaf ? null : localStorage.getItem(`fixed_branch_${currentThreadId}`);
            const focusLeaf = targetMessageId;
            if (focusLeaf && allMessages.some(m => String(m.id) === String(focusLeaf))) {
                currentLeafId = Number(focusLeaf);
            } else if (storedFixedId && allMessages.find(m => String(m.id) === String(storedFixedId))) {
                currentLeafId = storedFixedId;
            } else if (allMessages.length > 0) {
                currentLeafId = allMessages[allMessages.length - 1].id;
            } else {
                currentLeafId = null;
            }

            if (silent) {
                renderThreadTree({ silent, keepScroll: silent });
            } else {
                renderThreadTree({ silent, keepScroll: silent, animate: true });
            }
            if (!preserveDraft) restoreLatestPromptOptions();
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
            if (targetMessageId && new URLSearchParams(location.search).has('message')) history.replaceState({}, '', location.pathname);
            if (!preserveDraft) schedulePromptTokenEstimate(true);
            if(window.innerWidth < 768) get('overlay').click();
            if (typeof window.__refreshAdminThreadEncState === 'function') {
                try { window.__refreshAdminThreadEncState(); } catch (_) {}
            }
            return true;
            } catch (err) {
                if (loadSequence !== threadLoadSequence) return false;
                console.error('Failed to load chat thread:', err);
                if (!silent) showChatLoadError(tid);
                if (!silent) showToast('チャットの読み込みに失敗しました', 'error', true);
                return false;
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

            // When a silent reload keeps the current thread view (e.g. right after a
            // streamed answer completes), preserve its scroll position. Build the new
            // tree offscreen and replace the old children in one operation, so the
            // scroll range never collapses to zero during the rebuild.
            // Raw TeX is much shorter than the typeset answer. Keep each message's
            // height until MathJax finishes, otherwise the browser clamps scrollTop.
            const previousChildren = keepScroll ? Array.from(container.children) : [];
            const previousHeights = previousChildren.map(el => el.getBoundingClientRect().height);
            let previousScrollTop = null;
            if (keepScroll) {
                previousScrollTop = container.scrollTop;
            }

            if (allMessages.length === 0) {
                container.replaceChildren();
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
                    m.skill_name,
                    m.batch_job,
                    m.prompt_options && typeof m.prompt_options === 'object' ? m.prompt_options.image_mask : null,
                    m.processing_time,
                    m.math_steps_state
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

            const heldMessages = [];
            if (keepScroll && !lowBandwidthMode) {
                Array.from(fragment.children).forEach((el, index) => {
                    const old = previousChildren.find(child => child.id && child.id === el.id)
                        || (previousChildren.length === fragment.children.length ? previousChildren[index] : null);
                    const height = old ? previousHeights[previousChildren.indexOf(old)] : 0;
                    if (height > 0) {
                        el.style.minHeight = `${height}px`;
                        heldMessages.push(el);
                    }
                });
            }
            container.replaceChildren(fragment);
            displayMissingMessageFiles(container, path);

            updateTotalTokenBar(pathTotals.tokens_total, pathTotals);
            currentParentId = currentLeafId;
            if (keepScroll && previousScrollTop !== null) {
                restoreThreadTreeScroll(container, previousScrollTop);
            } else {
                scrollToBottom();
            }
            if (lowBandwidthMode) {
                queueMessageDecorations(container, container ? (container.textContent || '') : '');
            } else {
                queueHighlight(container);
                if (path.length) {
                    const mathReady = queueMathTypeset(container, path.map(message => message.content || '').join('\n'));
                    Promise.resolve(mathReady).finally(() => {
                        if (!heldMessages.length || !heldMessages[0].isConnected) return;
                        const top = container.scrollTop;
                        heldMessages.forEach(el => { el.style.minHeight = ''; });
                        restoreThreadTreeScroll(container, top);
                    });
                }
            }
        }

        function restoreThreadTreeScroll(container, previousScrollTop) {
            if (!container) return;
            const maxScroll = container.scrollHeight - container.clientHeight;
            if (userAutoScroll && !chatManualPauseIntent) {
                // The user is still following the stream (pinned at the bottom): snap
                // to the new bottom so the view stays pinned.
                container.scrollTop = container.scrollHeight;
            } else {
                // Auto-scroll is paused (the user scrolled away to re-read), or a
                // scroll-up gesture was just made: keep the exact document position.
                // Completing the answer must never drag a paused user back down to
                // the bottom, no matter how close to the bottom they were.
                container.scrollTop = Math.max(0, Math.min(previousScrollTop, maxScroll));
            }
            chatLastScrollTop = container.scrollTop;
            syncScrollToBottomButton();
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
                restoreLatestPromptOptions();
                return;
            }
            while (msgMap[currId] && msgMap[currId].childrenIds.length > 0) {
                // Pick the first child (or latest? let's pick latest child by ID)
                const children = msgMap[currId].childrenIds;
                currId = Math.max(...children);
            }
            currentLeafId = currId;
            renderThreadTree({ animate: true });
            restoreLatestPromptOptions();
        }
        async function loadSkills() {
            try {
                const r = await apiFetch(CHAT_CONFIG.urls.handleSkills);
                const rows = await r.json();
                loadedSkills = rows;
                const l = get('skill-list');
                if (!l) return;
                l.innerHTML = '<div id="skill-pull-indicator" class="ptr-pull-indicator" aria-hidden="true"><i class="fas fa-arrow-down ptr-pull-icon"></i><i class="fas fa-spinner fa-spin ptr-pull-spinner"></i><span class="ptr-pull-label"></span></div>';
                if (Array.isArray(rows)) {
                    rows.forEach((skill) => {
                        const d = document.createElement('div');
                        d.className = 'skill-item p-2 rounded hover:bg-gray-700 cursor-pointer text-sm text-gray-300 flex justify-between items-center group';
                        d.innerHTML = `<div class="flex items-center gap-2 overflow-hidden"><i class="fas fa-wand-magic-sparkles text-blue-500"></i><span class="truncate">${escapeHtml(skill.name)}</span></div><div class="flex items-center gap-1"><button class="text-gray-400 hover:text-blue-400 opacity-100 md:opacity-0 md:group-hover:opacity-100 px-2 transition" onclick="openEditSkillModal(event,'${skill.uuid}')"><i class="fas fa-pencil-alt text-[10px]"></i></button><button class="text-gray-400 hover:text-red-400 opacity-100 md:opacity-0 md:group-hover:opacity-100 px-2 transition" onclick="deleteSkill(event,'${skill.uuid}')"><i class="fas fa-trash text-[10px]"></i></button></div>`;
                        d.onclick = (e) => { if(!e.target.closest('button')) activateSkill(skill); };
                        l.appendChild(d);
                    });
                }
            } catch (err) {
                console.error('Failed to load skills:', err);
            }
        }
        async function openEditSkillModal(e, id) {
            e.stopPropagation();
            editingSkillUuid = id;
            try {
                const r = await apiFetch(`/api/skills/${id}`);
                const skill = await r.json();
                get('skill-name').value = skill.name;
                get('skill-desc').value = skill.description || '';
                get('skill-inst').value = skill.instruction;
                get('skill-modal-title').textContent = 'Edit Skill';
                get('save-skill-btn').innerText = "Save Changes";
                showModal('skill-modal');
                if (location.pathname !== '/skill') {
                    history.pushState({ modal: 'skill' }, '', '/skill');
                }
            } catch (err) {
                showToast("Skillの取得に失敗しました", "error", true);
            }
        }
        async function createSkill(name, inst) { await apiFetch(CHAT_CONFIG.urls.handleSkills, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({name, instruction:inst})}); loadSkills(); }

        function applyActiveSkill(skill) {
            activeSkill = skill || null;
            if (activeSkill) {
                get('active-skill-name').innerText = activeSkill.name;
                get('skill-active-indicator').classList.remove('hidden');
            } else {
                get('skill-active-indicator').classList.add('hidden');
            }
            const sysPromptOption = get('sys-prompt-option');
            if (sysPromptOption) sysPromptOption.style.opacity = '1';
        }
        function syncActiveSkillForThread(tid) {
            const skill = tid && threadSkillMap[tid] ? threadSkillMap[tid] : null;
            applyActiveSkill(skill);
        }
        async function saveThreadSkillUuid(threadId, skillUuid) {
            try {
                await apiFetch(CHAT_CONFIG.urls.handleSettings, {
                    method:'POST',
                    headers:{'Content-Type':'application/json'},
                    body:JSON.stringify({last_skill_uuid: skillUuid, thread_id: threadId})
                });
            } catch(e) {}
        }
        function activateSkill(skill, skipSave) {
            if (currentThreadId) {
                threadSkillMap[currentThreadId] = skill;
                applyActiveSkill(skill);
                showToast(`Skill "${skill.name}" をこのチャットに適用しました`, "success");
                if (!skipSave) saveThreadSkillUuid(currentThreadId, skill ? skill.uuid : null);
            } else {
                pendingSkillForNewThread = skill;
                applyActiveSkill(skill);
                if (allMessages && allMessages.length > 0) {
                    startNewChat({ preserveSkill: true });
                }
            }
        }
        function clearActiveSkill() { if (currentThreadId) { delete threadSkillMap[currentThreadId]; saveThreadSkillUuid(currentThreadId, null); } pendingSkillForNewThread = null; applyActiveSkill(null); }
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
