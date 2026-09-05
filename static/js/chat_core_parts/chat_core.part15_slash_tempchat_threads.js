
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', initPullToRefreshAll, { once: true });
        } else {
            initPullToRefreshAll();
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
                setEditUi(false);
            } else {
                cancelEdit();
            }
            currentThreadId = tid !== null && tid !== undefined ? String(tid) : tid;
            if (!opts.skipHistory) history.pushState({}, '', '/c/' + tid);
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
                if (loadSequence !== threadLoadSequence) return false;
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

            renderThreadTree({ silent, keepScroll: silent });
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
            // streamed answer completes), preserve the scroll position. Clearing the
            // container collapses its height, which would otherwise reset the view to
            // the top. The decision to stay put or snap to the bottom is made by the
            // restore helper based on the auto-scroll state: users who scrolled away
            // (auto-scroll paused) must never be dragged back down, while users still
            // pinned at the bottom stay pinned without a one-frame jump to the top.
            let previousScrollTop = null;
            if (keepScroll) {
                previousScrollTop = container.scrollTop;
            }

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
                    const latestText = path[path.length - 1] && path[path.length - 1].content;
                    queueMathTypeset(container, latestText);
                }
            }
        }

        function restoreThreadTreeScroll(container, previousScrollTop) {
            if (!container) return;
            const maxScroll = container.scrollHeight - container.clientHeight;
            if (userAutoScroll && !chatManualPauseIntent) {
                // The user is still following the stream (pinned at the bottom): snap
                // to the new bottom synchronously so the view stays pinned without a
                // one-frame jump to the top.
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
            try {
                const r = await apiFetch(CHAT_CONFIG.urls.handleGems);
                const gs = await r.json();
                loadedGems = gs;
                const l = get('gem-list');
                if (!l) return;
                l.innerHTML = '<div id="gem-pull-indicator" class="ptr-pull-indicator" aria-hidden="true"><i class="fas fa-arrow-down ptr-pull-icon"></i><i class="fas fa-spinner fa-spin ptr-pull-spinner"></i><span class="ptr-pull-label"></span></div>';
                if (Array.isArray(gs)) {
                    gs.forEach((g) => {
                        const d = document.createElement('div');
                        d.className = 'gem-item p-2 rounded hover:bg-gray-700 cursor-pointer text-sm text-gray-300 flex justify-between items-center group';
                        d.innerHTML = `<div class="flex items-center gap-2 overflow-hidden"><i class="fas fa-gem text-blue-500"></i><span class="truncate">${escapeHtml(g.name)}</span></div><div class="flex items-center gap-1"><button class="text-gray-400 hover:text-blue-400 opacity-100 md:opacity-0 md:group-hover:opacity-100 px-2 transition" onclick="openEditGemModal(event,'${g.uuid}')"><i class="fas fa-pencil-alt text-[10px]"></i></button><button class="text-gray-400 hover:text-red-400 opacity-100 md:opacity-0 md:group-hover:opacity-100 px-2 transition" onclick="deleteGem(event,'${g.uuid}')"><i class="fas fa-trash text-[10px]"></i></button></div>`;
                        d.onclick = (e) => { if(!e.target.closest('button')) activateGem(g); };
                        l.appendChild(d);
                    });
                }
            } catch (err) {
                console.error('Failed to load gems:', err);
            }
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
            if (typeof window.__refreshAdminThreadEncState === 'function') {
                try { window.__refreshAdminThreadEncState(); } catch (_) {}
            }
            if (!opts.skipHistory) history.pushState({}, '', '/');
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
            syncBack('modal-grok-image-quality', 'grok-image-quality');
            syncBack('modal-ocr-table-format', 'ocr-table-format');
            syncBack('modal-ocr-pages', 'ocr-pages');
            const syncBackChk = (modalId, targetId) => {
                if (get(modalId) && get(targetId)) get(targetId).checked = get(modalId).checked;
            };
            syncBackChk('modal-ocr-extract-header', 'ocr-extract-header');
            syncBackChk('modal-ocr-extract-footer', 'ocr-extract-footer');
            syncBackChk('modal-ocr-include-blocks', 'ocr-include-blocks');
            syncBackChk('modal-ocr-include-images', 'ocr-include-images');

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
                                    inlineMath: [['\\(', '\\)'], ['$', '$']],
                                    displayMath: [['$$', '$$'], ['\\[', '\\]']],
                                    processEscapes: true
                                },
                                options: {
                                    ignoreHtmlClass: 'tex2jax_ignore|mathjax_ignore',
                                    processHtmlClass: 'tex2jax_process|mathjax_process'
                                },
                                startup: {
                                    typeset: false
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
        function updateLibFavoriteFilterUi() {
            const btn = get('lib-favorite-filter-btn');
            if (!btn) return;
            const active = !!lib.favoritesOnly;
            btn.classList.toggle('is-active', active);
            btn.setAttribute('aria-pressed', active ? 'true' : 'false');
            const icon = btn.querySelector('i');
            if (icon) icon.className = active ? 'fas fa-star' : 'far fa-star';
        }
        function fileNameForSearch(item) {
            return String((item && item.filename) || '').toLocaleLowerCase();
        }
        function renderLibraryGrid() {
            const grid = get('lib-grid');
            if (!grid) return;
            updateLibFavoriteFilterUi();
            grid.innerHTML = '';
            if (!lib.files || !lib.files.length) {
                grid.innerHTML = '<div class="lib-empty-state"><div class="lib-empty-icon"><i class="fas fa-folder"></i></div><p class="lib-empty-title">ファイルがまだありません</p><p class="lib-empty-sub">アップロードしたファイルがここに表示されます。</p></div>';
                const countEl = get('lib-total-count');
                if (countEl) countEl.innerText = "0 files";
                return;
            }
            const ordered = sortLibraryFiles(lib.files);
            const q = getLibSearchQuery();
            const filtered = ordered.filter((f) => {
                if (lib.favoritesOnly && !f.is_favorite) return false;
                return !q || fileNameForSearch(f).includes(q);
            });
            const countEl = get('lib-total-count');
            if (countEl) {
                if (q || lib.favoritesOnly) countEl.innerText = `${filtered.length} / ${lib.files.length} files`;
                else countEl.innerText = `${lib.files.length} files`;
            }
            if (!filtered.length) {
                const icon = lib.favoritesOnly && !q ? 'fa-star' : 'fa-search';
                const title = lib.favoritesOnly && !q ? 'お気に入りがありません' : '一致するファイルがありません';
                const sub = lib.favoritesOnly && !q ? 'ファイルの星ボタンからお気に入りに追加できます。' : '検索条件や並び順を変更してください。';
                grid.innerHTML = `<div class="lib-empty-state"><div class="lib-empty-icon"><i class="fas ${icon}"></i></div><p class="lib-empty-title">${title}</p><p class="lib-empty-sub">${sub}</p></div>`;
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
