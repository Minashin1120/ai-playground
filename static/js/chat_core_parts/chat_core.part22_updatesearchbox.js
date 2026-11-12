        function updateSearchBox(bubble, box, phase) {
            if (phase !== 'searching' && phase !== 'done') return box;
            if (!box) {
                bubble.insertAdjacentHTML('afterbegin', SEARCH_COMPLETE_HTML);
                box = bubble.querySelector('.search-box');
            }
            box.classList.toggle('animate-pulse', phase === 'searching');
            box.innerHTML = phase === 'searching'
                ? '<i class="fas fa-globe"></i> Searching web...'
                : '<i class="fas fa-check-circle text-green-400"></i> Search complete';
            return box;
        }

        function renderMessage(id, role, text, imgUrl, thoughtData, modelName, versionInfo = null, animate = true, quoteText = null, tokenCount = null, tokenIn = null, tokenOut = null, isEncrypted = null, tokensContent = null, tokensThought = null, target = null, doScroll = true, parentId = null, skillName = null, batchInfo = null, maskImageUrl = null, processingTime = null) {
            const isUser = role === 'user';
            const bg = isUser ? 'bg-blue-600' : 'bg-gray-700';
            const align = isUser ? 'justify-end' : 'justify-start';
            messageStore[id] = text;
            const pythonExtract = (!isUser && text) ? extractPythonExecutionsFromContent(text) : { text: text || '', executions: [] };
            const displayText = isUser ? text : pythonExtract.text;
            let totalTokens = tokenCount;
            if (totalTokens === null || totalTokens === undefined) {
                const inVal = (tokenIn !== null && tokenIn !== undefined) ? Number(tokenIn) : 0;
                const outVal = (tokenOut !== null && tokenOut !== undefined) ? Number(tokenOut) : 0;
                if ((tokenIn !== null && tokenIn !== undefined) || (tokenOut !== null && tokenOut !== undefined)) {
                    totalTokens = inVal + outVal;
                }
            }
            const cacheUsage = costCacheUsage(thoughtData);
            messageMeta[id] = {
                tokens_in: tokenIn,
                tokens_out: tokenOut,
                tokens_total: totalTokens,
                tokens_content: tokensContent,
                tokens_thought: tokensThought,
                cache_read_tokens: cacheUsage ? cacheUsage.read : null,
                cache_write_tokens: cacheUsage ? cacheUsage.write : null,
                cache_write_ttls: cacheUsage ? cacheUsage.ttls : null,
                is_encrypted: isEncrypted,
                role: role,
                model: modelName,
                parent_id: parentId,
                quote_text: quoteText,
                image_url: imgUrl,
                image_mask: maskImageUrl,
                skill_name: skillName,
                batch_job: batchInfo,
                processing_time: processingTime,
                python_executions: isUser ? [] : (pythonExtract.executions || [])
            };

            let qh = '';
            if (quoteText) {
                qh = `<div class="mb-2 p-2 bg-black/20 rounded border-l-4 border-blue-400 text-xs text-gray-300 italic truncate max-w-full"><i class="fas fa-quote-left mr-1 opacity-50"></i>${escapeHtml(quoteText)}</div>`;
            }

            const skillNoticeHtml = skillName
                ? `<div class="skill-notice visible"><i class="fas fa-wand-magic-sparkles"></i><span>Skill: <span class="font-bold">${escapeHtml(skillName)}</span></span></div>`
                : '';

            let th = '';
            if (thoughtData && !isUser) {
                let tx = "";
                try {
                    const o = JSON.parse(thoughtData);
                    tx = o.text || "";
                    if (o.search_used === true) th = SEARCH_COMPLETE_HTML;
                } catch(e) { tx = thoughtData; }
                if (tx) th += `<div class="thought-container"><div class="thought-header" onclick="toggleThinking(this)"><i class="fas fa-brain text-purple-400"></i> Thinking Process</div><div class="thought-content collapsed">${escapeHtml(tx)}</div></div>`;
            }

            let at = '';
            const representedImages = new Set();
            if (!isUser) {
                const visibleText = String(displayText || '').replace(/`{3,}[\s\S]*?`{3,}|~{3,}[\s\S]*?~{3,}|`[^`\n]*`/g, '');
                const refs = Array.from(visibleText.matchAll(/!\[[^\]\n]*\]\(([^)\s]+)\)/g), m => m[1]);
                refs.push(...(pythonExtract.executions || []).flatMap(ex => ex.images || []));
                refs.forEach(ref => {
                    if (/^https?:\/\//i.test(ref) && !ref.startsWith(FILE_BASE_URL)) {
                        try { if (new URL(ref).origin !== window.location.origin) return; }
                        catch (e) { return; }
                    }
                    const path = normalizeAttachmentPath(ref);
                    if (path) representedImages.add(path);
                });
            }
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
                            if (!isUser && representedImages.has(path)) return;
                            setAttachmentSourceForPath(path, source);
                            const displayPath = path.replace(/^\d+\//, '');
                            const url = buildFileUrl(displayPath);
                            const previewUrl = buildAttachmentPreviewUrl(displayPath);
                            const fn = path.split('/').pop();
                            const ext = fn.split('.').pop().toLowerCase();
                            if(['jpg','jpeg','png','webp','gif'].includes(ext)) {
                                items.push(buildChatImageHtml(previewUrl, { viewerSrc: url, alt: fn, title: fn, filename: fn }));
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

            let maskAt = '';
            if (isUser && maskImageUrl) {
                try {
                    const path = normalizeAttachmentPath(maskImageUrl) || String(maskImageUrl);
                    const displayPath = path.replace(/^\d+\//, '');
                    const url = buildFileUrl(displayPath);
                    const previewUrl = buildAttachmentPreviewUrl(displayPath);
                    const filename = path.split('/').pop();
                    maskAt = `<div class="mt-2 max-w-xs rounded-lg border border-amber-300/40 bg-black/10 p-2"><div class="mb-1 text-[10px] font-semibold text-amber-100"><i class="fas fa-mask mr-1" aria-hidden="true"></i>マスク画像（削除範囲）</div>${buildChatImageHtml(previewUrl, { viewerSrc: url, alt: 'Object Remover mask', title: 'マスク画像（削除範囲）', filename })}</div>`;
                } catch (e) {}
            }

            const regenBtn = !isUser ? `<button class="ctrl-btn" onclick="regenerateMessage('${id}')"><i class="fas fa-rotate-right"></i></button>` : '';
            const ctrl = `<div class="msg-controls absolute -top-5 right-0 hidden group-hover:flex gap-1 z-10"><button class="ctrl-btn" onclick="window.copyMessage('${id}', this)"><i class="fas fa-copy"></i></button>${isUser?`<button class="ctrl-btn edit-btn" data-id="${id}"><i class="fas fa-pen"></i></button>`:''}${regenBtn}${/^\d+$/.test(String(id)) ? `<button class="ctrl-btn" title="新しいチャットに分岐" aria-label="新しいチャットに分岐" onclick="branchChatMessage('${id}', this)"><i class="fas fa-code-branch"></i></button>` : ''}<button class="ctrl-btn" onclick="deleteMessage('${id}')"><i class="fas fa-trash"></i></button></div>`;
            const footerParts = [];
            if (!isUser && modelName) footerParts.push(escapeHtml(modelName));
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
                const lockTitle = isAdminUser
                    ? (isEncrypted ? '暗号化状態（タップで復号化）' : '平文状態（タップで再暗号化）')
                    : (isEncrypted ? 'Encrypted' : 'Plain');
                const lockColor = isAdminUser
                    ? (isEncrypted ? 'text-amber-300/90 hover:text-amber-200' : 'text-cyan-300/90 hover:text-cyan-200')
                    : 'text-slate-300/80 hover:text-white';
                footerParts.push(`<button class="${lockColor}" title="${lockTitle}" onclick="openEncryptionSettings('${id}')"><i class="fas ${lockIcon}"></i></button>`);
            }
            if (!isUser && pythonExtract.executions && pythonExtract.executions.length) {
                const pyCount = pythonExtract.executions.length;
                const pyLabel = pyCount > 1 ? `Python ×${pyCount}` : 'Python';
                footerParts.push(`<button type="button" class="python-exec-btn" onclick="openPythonExecDetail('${id}')" title="Python実行結果を表示" aria-label="Python実行結果を表示"><i class="fas fa-terminal"></i><span>${pyLabel}</span></button>`);
            }
            const mHtml = footerParts.length ? `<div class="text-[10px] text-slate-300/90 mt-2 text-right font-mono message-footer-meta">${footerParts.join(' • ')}</div>` : '';
            const processingHtml = isUser ? '' : buildProcessingTimeHtml(id, processingTime);

            let contentHtml;
            const batchStatusHtml = (!isUser && batchInfo) ? (() => {
                const state = String(batchInfo.state || '').toUpperCase();
                const statusText = batchInfo.status_text || (
                    state === 'JOB_STATE_SUCCEEDED' ? 'Batch処理が完了しました' :
                    state === 'JOB_STATE_FAILED' ? 'Batch処理に失敗しました' : 'Batch APIで処理中です'
                );
                const tone = state === 'JOB_STATE_FAILED' || state === 'JOB_STATE_CANCELLED' || state === 'JOB_STATE_EXPIRED'
                    ? 'border-red-400/40 bg-red-950/30 text-red-100'
                    : state === 'JOB_STATE_SUCCEEDED'
                        ? 'border-emerald-400/40 bg-emerald-950/30 text-emerald-100'
                        : 'border-violet-400/40 bg-violet-950/30 text-violet-100';
                return `<div class="batch-status-card mb-3 rounded-lg border ${tone} px-3 py-2 text-xs"><div class="font-semibold"><i class="fas fa-layer-group mr-1"></i>Batch</div><div class="mt-1 opacity-90">${escapeHtml(statusText)}</div></div>`;
            })() : '';
            if (isUser) {
                // User message: RAW TEXT DISPLAY (Preserve whitespace, no markdown)
                contentHtml = `<div class="content-area whitespace-pre-wrap font-sans text-sm break-words">${escapeHtml(text||'')}</div>`;
            } else {
                // AI message: Markdown rendered with tool notices grouped after the prose.
                contentHtml = batchStatusHtml + (displayText && String(displayText).trim()
                    ? buildAiMarkdownHtml(displayText)
                    : (batchInfo ? '<div class="content-area prose prose-invert text-sm break-words text-gray-300">回答を準備しています…</div>' : buildAiMarkdownHtml(displayText)));
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
            msgEl.innerHTML = `<div class="message-bubble ${bg} text-white p-4 rounded-2xl shadow-md relative">${ctrl}${qh}${skillNoticeHtml}${th}${contentHtml}${at}${maskAt}${versionSwitcher}${mHtml}${processingHtml}</div>`;

            const container = target || get('chat-container');
            if (container) {
                container.appendChild(msgEl);
                if (doScroll) scrollToBottom();
                if (!isUser) {
                    if (msgEl.isConnected) queueMessageDecorations(msgEl, displayText);
                    syncCodingTargetButtons(msgEl);
                    syncCodingModeUi(codingModeEnabled, { persist: false });
                }
            }
            return msgEl;
        }

        const PROCESSING_PHASE_LABELS = {
            send: '送信〜受付',
            queue: '待ち行列',
            provider: '初回応答まで',
            generation: '生成',
            receive: '受信・表示'
        };

        function formatProcessingDuration(ms) {
            const value = Number(ms);
            if (!Number.isFinite(value) || value < 0) return '-';
            if (value < 60000) return `${(value / 1000).toFixed(2)}秒`;
            const seconds = value / 1000;
            const minutes = Math.floor(seconds / 60);
            return `${minutes}分${(seconds - minutes * 60).toFixed(1)}秒`;
        }

        function buildProcessingTimeHtml(id, processingTime) {
            if (!processingTime || processingTime.total_ms === null || processingTime.total_ms === undefined) return '';
            const totalText = formatProcessingDuration(processingTime.total_ms);
            return `<div class="mt-1 text-right font-mono text-[10px] text-slate-400/80 processing-time-row">`
                + `<button type="button" class="underline decoration-dotted hover:text-white processing-time-btn" onclick="openProcessingTimeDetail('${id}')" title="回答処理時間の内訳" aria-label="回答処理時間の内訳">処理時間 ${escapeHtml(totalText)}</button>`
                + `</div>`;
        }

        function openProcessingTimeDetail(id) {
            const meta = messageMeta[id];
            const processing = meta && meta.processing_time;
            if (!processing || processing.total_ms === null || processing.total_ms === undefined) return;
            const modal = get('processing-time-modal');
            if (!modal) return;
            get('processing-time-total').textContent = formatProcessingDuration(processing.total_ms);
            const firstRow = get('processing-time-first-row');
            if (firstRow) {
                const hasFirst = processing.first_response_ms !== null && processing.first_response_ms !== undefined;
                firstRow.classList.toggle('hidden', !hasFirst);
                if (hasFirst && get('processing-time-first')) {
                    get('processing-time-first').textContent = formatProcessingDuration(processing.first_response_ms);
                }
            }
            const phasesEl = get('processing-time-phases');
            if (phasesEl) {
                phasesEl.replaceChildren();
                const phases = Array.isArray(processing.phases) ? processing.phases : [];
                for (const phase of phases) {
                    if (!phase || phase.ms === null || phase.ms === undefined) continue;
                    const label = PROCESSING_PHASE_LABELS[phase.key] || phase.key;
                    const row = document.createElement('div');
                    row.className = 'flex justify-between gap-4';
                    const name = document.createElement('span');
                    name.textContent = label;
                    const value = document.createElement('span');
                    value.className = 'font-mono';
                    value.textContent = formatProcessingDuration(phase.ms);
                    row.appendChild(name);
                    row.appendChild(value);
                    phasesEl.appendChild(row);
                }
                phasesEl.classList.toggle('hidden', phasesEl.childElementCount === 0);
            }
            const note = get('processing-time-note');
            if (note) {
                if (processing.source === 'client') {
                    note.textContent = '回答完了時に端末で記録した時間です。詳細な内訳は履歴の再読み込み後に表示されます。';
                } else if (processing.phases && processing.phases.length) {
                    note.textContent = '記録できた処理区間の内訳です。';
                } else {
                    note.textContent = 'この回答では合計時間だけを記録しています。';
                }
            }
            const title = get('processing-time-title');
            if (title) title.textContent = meta && meta.model ? `${meta.model} (${meta.role || 'assistant'})` : '回答処理時間';
            showModal('processing-time-modal');
        }

        function closeProcessingTimeDetail() {
            hideModal('processing-time-modal');
        }

        function showTokenDetailModal(messageId = null, promptEstimate = false) {
            if (location.pathname !== '/token-details') {
                const state = { modal: 'token-details' };
                if (promptEstimate) state.promptEstimate = true;
                if (messageId !== null) state.messageId = messageId;
                history.pushState(state, '', '/token-details');
            }
            showModal('token-detail-modal');
        }

        function openTokenDetail(id) {
            const meta = messageMeta[id];
            if (!meta) return;
            const modal = get('token-detail-modal');
            if (!modal) return;
            get('token-detail-usage').classList.remove('hidden');
            get('token-detail-estimate').classList.add('hidden');
            cancelPromptTokenDetail();
            const total = meta.tokens_total !== null && meta.tokens_total !== undefined ? meta.tokens_total : '-';
            const tIn = meta.tokens_in !== null && meta.tokens_in !== undefined ? meta.tokens_in : '-';
            const tOut = meta.tokens_out !== null && meta.tokens_out !== undefined ? meta.tokens_out : '-';
            const tContent = meta.tokens_content !== null && meta.tokens_content !== undefined ? meta.tokens_content : '-';
            const tThought = meta.tokens_thought !== null && meta.tokens_thought !== undefined ? meta.tokens_thought : '-';
            const enc = meta.is_encrypted === null || meta.is_encrypted === undefined ? '-' : (meta.is_encrypted ? 'Encrypted' : 'Plain');
            const cacheRead = meta.cache_read_tokens !== null && meta.cache_read_tokens !== undefined ? meta.cache_read_tokens : null;
            const cacheWrite = meta.cache_write_tokens !== null && meta.cache_write_tokens !== undefined ? meta.cache_write_tokens : null;
            const hasCache = cacheRead !== null || cacheWrite !== null;
            get('token-detail-total').innerText = total;
            get('token-detail-in').innerText = tIn;
            get('token-detail-out').innerText = tOut;
            get('token-detail-content').innerText = tContent;
            get('token-detail-thought').innerText = tThought;
            get('token-detail-encrypted').innerText = enc;
            const cacheReadRow = get('token-detail-cache-read-row');
            const cacheWriteRow = get('token-detail-cache-write-row');
            const cacheTtlRow = get('token-detail-cache-write-ttl');
            if (cacheReadRow) cacheReadRow.classList.toggle('hidden', !hasCache);
            if (cacheWriteRow) cacheWriteRow.classList.toggle('hidden', !hasCache);
            if (get('token-detail-cache-read')) get('token-detail-cache-read').innerText = cacheRead === null ? '-' : cacheRead;
            if (get('token-detail-cache-write')) get('token-detail-cache-write').innerText = cacheWrite === null ? '-' : cacheWrite;
            if (cacheTtlRow) {
                const ttls = ['5m', '1h', '30m', '12h'].map(ttl => {
                    const value = Number(meta.cache_write_ttls?.[ttl] || 0);
                    return value > 0 ? `${ttl}: ${value}` : '';
                }).filter(Boolean);
                cacheTtlRow.textContent = ttls.length ? `TTL ${ttls.join(' · ')}` : '';
                cacheTtlRow.classList.toggle('hidden', !hasCache || !ttls.length);
            }
            const costDetail = get('token-detail-cost');
            if (costDetail) {
                costDetail.replaceChildren();
                costDetail.classList.toggle('hidden', meta.role !== 'total');
                if (meta.role === 'total') {
                    for (const text of [costEstimateLabel(meta.cost_estimate),
                        `集計したAPI呼び出し: ${meta.cost_estimate?.requests || 0}回`, COST_ESTIMATE_NOTE,
                        `公式料金確認日: ${costPricing?.checked_at || '取得できませんでした'}`]) {
                        const p = document.createElement('p'); p.textContent = text; costDetail.appendChild(p);
                    }
                    for (const [provider, url] of Object.entries(costPricing?.sources || {})) {
                        const link = document.createElement('a'); link.href = url; link.textContent = `${provider} 公式料金 `;
                        link.target = '_blank'; link.rel = 'noopener noreferrer'; link.className = 'underline'; costDetail.appendChild(link);
                    }
                }
            }
            const title = meta.model ? `${meta.model} (${meta.role})` : `${meta.role}`;
            get('token-detail-title').innerText = title;
            showTokenDetailModal(id);
        }

        function closeTokenDetail(skipHistory = false) {
            cancelPromptTokenDetail();
            const modal = get('token-detail-modal');
            if (!modal) return;
            hideModal('token-detail-modal');
            if (!skipHistory && location.pathname === '/token-details') {
                history.back();
            }
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
            const adminBox = get('encryption-status-admin-actions');
            const adminBtn = get('encryption-status-admin-toggle');
            const enc = !!isEncrypted;
            if (enc) {
                if (title) title.innerText = '暗号化されています';
                if (body) {
                    body.innerText = isAdminUser
                        ? 'このメッセージはE2EEで暗号化されています。管理者は下のボタンでこのチャット全体を復号化できます。'
                        : 'このメッセージはE2EEで暗号化されています。';
                }
            } else {
                if (title) title.innerText = '暗号化されていません';
                if (body) {
                    body.innerText = isAdminUser
                        ? 'このメッセージは暗号化されていません。管理者は下のボタンでこのチャット全体を再暗号化できます。'
                        : 'このメッセージは暗号化されていません。';
                }
            }
            if (adminBox && adminBtn) {
                const canToggle = !!(isAdminUser && currentThreadId);
                if (canToggle) {
                    adminBox.classList.remove('hidden');
                    // Encrypted message → offer decrypt (enable=false). Plain → re-encrypt (enable=true).
                    adminBtn.dataset.enable = enc ? '0' : '1';
                    adminBtn.disabled = false;
                    adminBtn.textContent = enc ? 'このチャットを復号化' : 'このチャットを再暗号化';
                    adminBtn.className = enc
                        ? 'w-full px-3 py-2 text-xs font-bold rounded text-white bg-amber-600 hover:bg-amber-500 btn-hover'
                        : 'w-full px-3 py-2 text-xs font-bold rounded text-white bg-cyan-700 hover:bg-cyan-600 btn-hover';
                } else {
                    adminBox.classList.add('hidden');
                }
            }
            showEncryptionStatusModal();
        }

        function showEncryptionStatusModal() {
            if (location.pathname !== '/encryption-status') {
                history.pushState({ modal: 'encryption-status' }, '', '/encryption-status');
            }
            showModal('encryption-status-modal');
        }

        async function toggleThreadEncryptionFromModal() {
            const adminBtn = get('encryption-status-admin-toggle');
            if (!adminBtn || !isAdminUser || !currentThreadId) return;
            if (adminBtn.disabled) return;
            const enable = adminBtn.getAttribute('data-enable') === '1';
            const action = enable ? '再暗号化' : '復号化';
            if (!confirm(`このチャットを${action}しますか？`)) return;
            adminBtn.disabled = true;
            const original = adminBtn.textContent;
            adminBtn.textContent = '処理中...';
            try {
                if (typeof window.__setAdminThreadEncryption !== 'function') {
                    showToast('暗号化操作を利用できません', 'error', true);
                    return;
                }
                const ok = await window.__setAdminThreadEncryption(currentThreadId, enable, {
                    confirmPrompt: false,
                    reloadCurrent: true
                });
                if (ok) closeEncryptionModal();
            } finally {
                adminBtn.disabled = false;
                adminBtn.textContent = original;
            }
        }

        function closeEncryptionModal(skipHistory = false) {
            hideModal('encryption-status-modal');
            if (!skipHistory && location.pathname === '/encryption-status') {
                history.back();
            }
        }

        function goToEncryptionSettings() {
            hideModal('encryption-status-modal');
            if (location.pathname === '/encryption-status') {
                history.replaceState({ modal: 'settings', from: '/encryption-status' }, '', '/settings');
            }
            if (typeof openSettingsModal === 'function') {
                openSettingsModal();
                switchTab('security');
                setTimeout(() => {
                    const card = (isAdminUser && get('admin-enc-card')) || get('e2ee-card');
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
            const html = `<div class="flex justify-start mb-4 ai-pending-row ${fadeClass}"><div${idAttr} class="message-bubble ai-pending-bubble bg-gray-700 text-white p-4 rounded-2xl rounded-tl-none shadow-md relative">${skeletonHtml}</div></div>`;
            const container = target || get('chat-container');
            if (!container) return;
            if (typeof container.insertAdjacentHTML === 'function') {
                container.insertAdjacentHTML('beforeend', html);
            } else {
                // renderThreadTree builds into a DocumentFragment, which has no
                // insertAdjacentHTML. Build the node and append it so a pending
                // stream bubble renders on reload during generation.
                const wrap = document.createElement('div');
                wrap.innerHTML = html;
                const node = wrap.firstElementChild;
                if (node) container.appendChild(node);
            }
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
        function visibleSlashCommands(filter = '') {
            const normalized = String(filter || '').toLowerCase();
            return SLASH_COMMANDS.filter((command) => {
                if (command.kind === 'minimal' && !minimalPromptMode) return false;
                return command.label.toLowerCase().includes(normalized)
                    || command.description.toLowerCase().includes(normalized);
            });
        }

        function slashCommandSuggestionFilter(token, value) {
            if (String(token || '').toLowerCase() !== 'thinking') return token;
            const trimmed = String(value || '').trimStart();
            const match = trimmed.match(/^\/thinking(\s+.*)$/i);
            return match ? `thinking${match[1]}`.toLowerCase() : token;
        }

        function parseSlashToggleArgument(argument) {
            const value = String(argument || '').trim().toLowerCase();
            if (!value || value === 'toggle' || value === '切替' || value === '切り替え') return null;
            if (['on', 'true', '1', 'オン', '有効'].includes(value)) return true;
            if (['off', 'false', '0', 'オフ', '無効'].includes(value)) return false;
            return undefined;
        }

        function executeMinimalSlashCommand(commandId, argument = '') {
            const command = MINIMAL_SLASH_COMMANDS.find((item) => item.id === commandId);
            if (!command || !minimalPromptMode) return false;
            if (command.action === 'options') {
                openMinimalOptions();
                return true;
            }
            const item = MINIMAL_POPUP_ITEMS.find((candidate) => candidate.key === command.itemKey);
            if (!item || !minimalOptionVisible(item)) {
                showToast(`/${commandId} は現在のモデルでは利用できません`, 'warning');
                return true;
            }
            if (minimalOptionDisabled(item) && item.special !== 'thinking') {
                showToast(`/${commandId} は現在変更できません`, 'warning');
                return true;
            }

            const rawArgument = String(command.presetArgument || argument || '').trim();
            if (item.selectId) {
                if (!rawArgument) {
                    showToast(`使い方: ${command.label} default / none`, 'info');
                    return false;
                }
                const select = get(item.selectId);
                const normalized = rawArgument.toLowerCase();
                const option = select ? Array.from(select.options).find((candidate) =>
                    candidate.value.toLowerCase() === normalized || candidate.textContent.trim().toLowerCase() === normalized
                ) : null;
                if (!select || !option) {
                    showToast(`${command.label}: 指定値「${rawArgument}」は利用できません`, 'warning');
                    return false;
                }
                select.value = option.value;
                select.dispatchEvent(new Event('change', { bubbles: true }));
                refreshMinimalOptionItems();
                showToast(`${item.label}: ${option.textContent.trim()}`, 'success');
                return true;
            }

            if (item.special === 'thinking' && rawArgument) {
                const normalized = rawArgument.toLowerCase();
                const level = activeThinkingSelect();
                const isEffort = !!(level && level.id === 'reasoning-effort');
                const levelAliases = isEffort
                    ? { off: 'none', none: 'none', min: 'low', minimal: 'low', low: 'low', mid: 'medium', medium: 'medium', high: 'high', xhigh: 'xhigh', max: 'max' }
                    : { min: 'minimal', minimal: 'minimal', low: 'low', mid: 'medium', medium: 'medium', high: 'high' };
                const desired = parseSlashToggleArgument(rawArgument);
                const checkbox = get(item.checkboxId);
                if (Object.prototype.hasOwnProperty.call(levelAliases, normalized)) {
                    const targetValue = levelAliases[normalized];
                    const targetOption = level ? Array.from(level.options).find((opt) =>
                        opt.value === targetValue && !opt.disabled && !opt.classList.contains('hidden')
                    ) : null;
                    if (level && !targetOption) {
                        showToast(`Thinking: 「${normalized}」はこのモデルでは利用できません`, 'warning');
                        return false;
                    }
                    if (targetValue === 'none') {
                        if (checkbox && checkbox.disabled) {
                            showToast('Thinking: このモデルではOFFにできません', 'warning');
                            return false;
                        }
                        if (checkbox && checkbox.checked) {
                            checkbox.checked = false;
                            checkbox.dispatchEvent(new Event('change', { bubbles: true }));
                        } else if (level) {
                            level.value = 'none';
                            level.dispatchEvent(new Event('change', { bubbles: true }));
                        }
                    } else {
                        if (checkbox && !checkbox.checked && !checkbox.disabled) {
                            checkbox.checked = true;
                            checkbox.dispatchEvent(new Event('change', { bubbles: true }));
                        }
                        if (level) {
                            level.value = targetValue;
                            level.dispatchEvent(new Event('change', { bubbles: true }));
                        }
                    }
                    refreshMinimalOptionItems();
                    showToast(`Thinking: ${normalized}`, 'success');
                    return true;
                }
                if (desired === undefined) {
                    showToast(isEffort
                        ? '使い方: /thinking on / off / min / low / mid / high / xhigh / max'
                        : '使い方: /thinking on / off / min / low / mid / high', 'info');
                    return false;
                }
            }

            if (item.checkboxId && rawArgument) {
                const desired = parseSlashToggleArgument(rawArgument);
                if (desired === undefined) {
                    showToast(`使い方: ${command.label} on / off`, 'info');
                    return false;
                }
                const checkbox = get(item.checkboxId);
                if (desired !== null && checkbox && checkbox.checked === desired) {
                    showToast(`${item.label}: ${desired ? 'ON' : 'OFF'}`, 'info');
                    return true;
                }
            }

            handleMinimalOptionClick(item);
            return true;
        }

        // Extract the leading slash command token (Latin word chars) from the
        // input, so text typed right after the command name without a space
        // (e.g. "/settingsデフォルトモデルを...") does not prevent the palette
        // from recognizing the command. Returns null when the input does not
        // begin with '/'.
        function extractSlashCommandToken(val) {
            const trimmed = String(val || '').trimStart();
            if (!trimmed.startsWith('/')) return null;
            const token = trimmed.substring(1).split(/\s+/)[0] || '';
            const m = token.match(/^[a-z][\w-]*/i);
            return m ? m[0] : token;
        }

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
                if (cmd) {
                    input.dataset.originalPlaceholder = input.placeholder;
                    input.placeholder = cmd.argumentHint || '設定変更の指示を入力（例: デフォルトモデルをgemini-2.5-flashに変更）...';
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

            const wasSettingsCommand = pendingSlashCommand === 'settings';
            pendingSlashCommand = null;
            if (wasSettingsCommand) clearAiSettingsConversation();
        }

        function showSlashCommandSuggestions(filter = '') {
            const box = get('slash-command-suggestions');
            const listEl = get('slash-command-list');
            const inputRow = get('input-row');
            if (!box || !listEl || !inputRow) return;

            const filtered = visibleSlashCommands(filter);

            if (filtered.length === 0) {
                hideSlashCommandSuggestions();
                return;
            }
            slashSelectedIndex = Math.min(slashSelectedIndex, filtered.length - 1);

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
                let selectedByPointer = false;
                item.addEventListener('pointerdown', (event) => {
                    if (typeof event.button === 'number' && event.button !== 0) return;
                    // On touch devices, textarea blur can hide and remove this
                    // item before the delayed click event is dispatched.
                    event.preventDefault();
                    selectedByPointer = true;
                    selectSlashCommand(cmd.id);
                });
                item.addEventListener('click', (event) => {
                    event.preventDefault();
                    if (!selectedByPointer) selectSlashCommand(cmd.id);
                });
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
