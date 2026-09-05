
        async function sendBrowserFastMessage(rawText) {
            const model = String(get('model-select').value || '').trim();
            const bootstrap = await fetchBrowserFastBootstrap(false);
            if (!browserFastApiKey || browserFastApiKeyModel !== model) {
                throw new Error('選択中モデルの保存済みGemini APIキーを取得できませんでした');
            }
            const localEntries = Array.from(browserFastLocalFiles.values());
            const userParts = [];
            for (const entry of localEntries) {
                userParts.push({ inlineData: { mimeType: entry.file.type, data: await fileToBase64Payload(entry.file) } });
            }
            userParts.push({ text: rawText });
            const generationConfig = {};
            const thinkingConfig = browserFastThinkingConfig(model.toLowerCase());
            if (thinkingConfig) generationConfig.thinkingConfig = thinkingConfig;
            const payload = {
                contents: [
                    ...(await buildBrowserFastHistoryContents(bootstrap.history)),
                    { role: 'user', parts: userParts },
                ],
                generationConfig,
            };
            const fastPythonEnabled = !!(get('enable-python') && get('enable-python').checked);
            if (fastPythonEnabled) {
                payload.tools = [{ codeExecution: {} }];
            }

            if (rawText.trim() && (promptHistory.length === 0 || promptHistory[0] !== rawText)) {
                promptHistory.unshift(rawText);
                if (promptHistory.length > 100) promptHistory.pop();
            }
            historyIndex = -1;
            tempPrompt = '';

            playSendAnimation();
            get('welcome-screen').classList.add('hidden');
            renderMessage(Date.now(), 'user', rawText, null, null, null, null, true, null, null, null, null, null, null, null, null, true);
            const aid = `browser-fast-${Date.now()}`;
            get('chat-container').insertAdjacentHTML('beforeend', `<div class="flex justify-start mb-4 fade-in"><div id="${aid}" class="message-bubble ai-pending-bubble bg-gray-700 text-white p-4 rounded-2xl rounded-tl-none shadow-md relative">${buildPendingSkeletonHtml(model, 'Geminiへ直接送信中...')}</div></div>`);
            const adiv = get(aid);
            activeStreamingBubbleId = aid;
            setSendBtnToStopMode();
            resumeChatAutoScroll();
            abortController = new AbortController();
            let content = '';
            let thought = '';
            const thoughtSignatures = [];
            let contentEl = null;
            let thoughtEl = null;
            let started = false;
            const pyBoxes = {};
            const pyExecPayloads = [];
            let currentPyId = null;
            let currentPyCode = '';
            const finishProgress = window.ProgressSpinner ? window.ProgressSpinner.startFlow('browserFast') : null;
            let browserFastOpStarted = false;
            try {
                const response = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:streamGenerateContent?alt=sse`, manualSpinnerRequestOptions({
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json', 'x-goog-api-key': browserFastApiKey },
                    body: JSON.stringify(payload),
                    signal: abortController.signal,
                }));
                if (!response.ok) {
                    const errorData = await response.json().catch(() => ({}));
                    throw new Error(errorData && errorData.error && errorData.error.message ? errorData.error.message : `Gemini API HTTP ${response.status}`);
                }
                if (window.ConnectionMonitor) {
                    browserFastOpStarted = true;
                    window.ConnectionMonitor.operationStarted();
                }
                if (finishProgress) finishProgress.setPhase('waiting');
                get('prompt-input').value = '';
                get('prompt-input').style.height = 'auto';
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';
                const consumeEvent = (block) => {
                    const dataText = block.split(/\r?\n/).filter((line) => line.startsWith('data:')).map((line) => line.slice(5).trim()).join('');
                    if (!dataText || dataText === '[DONE]') return;
                    const evt = JSON.parse(dataText);
                    if (evt.error) throw new Error(evt.error.message || 'Gemini API error');
                    const candidates = Array.isArray(evt.candidates) ? evt.candidates : [];
                    candidates.forEach((candidate) => {
                        const parts = candidate && candidate.content && Array.isArray(candidate.content.parts) ? candidate.content.parts : [];
                        parts.forEach((part) => {
                            if (part && typeof part.thoughtSignature === 'string' && !thoughtSignatures.includes(part.thoughtSignature)) {
                                thoughtSignatures.push(part.thoughtSignature);
                            }
                            if (part && part.executableCode && typeof part.executableCode.code === 'string') {
                                const pyCode = part.executableCode.code;
                                content += `\n\`\`\`python\n${pyCode}\n\`\`\`\n`;
                                currentPyId = `browserFastPy_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
                                currentPyCode = pyCode;
                                if (!pyBoxes[currentPyId]) {
                                    adiv.insertAdjacentHTML('afterbegin', browserFastPythonBoxHtml(currentPyId));
                                    pyBoxes[currentPyId] = adiv.querySelector(`[data-py-id="${currentPyId}"]`);
                                }
                                updateBrowserFastPythonBox(pyBoxes[currentPyId], 'code', pyCode);
                                return;
                            }
                            if (part && part.codeExecutionResult && typeof part.codeExecutionResult.output === 'string') {
                                const pyOutput = part.codeExecutionResult.output;
                                content += `\n**Output:**\n\`\`\`\n${pyOutput}\n\`\`\`\n`;
                                const pyId = currentPyId || `browserFastPy_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
                                pyExecPayloads.push({ code: currentPyCode || '', output: pyOutput });
                                if (!pyBoxes[pyId]) {
                                    adiv.insertAdjacentHTML('afterbegin', browserFastPythonBoxHtml(pyId));
                                    pyBoxes[pyId] = adiv.querySelector(`[data-py-id="${pyId}"]`);
                                }
                                updateBrowserFastPythonBox(pyBoxes[pyId], 'output', pyOutput);
                                return;
                            }
                            const text = typeof part.text === 'string' ? part.text : '';
                            if (!text) return;
                            if (part.thought === true) thought += text;
                            else content += text;
                        });
                    });
                    if (!started && (content || thought)) {
                        beginPendingToStreamTransition(adiv);
                        const pending = adiv.querySelector('.content-area');
                        if (pending) pending.remove();
                        started = true;
                    }
                    if (thought) {
                        if (!thoughtEl) {
                            adiv.insertAdjacentHTML('afterbegin', '<div class="thought-container"><div class="thought-header" onclick="toggleThinking(this)"><i class="fas fa-brain text-purple-400"></i> Thinking Process</div><div class="thought-content"></div></div>');
                            thoughtEl = adiv.querySelector('.thought-content');
                        }
                        thoughtEl.textContent = thought;
                    }
                    if (content) {
                        if (!contentEl) {
                            contentEl = document.createElement('div');
                            contentEl.className = 'content-area prose prose-invert text-sm break-words';
                            adiv.appendChild(contentEl);
                        }
                        renderAiMarkdownInto(contentEl, content, { incrementalMath: true });
                    }
                    scrollToBottom();
                };
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    if (window.ConnectionMonitor) window.ConnectionMonitor.reportActivity();
                    if (finishProgress) finishProgress.setPhase('receiving');
                    buffer += decoder.decode(value, { stream: true });
                    const blocks = buffer.split(/\r?\n\r?\n/);
                    buffer = blocks.pop() || '';
                    blocks.forEach(consumeEvent);
                }
                buffer += decoder.decode();
                if (buffer.trim()) consumeEvent(buffer);
                if (!content.trim()) throw new Error('Geminiから回答本文が返されませんでした');
                if (contentEl) renderAiMarkdownInto(contentEl, content, { incrementalMath: true });
                if (thoughtEl) thoughtEl.classList.add('collapsed');
                if (pyExecPayloads.length) {
                    content += pyExecPayloads.map((payload) => `\n\`\`\`pyexec\n${JSON.stringify(payload)}\n\`\`\`\n`).join('');
                }

                if (localEntries.length) {
                    if (finishProgress) finishProgress.setPhase('saving');
                    showToast('回答が完了しました。画像と履歴をサーバーへ保存しています。', 'info', false);
                    await uploadBrowserFastLocalFiles();
                }
                if (finishProgress) finishProgress.setPhase('saving');
                const refs = collectImageUrlsForSend();
                const saveResponse = await fetchChatStreamWithUnavailableRetry('/api/browser_fast_mode/save', manualSpinnerRequestOptions({
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        client_request_id: createClientRequestId(),
                        message: rawText,
                        assistant_content: content,
                        thought_content: thought,
                        model,
                        image_urls: refs,
                        temporary_chat: temporaryChatEnabled,
                        thread_id: currentThreadId || null,
                        parent_id: bootstrap.parent_id || null,
                        thought_signatures: thoughtSignatures,
                        turnstile_token: botTurnstileTokenForRequest(),
                    }),
                    signal: abortController.signal,
                }), adiv);
                const saved = await saveResponse.json().catch(() => ({}));
                if (!saveResponse.ok || !saved.thread_id) throw new Error(saved.error || 'DB保存に失敗しました');
                const createdThread = !currentThreadId;
                currentThreadId = String(saved.thread_id);
                currentParentId = saved.assistant_message_id || null;
                currentLeafId = saved.assistant_message_id || null;
                resetUploadState();
                browserFastBootstrap = null;
                await loadMessages(currentThreadId, { preserveDraft: true, silent: true, skipHistory: !createdThread });
                applyBrowserFastModeRestrictions();
                loadThreads(false);
                showToast('高速モードの回答を履歴へ保存しました', 'success', false);
            } catch (error) {
                if (error.name !== 'AbortError') {
                    showToast(`高速モード: ${error.message}`, 'error', true);
                    if (!get('prompt-input').value) get('prompt-input').value = rawText;
                    const errMsg = error.message || 'エラー';
                    if (adiv) adiv.insertAdjacentHTML('beforeend', buildChatErrorBubbleHtml(errMsg));
                    // Persist the error (and any partial answer) so it remains after reload.
                    try {
                        let partial = content || '';
                        if (pyExecPayloads.length) {
                            partial += pyExecPayloads.map((payload) => `\n\`\`\`pyexec\n${JSON.stringify(payload)}\n\`\`\`\n`).join('');
                        }
                        const assistantContent = buildChatErrorMarkdown(errMsg, partial);
                        // Local images may not be on the server yet; skip attachments on error path.
                        const refs = localEntries.length ? [] : collectImageUrlsForSend();
                        const saveResponse = await fetchChatStreamWithUnavailableRetry('/api/browser_fast_mode/save', manualSpinnerRequestOptions({
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                client_request_id: createClientRequestId(),
                                message: rawText,
                                assistant_content: assistantContent,
                                thought_content: thought || '',
                                model,
                                image_urls: refs,
                                temporary_chat: temporaryChatEnabled,
                                thread_id: currentThreadId || null,
                                parent_id: bootstrap && bootstrap.parent_id ? bootstrap.parent_id : null,
                                thought_signatures: thoughtSignatures,
                                turnstile_token: botTurnstileTokenForRequest(),
                            }),
                            signal: abortController && !abortController.signal.aborted ? abortController.signal : undefined,
                        }), adiv);
                        const saved = await saveResponse.json().catch(() => ({}));
                        if (saveResponse.ok && saved.thread_id) {
                            const createdThread = !currentThreadId;
                            currentThreadId = String(saved.thread_id);
                            currentParentId = saved.assistant_message_id || null;
                            currentLeafId = saved.assistant_message_id || null;
                            resetUploadState();
                            browserFastBootstrap = null;
                            await loadMessages(currentThreadId, { preserveDraft: true, silent: true, skipHistory: !createdThread });
                            applyBrowserFastModeRestrictions();
                            loadThreads(false);
                        }
                    } catch (persistError) {
                        // Keep the live error bubble even if persistence fails.
                        sendClientDebugLog('error', `Browser fast error persist failed: ${persistError && persistError.message ? persistError.message : persistError}`);
                    }
                }
            } finally {
                if (browserFastOpStarted && window.ConnectionMonitor) window.ConnectionMonitor.operationEnded();
                if (finishProgress) finishProgress();
                setSendBtnToSendMode();
                if (activeStreamingBubbleId === aid) activeStreamingBubbleId = null;
                abortController = null;
                updateFilePreview();
            }
        }

        async function sendMessage() {
            vibrateHelper(50);
            if (abortController) {
                showToast("回答生成中です。完了までお待ちいただくか、停止してください。", "warning", true);
                return;
            }
            if (uploadProgressState.active > 0) {
                showToast("ファイルの送信・処理中です。しばらくお待ちください。", "warning", true);
                return;
            }
            // Lyria RealTime has no text-chat generation; route to the studio.
            if (isLyriaRealtimeModel()) {
                const promptText = get('prompt-input').value;
                get('prompt-input').value = '';
                get('prompt-input').style.height = 'auto';
                if (window.openLyriaStudio) window.openLyriaStudio(promptText);
                return;
            }
            // Rapid send-button clicking would fire many chat_stream requests
            // and load the server, so once the threshold is hit we lock the
            // account (10 min) with a visible reason. This applies even to
            // already-verified users so DOM-driven rapid clicking is caught too.
            // Threshold is intentionally lenient (8 presses / 3s) so normal
            // retries and double-taps do not immediately lock the account.
            if (isBotDetectionActive()) {
                const sendCount = registerSendButtonSpam();
                if (sendCount >= 8) {
                    const ok = await runSendSpamVerification();
                    if (!ok) {
                        showToast("送信操作が速すぎるため、確認後に再度お試しください。", "warning", true);
                        return;
                    }
                }
            }
            let botTurnstileToken = null;
            if (isBotDetectionActive()) {
                botTurnstileToken = await getTurnstileToken();
                if (!botTurnstileToken && !botDetectionVerified) {
                    // Token could not be obtained and the user is not yet verified.
                    // Run the gate (which shows the dialog only when suspicious)
                    // before giving up, so we never accumulate turnstile failures
                    // towards a ban without the dialog ever being shown.
                    try { await runBotDetectionGate(); } catch (e) {}
                    botTurnstileToken = await getTurnstileToken();
                }
                if (!botTurnstileToken && !botDetectionVerified) {
                    showToast("安全性の確認を完了できませんでした。しばらく待ってから再送信してください。", "error", true);
                    botTelemetry.send(true);
                    return;
                }
                if (botTurnstileToken) await verifyTurnstileOnServer(botTurnstileToken);
            }
            const rawText = get('prompt-input').value; // RAW INPUT (No trim)

            // Handle pending slash command (e.g. after selecting /settings via the palette)
            if (pendingSlashCommand) {
                const cmd = pendingSlashCommand;
                const instruction = rawText.trim();
                const modelForCmd = get('model-select') ? get('model-select').value : null;

                if (cmd === 'settings') {
                    if (!instruction) {
                        showToast('設定変更の指示を入力してください（例: デフォルトモデルをgemini-2.5-flashに）', 'info');
                        get('prompt-input').focus();
                        return;
                    }
                    if (!modelForCmd) {
                        showToast('モデルを選択してください', 'error', true);
                        return;
                    }

                    get('prompt-input').value = '';
                    get('prompt-input').style.height = 'auto';

                    await runAiSettingsCommand(instruction, modelForCmd);
                }
                return; // Do not treat as normal message
            }

            if (browserFastModeEnabled) {
                const reason = browserFastModeIneligibility(rawText);
                if (!reason) {
                    try {
                        await sendBrowserFastMessage(rawText);
                    } catch (error) {
                        showToast(`高速モード: ${error.message || '開始準備に失敗しました'}`, 'error', true);
                    }
                    return;
                }
                showToast(`高速モード条件外: ${reason}。通常モードへ切り替えます。`, 'warning', true);
                if (browserFastLocalFiles.size) {
                    try { await uploadBrowserFastLocalFiles(); }
                    catch (error) { showToast(error.message || '通常モード用アップロードに失敗しました', 'error', true); return; }
                }
                setBrowserFastModeEnabled(false);
                return sendMessage();
            }

            // Save to prompt history
            if (rawText.trim()) {
                if (promptHistory.length === 0 || promptHistory[0] !== rawText) {
                    promptHistory.unshift(rawText);
                    if (promptHistory.length > 100) promptHistory.pop();
                }
            }
            historyIndex = -1;
            tempPrompt = "";

            const attachmentItemsToSend = collectAttachmentItemsForSend();
            const imageUrlsToSend = attachmentItemsToSend.map((it) => it.path);
            const uploadedImageUrlsToSend = attachmentItemsToSend
                .filter((it) => normalizeAttachmentSource(it.source) === 'upload')
                .map((it) => it.path);
            if (imageUrlsToSend.length > ATTACHMENT_MAX_FILES) {
                showToast(`添付は最大${ATTACHMENT_MAX_FILES}件です。添付を減らして再送してください。`, "error", true);
                return;
            }
            const support = getModelMediaSupport(get('model-select').value);
            const hasAudio = imageUrlsToSend.some((fp) => isAudioPath(fp));
            const hasVideo = imageUrlsToSend.some((fp) => isVideoPath(fp));
            const modelId = (get('model-select').value || '').toLowerCase();
            const pyChk = get('enable-python');
            const pyEnabled = !!(pyChk && pyChk.checked);
            if ((hasAudio && !support.audio) || (hasVideo && !support.video)) {
                showToast("このモデルは音声/動画入力に対応していません", "error", true);
                purgeUnsupportedAttachments(true);
                return;
            }
            if(!rawText.trim() && imageUrlsToSend.length === 0) return; // Check trim only for empty check
            if (isMistralOcrModel(modelId)) {
                const hasOcrUrl = /https?:\/\/\S+/i.test(rawText);
                const unsupported = imageUrlsToSend.filter((fp) => isAudioPath(fp) || isVideoPath(fp));
                if (unsupported.length) {
                    showToast('Mistral OCR は音声・動画に対応していません。PDF / 画像 / DOCX / PPTX を添付してください。', 'error', true);
                    return;
                }
                if (!imageUrlsToSend.length && !hasOcrUrl) {
                    showToast('Mistral OCR は文書専用です。PDF・画像・DOCX・PPTX を添付するか、公開URLを入力してください。', 'error', true);
                    return;
                }
            }

            // === /settings command: natural language settings change via AI (reuses prompt bar model + toggles) ===
            const trimmedRaw = rawText.trim();
            if (/^\/settings(?:\s|$)/i.test(trimmedRaw) && isMistralOcrModel()) {
                showToast('Mistral OCR は設定変更コマンドに使えません。チャットモデルを選んでください。', 'error', true);
                return;
            }
            if (/^\/settings(?:\s|$)/i.test(trimmedRaw)) {
                const instruction = trimmedRaw.replace(/^\/settings\s*/i, '').trim();
                if (!instruction) {
                    showToast('使い方: /settings デフォルトモデルを gemini-2.5-flash に変更して thinking をオンに', 'info');
                    const input = get('prompt-input');
                    input.value = '/settings ';
                    const hintFilter = extractSlashCommandToken(input.value);
                    lastSlashFilter = hintFilter;
                    showSlashCommandSuggestions(hintFilter);
                    input.focus();
                    return;
                }
                const settingsModel = get('model-select') ? get('model-select').value : null;
                if (!settingsModel) {
                    showToast('モデルが選択されていません', 'error', true);
                    return;
                }
                // Optimistic UI: clear input immediately
                get('prompt-input').value = '';
                get('prompt-input').style.height = 'auto';
                await runAiSettingsCommand(instruction, settingsModel);
                return; // Do not proceed to normal chat send
            }

            if (isGeminiLocalPythonMode(modelId, hasAudio, hasVideo, pyEnabled)) {
                const proceed = await confirmGeminiLocalPythonSwitch();
                if (!proceed) return;
            }
            let codingTargetForSend = null;
            let codingCandidatesForSend = [];
            if (codingModeEnabled) {
                const allCandidates = collectCodingCandidates(rawText);
                const promptCandidates = allCandidates.filter(item => item.prompt_source);
                const historyCandidates = allCandidates.filter(item => !item.prompt_source);
                const promptChars = promptCandidates.reduce((sum, item) => sum + String(item.code || '').length, 0);
                if (promptChars > 300000) {
                    showToast('入力内の編集候補コード合計が大きすぎます（上限300,000文字）', 'error', true);
                    return;
                }
                let remainingChars = 300000 - promptChars;
                const selectedHistory = [];
                for (let index = historyCandidates.length - 1; index >= 0; index--) {
                    const candidateLength = String(historyCandidates[index].code || '').length;
                    if (candidateLength > remainingChars) continue;
                    selectedHistory.unshift(historyCandidates[index]);
                    remainingChars -= candidateLength;
                }
                codingCandidatesForSend = codingTargetSelection
                    ? selectedHistory.slice(-1)
                    : [...promptCandidates, ...selectedHistory];
                const latestPromptTarget = promptCandidates.length ? promptCandidates[promptCandidates.length - 1] : null;
                codingTargetForSend = codingTargetSelection
                    ? codingCandidatesForSend[0]
                    : (latestPromptTarget || codingCandidatesForSend[codingCandidatesForSend.length - 1] || null);
                codingModeEffective = !!(codingTargetForSend && String(codingTargetForSend.code || '').trim());
                if (codingModeEffective && codingTargetForSend.code.length > 300000) {
                    showToast('編集対象コードが大きすぎます（上限300,000文字）', 'error', true);
                    return;
                }
                if (codingModeEffective) {
                    const codingModel = String(get('model-select')?.value || '').toLowerCase();
                    if (/(image|video|tts|audio|native-audio)/.test(codingModel)) {
                        showToast('Coding Modeではテキスト生成モデルを選択してください', 'error', true);
                        return;
                    }
                }
            }
            // Freeze the model-facing state for this request. A code fence completed
            // by the streaming response may arm the next request, never this one.
            const codingModeActiveForSend = codingModeEnabled && codingModeEffective;
            sendClientDebugLog(
                'info',
                `Prompt send start: model=${get('model-select').value} thread=${currentThreadId || '-'} text_len=${rawText.length} attachments=${imageUrlsToSend.length} search=${get('enable-search').checked}`
            );
            const t = rawText;
            const markerSysPrompt = hasMarkerHint() ? MARKER_HINT_TEXT : null;
            if (isGptImageModel() && currentMaskImage && imageUrlsToSend.length === 0) {
                showToast("Mask は画像入力が必要です", "error", true);
                return;
            }
            const editingId = editingMessageId;
            const capturedParentId = currentParentId;
            const parentIdExplicit = (editingId !== null && editingId !== undefined);
            if (editingId) {
                editingMessageId = null;
                setEditUi(false);
            }
            playSendAnimation();
            get('welcome-screen').classList.add('hidden');

            const hiddenBranch = [];
            const hideRenderedBranchFrom = (startId) => {
                if (startId === null || startId === undefined) return;
                let node = document.getElementById(`msg-${startId}`);
                while (node) {
                    if (node.classList && node.classList.contains('message-group')) {
                        hiddenBranch.push({ node, prevDisplay: node.style.display });
                        node.style.display = 'none';
                    }
                    node = node.nextElementSibling;
                }
            };
            const restoreHiddenBranch = () => {
                hiddenBranch.forEach(({ node, prevDisplay }) => {
                    if (!node) return;
                    node.style.display = prevDisplay || '';
                });
                hiddenBranch.length = 0;
            };

            // If editing/regenerating, hide the current branch before optimistic render.
            if (editingId) {
                hideRenderedBranchFrom(editingId);
            }

            // Render UI immediately to reduce perceived latency.
            const optimisticUserId = Date.now();
            const optimisticUserMessageEl = renderMessage(optimisticUserId, 'user', t, JSON.stringify(imageUrlsToSend), null, null, null, true, currentQuote, null, null, null, null, null, null, null, true, capturedParentId, activeGem ? activeGem.name : null);

            let disableAutoSearch = false;
            const xLinkPattern = /(https?:\/\/)?(x\.com|twitter\.com)\//i;
            const hasXLink = xLinkPattern.test(t || '') || xLinkPattern.test(currentQuote || '');
            const grokXModel = "grok-4-fast-reasoning";
            const applyXLinkAuto = () => {
                get('enable-search').checked = true;
                if (get('model-select').value !== grokXModel) {
                    selectModelById(grokXModel);
                }
            };
            if (hasXLink && !isMistralOcrModel() && !get('enable-search').checked) {
                if (autoSearchOnLinks) {
                    applyXLinkAuto();
                } else {
                    const banner = get('auto-search-banner');
                    const onBtn = get('auto-search-on-btn');
                    const offBtn = get('auto-search-off-btn');
                    const remember = get('auto-search-remember');
                    if (banner && onBtn && offBtn) {
                        if (remember) remember.checked = false;
                        await new Promise((resolve) => {
                            banner.classList.remove('hidden');
                            const cleanup = (choice) => {
                                banner.classList.add('hidden');
                                onBtn.onclick = null;
                                offBtn.onclick = null;
                                resolve(choice);
                            };
                            onBtn.onclick = () => cleanup('enable');
                            offBtn.onclick = () => cleanup('disable');
                        }).then(async (choice) => {
                            if (choice === 'enable') {
                                applyXLinkAuto();
                                if (remember && remember.checked) {
                                    autoSearchOnLinks = true;
                                    await apiFetch(CHAT_CONFIG.urls.handleSettings, {
                                        method: 'POST',
                                        headers: {'Content-Type': 'application/json'},
                                        body: JSON.stringify({ auto_search_on_links: true })
                                    });
                                }
                            } else {
                                disableAutoSearch = true;
                            }
                        });
                    }
                }
            }

            const effortLower = String(get('reasoning-effort').value || '').toLowerCase();
            const isDeepSeekNonThinking = String(get('model-select').value || '').toLowerCase().includes('deepseek') && effortLower === 'none';
            const p = {
                client_request_id: createClientRequestId(),
                thread_id: currentThreadId,
                message: t,
                model: get('model-select').value,
                image_urls: imageUrlsToSend,
                image_items: attachmentItemsToSend,
                uploaded_image_urls: uploadedImageUrlsToSend,
                temporary_chat: temporaryChatEnabled,
                enable_search: get('enable-search').checked,
                enable_url_context: get('enable-url-context') ? get('enable-url-context').checked : false,
                enable_maps: get('enable-maps') ? get('enable-maps').checked : false,
                enable_python: get('enable-python').checked,
                enable_mcp: isMcpEnabledForSend(),
                enable_file_creation: get('enable-file-creation') ? get('enable-file-creation').checked : true,
                enable_thinking: isDeepSeekNonThinking ? false : get('enable-thinking').checked,
                thinking_level: get('thinking-level').value,
                thinking_budget: get('thinking-budget') ? get('thinking-budget').value : null,
                reasoning_effort: get('reasoning-effort').value,
                enable_system_prompt: get('enable-sys-prompt').checked,
                enable_prompt_caching: get('enable-prompt-cache') ? get('enable-prompt-cache').checked : false,
                marker_system_prompt: markerSysPrompt,
                safety_setting: get('safety-setting').value,
                tts_voice: isTtsModel() && get('tts-voice') ? get('tts-voice').value : null,
                tts_voice_custom: isTtsModel() && get('tts-voice-custom') ? get('tts-voice-custom').value : null,
                tts_language: isTtsModel() && get('tts-language') ? get('tts-language').value : null,
                tts_speed: isTtsModel() && get('tts-speed') ? get('tts-speed').value : null,
                image_size: isGptImageModel() && get('gpt-image-size') ? get('gpt-image-size').value : null,
                image_quality: isGptImageModel() && get('gpt-image-quality') ? get('gpt-image-quality').value : null,
                image_format: isGptImageModel() && get('gpt-image-format') ? get('gpt-image-format').value : null,
                image_compression: isGptImageModel() && get('gpt-image-compression') ? get('gpt-image-compression').value : null,
                image_mask: isGptImageModel() ? currentMaskImage : null,
                gemini_image_aspect: isGeminiImageModel() && get('gemini-image-aspect') ? get('gemini-image-aspect').value : null,
                gemini_image_size: isGeminiImageModel() && get('gemini-image-size') ? get('gemini-image-size').value : null,
                grok_image_aspect: isGrokImageModel() && get('grok-image-aspect') ? get('grok-image-aspect').value : null,
                grok_image_resolution: isGrokImageModel() && get('grok-image-resolution') ? get('grok-image-resolution').value : null,
                grok_image_quality: isGrokImageModel() && get('grok-image-quality') ? get('grok-image-quality').value : null,
                grok_image_format: isGrokImageModel() && get('grok-image-format') ? get('grok-image-format').value : null,
                grok_image_count: isGrokImageModel() && get('grok-image-count') ? get('grok-image-count').value : null,
                xai_temperature: get('xai-temperature') ? get('xai-temperature').value : null,
                xai_top_p: get('xai-top-p') ? get('xai-top-p').value : null,
                xai_max_completion_tokens: get('xai-max-completion-tokens') ? get('xai-max-completion-tokens').value : null,
                xai_seed: get('xai-seed') ? get('xai-seed').value : null,
                xai_presence_penalty: get('xai-presence-penalty') ? get('xai-presence-penalty').value : null,
                xai_frequency_penalty: get('xai-frequency-penalty') ? get('xai-frequency-penalty').value : null,
                xai_stop: get('xai-stop') ? get('xai-stop').value : null,
                xai_response_format: get('xai-response-format') ? get('xai-response-format').value : null,
                xai_tool_choice: get('xai-tool-choice') ? get('xai-tool-choice').value : null,
                xai_parallel_tool_calls: get('xai-parallel-tool-calls') ? get('xai-parallel-tool-calls').checked : true,
                xai_logprobs: get('xai-logprobs') ? get('xai-logprobs').checked : false,
                xai_top_logprobs: get('xai-top-logprobs') ? get('xai-top-logprobs').value : null,
                grok_video_duration: isGrokVideoModel() && get('grok-video-duration') ? get('grok-video-duration').value : null,
                grok_video_aspect: isGrokVideoModel() && get('grok-video-aspect') ? get('grok-video-aspect').value : null,
                grok_video_resolution: isGrokVideoModel() && get('grok-video-resolution') ? get('grok-video-resolution').value : null,
                gemini_video_duration: isGeminiVideoModel() && get('gemini-video-duration') ? get('gemini-video-duration').value : null,
                gemini_video_aspect: isGeminiVideoModel() && get('gemini-video-aspect') ? get('gemini-video-aspect').value : null,
                gemini_video_resolution: isGeminiVideoModel() && get('gemini-video-resolution') ? get('gemini-video-resolution').value : null,
                music_instrumental: isGeminiMusicModel() && get('music-instrumental') ? get('music-instrumental').checked : false,
                ocr_table_format: isMistralOcrModel() && get('ocr-table-format') ? get('ocr-table-format').value : null,
                ocr_extract_header: isMistralOcrModel() && get('ocr-extract-header') ? get('ocr-extract-header').checked : false,
                ocr_extract_footer: isMistralOcrModel() && get('ocr-extract-footer') ? get('ocr-extract-footer').checked : false,
                ocr_include_blocks: isMistralOcrModel() && get('ocr-include-blocks') ? get('ocr-include-blocks').checked : false,
                ocr_include_image_base64: isMistralOcrModel() && get('ocr-include-images') ? get('ocr-include-images').checked : true,
                ocr_pages: isMistralOcrModel() && get('ocr-pages') ? get('ocr-pages').value : null,
                transcription_language_codes: [],
                transcription_custom_vocabulary: [],
                transcription_mode: 'verbatim',
                transcription_diarization: false,
                transcription_word_timestamps: false,
                quote_text: currentQuote,
                parent_id: capturedParentId,
                parent_id_explicit: parentIdExplicit,
                disable_auto_search: disableAutoSearch,
                image_vision_model: currentVisionModel || null,
                coding_mode: codingModeActiveForSend,
                coding_target: codingModeActiveForSend ? {
                    id: codingTargetForSend.candidate_id,
                    code: codingTargetForSend.prompt_source ? null : codingTargetForSend.code,
                    language: codingTargetForSend.language || 'text',
                    key: codingTargetForSend.key || null,
                    message_id: codingTargetForSend.message_id || null,
                    source: codingTargetForSend.prompt_source ? 'prompt' : 'history',
                    explicit: codingTargetForSend.explicit === true
                } : null,
                coding_candidates: codingModeActiveForSend ? codingCandidatesForSend.map((candidate) => ({
                    id: candidate.candidate_id,
                    source: candidate.prompt_source ? 'prompt' : 'history',
                    prompt_index: candidate.prompt_source ? candidate.prompt_index : null,
                    code: candidate.prompt_source ? null : candidate.code,
                    language: candidate.language || 'text',
                    explicit: candidate.explicit === true
                })) : []
            };
            if (botTurnstileToken) p.turnstile_token = botTurnstileToken;
            const threadCustomInstructionEl = get('thread-custom-instruction');
            if (threadCustomInstructionEl) {
                p.thread_custom_instruction = threadCustomInstructionEl.value || '';
            }
            if (activeGem) { p.system_prompt = activeGem.instruction; p.enable_system_prompt = true; p.gem_uuid = activeGem.uuid; } else { p.gem_uuid = null; }
            setSendBtnToStopMode(); const aid = 'ai-' + Date.now();
            const modelLower = String(p.model || '').toLowerCase();
            const reasoningRequested = !!p.enable_thinking || (!!effortLower && effortLower !== 'none');
            const reasoningCapableModel =
                modelLower.includes('gemini') ||
                modelLower.includes('o1') ||
                modelLower.includes('o3') ||
                modelLower.includes('gpt-5') ||
                (modelLower.includes('reasoning') && !modelLower.includes('non-reasoning'));
            const shouldShowReasoningProgress = reasoningRequested && reasoningCapableModel;

            let initialHtml = buildPendingSkeletonHtml(p.model, 'APIに送信中...');
            get('chat-container').insertAdjacentHTML('beforeend', `<div class="flex justify-start mb-4 fade-in"><div id="${aid}" class="message-bubble ai-pending-bubble bg-gray-700 text-white p-4 rounded-2xl rounded-tl-none shadow-md relative">${initialHtml}</div></div>`);
            resumeChatAutoScroll();
            const adiv = get(aid);
            activeStreamingBubbleId = aid;
            if (canvasModeEnabled) {
                resetCanvasPreviewPanel();
            }
            let thoughtPlaceholderEl = null;
            const ensureThoughtPlaceholder = (text) => {
                if (!shouldShowReasoningProgress || !adiv) return null;
                if (!thoughtPlaceholderEl || !adiv.contains(thoughtPlaceholderEl)) {
                    thoughtPlaceholderEl = adiv.querySelector('.thought-content');
                }
                if (!thoughtPlaceholderEl) {
                    const tHtml = `<div class="thought-container"><div class="thought-header thinking-shimmer" onclick="toggleThinking(this)"><i class="fas fa-brain text-purple-400"></i> Thinking Process</div><div class="thought-content collapsed" data-placeholder="1"></div></div>`;
                    adiv.insertAdjacentHTML('afterbegin', tHtml);
                    thoughtPlaceholderEl = adiv.querySelector('.thought-content');
                }
                if (thoughtPlaceholderEl) {
                    thoughtPlaceholderEl.setAttribute('data-placeholder', '1');
                    thoughtPlaceholderEl.textContent = text || '推論プロセスを準備中...';
                }
                return thoughtPlaceholderEl;
            };
            if (shouldShowReasoningProgress) {
                ensureThoughtPlaceholder('推論プロセスを準備中...');
            }

            abortController = new AbortController();
            const streamStartedThreadId = currentThreadId;
            const sendStartPerfMs = nowPerfMs();
            const sendStartEpochMs = Date.now();
            let firstStatusLatencySent = false;
            let firstThoughtLatencySent = false;
            let firstContentLatencySent = false;
            let firstStatusLatencyMs = null;
            let firstThoughtLatencyMs = null;
            let firstContentLatencyMs = null;

            let streamThreadIdForMetric = (currentThreadId !== null && currentThreadId !== undefined && currentThreadId !== '')
                ? String(currentThreadId)
                : null;
            const maybeReportFirstEventLatency = (eventType, shouldReport) => {
                if (!shouldReport) return;
                if (eventType === 'status' && firstStatusLatencySent) return;
                if (eventType === 'thought' && firstThoughtLatencySent) return;
                if (eventType === 'content' && firstContentLatencySent) return;
                const elapsedMs = Math.max(0, nowPerfMs() - sendStartPerfMs);

                if (eventType === 'status') firstStatusLatencyMs = elapsedMs;
                else if (eventType === 'thought') firstThoughtLatencyMs = elapsedMs;
                else if (eventType === 'content') firstContentLatencyMs = elapsedMs;

                reportFirstTokenLatency({
                    latency_seconds: elapsedMs / 1000,
                    latency_ms: elapsedMs,
                    thread_id: streamThreadIdForMetric || currentThreadId,
                    job_id: currentJobId,
                    model: p.model,
                    first_event_type: eventType,
                    client_sent_at_ms: sendStartEpochMs
                });
                if (eventType === 'status') firstStatusLatencySent = true;
                else if (eventType === 'thought') firstThoughtLatencySent = true;
                else if (eventType === 'content') firstContentLatencySent = true;
            };
            // fetch() resolves when response headers arrive, before its streaming body is done.
            // Keep the dedicated progress flow alive until the reader loop reaches EOF.
            const finishStreamProgress = window.ProgressSpinner
                ? window.ProgressSpinner.startFlow('chat')
                : null;
            let requestAccepted = false;
            let retryAfterApiKeySetup = false;
            let resumeAcceptedSubmission = null;
            let reconnectAfterStreamDisconnect = null;
            let streamOpStarted = false;
            try {
                if (p.thread_id && activeGem) {
                    threadGemMap[p.thread_id] = activeGem;
                    pendingGemForNewThread = null;
                }
                const r = await fetchChatStreamWithUnavailableRetry(
                    CHAT_CONFIG.urls.chatStream,
                    manualSpinnerRequestOptions({method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(p), signal:abortController.signal}),
                    adiv
                );
                sendClientDebugLog('info', `Prompt stream response status: ${r.status}`);
                if (!r.ok) {
                    const errorPayload = await r.json().catch(() => ({}));
                    const requestError = new Error(errorPayload.error || `HTTP ${r.status}`);
                    requestError.serverCode = errorPayload.code || null;
                    requestError.serverModel = errorPayload.model || p.model;
                    requestError.acceptedJobId = errorPayload.job_id || null;
                    requestError.acceptedThreadId = errorPayload.thread_id || null;
                    throw requestError;
                }
                requestAccepted = true;
                if (window.ConnectionMonitor) {
                    streamOpStarted = true;
                    window.ConnectionMonitor.operationStarted();
                }
                if (finishStreamProgress) finishStreamProgress.setPhase('waiting');
                get('prompt-input').value = '';
                get('prompt-input').style.height = 'auto';
                schedulePromptTokenEstimate(true);
                if (codingModeEnabled) {
                    syncCodingModeUi(true, { persist: false });
                }
                resetUploadState();
                clearQuote();
                const markApiAccepted = () => {
                    if (!adiv) return;
                    const ca = adiv.querySelector('.content-area');
                    if (!ca) return;
                    if (ca.getAttribute('data-api-accepted') === '1') return;
                    ca.setAttribute('data-api-accepted', '1');
                    if (!updatePendingSkeletonStatus(adiv, '接続完了。モデル応答を待機中...', 'キュー待機や初期化中の可能性があります')) {
                        ca.outerHTML = buildPendingSkeletonHtml(p.model, '接続完了。モデル応答を待機中...');
                        const newCa = adiv.querySelector('.content-area');
                        if (newCa) newCa.setAttribute('data-api-accepted', '1');
                        updatePendingSkeletonStatus(adiv, '接続完了。モデル応答を待機中...', 'キュー待機や初期化中の可能性があります');
                    }
                };
                markApiAccepted();
                const reader = r.body.getReader();
                const dec = new TextDecoder();
                let buf="", acc="", tht="", first=true, thEl=null, cEl=null, searchBox=null, hadError=false;
                const pyBoxes = {};
                let lastRenderTime = 0;
                let streamEndedByError = false;

                while(!streamEndedByError) {
                    const {done, value} = await reader.read();
                    if(done) break;
                    if (window.ConnectionMonitor) window.ConnectionMonitor.reportActivity();
                    if (finishStreamProgress) finishStreamProgress.setPhase('receiving');
                    buf += dec.decode(value, {stream:true});
                    let ls = buf.split("\n");
                    buf = ls.pop();
                    let contentChanged = false;
                    let thoughtChanged = false;
                    for(let l of ls) {
                        if(!l.trim()) continue;
                        try {
                            const j = JSON.parse(l);
                            if (j.type === 'thread_id') {
                                markApiAccepted();
                                const streamThreadId = j.content !== null && j.content !== undefined ? String(j.content) : j.content;
                                if (streamThreadId) {
                                    streamThreadIdForMetric = streamThreadId;
                                    if (currentThreadId !== streamThreadId) {
                                        currentThreadId = streamThreadId;
                                        history.pushState({}, '', '/c/' + streamThreadId);
                                    }
                                    if (activeGem) {
                                        threadGemMap[streamThreadId] = activeGem;
                                        pendingGemForNewThread = null;
                                    }
                                    ensureTemporaryChatHeartbeat(true);
                                }
                                continue;
                            }
                            if (j.type === 'job_id') { markApiAccepted(); currentJobId = j.content; continue; }
                            if (j.type === 'search_status') {
                                if (j.content === 'searching' && !searchBox) {
                                     adiv.insertAdjacentHTML('afterbegin', `<div class="search-box visible animate-pulse mb-2"><i class="fas fa-globe"></i> Searching web...</div>`);
                                     searchBox = adiv.querySelector('.search-box');
                                } else if (j.content === 'done' && searchBox) {
                                     searchBox.classList.remove('animate-pulse');
                                     searchBox.innerHTML = '<i class="fas fa-check-circle text-green-400"></i> Search complete';
                                     setTimeout(() => { if(searchBox) searchBox.remove(); searchBox=null; }, 2000);
                                }
                                continue;
                            }
                            if (j.type === 'mcp') {
                                handleMcpStreamEvent(adiv, j.content || {});
                                continue;
                            }
                            if (j.type === 'mcp_decision_request') {
                                openMcpDecisionModal(j.content || {});
                                continue;
                            }
                            if (j.type === 'status') {
                                markApiAccepted();
                                const statusText = (j.content === null || j.content === undefined) ? '' : String(j.content);
                                maybeReportFirstEventLatency('status', !!statusText);
                                if (first && adiv) {
                                    const headline = statusText || 'モデル処理中...';
                                    if (!updatePendingSkeletonStatus(adiv, headline, '応答開始までの進捗を表示しています')) {
                                        const ca = adiv.querySelector('.content-area');
                                        if (ca) {
                                            ca.outerHTML = buildPendingSkeletonHtml(p.model, headline);
                                            updatePendingSkeletonStatus(adiv, headline, '応答開始までの進捗を表示しています');
                                        }
                                    }
                                }
                                if (shouldShowReasoningProgress) {
                                    ensureThoughtPlaceholder(statusText || '推論プロセスを準備中...');
                                }
                                continue;
                            }
                            if(first){
                                 beginPendingToStreamTransition(adiv);
                                 const ca = adiv.querySelector('.content-area');
                                 if(ca) ca.innerHTML = '';
                                 first=false;
                            }
                            if(j.type==='coding_diff'){
                                appendCodingLiveDiff(adiv, j.content || {});
                                maybeReportFirstEventLatency('content', true);
                            } else if(j.type==='thought'){
                                if(!thEl){
                                    thEl = adiv.querySelector('.thought-content');
                                }
                                tht+=j.content;
                                maybeReportFirstEventLatency('thought', !!j.content);
                                if(!thEl){
                                    const tHtml = `<div class="thought-container"><div class="thought-header" onclick="toggleThinking(this)"><i class="fas fa-brain text-purple-400"></i> Thinking Process</div><div class="thought-content"></div></div>`;
                                    if(searchBox) searchBox.insertAdjacentHTML('afterend', tHtml);
                                    else adiv.insertAdjacentHTML('afterbegin', tHtml);
                                    thEl=adiv.querySelector('.thought-content');
                                }
                                if (thEl && thEl.getAttribute('data-placeholder') === '1') {
                                    thEl.textContent = '';
                                    thEl.removeAttribute('data-placeholder');
                                    if (thEl) {
                                        const thoughtHeader = thEl.parentElement.querySelector('.thought-header');
                                        if (thoughtHeader) thoughtHeader.classList.remove('thinking-shimmer');
                                    }
                                    tht = j.content;
                                }
                                thEl.classList.remove('collapsed');
                                thoughtChanged = true;
                            } else if (j.type === 'image_analysis') {
                                const iaText = (j.content === null || j.content === undefined) ? '' : String(j.content);
                                if (!adiv) continue;
                                let iaEl = adiv.querySelector('.image-analysis-box');
                                if (!iaEl) {
                                    const iaHtml = `<div class="image-analysis-box mb-2 p-2 bg-blue-900/20 border border-blue-500/30 rounded"><div class="text-[10px] text-blue-300 font-medium mb-1"><i class="fas fa-image mr-1"></i>Image Analysis</div><div class="image-analysis-text text-[11px] text-gray-300"></div></div>`;
                                    if(searchBox) searchBox.insertAdjacentHTML('afterend', iaHtml);
                                    else adiv.insertAdjacentHTML('afterbegin', iaHtml);
                                    iaEl = adiv.querySelector('.image-analysis-box');
                                }
                                const iaTxt = iaEl.querySelector('.image-analysis-text');
                                if (iaTxt) iaTxt.textContent = iaText;
                            } else if(j.type==='python'){
                                const py = j.content || {};
                                const pyId = py.id || `py_${Date.now()}`;
                                if(!pyBoxes[pyId]){
                                    const boxHtml = `<div class="code-wrapper python-box collapsed" data-py-id="${pyId}" data-collapsed="true" data-code-key="${pyId}"><div class="code-header"><span class="code-lang"><i class="fas fa-terminal"></i> Python Execution</span><div class="code-actions"><button class="code-toggle" aria-expanded="false" title="展開" aria-label="展開"><i class="fas fa-chevron-down"></i></button><button class="copy-btn" data-copy="code" data-code="" title="コードをコピー" aria-label="コードをコピー"><i class="fas fa-copy"></i></button><button class="copy-btn" data-copy="output" data-code="" title="出力をコピー" aria-label="出力をコピー"><i class="fas fa-align-left"></i></button></div></div><div class="code-body"><div class="python-section"><div class="python-label">Code</div><pre><code class="hljs language-python python-code"></code></pre></div><div class="python-section"><div class="python-label">Output</div><pre><code class="hljs language-plaintext python-output"></code></pre></div></div></div>`;
                                    if(searchBox) searchBox.insertAdjacentHTML('afterend', boxHtml);
                                    else adiv.insertAdjacentHTML('afterbegin', boxHtml);
                                    pyBoxes[pyId] = adiv.querySelector(`[data-py-id="${pyId}"]`);
                                }
                                const box = pyBoxes[pyId];
                                if(box){
                                    if(py.code !== undefined){
                                        const codeText = py.code == null ? '' : String(py.code);
                                        const codeEl = box.querySelector('.python-code');
                                        if (codeEl) {
                                            codeEl.textContent = codeText;
                                            codeEl.removeAttribute('data-highlighted');
                                            queueHighlight(box, codeText);
                                        }
                                        const codeBtn = box.querySelector('.copy-btn[data-copy="code"]');
                                        if (codeBtn) codeBtn.setAttribute('data-code', encodeURIComponent(codeText).replace(/'/g, "%27"));
                                    }
                                    if(py.output !== undefined){
                                        const outText = py.output == null ? '' : String(py.output);
                                        const outEl = box.querySelector('.python-output');
                                        if (outEl) outEl.textContent = outText;
                                        const outBtn = box.querySelector('.copy-btn[data-copy="output"]');
                                        if (outBtn) outBtn.setAttribute('data-code', encodeURIComponent(outText).replace(/'/g, "%27"));
                                    }
                                }
                            } else if(j.type==='content'){
                                const contentDelta = (j.content === null || j.content === undefined) ? '' : String(j.content);
                                acc += contentDelta;
                                if (/[`~]/.test(contentDelta)) activateDeferredCodingModeFromStream(acc);
                                if(!cEl){
                                    cEl = adiv.querySelector('.content-area') || document.createElement('div');
                                    cEl.className='prose prose-invert text-sm break-words';
                                    if(!adiv.contains(cEl)) adiv.appendChild(cEl);
                                }
                                contentChanged = true;
                                maybeReportFirstEventLatency('content', !!contentDelta);
                            } else if(j.type==='error'){
                                hadError = true;
                                streamEndedByError = true;
                                adiv.insertAdjacentHTML('beforeend', buildChatErrorBubbleHtml(j.content));
                                showToast(j.content || "Unknown error", "error", true);
                                break;
                            }
                        } catch(e){}
                    }
                    if (thoughtChanged && thEl) {
                        thEl.textContent = tht;
                        if (userAutoScroll) thEl.scrollTop = thEl.scrollHeight;
                    }
                    if (contentChanged && cEl) {
                        const now = Date.now();
                        if (now - lastRenderTime > 100) {
                            const collapseState = snapshotCodeCollapse(cEl);
                            renderAiMarkdownInto(cEl, acc, { incrementalMath: true });
                            applyCodeCollapse(cEl, collapseState, true);
                            lastRenderTime = now;
                        }
                    }
                    scrollToBottom();
                }
                if (finishStreamProgress) finishStreamProgress();
                // Final render to catch any remaining content
                if (cEl) {
                    const collapseState = snapshotCodeCollapse(cEl);
                    renderAiMarkdownInto(cEl, acc, { incrementalMath: true });
                    applyCodeCollapse(cEl, collapseState, true);
                }
                scrollToBottom();

                vibrateHelper([100, 50, 100]);

                if (adiv) {
                    queueHighlight(adiv, acc);

                    if (enableLatencyMetrics) {
                        const totalLatencyMs = nowPerfMs() - sendStartPerfMs;

                        // Send total latency to server
                        reportFirstTokenLatency({
                            is_total: true,
                            latency_seconds: totalLatencyMs / 1000,
                            latency_ms: totalLatencyMs,
                            thread_id: streamThreadIdForMetric || currentThreadId,
                            job_id: currentJobId,
                            model: p.model,
                            client_sent_at_ms: sendStartEpochMs,
                            client_done_at_ms: Date.now()
                        });

                        let latencyHtml = `<div class="mt-2 pt-2 border-t border-gray-700/30 flex flex-col gap-1 items-end opacity-70 text-[10px] font-mono text-gray-400">`;

                        // First Token
                        let firstAnyMs = null;
                        if (firstStatusLatencyMs !== null) firstAnyMs = firstStatusLatencyMs;
                        if (firstThoughtLatencyMs !== null && (firstAnyMs === null || firstThoughtLatencyMs < firstAnyMs)) firstAnyMs = firstThoughtLatencyMs;
                        if (firstContentLatencyMs !== null && (firstAnyMs === null || firstContentLatencyMs < firstAnyMs)) firstAnyMs = firstContentLatencyMs;

                        if (firstAnyMs !== null) {
                            latencyHtml += `<div>Initial: ${(firstAnyMs/1000).toFixed(2)}s</div>`;
                        }

                        // Breakdown if available
                        if (firstContentLatencyMs !== null && firstContentLatencyMs !== firstAnyMs) {
                            latencyHtml += `<div>Content: ${(firstContentLatencyMs/1000).toFixed(2)}s</div>`;
                        }

                        latencyHtml += `<div class="font-bold text-gray-300">Total: ${(totalLatencyMs/1000).toFixed(2)}s</div>`;
                        if (currentJobId) {
                            latencyHtml += `<div class="text-[9px] opacity-50">Job ID: ${escapeHtml(currentJobId)}</div>`;
                        }
                        latencyHtml += `<div class="text-[10px] mt-1">${escapeHtml(get('model-select').value)}</div>`;
                        latencyHtml += `</div>`;
                        adiv.insertAdjacentHTML('beforeend', latencyHtml);
                    } else {
                        adiv.insertAdjacentHTML('beforeend', `<div class="text-[10px] text-gray-500/50 mt-2 text-right font-mono">${escapeHtml(get('model-select').value)}</div>`);
                    }
                }

                // Reset editing state
                editingMessageId = null;
                setEditUi(false);

                if (adiv) {
                    const thoughts = adiv.querySelectorAll('.thought-content');
                    thoughts.forEach(t => t.classList.add('collapsed'));
                }
                // Full reload to establish new tree structure. Errors are persisted
                // server-side as assistant messages (```chat_error), so reload keeps them visible.
                await loadMessages(currentThreadId, { preserveDraft: true, silent: true });
                if (!hadError && codingModeEnabled) {
                    codingTargetSelection = null;
                    syncCodingModeUi(true, { persist: false });
                }

                // Only auto-scroll if user was already at bottom or auto-scroll is active
                if(userAutoScroll) scrollToBottom();

                if (document.querySelectorAll('.message-group').length <= 2 || !currentThreadTitle || currentThreadTitle === 'New Chat' || currentThreadTitle === 'No Title') {
                     apiFetch("/api/generate_title", {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({thread_id: currentThreadId, model_id: get('model-select').value})}).then(r=>r.json()).then(d=>{ if(d.title) { document.title = d.title + " - AI Chat"; setCurrentChatHeaderTitle(d.title); loadThreads(); } });
                } else loadThreads(false);

            } catch(e){
                let syncedAfterAbort = false;
                const manuallyStopped = e.name === 'AbortError' && isManualStopAbortForThread(streamStartedThreadId);
                if (e.name === 'AbortError' && !manuallyStopped) {
                    syncedAfterAbort = await syncThreadAfterAbortedStream(streamStartedThreadId, { retries: 2, retryDelayMs: 180, notifyOnFailure: true });
                }
                sendClientDebugLog('error', `Prompt send error: ${e.message}`);
                if (!requestAccepted) {
                    if (optimisticUserMessageEl) optimisticUserMessageEl.remove();
                    const pendingBubbleGroup = adiv && adiv.closest('.fade-in');
                    if (pendingBubbleGroup) pendingBubbleGroup.remove();
                    delete messageStore[optimisticUserId];
                    delete messageMeta[optimisticUserId];
                }
                if (e.serverCode === 'request_already_accepted' && e.acceptedJobId && e.acceptedThreadId) {
                    requestAccepted = true;
                    resumeAcceptedSubmission = {
                        job_id: e.acceptedJobId,
                        thread_id: String(e.acceptedThreadId),
                        model: p.model
                    };
                    get('prompt-input').value = '';
                    get('prompt-input').style.height = 'auto';
                    resetUploadState();
                    clearQuote();
                } else if (requestAccepted && !manuallyStopped) {
                    reconnectAfterStreamDisconnect = {
                        job_id: normalizeJobIdForUi(currentJobId),
                        thread_id: currentThreadId !== null && currentThreadId !== undefined ? String(currentThreadId) : null,
                        model: p.model
                    };
                    window.ConnectionMonitor.setUnavailable('offline');
                    showToast('回答への接続が切れました。バックグラウンド処理へ自動再接続します。', 'warning', false);
                } else if (e.serverCode === 'turnstile_required') {
                    // APIレベルでもTurnstile未検証はブロックされるため、再検証を試みる。
                    const retryToken = await getTurnstileToken();
                    if (retryToken) {
                        await verifyTurnstileOnServer(retryToken, true);
                        showToast('安全性の確認を完了しました。もう一度送信してください。', 'warning', false);
                    } else {
                        showToast('安全性の確認を完了できませんでした。しばらく待ってから再送信してください。', 'error', true);
                    }
                } else if (e.serverCode === 'api_key_missing') {
                    const missingKeyModel = e.serverModel || p.model;
                    const action = await showApiKeyRequiredModalAsync(missingKeyModel);
                    if (action === 'set') {
                        retryAfterApiKeySetup = true;
                    } else if (action === 'switch') {
                        showModal('model-modal');
                    } else {
                        showToast(e.message || `${getModelNameById(missingKeyModel)} のAPIキーが設定されていません`, "error", true);
                    }
                } else if(e.name!=='AbortError') {
                    const msg = "Connection Error: " + e.message;
                    showToast(msg, "error", true);
                }
                // Restore old message if error occurred during edit
                if (editingId && !syncedAfterAbort) restoreHiddenBranch();
            } finally {
                if (streamOpStarted && window.ConnectionMonitor) window.ConnectionMonitor.operationEnded();
                if (finishStreamProgress) finishStreamProgress();
                setSendBtnToSendMode();
                updateFilePreview();
                if (activeStreamingBubbleId === aid) activeStreamingBubbleId = null;
                abortController=null; currentJobId=null; editingMessageId=null; setEditUi(false);
            }
            if (resumeAcceptedSubmission) {
                const previousThreadId = currentThreadId !== null && currentThreadId !== undefined
                    ? String(currentThreadId)
                    : null;
                currentThreadId = resumeAcceptedSubmission.thread_id;
                if (previousThreadId !== currentThreadId || location.pathname !== '/c/' + currentThreadId) {
                    history.pushState({}, '', '/c/' + currentThreadId);
                }
                return reconnectPendingStreamUntilAvailable(resumeAcceptedSubmission, currentThreadId);
            }
            if (reconnectAfterStreamDisconnect && reconnectAfterStreamDisconnect.thread_id) {
                return reconnectPendingStreamUntilAvailable(
                    reconnectAfterStreamDisconnect,
                    reconnectAfterStreamDisconnect.thread_id
                );
            }
            if (retryAfterApiKeySetup) return sendMessage();
        }

        async function resumePendingStream(pending) {
            if (abortController) return;
            if (!pending || !pending.job_id || !currentThreadId) return;
            if (isPendingJobSuppressed(pending.job_id)) return;
            const jobId = pending.job_id;
            const bubbleId = `pending-${jobId}`;
            const pendingModelRaw = (pending && pending.model) ? String(pending.model) : '';
            if (!get(bubbleId)) {
                renderPendingMessage(get('chat-container'), true, true, bubbleId, pendingModelRaw);
            }
            const adiv = get(bubbleId);
            if (!adiv) return;
            activeStreamingBubbleId = bubbleId;
            adiv.classList.add('ai-pending-bubble');
            // Ensure skeleton matches pending model even if bubble already existed
            if (!adiv.querySelector('.content-area.skeleton-pending')) {
                const ca = adiv.querySelector('.content-area');
                if (ca) ca.outerHTML = buildPendingSkeletonHtml(pendingModelRaw, '回答を生成中...');
                else adiv.insertAdjacentHTML('afterbegin', buildPendingSkeletonHtml(pendingModelRaw, '回答を生成中...'));
            }
            currentJobId = jobId;
            setSendBtnToStopMode();
            resumeChatAutoScroll();
            if (canvasModeEnabled) {
                resetCanvasPreviewPanel();
            }
            abortController = new AbortController();
            const resumeStartedThreadId = currentThreadId;
            const pendingModel = pendingModelRaw.toLowerCase();
            const pendingReasoningModel =
                pendingModel.includes('gemini') ||
                pendingModel.includes('o1') ||
                pendingModel.includes('o3') ||
                pendingModel.includes('gpt-5') ||
                (pendingModel.includes('reasoning') && !pendingModel.includes('non-reasoning'));
            let thoughtPlaceholderEl = null;
            const ensureThoughtPlaceholder = (text) => {
                if (!pendingReasoningModel || !adiv) return null;
                if (!thoughtPlaceholderEl || !adiv.contains(thoughtPlaceholderEl)) {
                    thoughtPlaceholderEl = adiv.querySelector('.thought-content');
                }
                if (!thoughtPlaceholderEl) {
                    const tHtml = `<div class="thought-container"><div class="thought-header thinking-shimmer" onclick="toggleThinking(this)"><i class="fas fa-brain text-purple-400"></i> Thinking Process</div><div class="thought-content collapsed" data-placeholder="1"></div></div>`;
                    adiv.insertAdjacentHTML('afterbegin', tHtml);
                    thoughtPlaceholderEl = adiv.querySelector('.thought-content');
                }
                if (thoughtPlaceholderEl) {
                    thoughtPlaceholderEl.setAttribute('data-placeholder', '1');
                    thoughtPlaceholderEl.textContent = text || '推論プロセスを準備中...';
                }
                return thoughtPlaceholderEl;
            };
            if (pendingReasoningModel) {
                ensureThoughtPlaceholder('推論プロセスを準備中...');
            }
            let buf="", acc="", tht="", first=true, thEl=null, cEl=null, searchBox=null, hadError=false;
            const pyBoxes = {};
            let lastRenderTime = 0;
            let streamEndedByError = false;
            const finishResumeProgress = window.ProgressSpinner
                ? window.ProgressSpinner.startFlow('chatResume')
                : null;
            let reconnectAfterResumeDisconnect = false;
            let resumeOpStarted = false;
            try {
                const r = await apiFetch("/chat_stream_resume", manualSpinnerRequestOptions({
                    method: 'POST',
                    headers: {'Content-Type':'application/json'},
                    body: JSON.stringify({ thread_id: currentThreadId, job_id: jobId, turnstile_token: botTurnstileTokenForRequest() }),
                    signal: abortController.signal
                }));
                if (!r.ok) {
                    throw new Error(`Resume failed (${r.status})`);
                }
                if (window.ConnectionMonitor) {
                    resumeOpStarted = true;
                    window.ConnectionMonitor.operationStarted();
                }
                if (finishResumeProgress) finishResumeProgress.setPhase('waiting');
                const reader = r.body.getReader();
                const dec = new TextDecoder();
                while(!streamEndedByError) {
                    const {done, value} = await reader.read();
                    if (done) break;
                    if (window.ConnectionMonitor) window.ConnectionMonitor.reportActivity();
                    if (finishResumeProgress) finishResumeProgress.setPhase('receiving');
                    buf += dec.decode(value, {stream:true});
                    let ls = buf.split("\n");
                    buf = ls.pop();
                    let contentChanged = false;
                    let thoughtChanged = false;
                    for (let l of ls) {
                        if (!l.trim()) continue;
                        try {
                            const j = JSON.parse(l);
                            if (j.type === 'job_id') { currentJobId = j.content || jobId; continue; }
                            if (j.type === 'search_status') {
                                if (j.content === 'searching' && !searchBox) {
                                     adiv.insertAdjacentHTML('afterbegin', `<div class="search-box visible animate-pulse mb-2"><i class="fas fa-globe"></i> Searching web...</div>`);
                                     searchBox = adiv.querySelector('.search-box');
                                } else if (j.content === 'done' && searchBox) {
                                     searchBox.classList.remove('animate-pulse');
                                     searchBox.innerHTML = '<i class="fas fa-check-circle text-green-400"></i> Search complete';
                                     setTimeout(() => { if(searchBox) searchBox.remove(); searchBox=null; }, 2000);
                                }
                                continue;
                            }
                            if (j.type === 'mcp') {
                                handleMcpStreamEvent(adiv, j.content || {});
                                continue;
                            }
                            if (j.type === 'mcp_decision_request') {
                                openMcpDecisionModal(j.content || {});
                                continue;
                            }
                            if (j.type === 'status') {
                                const statusText = (j.content === null || j.content === undefined) ? '' : String(j.content);
                                if (first && adiv) {
                                    const headline = statusText || 'モデル処理中...';
                                    if (!updatePendingSkeletonStatus(adiv, headline, '応答開始までの進捗を表示しています')) {
                                        const ca = adiv.querySelector('.content-area');
                                        if (ca) {
                                            ca.outerHTML = buildPendingSkeletonHtml(pendingModelRaw, headline);
                                            updatePendingSkeletonStatus(adiv, headline, '応答開始までの進捗を表示しています');
                                        }
                                    }
                                }
                                if (pendingReasoningModel) {
                                    ensureThoughtPlaceholder(statusText || '推論プロセスを準備中...');
                                }
                                continue;
                            }
                            if (first) {
                                beginPendingToStreamTransition(adiv);
                                const ca = adiv.querySelector('.content-area');
                                if (ca) ca.innerHTML = '';
                                first = false;
                            }
                            if (j.type === 'coding_diff') {
                                appendCodingLiveDiff(adiv, j.content || {});
                            } else if (j.type === 'thought') {
                                if (!thEl) {
                                    thEl = adiv.querySelector('.thought-content');
                                }
                                tht += j.content;
                                if (!thEl) {
                                    const tHtml = `<div class="thought-container"><div class="thought-header" onclick="toggleThinking(this)"><i class="fas fa-brain text-purple-400"></i> Thinking Process</div><div class="thought-content"></div></div>`;
                                    if (searchBox) searchBox.insertAdjacentHTML('afterend', tHtml);
                                    else adiv.insertAdjacentHTML('afterbegin', tHtml);
                                    thEl = adiv.querySelector('.thought-content');
                                }
                                if (thEl && thEl.getAttribute('data-placeholder') === '1') {
                                    thEl.textContent = '';
                                    thEl.removeAttribute('data-placeholder');
                                    if (thEl) {
                                        const thoughtHeader = thEl.parentElement.querySelector('.thought-header');
                                        if (thoughtHeader) thoughtHeader.classList.remove('thinking-shimmer');
                                    }
                                    tht = j.content;
                                }
                                thEl.classList.remove('collapsed');
                                thoughtChanged = true;
                            } else if (j.type === 'image_analysis') {
                                const iaText = (j.content === null || j.content === undefined) ? '' : String(j.content);
                                if (!adiv) continue;
                                let iaEl = adiv.querySelector('.image-analysis-box');
                                if (!iaEl) {
                                    const iaHtml = `<div class="image-analysis-box mb-2 p-2 bg-blue-900/20 border border-blue-500/30 rounded"><div class="text-[10px] text-blue-300 font-medium mb-1"><i class="fas fa-image mr-1"></i>Image Analysis</div><div class="image-analysis-text text-[11px] text-gray-300"></div></div>`;
                                    if(searchBox) searchBox.insertAdjacentHTML('afterend', iaHtml);
                                    else adiv.insertAdjacentHTML('afterbegin', iaHtml);
                                    iaEl = adiv.querySelector('.image-analysis-box');
                                }
                                const iaTxt = iaEl.querySelector('.image-analysis-text');
                                if (iaTxt) iaTxt.textContent = iaText;
                            } else if (j.type === 'python') {
                                const py = j.content || {};
                                const pyId = py.id || `py_${Date.now()}`;
                                if (!pyBoxes[pyId]) {
                                    const boxHtml = `<div class="code-wrapper python-box collapsed" data-py-id="${pyId}" data-collapsed="true" data-code-key="${pyId}"><div class="code-header"><span class="code-lang"><i class="fas fa-terminal"></i> Python Execution</span><div class="code-actions"><button class="code-toggle" aria-expanded="false" title="展開" aria-label="展開"><i class="fas fa-chevron-down"></i></button><button class="copy-btn" data-copy="code" data-code="" title="コードをコピー" aria-label="コードをコピー"><i class="fas fa-copy"></i></button><button class="copy-btn" data-copy="output" data-code="" title="出力をコピー" aria-label="出力をコピー"><i class="fas fa-align-left"></i></button></div></div><div class="code-body"><div class="python-section"><div class="python-label">Code</div><pre><code class="hljs language-python python-code"></code></pre></div><div class="python-section"><div class="python-label">Output</div><pre><code class="hljs language-plaintext python-output"></code></pre></div></div></div>`;
                                    if (searchBox) searchBox.insertAdjacentHTML('afterend', boxHtml);
                                    else adiv.insertAdjacentHTML('afterbegin', boxHtml);
                                    pyBoxes[pyId] = adiv.querySelector(`[data-py-id="${pyId}"]`);
                                }
                                const box = pyBoxes[pyId];
                                if (box) {
                                    if (py.code !== undefined) {
                                        const codeText = py.code == null ? '' : String(py.code);
                                        const codeEl = box.querySelector('.python-code');
                                        if (codeEl) {
                                            codeEl.textContent = codeText;
                                            codeEl.removeAttribute('data-highlighted');
                                            queueHighlight(box, codeText);
                                        }
                                        const codeBtn = box.querySelector('.copy-btn[data-copy="code"]');
                                        if (codeBtn) codeBtn.setAttribute('data-code', encodeURIComponent(codeText).replace(/'/g, "%27"));
                                    }
                                    if (py.output !== undefined) {
                                        const outText = py.output == null ? '' : String(py.output);
                                        const outEl = box.querySelector('.python-output');
                                        if (outEl) outEl.textContent = outText;
                                        const outBtn = box.querySelector('.copy-btn[data-copy="output"]');
                                        if (outBtn) outBtn.setAttribute('data-code', encodeURIComponent(outText).replace(/'/g, "%27"));
                                    }
                                }
                            } else if (j.type === 'content') {
                                const contentDelta = (j.content === null || j.content === undefined) ? '' : String(j.content);
                                acc += contentDelta;
                                if (/[`~]/.test(contentDelta)) activateDeferredCodingModeFromStream(acc);
                                if (!cEl) {
                                    cEl = adiv.querySelector('.content-area') || document.createElement('div');
                                    cEl.className = 'prose prose-invert text-sm break-words';
                                    if (!adiv.contains(cEl)) adiv.appendChild(cEl);
                                }
                                contentChanged = true;
                            } else if (j.type === 'error') {
                                hadError = true;
                                streamEndedByError = true;
                                adiv.insertAdjacentHTML('beforeend', buildChatErrorBubbleHtml(j.content));
                                showToast(j.content || "Unknown error", "error", true);
                                break;
                            }
                        } catch (e) {}
                    }
                    if (thoughtChanged && thEl) {
                        thEl.textContent = tht;
                        if (userAutoScroll) thEl.scrollTop = thEl.scrollHeight;
                    }
                    if (contentChanged && cEl) {
                        const now = Date.now();
                        if (now - lastRenderTime > 100) {
                            const collapseState = snapshotCodeCollapse(cEl);
                            renderAiMarkdownInto(cEl, acc, { incrementalMath: true });
                            applyCodeCollapse(cEl, collapseState, true);
                            lastRenderTime = now;
                        }
                    }
                    scrollToBottom();
                }
                if (finishResumeProgress) finishResumeProgress();
                // Final render to catch any remaining content
                if (cEl) {
                    const collapseState = snapshotCodeCollapse(cEl);
                    renderAiMarkdownInto(cEl, acc, { incrementalMath: true });
                    applyCodeCollapse(cEl, collapseState, true);
                }

                vibrateHelper([100, 50, 100]);

                if (adiv) {
                    queueHighlight(adiv, acc);
                }

                // Errors are persisted server-side; always reload so history stays consistent.
                if (adiv) {
                    const thoughts = adiv.querySelectorAll('.thought-content');
                    thoughts.forEach(t => t.classList.add('collapsed'));
                }
                await loadMessages(currentThreadId, { preserveDraft: true, silent: true });
                loadThreads(false);
            } catch (e) {
                const manuallyStopped = e.name === 'AbortError' && isManualStopAbortForThread(resumeStartedThreadId);
                if (e.name === 'AbortError' && !manuallyStopped) {
                    await syncThreadAfterAbortedStream(resumeStartedThreadId, { retries: 2, retryDelayMs: 180, notifyOnFailure: true });
                }
                if (!manuallyStopped) {
                    reconnectAfterResumeDisconnect = true;
                    window.ConnectionMonitor.setUnavailable('offline');
                    showToast('回答への再接続が切れました。自動的に再試行します。', 'warning', false);
                }
            } finally {
                if (resumeOpStarted && window.ConnectionMonitor) window.ConnectionMonitor.operationEnded();
                if (finishResumeProgress) finishResumeProgress();
                setSendBtnToSendMode();
                updateFilePreview();
                if (activeStreamingBubbleId === bubbleId) activeStreamingBubbleId = null;
                abortController = null;
                currentJobId = null;
                currentThreadPending = null;
            }
            if (reconnectAfterResumeDisconnect) {
                return reconnectPendingStreamUntilAvailable(
                    { job_id: jobId, model: pendingModelRaw },
                    resumeStartedThreadId
                );
            }
        }

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
        const initGemPullToRefresh = () => initPullToRefresh('gem-list', () => loadGems());

        const initPullToRefreshAll = () => {
            initThreadPullToRefresh();
            initGemPullToRefresh();
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
