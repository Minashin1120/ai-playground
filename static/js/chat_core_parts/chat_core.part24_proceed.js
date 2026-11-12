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
            const ideogramObjectRemoval = isIdeogramObjectRemoval();
            if (isGptImageModel() && currentMaskImage && imageUrlsToSend.length === 0) {
                showToast("Mask は画像入力が必要です", "error", true);
                return;
            }
            if (ideogramObjectRemoval && !currentMaskImage) {
                showToast("Object Remover にはマスク画像が必要です", "error", true);
                return;
            }
            const followSend = userAutoScroll && isChatNearBottom();
            if (!followSend) pauseChatAutoScroll();
            const editingId = editingMessageId;
            const settingsBeforeSubmission = promptSettingsBeforeEdit;
            promptSettingsBeforeEdit = null;
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
            const maskImageToSend = (isGptImageModel() || ideogramObjectRemoval) ? currentMaskImage : null;
            const optimisticUserMessageEl = renderMessage(optimisticUserId, 'user', t, JSON.stringify(imageUrlsToSend), null, null, null, false, currentQuote, null, null, null, null, null, null, null, followSend, capturedParentId, activeSkill ? activeSkill.name : null, null, maskImageToSend);

            let disableAutoSearch = false;
            const xLinkPattern = /(https?:\/\/)?(x\.com|twitter\.com)\//i;
            const hasXLink = !removeBackground && (xLinkPattern.test(t || '') || xLinkPattern.test(currentQuote || ''));
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
                enable_skills: get('enable-skills') ? get('enable-skills').checked : true,
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
                tts_style: isTtsModel() && get('tts-style') && get('tts-style-wrap') && !get('tts-style-wrap').classList.contains('hidden') ? get('tts-style').value : null,
                tts_speed: isTtsModel() && get('tts-speed') ? get('tts-speed').value : null,
                image_size: isGptImageModel() && get('gpt-image-size') ? get('gpt-image-size').value : null,
                image_quality: isGptImageModel() && get('gpt-image-quality') ? get('gpt-image-quality').value : null,
                image_complexity: isGptImageModel() && get('gpt-image-complexity') ? get('gpt-image-complexity').value : null,
                image_format: isGptImageModel() && get('gpt-image-format') ? get('gpt-image-format').value : null,
                image_compression: isGptImageModel() && get('gpt-image-compression') ? get('gpt-image-compression').value : null,
                image_frame_count: isGptImageModel() && get('gpt-image-frame-count') ? get('gpt-image-frame-count').value : null,
                image_frame_duration_ms: isGptImageModel() && get('gpt-image-frame-duration') ? get('gpt-image-frame-duration').value : null,
                image_mask: maskImageToSend,
                gemini_image_aspect: isGeminiImageModel() && get('gemini-image-aspect') ? get('gemini-image-aspect').value : null,
                gemini_image_size: isGeminiImageModel() && get('gemini-image-size') ? get('gemini-image-size').value : null,
                grok_image_aspect: isGrokImageModel() && get('grok-image-aspect') ? get('grok-image-aspect').value : null,
                grok_image_resolution: isGrokImageModel() && get('grok-image-resolution') ? get('grok-image-resolution').value : null,
                grok_image_quality: isGrokImageModel() && get('grok-image-quality') ? get('grok-image-quality').value : null,
                grok_image_format: isGrokImageModel() && get('grok-image-format') ? get('grok-image-format').value : null,
                grok_image_count: isGrokImageModel() && get('grok-image-count') ? get('grok-image-count').value : null,
                ideogram_operation: isIdeogramModel() && get('ideogram-image-operation') ? get('ideogram-image-operation').value : null,
                ideogram_aspect: isIdeogramModel() && get('ideogram-image-aspect') ? get('ideogram-image-aspect').value : null,
                ideogram_resolution: isIdeogramModel() && get('ideogram-image-resolution') ? get('ideogram-image-resolution').value : null,
                ideogram_quality: isIdeogramModel() && get('ideogram-image-quality') ? get('ideogram-image-quality').value : null,
                ideogram_speed: isIdeogramModel() && get('ideogram-image-speed') ? get('ideogram-image-speed').value : null,
                ideogram_magic_prompt: isIdeogramModel() && get('ideogram-image-magic') ? get('ideogram-image-magic').value : null,
                ideogram_style_type: isIdeogramModel() && get('ideogram-image-style') ? get('ideogram-image-style').value : null,
                ideogram_negative_prompt: isIdeogramModel() && get('ideogram-image-negative') ? get('ideogram-image-negative').value : null,
                ideogram_count: isIdeogramModel() && get('ideogram-image-count') ? get('ideogram-image-count').value : null,
                ideogram_seed: isIdeogramModel() && get('ideogram-image-seed') ? get('ideogram-image-seed').value : null,
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
                quote_message_id: currentQuote ? currentQuoteMessageId : null,
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
                })) : [],
                batch_mode: batchModeRequested,
                canvas_mode: !!canvasModeEnabled
            };
            if (botTurnstileToken) p.turnstile_token = botTurnstileToken;
            const threadCustomInstructionEl = get('thread-custom-instruction');
            if (threadCustomInstructionEl) {
                p.thread_custom_instruction = threadCustomInstructionEl.value || '';
            }
            if (activeSkill) { p.system_prompt = activeSkill.instruction; p.enable_system_prompt = true; p.skill_uuid = activeSkill.uuid; } else { p.skill_uuid = null; }
            const sentPrompt = takePromptForSend(options.animate !== false);
            const stopSendMotion = animatePromptToMessage(optimisticUserMessageEl, sentPrompt, followSend && options.animate !== false);
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
            get('chat-container').insertAdjacentHTML('beforeend', `<div class="flex justify-start mb-4 ai-pending-row fade-in"><div id="${aid}" class="message-bubble ai-pending-bubble bg-gray-700 text-white p-4 rounded-2xl rounded-tl-none shadow-md relative">${initialHtml}</div></div>`);
            if (followSend) resumeChatAutoScroll();
            const adiv = get(aid);
            const stopPendingElapsed = !batchModeRequested ? startPendingElapsed(adiv, pendingStartedAt) : null;
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
                if (p.thread_id && activeSkill) {
                    threadSkillMap[p.thread_id] = activeSkill;
                    pendingSkillForNewThread = null;
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
                sentPrompt.accept();
                if (codingModeEnabled) {
                    syncCodingModeUi(true, { persist: false });
                }
                const markApiAccepted = () => {
                    if (!adiv) return;
                    const ca = adiv.querySelector('.content-area');
                    if (!ca) return;
                    if (ca.getAttribute('data-api-accepted') === '1') return;
                    ca.setAttribute('data-api-accepted', '1');
                    if (!updatePendingSkeletonStatus(adiv, '接続しました。応答を待っています...', '応答開始までの状況を表示しています')) {
                        ca.outerHTML = buildPendingSkeletonHtml(p.model, '接続しました。応答を待っています...');
                        const newCa = adiv.querySelector('.content-area');
                        if (newCa) newCa.setAttribute('data-api-accepted', '1');
                        updatePendingSkeletonStatus(adiv, '接続しました。応答を待っています...', '応答開始までの状況を表示しています');
                    }
                };
                markApiAccepted();
                window.ActivityLog?.log('send.stream.start');
                let loggedFirstChunk = false;
                const reader = r.body.getReader();
                const dec = new TextDecoder();
                let buf="", acc="", tht="", first=true, thEl=null, cEl=null, searchBox=null, hadError=false;
                const pyBoxes = {};
                let lastRenderTime = 0;
                let streamEndedByError = false;

                while(!streamEndedByError) {
                    const {done, value} = await reader.read();
                    if (!done && !loggedFirstChunk) { loggedFirstChunk = true; window.ActivityLog?.log('send.stream.first'); }
                    if (done) window.ActivityLog?.log('send.stream.done');
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
                                    if (activeSkill) {
                                        threadSkillMap[streamThreadId] = activeSkill;
                                        pendingSkillForNewThread = null;
                                    }
                                    ensureTemporaryChatHeartbeat(true);
                                }
                                continue;
                            }
                            if (j.type === 'job_id') {
                                markApiAccepted();
                                currentJobId = j.content;
                                if (batchModeRequested) {
                                    showToast('Batch登録', 'info');
                                }
                                continue;
                            }
                            if (j.type === 'search_status') {
                                searchBox = updateSearchBox(adiv, searchBox, j.content);
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

                    const totalLatencyMs = nowPerfMs() - sendStartPerfMs;

                    if (enableLatencyMetrics) {
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
                    }

                    let firstAnyMs = null;
                    if (firstStatusLatencyMs !== null) firstAnyMs = firstStatusLatencyMs;
                    if (firstThoughtLatencyMs !== null && (firstAnyMs === null || firstThoughtLatencyMs < firstAnyMs)) firstAnyMs = firstThoughtLatencyMs;
                    if (firstContentLatencyMs !== null && (firstAnyMs === null || firstContentLatencyMs < firstAnyMs)) firstAnyMs = firstContentLatencyMs;

                    const liveProcessingTime = {
                        total_ms: totalLatencyMs,
                        first_response_ms: firstAnyMs,
                        phases: firstAnyMs !== null
                            ? [{ key: 'generation', ms: Math.max(0, totalLatencyMs - firstAnyMs) }]
                            : [],
                        source: 'client'
                    };
                    messageMeta[aid] = Object.assign(messageMeta[aid] || {}, {
                        role: 'assistant',
                        model: p.model,
                        processing_time: liveProcessingTime
                    });
                    const existingProcessingRow = adiv.querySelector('.processing-time-row');
                    if (existingProcessingRow) existingProcessingRow.remove();
                    adiv.insertAdjacentHTML('beforeend', buildProcessingTimeHtml(aid, liveProcessingTime));
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
                await loadMessages(currentThreadId, { preserveDraft: true, silent: true, forceLatestLeaf: !!editingId });
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
                window.ActivityLog?.log(e.name === 'AbortError' ? 'send.cancelled' : 'send.error');
                stopSendMotion();
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
                        model: p.model,
                        pendingStartedAt: batchModeRequested ? null : pendingStartedAt
                    };
                    sentPrompt.accept();
                } else if (requestAccepted && !manuallyStopped) {
                    reconnectAfterStreamDisconnect = {
                        job_id: normalizeJobIdForUi(currentJobId),
                        thread_id: currentThreadId !== null && currentThreadId !== undefined ? String(currentThreadId) : null,
                        model: p.model,
                        pendingStartedAt: batchModeRequested ? null : pendingStartedAt
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
                if (!requestAccepted) {
                    const restoreEditing = editingId && sentPrompt.canRetry() && !editingMessageId;
                    sentPrompt.restore();
                    if (restoreEditing) {
                        editingMessageId = editingId;
                        currentParentId = capturedParentId;
                        promptSettingsBeforeEdit = settingsBeforeSubmission;
                        setEditUi(true);
                    }
                }
                // Restore old message if error occurred during edit
                if (editingId && !syncedAfterAbort) restoreHiddenBranch();
            } finally {
                if (stopPendingElapsed) stopPendingElapsed();
                if (streamOpStarted && window.ConnectionMonitor) window.ConnectionMonitor.operationEnded();
                if (finishStreamProgress) finishStreamProgress();
                setSendBtnToSendMode();
                updateFilePreview();
                if (activeStreamingBubbleId === aid) activeStreamingBubbleId = null;
                abortController=null; currentJobId=null; setEditUi(!!editingMessageId);
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
            if (retryAfterApiKeySetup && sentPrompt.canRetry()) return sendMessageLogged({ animate: false });
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
            const stopPendingElapsed = startPendingElapsed(adiv, pending.pendingStartedAt);
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
                window.ActivityLog?.log('send.stream.start');
                let loggedFirstChunk = false;
                const reader = r.body.getReader();
                const dec = new TextDecoder();
                while(!streamEndedByError) {
                    const {done, value} = await reader.read();
                    if (!done && !loggedFirstChunk) { loggedFirstChunk = true; window.ActivityLog?.log('send.stream.first'); }
                    if (done) window.ActivityLog?.log('send.stream.done');
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
                                searchBox = updateSearchBox(adiv, searchBox, j.content);
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
                window.ActivityLog?.log(e.name === 'AbortError' ? 'send.cancelled' : 'send.stream.error');
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
                if (stopPendingElapsed) stopPendingElapsed();
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
                    { job_id: jobId, model: pendingModelRaw, pendingStartedAt: pending.pendingStartedAt },
                    resumeStartedThreadId
                );
            }
        }
