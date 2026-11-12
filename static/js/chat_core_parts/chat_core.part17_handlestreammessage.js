                    }
                }

                function handleStreamMessage(msg) {
                    if (msg && msg.snapshot) {
                        // Snapshot lets the recording reconstruct; live playback continues with deltas.
                        const st = msg.status;
                        if (st === 'error') {
                            setStatus('エラー', 'error');
                            stopElapsedTimer();
                            return;
                        }
                        if (st === 'closed' || st === 'stopped') {
                            setStatus('終了', 'closed');
                            stopElapsedTimer();
                            return;
                        }
                        setStatus(st === 'paused' ? '一時停止中' : '接続中...', st === 'paused' ? 'paused' : 'connecting');
                        return;
                    }
                    if (msg && msg.audio) {
                        setStatus('再生中...', 'streaming');
                        startElapsedTimer();
                        playChunk(msg.audio);
                        return;
                    }
                    if (msg && msg.error) {
                        setStatus('エラー: ' + msg.error, 'error');
                        stopElapsedTimer();
                        return;
                    }
                    if (msg && msg.final) {
                        setStatus('終了', 'closed');
                        stopElapsedTimer();
                        updateTransportButtons();
                        return;
                    }
                }

                function playChunk(b64) {
                    if (!b64) return;
                    if (!audioCtx) {
                        const AC = window.AudioContext || window.webkitAudioContext;
                        if (!AC) return;
                        audioCtx = new AC({ sampleRate: SAMPLE_RATE });
                        playbackStarted = false;
                        playbackNextTime = 0;
                    }
                    let bytes;
                    try {
                        const binary = atob(b64);
                        bytes = new Uint8Array(binary.length);
                        for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
                    } catch (e) { return; }
                    const pcm = new Int16Array(bytes.buffer);
                    const frames = Math.floor(pcm.length / CHANNELS);
                    if (frames < 1) return;
                    const buffer = audioCtx.createBuffer(CHANNELS, frames, SAMPLE_RATE);
                    for (let ch = 0; ch < CHANNELS; ch++) {
                        const chData = buffer.getChannelData(ch);
                        for (let i = 0; i < frames; i++) {
                            chData[i] = pcm[i * CHANNELS + ch] / 32768.0;
                        }
                    }
                    if (audioCtx.state === 'suspended') audioCtx.resume();
                    const source = audioCtx.createBufferSource();
                    source.buffer = buffer;
                    source.connect(audioCtx.destination);
                    if (!playbackStarted) {
                        playbackNextTime = audioCtx.currentTime + 0.08;
                        playbackStarted = true;
                    }
                    const startTime = Math.max(audioCtx.currentTime, playbackNextTime);
                    source.start(startTime);
                    playbackNextTime = startTime + buffer.duration;
                }

                async function apiCommand(type, payload) {
                    const res = await fetch('/api/skillini/music/command', noSpinner({
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(Object.assign({ session_id: sessionId, type }, payload || {}))
                    }));
                    const data = await res.json().catch(() => ({}));
                    if (!res.ok) throw new Error(data.error || 'コマンド送信に失敗しました');
                    return data;
                }

                async function startSession() {
                    if (busy) return;
                    const prompts = collectPrompts();
                    if (!prompts.length) {
                        showToast('プロンプトを入力してください', 'warning', true);
                        return;
                    }
                    busy = true;
                    updateTransportButtons();
                    setStatus('接続中...', 'connecting');
                    try {
                        const res = await fetch('/api/skillini/music/start', noSpinner({
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ weighted_prompts: prompts, config: collectConfig() })
                        }));
                        const data = await res.json().catch(() => ({}));
                        if (!res.ok) throw new Error(data.error || 'セッション開始に失敗しました');
                        sessionId = data.session_id;
                        lastConfig = collectConfig();
                        setStatus('接続中...', 'connecting');
                        openStream();
                    } catch (err) {
                        setStatus('エラー: ' + err.message, 'error');
                        showToast('Lyria RealTime: ' + err.message, 'error', true);
                    } finally {
                        busy = false;
                        updateTransportButtons();
                    }
                }

                async function control(action) {
                    if (!sessionId) return;
                    busy = true;
                    updateTransportButtons();
                    try {
                        await apiCommand('control', { action });
                        if (action === 'PLAY') setStatus('再生中...', 'streaming');
                        else if (action === 'PAUSE') setStatus('一時停止中', 'paused');
                        else if (action === 'STOP') setStatus('停止中', 'stopped');
                        else if (action === 'RESET_CONTEXT') setStatus('コンテキストをリセット...', 'connecting');
                    } catch (err) {
                        showToast('Lyria RealTime: ' + err.message, 'error', true);
                        setStatus('エラー: ' + err.message, 'error');
                    } finally {
                        busy = false;
                        updateTransportButtons();
                    }
                }

                async function applyPrompts() {
                    if (!sessionId) return;
                    const prompts = collectPrompts();
                    if (!prompts.length) {
                        showToast('プロンプトを入力してください', 'warning', true);
                        return;
                    }
                    busy = true;
                    try {
                        await apiCommand('prompts', { weighted_prompts: prompts });
                        setStatus('プロンプトを適用しました', state === 'paused' ? 'paused' : 'streaming');
                        showToast('プロンプトを適用しました', 'success');
                    } catch (err) {
                        showToast('Lyria RealTime: ' + err.message, 'error', true);
                    } finally {
                        busy = false;
                        updateTransportButtons();
                    }
                }

                async function applyConfig() {
                    if (!sessionId) return;
                    const cfg = collectConfig();
                    const prev = lastConfig || {};
                    // BPM and scale need a context reset to take effect; other
                    // parameters can morph smoothly without one.
                    const bpmChanged = (cfg.bpm !== undefined && cfg.bpm !== prev.bpm);
                    const scaleChanged = (cfg.scale !== undefined && cfg.scale !== prev.scale);
                    const resetContext = bpmChanged || scaleChanged;
                    busy = true;
                    try {
                        await apiCommand('config', { config: cfg, reset_context: resetContext });
                        lastConfig = cfg;
                        setStatus(resetContext ? '設定を適用しました（コンテキストをリセット）' : '設定を適用しました', state === 'paused' ? 'paused' : 'streaming');
                        showToast(resetContext ? '設定を適用しました（コンテキストをリセット）' : '設定を適用しました', 'success');
                    } catch (err) {
                        showToast('Lyria RealTime: ' + err.message, 'error', true);
                    } finally {
                        busy = false;
                        updateTransportButtons();
                    }
                }

                async function saveSession() {
                    if (!sessionId) return;
                    busy = true;
                    setStatus('保存中...', 'connecting');
                    updateTransportButtons();
                    try {
                        const res = await fetch('/api/skillini/music/save', noSpinner({
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ session_id: sessionId, thread_id: currentThreadId || null })
                        }));
                        const data = await res.json().catch(() => ({}));
                        if (!res.ok) throw new Error(data.error || '保存に失敗しました');
                        setStatus('保存しました', 'closed');
                        stopElapsedTimer();
                        showToast('チャットに保存しました', 'success');
                        if (data.thread_id) {
                            currentThreadId = String(data.thread_id);
                            history.pushState({}, '', '/c/' + data.thread_id);
                            get('welcome-screen').classList.add('hidden');
                        }
                        await loadMessages(data.thread_id || currentThreadId);
                        closeAndCleanup(true);
                    } catch (err) {
                        setStatus('エラー: ' + err.message, 'error');
                        showToast('Lyria RealTime: ' + err.message, 'error', true);
                    } finally {
                        busy = false;
                        updateTransportButtons();
                    }
                }

                async function cancelSession() {
                    closeStream();
                    if (sessionId) {
                        try {
                            await fetch('/api/skillini/music/cancel', noSpinner({
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({ session_id: sessionId })
                            }));
                        } catch (e) {}
                    }
                    sessionId = null;
                    stopElapsedTimer();
                    resetPlayback();
                    setStatus('準備完了', 'idle');
                }

                function resetControls() {
                    const container = $('lyria-prompt-rows');
                    if (container) container.innerHTML = '';
                    addPromptRow('', 1.0);
                    lastConfig = null;
                    streamStartTime = 0;
                    ['lyria-bpm', 'lyria-guidance', 'lyria-density', 'lyria-brightness', 'lyria-temperature'].forEach((id) => {
                        const el = $(id);
                        if (!el) return;
                        el.value = id === 'lyria-bpm' ? '120' : (id === 'lyria-guidance' ? '4' : (id === 'lyria-temperature' ? '1.1' : '0.5'));
                    });
                    const scale = $('lyria-scale');
                    if (scale) scale.value = '';
                    const mode = $('lyria-mode');
                    if (mode) mode.value = 'QUALITY';
                    ['lyria-mute-bass', 'lyria-mute-drums', 'lyria-only-bass-drums'].forEach((id) => {
                        const el = $(id);
                        if (el) el.checked = false;
                    });
                    bindRangeLabels();
                }

                function closeAndCleanup(keepThread) {
                    closeStream();
                    if (sessionId) {
                        fetch('/api/skillini/music/cancel', noSpinner({
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ session_id: sessionId })
                        })).catch(() => {});
                    }
                    sessionId = null;
                    streamOpen = false;
                    stopElapsedTimer();
                    resetPlayback();
                    hideModal('lyria-studio-modal');
                }

                function open(promptText) {
                    if (!isLyriaRealtimeModel()) {
                        showToast('Lyria RealTime モデルを選択してから開いてください', 'warning', true);
                        return;
                    }
                    const modal = $('lyria-studio-modal');
                    const alreadyOpen = modal && modal.classList.contains('modal-open');
                    if (alreadyOpen && sessionId) {
                        if (promptText && typeof promptText === 'string') {
                            const container = $('lyria-prompt-rows');
                            if (container) container.innerHTML = '';
                            addPromptRow(promptText, 1.0);
                        }
                        return;
                    }
                    if (sessionId) cancelSession();
                    resetControls();
                    if (promptText && typeof promptText === 'string') {
                        const container = $('lyria-prompt-rows');
                        if (container) container.innerHTML = '';
                        addPromptRow(promptText, 1.0);
                    }
                    sessionId = null;
                    streamOpen = false;
                    stopElapsedTimer();
                    resetPlayback();
                    setStatus('準備完了', 'idle');
                    showModal('lyria-studio-modal');
                }

                function init() {
                    const openBtn = $('lyria-open-studio-btn');
                    if (openBtn) openBtn.addEventListener('click', () => open(''));
                    const closeBtn = $('lyria-studio-close');
                    if (closeBtn) closeBtn.addEventListener('click', () => closeAndCleanup(false));
                    const playBtn = $('lyria-play-btn');
                    if (playBtn) {
                        playBtn.addEventListener('click', () => {
                            if (!sessionId) { startSession(); return; }
                            control('PLAY');
                        });
                    }
                    const pauseBtn = $('lyria-pause-btn');
                    if (pauseBtn) pauseBtn.addEventListener('click', () => control('PAUSE'));
                    const stopBtn = $('lyria-stop-btn');
                    if (stopBtn) stopBtn.addEventListener('click', () => control('STOP'));
                    const resetBtn = $('lyria-reset-btn');
                    if (resetBtn) resetBtn.addEventListener('click', () => control('RESET_CONTEXT'));
                    const addPromptBtn = $('lyria-add-prompt-btn');
                    if (addPromptBtn) addPromptBtn.addEventListener('click', () => addPromptRow('', 1.0));
                    const applyPromptsBtn = $('lyria-apply-prompts-btn');
                    if (applyPromptsBtn) applyPromptsBtn.addEventListener('click', applyPrompts);
                    const applyConfigBtn = $('lyria-apply-config-btn');
                    if (applyConfigBtn) applyConfigBtn.addEventListener('click', applyConfig);
                    const saveBtn = $('lyria-save-btn');
                    if (saveBtn) saveBtn.addEventListener('click', saveSession);
                    bindRangeLabels();
                    resetControls();
                    // Expose for sendMessage interception.
                    window.openLyriaStudio = open;
                }

                // Overlay click: same as the × button, but never while saving.
                function requestClose() {
                    if (busy) return;
                    closeAndCleanup(false);
                }

                return { init, open, requestClose };
            })();
            LyriaRealtimeStudio.init();
            window.closeLyriaStudio = () => LyriaRealtimeStudio.requestClose();

            // ===========================================================================
            // Voice Studio (WebSocket voice / STS models) — dedicated studio UI
            // ===========================================================================
            const VoiceStudio = (() => {
                let originalPanelParent = null;
                let originalPanelNext = null;
                let originalFileParent = null;
                const SETTINGS_OPEN_KEY = 'voiceDockSettingsOpen';
                const PLACEHOLDER = '会話の文字起こしがここに表示されます。';
                const $ = (id) => document.getElementById(id);

                function isStudioMode() {
                    return isStsModel() && voiceStudioUiEnabled !== false;
                }

                function updateTitle() {
                    const model = get('model-select') ? get('model-select').value : '';
                    const titleEl = $('voice-studio-title');
                    if (!titleEl) return;
                    if (model === 'gpt-transcribe' || model === 'gpt-live-transcribe' || model === 'grok-voice-transcribe-2.0-file') {
                        titleEl.textContent = '音声文字起こしスタジオ';
                    } else if (model === 'gemini-3.5-live-translate-preview') {
                        titleEl.textContent = 'リアルタイム音声翻訳スタジオ';
                    } else {
                        titleEl.textContent = '音声スタジオ';
                    }
                }

                function resetTranscript() {
                    const modalHost = $('voice-studio-transcript');
                    if (modalHost) modalHost.innerHTML = `<div class="voice-studio-placeholder text-[10px] text-gray-500">${PLACEHOLDER}</div>`;
                    const dockHost = $('sts-live-transcript');
                    if (dockHost) {
                        dockHost.innerHTML = '';
                        dockHost.classList.add('hidden');
                    }
                }

                function writeLine(host, role, inner) {
                    const lines = host.querySelectorAll('.voice-studio-line');
                    let target = null;
                    for (let i = lines.length - 1; i >= 0; i--) {
                        if (lines[i].dataset.role === role) { target = lines[i]; break; }
                    }
                    if (target) {
                        target.innerHTML = inner;
                    } else {
                        const placeholder = host.querySelector('.voice-studio-placeholder');
                        if (placeholder) placeholder.remove();
                        const line = document.createElement('div');
                        line.className = 'voice-studio-line';
                        line.dataset.role = role;
                        line.innerHTML = inner;
                        host.appendChild(line);
                    }
                    host.classList.remove('hidden');
                    host.scrollTop = host.scrollHeight;
                }

                function log(role, text) {
                    if (!text || !String(text).trim()) return;
                    if (!isStudioMode()) return;
                    const label = role === 'user' ? 'あなた' : 'AI';
                    const cls = role === 'user' ? 'text-cyan-300' : 'text-gray-100';
                    const inner = `<span class="${cls} font-bold">${escapeHtml(label)}:</span> <span class="text-gray-200">${escapeHtml(text)}</span>`;
                    [$('voice-studio-transcript'), $('sts-live-transcript')].filter(Boolean).forEach(host => writeLine(host, role, inner));
                }

                function setSettingsOpen(isOpen, persist = true) {
                    const panel = $('sts-panel');
                    const toggle = $('sts-settings-toggle');
                    if (panel) panel.classList.toggle('settings-open', !!isOpen);
                    if (toggle) toggle.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
                    if (persist) {
                        try { localStorage.setItem(SETTINGS_OPEN_KEY, isOpen ? '1' : '0'); } catch (e) {}
                    }
                }

                function readSettingsOpen() {
                    try { return localStorage.getItem(SETTINGS_OPEN_KEY) === '1'; } catch (e) { return false; }
                }

                // Called from updateStsUi whenever the model or the studio preference changes.
                let lastDockModel = null;
                function syncDock() {
                    const model = get('model-select') ? get('model-select').value : '';
                    if (model !== lastDockModel) {
                        lastDockModel = model;
                        resetTranscript();
                    }
                    setSettingsOpen(readSettingsOpen(), false);
                    updateTitle();
                }

                function movePanelIntoModal() {
                    const panel = $('sts-panel');
                    const host = $('voice-studio-panel-host');
                    if (panel && host && panel.parentNode !== host) {
                        originalPanelParent = panel.parentNode;
                        originalPanelNext = panel.nextSibling;
                        host.appendChild(panel);
                    }
                    const filePreview = $('file-preview');
                    const fileHost = $('voice-studio-file-host');
                    if (filePreview && fileHost && filePreview.parentNode !== fileHost) {
                        originalFileParent = filePreview.parentNode;
                        fileHost.appendChild(filePreview);
                        fileHost.classList.remove('hidden');
                    }
                }

                function movePanelBack() {
                    const panel = $('sts-panel');
                    if (panel && originalPanelParent && panel.parentNode !== originalPanelParent) {
                        if (originalPanelNext && originalPanelNext.parentNode === originalPanelParent) {
                            originalPanelParent.insertBefore(panel, originalPanelNext);
                        } else {
                            originalPanelParent.appendChild(panel);
                        }
                    }
                    const filePreview = $('file-preview');
                    if (filePreview && originalFileParent && filePreview.parentNode !== originalFileParent) {
                        originalFileParent.appendChild(filePreview);
                    }
                    const fileHost = $('voice-studio-file-host');
                    if (fileHost) fileHost.classList.add('hidden');
                    originalPanelParent = null;
                    originalPanelNext = null;
                    originalFileParent = null;
                }

                // Enlarged view. The conversation keeps running when switching between
                // the inline dock and this view, so opening/closing never cancels it.
                function open() {
                    if (!isStudioMode()) return;
                    movePanelIntoModal();
                    const panel = $('sts-panel');
                    if (panel) panel.classList.remove('hidden');
                    updateTitle();
                    window.VoiceStudioOpen = true;
                    showModal('voice-studio-modal');
                }

                function close() {
                    window.VoiceStudioOpen = false;
                    movePanelBack();
                    hideModal('voice-studio-modal');
                }

                function closeIfOpen() {
                    if (window.VoiceStudioOpen) close();
                }

                function init() {
                    window.VoiceStudioOpen = false;
                    const openBtn = $('voice-studio-open-btn');
                    if (openBtn) openBtn.addEventListener('click', () => open());
                    const closeBtn = $('voice-studio-close');
                    if (closeBtn) closeBtn.addEventListener('click', () => close());
                    const settingsToggle = $('sts-settings-toggle');
                    if (settingsToggle) settingsToggle.addEventListener('click', () => {
                        const panel = $('sts-panel');
                        setSettingsOpen(!(panel && panel.classList.contains('settings-open')));
                    });
                    window.VoiceStudio = { open, close, closeIfOpen, log, isStudioMode, syncDock };
                    syncDock();
                }

                return { init };
            })();
            VoiceStudio.init();

            let currentRtPlayer = null;

            function stopStsPlayback() {
                if (currentRtPlayer) {
                    currentRtPlayer.stop();
                    currentRtPlayer = null;
                }
                if (stsPlaybackAudio) {
                    try { stsPlaybackAudio.pause(); } catch (e) {}
                    try { stsPlaybackAudio.src = ''; } catch (e) {}
                    stsPlaybackAudio = null;
                }
            }
            async function playStsAudio(url) {
                stopStsPlayback();
                const audio = new Audio();
                audio.src = url;
                audio.preload = 'auto';
                audio.autoplay = true;
                audio.playsInline = true;
                stsPlaybackAudio = audio;
                await audio.play();
                return new Promise(resolve => {
                    audio.onended = () => resolve('ended');
                    audio.onerror = () => resolve('error');
                });
            }
            function cancelRecording() {
                if (rtVoiceSession.isActive()) {
                    rtVoiceSession._cancel();
                    return;
                }
                if (currentGeminiLive) {
                    currentGeminiLive.stop();
                    currentGeminiLive = null;
                    stopStsPlayback();
                    get('mic-btn').classList.remove('bg-red-600', 'animate-pulse');
                    get('mic-btn').classList.add('bg-gray-700');
                    setStsStatus('Canceled', false);
                    setTimeout(() => setStsStatus('Tap to speak', false), 800);
                    stopMicWaveform();
                    return;
                }
                if (mediaRecorder && mediaRecorder.state === "recording") {
                    stsCancelPending = true;
                    mediaRecorder.stop();
                }
            }
            function getMicCaptureConstraints() {
                const stsMode = isStsModel();
                if (stsMode) return { audio: true };
                const supported = (navigator.mediaDevices && navigator.mediaDevices.getSupportedConstraints)
                    ? navigator.mediaDevices.getSupportedConstraints()
                    : {};
                const audio = { channelCount: 1 };
                if (supported.echoCancellation) audio.echoCancellation = false;
                if (supported.noiseSuppression) audio.noiseSuppression = false;
                if (supported.autoGainControl) audio.autoGainControl = false;
                return { audio };
            }
            // Open the mic during the click. A later await (the session token)
            // drops mobile Chrome's user activation, and an AudioContext created
            // after that stays suspended, so Live never receives PCM.
            function beginGeminiLiveMic() {
                const AC = window.AudioContext || window.webkitAudioContext;
                if (!AC || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                    throw new Error('マイクを利用できません');
                }
                let ctx = null;
                try {
                    ctx = new AC({ sampleRate: 16000 });
                } catch (e) {
                    if (ctx) { try { ctx.close(); } catch (err) {} ctx = null; }
                    ctx = new AC();
                }
                const resumePromise = ctx.resume();
                const streamPromise = navigator.mediaDevices.getUserMedia(getMicCaptureConstraints());
                return { ctx, resumePromise, streamPromise };
            }
            function releaseGeminiLiveMic(prepared) {
                if (!prepared) return;
                if (prepared.streamPromise) {
                    prepared.streamPromise.then(stream => {
                        try { stream.getTracks().forEach(track => track.stop()); } catch (e) {}
                    }).catch(() => {});
                } else if (prepared.stream) {
                    try { prepared.stream.getTracks().forEach(track => track.stop()); } catch (e) {}
                }
                if (prepared.ctx) { try { prepared.ctx.close(); } catch (e) {} }
            }
            get('mic-btn').onclick = async () => {
                if (abortController) {
                    showToast("回答生成中です。完了までお待ちいただくか、停止してください。", "warning", true);
                    return;
                }
                if (uploadProgressState.active > 0) {
                    showToast("ファイルの送信・処理中です。しばらくお待ちください。", "warning", true);
                    return;
                }
                if (currentGeminiLive) {
                    setStsStatus('Processing...', true);
                    const client = currentGeminiLive;
                    currentGeminiLive = null;
                    client.stop();

                    get('mic-btn').classList.remove('bg-red-600', 'animate-pulse');
                    get('mic-btn').classList.add('bg-gray-700');

                    try {
                        const finalData = await client.getFinalData();

                        if (isGeminiLiveTranscribeModel()) {
                            // Live transcription: transcript is the assistant output;
                            // user message keeps the recorded audio + placeholder text.
                            finalData.user_text = '音声文字起こし';
                            finalData.assistant_text = (client.inputTranscript || '').trim();
                            finalData.assistant_thought = '';
                            if (!finalData.assistant_text) {
                                setStsStatus('No transcript', false);
                                setTimeout(() => setStsStatus('Tap to speak', false), 1000);
                                return;
                            }
                        }

                        const hasLiveText = (finalData.user_text || '').trim()
                            || (finalData.assistant_text || '').trim()
                            || (finalData.assistant_thought || '').trim();
                        const hasLiveAudio = !!(finalData.audio_base64 || finalData.user_audio_base64);
                        if (!hasLiveText && !hasLiveAudio) {
                            setStsStatus('Error', false);
                            showToast('Gemini Live の音声を取得できませんでした。もう一度お試しください。', 'error', true);
                            setTimeout(() => setStsStatus('Tap to speak', false), 1000);
                            return;
                        }

                        if (!currentThreadId) {
                            const r = await apiFetch(CHAT_CONFIG.urls.handleThreads, {
                                method:'POST',
                                headers:{'Content-Type':'application/json'},
                                body: JSON.stringify({ is_temporary: temporaryChatEnabled })
                            });
                            const d = await r.json();
                            currentThreadId = String(d.id);
                            history.pushState({}, '', '/c/' + d.id);
                            get('welcome-screen').classList.add('hidden');
                        }

                        finalData.thread_id = currentThreadId;
                        finalData.model = get('model-select').value;

                        await apiFetch('/api/skillini/save_sts', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify(finalData)
                        });

                        setStsStatus('Saved', false);
                        setTimeout(() => setStsStatus('Tap to speak', false), 1000);
                        await loadMessages(currentThreadId);
                    } catch (e) {
                        console.error("Failed to save Gemini Live session:", e);
                        setStsStatus('Error saving', false);
                    }
                    return;
                }
                if (rtVoiceSession.isActive()) {
                    get('mic-btn').classList.remove('bg-red-600', 'animate-pulse');
                    get('mic-btn').classList.add('bg-gray-700');
                    rtVoiceSession.stop();
                    return;
                }
                if (mediaRecorder && mediaRecorder.state === "recording") {
                    mediaRecorder.stop();
                    get('mic-btn').classList.remove('bg-red-600', 'animate-pulse');
                    get('mic-btn').classList.add('bg-gray-700');
                    if (!isStsModel()) setMicRecordingIndicator('録音を処理中…', 'processing');
                    if (isStsModel()) setStsStatus('Processing...', true);
                    return;
                }
                try {
                    // Pre-activate audio for STS to bypass browser autoplay restrictions
                    if (isStsModel()) {
                        try {
                            const dummyAudio = new Audio();
                            dummyAudio.src = 'data:audio/wav;base64,UklGRiQAAABXQVZFRm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA=';
                            dummyAudio.play().catch(() => {});
                        } catch (e) {}
                    }

                    if (isGeminiLiveModel()) {
                        setStsStatus('Connecting...', true);
                        let liveMic = null;
                        try {
                            liveMic = beginGeminiLiveMic();
                            const modelKeyForSession = get('model-select').value;
                            const sessionBody = {
                                model: modelKeyForSession
                            };
                            if (isGeminiLiveTranscribeModel()) {
                                sessionBody.transcription_mode = get('sts-transcribe-mode') ? get('sts-transcribe-mode').value : 'VERBATIM';
                                if (get('sts-custom-vocab')) {
                                    const cv = get('sts-custom-vocab').value.split(/[,、\n]/).map(s => s.trim()).filter(Boolean);
                                    if (cv.length) sessionBody.custom_vocabulary = cv.slice(0, 1000);
                                }
                            } else {
                                sessionBody.voice = get('sts-voice') ? get('sts-voice').value : 'Kore';
                                if (isGeminiLiveExtendedThinkingModel()) {
                                    sessionBody.thinking_level = get('sts-thinking-level') ? get('sts-thinking-level').value : 'medium';
                                    sessionBody.include_thoughts = get('sts-include-thoughts') ? get('sts-include-thoughts').checked : false;
                                }
                                if (isGeminiLiveTranslateModel() && get('sts-target-lang')) {
                                    sessionBody.target_lang = get('sts-target-lang').value;
                                }
                            }
                            const micStream = await liveMic.streamPromise;
                            try { await liveMic.resumePromise; } catch (e) {}
                            liveMic = {
                                stream: micStream,
                                ctx: liveMic.ctx,
                                source: liveMic.ctx.createMediaStreamSource(micStream)
                            };
                            const res = await apiFetch('/api/skillini/session', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify(sessionBody)
                            });
                            if (!res.ok) throw new Error("Failed to get session token");
                            const { token, url } = await res.json();

                            const modelKey = get('model-select').value;
                            const voice = get('sts-voice') ? get('sts-voice').value : 'Kore';
                            const thinking_level = get('sts-thinking-level') ? get('sts-thinking-level').value : 'minimal';
                            const include_thoughts = get('sts-include-thoughts') ? get('sts-include-thoughts').checked : false;

                            currentGeminiLive = new GeminiLiveClient();
                            if (stsOpt('sts-auto-play') && !isGeminiLiveTranscribeModel()) {
                                currentGeminiLive.rtPlayer = new RealTimeAudioPlayer();
                            }

                            if (isGeminiLiveTranscribeModel()) {
                                const tMode = get('sts-transcribe-mode') ? get('sts-transcribe-mode').value : 'VERBATIM';
                                const transcriptionConfig = { languageCodes: [] };
                                if (tMode === 'SMART' || tMode === 'VERBATIM') transcriptionConfig.mode = tMode;
                                if (get('sts-custom-vocab')) {
                                    const cv = get('sts-custom-vocab').value.split(/[,、\n]/).map(s => s.trim()).filter(Boolean);
                                    if (cv.length) transcriptionConfig.customVocabulary = cv.slice(0, 1000);
                                }
                                await currentGeminiLive.start(token, url, modelKey, {
                                    transcriptionConfig
                                }, liveMic);
                            } else if (isGeminiLiveTranslateModel()) {
                                const targetLang = get('sts-target-lang') ? get('sts-target-lang').value : 'ja';
                                await currentGeminiLive.start(token, url, modelKey, {
                                    translationConfig: { targetLanguageCode: targetLang, echoTargetLanguage: true }
                                }, liveMic);
                            } else {
                                const liveConfig = {
                                    speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: voice } } }
                                };
                                if (isGeminiLiveExtendedThinkingModel()) {
                                    liveConfig.thinkingConfig = { thinkingLevel: thinking_level, includeThoughts: include_thoughts };
                                }
                                await currentGeminiLive.start(token, url, modelKey, liveConfig, liveMic);
                            }
                            liveMic = null;
                            mediaRecorder = currentGeminiLive.backupRecorder; // For silence monitor

                            // Bind onstop to trigger the save logic (clicking mic-btn again)
                            mediaRecorder.onstop = () => {
                                if (currentGeminiLive) get('mic-btn').click();
                            };

                            let firstAudio = true;
                            let liveMsgId = 'live-sts-' + Date.now();
                            currentGeminiLive.onMessage = (data) => {
                                if (data.serverContent) {
                                    if (isGeminiLiveTranscribeModel()) {
                                        // Live transcription: show interim + finalized user transcript.
                                        const interim = currentGeminiLive.interimInputTranscript;
                                        const finalText = currentGeminiLive.inputTranscript;
                                        const displayText = finalText + (interim && !finalText.endsWith(interim) ? (finalText ? '\n' : '') + interim : '');
                                        const container = get('chat-messages');
                                        let msgEl = document.getElementById(liveMsgId);
                                        if (!msgEl) {
                                            msgEl = document.createElement('div');
                                            msgEl.id = liveMsgId;
                                            msgEl.className = 'flex flex-col gap-2 mb-4 assistant-message bg-slate-800/40 p-3 rounded-lg border border-slate-700/50';
                                            msgEl.innerHTML = `
                                                <div class="text-[10px] text-teal-400 font-bold uppercase tracking-wider flex items-center gap-2">
                                                    <i class="fas fa-microphone"></i> Gemini 3.5 Transcribe Live
                                                </div>
                                                <div class="message-content text-sm text-slate-100 leading-relaxed"></div>
                                            `;
                                            container.appendChild(msgEl);
                                            container.scrollTop = container.scrollHeight;
                                        }
                                        const contentEl = msgEl.querySelector('.message-content');
                                        contentEl.innerText = displayText || '聴き取り中...';
                                        container.scrollTop = container.scrollHeight;
                                        if (window.VoiceStudio && finalText) window.VoiceStudio.log('user', finalText);
                                        return;
                                    }
                                    if (data.serverContent.modelTurn) {
                                        if (firstAudio) {
                                            setStsStatus('Gemini is speaking...', false);
                                            firstAudio = false;
                                        }
                                        // Update UI with transcript so far
                                        const container = get('chat-messages');
                                        let msgEl = document.getElementById(liveMsgId);
                                        if (!msgEl) {
                                            msgEl = document.createElement('div');
                                            msgEl.id = liveMsgId;
                                            msgEl.className = 'flex flex-col gap-2 mb-4 assistant-message bg-slate-800/40 p-3 rounded-lg border border-slate-700/50';
                                            msgEl.innerHTML = `
                                                <div class="text-[10px] text-cyan-400 font-bold uppercase tracking-wider flex items-center gap-2">
                                                    <i class="fas fa-robot"></i> Gemini Live (Streaming)
                                                </div>
                                                <div class="thought-container hidden italic text-slate-400 text-xs border-l-2 border-slate-600 pl-2 my-1"></div>
                                                <div class="message-content text-sm text-slate-100 leading-relaxed"></div>
                                            `;
                                            container.appendChild(msgEl);
                                            container.scrollTop = container.scrollHeight;
                                        }

                                        const thoughtEl = msgEl.querySelector('.thought-container');
                                        const contentEl = msgEl.querySelector('.message-content');

                                        if (currentGeminiLive.assistantThought) {
                                            thoughtEl.classList.remove('hidden');
                                            thoughtEl.innerText = currentGeminiLive.assistantThought;
                                        }
                                        contentEl.innerText = currentGeminiLive.assistantText;
                                        container.scrollTop = container.scrollHeight;
                                        if (window.VoiceStudio) {
                                            if (currentGeminiLive.inputTranscript) window.VoiceStudio.log('user', currentGeminiLive.inputTranscript);
                                            if (currentGeminiLive.assistantText) window.VoiceStudio.log('assistant', currentGeminiLive.assistantText);
                                        }
                                    }
                                }
                            };

                            setStsStatus('Listening...', true);
                            get('mic-btn').classList.remove('bg-gray-700');
                            get('mic-btn').classList.add('bg-red-600', 'animate-pulse');
                            startMicWaveform(currentGeminiLive.stream);
                            startSilenceMonitor(currentGeminiLive.stream);

                            // Provider-side end (setup rejected, session limit, network):
                            // report it and close out like a manual stop so the
                            // conversation so far is saved.
                            const liveClient = currentGeminiLive;
                            liveClient.onError = (err) => {
                                const msg = err && err.message ? err.message : (typeof err === 'string' ? err : '');
                                if (msg) showToast('Gemini Live: ' + msg, 'error', true);
                            };
                            liveClient.onClose = (ev) => {
                                if (currentGeminiLive !== liveClient) return;
                                if (ev && ev.code && ev.code !== 1000) {
                                    const reason = ev.reason ? ': ' + ev.reason : ' (code ' + ev.code + ')';
                                    showToast('Gemini Live の接続が終了しました' + reason, 'error', true);
                                }
                                get('mic-btn').click();
                            };
                            if (liveClient.closedEvent) liveClient.onClose(liveClient.closedEvent);
                            return;
                        } catch (e) {
                            releaseGeminiLiveMic(liveMic);
                            if (currentGeminiLive) {
                                try { currentGeminiLive.stop(); } catch (err) {}
                                currentGeminiLive = null;
                            }
                            showToast("Gemini Live connection failed: " + e.message, "error", true);
                            setStsStatus('Error', false);
                            return;
                        }
                    }

                    if (isRealtimeSessionModel()) {
                        await rtVoiceSession.start();
                        return;
                    }

                    if (!isStsModel()) {
                        resetMicWaveformBars();
                        setMicRecordingIndicator('録音準備中…', 'processing');
                    }
                    const stream = await navigator.mediaDevices.getUserMedia(getMicCaptureConstraints());
                    mediaRecorder = new MediaRecorder(stream);
                    audioChunks = [];
                    stsCancelPending = false;
                    const recordingStartedInStsMode = isStsModel();
                    mediaRecorder.ondataavailable = (e) => audioChunks.push(e.data);
                    mediaRecorder.onstop = async () => {
                        if (stsCancelPending) {
                            audioChunks = [];
                            get('file-preview').classList.add('hidden');
                            stream.getTracks().forEach(track => track.stop());
                            stopSilenceMonitor();
                            stopMicWaveform();
                            if (!recordingStartedInStsMode) {
                                setMicRecordingIndicator('録音をキャンセルしました', 'idle');
                                micIndicatorHideTimer = setTimeout(() => setMicRecordingIndicator('', 'hidden'), 900);
                            }
                            if (isStsModel()) setStsStatus('Canceled', false);
                            setTimeout(() => { if (isStsModel()) setStsStatus('Tap to speak', false); }, 800);
                            return;
                        }
                        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                        const file = new File([audioBlob], "recording.webm", { type: 'audio/webm' });
                        const fd = new FormData();
                        fd.append('file', file);
                        get('file-preview').classList.remove('hidden');
                        const stsMode = recordingStartedInStsMode;
                        get('file-name').innerText = stsMode ? "Processing voice..." : "Transcribing...";
                        try {
                            if (stsMode) {
                                if (!currentThreadId) {
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
                                    get('welcome-screen').classList.add('hidden');
                                }
                                if (currentThreadId && activeSkill) {
                                    threadSkillMap[currentThreadId] = activeSkill;
                                    pendingSkillForNewThread = null;
                                }
                                fd.append('model', get('model-select').value);
                                fd.append('thread_id', currentThreadId);
                                if (get('sts-voice')) fd.append('sts_voice', get('sts-voice').value || '');
                                if (get('sts-reasoning-effort')) fd.append('sts_reasoning_effort', get('sts-reasoning-effort').value || '');
                                if (get('sts-speed')) fd.append('sts_speed', get('sts-speed').value || '');
                                if (get('sts-rate-in')) fd.append('sts_rate_in', get('sts-rate-in').value || '');
                                if (get('sts-rate-out')) fd.append('sts_rate_out', get('sts-rate-out').value || '');
                                if (get('sts-thinking-level')) fd.append('sts_thinking_level', get('sts-thinking-level').value || '');
                                if (get('sts-include-thoughts')) fd.append('sts_include_thoughts', get('sts-include-thoughts').checked ? 'true' : '');
                                setStsStatus('Sending audio...', true);
                                const stsRes = await apiFetch("/sts", { method: 'POST', body: fd });
                                if (!stsRes.ok) {
                                    const errData = await stsRes.json().catch(() => ({}));
                                    throw new Error(errData.error || "Speech-to-speech failed");
                                }

                                const reader = stsRes.body.getReader();
                                const decoder = new TextDecoder();
                                let buffer = '';
                                let stsData = null;
                                let rtPlayer = null;
                                if (stsOpt('sts-auto-play')) {
                                    rtPlayer = new RealTimeAudioPlayer();
                                    currentRtPlayer = rtPlayer;
                                }

                                setStsStatus(isTranscriptionModel() ? 'Transcribing...' : 'Processing audio...', true);

                                let firstChunk = true;
                                let studioInput = '';
                                let studioAssistant = '';
                                while (true) {
                                    const { done, value } = await reader.read();
                                    if (done) break;

                                    buffer += decoder.decode(value, { stream: true });
                                    const lines = buffer.split('\n');
                                    buffer = lines.pop();

                                    for (const line of lines) {
                                        if (!line.trim()) continue;
                                        const chunk = JSON.parse(line);
                                        if (chunk.error) throw new Error(chunk.error);

                                        if (chunk.audio_delta && rtPlayer) {
                                            if (firstChunk) {
                                                setStsStatus('Playing response...', false);
                                                firstChunk = false;
                                            }
                                            await rtPlayer.addChunk(chunk.audio_delta);
                                        }
                                        if (chunk.input_delta) {
                                            studioInput += chunk.input_delta;
                                            if (window.VoiceStudio) window.VoiceStudio.log('user', studioInput);
                                        }
                                        if (chunk.transcript_delta) {
                                            studioAssistant += chunk.transcript_delta;
                                            if (window.VoiceStudio) window.VoiceStudio.log('assistant', studioAssistant);
                                        }
                                        // Streaming models end with `final`; the one-shot reply (Realtime Whisper) carries `audio_url`.
                                        if (chunk.final || chunk.audio_url) {
                                            stsData = chunk;
                                        }
                                    }
                                }
                                if (window.VoiceStudio && !studioInput.trim()) {
                                    window.VoiceStudio.log('user', '（音声メッセージ）');
                                }

                                if (stsData && (stsData.audio_url || stsData.transcription_only)) {
                                    if (stsOpt('sts-auto-restart') && isStsModel()) {
                                        // Wait a bit for rtPlayer to finish if it's still playing
                                        setTimeout(() => {
                                            setStsStatus('Listening...', true);
                                            get('mic-btn').click();
                                        }, 500);
                                    } else {
                                        setStsStatus('Tap to speak', false);
                                    }
                                    await loadMessages(currentThreadId);
                                } else if (!stsData && !rtPlayer) {
                                     // Fallback for sync responses if any
                                }
                            } else {
                                const micModeEl = get('set-mic-transcribe-mode');
                                const llmMicMode = !!(micModeEl && micModeEl.value === 'llm');
                                if (llmMicMode && !supportsAudioInputModel()) {
                                    showToast("現在のモデルはLLM音声文字起こし（音声入力）に対応していません", "error", true);
                                    return;
                                }
                                fd.append('llm_model', get('model-select') ? (get('model-select').value || '') : '');
                                const transRes = await apiFetch(CHAT_CONFIG.urls.transcribe, {
                                    method: 'POST',
                                    body: fd
                                });
                                const transData = await transRes.json();
                                if (transData.transcript) {
                                    const input = get('prompt-input');
                                    input.value += (input.value ? " " : "") + transData.transcript;
                                    input.style.height = 'auto';
                                    input.style.height = input.scrollHeight + 'px';
                                } else {
                                    showToast(transData.error || "Transcription failed", "error", true);
                                }
                            }
                        } catch (err) {
                            showToast("Audio processing error: " + err.message, "error", true);
                        } finally {
                            get('file-preview').classList.add('hidden');
                            stream.getTracks().forEach(track => track.stop());
                            stopSilenceMonitor();
                            stopMicWaveform();
                            if (!stsMode) setMicRecordingIndicator('', 'hidden');
                            if (stsMode) setStsStatus('Tap to speak', false);
                        }
                    };
                    mediaRecorder.start();
                    get('mic-btn').classList.remove('bg-gray-700');
                    get('mic-btn').classList.add('bg-red-600', 'animate-pulse');
                    if (!isStsModel()) {
                        setMicRecordingIndicator('録音中…', 'recording');
                        startMicWaveform(stream);
                    }
                    startSilenceMonitor(stream);
                    if (isStsModel()) setStsStatus('Recording... Tap to stop', true);
                } catch (err) {
                    stopMicWaveform();
                    if (!isStsModel()) setMicRecordingIndicator('', 'hidden');
                    alert("Microphone access denied or not available.");
                }
            };
