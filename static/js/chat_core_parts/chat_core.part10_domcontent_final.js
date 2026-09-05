
            const rtVoiceSession = new RealtimeVoiceSession();

            // ===========================================================================
            // Lyria RealTime Studio (lyria-realtime-exp) — dedicated realtime music UI
            // ===========================================================================
            const LyriaRealtimeStudio = (() => {
                const SAMPLE_RATE = 48000;
                const CHANNELS = 2;
                const STATUS_CLASS = {
                    idle: 'bg-gray-600',
                    connecting: 'bg-amber-500 animate-pulse',
                    streaming: 'bg-emerald-600 animate-pulse',
                    paused: 'bg-amber-500',
                    stopped: 'bg-gray-600',
                    error: 'bg-red-600',
                    closed: 'bg-gray-600'
                };
                let sessionId = null;
                let streamController = null;
                let streamOpen = false;
                let audioCtx = null;
                let playbackStarted = false;
                let playbackNextTime = 0;
                let streamStartTime = 0;
                let elapsedTimer = null;
                let state = 'idle';
                let busy = false;
                let lastConfig = null;

                const $ = (id) => document.getElementById(id);
                const noSpinner = (options) => {
                    const opts = Object.assign({}, options || {});
                    if (window.ProgressSpinner && typeof window.ProgressSpinner.manualRequestOptions === 'function') {
                        return window.ProgressSpinner.manualRequestOptions(opts);
                    }
                    opts.progressSpinner = false;
                    return opts;
                };

                function setStatus(text, statusKey) {
                    state = statusKey;
                    const txt = $('lyria-status-text');
                    const dot = $('lyria-status-dot');
                    if (txt) txt.textContent = text;
                    if (dot) dot.className = 'w-2 h-2 rounded-full inline-block ' + (STATUS_CLASS[statusKey] || STATUS_CLASS.idle);
                    updateTransportButtons();
                    updateSaveButton();
                }

                function formatElapsed() {
                    const secs = streamStartTime ? Math.floor((Date.now() - streamStartTime) / 1000) : 0;
                    const m = String(Math.floor(secs / 60)).padStart(2, '0');
                    const s = String(secs % 60).padStart(2, '0');
                    return `${m}:${s}`;
                }

                function startElapsedTimer() {
                    if (!streamStartTime) streamStartTime = Date.now();
                    const el = $('lyria-elapsed');
                    if (el) el.textContent = formatElapsed();
                    if (!elapsedTimer) {
                        elapsedTimer = window.setInterval(() => {
                            const target = $('lyria-elapsed');
                            if (target) target.textContent = formatElapsed();
                        }, 1000);
                    }
                }

                function stopElapsedTimer() {
                    if (elapsedTimer) {
                        window.clearInterval(elapsedTimer);
                        elapsedTimer = null;
                    }
                }

                function updateTransportButtons() {
                    const playBtn = $('lyria-play-btn');
                    const pauseBtn = $('lyria-pause-btn');
                    const stopBtn = $('lyria-stop-btn');
                    const resetBtn = $('lyria-reset-btn');
                    const hasSession = !!sessionId;
                    const playing = state === 'streaming' || state === 'connecting';
                    if (playBtn) {
                        playBtn.disabled = busy || !hasSession;
                        const icon = playBtn.querySelector('i');
                        if (icon) icon.className = 'fas fa-play';
                    }
                    if (pauseBtn) pauseBtn.disabled = busy || !playing;
                    if (stopBtn) stopBtn.disabled = busy || !hasSession || !playing;
                    if (resetBtn) resetBtn.disabled = busy || !hasSession || !playing;
                }

                function updateSaveButton() {
                    const saveBtn = $('lyria-save-btn');
                    if (!saveBtn) return;
                    const visible = !!sessionId && state !== 'idle' && state !== 'connecting' && state !== 'error';
                    saveBtn.classList.toggle('hidden', !visible);
                }

                function addPromptRow(text, weight) {
                    const container = $('lyria-prompt-rows');
                    if (!container) return;
                    const row = document.createElement('div');
                    row.className = 'flex items-center gap-2';
                    row.innerHTML = `
                        <input type="text" value="${escapeHtml(text || '')}" placeholder="例: minimal techno / warm acoustic guitar" class="flex-1 bg-gray-700 border border-gray-600 rounded px-2 py-1.5 text-[11px] text-white outline-none min-w-0" maxlength="4000">
                        <label class="flex items-center gap-1 text-[10px] text-gray-400 shrink-0">
                            <span>w</span>
                            <input type="range" min="0.1" max="5" step="0.1" value="${typeof weight === 'number' ? weight : 1.0}" class="accent-purple-400 w-16">
                            <span class="lyria-weight-label font-mono text-purple-300 w-8 text-right">${(typeof weight === 'number' ? weight : 1.0).toFixed(1)}</span>
                        </label>
                        <button type="button" data-progress-no-spinner="true" class="lyria-prompt-remove shrink-0 w-6 h-6 rounded-full bg-gray-800 hover:bg-red-600 text-gray-400 hover:text-white text-[10px] flex items-center justify-center transition btn-hover"><i class="fas fa-times"></i></button>
                    `;
                    const range = row.querySelector('input[type="range"]');
                    const label = row.querySelector('.lyria-weight-label');
                    if (range && label) {
                        range.addEventListener('input', () => { label.textContent = parseFloat(range.value).toFixed(1); });
                    }
                    const removeBtn = row.querySelector('.lyria-prompt-remove');
                    if (removeBtn) {
                        removeBtn.addEventListener('click', () => {
                            if (container.querySelectorAll('.lyria-prompt-row-wrap').length <= 1) return;
                            row.remove();
                        });
                    }
                    row.classList.add('lyria-prompt-row-wrap');
                    container.appendChild(row);
                }

                function collectPrompts() {
                    const rows = document.querySelectorAll('#lyria-prompt-rows .lyria-prompt-row-wrap');
                    const prompts = [];
                    rows.forEach((row) => {
                        const input = row.querySelector('input[type="text"]');
                        const range = row.querySelector('input[type="range"]');
                        const text = (input ? input.value : '').trim();
                        if (!text) return;
                        prompts.push({ text, weight: parseFloat(range ? range.value : 1.0) || 1.0 });
                    });
                    return prompts;
                }

                function collectConfig() {
                    const cfg = {};
                    const num = (id) => {
                        const el = $(id);
                        return el && el.value !== '' ? parseFloat(el.value) : undefined;
                    };
                    const bpm = num('lyria-bpm');
                    if (bpm !== undefined) cfg.bpm = Math.round(bpm);
                    const guidance = num('lyria-guidance');
                    if (guidance !== undefined) cfg.guidance = guidance;
                    const density = num('lyria-density');
                    if (density !== undefined) cfg.density = density;
                    const brightness = num('lyria-brightness');
                    if (brightness !== undefined) cfg.brightness = brightness;
                    const temperature = num('lyria-temperature');
                    if (temperature !== undefined) cfg.temperature = temperature;
                    const scale = $('lyria-scale');
                    if (scale && scale.value) cfg.scale = scale.value;
                    const mode = $('lyria-mode');
                    if (mode && mode.value) cfg.music_generation_mode = mode.value;
                    const muteBass = $('lyria-mute-bass');
                    const muteDrums = $('lyria-mute-drums');
                    const onlyBassDrums = $('lyria-only-bass-drums');
                    if (muteBass) cfg.mute_bass = muteBass.checked;
                    if (muteDrums) cfg.mute_drums = muteDrums.checked;
                    if (onlyBassDrums) cfg.only_bass_and_drums = onlyBassDrums.checked;
                    return cfg;
                }

                function bindRangeLabels() {
                    const bindings = [
                        ['lyria-bpm', 'lyria-bpm-label'],
                        ['lyria-guidance', 'lyria-guidance-label'],
                        ['lyria-density', 'lyria-density-label'],
                        ['lyria-brightness', 'lyria-brightness-label'],
                        ['lyria-temperature', 'lyria-temperature-label']
                    ];
                    bindings.forEach(([rangeId, labelId]) => {
                        const range = $(rangeId);
                        const label = $(labelId);
                        if (!range || !label) return;
                        range.addEventListener('input', () => {
                            const v = parseFloat(range.value);
                            label.textContent = rangeId === 'lyria-bpm' ? String(Math.round(v)) : v.toFixed(1);
                        });
                    });
                }

                function resetPlayback() {
                    if (audioCtx) {
                        try { audioCtx.close(); } catch (e) {}
                        audioCtx = null;
                    }
                    playbackStarted = false;
                    playbackNextTime = 0;
                }

                function closeStream() {
                    streamOpen = false;
                    if (streamController && typeof streamController.abort === 'function') {
                        try { streamController.abort(); } catch (e) {}
                    }
                    streamController = null;
                }

                async function openStream() {
                    closeStream();
                    streamController = new AbortController();
                    streamOpen = true;
                    try {
                        const res = await fetch(`/api/gemini/music/stream?session_id=${encodeURIComponent(sessionId)}`, noSpinner({
                            method: 'GET',
                            signal: streamController.signal,
                            headers: { 'Accept': 'text/event-stream' },
                            cache: 'no-store'
                        }));
                        if (!res.ok) {
                            const errData = await res.json().catch(() => ({}));
                            throw new Error(errData.error || 'ストリーム接続に失敗しました');
                        }
                        const reader = res.body.getReader();
                        const decoder = new TextDecoder();
                        let buffer = '';
                        while (streamOpen) {
                            const { done, value } = await reader.read();
                            if (done) break;
                            buffer += decoder.decode(value, { stream: true });
                            const events = buffer.split('\n\n');
                            buffer = events.pop();
                            for (const evt of events) {
                                const line = evt.split('\n').find((l) => l.startsWith('data: '));
                                if (!line) continue;
                                const payload = line.slice(6);
                                try {
                                    const msg = JSON.parse(payload);
                                    handleStreamMessage(msg);
                                } catch (e) {}
                            }
                        }
                    } catch (err) {
                        if (err && err.name === 'AbortError') return;
                        if (streamOpen) {
                            setStatus('ストリーム切断。再接続します…', 'connecting');
                            window.setTimeout(() => {
                                if (streamOpen && sessionId) openStream();
                            }, 1200);
                        }
                    } finally {
                        streamOpen = false;
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
                    const res = await fetch('/api/gemini/music/command', noSpinner({
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
                        const res = await fetch('/api/gemini/music/start', noSpinner({
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
                        const res = await fetch('/api/gemini/music/save', noSpinner({
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
                            await fetch('/api/gemini/music/cancel', noSpinner({
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
                        fetch('/api/gemini/music/cancel', noSpinner({
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

                return { init, open };
            })();
            LyriaRealtimeStudio.init();

            // ===========================================================================
            // Voice Studio (WebSocket voice / STS models) — dedicated studio UI
            // ===========================================================================
            const VoiceStudio = (() => {
                let originalPanelParent = null;
                let originalFileParent = null;
                const $ = (id) => document.getElementById(id);

                function isStudioMode() {
                    return isStsModel() && voiceStudioUiEnabled !== false;
                }

                function updateTitle() {
                    const model = get('model-select') ? get('model-select').value : '';
                    const titleEl = $('voice-studio-title');
                    if (!titleEl) return;
                    if (model === 'gpt-transcribe' || model === 'gpt-live-transcribe') {
                        titleEl.textContent = '音声文字起こしスタジオ';
                    } else if (model === 'gemini-3.5-live-translate-preview') {
                        titleEl.textContent = 'リアルタイム音声翻訳スタジオ';
                    } else {
                        titleEl.textContent = '音声スタジオ';
                    }
                }

                function resetTranscript() {
                    const host = $('voice-studio-transcript');
                    if (!host) return;
                    host.innerHTML = '<div class="text-[10px] text-gray-500">会話の文字起こしがここに表示されます。</div>';
                }

                function log(role, text) {
                    if (!text || !String(text).trim()) return;
                    const host = $('voice-studio-transcript');
                    if (!host || !window.VoiceStudioOpen) return;
                    const label = role === 'user' ? 'あなた' : 'AI';
                    const cls = role === 'user' ? 'text-cyan-300' : 'text-gray-100';
                    const lines = host.querySelectorAll('.voice-studio-line');
                    let target = null;
                    for (let i = lines.length - 1; i >= 0; i--) {
                        if (lines[i].dataset.role === role) { target = lines[i]; break; }
                    }
                    const inner = `<span class="${cls} font-bold">${escapeHtml(label)}:</span> <span class="text-gray-200">${escapeHtml(text)}</span>`;
                    if (target) {
                        target.innerHTML = inner;
                    } else {
                        const placeholder = host.querySelector('.text-gray-500');
                        if (placeholder) placeholder.remove();
                        const line = document.createElement('div');
                        line.className = 'voice-studio-line';
                        line.dataset.role = role;
                        line.innerHTML = inner;
                        host.appendChild(line);
                    }
                    host.scrollTop = host.scrollHeight;
                }

                function movePanelIntoModal() {
                    const panel = $('sts-panel');
                    const host = $('voice-studio-panel-host');
                    if (panel && host && panel.parentNode !== host) {
                        originalPanelParent = panel.parentNode;
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
                        originalPanelParent.appendChild(panel);
                    }
                    const filePreview = $('file-preview');
                    if (filePreview && originalFileParent && filePreview.parentNode !== originalFileParent) {
                        originalFileParent.appendChild(filePreview);
                    }
                    const fileHost = $('voice-studio-file-host');
                    if (fileHost) fileHost.classList.add('hidden');
                    originalPanelParent = null;
                    originalFileParent = null;
                }

                function open() {
                    if (!isStudioMode()) {
                        showToast('音声系モデルを選択してから開いてください', 'warning', true);
                        return;
                    }
                    movePanelIntoModal();
                    const panel = $('sts-panel');
                    if (panel) panel.classList.remove('hidden');
                    updateTitle();
                    resetTranscript();
                    window.VoiceStudioOpen = true;
                    showModal('voice-studio-modal');
                }

                function close() {
                    if (window.VoiceStudioOpen && (currentGeminiLive || (mediaRecorder && mediaRecorder.state === 'recording') || rtVoiceSession.isActive())) {
                        cancelRecording();
                    }
                    window.VoiceStudioOpen = false;
                    movePanelBack();
                    hideModal('voice-studio-modal');
                    if (isStsModel() && voiceStudioUiEnabled !== false) {
                        const panel = $('sts-panel');
                        if (panel) panel.classList.add('hidden');
                    }
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
                    window.VoiceStudio = { open, close, closeIfOpen, log, isStudioMode };
                }

                return { init, open, close, closeIfOpen, log, isStudioMode };
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

                        await apiFetch('/api/gemini/save_sts', {
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
                        try {
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
                                sessionBody.thinking_level = get('sts-thinking-level') ? get('sts-thinking-level').value : 'minimal';
                                sessionBody.include_thoughts = get('sts-include-thoughts') ? get('sts-include-thoughts').checked : false;
                                if (isGeminiLiveTranslateModel() && get('sts-target-lang')) {
                                    sessionBody.target_lang = get('sts-target-lang').value;
                                }
                            }
                            const res = await apiFetch('/api/gemini/session', {
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
                                });
                            } else if (isGeminiLiveTranslateModel()) {
                                const targetLang = get('sts-target-lang') ? get('sts-target-lang').value : 'ja';
                                await currentGeminiLive.start(token, url, modelKey, {
                                    translationConfig: { targetLanguageCode: targetLang, echoTargetLanguage: true }
                                });
                            } else {
                                await currentGeminiLive.start(token, url, modelKey, {
                                    speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: voice } } },
                                    thinkingConfig: { thinkingLevel: thinking_level, includeThoughts: include_thoughts }
                                });
                            }
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
                            return;
                        } catch (e) {
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
                                if (currentThreadId && activeGem) {
                                    threadGemMap[currentThreadId] = activeGem;
                                    pendingGemForNewThread = null;
                                }
                                fd.append('model', get('model-select').value);
                                fd.append('thread_id', currentThreadId);
                                if (get('sts-voice')) fd.append('sts_voice', get('sts-voice').value || '');
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
                                        if (chunk.final) {
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

        const setLibBtnLabel = (btn, label) => {
            if (!btn) return;
            const span = btn.querySelector('span');
            if (span) span.textContent = label;
            else btn.textContent = label;
        };
        window.updateLibSelectionUi = function () {
            if (!lib.selected) lib.selected = new Set();
            const count = lib.selected.size;
            const delBtn = get('lib-del-btn');
            const downloadBtn = get('lib-download-btn');
            const attachBtn = get('lib-attach-btn');
            const renameBtn = get('lib-rename-btn');
            if (delBtn) {
                delBtn.disabled = count === 0;
                setLibBtnLabel(delBtn, count ? `削除 (${count})` : "削除");
            }
            if (downloadBtn) {
                downloadBtn.disabled = count === 0;
                setLibBtnLabel(downloadBtn, count ? `ダウンロード (${count})` : "ダウンロード");
            }
            if (attachBtn) {
                attachBtn.disabled = count === 0;
                setLibBtnLabel(attachBtn, count ? `添付 (${count})` : "添付");
            }
            if (renameBtn) {
                renameBtn.disabled = count !== 1;
                setLibBtnLabel(renameBtn, "名前変更");
            }
            if (lib.modal) {
                const isMobile = window.matchMedia('(max-width: 768px)').matches;
                lib.modal.classList.toggle('lib-selecting', isMobile && count > 0);
            }
        };
        function setLibAttachMode(flag) {
            lib.attachMode = !!flag;
        }
        const openLibModal = (attachMode = false) => {
            setLibAttachMode(attachMode);
            showModal('lib-modal');
            loadLibraryFiles();
            if (location.pathname !== '/library') {
                history.pushState({ modal: 'library' }, '', '/library');
            }
        };
        window.closeLibModal = (skipHistory = false) => {
            hideModal('lib-modal');
            if (!skipHistory && location.pathname === '/library') {
                history.back();
            }
        };
        get('lib-btn').onclick = () => openLibModal(false);
        get('lib-del-btn').onclick = deleteSelectedFiles;
        if (get('lib-download-btn')) get('lib-download-btn').onclick = () => downloadSelectedLibraryFiles();
        if (get('lib-attach-btn')) get('lib-attach-btn').onclick = () => attachSelectedLibraryFiles();
        if (get('lib-rename-btn')) get('lib-rename-btn').onclick = () => renameSelectedLibraryFile();
        if (get('upload-lib-btn')) get('upload-lib-btn').onclick = () => openLibModal(true);
        if (get('lib-search')) {
            get('lib-search').oninput = () => {
                lib.searchQuery = (get('lib-search').value || '').trim();
                renderLibraryGrid();
            };
        }
        if (get('lib-sort')) {
            const storedSort = localStorage.getItem(LIB_SORT_KEY) || 'newest';
            get('lib-sort').value = storedSort;
            get('lib-sort').onchange = () => {
                const v = get('lib-sort').value || 'newest';
                localStorage.setItem(LIB_SORT_KEY, v);
                renderLibraryGrid();
            };
        }
        if (get('lib-favorite-filter-btn')) {
            lib.favoritesOnly = localStorage.getItem(LIB_FAVORITES_ONLY_KEY) === 'true';
            get('lib-favorite-filter-btn').onclick = () => {
                lib.favoritesOnly = !lib.favoritesOnly;
                localStorage.setItem(LIB_FAVORITES_ONLY_KEY, String(lib.favoritesOnly));
                renderLibraryGrid();
            };
        }
            if (get('add-gem-fixed-prompt-row')) {
                get('add-gem-fixed-prompt-row').onclick = () => addGemFixedPromptRow();
            }

        const openGemModal = () => {
            editingGemUuid = null;
            get('gem-modal-title').innerHTML = `<i class="fas fa-gem text-blue-500 mr-2"></i>Create New Gem`;
            get('save-gem-btn').innerText = "Create Gem";
            showModal('gem-modal');
            get('gem-name').value=''; get('gem-desc').value=''; get('gem-inst').value=''; get('gem-default-model').value='';
            if (get('gem-fixed-prompts-container')) get('gem-fixed-prompts-container').innerHTML = '';
            if (location.pathname !== '/gem') {
                history.pushState({ modal: 'gem' }, '', '/gem');
            }
        };
        window.closeGemModal = (skipHistory = false) => {
            hideModal('gem-modal');
            if (!skipHistory && location.pathname === '/gem') {
                history.back();
            }
        };
        get('add-gem-btn').onclick = () => openGemModal();
            get('save-gem-btn').onclick = async () => {
                const name = get('gem-name').value;
                const desc = get('gem-desc').value;
                const inst = get('gem-inst').value;
                const fixed_prompts = collectGemFixedPrompts();

                if(name && inst) {
                    const method = editingGemUuid ? 'PUT' : 'POST';
                    const url = editingGemUuid ? `/api/gems/${editingGemUuid}` : CHAT_CONFIG.urls.handleGems;
                    await apiFetch(url, {
                        method: method,
                        headers:{'Content-Type':'application/json'},
                        body:JSON.stringify({name, description:desc, instruction:inst, fixed_prompts, default_model: get('gem-default-model').value || null})
                    });
                    window.closeGemModal();
                    loadGems();
                    if (editingGemUuid && activeGem && activeGem.uuid === editingGemUuid) {
                       activeGem.name = name;
                       activeGem.instruction = inst;
                       activeGem.fixed_prompts = fixed_prompts;
                       applyActiveGem(activeGem);
                    }
                } else alert("Name and Instruction are required.");
            };
            document.addEventListener('click', function(e) {
                if (e.target.closest('.edit-btn')) {
                    const btn = e.target.closest('.edit-btn');
                    const id = btn.getAttribute('data-id');
                    beginEditMessage(id);
                }
                if (e.target.closest('.code-toggle')) {
                    const btn = e.target.closest('.code-toggle');
                    const wrapper = btn.closest('.code-wrapper');
                    if (!wrapper) return;
                    const isCollapsed = wrapper.classList.toggle('collapsed');
                    wrapper.setAttribute('data-collapsed', isCollapsed ? 'true' : 'false');
                    btn.setAttribute('aria-expanded', isCollapsed ? 'false' : 'true');
                    btn.innerHTML = isCollapsed
                        ? '<i class="fas fa-chevron-down"></i>'
                        : '<i class="fas fa-chevron-up"></i>';
                    btn.title = isCollapsed ? '展開' : '折りたたむ';
                    btn.setAttribute('aria-label', isCollapsed ? '展開' : '折りたたむ');
                }
                if (e.target.closest('.download-btn')) {
                    const btn = e.target.closest('.download-btn');
                    const codeEnc = btn.getAttribute('data-code');
                    const lang = (btn.getAttribute('data-lang') || 'txt').toLowerCase();
                    if (codeEnc) {
                        try {
                            const code = decodeURIComponent(codeEnc);
                            const blob = new Blob([code], { type: 'text/plain' });
                            const url = URL.createObjectURL(blob);
                            const a = document.createElement('a');
                            a.href = url;
                            const map = {
                                python: 'py', javascript: 'js', typescript: 'ts', markdown: 'md',
                                html: 'html', css: 'css', json: 'json', xml: 'xml', sql: 'sql',
                                bash: 'sh', sh: 'sh', shell: 'sh', zsh: 'sh',
                                c: 'c', cpp: 'cpp', csharp: 'cs', cs: 'cs',
                                java: 'java', kotlin: 'kt', swift: 'swift',
                                go: 'go', rust: 'rs', ruby: 'rb', php: 'php',
                                perl: 'pl', lua: 'lua', r: 'r', matlab: 'm',
                                yaml: 'yaml', yml: 'yaml', toml: 'toml', ini: 'ini',
                                plaintext: 'txt', text: 'txt'
                            };
                            let ext = map[lang] || lang;
                            if (lang.length > 8 || /[^a-z0-9]/.test(lang)) ext = 'txt';
                            let filename = `code.${ext}`;
                            if (lang === 'dockerfile') filename = 'Dockerfile';
                            if (lang === 'makefile') filename = 'Makefile';
                            a.download = filename;
                            document.body.appendChild(a);
                            a.click();
                            document.body.removeChild(a);
                            URL.revokeObjectURL(url);
                        } catch (err) {
                            console.error('Download failed', err);
                        }
                    }
                }
                if (e.target.closest('.coding-target-btn')) {
                    selectCodingTargetFromButton(e.target.closest('.coding-target-btn'));
                }
                if (e.target.closest('.copy-btn')) {
                    const btn = e.target.closest('.copy-btn');
                    const code = btn.getAttribute('data-code');
                    if (code) window.copyCode(btn, code);
                }
                if (e.target.closest('.html-preview-btn')) {
                    const btn = e.target.closest('.html-preview-btn');
                    const code = btn.getAttribute('data-code');
                    if (code) openHtmlCodePreview(code);
                }
                if (e.target.closest('.canvas-preview-btn')) {
                    const btn = e.target.closest('.canvas-preview-btn');
                    previewCanvasCodeFromButton(btn);
                }
            });
            document.querySelectorAll('.modal-overlay').forEach((overlay) => {
                overlay.addEventListener('click', (e) => {
                    if (e.target === overlay) closeModalById(overlay.id);
                });
            });

            // Initial Load Logic for Permalink
            if(currentThreadId) { loadMessages(currentThreadId); }
            else { schedulePromptTokenEstimate(true); }
        });
