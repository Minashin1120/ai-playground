                // (RealtimeVoiceSession and the remaining voice helpers continued from part16b.)
                _openStream() {
                    const url = '/api/realtime/stream?session_id=' + encodeURIComponent(this.sessionId);
                    const opts = window.ProgressSpinner && typeof window.ProgressSpinner.manualRequestOptions === 'function'
                        ? window.ProgressSpinner.manualRequestOptions({ credentials: 'include', signal: this.abortCtrl.signal })
                        : { credentials: 'include', signal: this.abortCtrl.signal };
                    fetch(url, opts).then((resp) => {
                        if (!resp.ok) throw new Error('SSE stream failed (' + resp.status + ')');
                        this.reader = resp.body.getReader();
                        this._readLoop();
                    }).catch((e) => {
                        if (e && e.name === 'AbortError') return;
                        this.streamError = e && e.message ? e.message : 'ストリームエラー';
                        if (this.active) {
                            setStsStatus('ストリームエラー', false);
                            showToast('リアルタイム接続が切断されました', 'error', true);
                        }
                    });
                }

                async _readLoop() {
                    const decoder = new TextDecoder();
                    let buf = '';
                    try {
                        while (this.reader) {
                            const { done, value } = await this.reader.read();
                            if (done) break;
                            buf += decoder.decode(value, { stream: true });
                            let idx;
                            while ((idx = buf.indexOf('\n\n')) >= 0) {
                                const chunk = buf.slice(0, idx);
                                buf = buf.slice(idx + 2);
                                for (const line of chunk.split('\n')) {
                                    if (!line.startsWith('data: ')) continue;
                                    let ev = null;
                                    try { ev = JSON.parse(line.slice(6)); } catch (e) { continue; }
                                    this._handleEvent(ev);
                                }
                            }
                        }
                    } catch (e) {
                        if (e && e.name === 'AbortError') return;
                        if (this.active) {
                            this.streamError = e && e.message ? e.message : 'ストリームエラー';
                        }
                    } finally {
                        this.reader = null;
                    }
                }

                _handleEvent(ev) {
                    if (!ev) return;
                    switch (ev.type) {
                        case 'audio':
                            this.lastAudioAt = Date.now();
                            if (stsOpt('sts-auto-play')) {
                                if (!this.rtPlayer) {
                                    this.rtPlayer = new RealTimeAudioPlayer(this.rateOut || 24000);
                                    currentRtPlayer = this.rtPlayer;
                                }
                                setStsStatus('再生中...', true);
                                this.rtPlayer.addChunk(ev.data);
                            }
                            break;
                        case 'transcript':
                            if (ev.role === 'user') {
                                if (ev.cumulative) {
                                    this.userTranscript = ev.delta;
                                } else {
                                    this.userTranscript += ev.delta;
                                }
                                if (window.VoiceStudio) window.VoiceStudio.log('user', this.userTranscript);
                            } else if (ev.role === 'assistant') {
                                this.assistantTranscript += ev.delta;
                                if (window.VoiceStudio) window.VoiceStudio.log('assistant', this.assistantTranscript);
                            } else if (ev.role === 'thought') {
                                this.assistantThought += ev.delta;
                            }
                            break;
                        case 'speech_started':
                            this.speechActive = true;
                            this._stopPlayback();
                            setStsStatus('聞き取り中...', true);
                            break;
                        case 'speech_stopped':
                            this.speechActive = false;
                            setStsStatus('応答待ち...', true);
                            break;
                        case 'interrupted':
                            this._stopPlayback();
                            break;
                        case 'response_done':
                        case 'turn_complete':
                            this.responseDoneCount += 1;
                            break;
                        case 'status':
                            if (ev.status === 'ready' && this.active) setStsStatus('話してください...', true);
                            break;
                        case 'interaction_status':
                            if (this.active) setStsStatus(ev.status === 'IN_PROGRESS' ? '考え中...' : '話してください...', true);
                            break;
                        case 'notice':
                            // Recoverable provider error: the session stays open.
                            if (ev.message) showToast('リアルタイム音声: ' + ev.message, 'warning', true);
                            break;
                        case 'error':
                            this.streamError = ev.message || 'リアルタイムエラー';
                            setStsStatus('エラー', false);
                            break;
                        case 'final':
                            if (this.active && !this.saved) {
                                // The worker stopped; close out cleanly.
                                this._save();
                            }
                            break;
                    }
                }

                _stopPlayback() {
                    if (this.rtPlayer) {
                        try { this.rtPlayer.stop(); } catch (e) {}
                        this.rtPlayer = null;
                    }
                    currentRtPlayer = null;
                }

                _startCapture() {
                    return navigator.mediaDevices.getUserMedia(getMicCaptureConstraints()).then((stream) => {
                        this.stream = stream;
                        if (this.direct && window.MediaRecorder) {
                            try {
                                this.userAudioChunks = [];
                                try {
                                    // Keep the saved microphone audio small
                                    // enough for the 25 MB save limit.
                                    this.userRecorder = new MediaRecorder(stream, { audioBitsPerSecond: 32000 });
                                } catch (e) {
                                    this.userRecorder = new MediaRecorder(stream);
                                }
                                this.userRecorder.ondataavailable = (ev) => {
                                    if (ev.data && ev.data.size) this.userAudioChunks.push(ev.data);
                                };
                                this.userRecorder.start(500);
                            } catch (e) {
                                this.userRecorder = null;
                            }
                        }
                        const targetRate = this.rateIn || 24000;
                        const mic = openMicAudioSource(stream, targetRate);
                        this.audioCtx = mic.ctx;
                        const source = mic.source;
                        const ctxRate = this.audioCtx.sampleRate;
                        const bufSize = 4096;
                        this.processor = this.audioCtx.createScriptProcessor(bufSize, 1, 1);
                        this.processor.onaudioprocess = (e) => {
                            if (!this.active || !this.capturing) return;
                            const input = e.inputBuffer.getChannelData(0);
                            const pcm = pcm16FromFloat32(input, ctxRate, targetRate);
                            if (!pcm || !pcm.byteLength) return;
                            this._sendAudio(pcm);
                        };
                        source.connect(this.processor);
                        this.processor.connect(this.audioCtx.destination);
                    });
                }

                _sendAudio(pcmBytes) {
                    if (!this.active) return;
                    if (this.direct) {
                        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
                        try {
                            this.ws.send(JSON.stringify({
                                realtimeInput: {
                                    audio: {
                                        data: bytesToBase64(new Uint8Array(pcmBytes)),
                                        mimeType: 'audio/pcm;rate=' + (this.rateIn || 16000)
                                    }
                                }
                            }));
                        } catch (e) {}
                        return;
                    }
                    if (!this.sessionId) return;
                    // One request at a time: parallel POSTs can reach the server
                    // out of order and scramble the audio stream.
                    this.audioQueue.push(new Uint8Array(pcmBytes));
                    if (!this.audioFlush) this.audioFlush = this._flushAudio();
                }

                async _flushAudio() {
                    try {
                        while (this.audioQueue.length && this.sessionId) {
                            const chunks = this.audioQueue.splice(0, this.audioQueue.length);
                            const total = chunks.reduce((n, c) => n + c.byteLength, 0);
                            const body = new Uint8Array(total);
                            let offset = 0;
                            chunks.forEach((c) => { body.set(c, offset); offset += c.byteLength; });
                            const url = '/api/realtime/audio?session_id=' + encodeURIComponent(this.sessionId);
                            const opts = {
                                method: 'POST',
                                credentials: 'include',
                                headers: { 'X-CSRF-Token': csrfToken, 'Content-Type': 'application/octet-stream' },
                                body: body.buffer
                            };
                            const finalOpts = window.ProgressSpinner && typeof window.ProgressSpinner.manualRequestOptions === 'function'
                                ? window.ProgressSpinner.manualRequestOptions(opts)
                                : opts;
                            try { await fetch(url, finalOpts); } catch (e) {}
                        }
                    } finally {
                        this.audioFlush = null;
                    }
                }

                _stopCapture() {
                    this.capturing = false;
                    if (this.processor) { try { this.processor.disconnect(); } catch (e) {} this.processor = null; }
                    if (this.userRecorder) {
                        const rec = this.userRecorder;
                        this.userRecorder = null;
                        this.userRecorderDone = new Promise((resolve) => {
                            let done = false;
                            const finish = () => { if (!done) { done = true; resolve(); } };
                            rec.addEventListener('stop', finish, { once: true });
                            try { rec.stop(); } catch (e) { finish(); }
                            setTimeout(finish, 1500);
                        });
                    }
                    if (this.stream) { try { this.stream.getTracks().forEach((t) => t.stop()); } catch (e) {} this.stream = null; }
                    if (this.audioCtx) { try { this.audioCtx.close(); } catch (e) {} this.audioCtx = null; }
                    stopSilenceMonitor();
                    stopMicWaveform();
                }

                async stop() {
                    if (!this.active) return;
                    this.active = false;
                    this.stopping = true;
                    this._stopCapture();
                    setStsStatus('応答を待っています...', true);
                    // Deliver the audio still queued before ending the input.
                    if (this.audioFlush) { try { await this.audioFlush; } catch (e) {} }
                    // Finalize any trailing audio so the last turn is included.
                    if (this.direct) {
                        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                            try { this.ws.send(JSON.stringify({ realtimeInput: { audioStreamEnd: true } })); } catch (e) {}
                        }
                    } else {
                        try {
                            await apiFetch('/api/realtime/commit', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({ session_id: this.sessionId })
                            });
                        } catch (e) {}
                    }
                    // Wait for the final response (or a quiet timeout) before saving.
                    const startedAt = Date.now();
                    const beforeCount = this.responseDoneCount;
                    let lastActivity = this.lastAudioAt;
                    while (Date.now() - startedAt < 20000) {
                        if (this.responseDoneCount > beforeCount) break;
                        if (this.lastAudioAt > lastActivity) lastActivity = this.lastAudioAt;
                        if (!this.speechActive && Date.now() - startedAt > 2000 && Date.now() - lastActivity > 2500) break;
                        await new Promise((r) => setTimeout(r, 250));
                    }
                    await this._save();
                }

                async _save() {
                    if (this.saved) return;
                    this.saved = true;
                    this.saving = true;
                    try {
                        if (this.direct) {
                            await this._postDirectSave();
                        } else {
                            const resp = await apiFetch('/api/realtime/save', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({ session_id: this.sessionId, thread_id: currentThreadId })
                            });
                            const data = await resp.json().catch(() => ({}));
                            if (!resp.ok) throw new Error(data.error || '保存に失敗しました');
                            if (data.thread_id && !currentThreadId) currentThreadId = String(data.thread_id);
                        }
                        if (this.streamError) {
                            setStsStatus('エラー', false);
                            showToast('リアルタイム会話でエラーが発生しました: ' + this.streamError, 'error', true);
                        } else {
                            setStsStatus('保存しました', false);
                            setTimeout(() => setStsStatus('Tap to speak', false), 1200);
                            try { await loadMessages(currentThreadId); } catch (e) {}
                        }
                    } catch (e) {
                        setStsStatus('保存エラー', false);
                        showToast('音声会話の保存に失敗しました: ' + (e && e.message ? e.message : e), 'error', true);
                    } finally {
                        this.saving = false;
                        this.stopping = false;
                        this._cleanup();
                    }
                }

                async _postDirectSave() {
                    // The browser collected PCM 24 kHz assistant audio and the
                    // raw microphone recording; persist them like the previous
                    // direct Gemini flow did.
                    let assistantB64 = '';
                    if (this.assistantAudioChunks.length) {
                        const total = this.assistantAudioChunks.reduce((n, c) => n + c.byteLength, 0);
                        const merged = new Uint8Array(total);
                        let offset = 0;
                        this.assistantAudioChunks.forEach((c) => { merged.set(c, offset); offset += c.byteLength; });
                        assistantB64 = bytesToBase64(merged);
                    }
                    if (this.userRecorderDone) { try { await this.userRecorderDone; } catch (e) {} }
                    const userB64 = this.userAudioChunks.length
                        ? await blobToBase64(new Blob(this.userAudioChunks))
                        : '';
                    const resp = await apiFetch('/api/gemini/save_sts', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            thread_id: currentThreadId,
                            model: this.directModel || get('model-select').value,
                            user_text: this.userTranscript,
                            assistant_text: this.assistantTranscript,
                            assistant_thought: this.assistantThought,
                            audio_base64: assistantB64,
                            user_audio_base64: userB64,
                            usage_metadata: this.usageMetadata || null,
                            duration_ms: this.directStartedAt ? Math.max(0, Date.now() - this.directStartedAt) : null
                        })
                    });
                    const data = await resp.json().catch(() => ({}));
                    if (!resp.ok) throw new Error(data.error || '保存に失敗しました');
                    if (data.thread_id && !currentThreadId) currentThreadId = String(data.thread_id);
                }

                _cancel() {
                    if (this.direct) {
                        this.active = false;
                        if (this.ws) { try { this.ws.close(1000); } catch (e) {} this.ws = null; }
                    } else if (this.sessionId) {
                        apiFetch('/api/realtime/cancel', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ session_id: this.sessionId })
                        }).catch(() => {});
                    }
                    this._cleanup();
                    setStsStatus('Canceled', false);
                    setTimeout(() => setStsStatus('Tap to speak', false), 800);
                }

                _cleanup() {
                    this.active = false;
                    this.capturing = false;
                    this.stopping = false;
                    this.audioQueue = [];
                    if (this.ws) { try { this.ws.close(1000); } catch (e) {} this.ws = null; }
                    this.direct = false;
                    this.directReady = null;
                    this.directSetup = null;
                    this.resumeHandle = null;
                    this.assistantAudioChunks = [];
                    this.assistantAudioBytes = 0;
                    this.userAudioChunks = [];
                    this.userRecorderDone = null;
                    this._stopCapture();
                    this._stopPlayback();
                    if (this.abortCtrl) { try { this.abortCtrl.abort(); } catch (e) {} this.abortCtrl = null; }
                    this.reader = null;
                    this.sessionId = null;
                    const micBtn = get('mic-btn');
                    if (micBtn) {
                        micBtn.classList.remove('bg-red-600', 'animate-pulse');
                        micBtn.classList.add('bg-gray-700');
                    }
                }
            }

            // Microphone source at the provider rate when the browser allows it.
            // Firefox cannot connect a mic stream to a context with another
            // sample rate; fall back to the device rate (callers resample).
            function openMicAudioSource(stream, targetRate) {
                const AC = window.AudioContext || window.webkitAudioContext;
                if (!AC) throw new Error('AudioContext not supported');
                let ctx = null;
                try {
                    ctx = new AC({ sampleRate: targetRate });
                    return { ctx, source: ctx.createMediaStreamSource(stream) };
                } catch (e) {
                    if (ctx) { try { ctx.close(); } catch (err) {} }
                    ctx = new AC();
                    return { ctx, source: ctx.createMediaStreamSource(stream) };
                }
            }

            function pcm16FromFloat32(input, ctxRate, targetRate) {
                let data = input;
                if (ctxRate !== targetRate && ctxRate > 0 && targetRate > 0) {
                    const ratio = ctxRate / targetRate;
                    const outLen = Math.floor(data.length / ratio);
                    const down = new Float32Array(outLen);
                    for (let i = 0; i < outLen; i++) {
                        down[i] = data[Math.min(Math.floor(i * ratio), data.length - 1)];
                    }
                    data = down;
                }
                const pcm = new Int16Array(data.length);
                for (let i = 0; i < data.length; i++) {
                    const s = Math.max(-1, Math.min(1, data[i]));
                    pcm[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
                }
                return pcm.buffer;
            }

            function bytesToBase64(bytes) {
                let binary = '';
                const step = 0x8000;
                for (let i = 0; i < bytes.length; i += step) {
                    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + step));
                }
                return btoa(binary);
            }

            function blobToBase64(blob) {
                return new Promise((resolve, reject) => {
                    const reader = new FileReader();
                    reader.onload = () => resolve(String(reader.result || '').split(',')[1] || '');
                    reader.onerror = () => reject(reader.error || new Error('音声の読み取りに失敗しました'));
                    reader.readAsDataURL(blob);
                });
            }

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
                        // Play starts a new session when none is open yet.
                        playBtn.disabled = busy;
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
