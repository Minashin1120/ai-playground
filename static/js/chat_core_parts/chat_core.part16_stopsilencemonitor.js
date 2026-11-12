            function stopSilenceMonitor() {
                if (stsSilenceInterval) { clearInterval(stsSilenceInterval); stsSilenceInterval = null; }
                if (stsSource) { try { stsSource.disconnect(); } catch(e) {} stsSource = null; }
                if (stsAudioCtx) { try { stsAudioCtx.close(); } catch(e) {} stsAudioCtx = null; }
                stsAnalyser = null;
            }
            function startSilenceMonitor(stream) {
                if (!isStsModel() || !stsOpt('sts-auto-send')) return;
                stopSilenceMonitor();
                const AC = window.AudioContext || window.webkitAudioContext;
                if (!AC) return;
                stsAudioCtx = new AC();
                stsAnalyser = stsAudioCtx.createAnalyser();
                stsAnalyser.fftSize = 2048;
                stsSource = stsAudioCtx.createMediaStreamSource(stream);
                stsSource.connect(stsAnalyser);
                const data = new Uint8Array(stsAnalyser.fftSize);
                const silenceMs = getStsSilenceMs();
                const threshold = 0.02;
                stsLastSoundTs = 0;
                stsHasSound = false;
                stsSilenceInterval = setInterval(() => {
                    if (!stsAnalyser) return;
                    stsAnalyser.getByteTimeDomainData(data);
                    let sum = 0;
                    for (let i = 0; i < data.length; i++) {
                        const v = (data[i] - 128) / 128;
                        sum += v * v;
                    }
                    const rms = Math.sqrt(sum / data.length);
                    if (rms > threshold) {
                        if (!stsHasSound) stsHasSound = true;
                        stsLastSoundTs = Date.now();
                        return;
                    }
                    if (stsHasSound && Date.now() - stsLastSoundTs > silenceMs) {
                        if (mediaRecorder && mediaRecorder.state === "recording") {
                            mediaRecorder.stop();
                        }
                    }
                }, 200);
            }
            class GeminiLiveClient {
                constructor() {
                    this.ws = null;
                    this.audioContext = null;
                    this.processor = null;
                    this.stream = null;
                    this.rtPlayer = null;
                    this.assistantText = '';
                    this.assistantThought = '';
                    this.inputTranscript = '';
                    this.interimInputTranscript = '';
                    this.assistantAudioChunks = [];
                    this.userAudioChunks = [];
                    this.onMessage = null;
                    this.onClose = null;
                    this.onError = null;
                    this.setupComplete = false;
                    this.model = null;
                    this.assistantTurnBreak = false;
                    this.userTurnBreak = false;
                }
                async start(token, url, model, config = {}, prepared = null) {
                    this.model = model;
                    this.ws = new WebSocket(`${url}?access_token=${encodeURIComponent(token)}`);
                    this.ws.binaryType = 'arraybuffer';

                    this.ws.onopen = () => {
                        console.log("Gemini Live WebSocket opened. Sending setup...");
                        // Live Transcription: TEXT output + input audio transcription config.
                        // Otherwise: audio-to-audio Live agent with transcription metadata.
                        const isTranscribeMode = !!(config && config.transcriptionConfig);
                        const setupMsg = {
                            setup: {
                                model: `models/${model}`,
                                generationConfig: {
                                    responseModalities: isTranscribeMode ? ["TEXT"] : ["AUDIO"]
                                },
                                // Enable transcription at setup level (as per docs)
                                inputAudioTranscription: isTranscribeMode
                                    ? (config.transcriptionConfig || {})
                                    : {}
                            }
                        };
                        // Text-only transcription has no model audio to transcribe.
                        if (!isTranscribeMode) setupMsg.setup.outputAudioTranscription = {};

                        // speechConfig is inside generationConfig
                        if (config.speechConfig) {
                            setupMsg.setup.generationConfig.speechConfig = config.speechConfig;
                        }

                        // thinkingConfig is also inside generationConfig in the Live API schema
                        if (config.thinkingConfig) {
                            setupMsg.setup.generationConfig.thinkingConfig = config.thinkingConfig;
                        }

                        // Live Translate: translationConfig is a generationConfig field
                        if (config.translationConfig) {
                            setupMsg.setup.generationConfig.translationConfig = config.translationConfig;
                        }

                        console.log("Sending setup:", JSON.stringify(setupMsg));
                        this.ws.send(JSON.stringify(setupMsg));
                    };

                    this.ws.onmessage = (e) => this._handleMessage(e);
                    this.ws.onerror = (e) => {
                        console.error("Gemini Live WebSocket error:", e);
                        if (this.onError) this.onError(e);
                    };
                    this.ws.onclose = (e) => {
                        console.log("Gemini Live WebSocket closed:", e.code, e.reason);
                        this.closedEvent = e;
                        if (this.onClose) this.onClose(e);
                    };

                    if (prepared && prepared.stream && prepared.ctx && prepared.source) {
                        this.stream = prepared.stream;
                        this.audioContext = prepared.ctx;
                        var source = prepared.source;
                    } else {
                        this.stream = await navigator.mediaDevices.getUserMedia(getMicCaptureConstraints());
                        const mic = openMicAudioSource(this.stream, 16000);
                        this.audioContext = mic.ctx;
                        var source = mic.source;
                    }
                    if (this.audioContext.state === 'suspended') {
                        try { await this.audioContext.resume(); } catch (e) {}
                    }
                    this.processor = this.audioContext.createScriptProcessor(4096, 1, 1);

                    this.userAudioChunks = [];
                    const backupRecorder = new MediaRecorder(this.stream);
                    backupRecorder.ondataavailable = (e) => {
                        if (e.data.size > 0) this.userAudioChunks.push(e.data);
                    };
                    backupRecorder.start(500);
                    this.backupRecorder = backupRecorder;

                    this.processor.onaudioprocess = (e) => {
                        if (!this.ws || this.ws.readyState !== WebSocket.OPEN || !this.setupComplete) return;
                        const pcm = pcm16FromFloat32(e.inputBuffer.getChannelData(0), this.audioContext.sampleRate, 16000);
                        if (!pcm || !pcm.byteLength) return;
                        this.ws.send(JSON.stringify({
                            realtimeInput: {
                                audio: {
                                    data: btoa(String.fromCharCode.apply(null, new Uint8Array(pcm))),
                                    mimeType: "audio/pcm;rate=16000"
                                }
                            }
                        }));
                    };
                    source.connect(this.processor);
                    this.processor.connect(this.audioContext.destination);
                }
                _handleMessage(e) {
                    let data = null;
                    try {
                        data = JSON.parse(typeof e.data === 'string' ? e.data : new TextDecoder().decode(e.data));
                    } catch (err) {
                        return;
                    }

                    if (data.setupComplete) {
                        console.log("Gemini Live setup complete confirmed");
                        this.setupComplete = true;
                    }
                    if (data.serverContent) {
                        const sc = data.serverContent;
                        if (sc.modelTurn) {
                            sc.modelTurn.parts.forEach(p => {
                                if (p.text) {
                                    if (p.thought) {
                                        console.log("Gemini thought delta:", p.text);
                                        this.assistantThought += p.text;
                                    } else if (!(this.model === 'gemini-3.5-transcribe-live')) {
                                        this._appendAssistantText(p.text);
                                    }
                                }
                                if (p.inlineData && p.inlineData.data) {
                                    const audioBase64 = p.inlineData.data;
                                    console.log("Gemini audio chunk received, size:", audioBase64.length);
                                    if (this.rtPlayer) this.rtPlayer.addChunk(audioBase64);
                                    const binary = atob(audioBase64);
                                    const bytes = new Uint8Array(binary.length);
                                    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
                                    this.assistantAudioChunks.push(bytes);
                                }
                            });
                        }
                        // Native-audio models deliver the spoken text only here.
                        if (sc.outputTranscription && sc.outputTranscription.text) {
                            this._appendAssistantText(sc.outputTranscription.text);
                        }
                        if (sc.inputTranscription && sc.inputTranscription.text) {
                            if (this.userTurnBreak && this.inputTranscript && !this.inputTranscript.endsWith('\n')) this.inputTranscript += '\n';
                            this.userTurnBreak = false;
                            this.inputTranscript += sc.inputTranscription.text;
                            this.interimInputTranscript = '';
                        }
                        if (sc.interimInputTranscription) {
                            this.interimInputTranscript = sc.interimInputTranscription.text || '';
                        }
                        if (sc.turnComplete) {
                            this.assistantTurnBreak = true;
                            if (this.model !== 'gemini-3.5-transcribe-live') this.userTurnBreak = true;
                        }
                    }
                    if (data.error && this.onError) this.onError(data.error);
                    if (this.onMessage) this.onMessage(data);
                }
                _appendAssistantText(text) {
                    if (!text) return;
                    if (this.assistantTurnBreak && this.assistantText && !this.assistantText.endsWith('\n')) this.assistantText += '\n';
                    this.assistantTurnBreak = false;
                    this.assistantText += text;
                }
                stop() {
                    // Stop the recorder before the mic tracks. Stopping the tracks first drops
                    // the buffered recording, and getFinalData used to read it before `stop` fired.
                    const recorder = this.backupRecorder;
                    this.backupRecorder = null;
                    this._stopped = new Promise((resolve) => {
                        let settled = false;
                        const finish = () => { if (!settled) { settled = true; resolve(); } };
                        if (!recorder || recorder.state === 'inactive') { finish(); return; }
                        recorder.addEventListener('stop', finish, { once: true });
                        try { recorder.stop(); } catch (e) { finish(); }
                        setTimeout(finish, 1500);
                    });
                    if (this.ws) { try { this.ws.close(); } catch (e) {} }
                    if (this.processor) { try { this.processor.disconnect(); } catch (e) {} }
                    const release = () => {
                        if (this.stream) this.stream.getTracks().forEach(t => { try { t.stop(); } catch (e) {} });
                        if (this.audioContext) { try { this.audioContext.close(); } catch (e) {} }
                    };
                    this._stopped.then(release);
                }
                async getFinalData() {
                    if (this._stopped) await this._stopped;
                    const assistantAudioBlob = new Blob(this.assistantAudioChunks);
                    const assistantAudioBase64 = await this._blobToBase64(assistantAudioBlob);
                    const userAudioBlob = new Blob(this.userAudioChunks);
                    const userAudioBase64 = await this._blobToBase64(userAudioBlob);
                    return {
                        user_text: this.inputTranscript,
                        assistant_text: this.assistantText,
                        assistant_thought: this.assistantThought,
                        audio_base64: assistantAudioBase64,
                        user_audio_base64: userAudioBase64
                    };
                }
                _blobToBase64(blob) {
                    return new Promise(resolve => {
                        const reader = new FileReader();
                        reader.onloadend = () => resolve(reader.result.split(',')[1]);
                        reader.readAsDataURL(blob);
                    });
                }
            }
            class RealTimeAudioPlayer {
                constructor(sampleRate = 24000) {
                    const AC = window.AudioContext || window.webkitAudioContext;
                    this.ctx = new AC({ sampleRate });
                    this.nextStartTime = 0;
                    this.bufferDelay = 0.1; // 100ms jitter buffer
                    this.started = false;
                }
                async addChunk(base64Data) {
                    if (!this.ctx) return;
                    const binary = atob(base64Data);
                    const bytes = new Uint8Array(binary.length);
                    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
                    const pcm = new Int16Array(bytes.buffer);
                    const float32 = new Float32Array(pcm.length);
                    for (let i = 0; i < pcm.length; i++) float32[i] = pcm[i] / 32768.0;

                    const buffer = this.ctx.createBuffer(1, float32.length, this.ctx.sampleRate);
                    buffer.getChannelData(0).set(float32);

                    if (this.ctx.state === 'suspended') await this.ctx.resume();

                    const source = this.ctx.createBufferSource();
                    source.buffer = buffer;
                    source.connect(this.ctx.destination);

                    if (!this.started) {
                        this.nextStartTime = this.ctx.currentTime + this.bufferDelay;
                        this.started = true;
                    }
                    const startTime = Math.max(this.ctx.currentTime, this.nextStartTime);
                    source.start(startTime);
                    this.nextStartTime = startTime + buffer.duration;
                }
                stop() {
                    if (this.ctx) {
                        this.ctx.close();
                        this.ctx = null;
                    }
                }
            }

            // ===========================================================================
            // RealtimeVoiceSession — true real-time STS over a server-held session.
            // The server keeps the provider WebSocket open; this controller streams
            // microphone PCM to the server via HTTP, consumes audio/transcripts over
            // SSE, and plays responses while the user keeps talking (server VAD or
            // Gemini's natural turn handling drives turn-taking).
            // ===========================================================================
            class RealtimeVoiceSession {
                constructor() {
                    this.active = false;
                    this.capturing = false;
                    this.sessionId = null;
                    this.abortCtrl = null;
                    this.reader = null;
                    this.audioCtx = null;
                    this.processor = null;
                    this.stream = null;
                    this.rtPlayer = null;
                    this.rateIn = 24000;
                    this.rateOut = 24000;
                    this.userTranscript = '';
                    this.assistantTranscript = '';
                    this.assistantThought = '';
                    this.speechActive = false;
                    this.responseDoneCount = 0;
                    this.lastAudioAt = 0;
                    this.streamError = null;
                    this.saved = false;
                    this.saving = false;
                    this.stopping = false;
                    this.audioQueue = [];
                    this.audioFlush = null;
                }

                isActive() {
                    return this.active;
                }

                async start() {
                    if (this.active) return;
                    if (this.saving || this.stopping) {
                        showToast('前の会話を処理中です。しばらくお待ちください。', 'warning', true);
                        return;
                    }
                    const model = get('model-select') ? get('model-select').value : '';
                    if (!isRealtimeSessionModel()) {
                        showToast('このモデルはリアルタイム会話に対応していません', 'warning', true);
                        return;
                    }
                    if (!currentThreadId) {
                        try {
                            const r = await apiFetch(CHAT_CONFIG.urls.handleThreads, {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
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
                        } catch (e) {
                            showToast('スレッドの作成に失敗しました: ' + e.message, 'error', true);
                            return;
                        }
                    }
                    const payload = {
                        model,
                        thread_id: currentThreadId,
                        voice: get('sts-voice') ? get('sts-voice').value : '',
                        speed: get('sts-speed') ? get('sts-speed').value : '',
                        rate_in: get('sts-rate-in') ? get('sts-rate-in').value : '',
                        rate_out: get('sts-rate-out') ? get('sts-rate-out').value : '',
                        thinking_level: get('sts-thinking-level') ? get('sts-thinking-level').value : '',
                        include_thoughts: get('sts-include-thoughts') ? get('sts-include-thoughts').checked : false,
                        reasoning_effort: get('sts-reasoning-effort') ? get('sts-reasoning-effort').value : '',
                        target_lang: ((isGeminiLiveTranslateModel() || model === 'gpt-realtime-translate') && get('sts-target-lang')) ? get('sts-target-lang').value : ''
                    };
                    if (isXaiLiveTranscribeModel() && get('sts-custom-vocab')) payload.custom_vocabulary = get('sts-custom-vocab').value.split(/[,、\n]/);
                    setStsStatus('接続中...', true);
                    try {
                        const resp = await apiFetch('/api/realtime/start', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify(payload)
                        });
                        const data = await resp.json().catch(() => ({}));
                        if (!resp.ok) throw new Error(data.error || 'セッション開始に失敗しました');
                        this.sessionId = data.session_id;
                        this.rateIn = data.rate_in || this.rateIn;
                        this.rateOut = data.rate_out || this.rateOut;
                        this.active = true;
                        this.capturing = true;
                        this.saved = false;
                        this.userTranscript = '';
                        this.assistantTranscript = '';
                        this.assistantThought = '';
                        this.responseDoneCount = 0;
                        this.lastAudioAt = 0;
                        this.streamError = null;
                        this.rtPlayer = null;
                        this.audioQueue = [];
                        this.audioFlush = null;
                    } catch (e) {
                        setStsStatus('接続エラー', false);
                        showToast('リアルタイムセッションを開始できませんでした: ' + e.message, 'error', true);
                        return;
                    }

                    // Open the SSE stream first so audio events are not missed.
                    this.abortCtrl = new AbortController();
                    this._openStream();

                    // Start mic capture and stream PCM chunks to the server.
                    try {
                        await this._startCapture();
                    } catch (e) {
                        setStsStatus('マイクエラー', false);
                        showToast('マイクを利用できません: ' + e.message, 'error', true);
                        this._cancel();
                        return;
                    }

                    get('mic-btn').classList.remove('bg-gray-700');
                    get('mic-btn').classList.add('bg-red-600', 'animate-pulse');
                    setStsStatus('話してください...', true);
                }

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
                    if (!this.sessionId || !this.active) return;
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
                    try {
                        await apiFetch('/api/realtime/commit', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ session_id: this.sessionId })
                        });
                    } catch (e) {}
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
                        const resp = await apiFetch('/api/realtime/save', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ session_id: this.sessionId, thread_id: currentThreadId })
                        });
                        const data = await resp.json().catch(() => ({}));
                        if (!resp.ok) throw new Error(data.error || '保存に失敗しました');
                        if (data.thread_id && !currentThreadId) currentThreadId = String(data.thread_id);
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

                _cancel() {
                    if (this.sessionId) {
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
                        const res = await fetch(`/api/skillini/music/stream?session_id=${encodeURIComponent(sessionId)}`, noSpinner({
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
