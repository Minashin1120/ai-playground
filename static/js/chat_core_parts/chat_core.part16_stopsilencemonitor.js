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
                    // Browser-direct Gemini Live transport (ephemeral token).
                    // Non-Gemini models (and Vertex AI) use the server session.
                    this.direct = false;
                    this.ws = null;
                    this.directReady = null;
                    this.directUrl = '';
                    this.directToken = '';
                    this.directExpiresAt = 0;
                    this.resumeHandle = null;
                    this.directResumes = 0;
                    this.directSetup = null;
                    this.directModel = '';
                    this.directTranscribe = false;
                    this.directTranslate = false;
                    this.directThinking = false;
                    this.transcriptionMode = 'VERBATIM';
                    this.assistantAudioChunks = [];
                    this.assistantAudioBytes = 0;
                    this.userRecorder = null;
                    this.userAudioChunks = [];
                    this.userRecorderDone = null;
                    this.userTurnBreak = false;
                    this.assistantTurnBreak = false;
                    this.goAwayNotified = false;
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
                    if (isGeminiLiveTranscribeModel()) {
                        payload.transcription_mode = get('sts-transcribe-mode') ? get('sts-transcribe-mode').value : 'VERBATIM';
                        payload.custom_vocabulary = get('sts-custom-vocab')
                            ? get('sts-custom-vocab').value.split(/[,、\n]/).map((s) => s.trim()).filter(Boolean).slice(0, 1000)
                            : [];
                    }
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
                    this.assistantAudioChunks = [];
                    this.assistantAudioBytes = 0;
                    this.userAudioChunks = [];
                    this.userTurnBreak = false;
                    this.assistantTurnBreak = false;
                    setStsStatus('接続中...', true);
                    let connected = false;
                    if (isGeminiLiveModel()) {
                        connected = await this._tryStartDirect(payload);
                    }
                    if (!connected) {
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
                        } catch (e) {
                            setStsStatus('接続エラー', false);
                            showToast('リアルタイムセッションを開始できませんでした: ' + e.message, 'error', true);
                            return;
                        }
                        // Open the SSE stream first so audio events are not missed.
                        this.abortCtrl = new AbortController();
                        this._openStream();
                    }
                    this.active = true;
                    this.capturing = true;

                    // Start mic capture: direct models send PCM over the
                    // WebSocket, server sessions POST it to /api/realtime/audio.
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

