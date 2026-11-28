                // ------------------------------------------------------------------
                // Browser-direct Gemini Live (ephemeral token, v1beta)
                // ------------------------------------------------------------------
                _buildDirectSetup(payload) {
                    const model = payload.model;
                    const transcribe = this.directTranscribe;
                    const translate = this.directTranslate;
                    const generationConfig = {
                        responseModalities: transcribe ? ['TEXT'] : ['AUDIO']
                    };
                    const setup = {
                        model: 'models/' + model,
                        generationConfig,
                        // Ask for resumption handles so a provider-side reset
                        // can reconnect within the token lifetime (docs).
                        sessionResumption: {}
                    };
                    if (transcribe) {
                        setup.inputAudioTranscription = {
                            mode: payload.transcription_mode || 'VERBATIM',
                            languageCodes: []
                        };
                        if (payload.custom_vocabulary && payload.custom_vocabulary.length) {
                            setup.inputAudioTranscription.customVocabulary = payload.custom_vocabulary;
                        }
                    } else {
                        setup.inputAudioTranscription = {};
                        setup.outputAudioTranscription = {};
                        if (translate) {
                            generationConfig.translationConfig = {
                                targetLanguageCode: normalizeGeminiTargetLang(payload.target_lang),
                                echoTargetLanguage: false
                            };
                        } else if (payload.voice) {
                            generationConfig.speechConfig = {
                                voiceConfig: { prebuiltVoiceConfig: { voiceName: payload.voice } }
                            };
                        }
                        if (this.directThinking && payload.thinking_level) {
                            generationConfig.thinkingConfig = {
                                thinkingLevel: payload.thinking_level,
                                includeThoughts: !!payload.include_thoughts
                            };
                        }
                    }
                    return { setup };
                }

                async _tryStartDirect(payload) {
                    let info = null;
                    // Capture the panel choices now so a model change during
                    // the session cannot alter the running setup or save model.
                    this.directModel = payload.model;
                    this.directTranscribe = isGeminiLiveTranscribeModel();
                    this.directTranslate = isGeminiLiveTranslateModel();
                    this.directThinking = isGeminiLiveThinkingModel();
                    const body = { model: payload.model };
                    if (this.directTranscribe) {
                        body.transcription_mode = payload.transcription_mode || 'VERBATIM';
                        body.custom_vocabulary = payload.custom_vocabulary || [];
                    } else {
                        body.voice = payload.voice;
                        body.thinking_level = payload.thinking_level;
                        body.include_thoughts = payload.include_thoughts;
                        body.target_lang = payload.target_lang;
                    }
                    try {
                        const resp = await apiFetch('/api/gemini/session', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify(body)
                        });
                        const data = await resp.json().catch(() => ({}));
                        if (!resp.ok || !data.direct || !data.token) {
                            if (data.reason !== 'vertex_ai') {
                                console.warn('Gemini Live: browser-direct unavailable, using the server session', data.reason || data.error || resp.status);
                            }
                            return false;
                        }
                        info = data;
                    } catch (e) {
                        console.warn('Gemini Live: token request failed, using the server session', e);
                        return false;
                    }
                    this.rateIn = info.rate_in || 16000;
                    this.rateOut = info.rate_out || 24000;
                    this.directUrl = info.url;
                    this.directToken = info.token;
                    this.directExpiresAt = Date.parse(info.expires_at || '') || 0;
                    this.directSetup = this._buildDirectSetup(payload);
                    this.transcriptionMode = (payload.transcription_mode || 'VERBATIM').toUpperCase();
                    this.resumeHandle = null;
                    this.directResumes = 0;
                    this.goAwayNotified = false;
                    this.usageMetadata = null;
                    this.directStartedAt = Date.now();
                    try {
                        await this._connectDirect();
                        return true;
                    } catch (e) {
                        console.warn('Gemini Live: direct connection failed, using the server session', e);
                        this.direct = false;
                        this.directSetup = null;
                        if (this.ws) { try { this.ws.close(); } catch (err) {} this.ws = null; }
                        this.rateIn = 24000;
                        this.rateOut = 24000;
                        return false;
                    }
                }

                _connectDirect() {
                    return new Promise((resolve, reject) => {
                        let settled = false;
                        const finish = (err) => {
                            if (settled) return;
                            settled = true;
                            clearTimeout(timer);
                            if (err) reject(err); else resolve();
                        };
                        const timer = setTimeout(() => finish(new Error('接続がタイムアウトしました')), 12000);
                        let ws;
                        try {
                            ws = new WebSocket(this.directUrl + '?access_token=' + encodeURIComponent(this.directToken));
                        } catch (e) {
                            finish(e);
                            return;
                        }
                        this.ws = ws;
                        this.direct = true;
                        this.directReady = finish;
                        ws.binaryType = 'arraybuffer';
                        ws.onopen = () => {
                            if (this.ws !== ws) return;
                            const msg = JSON.parse(JSON.stringify(this.directSetup));
                            if (this.resumeHandle) msg.setup.sessionResumption = { handle: this.resumeHandle };
                            try { ws.send(JSON.stringify(msg)); } catch (e) { finish(e); }
                        };
                        ws.onmessage = (e) => { if (this.ws === ws) this._handleDirectMessage(e); };
                        ws.onerror = () => finish(new Error('接続に失敗しました'));
                        ws.onclose = (ev) => { if (this.ws === ws) this._directClosed(ev); };
                    });
                }

                _handleDirectMessage(e) {
                    let msg = null;
                    try {
                        msg = JSON.parse(typeof e.data === 'string' ? e.data : new TextDecoder().decode(e.data));
                    } catch (err) { return; }
                    if (msg.setupComplete !== undefined) {
                        const ready = this.directReady;
                        this.directReady = null;
                        if (ready) ready();
                    }
                    const um = msg.usageMetadata || msg.usage_metadata;
                    if (um && typeof um === 'object') {
                        this.usageMetadata = um;
                    }
                    const sc = msg.serverContent;
                    if (sc) {
                        if (sc.modelTurn) {
                            (sc.modelTurn.parts || []).forEach((p) => {
                                const inline = p.inlineData || {};
                                if (inline.data) this._playDirectAudio(inline.data);
                                if (p.text) {
                                    if (p.thought) {
                                        this.assistantThought += p.text;
                                    } else if (!this.directTranscribe) {
                                        this._appendDirectAssistant(p.text);
                                    }
                                }
                            });
                        }
                        if (sc.outputTranscription && sc.outputTranscription.text) {
                            this._appendDirectAssistant(sc.outputTranscription.text);
                        }
                        if (sc.inputTranscription && sc.inputTranscription.text) {
                            const text = String(sc.inputTranscription.text);
                            const smart = this.directTranscribe && this.transcriptionMode === 'SMART';
                            if (smart) {
                                this.userTranscript = text;
                            } else {
                                this.userTranscript += (this.userTurnBreak && this.userTranscript ? '\n' : '') + text;
                            }
                            this.userTurnBreak = false;
                            if (window.VoiceStudio) window.VoiceStudio.log('user', this.userTranscript);
                        }
                        if (sc.turnComplete) {
                            this.userTurnBreak = true;
                            this.assistantTurnBreak = true;
                            this.responseDoneCount += 1;
                        }
                    }
                    const interactionStatus = msg.interactionStatus
                        || (msg.serverContent && msg.serverContent.interactionStatus);
                    if (interactionStatus) {
                        const st = String(interactionStatus).toUpperCase();
                        setStsStatus(st === 'IN_PROGRESS' ? '考え中...' : '話してください...', true);
                        if (st === 'IDLE') this.responseDoneCount += 1;
                    }
                    const update = msg.sessionResumptionUpdate;
                    if (update && update.resumable && update.newHandle) this.resumeHandle = update.newHandle;
                    if (msg.goAway && !this.goAwayNotified) {
                        this.goAwayNotified = true;
                        showToast('Gemini Liveのセッション上限が近づいています。まもなく自動で終了します。', 'warning', true);
                    }
                    if (msg.error) {
                        this.streamError = (msg.error && msg.error.message) || 'Gemini Live エラー';
                        setStsStatus('エラー', false);
                    }
                }

                _appendDirectAssistant(text) {
                    if (!text) return;
                    if (this.assistantTurnBreak && this.assistantTranscript && !this.assistantTranscript.endsWith('\n')) text = '\n' + text;
                    this.assistantTurnBreak = false;
                    this.assistantTranscript += text;
                    if (window.VoiceStudio) window.VoiceStudio.log('assistant', this.assistantTranscript);
                }

                _playDirectAudio(data) {
                    this.lastAudioAt = Date.now();
                    const binary = atob(data);
                    const bytes = new Uint8Array(binary.length);
                    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
                    // save_sts accepts 25 MB; stop buffering before that so a
                    // long session still saves its transcript and newest audio.
                    if (this.assistantAudioBytes + bytes.length <= 20 * 1024 * 1024) {
                        this.assistantAudioChunks.push(bytes);
                        this.assistantAudioBytes += bytes.length;
                    }
                    if (!stsOpt('sts-auto-play')) return;
                    if (!this.rtPlayer) {
                        this.rtPlayer = new RealTimeAudioPlayer(this.rateOut || 24000);
                        currentRtPlayer = this.rtPlayer;
                    }
                    setStsStatus('再生中...', true);
                    this.rtPlayer.addChunk(data);
                }

                _directClosed(ev) {
                    const ready = this.directReady;
                    this.directReady = null;
                    if (ready) ready(new Error('接続が終了しました'));
                    this.ws = null;
                    if (!this.direct || !this.active || this.saved || this.stopping) return;
                    if (this.resumeHandle && (!this.directExpiresAt || Date.now() < this.directExpiresAt) && this.directResumes < 3) {
                        this.directResumes += 1;
                        setStsStatus('再接続中...', true);
                        this._connectDirect().catch(() => {
                            this.streamError = 'Gemini Live の再接続に失敗しました';
                            this.stop();
                        });
                        return;
                    }
                    if (ev && ev.code && ev.code !== 1000) {
                        this.streamError = 'Gemini Live の接続が終了しました' + (ev.reason ? ': ' + ev.reason : ' (code ' + ev.code + ')');
                    }
                    this.stop();
                }

