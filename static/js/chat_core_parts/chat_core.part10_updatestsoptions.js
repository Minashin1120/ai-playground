        function updateStsOptions() {
            if (!isStsModel()) return;
            const model = get('model-select').value || '';
            const provider = getStsProvider(model);
            const voiceSel = get('sts-voice');
            const speedWrap = get('sts-speed-wrap');
            const speedInput = get('sts-speed');
            const speedLabel = get('sts-speed-label');
            const rateWrap = get('sts-rate-wrap');
            const rateIn = get('sts-rate-in');
            const rateOut = get('sts-rate-out');
            const thinkingWrap = get('sts-thinking-wrap');
            const note = get('sts-note');
            const voiceWrap = get('sts-voice-wrap');
            const autoPlayWrap = get('sts-auto-play-wrap');
            const modeLabel = get('sts-mode-label');
            const transcription = isTranscriptionModel() || isGeminiLiveTranscribeModel() || isXaiLiveTranscribeModel();
            const langWrap = get('sts-lang-wrap');
            const reasoningWrap = get('sts-reasoning-wrap');
            if (reasoningWrap) reasoningWrap.classList.toggle('hidden', !OPENAI_REASONING_STS_MODELS.has(model));

            if (transcription) {
                if (modeLabel) modeLabel.textContent = 'Realtime Speech-to-Text';
                if (voiceWrap) voiceWrap.classList.add('hidden');
                if (autoPlayWrap) autoPlayWrap.classList.add('hidden');
                if (speedWrap) speedWrap.classList.add('hidden');
                if (rateWrap) rateWrap.classList.add('hidden');
                if (thinkingWrap) thinkingWrap.classList.add('hidden');
                if (langWrap) langWrap.classList.add('hidden');
                const transcribeWrap = get('sts-transcribe-wrap');
                const customVocabWrap = get('sts-custom-vocab-wrap');
                if (transcribeWrap) transcribeWrap.classList.toggle('hidden', !isGeminiLiveTranscribeModel());
                if (customVocabWrap) customVocabWrap.classList.toggle('hidden', !isGeminiLiveTranscribeModel() && !isXaiLiveTranscribeModel());
                if (note) {
                    note.textContent = isGeminiLiveTranscribeModel()
                        ? 'リアルタイム低遅延文字起こし（16kHz PCM / 最大10分）'
                        : isXaiLiveTranscribeModel()
                            ? 'xAI ストリーミング文字起こし（16kHz PCM）'
                        : model === 'grok-voice-transcribe-2.0-file'
                            ? 'xAI 音声ファイル文字起こし（録音単位）'
                        : model === 'gpt-live-transcribe'
                            ? '低遅延ライブ文字起こし（24kHz PCM）'
                        : model === 'gpt-realtime-whisper'
                            ? 'ストリーミング音声認識モデルによる文字起こし（24kHz PCM）'
                            : '高精度なコミット単位の文字起こし（24kHz PCM）';
                }
            } else if (provider === 'openai') {
                if (modeLabel) modeLabel.textContent = 'Speech-to-Speech Live';
                if (voiceWrap) voiceWrap.classList.remove('hidden');
                if (autoPlayWrap) autoPlayWrap.classList.remove('hidden');
                setSelectOptions(voiceSel, OPENAI_STS_VOICES, voiceSel.value || 'alloy');
                if (speedWrap) speedWrap.classList.remove('hidden');
                if (speedInput) {
                    speedInput.min = 0.25; speedInput.max = 1.5; speedInput.step = 0.05;
                    if (!speedInput.value) speedInput.value = 1;
                    if (Number(speedInput.value) < 0.25) speedInput.value = 0.25;
                    if (Number(speedInput.value) > 1.5) speedInput.value = 1.5;
                }
                if (rateWrap) rateWrap.classList.add('hidden');
                if (thinkingWrap) thinkingWrap.classList.add('hidden');
                if (langWrap) langWrap.classList.add('hidden');
                if (note) note.textContent = 'OpenAI Realtimeは24kHz PCM固定';
                if (OPENAI_REASONING_STS_MODELS.has(model) && note) {
                    note.textContent = 'OpenAI Realtimeは24kHz PCM固定（Reasoningで推論の強さを指定）';
                }
                if (OPENAI_LIVE_STS_MODELS.has(model)) {
                    setSelectOptions(voiceSel, OPENAI_LIVE_VOICES, OPENAI_LIVE_VOICES.includes(voiceSel.value) ? voiceSel.value : 'marin');
                    if (speedWrap) speedWrap.classList.add('hidden');
                    if (note) note.textContent = 'GPT-Liveは全二重の音声会話（24kHz PCM・速度変更非対応・推論はgpt-5.6-lunaに委任）';
                }
                if (model === 'gpt-realtime-translate') {
                    if (modeLabel) modeLabel.textContent = 'Realtime Translation';
                    if (voiceWrap) voiceWrap.classList.add('hidden');
                    if (speedWrap) speedWrap.classList.add('hidden');
                    if (langWrap) langWrap.classList.remove('hidden');
                    if (note) note.textContent = '話した内容を選択した言語へリアルタイムで音声翻訳（24kHz PCM・音声選択不可）';
                }
            } else if (provider === 'xai') {
                if (modeLabel) modeLabel.textContent = 'Speech-to-Speech Live';
                if (voiceWrap) voiceWrap.classList.remove('hidden');
                if (autoPlayWrap) autoPlayWrap.classList.remove('hidden');
                setSelectOptions(voiceSel, GROK_STS_VOICES, voiceSel.value || 'Ara');
                if (speedWrap) speedWrap.classList.add('hidden');
                if (rateWrap) rateWrap.classList.remove('hidden');
                if (thinkingWrap) thinkingWrap.classList.add('hidden');
                if (langWrap) langWrap.classList.add('hidden');
                setSelectOptions(rateIn, GROK_PCM_RATES, Number(rateIn.value || 24000));
                setSelectOptions(rateOut, GROK_PCM_RATES, Number(rateOut.value || 24000));
                if (note) note.textContent = 'xAIはPCMサンプルレート変更可';
            } else if (provider === 'gemini') {
                if (modeLabel) modeLabel.textContent = 'Speech-to-Speech Live';
                if (voiceWrap) voiceWrap.classList.remove('hidden');
                if (autoPlayWrap) autoPlayWrap.classList.remove('hidden');
                setSelectOptions(voiceSel, GEMINI_STS_VOICES, voiceSel.value || 'Kore');
                if (speedWrap) speedWrap.classList.add('hidden');
                if (rateWrap) rateWrap.classList.add('hidden');
                if (thinkingWrap) thinkingWrap.classList.remove('hidden');
                if (langWrap) langWrap.classList.add('hidden');
                if (note) note.textContent = 'Gemini Liveは音声速度変更非対応';
                const thinkingLevelSelect = get('sts-thinking-level');
                Array.from((thinkingLevelSelect && thinkingLevelSelect.options) || []).forEach(opt => { opt.disabled = false; });
                if (model === 'gemini-2.5-flash-native-audio-preview-12-2025') {
                    if (thinkingWrap) thinkingWrap.classList.add('hidden');
                } else if (model === 'gemini-3.8-live') {
                    if (thinkingWrap) thinkingWrap.classList.add('hidden');
                    if (note) note.textContent = 'Gemini 3.8 Flash Liveは固定レイテンシのLive APIモデル（Thinking level非対応）';
                } else if (model === 'gemini-3.8-live-extended-thinking') {
                    if (note) note.textContent = 'Gemini 3.8 Live Extended Thinkingはlow / medium / highのバックグラウンド推論に対応';
                    const thinkingSelect = get('sts-thinking-level');
                    Array.from((thinkingSelect && thinkingSelect.options) || []).forEach(opt => { opt.disabled = opt.value === 'minimal'; });
                    if (thinkingSelect && !['low', 'medium', 'high'].includes(thinkingSelect.value)) thinkingSelect.value = 'medium';
                }
                if (model === 'gemini-3.5-live-translate-preview') {
                    if (modeLabel) modeLabel.textContent = 'Realtime Translation';
                    if (thinkingWrap) thinkingWrap.classList.add('hidden');
                    if (voiceWrap) voiceWrap.classList.add('hidden');
                    if (langWrap) langWrap.classList.remove('hidden');
                    if (note) note.textContent = '70以上の言語に対応するリアルタイム音声翻訳（Think非対応・音声選択不可）';
                }
            }
            if (speedWrap && speedLabel && speedInput && !speedWrap.classList.contains('hidden')) {
                speedLabel.textContent = `${Number(speedInput.value || 1).toFixed(2)}x`;
            }
        }
        function stsOpt(id) {
            const el = get(id);
            if (id === 'sts-auto-play' || id === 'sts-auto-restart') {
                return el ? !!el.checked : true;
            }
            return el ? !!el.checked : false;
        }
        function getStsSilenceMs() {
            const el = get('sts-silence-sec');
            let v = el ? parseFloat(el.value) : 1.5;
            if (isNaN(v) || v < 0.5) v = 0.5;
            if (v > 10) v = 10;
            return Math.round(v * 1000);
        }
        function getTtsProvider(model) {
            if (!model) return null;
            const m = model.toLowerCase();
            if (m.includes('google-tts')) return 'google';
            if (m.includes('gemini') && m.includes('tts')) return 'gemini';
            if (m.includes('grok-tts') || m.includes('xai-tts')) return 'xai';
            if (m.includes('tts')) return 'openai';
            return null;
        }
        function setSelectOptions(el, list, selected) {
            if (!el) return;
            el.innerHTML = '';
            list.forEach(v => {
                const opt = document.createElement('option');
                opt.value = v.value || v;
                opt.textContent = v.label || v;
                if ((v.value || v) === selected) opt.selected = true;
                el.appendChild(opt);
            });
        }
        function updateTtsUi() {
            const model = get('model-select').value || '';
            const provider = getTtsProvider(model);
            const wrap = get('audio-gen-options');
            if (!wrap) return;
            if (!provider) {
                wrap.classList.add('hidden');
                return;
            }
            wrap.classList.remove('hidden');
            const voiceSel = get('tts-voice');
            const voiceCustomWrap = get('tts-voice-custom-wrap');
            const voiceCustom = get('tts-voice-custom');
            const langWrap = get('tts-language-wrap');
            const langInput = get('tts-language');
            const speedWrap = get('tts-speed-wrap');
            const speedInput = get('tts-speed');
            const speedLabel = get('tts-speed-label');
            const speedNote = get('tts-speed-note');
            const styleWrap = get('tts-style-wrap');
            const interactionsTts = provider === 'gemini' && GEMINI_INTERACTIONS_TTS_MODELS.has(model);
            if (styleWrap) styleWrap.classList.toggle('hidden', !interactionsTts);
            if (voiceCustom) {
                voiceCustom.placeholder = interactionsTts ? 'voice_... / voicekey_...' : 'e.g. en-US-Wavenet-D';
            }

            if (provider === 'openai') {
                setSelectOptions(voiceSel, OPENAI_TTS_VOICES, voiceSel.value || 'alloy');
                voiceCustomWrap.classList.add('hidden');
                langWrap.classList.add('hidden');
                if (speedInput) {
                    speedInput.min = 0.25; speedInput.max = 4; speedInput.step = 0.05;
                    if (!speedInput.value) speedInput.value = 1;
                    if (Number(speedInput.value) < 0.25) speedInput.value = 0.25;
                    if (Number(speedInput.value) > 4) speedInput.value = 4;
                    speedInput.disabled = false;
                }
                if (speedNote) speedNote.textContent = '';
            } else if (provider === 'gemini') {
                setSelectOptions(voiceSel, GEMINI_TTS_VOICES, voiceSel.value || 'Kore');
                // Gemini 3.8 TTS also accepts Voice design / Voice replication IDs.
                voiceCustomWrap.classList.toggle('hidden', !interactionsTts);
                if (!interactionsTts && voiceCustom) voiceCustom.value = '';
                langWrap.classList.add('hidden');
                if (speedInput) { speedInput.disabled = true; }
                if (speedNote) speedNote.textContent = interactionsTts
                    ? '(本文はそのまま読み上げ。話し方はStyleで指定・速度はStyleで調整)'
                    : '(Gemini TTSは速度変更非対応)';
            } else if (provider === 'google') {
                setSelectOptions(voiceSel, [
                    { value: 'auto', label: 'Auto (Studio/Neural2)' },
                    { value: 'custom', label: 'Custom Voice Name' }
                ], voiceSel.value || 'auto');
                if (voiceSel.value === 'custom') {
                    voiceCustomWrap.classList.remove('hidden');
                } else {
                    voiceCustomWrap.classList.add('hidden');
                    if (voiceCustom) voiceCustom.value = '';
                }
                langWrap.classList.remove('hidden');
                if (langInput && !langInput.value) langInput.value = 'ja-JP';
                if (speedInput) {
                    speedInput.min = 0.25; speedInput.max = 2; speedInput.step = 0.05;
                    if (!speedInput.value) speedInput.value = 1;
                    if (Number(speedInput.value) < 0.25) speedInput.value = 0.25;
                    if (Number(speedInput.value) > 2) speedInput.value = 2;
                    speedInput.disabled = false;
                }
                if (speedNote) speedNote.textContent = '';
            } else if (provider === 'xai') {
                setSelectOptions(voiceSel, GROK_TTS_VOICES, voiceSel.value || 'Eve');
                voiceCustomWrap.classList.remove('hidden');
                langWrap.classList.remove('hidden');
                if (langInput && !langInput.value) langInput.value = 'ja';
                if (speedInput) {
                    speedInput.min = 0.7; speedInput.max = 1.5; speedInput.step = 0.05;
                    if (!speedInput.value) speedInput.value = 1;
                    if (Number(speedInput.value) < 0.7) speedInput.value = 0.7;
                    if (Number(speedInput.value) > 1.5) speedInput.value = 1.5;
                    speedInput.disabled = false;
                }
                if (speedNote) speedNote.textContent = 'xAI TTS supports speed 0.7–1.5 and speech tags';
            }
            if (speedInput && speedLabel) {
                speedLabel.textContent = `${Number(speedInput.value || 1).toFixed(2)}x`;
            }
        }

        // endregion
/* ================= MCP（外部モデル連携）設定UI ================= */
        // 状態
        let mcpServers = [];
        let mcpLoaded = false;
        let mcpLoadPromise = null;
        let mcpOauthPopups = [];
        const MCP_URLS = {
            servers: () => '/api/mcp/servers',
            server: (id) => `/api/mcp/servers/${encodeURIComponent(id)}`,
            test: (id) => `/api/mcp/servers/${encodeURIComponent(id)}/test`,
            authStart: (id) => `/api/mcp/servers/${encodeURIComponent(id)}/auth/start`,
            authDisconnect: (id) => `/api/mcp/servers/${encodeURIComponent(id)}/auth/disconnect`,
            tools: (id) => `/api/mcp/servers/${encodeURIComponent(id)}/tools`,
            oauthClient: () => '/api/mcp/oauth-client',
            permission: (id, tool) => `/api/mcp/servers/${encodeURIComponent(id)}/tools/${encodeURIComponent(tool)}/permission`
        };
        const mcpGoogleProviderKey = 'google_workspace';

        const mcpEsc = (s) => {
            return String(s == null ? '' : s).replace(/[&<>"']/g, (c) => ({
                '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
            }[c]));
        };
        const mcpStatusMsg = (elId, msg, isError) => {
            const el = get(elId);
            if (!el) return;
            el.textContent = msg || '';
            el.style.color = isError ? '#f87171' : '#9ca3af';
        };

        function mcpAuthStatusLabel(srv) {
            if (srv.auth_type === 'none') return '認証不要';
            if (srv.auth_status === 'connected') return '接続済み';
            if (srv.auth_status === 'expired') return '期限切れ（再認証）';
            if (srv.auth_status === 'needs_auth') return '認証が必要';
            return '未認証';
        }
        function mcpConnectionStateLabel(srv) {
            if (srv.connection_state === 'error') return 'エラー';
            if (srv.connection_state === 'connected') return '接続OK';
            if (srv.connection_state === 'needs_auth') return '認証待ち';
            return '未接続';
        }
        function mcpBadgeClass(kind) {
            if (kind === 'ok' || kind === 'connected') return 'bg-emerald-700/60 text-emerald-100';
            if (kind === 'error' || kind === 'expired') return 'bg-red-700/60 text-red-100';
            if (kind === 'auth') return 'bg-amber-600/50 text-amber-100';
            return 'bg-gray-700 text-gray-300';
        }
        function mcpStateBadge(srv) {
            const label = mcpAuthStatusLabel(srv);
            const cls = (srv.auth_status === 'connected') ? 'ok'
                : (srv.auth_status === 'expired') ? 'expired'
                : (srv.auth_status === 'needs_auth') ? 'auth' : 'neutral';
            return `<span class="text-[9px] font-bold px-2 py-0.5 rounded-full ${mcpBadgeClass(cls)}">${mcpEsc(label)}</span>`;
        }

        function mcpOauthProviderLabel(pk) {
            if (pk === 'google_workspace') return 'Google Workspace';
            return pk || 'OAuth';
        }

        async function loadMcpServers(force) {
            const listEl = get('mcp-server-list');
            if (!listEl) return;
            // Settings can be opened while the initial preload is still running.
            // Share that request so opening the modal never starts a second slow
            // request (and so the prompt-bar state has one source of truth).
            if (mcpLoadPromise) {
                await mcpLoadPromise;
                if (!force) return;
            }
            if (!force && mcpLoaded) { renderMcpServers(); return; }
            mcpStatusMsg('mcp-status-msg', '読み込み中...', false);
            let request;
            request = (async () => {
                try {
                    const res = await apiFetch(MCP_URLS.servers());
                    if (!res.ok) {
                        const d = await res.json().catch(() => ({}));
                        mcpStatusMsg('mcp-status-msg', d.error || 'MCPサーバー一覧の取得に失敗しました', true);
                        return;
                    }
                    const data = await res.json();
                    mcpServers = (data && Array.isArray(data.servers)) ? data.servers : [];
                    mcpLoaded = true;
                    renderMcpServers();
                    applyMcpPromptChipUi();
                } catch (e) {
                    mcpStatusMsg('mcp-status-msg', 'MCPサーバー一覧の取得に失敗しました: ' + (e && e.message ? e.message : e), true);
                } finally {
                    if (mcpLoadPromise === request) mcpLoadPromise = null;
                }
            })();
            mcpLoadPromise = request;
            await request;
        }

        // 有効（enabled）なMCPサーバーが1つ以上あるか（プロンプトバーMCPチップの表示条件）。
        function mcpHasEnabledServer() {
            return (mcpServers || []).some((srv) => !!srv.enabled);
        }

        // プロンプトバーのMCPスイッチが実効ONか（チップ表示中かつチェックON）。
        function isMcpEnabledForSend() {
            const cont = get('mcp-container');
            if (!cont || cont.classList.contains('hidden')) return false;
            const chk = get('enable-mcp');
            return !!chk && chk.checked;
        }

        // このモデルでMCPツールが利用可能か（バックエンドのテキストLLM付与対象と揃える）。
        function mcpModelSupported() {
            try {
                const m = String((get('model-select') && get('model-select').value) || '').toLowerCase();
                if (!m) return false;
                if (m.startsWith('glm-')) return false;
                // Claude / Kimi は isLlmModel に含まれないが、バックエンドではMCPを付与する。
                if (m.includes('claude') || m.startsWith('kimi')) return true;
                if (typeof isLlmModel === 'function' && isLlmModel()) return true;
                return false;
            } catch (e) { return false; }
        }

        // プロンプトバーのMCPチップ表示/非表示を、現在のモデルとMCP利用可否で更新する。
        function applyMcpPromptChipUi() {
            const cont = get('mcp-container');
            if (!cont) return;
            const show = mcpModelSupported() && mcpHasEnabledServer();
            cont.classList.toggle('hidden', !show);
            syncMcpAutoSysRows();
            if (typeof refreshMinimalOptionsIfOpen === 'function') {
                try { refreshMinimalOptionsIfOpen(); } catch (e) {}
            }
        }

        // 「自動注入システムプロンプト（ユーザー単位）」のMCP / Skills行を
        // 読取専用（プロンプトバー連動）に保つ。設定モーダル(set)とスレッド設定(thread)の
        // 両方に反映する。
        function syncMcpAutoSysRows() {
            const skillsOn = get('enable-skills') ? !!get('enable-skills').checked : true;
            ['set', 'thread'].forEach((prefix) => {
                const mcpEl = get(`${prefix}-auto-sys-mcp-enabled`);
                if (mcpEl) {
                    mcpEl.disabled = true;
                    mcpEl.checked = isMcpEnabledForSend();
                }
                const skillsEl = get(`${prefix}-auto-sys-skills-enabled`);
                if (skillsEl) {
                    skillsEl.disabled = true;
                    skillsEl.checked = skillsOn;
                }
            });
        }

        function renderMcpServers() {
            const listEl = get('mcp-server-list');
            const countEl = get('mcp-server-count');
            if (!listEl) return;
            const count = mcpServers.length;
            if (countEl) countEl.textContent = `${count}件`;
            if (!count) {
                listEl.innerHTML = '<div class="text-[11px] text-gray-600 py-2">まだサーバーがありません。上のカスタム追加フォームから登録するか、Google Workspace の認証をしてください。</div>';
                mcpStatusMsg('mcp-status-msg', '');
                return;
            }
            const html = mcpServers.map((srv, idx) => mcpServerCard(srv, idx)).join('');
            listEl.innerHTML = html;
            mcpStatusMsg('mcp-status-msg', '');
        }

        function mcpServerCard(srv, idx) {
            const isPreset = !!srv.is_preset;
            const isOauth = srv.auth_type === 'oauth';
            const isBearer = srv.auth_type === 'bearer';
            const showAuthAction = isOauth || isBearer;
            const needOauthClient = isOauth && !srv.oauth_client_registered;
            const toolCount = Number(srv.tool_count || 0);
            const toolHint = toolCount > 0 ? `${toolCount}ツール` : 'ツール未取得';
            const badge = mcpStateBadge(srv);
            const tag = isPreset
                ? `<span class="text-[9px] font-bold px-1.5 py-0.5 rounded bg-blue-700/50 text-blue-100">プリセット</span>`
                : `<span class="text-[9px] font-bold px-1.5 py-0.5 rounded bg-purple-700/50 text-purple-100">カスタム</span>`;
            const authBlock = mcpAuthBlock(srv);
            const oauthClientBlock = isOauth ? mcpOauthClientBlock(srv) : '';
            return `
<div class="rounded border border-gray-700 bg-gray-950/50 p-3" data-mcp-server="${mcpEsc(srv.slug)}">
    <div class="flex items-center justify-between gap-2 flex-wrap">
        <div class="flex items-center gap-2 min-w-0">
            <i class="fas fa-plug ${srv.enabled ? 'text-cyan-300' : 'text-gray-600'}"></i>
            <div class="min-w-0">
                <span class="text-xs font-bold text-white">${mcpEsc(srv.name)}</span>
                ${tag} ${badge}
            </div>
        </div>
        <div class="flex items-center gap-1 shrink-0">
            ${showAuthAction ? mcpAuthActionButton(srv) : ''}
            ${isBearer && srv.auth_status !== 'connected' ? '' : ''}
            ${!isPreset ? `<button type="button" data-progress-no-spinner="true" class="mcp-mini-btn mcp-danger-btn" data-act="delete" data-id="${srv.id}">削除</button>` : ''}
            <label class="relative inline-flex items-center cursor-pointer ml-1" title="${srv.enabled ? '無効化' : '有効化'}">
                <input type="checkbox" class="sr-only peer mcp-enable-toggle" data-id="${srv.id}" ${srv.enabled ? 'checked' : ''}>
                <div class="w-9 h-5 bg-gray-700 peer-focus:outline-none rounded-full peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-[var(--theme-600)]"></div>
            </label>
        </div>
    </div>
    <div class="text-[10px] text-gray-500 mt-1 break-all">${mcpEsc(srv.url)}</div>
    ${srv.description ? `<div class="text-[10px] text-gray-500 mt-0.5">${mcpEsc(srv.description)}</div>` : ''}
    ${srv.last_error ? `<div class="text-[10px] text-red-400 mt-1">${mcpEsc(srv.last_error)}</div>` : ''}
    <div class="flex items-center justify-between gap-2 mt-2 flex-wrap">
        <div class="text-[10px] text-gray-500 flex items-center gap-2">
            <span class="${toolCount > 0 ? 'text-emerald-300' : 'text-gray-500'}">${toolHint}</span>
            <button type="button" data-progress-no-spinner="true" class="mcp-mini-btn" data-act="tools" data-id="${srv.id}">ツール一覧</button>
        </div>
        <div class="flex items-center gap-1 flex-wrap">
            <button type="button" data-progress-no-spinner="true" class="mcp-mini-btn" data-act="test" data-id="${srv.id}"><i class="fas fa-plug"></i> 接続テスト</button>
            <span class="text-[9px] text-gray-600">${mcpEsc(mcpConnectionStateLabel(srv))}</span>
        </div>
    </div>
    ${toolCount > 0 ? `<div class="hidden mt-2" data-mcp-toolbox="${srv.id}"></div>` : `<div class="hidden mt-2" data-mcp-toolbox="${srv.id}"><div class="text-[10px] text-gray-600">接続テスト後にツール一覧が表示されます。</div></div>`}
    ${oauthClientBlock}
    ${authBlock}
</div>`;
        }

        function mcpAuthActionButton(srv) {
            if (srv.auth_type === 'bearer') {
                return '';
            }
            if (srv.auth_status === 'connected' || srv.auth_status === 'expired') {
                return `<button type="button" data-progress-no-spinner="true" class="mcp-mini-btn mcp-auth-btn" data-act="reconnect" data-id="${srv.id}"><i class="fas fa-sync"></i> 再認証</button>
                        <button type="button" data-progress-no-spinner="true" class="mcp-mini-btn mcp-danger-btn" data-act="disconnect" data-id="${srv.id}"><i class="fas fa-unlink"></i> 接続解除</button>`;
            }
            return `<button type="button" data-progress-no-spinner="true" class="mcp-mini-btn mcp-auth-btn" data-act="auth" data-id="${srv.id}"><i class="fas fa-key"></i> 認証する</button>`;
        }

        // OAuthクライアント情報の編集ブロック（サーバーごと）
        function mcpOauthClientBlock(srv) {
            const pk = srv.oauth_provider_key || srv.slug || '';
            const providerLabel = mcpOauthProviderLabel(pk);
            if (srv.oauth_client_registered) {
                return `
<div class="mt-2 rounded border border-gray-800 bg-black/20 p-2">
    <div class="text-[10px] text-gray-400 flex items-center justify-between">
        <span>OAuthクライアント（${mcpEsc(providerLabel)}）: ${mcpEsc(srv.oauth_client_id_masked || '登録済み')}</span>
        <button type="button" data-progress-no-spinner="true" class="mcp-mini-btn" data-act="edit-oauth" data-id="${srv.id}">変更</button>
    </div>
</div>`;
            }
            return `
<div class="mt-2 rounded border border-amber-700/50 bg-amber-950/20 p-2">
    <div class="text-[10px] text-amber-300 mb-1">${mcpEsc(providerLabel)} の OAuth クライアント情報（Client ID / Secret）が必要です。</div>
    <div class="grid grid-cols-1 md:grid-cols-2 gap-1">
        <input type="text" data-oauth-pk="${mcpEsc(pk)}" data-oauth-role="cid" placeholder="Client ID" autocomplete="off" data-1p-ignore="true" class="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs text-white">
        <input type="password" data-oauth-pk="${mcpEsc(pk)}" data-oauth-role="secret" placeholder="Client Secret" autocomplete="off" data-1p-ignore="true" class="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs text-white">
    </div>
    <div class="flex justify-end mt-1">
        <button type="button" data-progress-no-spinner="true" class="mcp-mini-btn" data-act="save-oauth" data-id="${srv.id}" data-pk="${mcpEsc(pk)}">保存</button>
    </div>
</div>`;
        }

        // 認証ブロック（bearer入力など）
        function mcpAuthBlock(srv) {
            if (srv.auth_type === 'bearer') {
                const hasToken = !!srv.auth_has_token;
                return `
<div class="mt-2 rounded border border-gray-800 bg-black/20 p-2">
    <div class="text-[10px] text-gray-400 mb-1">Bearer トークン ${hasToken ? '<span class="text-emerald-300">（保存済み・********）</span>' : '<span class="text-amber-300">（未設定）</span>'}</div>
    <div class="flex gap-1">
        <input type="password" data-bearer-id="${srv.id}" placeholder="Bearer トークン" autocomplete="off" data-1p-ignore="true" class="flex-1 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs text-white">
        <button type="button" data-progress-no-spinner="true" class="mcp-mini-btn mcp-auth-btn" data-act="save-bearer" data-id="${srv.id}">保存</button>
    </div>
</div>`;
            }
            if (srv.auth_type === 'oauth') {
                const pk = srv.oauth_provider_key || srv.slug || '';
                const needClient = !srv.oauth_client_registered;
                return `
<div class="text-[10px] text-gray-600 mt-1">${needClient ? 'OAuthクライアント情報を保存すると「認証する」が使えます。' : ''}</div>`;
            }
            return '';
        }

        async function mcpToggleEnabled(serverId, enabled) {
            mcpStatusMsg('mcp-status-msg', enabled ? '有効化しています...' : '無効化しています...', false);
            try {
                const res = await apiFetch(MCP_URLS.server(serverId), {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ enabled: enabled })
                });
                if (!res.ok) {
                    const d = await res.json().catch(() => ({}));
                    mcpStatusMsg('mcp-status-msg', d.error || '更新に失敗しました', true);
                    return;
                }
                const data = await res.json();
                mcpStatusMsg('mcp-status-msg', enabled ? '有効にしました。チャットのモデルへツールが公開されます。' : '無効にしました。', false);
                loadMcpServers(true);
            } catch (e) {
                mcpStatusMsg('mcp-status-msg', '更新に失敗しました: ' + (e && e.message ? e.message : e), true);
            }
        }

        async function mcpOpenAuth(serverId) {
            mcpStatusMsg('mcp-status-msg', '認可URLを準備しています...', false);
            try {
                const res = await apiFetch(MCP_URLS.authStart(serverId), {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({})
                });
                if (!res.ok) {
                    const d = await res.json().catch(() => ({}));
                    if (d.requires_oauth_client) {
                        mcpStatusMsg('mcp-status-msg', d.error || 'OAuthクライアント情報を先に登録してください。', true);
                    } else {
                        mcpStatusMsg('mcp-status-msg', d.error || '認可URLの取得に失敗しました', true);
                    }
                    return;
                }
                const data = await res.json();
                if (!data.url) {
                    mcpStatusMsg('mcp-status-msg', '認可URLが返りませんでした', true);
                    return;
                }
                const pop = window.open(data.url, '_blank', 'width=520,height=680');
                if (pop) {
                    mcpOauthPopups.push(pop);
                    mcpStatusMsg('mcp-status-msg', 'Googleの画面で許可してください。完了後このタブに反映されます。', false);
                    const poll = window.setInterval(() => {
                        if (!pop || pop.closed) {
                            window.clearInterval(poll);
                            loadMcpServers(true);
                        }
                    }, 1200);
                } else {
                    mcpStatusMsg('mcp-status-msg', 'ポップアップがブロックされました。', true);
                }
            } catch (e) {
                mcpStatusMsg('mcp-status-msg', '認可URLの取得に失敗しました: ' + (e && e.message ? e.message : e), true);
            }
        }

        async function mcpDisconnect(serverId) {
            if (!window.confirm('このサーバーの認証情報（トークン）を削除して接続を解除しますか？')) return;
            try {
                const res = await apiFetch(MCP_URLS.authDisconnect(serverId), {
                    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}'
                });
                if (!res.ok) {
                    const d = await res.json().catch(() => ({}));
                    mcpStatusMsg('mcp-status-msg', d.error || '接続解除に失敗しました', true);
                    return;
                }
                mcpStatusMsg('mcp-status-msg', '接続を解除しました。', false);
                loadMcpServers(true);
            } catch (e) {
                mcpStatusMsg('mcp-status-msg', '接続解除に失敗しました: ' + (e && e.message ? e.message : e), true);
            }
        }

        async function mcpDeleteServer(serverId) {
            if (!window.confirm('このカスタムMCPサーバーを削除しますか？')) return;
            try {
                const res = await apiFetch(MCP_URLS.server(serverId), { method: 'DELETE' });
                if (!res.ok) {
                    const d = await res.json().catch(() => ({}));
                    mcpStatusMsg('mcp-status-msg', d.error || '削除に失敗しました', true);
                    return;
                }
                mcpStatusMsg('mcp-status-msg', '削除しました。', false);
                loadMcpServers(true);
            } catch (e) {
                mcpStatusMsg('mcp-status-msg', '削除に失敗しました: ' + (e && e.message ? e.message : e), true);
            }
        }

        async function mcpTestServer(serverId, showTools) {
            mcpStatusMsg('mcp-status-msg', '接続テスト中...', false);
            try {
                const res = await apiFetch(MCP_URLS.test(serverId), {
                    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}'
                });
                const d = await res.json().catch(() => ({}));
                if (!res.ok) {
                    mcpStatusMsg('mcp-status-msg', d.error || '接続テストに失敗しました', true);
                    return;
                }
                if (d.probe && d.probe.message) mcpStatusMsg('mcp-status-msg', d.probe.message, !d.probe.ok);
                loadMcpServers(true);
            } catch (e) {
                mcpStatusMsg('mcp-status-msg', '接続テストに失敗しました: ' + (e && e.message ? e.message : e), true);
            }
        }

        async function mcpLoadTools(serverId) {
            const box = document.querySelector(`[data-mcp-toolbox="${serverId}"]`);
            if (!box) return;
            box.classList.remove('hidden');
            box.innerHTML = '<div class="text-[10px] text-gray-500">読み込み中...</div>';
            try {
                const res = await apiFetch(MCP_URLS.tools(serverId));
                const d = await res.json().catch(() => ({}));
                if (!res.ok) {
                    box.innerHTML = `<div class="text-[10px] text-red-400">${mcpEsc(d.error || '取得に失敗しました')}</div>`;
                    return;
                }
                const tools = (d && Array.isArray(d.tools)) ? d.tools : [];
                if (!tools.length) {
                    box.innerHTML = '<div class="text-[10px] text-gray-600">ツール一覧がありません。「接続テスト」で取得してください。</div>';
                    return;
                }
                const rows = tools.map((t, i) => `
<div class="flex items-start justify-between gap-2 py-1 border-b border-gray-800 last:border-0">
    <div class="min-w-0">
        <div class="text-[11px] text-cyan-200 font-mono">${mcpEsc(t.name)}</div>
        <div class="text-[10px] text-gray-500 line-clamp-2">${mcpEsc(t.description || '')}</div>
    </div>
    <span class="text-[9px] shrink-0 px-1.5 py-0.5 rounded ${t.read_only ? 'bg-emerald-800/40 text-emerald-200' : 'bg-amber-800/40 text-amber-200'}">${t.read_only ? '読み取り' : '変更'}</span>
</div>`).join('');
                box.innerHTML = `<div class="rounded border border-gray-800 bg-black/20 p-2">${rows}</div>`;
            } catch (e) {
                box.innerHTML = '<div class="text-[10px] text-red-400">取得に失敗しました</div>';
            }
        }

        async function mcpSaveOauthClient(providerKey, clientId, clientSecret, fromCardId) {
            mcpStatusMsg('mcp-status-msg', '保存しています...', false);
            const body = { provider_key: providerKey, client_id: clientId, client_secret: clientSecret };
            try {
                const res = await apiFetch(MCP_URLS.oauthClient(), {
                    method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body)
                });
                const d = await res.json().catch(() => ({}));
                if (!res.ok) {
                    mcpStatusMsg('mcp-status-msg', d.error || '保存に失敗しました', true);
                    return;
                }
                mcpStatusMsg('mcp-status-msg', 'OAuthクライアント情報を保存しました。', false);
                loadMcpServers(true);
            } catch (e) {
                mcpStatusMsg('mcp-status-msg', '保存に失敗しました: ' + (e && e.message ? e.message : e), true);
            }
        }

        async function mcpAddCustomServer() {
            const nameEl = get('mcp-custom-name');
            const urlEl = get('mcp-custom-url');
            const authEl = get('mcp-custom-auth');
            const descEl = get('mcp-custom-desc');
            const bearerEl = get('mcp-custom-bearer');
            const statusEl = get('mcp-custom-status');
            const btnEl = get('mcp-add-server-btn');
            if (!nameEl || !urlEl || !authEl) return;
            const name = (nameEl.value || '').trim();
            const url = (urlEl.value || '').trim();
            const auth = authEl.value || 'none';
            const desc = descEl ? (descEl.value || '').trim() : '';
            const bearer = (bearerEl && bearerEl.value || '').trim();
            if (!name || !url) {
                if (statusEl) { statusEl.textContent = '表示名とURLは必須です'; statusEl.style.color = '#f87171'; }
                return;
            }
            if (btnEl) btnEl.disabled = true;
            if (statusEl) { statusEl.textContent = '接続テスト中...'; statusEl.style.color = '#9ca3af'; }
            const body = { name, url, auth_type: auth, description: desc };
            if (auth === 'bearer' && bearer) body.bearer_token = bearer;
            try {
                const res = await apiFetch(MCP_URLS.servers(), {
                    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body)
                });
                const d = await res.json().catch(() => ({}));
                if (!res.ok) {
                    if (statusEl) { statusEl.textContent = d.error || '追加に失敗しました'; statusEl.style.color = '#f87171'; }
                    return;
                }
                if (statusEl) { statusEl.textContent = (d.probe && d.probe.message) || '追加しました'; statusEl.style.color = (d.probe && d.probe.ok) ? '#34d399' : '#fbbf24'; }
                nameEl.value = '';
                urlEl.value = '';
                if (descEl) descEl.value = '';
                if (bearerEl) bearerEl.value = '';
                mcpLoaded = false;
                loadMcpServers(true);
            } catch (e) {
                if (statusEl) { statusEl.textContent = '追加に失敗しました: ' + (e && e.message ? e.message : e); statusEl.style.color = '#f87171'; }
            } finally {
                if (btnEl) btnEl.disabled = false;
            }
        }

        function bindMcpSettingsUi() {
            const listEl = get('mcp-server-list');
            if (!listEl) return;
            // 追加ボタン
            const addBtn = get('mcp-add-server-btn');
            if (addBtn) addBtn.addEventListener('click', mcpAddCustomServer);
            const authSel = get('mcp-custom-auth');
            const bearerWrap = get('mcp-custom-bearer-wrap');
            if (authSel && bearerWrap) {
                const syncBearer = () => {
                    bearerWrap.classList.toggle('hidden', authSel.value !== 'bearer');
                };
                authSel.addEventListener('change', syncBearer);
                syncBearer();
            }
            // Google連携クライアント情報（上部カード）
            const saveGoogleBtn = get('mcp-save-google-client-btn');
            if (saveGoogleBtn) {
                saveGoogleBtn.addEventListener('click', async () => {
                    const cidEl = get('mcp-google-client-id');
                    const secEl = get('mcp-google-client-secret');
                    const stateEl = get('mcp-google-client-state');
                    const cid = cidEl ? cidEl.value : '';
                    const sec = secEl ? secEl.value : '';
                    if (!cid && !sec) {
                        if (stateEl) { stateEl.textContent = 'Client ID を入力してください'; stateEl.style.color = '#f87171'; }
                        return;
                    }
                    await mcpSaveOauthClient(mcpGoogleProviderKey, cid || '********', sec || '********', null);
                });
            }
            // 一覧内の操作（イベント委譲）
            listEl.addEventListener('click', async (ev) => {
                const btn = ev.target.closest('[data-act]');
                if (!btn) return;
                const act = btn.getAttribute('data-act');
                const id = btn.getAttribute('data-id');
                if (act === 'test') { ev.preventDefault(); mcpTestServer(id); return; }
                if (act === 'tools') { ev.preventDefault(); mcpLoadTools(id); return; }
                if (act === 'auth' || act === 'reconnect') { ev.preventDefault(); mcpOpenAuth(id); return; }
                if (act === 'disconnect') { ev.preventDefault(); mcpDisconnect(id); return; }
                if (act === 'delete') { ev.preventDefault(); mcpDeleteServer(id); return; }
                if (act === 'edit-oauth') {
                    ev.preventDefault();
                    const card = btn.closest('[data-mcp-server]');
                    if (card) {
                        const pk = btn.getAttribute('data-oauth-pk') || '';
                        const srv = mcpServers.find(s => String(s.id) === String(id));
                        const wrap = document.createElement('div');
                        wrap.className = 'mt-2 rounded border border-amber-700/50 bg-amber-950/20 p-2';
                        wrap.innerHTML = `
    <div class="grid grid-cols-1 md:grid-cols-2 gap-1">
        <input type="text" placeholder="Client ID" autocomplete="off" data-1p-ignore="true" class="mcp-oauth-edit-cid w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs text-white" value="">
        <input type="password" placeholder="Client Secret" autocomplete="off" data-1p-ignore="true" class="mcp-oauth-edit-sec w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs text-white">
    </div>
    <div class="flex justify-end mt-1 gap-1">
        <button type="button" data-progress-no-spinner="true" class="mcp-mini-btn" data-act="save-oauth" data-id="${id}" data-pk="${mcpEsc((srv && (srv.oauth_provider_key || srv.slug)) || '')}">保存</button>
    </div>`;
                        const prev = btn.closest('div');
                        prev.parentNode.insertBefore(wrap, prev.nextSibling);
                        btn.remove();
                    }
                    return;
                }
                if (act === 'save-oauth') {
                    ev.preventDefault();
                    const pk = btn.getAttribute('data-pk') || '';
                    // 上部カード内の入力 or カード内入力 or インライン編集
                    const scope = btn.closest('[data-mcp-server]') || document;
                    const cidInputs = scope.querySelectorAll('[data-oauth-role="cid"], .mcp-oauth-edit-cid');
                    const secInputs = scope.querySelectorAll('[data-oauth-role="secret"], .mcp-oauth-edit-sec');
                    const cid = cidInputs.length ? cidInputs[cidInputs.length - 1].value : '';
                    const sec = secInputs.length ? secInputs[secInputs.length - 1].value : '';
                    if (!cid && !sec) { mcpStatusMsg('mcp-status-msg', 'Client ID を入力してください', true); return; }
                    mcpSaveOauthClient(pk, cid || '********', sec || '********', id);
                    return;
                }
                if (act === 'save-bearer') {
                    ev.preventDefault();
                    const input = document.querySelector(`[data-bearer-id="${id}"]`);
                    const token = input ? input.value : '';
                    if (!token || token.trim() === '') { mcpStatusMsg('mcp-status-msg', 'Bearerトークンを入力してください', true); return; }
                    mcpStatusMsg('mcp-status-msg', '保存しています...', false);
                    try {
                        const res = await apiFetch(MCP_URLS.server(id), {
                            method: 'PUT', headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ bearer_token: token })
                        });
                        const d = await res.json().catch(() => ({}));
                        if (!res.ok) { mcpStatusMsg('mcp-status-msg', d.error || '保存に失敗しました', true); return; }
                        mcpStatusMsg('mcp-status-msg', 'Bearerトークンを保存しました。', false);
                        loadMcpServers(true);
                    } catch (e) {
                        mcpStatusMsg('mcp-status-msg', '保存に失敗しました', true);
                    }
                    return;
                }
            });
            // 有効トグル
            listEl.addEventListener('change', (ev) => {
                const t = ev.target.closest('.mcp-enable-toggle');
                if (!t) return;
                mcpToggleEnabled(t.getAttribute('data-id'), t.checked);
            });
        }

        const initMcpUi = () => {
            try { bindMcpSettingsUi(); } catch (e) {}
            // Load MCP metadata as part of chat initialization.  The settings
            // modal is hidden by default, so waiting for it to open made the
            // prompt-bar MCP switch appear late or not at all.
            try { loadMcpServers(); } catch (e) {}
        };
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', initMcpUi, { once: true });
        } else {
            initMcpUi();
        }

        function bindMcpPromptToggle() {
            const chk = get('enable-mcp');
            const skillsChk = get('enable-skills');
            const refresh = () => {
                syncMcpAutoSysRows();
                if (typeof refreshMinimalOptionsIfOpen === 'function') {
                    try { refreshMinimalOptionsIfOpen(); } catch (e) {}
                }
            };
            if (chk) chk.addEventListener('change', refresh);
            if (skillsChk) skillsChk.addEventListener('change', refresh);
        }
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => { try { bindMcpPromptToggle(); } catch (e) {} });
        } else {
            try { bindMcpPromptToggle(); } catch (e) {}
        }

        function getModelTags(m, group) {
            const tags = [];
            const id = (m.id || '').toLowerCase();
            const name = (m.name || '').toLowerCase();
            const desc = (m.desc || '').toLowerCase();
            const cat = (group.category || '').toLowerCase();
            if (
                cat.includes('gemini') ||
                id.includes('gemini') ||
                name.includes('gemini') ||
                desc.includes('gemini') ||
                cat.includes('banana') ||
                name.includes('banana')
            ) tags.push('gemini');
            if (
                cat.includes('deepseek') ||
                id.includes('deepseek') ||
                name.includes('deepseek') ||
                desc.includes('deepseek')
            ) tags.push('deepseek');
            if (id.startsWith('glm-') || cat.includes('z.ai')) tags.push('zai');
            if (
                cat.includes('mistral') ||
                id.includes('mistral') ||
                name.includes('mistral') ||
                desc.includes('mistral') ||
                id.includes('ocr') ||
                cat.includes('ocr')
            ) tags.push('mistral');
            if (cat.includes('ideogram') || id.includes('ideogram') || name.includes('ideogram')) tags.push('ideogram');
            if (
                cat.includes('gpt') ||
                cat.includes('openai') ||
                id.includes('gpt') ||
                name.includes('gpt') ||
                desc.includes('openai')
            ) tags.push('openai');
            if (
                cat.includes('xai') ||
                cat.includes('grok') ||
                id.includes('grok') ||
                name.includes('grok') ||
                desc.includes('xai')
            ) tags.push('xai');
            if (cat.includes('image') || id.includes('image') || name.includes('image') || desc.includes('image')) tags.push('image');
            if (
                cat.includes('audio') ||
                cat.includes('music') ||
                cat.includes('transcription') ||
                cat.includes('speech') ||
                id.includes('tts') ||
                id.includes('transcri') ||
                name.includes('tts') ||
                name.includes('transcri') ||
                name.includes('voice') ||
                desc.includes('tts') ||
                id.includes('realtime') ||
                id.includes('live') ||
                id.includes('voice-agent') ||
                id.includes('native-audio') ||
                name.includes('audio') ||
                desc.includes('audio') ||
                desc.includes('speech-to-text')
            ) tags.push('audio');
            if (id.includes('reasoning') || name.includes('reasoning') || desc.includes('reasoning')) tags.push('reasoning');
            if ((cat.includes('deepseek') || id.includes('deepseek') || name.includes('deepseek')) && !tags.includes('reasoning')) tags.push('reasoning');
            if (id.includes('fast') || name.includes('fast') || desc.includes('fast') || cat.includes('fast')) tags.push('fast');
            if ((id.includes('deepseek-v4-flash') || (cat.includes('deepseek') && name.includes('flash'))) && !tags.includes('fast')) tags.push('fast');
            if (cat.includes('anthropic') || id.includes('claude') || name.includes('claude') || desc.includes('anthropic')) tags.push('anthropic');
            if (cat.includes('kimi') || id.includes('kimi') || name.includes('kimi') || desc.includes('moonshot')) tags.push('kimi');
            if (
                cat.includes('video') ||
                id.includes('video') ||
                id.startsWith('veo-') ||
                id.includes('omni-') ||
                name.includes('video') ||
                desc.includes('video')
            ) tags.push('video');
            if (
                cat.includes('music') ||
                id.startsWith('lyria-') ||
                name.includes('music') ||
                desc.includes('music') ||
                desc.includes('song')
            ) tags.push('music');
            if (
                cat.includes('transcription') ||
                id.includes('transcri') ||
                name.includes('transcri') ||
                desc.includes('transcription') ||
                desc.includes('speech-to-text')
            ) tags.push('transcription');
            if (cat.includes('ocr') || id.includes('ocr') || name.includes('ocr') || desc.includes('ocr')) tags.push('ocr');
            if (
                cat.includes('agent') ||
                id.includes('agent') ||
                name.includes('agent') ||
                desc.includes('agentic') ||
                desc.includes('computer use') ||
                desc.includes('deep research')
            ) tags.push('agent');
            if (m.agenticView) tags.push('agentic view');
            return tags;
        }

        function updateModelTagUi() {
            const bar = get('model-tag-bar');
            if (!bar) return;
            const btns = bar.querySelectorAll('.model-tag-btn');
            btns.forEach(b => {
                const t = b.innerText.trim().toLowerCase();
                const active = (t === 'all' ? 'all' : t) === activeModelTag;
                b.classList.toggle('is-active', active);
            });
        }

