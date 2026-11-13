
        function selectSlashCommand(cmdId) {
            const input = get('prompt-input');
            if (!input) return;

            const val = input.value;
            const token = extractSlashCommandToken(val);
            if (token !== null) {
                // Input begins with a slash command token, possibly followed
                // by argument text typed without a space. Strip only the
                // command part and keep the rest as the instruction.
                const trimmed = String(val || '').trimStart();
                input.value = trimmed.substring(1 + token.length).trimStart();
            } else {
                // Find the last '/' before cursor or end, and remove from there to the command
                const lastSlash = val.lastIndexOf('/');
                if (lastSlash !== -1) {
                    // Keep everything before the command start
                    input.value = val.substring(0, lastSlash).trimEnd();
                } else {
                    input.value = '';
                }
            }

            hideSlashCommandSuggestions();
            const command = SLASH_COMMANDS.find((candidate) => candidate.id === cmdId);
            const argument = input.value.trim();
            if (command && command.autocompleteArgument && !argument) {
                input.value = `${command.label} `;
                slashSelectedIndex = 0;
                lastSlashFilter = null;
                input.dispatchEvent(new Event('input', { bubbles: true }));
                input.focus();
                return;
            }
            if (command && command.kind === 'minimal' && (!command.requiresArgument || argument)) {
                input.value = '';
                executeMinimalSlashCommand(cmdId, argument);
                input.dispatchEvent(new Event('input', { bubbles: true }));
                input.focus();
                return;
            }
            pendingSlashCommand = cmdId;

            // Show visual feedback that we are now in command argument mode
            showPendingSlashCommandIndicator(cmdId);

            // Optional: give a small hint in placeholder or just let user start typing the instruction
            input.focus();
            // Dispatch input to update any token estimate etc.
            input.dispatchEvent(new Event('input', { bubbles: true }));
        }

        const AI_SETTING_JUMP_TARGETS = {
            default_model: { label: '既定のモデル', tab: 'general', control: 'set-default-model' },
            default_vision_model: { label: 'Vision Model', tab: 'general', control: 'set-default-vision-model' },
            use_last_chat_settings: { label: '前回の設定を継続', tab: 'general', control: 'set-use-last-settings' },
            default_enable_search: { label: '既定のSearch', tab: 'general', control: 'set-default-search' },
            default_gemini_free_tier: { label: '既定の無料枠', tab: 'general', control: 'set-default-gemini-free-tier' },
            default_enable_url_context: { label: '既定のURLs', tab: 'general', control: 'set-default-url-context' },
            default_enable_maps: { label: '既定のMaps', tab: 'general', control: 'set-default-maps' },
            default_enable_python: { label: '既定のPython', tab: 'general', control: 'set-default-python' },
            default_enable_file_creation: { label: '既定のFile', tab: 'general', control: 'set-default-file-creation' },
            default_enable_thinking: { label: '既定のThinking', tab: 'general', control: 'set-default-thinking' },
            default_thinking_level: { label: 'Thinking Level', tab: 'general', control: 'set-default-thinking-level' },
            default_thinking_budget: { label: 'Thinking Budget', tab: 'general', control: 'set-default-thinking-budget' },
            default_reasoning_effort: { label: 'Reasoning Effort', tab: 'general', control: 'set-default-reasoning-effort' },
            default_enable_system_prompt: { label: '既定のSysPrompt', tab: 'general', control: 'set-default-sys-prompt' },
            default_enable_mcp: { label: '既定のMCP', tab: 'general', control: 'set-default-mcp' },
            default_safety_setting: { label: '既定のSafety', tab: 'general', control: 'set-default-safety' },
            auto_search_on_links: { label: 'Xリンクの自動検索', tab: 'general', control: 'set-auto-search-links' },
            mic_transcribe_mode: { label: 'マイク文字起こし方式', tab: 'general', control: 'set-mic-transcribe-mode' },
            stt_model: { label: 'STTモデル', tab: 'general', control: 'set-stt-model' },
            llm_transcribe_prompt: { label: 'LLM文字起こしプロンプト', tab: 'general', control: 'set-llm-transcribe-prompt' },
            enter_to_send: { label: 'Enterで送信', tab: 'general', control: 'set-enter-to-send' },
            compact_prompt_mode: { label: 'プロンプトバー表示', tab: 'general', control: 'set-compact-prompt-mode' },
            minimal_prompt_mode: { label: 'ミニマル表示', tab: 'general', control: 'set-minimal-prompt-mode' },
            temp_chat_timeout_seconds: { label: '一時チャット保持時間', tab: 'general', control: 'set-temp-chat-timeout-seconds' },
            system_prompt: { label: 'ユーザーシステムプロンプト', tab: 'prompt', control: 'sys-prompt-text' },
            system_prompt_enabled: { label: 'システムプロンプト', tab: 'prompt', control: 'set-global-sys-prompt-enabled' },
            apply_global_system_prompt: { label: 'ユーザープロンプトの適用', tab: 'prompt', control: 'set-apply-global-sys-prompt' },
            apply_auto_system_prompt_notices: { label: '自動注入プロンプト', tab: 'prompt', control: 'set-apply-auto-sys-prompt-notices' },
            auto_system_prompt_notices_config: { label: '自動注入プロンプト設定', tab: 'prompt', control: 'auto-sys-prompt-settings' },
            theme_color: { label: 'テーマカラー', tab: 'display', control: 'set-theme-color' },
            liquid_glass_enabled: { label: 'Liquid Glass', tab: 'display', control: 'set-liquid-glass' },
            use_sw_cache: { label: '高速キャッシュ', tab: 'data', control: 'set-use-sw-cache' },
            enable_latency_metrics: { label: 'レスポンス速度の計測', tab: 'data', control: 'set-latency-metrics' },
            enable_client_debug_log: { label: 'デバッグログの拡張送信', tab: 'data', control: 'set-client-debug-log' },
            bot_detection_enabled: { label: 'Bot Detection', tab: 'security', control: 'set-bot-detect' },
            skip_2fa_on_google_login: { label: 'Googleログイン時の2FA', tab: '2fa', control: 'set-skip-2fa-google' },
            default_2fa_method: { label: '既定の2FA方式', tab: '2fa', control: 'set-default-2fa-method' },
            rich_paste_prompt_default: { label: 'リッチ貼り付けプロンプト', modal: 'rich-paste', control: 'rich-paste-prompt' },
            rich_paste_prompt_use_custom_default: { label: 'リッチ貼り付けの既定値', modal: 'rich-paste', control: 'rich-paste-use-default' }
        };

        function formatAiSettingValue(value) {
            if (value === true) return 'ON';
            if (value === false) return 'OFF';
            if (value === '(更新)') return '更新済み';
            if (value === null || value === undefined || value === '') return '未設定';
            if (typeof value === 'object') {
                try { return JSON.stringify(value); } catch (e) { return '更新済み'; }
            }
            return String(value);
        }

        function findSettingsJumpElement(tabId, controlId) {
            const tab = get(`tab-${tabId}`);
            let element = get(controlId);
            if (!tab || !element) return null;
            while (element.parentElement && element.parentElement !== tab) element = element.parentElement;
            return element.parentElement === tab ? element : get(controlId);
        }

        function openAiSettingJumpTarget(key) {
            const target = AI_SETTING_JUMP_TARGETS[key];
            if (!target) {
                if (typeof window.openSettingsModal === 'function') window.openSettingsModal();
                return;
            }
            if (target.modal === 'rich-paste') {
                openRichPasteModal();
                setTimeout(() => {
                    const control = get(target.control);
                    if (control) {
                        control.scrollIntoView({ behavior: 'smooth', block: 'center' });
                        control.focus({ preventScroll: true });
                    }
                }, 260);
                return;
            }
            if (typeof window.openSettingsModal === 'function') window.openSettingsModal();
            setTimeout(() => {
                const element = findSettingsJumpElement(target.tab, target.control);
                if (element) jumpToSetting(target.tab, element);
                else switchTab(target.tab || 'general');
            }, 320);
        }

        function removeEphemeralMessageControls(messageEl) {
            if (!messageEl) return;
            const controls = messageEl.querySelector('.msg-controls');
            if (controls) controls.remove();
        }

        function renderAiSettingsResultBubble(values, modelId, mode = 'update') {
            const entries = Object.entries(values || {});
            const id = `settings-result-${Date.now()}`;
            const inspecting = mode === 'inspect';
            const message = entries.length
                ? (inspecting
                    ? '現在の設定を確認しました。\n\n確認した項目をタップすると、設定画面の該当箇所へ移動できます。'
                    : '設定を更新しました。\n\n変更した項目をタップすると、設定画面の該当箇所へ移動できます。')
                : (inspecting ? '確認できる設定項目がありませんでした。' : '変更された設定項目はありませんでした。');
            const messageEl = renderMessage(id, 'assistant', message, null, null, modelId, null, true, null, null, null, null, null, null, null, null, true);
            if (!messageEl) return;
            removeEphemeralMessageControls(messageEl);
            const bubble = messageEl.querySelector('.message-bubble');
            if (!bubble || !entries.length) return;

            const list = document.createElement('div');
            list.className = 'mt-3 space-y-2 ai-settings-result-list';
            entries.forEach(([key, value]) => {
                const target = AI_SETTING_JUMP_TARGETS[key] || { label: key };
                const button = document.createElement('button');
                button.type = 'button';
                button.className = 'w-full flex items-center gap-3 rounded-xl border border-white/10 bg-black/20 px-3 py-2.5 text-left hover:bg-black/30 hover:border-blue-400/40 transition ai-settings-result-item';
                const text = document.createElement('span');
                text.className = 'min-w-0 flex-1';
                const label = document.createElement('span');
                label.className = 'block text-xs font-bold text-blue-200';
                label.textContent = target.label;
                const valueEl = document.createElement('span');
                valueEl.className = 'block mt-0.5 text-[11px] text-gray-300 break-words';
                valueEl.textContent = formatAiSettingValue(value);
                const icon = document.createElement('i');
                icon.className = 'fas fa-arrow-up-right-from-square text-[10px] text-blue-300 shrink-0';
                text.appendChild(label);
                text.appendChild(valueEl);
                button.appendChild(text);
                button.appendChild(icon);
                button.addEventListener('click', () => openAiSettingJumpTarget(key));
                list.appendChild(button);
            });
            const footer = bubble.querySelector('.message-footer-meta');
            if (footer) bubble.insertBefore(list, footer);
            else bubble.appendChild(list);
            scrollToBottom();
        }

        async function runAiSettingsCommand(instruction, modelId) {
            // Keep the command context active so the next prompt is a follow-up
            // instruction instead of falling back to a normal chat request.
            if (pendingSlashCommand !== 'settings') {
                pendingSlashCommand = 'settings';
                showPendingSlashCommandIndicator('settings');
            }
            appendAiSettingsConversation('user', instruction);
            const timestamp = Date.now();
            const userEl = renderMessage(`settings-user-${timestamp}`, 'user', `/settings ${instruction}`, null, null, null, null, true, null, null, null, null, null, null, null, null, true);
            removeEphemeralMessageControls(userEl);
            const welcome = get('welcome-screen');
            if (welcome) welcome.classList.add('hidden');
            const pendingId = `settings-pending-${timestamp}`;
            const chat = get('chat-container');
            if (chat) {
                chat.insertAdjacentHTML('beforeend', `<div id="${pendingId}" class="flex justify-start mb-4 ai-pending-row fade-in"><div class="message-bubble ai-pending-bubble bg-gray-700 text-white p-4 rounded-2xl rounded-tl-none shadow-md relative">${buildPendingSkeletonHtml(modelId, '設定リクエストを確認しています...')}</div></div>`);
                scrollToBottom();
            }
            try {
                const res = await apiFetch('/api/settings/apply-ai-prompt', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        prompt: instruction,
                        model: modelId,
                        conversation: aiSettingsConversation,
                    })
                });
                const data = await res.json().catch(() => ({}));
                const pending = get(pendingId);
                if (pending) pending.remove();
                if (data && data.status === 'ok' && data.mode === 'inspect' && data.current) {
                    appendAiSettingsConversation('assistant', summarizeAiSettingsConversationValues(data.current, 'inspect'));
                    showToast(`現在の設定を確認しました（${Object.keys(data.current).length}項目）`, 'success');
                    renderAiSettingsResultBubble(data.current, modelId, 'inspect');
                    return;
                }
                if (data && data.status === 'ok' && data.applied) {
                    appendAiSettingsConversation('assistant', summarizeAiSettingsConversationValues(data.applied, 'update'));
                    showToast(`設定を更新しました（${Object.keys(data.applied).length}項目）`, 'success');
                    try {
                        const fresh = await apiFetch(CHAT_CONFIG.urls.handleSettingsQuery).then((response) => response.json());
                        populateAiSafeFormFields(fresh);
                        cacheUserSettings(fresh);
                    } catch (e) {}
                    renderAiSettingsResultBubble(data.applied, modelId);
                    return;
                }
                const msg = data.message || data.error || '設定変更に失敗しました';
                appendAiSettingsConversation('assistant', `設定操作に失敗しました: ${msg}`);
                const errorEl = renderMessage(`settings-error-${Date.now()}`, 'assistant', `設定変更に失敗しました。\n\n${msg}`, null, null, modelId, null, true, null, null, null, null, null, null, null, null, true);
                removeEphemeralMessageControls(errorEl);
                showToast(msg, 'error', true);
            } catch (error) {
                appendAiSettingsConversation('assistant', '設定操作の通信に失敗しました。');
                const pending = get(pendingId);
                if (pending) pending.remove();
                const errorEl = renderMessage(`settings-error-${Date.now()}`, 'assistant', '設定変更の通信に失敗しました。時間をおいて再度お試しください。', null, null, modelId, null, true, null, null, null, null, null, null, null, null, true);
                removeEphemeralMessageControls(errorEl);
                showToast('設定変更の通信に失敗しました', 'error', true);
            }
        }

        // === Skill suggestion helpers (triggered by @ in prompt bar) ===
        function hideSkillSuggestions() {
            const box = get('skill-suggestions');
            if (box) box.classList.add('hidden');
            skillSuggestionsVisible = false;
            skillSelectedIndex = 0;
        }

        function showSkillSuggestions(filter = '') {
            const box = get('skill-suggestions');
            const listEl = get('skill-suggestions-list');
            const inputRow = get('input-row');
            if (!box || !listEl || !inputRow) return;

            if (!loadedSkills || loadedSkills.length === 0) {
                hideSkillSuggestions();
                return;
            }

            const f = filter.toLowerCase();
            const filtered = loadedSkills.filter(g =>
                g.name.toLowerCase().includes(f) || (g.description && g.description.toLowerCase().includes(f))
            );

            if (filtered.length === 0) {
                hideSkillSuggestions();
                return;
            }

            // Clamp selected index
            if (skillSelectedIndex >= filtered.length) skillSelectedIndex = 0;

            listEl.innerHTML = '';
            filtered.forEach((skill, idx) => {
                const item = document.createElement('div');
                item.className = `px-3 py-2 flex items-center gap-3 cursor-pointer text-sm hover:bg-gray-700 ${idx === skillSelectedIndex ? 'bg-gray-700' : ''}`;
                item.innerHTML = `
                    <i class="fas fa-wand-magic-sparkles w-4 text-blue-400"></i>
                    <div class="flex-1 min-w-0">
                        <div class="text-blue-300 truncate font-medium">${escapeHtml(skill.name)}</div>
                        ${skill.description ? `<div class="text-[11px] text-gray-400 truncate">${escapeHtml(skill.description)}</div>` : ''}
                    </div>
                `;
                item.onclick = () => selectSkillSuggestion(skill);
                item.onmouseenter = () => {
                    skillSelectedIndex = idx;
                    showSkillSuggestions(filter);
                };
                listEl.appendChild(item);
            });

            // Smart vertical positioning (same logic as slash commands)
            const rect = inputRow.getBoundingClientRect();
            const viewportHeight = window.innerHeight;
            const spaceBelow = viewportHeight - rect.bottom;
            const spaceAbove = rect.top;

            const desiredMaxHeight = 260;
            const padding = 8;

            box.style.position = 'fixed';
            box.style.left = `${Math.max(8, rect.left)}px`;
            box.style.zIndex = '80';
            box.style.maxHeight = 'none';

            const showAbove = spaceBelow < 180 && spaceAbove > spaceBelow;

            if (showAbove) {
                const popupHeight = Math.min(desiredMaxHeight, spaceAbove - padding);
                box.style.top = 'auto';
                box.style.bottom = `${viewportHeight - rect.top + 4}px`;
                listEl.style.maxHeight = `${popupHeight}px`;
            } else {
                const popupHeight = Math.min(desiredMaxHeight, spaceBelow - padding);
                box.style.top = `${rect.bottom + 4}px`;
                box.style.bottom = 'auto';
                listEl.style.maxHeight = `${popupHeight}px`;
            }

            box.classList.remove('hidden');
            skillSuggestionsVisible = true;
        }

        function selectSkillSuggestion(skill) {
            const input = get('prompt-input');
            if (!input) return;

            const val = input.value;
            const lastAt = val.lastIndexOf('@');
            if (lastAt !== -1) {
                input.value = val.substring(0, lastAt).trimEnd();
            } else {
                input.value = '';
            }

            hideSkillSuggestions();
            activateSkill(skill);
            input.focus();
            input.dispatchEvent(new Event('input', { bubbles: true }));
        }

        function browserFastModeIneligibility(rawText) {
            const model = String(get('model-select') ? get('model-select').value : '').toLowerCase();
            if (!rawText || !rawText.trim()) return 'プロンプトを入力してください';
            if (!model.startsWith('gemini-') || /(image|native-audio|tts|live)/.test(model)) return 'Geminiテキストモデル専用です';
            if (currentImageUrls.length) return 'サーバー保存済み添付があるため通常モードが必要です';
            if (activeSkill) return 'Skills利用時は通常モードが必要です';
            if (currentQuote || editingMessageId) return '引用・編集時は通常モードが必要です';
            if (codingModeEnabled) return 'Coding Mode利用時は通常モードが必要です';
            const enabledIds = ['enable-search', 'enable-url-context', 'enable-maps', 'enable-sys-prompt', 'enable-prompt-cache', 'enable-mcp'];
            if (enabledIds.some((id) => { const el = get(id); return !!(el && el.checked); })) return '検索・URL参照・システム機能利用時は通常モードが必要です';
            const custom = get('thread-custom-instruction');
            if (custom && String(custom.value || '').trim()) return 'チャット固有指示利用時は通常モードが必要です';
            const entries = Array.from(browserFastLocalFiles.values());
            if (entries.length > BROWSER_FAST_MAX_IMAGES) return '画像は4枚までです';
            const total = entries.reduce((sum, entry) => sum + Number(entry.file && entry.file.size || 0), 0);
            if (total > BROWSER_FAST_MAX_BYTES) return '画像合計は12MBまでです';
            if (entries.some((entry) => !entry.file || !String(entry.file.type || '').startsWith('image/'))) return '画像以外は利用できません';
            return '';
        }

        function fileToBase64Payload(file) {
            return new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.onload = () => {
                    const raw = String(reader.result || '');
                    const comma = raw.indexOf(',');
                    if (comma < 0) return reject(new Error('画像の読み込みに失敗しました'));
                    resolve(raw.slice(comma + 1));
                };
                reader.onerror = () => reject(reader.error || new Error('画像の読み込みに失敗しました'));
                reader.readAsDataURL(file);
            });
        }

        async function buildBrowserFastHistoryContents(history) {
            const contents = [];
            let omittedImages = 0;
            for (const item of (Array.isArray(history) ? history : [])) {
                if (!item || !['user', 'model'].includes(item.role)) continue;
                const parts = [];
                if (item.role === 'model' && Array.isArray(item.thought_signatures)) {
                    item.thought_signatures.forEach((signature) => {
                        if (signature) parts.push({ thoughtSignature: String(signature) });
                    });
                }
                if (item.text) parts.push({ text: String(item.text) });
                for (const image of (Array.isArray(item.images) ? item.images : [])) {
                    try {
                        const imageResponse = await fetch(buildFileUrl(image.path), {
                            credentials: 'same-origin',
                            cache: 'no-store',
                        });
                        if (!imageResponse.ok) throw new Error(`HTTP ${imageResponse.status}`);
                        const blob = await imageResponse.blob();
                        parts.push({
                            inlineData: {
                                mimeType: image.mime_type || blob.type || 'application/octet-stream',
                                data: await fileToBase64Payload(blob),
                            },
                        });
                    } catch (error) {
                        omittedImages++;
                    }
                }
                if (parts.length) contents.push({ role: item.role, parts });
            }
            if (omittedImages) {
                showToast(`履歴画像${omittedImages}件を再取得できなかったため、テキスト履歴だけで続行します`, 'warning', true);
            }
            return contents;
        }

        async function uploadBrowserFastLocalFiles(files = browserFastLocalFiles, detached = false) {
            const entries = Array.from(files.entries());
            const uploaded = [];
            for (const [uploadId, entry] of entries) {
                if (!entry || !entry.file || !entry.rowObj) throw new Error('ローカル画像の状態が失われました');
                if (entry.rowObj.status) entry.rowObj.status.textContent = '回答完了・サーバー保存中...';
                const ok = await uploadFileWithProgress(entry.file, entry.rowObj);
                if (!ok) throw new Error(`${entry.file.name || '画像'}をサーバーへ保存できませんでした`);
                const path = entry.rowObj.row.getAttribute('data-filename');
                if (path) uploaded.push(path);
                if (detached) currentImageUrls = currentImageUrls.filter((value) => value !== path);
                else browserFastLocalFiles.delete(uploadId);
            }
            return uploaded;
        }

        function browserFastThinkingConfig(model) {
            const thinking = get('enable-thinking');
            if (!thinking || !thinking.checked) return null;
            const rawLevel = String(get('thinking-level') ? get('thinking-level').value : 'high').toLowerCase();
            if (model.includes('2.5')) {
                const manual = Number(get('thinking-budget') ? get('thinking-budget').value : 4096);
                const budget = Number.isFinite(manual) ? Math.max(0, Math.min(32768, Math.trunc(manual))) : 4096;
                return { includeThoughts: true, thinkingBudget: budget };
            }
            let level = rawLevel.toUpperCase();
            if (model.includes('3.6') && !['MEDIUM', 'HIGH'].includes(level)) level = 'MEDIUM';
            if (model.includes('3.5') && !['MINIMAL', 'MEDIUM', 'HIGH'].includes(level)) level = 'MINIMAL';
            return { includeThoughts: true, thinkingLevel: level };
        }

        function browserFastPythonBoxHtml(pyId) {
            return `<div class="code-wrapper python-box collapsed" data-py-id="${pyId}" data-collapsed="true" data-code-key="${pyId}"><div class="code-header"><span class="code-lang"><i class="fas fa-terminal"></i> Python Execution</span><div class="code-actions"><button class="code-toggle" aria-expanded="false" title="展開" aria-label="展開"><i class="fas fa-chevron-down"></i></button><button class="copy-btn" data-copy="code" data-code="" title="コードをコピー" aria-label="コードをコピー"><i class="fas fa-copy"></i></button><button class="copy-btn" data-copy="output" data-code="" title="出力をコピー" aria-label="出力をコピー"><i class="fas fa-align-left"></i></button></div></div><div class="code-body"><div class="python-section"><div class="python-label">Code</div><pre><code class="hljs language-python python-code"></code></pre></div><div class="python-section"><div class="python-label">Output</div><pre><code class="hljs language-plaintext python-output"></code></pre></div></div></div>`;
        }

        function updateBrowserFastPythonBox(box, field, value) {
            if (!box) return;
            if (field === 'code') {
                const codeText = value == null ? '' : String(value);
                const codeEl = box.querySelector('.python-code');
                if (codeEl) {
                    codeEl.textContent = codeText;
                    codeEl.removeAttribute('data-highlighted');
                    queueHighlight(box, codeText);
                }
                const codeBtn = box.querySelector('.copy-btn[data-copy="code"]');
                if (codeBtn) codeBtn.setAttribute('data-code', encodeURIComponent(codeText).replace(/'/g, "%27"));
            } else if (field === 'output') {
                const outText = value == null ? '' : String(value);
                const outEl = box.querySelector('.python-output');
                if (outEl) outEl.textContent = outText;
                const outBtn = box.querySelector('.copy-btn[data-copy="output"]');
                if (outBtn) outBtn.setAttribute('data-code', encodeURIComponent(outText).replace(/'/g, "%27"));
            }
        }

        // 高速モードの保存リクエストに載せる、プロンプトバーの選択内容（歯車から開く項目は含めない）。
        function collectBrowserFastPromptOptions() {
            const checked = (id, fallback) => (get(id) ? !!get(id).checked : fallback);
            const value = (id) => (get(id) ? get(id).value : null);
            // 高速モードはSkillsを強制オフにする。保存する設定は、切り替え前の
            // 利用者の選択へ戻せるよう、退避した値を記録する。
            const priorSkills = browserFastPreviousOptions && browserFastPreviousOptions.checks
                && Object.prototype.hasOwnProperty.call(browserFastPreviousOptions.checks, 'enable-skills')
                ? !!browserFastPreviousOptions.checks['enable-skills'] : true;
            return {
                canvas_mode: !!canvasModeEnabled,
                enable_search: checked('enable-search', false),
                gemini_free_tier: checked('enable-gemini-free-tier', false),
                enable_url_context: checked('enable-url-context', false),
                enable_maps: checked('enable-maps', false),
                enable_python: checked('enable-python', false),
                enable_file_creation: checked('enable-file-creation', true),
                enable_mcp: typeof isMcpEnabledForSend === 'function' ? !!isMcpEnabledForSend() : false,
                enable_skills: priorSkills,
                enable_thinking: checked('enable-thinking', false),
                thinking_level: value('thinking-level'),
                thinking_budget: value('thinking-budget'),
                reasoning_effort: value('reasoning-effort'),
                safety_setting: value('safety-setting'),
                enable_prompt_caching: checked('enable-prompt-cache', false),
                math_steps: (typeof window.collectMathStepsSendOption === 'function' ? window.collectMathStepsSendOption() : false),
            };
        }

        async function sendBrowserFastMessage(rawText, pendingStartedAt) {
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
            if (bootstrap.math_steps_system_prompt) {
                payload.systemInstruction = { parts: [{ text: String(bootstrap.math_steps_system_prompt) }] };
            }
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
            const followSend = userAutoScroll && isChatNearBottom();
            if (!followSend) pauseChatAutoScroll();
            const fastImageRefs = collectImageUrlsForSend();
            const fastThumbnails = Array.from(get('file-preview-thumbs')?.querySelectorAll('img') || []).map((img) => img.cloneNode(true));
            const fastUserRow = renderMessage(Date.now(), 'user', rawText, JSON.stringify(collectImageUrlsForSend()), null, null, null, false, currentQuote, null, null, null, null, null, null, null, followSend);
            if (fastThumbnails.length && !fastImageRefs.length) {
                const thumbnails = document.createElement('div');
                thumbnails.className = 'flex gap-2 flex-wrap';
                fastThumbnails.forEach((img) => { img.removeAttribute('onclick'); thumbnails.appendChild(img); });
                fastUserRow.querySelector('.message-bubble').appendChild(thumbnails);
            }
            const sentPrompt = takePromptForSend();
            const stopSendMotion = animatePromptToMessage(fastUserRow, sentPrompt, followSend);
            const aid = `browser-fast-${Date.now()}`;
            get('chat-container').insertAdjacentHTML('beforeend', `<div class="flex justify-start mb-4 ai-pending-row fade-in"><div id="${aid}" class="message-bubble ai-pending-bubble bg-gray-700 text-white p-4 rounded-2xl rounded-tl-none shadow-md relative">${buildPendingSkeletonHtml(model, 'Geminiへ直接送信中...')}</div></div>`);
            const adiv = get(aid);
            const stopPendingElapsed = startPendingElapsed(adiv, pendingStartedAt);
            activeStreamingBubbleId = aid;
            setSendBtnToStopMode();
            if (followSend) resumeChatAutoScroll();
            abortController = new AbortController();
            let content = '';
            let thought = '';
            let pricingUsage = null;
            const pricingQueries = new Set();
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
                sentPrompt.accept(false);
                window.ActivityLog?.log('send.stream.start');
                let loggedFirstChunk = false;
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';
                const consumeEvent = (block) => {
                    const dataText = block.split(/\r?\n/).filter((line) => line.startsWith('data:')).map((line) => line.slice(5).trim()).join('');
                    if (!dataText || dataText === '[DONE]') return;
                    const evt = JSON.parse(dataText);
                    if (Number.isInteger(evt.usageMetadata?.promptTokenCount)) {
                        pricingUsage = pricingUsage || {};
                        for (const key of ['promptTokenCount', 'candidatesTokenCount', 'thoughtsTokenCount', 'cachedContentTokenCount']) {
                            if (Number.isInteger(evt.usageMetadata[key]) && evt.usageMetadata[key] >= 0) pricingUsage[key] = evt.usageMetadata[key];
                        }
                    }
                    for (const candidate of evt.candidates || []) {
                        for (const query of candidate.groundingMetadata?.webSearchQueries || []) {
                            if (typeof query === 'string' && query.trim()) pricingQueries.add(query);
                        }
                    }
                    if (pricingQueries.size) { pricingUsage = pricingUsage || {}; pricingUsage.grounding_queries = pricingQueries.size; }
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
                        renderAiMarkdownInto(contentEl, content, { incrementalMath: true, streaming: true, mathSteps: { jobId: null, streaming: true } });
                    }
                    scrollToBottom();
                };
                while (true) {
                    const { done, value } = await reader.read();
                    if (!done && !loggedFirstChunk) { loggedFirstChunk = true; window.ActivityLog?.log('send.stream.first'); }
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
                if (contentEl) renderAiMarkdownInto(contentEl, content, { incrementalMath: true, streaming: false, mathSteps: { jobId: null, streaming: false } });
                if (thoughtEl) thoughtEl.classList.add('collapsed');
                if (pyExecPayloads.length) {
                    content += pyExecPayloads.map((payload) => `\n\`\`\`pyexec\n${JSON.stringify(payload)}\n\`\`\`\n`).join('');
                }

                let fastUploadedRefs = [];
                if (localEntries.length) {
                    if (finishProgress) finishProgress.setPhase('saving');
                    showToast('回答が完了しました。画像と履歴をサーバーへ保存しています。', 'info', false);
                    fastUploadedRefs = await uploadBrowserFastLocalFiles(sentPrompt.files, true);
                }
                if (finishProgress) finishProgress.setPhase('saving');
                const refs = [...fastImageRefs, ...fastUploadedRefs];
                const saveResponse = await fetchChatStreamWithUnavailableRetry('/api/browser_fast_mode/save', manualSpinnerRequestOptions({
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        client_request_id: createClientRequestId(),
                        message: rawText,
                        assistant_content: content,
                        pricing_usage: pricingUsage,
                        thought_content: thought,
                        model,
                        image_urls: refs,
                        prompt_options: collectBrowserFastPromptOptions(),
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
                if (saved.assistant_message_id !== undefined && saved.assistant_message_id !== null && typeof window.bindMathStepsMessageId === 'function') {
                    const stepRoot = adiv ? adiv.querySelector('.math-steps') : null;
                    if (stepRoot) await window.bindMathStepsMessageId(stepRoot, saved.assistant_message_id);
                }
                const createdThread = !currentThreadId;
                currentThreadId = String(saved.thread_id);
                currentParentId = saved.assistant_message_id || null;
                currentLeafId = saved.assistant_message_id || null;
                browserFastBootstrap = null;
                await loadMessages(currentThreadId, { preserveDraft: true, silent: true, skipHistory: !createdThread });
                applyBrowserFastModeRestrictions();
                loadThreads(false);
                showToast('高速モードの回答を履歴へ保存しました', 'success', false);
            } catch (error) {
                stopSendMotion();
                sentPrompt.restore();
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
                        const refs = localEntries.length ? [] : fastImageRefs;
                        const saveResponse = await fetchChatStreamWithUnavailableRetry('/api/browser_fast_mode/save', manualSpinnerRequestOptions({
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                client_request_id: createClientRequestId(),
                                message: rawText,
                                assistant_content: assistantContent,
                                pricing_usage: pricingUsage,
                                thought_content: thought || '',
                                model,
                                image_urls: refs,
                                prompt_options: collectBrowserFastPromptOptions(),
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
                sentPrompt.releaseFiles();
                if (stopPendingElapsed) stopPendingElapsed();
                if (browserFastOpStarted && window.ConnectionMonitor) window.ConnectionMonitor.operationEnded();
                if (finishProgress) finishProgress();
                setSendBtnToSendMode();
                if (activeStreamingBubbleId === aid) activeStreamingBubbleId = null;
                abortController = null;
                updateFilePreview();
            }
        }

        async function sendMessage() {
            if (!window.ActivityLog) return sendMessageLogged();
            return window.ActivityLog.operation('send', () => sendMessageLogged());
        }
        async function sendMessageLogged(options = {}) {
            const pendingStartedAt = performance.now();
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
            const removeBackground = isIdeogramBackgroundRemoval();
            const rawText = ideogramPromptForSend(get('prompt-input').value); // Keep ordinary prompts untrimmed.

            // Handle pending slash command (e.g. after selecting /settings via the palette)
            if (pendingSlashCommand && !removeBackground) {
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
                } else if (executeMinimalSlashCommand(cmd, instruction)) {
                    get('prompt-input').value = '';
                    get('prompt-input').style.height = 'auto';
                    hidePendingSlashCommandIndicator();
                } else {
                    get('prompt-input').focus();
                }
                return; // Do not treat as normal message
            }

            const directMinimalCommand = rawText.trim().match(/^\/([a-z][\w-]*)(?:\s+(.*))?$/i);
            if (directMinimalCommand && minimalPromptMode && MINIMAL_SLASH_COMMANDS.some((command) => command.id === directMinimalCommand[1].toLowerCase())) {
                if (executeMinimalSlashCommand(directMinimalCommand[1].toLowerCase(), directMinimalCommand[2] || '')) {
                    hideSlashCommandSuggestions();
                    get('prompt-input').value = '';
                    get('prompt-input').style.height = 'auto';
                }
                return;
            }

            const batchModeRequested = !!(get('enable-batch-mode') && get('enable-batch-mode').checked);
            if (batchModeRequested && codingModeEnabled) {
                showToast('Batch APIではCoding Modeを利用できません。Batchを解除するかCodingを解除してください。', 'warning', true);
                return;
            }

            if (browserFastModeEnabled) {
                if (batchModeRequested) {
                    setBrowserFastModeEnabled(false);
                } else {
                const reason = browserFastModeIneligibility(rawText);
                if (!reason) {
                    try {
                        await sendBrowserFastMessage(rawText, pendingStartedAt);
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
