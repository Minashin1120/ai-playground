            window.collectAutoSystemPromptConfigFromForm = collectAutoSystemPromptConfigFromForm;
            window.ensureAutoSystemPromptSettingsCard = () => {
                const promptToggle = get('set-global-sys-prompt-enabled');
                const wrapHost = promptToggle ? promptToggle.closest('.space-y-4') : null;
                if (!wrapHost || get('auto-sys-prompt-settings')) return;
                const box = document.createElement('div');
                box.id = 'auto-sys-prompt-settings';
                box.className = 'border-t border-gray-700 pt-3';
                box.innerHTML = `
                    <div class="flex items-center justify-between mb-2">
                        <label class="text-xs text-gray-500 block">自動注入システムプロンプト（ユーザー単位）</label>
                        <div class="flex items-center gap-2">
                            <button type="button" id="reset-set-auto-sys-prompt-defaults" class="bg-gray-700 hover:bg-gray-600 text-white px-2 py-1 rounded text-[10px] font-bold btn-hover">既定に戻す</button>
                            <label class="flex items-center gap-1 text-[10px] text-gray-400">
                                <input type="checkbox" id="set-apply-auto-sys-prompt-notices" class="accent-yellow-500 w-3 h-3">
                                <span>全体適用</span>
                            </label>
                        </div>
                    </div>
                    <div id="set-auto-sys-prompt-items" class="space-y-2">${window.buildAutoSystemPromptRows('set', false)}</div>
                    <div class="text-[10px] text-gray-500 mt-2">各文面はユーザー単位で編集されます。空欄で保存すると既定文面に戻ります。</div>
                `;
                wrapHost.appendChild(box);
            };
            window.ensureThreadAutoSystemPromptCard = () => {
                const threadPrompt = get('thread-global-sys-prompt');
                const wrapHost = threadPrompt ? threadPrompt.closest('.space-y-3') : null;
                if (!wrapHost || get('thread-auto-sys-prompt-settings')) return;
                const box = document.createElement('div');
                box.id = 'thread-auto-sys-prompt-settings';
                box.className = 'border-t border-gray-700 pt-3';
                box.innerHTML = `
                    <div class="flex items-center justify-between mb-2">
                        <div class="text-xs text-gray-400">自動注入システムプロンプト（ユーザー単位）</div>
                        <div class="flex items-center gap-2">
                            <button type="button" id="reset-thread-auto-sys-prompt-defaults" class="bg-gray-700 hover:bg-gray-600 text-white px-2 py-1 rounded text-[10px] font-bold btn-hover">既定に戻す</button>
                            <label class="flex items-center gap-1 text-[10px] text-gray-500">
                                <input type="checkbox" id="thread-apply-auto-sys-prompt-notices" class="accent-yellow-500 w-3 h-3">
                                <span>全体適用</span>
                            </label>
                        </div>
                    </div>
                    <div id="thread-auto-sys-prompt-items" class="space-y-2">${window.buildAutoSystemPromptRows('thread', true)}</div>
                `;
                wrapHost.appendChild(box);
            };
            ensureTemporaryChatSettingsCard();
            ensureLlmTranscribePromptSettingsUi();
            bindTemporaryChatToggle();
            const populateDefaultModelOptions = () => {
                const sel = get('set-default-model');
                if (!sel) return;
                const current = sel.value;
                sel.innerHTML = '';
                MODELS.forEach(group => {
                    const optgroup = document.createElement('optgroup');
                    optgroup.label = group.category;
                    (group.items || []).forEach(item => {
                        const opt = document.createElement('option');
                        opt.value = item.id;
                        opt.textContent = item.name;
                        optgroup.appendChild(opt);
                    });
                    sel.appendChild(optgroup);
                });
                const stored = (userSettingsSnapshot && userSettingsSnapshot.default_model) || current || 'gemini-3.6-flash';
                if (stored && Array.from(sel.options).some(o => o.value === stored)) sel.value = stored;
                syncSettingsModelPickerLabels();
            };
            const populateDefaultVisionModelOptions = () => {
                const sel = get('set-default-vision-model');
                if (!sel) return;
                const current = sel.value;
                sel.innerHTML = '';
                MODELS.forEach(group => {
                    const visionItems = (group.items || []).filter(m => {
                        const id = (m.id || '').toLowerCase();
                        return id.startsWith('gemini-') || id.startsWith('gpt-4o') || id.startsWith('claude-') || id.startsWith('grok-3') || ['glm-5.3-flash', 'glm-5.3-flashx', 'glm-4.6v', 'glm-4.6v-flashx', 'glm-4.6v-flash', 'glm-4.5v'].includes(id);
                    });
                    if (visionItems.length === 0) return;
                    const optgroup = document.createElement('optgroup');
                    optgroup.label = group.category;
                    visionItems.forEach(item => {
                        const opt = document.createElement('option');
                        opt.value = item.id;
                        opt.textContent = item.name + ' ★';
                        optgroup.appendChild(opt);
                    });
                    sel.appendChild(optgroup);
                });
                const stored = (userSettingsSnapshot && userSettingsSnapshot.default_vision_model) || current || 'gemini-3-flash-preview';
                if (stored && Array.from(sel.options).some(o => o.value === stored)) sel.value = stored;
                syncSettingsModelPickerLabels();
            };
            // Populate every settings-modal control from a settings payload.  Shared
            // by openSettingsModal and the account-import flow so imported settings
            // and API credentials are reflected in the open settings modal instead of
            // leaving stale (pre-import) values that a later save would overwrite.
            const populateSettingsFormFromData = (d) => {
                if (!d) return;
                cacheUserSettings(d);
                const globalPreview = get('app-global-sys-prompt-preview');
                if (globalPreview) {
                    globalPreview.value = d.global_system_prompt_effective || '';
                }
                const globalPreviewStatus = get('app-global-sys-prompt-preview-status');
                if (globalPreviewStatus) {
                    if (d.global_system_prompt_enabled === false) {
                        globalPreviewStatus.textContent = '現在は無効化されています。';
                    } else if (d.global_system_prompt_uses_time_fallback) {
                        globalPreviewStatus.textContent = '管理者設定が空欄のため、時刻の既定プロンプトが適用されています。';
                    } else {
                        globalPreviewStatus.textContent = '管理者が設定した全体システムプロンプトが適用されています。';
                    }
                }
                if(get('sys-prompt-text')) get('sys-prompt-text').value = d.system_prompt || '';
                if(get('set-global-sys-prompt-enabled')) get('set-global-sys-prompt-enabled').checked = d.system_prompt_enabled !== false;

                window.ensureAutoSystemPromptSettingsCard();
                if(get('set-apply-global-sys-prompt')) get('set-apply-global-sys-prompt').checked = d.apply_global_system_prompt !== false;
                if(get('set-apply-auto-sys-prompt-notices')) get('set-apply-auto-sys-prompt-notices').checked = d.apply_auto_system_prompt_notices !== false;
                window.applyAutoSystemPromptConfigToForm('set', d.auto_system_prompt_notices_config || {});

                if(get('set-latency-metrics')) get('set-latency-metrics').checked = d.enable_latency_metrics === true;
                if(get('set-client-debug-log')) syncClientDebugLogToggle(d.enable_client_debug_log === true, 'settings modal sync');
                if(get('set-openai')) get('set-openai').value = d.openai_key || '';
                if(get('set-gemini')) get('set-gemini').value = d.gemini_key || '';
                if(get('set-gemini-free')) get('set-gemini-free').value = d.gemini_free_key || '';
                if(get('set-deepseek')) get('set-deepseek').value = d.deepseek_key || '';
                if(get('set-zai')) get('set-zai').value = d.zai_key || '';
                if(get('set-kimi')) get('set-kimi').value = d.kimi_key || '';
                if(get('set-mistral')) get('set-mistral').value = d.mistral_key || '';
                if(get('set-ideogram')) get('set-ideogram').value = d.ideogram_key || '';
                if(get('set-anthropic')) get('set-anthropic').value = d.anthropic_key || '';
                if(get('set-gemini-backend')) get('set-gemini-backend').value = normalizeGeminiBackend(d.gemini_backend || 'gemini_api');
                if(get('set-gemini-vertex-project')) get('set-gemini-vertex-project').value = d.gemini_vertex_project || '';
                if(get('set-gemini-vertex-location')) get('set-gemini-vertex-location').value = d.gemini_vertex_location || 'global';
                ensureGeminiVertexCredentialsField();
                if(get('set-gemini-vertex-credentials-json')) get('set-gemini-vertex-credentials-json').value = d.gemini_vertex_credentials_json || '';
                syncGeminiBackendUi();
                if (typeof window.toggleOptions === 'function') window.toggleOptions();
                if(get('set-admin-api-key-mode')) get('set-admin-api-key-mode').value = normalizeAdminApiKeyMode(d.admin_api_key_mode || 'env_fallback');
                syncAdminApiKeyModeUi();
                if(get('set-xai')) get('set-xai').value = d.xai_key || '';
            if(get('set-google-key')) get('set-google-key').value = d.google_key || '';
            if(get('set-google-project')) get('set-google-project').value = d.google_project || '';
            modelApiKeyMap = normalizeModelApiKeyMap(d.model_api_keys || {});
            syncModelApiKeyModelOptions();
            renderModelApiKeyList();
            setModelApiKeyPanelOpen(false);
            if(get('set-mic-transcribe-mode')) get('set-mic-transcribe-mode').value = d.mic_transcribe_mode || 'stt_api';
            if(get('set-stt-model')) get('set-stt-model').value = d.stt_model || 'gpt-4o-mini-transcribe';
            if(get('set-llm-transcribe-prompt')) {
                get('set-llm-transcribe-prompt').value = d.llm_transcribe_prompt || '';
                get('set-llm-transcribe-prompt').placeholder = d.llm_transcribe_prompt_default || '';
            }
            syncRichPastePromptPreferencesUi(d);
            updateGoogleLinkUI(d);
            updateMinashinLinkUI(d);
            if(get('set-enter-to-send')) get('set-enter-to-send').checked = !!d.enter_to_send;
            if (typeof window.applyMathStepsSettings === 'function') window.applyMathStepsSettings(d);
            if(get('set-math-steps-enabled')) get('set-math-steps-enabled').checked = d.math_steps_enabled === true;
            if(get('set-default-math-steps')) get('set-default-math-steps').checked = d.default_math_steps === true;
            writePromptBarModeToForm(!!d.compact_prompt_mode, !!d.minimal_prompt_mode);
                if(get('set-use-sw-cache')) get('set-use-sw-cache').checked = !!d.use_sw_cache;
                if(get('set-clear-cache-on-version-update')) get('set-clear-cache-on-version-update').checked = !!d.clear_cache_on_version_update;
            if(get('set-liquid-glass')) get('set-liquid-glass').checked = !!d.liquid_glass_enabled;
            if(get('set-light-mode')) get('set-light-mode').checked = !!d.light_mode_enabled;
            if(get('set-auto-search-links')) get('set-auto-search-links').checked = d.auto_search_on_links !== false;
            if(get('set-use-last-settings')) get('set-use-last-settings').checked = !!d.use_last_chat_settings;
            if(get('set-default-model')) get('set-default-model').value = d.default_model || 'gemini-3.6-flash';
            if(get('set-default-vision-model')) get('set-default-vision-model').value = d.default_vision_model || 'gemini-3-flash-preview';
            syncSettingsModelPickerLabels();
            applyTemporaryChatTimeoutSeconds(d.temp_chat_timeout_seconds);
            if(get('set-default-search')) get('set-default-search').checked = !!d.default_enable_search;
            if(get('set-default-gemini-free-tier')) get('set-default-gemini-free-tier').checked = !!d.default_gemini_free_tier;
            if(get('set-default-url-context')) get('set-default-url-context').checked = !!d.default_enable_url_context;
            if(get('set-default-maps')) get('set-default-maps').checked = !!d.default_enable_maps;
            if(get('set-default-python')) get('set-default-python').checked = !!d.default_enable_python;
            if(get('set-default-file-creation')) get('set-default-file-creation').checked = !!d.default_enable_file_creation;
            if(get('set-default-thinking')) get('set-default-thinking').checked = !!d.default_enable_thinking;
            if(get('set-default-sys-prompt')) get('set-default-sys-prompt').checked = !!d.default_enable_system_prompt;
            if(get('set-default-mcp')) get('set-default-mcp').checked = d.default_enable_mcp !== false;
            if(get('set-default-thinking-level')) get('set-default-thinking-level').value = d.default_thinking_level || 'high';
            if(get('set-default-thinking-budget')) get('set-default-thinking-budget').value = d.default_thinking_budget || 4096;
            if(get('set-default-reasoning-effort')) get('set-default-reasoning-effort').value = d.default_reasoning_effort || 'medium';
            if(get('set-default-safety')) get('set-default-safety').value = d.default_safety_setting || 'default';
            get('set-e2ee').checked = d.enable_e2ee;
            if(get('set-bot-detect')) get('set-bot-detect').checked = d.bot_detection_enabled !== false;
            if(get('set-bot-detect-global')) get('set-bot-detect-global').checked = d.bot_detection_global_enabled !== false;
            const botStatus = get('bot-status');
            if (botStatus) {
                if (d.is_bot_banned) {
                    botStatus.textContent = `BAN中: ${d.bot_ban_reason || 'Bot detection'}`;
                    botStatus.classList.remove('hidden');
                    botStatus.classList.add('text-red-400');
                } else {
                    botStatus.classList.add('hidden');
                }
            }
            if (d && d.theme_color) {
                applyThemeColor(d.theme_color, true);
                syncThemeInputs(d.theme_color);
            } else {
                syncThemeInputs(localStorage.getItem(THEME_STORAGE_KEY) || INITIAL_THEME_COLOR || THEME_DEFAULT);
            }
            snapshotSidebarHistory('settings-theme-synced');
            syncGeminiLocalPyDialogSetting();
            syncCompressionSettingsUi();
            if(get('set-username')) get('set-username').value = d.username;

            // 2FA UI Update
            const badge = get('2fa-badge');
            const disBtn = get('disable-2fa-btn');
            if(d.is_2fa_enabled) {
                badge.innerText = "ENABLED";
                badge.classList.replace('bg-gray-700', 'bg-green-600');
                badge.classList.replace('text-gray-400', 'text-white');
                disBtn.classList.remove('hidden');
            } else {
                badge.innerText = "DISABLED";
                badge.classList.replace('bg-green-600', 'bg-gray-700');
                badge.classList.replace('text-white', 'text-gray-400');
                disBtn.classList.add('hidden');
            }
            if(get('set-skip-2fa-google')) get('set-skip-2fa-google').checked = !!d.skip_2fa_on_google_login;
            if(get('set-default-2fa-method')) get('set-default-2fa-method').value = d.default_2fa_method || 'totp';

            const pkOnly = get('set-passkey-only-login');
            const pkNote = get('passkey-only-note');
            const passkeys = Array.isArray(d.passkey_credentials) ? d.passkey_credentials : [];
            renderPasskeyList(passkeys);
            if (pkOnly) {
                pkOnly.checked = !!d.passkey_only_login;
                const hasKey = passkeys.length > 0 || !!d.has_webauthn;
                pkOnly.disabled = !hasKey;
                if (!hasKey) pkOnly.checked = false;
                if (pkNote) {
                    if (hasKey) pkNote.classList.add('hidden');
                    else pkNote.classList.remove('hidden');
                }
            }

            const migBox = get('mig-status-box');
            const migText = get('mig-progress-text');
            const migBar = get('mig-progress-bar');
            const status = d.migration_status || 'idle';
            if (status === 'processing') {
                migBox.classList.remove('hidden');
                const prog = (d.migration_progress || '').split('/');
                if (prog.length === 2) {
                    const done = parseInt(prog[0] || '0', 10);
                    const total = parseInt(prog[1] || '0', 10);
                    if (migText) migText.innerText = `${done} / ${total}`;
                    if (migBar && total > 0) migBar.style.width = `${Math.min(100, Math.floor((done/total)*100))}%`;
                }
            } else {
                migBox.classList.add('hidden');
                if (migBar) migBar.style.width = '0%';
                if (migText) migText.innerText = '';
            }
            settingsModalLoaded = true;
            setSettingsSaveEnabled(true);
            };

            window.openSettingsModal = async () => {
                settingsModalLoaded = false;
                setSettingsSaveEnabled(false);
                snapshotSidebarHistory('settings-open-before');
                const settingsData = await ensureUserSettingsSnapshot();
                if (settingsData) populateSettingsFormFromData(settingsData);
                const searchEl = get('search-box');
                const preservedThreadSearch = searchEl ? searchEl.value : '';
                clearTimeout(searchTimeout);
                const ss = get('settings-search');
                if (ss) { ss.value = ''; }
                filterSettings();
                populateDefaultModelOptions();
                populateDefaultVisionModelOptions();
                showModal('settings-modal');
                refreshSettingsTabsScroll();
                requestAnimationFrame(() => refreshSettingsTabsScroll());
                restoreThreadSearchValue(preservedThreadSearch, 'restored-search-box-open');
                revealPersistentSidebarLists();
                snapshotSidebarHistory('settings-open-after');
                [50, 200, 400, 800].forEach((ms) => {
                    setTimeout(() => {
                        restoreThreadSearchValue(preservedThreadSearch, 'restored-search-box-' + ms + 'ms');
                        snapshotSidebarHistory('settings-open-later-' + ms + 'ms');
                    }, ms);
                });
                syncAdaptiveBlurSettingsUi();
                loadStorageUsage();
                loadSiteCacheUsage();
                refreshLatestAccountExport();
                ensureLlmTranscribePromptSettingsUi();
                if (typeof window.__loadAdminEncThreads === 'function') {
                    try { window.__loadAdminEncThreads(); } catch (_) {}
                }
                if (location.pathname !== '/settings') {
                    history.pushState({ modal: 'settings', from: location.pathname }, '', '/settings');
                }
                    refreshBanAppealSummary(true);
                    loadBanAppeals();
                if (!settingsData) {
                    // The saved settings could not be loaded. Keep the form
                    // uneditable so a save cannot overwrite settings with default
                    // values (previously this silently toggled E2EE and other
                    // fields). The bounded request above also guarantees that a
                    // network stall cannot leave this state indefinitely.
                    settingsModalLoaded = false;
                    setSettingsSaveEnabled(false);
                    showToast('設定の読み込みに失敗しました。閉じて再度開いてください', 'error', true);
                }
                loadFeedback();
                bindSessionButtons();
                loadSessions();
                try { loadMcpServers(); } catch (e) {}
            };
            const closeSettingsModal = (skipHistory = false) => {
                if (settingsModelPickerTarget) window.closeModelModal(true);
                snapshotSidebarHistory('settings-close-before');
                hideModal('settings-modal');
                revealPersistentSidebarLists();
                snapshotSidebarHistory('settings-close-after');
                setTimeout(() => snapshotSidebarHistory('settings-close-later-300ms'), 300);
                if (!skipHistory && location.pathname === '/settings') {
                    history.back();
                }
            };
            const bindThemeControls = () => {
                const colorInput = get('set-theme-color');
                const textInput = get('set-theme-color-text');
                const resetBtn = get('theme-reset-btn');
                const swatches = document.querySelectorAll('#theme-presets .theme-swatch');
                const applyFromValue = (value, persist = true) => {
                    const hex = normalizeHex(value);
                    if (!hex) return;
                    applyThemeColor(hex, persist);
                    syncThemeInputs(hex);
                };
                if (colorInput) {
                    colorInput.addEventListener('input', () => applyFromValue(colorInput.value, true));
                }
                if (textInput) {
                    textInput.addEventListener('change', () => {
                        const hex = normalizeHex(textInput.value);
                        if (!hex) {
                            syncThemeInputs(localStorage.getItem(THEME_STORAGE_KEY) || THEME_DEFAULT);
                            return;
                        }
                        applyFromValue(hex, true);
                    });
                    textInput.addEventListener('keydown', (e) => {
                        if (e.key === 'Enter') {
                            e.preventDefault();
                            textInput.blur();
                        }
                    });
                }
                if (resetBtn) resetBtn.onclick = () => applyFromValue(THEME_DEFAULT, true);
                swatches.forEach((btn) => {
                    btn.addEventListener('click', () => applyFromValue(btn.getAttribute('data-color'), true));
                });
            };
            const bindSystemPromptControls = () => {
                const resetGlobal = get('reset-global-sys-prompt');
                if (resetGlobal) {
                    resetGlobal.onclick = () => {
                        if (get('sys-prompt-text')) get('sys-prompt-text').value = '';
                        if (get('set-global-sys-prompt-enabled')) get('set-global-sys-prompt-enabled').checked = false;
                        showToast('ユーザーシステムプロンプトをリセットしました（保存してください）', 'success');
                    };
                }
                const resetThreadUserPrompt = get('reset-thread-sys-prompt');
                if (resetThreadUserPrompt) {
                    resetThreadUserPrompt.onclick = () => {
                        if (get('thread-global-sys-prompt')) get('thread-global-sys-prompt').value = '';
                        if (get('thread-global-sys-prompt-enabled')) get('thread-global-sys-prompt-enabled').checked = false;
                        showToast('ユーザーシステムプロンプトをリセットしました（保存してください）', 'success');
                    };
                }
                const resetAutoSet = get('reset-set-auto-sys-prompt-defaults');
                if (resetAutoSet) {
                    resetAutoSet.onclick = () => {
                        resetAutoSystemPromptConfigToCodeDefaults('set', 'set-apply-auto-sys-prompt-notices');
                        showToast('自動注入システムプロンプトを既定値に戻しました（保存してください）', 'success');
                    };
                }
                const resetAutoThread = get('reset-thread-auto-sys-prompt-defaults');
                if (resetAutoThread) {
                    resetAutoThread.onclick = () => {
                        resetAutoSystemPromptConfigToCodeDefaults('thread', 'thread-apply-auto-sys-prompt-notices');
                        showToast('自動注入システムプロンプトを既定値に戻しました（保存してください）', 'success');
                    };
                }
            };
            get('settings-btn').onclick = () => {
                openSettingsModal();
            };
            get('close-settings-btn').onclick = () => closeSettingsModal();
            const settingsHeaderClose = get('settings-header-close');
            if (settingsHeaderClose) settingsHeaderClose.onclick = () => closeSettingsModal();
            const settingsSearchInput = get('settings-search');
            if (settingsSearchInput) {
                settingsSearchInput.addEventListener('input', filterSettings);
                settingsSearchInput.addEventListener('keydown', (e) => {
                    if (e.key === 'Enter') {
                        const tab = get('tab-' + activeSettingsTab);
                        if (!tab) return;
                        const first = tab.querySelector(':scope > .settings-match');
                        if (first) first.scrollIntoView({ behavior: 'smooth', block: 'start' });
                    }
                });
            }
            const settingsSearchClear = get('settings-search-clear');
            if (settingsSearchClear) {
                settingsSearchClear.addEventListener('click', () => {
                    if (settingsSearchInput) {
                        settingsSearchInput.value = '';
                        filterSettings();
                        settingsSearchInput.focus();
                    }
                });
            }
            bindThemeControls();
            bindSystemPromptControls();
            bindModelApiKeySettingsControls();
            syncGeminiLocalPyDialogSetting();
            syncCompressionSettingsUi();
            const localPyDialogSetting = get('set-gemini-local-python-dialog');
            if (localPyDialogSetting) {
                localPyDialogSetting.onchange = () => setGeminiLocalPyDialogEnabled(localPyDialogSetting.checked);
            }
            const geminiBackendSetting = get('set-gemini-backend');
            if (geminiBackendSetting) {
                geminiBackendSetting.onchange = () => {
                    syncGeminiBackendUi();
                    if (typeof window.toggleOptions === 'function') window.toggleOptions();
                };
            }
            const adminApiModeSetting = get('set-admin-api-key-mode');
            if (adminApiModeSetting) {
                adminApiModeSetting.onchange = () => syncAdminApiKeyModeUi();
            }
            const tempChatTimeoutSetting = get('set-temp-chat-timeout-seconds');
            if (tempChatTimeoutSetting) {
                tempChatTimeoutSetting.onchange = () => {
                    applyTemporaryChatTimeoutSeconds(tempChatTimeoutSetting.value);
                };
            }

            // Wire slash command cancel button (for pending command mode)
            const slashCancelBtn = get('slash-command-cancel-btn');
            if (slashCancelBtn) {
                slashCancelBtn.onclick = () => {
                    hidePendingSlashCommandIndicator();
                    const input = get('prompt-input');
                    if (input) input.focus();
                };
            }
            syncGeminiBackendUi();
            syncAdminApiKeyModeUi();
            get('save-settings-btn').onclick = async () => {
                if (!settingsModalLoaded) {
                    showToast('設定を読み込み中です。完了するまでお待ちください', 'error', true);
                    return;
                }
                const uEl = get('set-username');
                const pEl = get('set-password');
                const promptBarMode = readPromptBarModeFromForm();
                const b = {
                    system_prompt: get('sys-prompt-text') ? get('sys-prompt-text').value : '',
                    system_prompt_enabled: get('set-global-sys-prompt-enabled') ? get('set-global-sys-prompt-enabled').checked : true,
                    apply_global_system_prompt: get('set-apply-global-sys-prompt') ? get('set-apply-global-sys-prompt').checked : true,
                    apply_auto_system_prompt_notices: get('set-apply-auto-sys-prompt-notices') ? get('set-apply-auto-sys-prompt-notices').checked : true,
                    auto_system_prompt_notices_config: collectAutoSystemPromptConfigFromForm('set'),
                    theme_color: normalizeHex(get('set-theme-color-text') ? get('set-theme-color-text').value : '') || THEME_DEFAULT,
                    light_mode_enabled: get('set-light-mode') ? get('set-light-mode').checked : false,
                    mic_transcribe_mode: get('set-mic-transcribe-mode') ? get('set-mic-transcribe-mode').value : 'stt_api',
                    stt_model: get('set-stt-model') ? get('set-stt-model').value : null,
                    llm_transcribe_prompt: get('set-llm-transcribe-prompt') ? get('set-llm-transcribe-prompt').value : '',
                    enter_to_send: get('set-enter-to-send') ? get('set-enter-to-send').checked : false,
                    math_steps_enabled: get('set-math-steps-enabled') ? get('set-math-steps-enabled').checked : false,
                    default_math_steps: get('set-default-math-steps') ? get('set-default-math-steps').checked : false,
                    compact_prompt_mode: promptBarMode.compact_prompt_mode,
                    minimal_prompt_mode: promptBarMode.minimal_prompt_mode,
                    use_sw_cache: get('set-use-sw-cache') ? get('set-use-sw-cache').checked : false,
                    clear_cache_on_version_update: get('set-clear-cache-on-version-update') ? get('set-clear-cache-on-version-update').checked : false,
                    liquid_glass_enabled: get('set-liquid-glass') ? get('set-liquid-glass').checked : false,
                    auto_search_on_links: get('set-auto-search-links') ? get('set-auto-search-links').checked : true,
                    use_last_chat_settings: get('set-use-last-settings') ? get('set-use-last-settings').checked : false,
                    voice_studio_ui: get('set-voice-studio-ui') ? get('set-voice-studio-ui').checked : true,
                    default_model: get('set-default-model') ? get('set-default-model').value : null,
                    default_vision_model: get('set-default-vision-model') ? get('set-default-vision-model').value : null,
                    temp_chat_timeout_seconds: normalizeTemporaryChatTimeoutSeconds(
                        get('set-temp-chat-timeout-seconds') ? get('set-temp-chat-timeout-seconds').value : temporaryChatTimeoutSeconds
                    ),
                    default_enable_search: get('set-default-search') ? get('set-default-search').checked : false,
                    default_gemini_free_tier: get('set-default-gemini-free-tier') ? get('set-default-gemini-free-tier').checked : false,
                    default_enable_url_context: get('set-default-url-context') ? get('set-default-url-context').checked : false,
                    default_enable_maps: get('set-default-maps') ? get('set-default-maps').checked : false,
                    default_enable_python: get('set-default-python') ? get('set-default-python').checked : false,
                    default_enable_file_creation: get('set-default-file-creation') ? get('set-default-file-creation').checked : false,
                    default_enable_thinking: get('set-default-thinking') ? get('set-default-thinking').checked : false,
                    default_enable_mcp: get('set-default-mcp') ? get('set-default-mcp').checked : true,
                    default_thinking_level: get('set-default-thinking-level') ? get('set-default-thinking-level').value : null,
                    default_thinking_budget: get('set-default-thinking-budget') ? get('set-default-thinking-budget').value : null,
                    default_reasoning_effort: get('set-default-reasoning-effort') ? get('set-default-reasoning-effort').value : null,
                    default_enable_system_prompt: get('set-default-sys-prompt') ? get('set-default-sys-prompt').checked : false,
                    default_safety_setting: get('set-default-safety') ? get('set-default-safety').value : null,
                    enable_latency_metrics: get('set-latency-metrics') ? get('set-latency-metrics').checked : false,
                    server_diagnostic_log_retention_hours: Number(get('set-server-diagnostic-log-retention-hours')?.value || 1),
                    enable_server_diagnostic_log: get('set-server-diagnostic-log') ? get('set-server-diagnostic-log').checked : false,
                    enable_client_debug_log: get('set-client-debug-log') ? get('set-client-debug-log').checked : false,
                    passkey_only_login: get('set-passkey-only-login') ? get('set-passkey-only-login').checked : false,
                    skip_2fa_on_google_login: get('set-skip-2fa-google') ? get('set-skip-2fa-google').checked : false,
                    default_2fa_method: get('set-default-2fa-method') ? get('set-default-2fa-method').value : 'totp',
                    new_username: uEl ? uEl.value : null,
                    new_password: pEl ? pEl.value : null
                };
                // Only send enable_e2ee when the checkbox actually differs from the
                // value loaded from the server.  The HTML default is "off", so an
                // unchanged form used to POST enable_e2ee=false for E2EE accounts,
                // silently starting a decrypt migration.
                const e2eeCurrent = get('set-e2ee') ? get('set-e2ee').checked : false;
                const e2eeLoaded = userSettingsSnapshot && Object.prototype.hasOwnProperty.call(userSettingsSnapshot, 'enable_e2ee')
                    ? !!userSettingsSnapshot.enable_e2ee
                    : !!(window.CHAT_CONFIG && window.CHAT_CONFIG.enableE2EE);
                if (e2eeCurrent !== e2eeLoaded) {
                    b.enable_e2ee = e2eeCurrent;
                }
                if (get('set-openai')) b.openai_key = get('set-openai').value;
                if (get('set-gemini')) b.gemini_key = get('set-gemini').value;
                if (get('set-gemini-free')) b.gemini_free_key = get('set-gemini-free').value;
                if (get('set-deepseek')) b.deepseek_key = get('set-deepseek').value;
                if (get('set-zai')) b.zai_key = get('set-zai').value;
                if (get('set-kimi')) b.kimi_key = get('set-kimi').value;
                if (get('set-mistral')) b.mistral_key = get('set-mistral').value;
                if (get('set-ideogram')) b.ideogram_key = get('set-ideogram').value;
                if (get('set-anthropic')) b.anthropic_key = get('set-anthropic').value;
                b.model_api_keys = normalizeModelApiKeyMap(modelApiKeyMap);
                if (get('set-gemini-backend')) b.gemini_backend = normalizeGeminiBackend(get('set-gemini-backend').value);
                if (get('set-gemini-vertex-project')) b.gemini_vertex_project = get('set-gemini-vertex-project').value;
                if (get('set-gemini-vertex-location')) b.gemini_vertex_location = get('set-gemini-vertex-location').value;
                if (get('set-gemini-vertex-credentials-json')) b.gemini_vertex_credentials_json = get('set-gemini-vertex-credentials-json').value;
                if (get('set-xai')) b.xai_key = get('set-xai').value;
                if (get('set-google-key')) b.google_key = get('set-google-key').value;
                if (get('set-google-project')) b.google_project = get('set-google-project').value;
                if (get('set-admin-api-key-mode')) b.admin_api_key_mode = normalizeAdminApiKeyMode(get('set-admin-api-key-mode').value);
                if (get('set-bot-detect')) b.bot_detection_enabled = get('set-bot-detect').checked;
                if (get('set-bot-detect-global')) b.bot_detection_global_enabled = get('set-bot-detect-global').checked;
                const res = await apiFetch(CHAT_CONFIG.urls.handleSettings, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(b)});
                if (res.ok) {
                    // The server returns the result message in the JSON body
                    // (e.g. the E2EE migration notice).  It no longer uses
                    // flash(), which used to leak a stale "設定を保存しました"
                    // toast onto the next page load.
                    let saveMsg = "設定を保存しました";
                    try {
                        const d = await res.json();
                        if (d && d.message) saveMsg = d.message;
                    } catch (e) {}

                    // Keep the cached settings in sync with the values just saved.
                    // Without this, reopening the modal reused the pre-save snapshot
                    // and showed both display-mode switches as OFF until a full reload.
                    cacheUserSettings(Object.assign({}, userSettingsSnapshot || {}, {
                        light_mode_enabled: !!b.light_mode_enabled,
                        liquid_glass_enabled: !!b.liquid_glass_enabled,
                        math_steps_enabled: !!b.math_steps_enabled,
                        default_math_steps: !!b.default_math_steps
                    }));
                    if (typeof window.applyMathStepsSettings === 'function') {
                        window.applyMathStepsSettings({ math_steps_enabled: !!b.math_steps_enabled });
                        if (!b.math_steps_enabled) setMathStepsEnabled(false, { persist: false });
                    }
                    window.applySavedUserSystemPromptSettings({
                        system_prompt: b.system_prompt,
                        system_prompt_enabled: b.system_prompt_enabled,
                        apply_global_system_prompt: b.apply_global_system_prompt,
                        apply_auto_system_prompt_notices: b.apply_auto_system_prompt_notices,
                        auto_system_prompt_notices_config: b.auto_system_prompt_notices_config
                    });
                    closeSettingsModal();

                    const oldUsername = currentUsername;
                    const oldE2EE = CHAT_CONFIG.enableE2EE;

                    // Update client-side variables
                    enterToSend = b.enter_to_send;
                    autoSearchOnLinks = b.auto_search_on_links;
                    const previousUseSwCache = useSwCache;
                    useSwCache = b.use_sw_cache;
                    if (window.CHAT_CONFIG) {
                        window.CHAT_CONFIG.clearCacheOnVersionUpdate = !!b.clear_cache_on_version_update;
                    }
                    compactPromptMode = b.compact_prompt_mode;
                    minimalPromptMode = b.minimal_prompt_mode;
                    voiceStudioUiEnabled = b.voice_studio_ui !== false;
                    temporaryChatTimeoutSeconds = b.temp_chat_timeout_seconds;

                    // Apply theme color
                    applyThemeColor(b.theme_color, true);
                    syncThemeInputs(b.theme_color);
                    applyLightMode(b.light_mode_enabled);
                    applyLiquidGlassMode(b.liquid_glass_enabled);
                    applyAdaptiveBlurPreference(get('set-background-blur-mode') ? get('set-background-blur-mode').value : adaptiveBlurPreferenceMode);

                    // Update UI components
                    if (minimalPromptMode) setMinimalPromptMode(true);
                    else setCompactPromptMode(compactPromptMode);
                    updateStsUi();
                    if (previousUseSwCache !== useSwCache) {
                        applyCacheMode(useSwCache, { forceCleanup: !useSwCache });
                    }

                    showToast(saveMsg, "success");
                    syncClientDebugLogToggle(b.enable_client_debug_log, 'settings saved');

                    // Critical changes that still might benefit from a reload for full consistency
                    if (b.new_username && b.new_username !== oldUsername) {
                        setTimeout(() => location.reload(), 1000);
                    } else if (b.new_password) {
                        showToast("パスワードを変更しました。次回ログイン時から有効です。", "info");
                    }
                }
                else {
                    let d = {};
                    try { d = await res.json(); } catch (e) {}
                    showToast(d.error || "設定の保存に失敗しました", "error", true);
                }
            };
            get('disable-2fa-btn').onclick = async () => {
                if(confirm("Disable 2FA?")) {
                    const res = await apiFetch(CHAT_CONFIG.urls.handleSettings, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({disable_2fa: true})});
                    if (res.ok) {
                        showToast("2FAを無効化しました", "success");
                        get('disable-2fa-btn').classList.add('hidden');
                        const badge = get('2fa-badge');
                        if (badge) {
                            badge.innerText = 'DISABLED';
                            badge.className = 'px-2 py-0.5 rounded text-xs font-bold bg-gray-700 text-gray-400';
                        }
                    } else {
                        showToast("2FAの無効化に失敗しました", "error", true);
                    }
                }
            };
            if (get('bot-unban-btn')) {
                get('bot-unban-btn').onclick = async () => {
                    const u = get('bot-unban-username');
                    const username = u ? u.value.trim() : '';
                    if (!username) { showToast('ユーザー名を入力してください', 'error', true); return; }
                    if (!confirm(`ユーザー ${username} のBANを解除しますか？`)) return;
                    const res = await apiFetch('/api/bot/unban', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({username, mode:'single'})});
                    const data = await res.json();
                    const box = get('bot-unban-result');
                    if (res.ok && data && data.status === 'ok') {
                        if (box) { box.textContent = `${username} のBANを単独解除しました`; box.classList.remove('hidden'); }
                        if (u) u.value = '';
                    } else {
                        const msg = data && data.error ? data.error : '解除に失敗しました';
                        showToast(msg, 'error', true);
                    }
                };
            }
            if (get('bot-unban-linked-btn')) {
                get('bot-unban-linked-btn').onclick = async () => {
                    const u = get('bot-unban-username');
                    const username = u ? u.value.trim() : '';
                    if (!username) { showToast('ユーザー名を入力してください', 'error', true); return; }
                    if (!confirm(`ユーザー ${username} の連鎖BANを解除しますか？`)) return;
                    const res = await apiFetch('/api/bot/unban', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({username, mode:'linked'})});
                    const data = await res.json();
                    const box = get('bot-unban-result');
                    if (res.ok && data && data.status === 'ok') {
                        if (box) { box.textContent = `${username} の連鎖BANを解除しました`; box.classList.remove('hidden'); }
                        if (u) u.value = '';
                    } else {
                        const msg = data && data.error ? data.error : '解除に失敗しました';
                        showToast(msg, 'error', true);
                    }
                };
            }
            if (get('bot-speed-test-btn')) {
                get('bot-speed-test-btn').onclick = async () => {
                    const btn = get('bot-speed-test-btn');
                    const box = get('bot-speed-test-result');
                    if (btn) btn.disabled = true;
                    if (btn) btn.classList.add('opacity-60', 'cursor-not-allowed');
                    if (box) {
                        box.classList.remove('hidden');
                        box.textContent = '実行中...';
                    }
                    try {
                        const setBox = (text) => { if (box) box.textContent = text; };
                        const cacheBust = () => `${Date.now()}_${Math.random().toString(36).slice(2)}`;
                        const toMbps = (bytes, ms) => {
                            if (!bytes || !ms || ms <= 0) return 0;
                            return (bytes * 8) / (ms / 1000) / 1000 / 1000;
                        };
                        const fmtMs = (v) => Number.isFinite(v) ? `${v.toFixed(0)} ms` : '-';
                        const fmtMbps = (v) => Number.isFinite(v) ? `${v.toFixed(v >= 100 ? 0 : 1)} Mbps` : '-';
                        const parseErr = async (res, fallback) => {
                            const data = await res.json().catch(() => ({}));
                            return (data && data.error) ? data.error : fallback;
                        };

                        const pingSamples = [];
                        setBox('測定中... ping');
                        for (let i = 0; i < 4; i++) {
                            const t0 = performance.now();
                            const res = await apiFetch(`/api/speedtest/ping?_=${cacheBust()}`, { cache: 'no-store' });
                            const t1 = performance.now();
                            if (!res.ok) throw new Error(await parseErr(res, 'ping_failed'));
                            await res.json().catch(() => ({}));
                            pingSamples.push(t1 - t0);
                        }
                        const pingAvg = pingSamples.reduce((a, b) => a + b, 0) / Math.max(1, pingSamples.length);
                        const pingMin = Math.min(...pingSamples);

                        const runDownload = async (bytes) => {
                            const t0 = performance.now();
                            const res = await apiFetch(`/api/speedtest/download?bytes=${bytes}&_=${cacheBust()}`, { cache: 'no-store' });
                            if (!res.ok) throw new Error(await parseErr(res, 'download_failed'));
                            const buf = await res.arrayBuffer();
                            const t1 = performance.now();
                            return { bytes: buf.byteLength || bytes, ms: t1 - t0, mbps: toMbps(buf.byteLength || bytes, t1 - t0) };
                        };

                        setBox(`測定中... ping ${fmtMs(pingAvg)}\n測定中... download`);
                        const dlRuns = [];
                        for (const bytes of [2 * 1024 * 1024, 8 * 1024 * 1024]) {
                            dlRuns.push(await runDownload(bytes));
                            setBox(`測定中... ping ${fmtMs(pingAvg)}\ndownload ${fmtMbps(Math.max(...dlRuns.map(x => x.mbps)))}\n測定中... upload`);
                        }
                        const downloadBest = Math.max(...dlRuns.map(x => x.mbps));

                        const runUpload = async (bytes) => {
                            const payload = new Uint8Array(bytes);
                            const t0 = performance.now();
                            const res = await apiFetch(`/api/speedtest/upload?_=${cacheBust()}`, {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/octet-stream' },
                                body: payload,
                                cache: 'no-store'
                            });
                            const t1 = performance.now();
                            if (!res.ok) throw new Error(await parseErr(res, 'upload_failed'));
                            const data = await res.json().catch(() => ({}));
                            const actualBytes = Number(data.bytes_received || bytes) || bytes;
                            return { bytes: actualBytes, ms: t1 - t0, mbps: toMbps(actualBytes, t1 - t0), serverMs: Number(data.server_elapsed_ms || 0) || 0 };
                        };

                        const ulRuns = [];
                        for (const bytes of [1 * 1024 * 1024, 4 * 1024 * 1024]) {
                            ulRuns.push(await runUpload(bytes));
                        }
                        const uploadBest = Math.max(...ulRuns.map(x => x.mbps));

                        const lines = [
                            '結果 (ブラウザ⇔このサーバー)',
                            `Ping (avg/min): ${fmtMs(pingAvg)} / ${fmtMs(pingMin)}`,
                            `Download (best): ${fmtMbps(downloadBest)}`,
                            `Upload (best): ${fmtMbps(uploadBest)}`,
                            `Download runs: ${dlRuns.map(x => `${Math.round(x.bytes / 1024 / 1024)}MB=${fmtMbps(x.mbps)}`).join(', ')}`,
                            `Upload runs: ${ulRuns.map(x => `${Math.round(x.bytes / 1024 / 1024)}MB=${fmtMbps(x.mbps)}`).join(', ')}`,
                            '注記: fast.com のようなインターネット全体の速度ではなく、このアプリサーバーまでの回線速度の目安です。'
                        ];
                        setBox(lines.join('\n'));
                        showToast('回線速度テストを実行しました', 'success');
                    } catch (e) {
                        if (box) box.textContent = `エラー: ${e && e.message ? e.message : '回線速度テストに失敗しました'}`;
                        showToast('回線速度テストに失敗しました', 'error', true);
                    } finally {
                        if (btn) {
                            btn.disabled = false;
                            btn.classList.remove('opacity-60', 'cursor-not-allowed');
                        }
                    }
                };
            }
            if (get('ban-appeal-refresh')) {
                get('ban-appeal-refresh').onclick = () => loadBanAppeals();
            }
            if (get('ban-appeal-mark-read')) {
                get('ban-appeal-mark-read').onclick = () => markBanAppealsRead();
            }
            if (get('ban-appeal-list')) {
                get('ban-appeal-list').addEventListener('click', async (e) => {
                    const btn = e.target.closest('button');
                    if (!btn) return;
                    const id = btn.getAttribute('data-id');
                    if (btn.classList.contains('ban-appeal-mark')) {
                        if (id) await markBanAppealsRead([Number(id)]);
                        return;
                    }
                    if (btn.classList.contains('ban-appeal-status')) {
                        const status = btn.getAttribute('data-status');
                        if (id && status) await updateBanAppealStatus({ id: Number(id), status });
                        return;
                    }
                    if (btn.classList.contains('ban-appeal-reply-send')) {
                        const card = btn.closest('[data-appeal-id]');
                        const textarea = card ? card.querySelector('.ban-appeal-reply') : null;
                        const reply = textarea ? textarea.value : '';
                        if (id) await updateBanAppealStatus({ id: Number(id), admin_reply: reply });
                        return;
                    }
                    if (btn.classList.contains('ban-appeal-block')) {
                        if (!confirm('このユーザーの異議申し立てをブロックしますか？')) return;
                        const reason = prompt('ブロック理由 (任意)') || '';
                        if (id) await updateBanAppealStatus({ id: Number(id), block_user: true, block_reason: reason });
                        return;
                    }
                });
            }
            if (get('upload-modal-close')) get('upload-modal-close').onclick = () => closeUploadModal();
            if (get('upload-select-btn')) get('upload-select-btn').onclick = () => get('file-input').click();
            if (get('upload-camera-btn')) get('upload-camera-btn').onclick = () => openCameraCaptureModal();
            if (get('upload-photo-btn')) get('upload-photo-btn').onclick = () => get('photo-input').click();
            if (get('camera-modal-close')) get('camera-modal-close').onclick = () => closeCameraCaptureModal();
            if (get('camera-capture-btn')) get('camera-capture-btn').onclick = () => captureCameraShot();
            if (get('camera-attach-btn')) get('camera-attach-btn').onclick = () => attachCameraCapturedFiles();
            if (get('camera-switch-btn')) get('camera-switch-btn').onclick = () => toggleCameraCaptureFacing();
            if (get('camera-clear-btn')) get('camera-clear-btn').onclick = () => resetCameraCapturePending();
            if (get('camera-fallback-btn')) {
                get('camera-fallback-btn').onclick = () => {
                    closeCameraCaptureModal();
                    const photoInput = get(get('camera-split-enabled').checked ? 'image-split-input' : 'photo-input');
                    if (photoInput) photoInput.click();
                };
            }
            if (get('upload-clear-btn')) {
                get('upload-clear-btn').onclick = () => {
                    resetUploadState();
                };
            }
            if (get('marker-modal-close')) {
                get('marker-modal-close').onclick = () => {
                    closeMarkerModal();
                    markerState.row = null;
                };
            }
            if (get('marker-tool-draw')) get('marker-tool-draw').onclick = () => setMarkerMode('draw');
            if (get('marker-tool-auto')) get('marker-tool-auto').onclick = () => setMarkerMode('auto');
            if (get('marker-tool-mosaic')) get('marker-tool-mosaic').onclick = () => setMarkerMode('mosaic');
            if (get('marker-tool-crop')) get('marker-tool-crop').onclick = () => setMarkerMode('crop');
            if (get('marker-auto-tolerance')) get('marker-auto-tolerance').oninput = (e) => {
                const label = get('marker-auto-tolerance-value');
                if (label) label.textContent = String(e.target.value);
            };
            const markerColorPicker = get('marker-color-picker');
            if (markerColorPicker) {
                markerColorPicker.oninput = (e) => setMarkerColor(e.target.value);
                markerColorPicker.onchange = (e) => setMarkerColor(e.target.value);
            }
            const markerOpacity = get('marker-opacity');
            if (markerOpacity) {
                markerOpacity.oninput = (e) => setMarkerOpacity(e.target.value);
                markerOpacity.onchange = (e) => setMarkerOpacity(e.target.value);
            }
            const markerOpacityNumber = get('marker-opacity-number');
            if (markerOpacityNumber) {
                markerOpacityNumber.onchange = (e) => setMarkerOpacity(e.target.value);
                markerOpacityNumber.onblur = (e) => setMarkerOpacity(e.target.value);
                markerOpacityNumber.onkeydown = (e) => {
                    if (e.key === 'Enter') {
                        setMarkerOpacity(e.target.value);
                        e.target.blur();
                    }
                };
            }
            document.querySelectorAll('#marker-toolbar .marker-color-chip[data-marker-color]').forEach((chip) => {
                chip.onclick = () => setMarkerColor(chip.getAttribute('data-marker-color'));
            });
            if (get('marker-view-reset')) get('marker-view-reset').onclick = () => resetMarkerTransform();
            if (get('marker-crop-reset')) get('marker-crop-reset').onclick = () => clearCropRect();
            if (get('marker-undo')) get('marker-undo').onclick = () => undoMarkerCanvas();
            if (get('marker-clear')) get('marker-clear').onclick = () => clearMarkerCanvas();
            if (get('marker-save')) get('marker-save').onclick = () => saveMarkerToRow();
            syncMarkerColorControls();
            initMarkerCanvas();
            initCropCanvas();
            window.addEventListener('resize', () => {
                const markerModal = get('marker-modal');
                if (!markerModal || markerModal.classList.contains('hidden')) return;
                applyMarkerTransform();
                renderCropOverlay();
            });
            const isUploadModalOpen = () => {
                const uploadModal = get('upload-modal');
                return !!(uploadModal && !uploadModal.classList.contains('hidden'));
            };
            const dropOverlay = get('drop-overlay');
            let dragCounter = 0;
            const showDropOverlay = () => {
                if (isUploadModalOpen()) return;
                if (dropOverlay) {
                    dropOverlay.classList.remove('hidden');
                    dropOverlay.classList.add('flex');
                }
            };
            const hideDropOverlay = () => {
                dragCounter = 0;
                if (dropOverlay) {
                    dropOverlay.classList.add('hidden');
                    dropOverlay.classList.remove('flex');
                }
            };
            window.hideDropOverlay = hideDropOverlay;
            const dropzone = get('upload-dropzone');
            if (dropzone) {
                dropzone.addEventListener('dragover', (e) => {
                    e.preventDefault();
                    dropzone.classList.add('dragover');
                });
                dropzone.addEventListener('dragleave', () => {
                    dropzone.classList.remove('dragover');
                });
                dropzone.addEventListener('drop', (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    dropzone.classList.remove('dragover');
                    hideDropOverlay();
                    const files = e.dataTransfer ? e.dataTransfer.files : null;
                    if (files && files.length) handleFiles(files);
                });
            }
            window.addEventListener('dragenter', (e) => {
                if (!e.dataTransfer || !e.dataTransfer.types || !e.dataTransfer.types.includes('Files')) return;
                dragCounter += 1;
                showDropOverlay();
            });
            window.addEventListener('dragover', (e) => {
                if (!e.dataTransfer || !e.dataTransfer.types || !e.dataTransfer.types.includes('Files')) return;
                e.preventDefault();
            });
            window.addEventListener('dragleave', (e) => {
                if (!e.dataTransfer || !e.dataTransfer.types || !e.dataTransfer.types.includes('Files')) return;
                dragCounter = Math.max(0, dragCounter - 1);
                if (dragCounter === 0 || !e.relatedTarget || e.clientY <= 0 || e.clientX <= 0 || e.clientX >= window.innerWidth || e.clientY >= window.innerHeight) {
                    hideDropOverlay();
                }
            });
            window.addEventListener('dragend', () => {
                hideDropOverlay();
            });
            window.addEventListener('drop', (e) => {
                hideDropOverlay();
                if (!e.dataTransfer || !e.dataTransfer.files || e.dataTransfer.files.length === 0) return;
                e.preventDefault();
                if (dropzone && dropzone.contains(e.target)) return;
                handleFiles(e.dataTransfer.files);
            });
            const botAdminModal = get('bot-admin-modal');
            const renderBotUsers = (users) => {
                const list = get('bot-admin-list');
                if (!list) return;
                list.innerHTML = '';
                if (!users || !users.length) {
                    list.innerHTML = '<div class="text-xs text-gray-400">該当ユーザーがいません。</div>';
                    return;
                }
                users.forEach((u, idx) => {
                    const isBanned = !!u.is_bot_banned;
                    const detOn = u.bot_detection_enabled !== false;
                    const lockRemaining = Number(u.lock_remaining_seconds) || 0;
                    const row = document.createElement('div');
                    row.className = 'flex flex-wrap items-center gap-2 bg-gray-900 border border-gray-700 rounded p-2 text-xs model-list-animate';
                    row.style.animationDelay = `${Math.min(idx, 12) * 0.02}s`;
                    row.innerHTML = `
                        <div class="flex-1 min-w-0">
                            <div class="text-gray-200 font-bold bot-log-wrap">${escapeHtml(u.username)}</div>
                            <div class="text-[10px] text-gray-500">${isBanned ? 'BAN中' : '正常'}${lockRemaining ? `・ロック中（残り${Math.ceil(lockRemaining / 60)}分）` : ''} ${u.bot_ban_reason ? ' / ' + escapeHtml(u.bot_ban_reason) : ''}</div>
                        </div>
                        <button class="bot-toggle-detect bg-gray-700 hover:bg-gray-600 text-white px-2 py-1 rounded" data-user="${escapeHtml(u.username)}" data-enabled="${detOn ? '1' : '0'}">${detOn ? '検出ON' : '検出OFF'}</button>
                        <button class="bot-toggle-ban ${isBanned ? 'bg-green-600 hover:bg-green-500' : 'bg-red-600 hover:bg-red-500'} text-white px-2 py-1 rounded" data-user="${escapeHtml(u.username)}" data-banned="${isBanned ? '1' : '0'}">${isBanned ? '単独解除' : 'BAN'}</button>                        ${isBanned ? `<button class=\"bot-toggle-unban-linked bg-rose-600 hover:bg-rose-500 text-white px-2 py-1 rounded\" data-user=\"${escapeHtml(u.username)}\">連鎖解除</button>` : ''}
                        ${u.is_admin ? '' : `<button class="bot-delete-account bg-red-800 hover:bg-red-700 text-white px-2 py-1 rounded" data-progress-expected-slow="true" data-user="${escapeHtml(u.username)}">削除</button>`}
                    `;
                    list.appendChild(row);
                });
            };
            const loadBotUsers = async (q = '') => {
                const list = get('bot-admin-list');
                if (list) {
                    list.innerHTML = '<div class="text-xs text-gray-400 py-2"><i class="fas fa-spinner fa-spin mr-1"></i>読み込み中...</div>';
                }
                try {
                    const res = await apiFetch(`/api/bot/users?q=${encodeURIComponent(q)}`);
                    const data = await res.json();
                    if (res.ok && data && data.users) renderBotUsers(data.users);
                    else {
                        if (list) list.innerHTML = '<div class="text-xs text-red-400">ユーザー一覧の取得に失敗しました。</div>';
                        showToast('ユーザー一覧取得に失敗しました', 'error', true);
                    }
                } catch (err) {
                    if (list) list.innerHTML = '<div class="text-xs text-red-400">ユーザー一覧の取得に失敗しました。</div>';
                    showToast('ユーザー一覧取得に失敗しました', 'error', true);
                }
            };
            const openBotAdminModal = async () => {
                if (!isAdminUser) return;
                const modal = get('bot-admin-modal') || botAdminModal;
                if (!modal) return;
                // 設定モーダルの上に重ねると背面オーバーレイで開閉アニメが見えにくいため、
                // 設定を先に閉じてからアカウント管理を開く（履歴は /settings を残し、閉じたときに戻れる）
                const settingsEl = get('settings-modal');
                if (settingsEl && (settingsEl.classList.contains('modal-open') || settingsEl.classList.contains('modal-prep'))) {
                    hideModal('settings-modal');
                }
                showModal('bot-admin-modal');
                if (location.pathname !== '/admin-bots') {
                    history.pushState({ modal: 'admin-bots' }, '', '/admin-bots');
                }
                await loadBotUsers(get('bot-admin-search') ? get('bot-admin-search').value.trim() : '');
            };
