
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
            if (
                cat.includes('mistral') ||
                id.includes('mistral') ||
                name.includes('mistral') ||
                desc.includes('mistral') ||
                id.includes('ocr') ||
                cat.includes('ocr')
            ) tags.push('mistral');
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
                cat.includes('speech') ||
                id.includes('tts') ||
                name.includes('tts') ||
                desc.includes('tts') ||
                id.includes('realtime') ||
                id.includes('live') ||
                id.includes('voice-agent') ||
                id.includes('native-audio') ||
                name.includes('audio') ||
                desc.includes('audio')
            ) tags.push('audio');
            if (id.includes('reasoning') || name.includes('reasoning') || desc.includes('reasoning')) tags.push('reasoning');
            if ((cat.includes('deepseek') || id.includes('deepseek') || name.includes('deepseek')) && !tags.includes('reasoning')) tags.push('reasoning');
            if (id.includes('fast') || name.includes('fast') || desc.includes('fast') || cat.includes('fast')) tags.push('fast');
            if ((id.includes('deepseek-v4-flash') || (cat.includes('deepseek') && name.includes('flash'))) && !tags.includes('fast')) tags.push('fast');
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
                b.className = `model-tag-btn px-2 py-1 text-[10px] rounded border transition ${active ? 'bg-blue-600/20 border-blue-500 text-blue-300' : 'bg-gray-800 border-gray-700 text-gray-300 hover:border-gray-500'}`;
            });
        }

        const modelListGroups = [];
        let modelListBanner = null;
        let modelListEmpty = null;
        let modelListBuilt = false;
        let modelListAnimated = false;
        let modelListRenderFrame = 0;
        function buildModelList() {
            const container = get('model-list-container');
            if (!container || modelListBuilt) return;
            container.innerHTML = '';
            modelListBanner = document.createElement('div');
            modelListBanner.className = 'hidden mb-4 px-3 py-2 rounded-lg border border-teal-500/40 bg-teal-900/20 text-[11px] text-teal-200';
            container.appendChild(modelListBanner);

            MODELS.forEach(group => {
                const availableItems = group.items.filter(m => !m.deprecated);
                if (!availableItems.length) return;
                const groupEl = document.createElement('section');
                groupEl.className = 'model-list-group';
                groupEl.innerHTML = `
                    <div class="flex items-center gap-2 mb-3 px-2">
                        <i class="${group.icon}"></i>
                        <div>
                            <h3 class="font-bold text-gray-200 text-sm">${group.category}</h3>
                            <p class="text-[10px] text-gray-500">${group.description}</p>
                        </div>
                    </div>
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-2 mb-6"></div>
                `;
                const grid = groupEl.querySelector('.grid');
                const entries = availableItems.map(m => {
                    const item = document.createElement('button');
                    const apiModelName = String(m.apiId || m.id || '').trim();
                    const agenticViewHtml = m.agenticView ? `<span class="inline-flex items-center gap-1 rounded-full border border-teal-500/40 bg-teal-900/20 px-2 py-0.5 text-[9px] font-semibold text-teal-200 whitespace-nowrap" title="Agentic View対応：画像をクロップして再観察しながら推論を継続できます"><i class="fas fa-eye" aria-hidden="true"></i>Agentic View</span>` : '';
                    const apiModelHtml = apiModelName ? `<div class="text-[10px] text-cyan-300/90 mt-1.5 font-mono break-all"><span class="font-sans text-gray-500 mr-1">API model:</span>${escapeHtml(apiModelName)}</div>` : '';
                    const priceHtml = m.price ? `<div class="text-[10px] text-amber-400/90 mt-1.5 font-mono flex items-start gap-1"><i class="fas fa-tag text-[9px] mt-0.5 opacity-70 shrink-0"></i><span>${m.price}</span></div>` : '';
                    item.type = 'button';
                    item.className = 'flex flex-col text-left p-3 rounded-lg border transition bg-gray-800 border-gray-700 hover:border-gray-500 hover:bg-gray-750';
                    item.dataset.selected = '0';
                    item.onclick = () => selectModel(m.id, m.name);
                    item.innerHTML = `
                        <div class="flex justify-between items-start gap-2 w-full mb-1">
                            <div class="flex flex-wrap items-center gap-2 min-w-0">
                                <span class="font-bold text-sm text-gray-200">${m.name}</span>
                                ${agenticViewHtml}
                            </div>
                            <i class="model-selected-icon fas fa-check-circle text-blue-400 hidden shrink-0 mt-0.5"></i>
                        </div>
                        <span class="text-[10px] text-gray-400">${m.desc}</span>
                        ${apiModelHtml}
                        ${priceHtml}
                    `;
                    grid.appendChild(item);
                    return {
                        model: m,
                        button: item,
                        searchText: `${m.name} ${m.id} ${apiModelName} ${m.agenticView ? 'agentic view' : ''}`.toLowerCase(),
                        provider: getModelApiProvider(m.id),
                        tags: new Set(getModelTags(m, group)),
                    };
                });
                modelListGroups.push({ element: groupEl, entries });
                container.appendChild(groupEl);
            });

            modelListEmpty = document.createElement('div');
            modelListEmpty.className = 'hidden text-center text-gray-500 py-8';
            container.appendChild(modelListEmpty);
            modelListBuilt = true;
        }

        function updateModelButtonSelection(entry, selectedModel) {
            const isSelected = selectedModel === entry.model.id;
            if (entry.button.dataset.selected === (isSelected ? '1' : '0')) return;
            entry.button.dataset.selected = isSelected ? '1' : '0';
            entry.button.classList.toggle('bg-blue-600/20', isSelected);
            entry.button.classList.toggle('border-blue-500', isSelected);
            entry.button.classList.toggle('ring-1', isSelected);
            entry.button.classList.toggle('ring-blue-500', isSelected);
            entry.button.classList.toggle('bg-gray-800', !isSelected);
            entry.button.classList.toggle('border-gray-700', !isSelected);
            entry.button.classList.toggle('hover:border-gray-500', !isSelected);
            entry.button.classList.toggle('hover:bg-gray-750', !isSelected);
            const icon = entry.button.querySelector('.model-selected-icon');
            if (icon) icon.classList.toggle('hidden', !isSelected);
        }

        function renderModelList(filter = "", options = {}) {
            const container = get('model-list-container');
            if (!container) return;
            buildModelList();
            const f = filter.toLowerCase();
            const lockedProvider = window._visionPickerActive ? null : getPromptCacheLockedProvider();
            const lockedLabel = lockedProvider ? (PROVIDER_LABELS[lockedProvider] || lockedProvider) : '';
            const selectedModel = get('model-select') ? get('model-select').value : '';
            let visibleCount = 0;

            modelListBanner.classList.toggle('hidden', !lockedProvider);
            if (lockedProvider) {
                modelListBanner.innerHTML = `<i class="fas fa-database mr-1.5"></i>PromptCache 有効中: <strong>${lockedLabel}</strong> のモデルのみ選択できます（他APIへの切替は不可）`;
            }

            modelListGroups.forEach(group => {
                let groupVisibleCount = 0;
                group.entries.forEach(entry => {
                    const visible = entry.searchText.includes(f)
                        && (!lockedProvider || entry.provider === lockedProvider)
                        && (activeModelTag === 'all' || entry.tags.has(activeModelTag));
                    entry.button.classList.toggle('hidden', !visible);
                    updateModelButtonSelection(entry, selectedModel);
                    if (visible) groupVisibleCount += 1;
                });
                group.element.classList.toggle('hidden', groupVisibleCount === 0);
                visibleCount += groupVisibleCount;
            });

            modelListEmpty.classList.toggle('hidden', visibleCount !== 0);
            if (visibleCount === 0) {
                modelListEmpty.textContent = lockedProvider
                    ? `No ${lockedLabel} models found.`
                    : 'No models found.';
            }
            if (options.animate && !modelListAnimated) {
                modelListAnimated = true;
                container.classList.add('model-list-animate');
            }
        }

        function scheduleModelListRender(filter) {
            if (modelListRenderFrame) cancelAnimationFrame(modelListRenderFrame);
            modelListRenderFrame = requestAnimationFrame(() => {
                modelListRenderFrame = 0;
                renderModelList(filter);
            });
        }

        function openModelModal() {
            if (location.pathname !== '/model') {
                history.pushState({ modal: 'model' }, '', '/model');
            }
            const search = get('model-search');
            if (search) search.value = '';
            updateModelTagUi();
            // Build/update while hidden so opening animation never competes with DOM construction.
            renderModelList('', { animate: true });
            showModal('model-modal');
            // Prevent auto-focus on mobile to avoid keyboard popup
            if (search && window.innerWidth > 768) {
                requestAnimationFrame(() => search.focus({ preventScroll: true }));
            }
        }
        window.closeModelModal = (skipHistory = false) => {
            hideModal('model-modal');
            if (!skipHistory && location.pathname === '/model') {
                history.back();
            }
        };

        function selectModel(id, name) {
            if (window._visionPickerActive) {
                currentVisionModel = id;
                window._visionPickerActive = false;
                window.closeModelModal();
                _syncVisionModelDisplay();
                return;
            }
            if (isPromptCacheEnabled()) {
                const currentProvider = getModelApiProvider(get('model-select') ? get('model-select').value : '');
                const nextProvider = getModelApiProvider(id);
                if (currentProvider && nextProvider && currentProvider !== nextProvider) {
                    const curLabel = PROVIDER_LABELS[currentProvider] || currentProvider;
                    const nextLabel = PROVIDER_LABELS[nextProvider] || nextProvider;
                    showToast(`PromptCache 有効中は他API（${nextLabel}）のモデルに変更できません。現在: ${curLabel}`, 'warning', true);
                    return;
                }
            }
            const el = get('model-select');
            el.value = id;
            get('model-selector-text').innerText = name;
            window.closeModelModal();
            const event = new Event('change');
            el.dispatchEvent(event);
        }
        function selectModelById(id) {
            let name = id;
            for (const g of MODELS) {
                const found = g.items.find(i => i.id === id);
                if (found) { name = found.name; break; }
            }
            selectModel(id, name);
        }

        function populateAiSafeFormFields(d) {
            // Mirror a subset of the population in openSettingsModal for live update after AI apply
            if (!d) return;
            try {
                if (get('set-default-model')) get('set-default-model').value = d.default_model || get('set-default-model').value;
                if (get('set-default-vision-model')) get('set-default-vision-model').value = d.default_vision_model || 'gemini-3-flash-preview';
                if (get('set-default-search')) get('set-default-search').checked = !!d.default_enable_search;
                if (get('set-default-url-context')) get('set-default-url-context').checked = !!d.default_enable_url_context;
                if (get('set-default-maps')) get('set-default-maps').checked = !!d.default_enable_maps;
                if (get('set-default-python')) get('set-default-python').checked = !!d.default_enable_python;
                if (get('set-default-file-creation')) get('set-default-file-creation').checked = !!d.default_enable_file_creation;
                if (get('set-default-thinking')) get('set-default-thinking').checked = !!d.default_enable_thinking;
                if (get('set-default-sys-prompt')) get('set-default-sys-prompt').checked = !!d.default_enable_system_prompt;
                if (get('set-default-thinking-level')) get('set-default-thinking-level').value = d.default_thinking_level || 'high';
                if (get('set-default-thinking-budget')) get('set-default-thinking-budget').value = d.default_thinking_budget || 4096;
                if (get('set-default-reasoning-effort')) get('set-default-reasoning-effort').value = d.default_reasoning_effort || 'medium';
                if (get('set-default-safety')) get('set-default-safety').value = d.default_safety_setting || 'default';
                if (get('sys-prompt-text')) get('sys-prompt-text').value = d.system_prompt || '';
                if (get('set-global-sys-prompt-enabled')) get('set-global-sys-prompt-enabled').checked = d.system_prompt_enabled !== false;
                if (get('set-apply-global-sys-prompt')) get('set-apply-global-sys-prompt').checked = d.apply_global_system_prompt !== false;
                if (get('set-apply-auto-sys-prompt-notices')) get('set-apply-auto-sys-prompt-notices').checked = d.apply_auto_system_prompt_notices !== false;
                if (get('set-mic-transcribe-mode')) get('set-mic-transcribe-mode').value = d.mic_transcribe_mode || 'stt_api';
                if (get('set-stt-model')) get('set-stt-model').value = d.stt_model || 'gpt-4o-mini-transcribe';
                if (get('set-llm-transcribe-prompt')) get('set-llm-transcribe-prompt').value = d.llm_transcribe_prompt || '';
                if (get('set-enter-to-send')) get('set-enter-to-send').checked = !!d.enter_to_send;
                if (get('set-compact-prompt-mode') || get('set-minimal-prompt-mode') || get('set-prompt-bar-mode-normal')) {
                    writePromptBarModeToForm(!!d.compact_prompt_mode, !!d.minimal_prompt_mode);
                }
                if (d.minimal_prompt_mode) setMinimalPromptMode(true);
                else if (Object.prototype.hasOwnProperty.call(d, 'compact_prompt_mode') || Object.prototype.hasOwnProperty.call(d, 'minimal_prompt_mode')) {
                    setCompactPromptMode(!!d.compact_prompt_mode);
                }
                if (get('set-use-sw-cache')) get('set-use-sw-cache').checked = !!d.use_sw_cache;
                if (get('set-liquid-glass')) get('set-liquid-glass').checked = !!d.liquid_glass_enabled;
                applyLiquidGlassMode(!!d.liquid_glass_enabled);
                if (get('set-auto-search-links')) get('set-auto-search-links').checked = d.auto_search_on_links !== false;
                if (get('set-use-last-settings')) get('set-use-last-settings').checked = !!d.use_last_chat_settings;
                if (get('set-voice-studio-ui')) get('set-voice-studio-ui').checked = d.voice_studio_ui !== false;
                if (get('set-latency-metrics')) get('set-latency-metrics').checked = !!d.enable_latency_metrics;
                if (get('set-client-debug-log')) syncClientDebugLogToggle(!!d.enable_client_debug_log, 'ai-settings');
                if (get('set-bot-detect')) get('set-bot-detect').checked = d.bot_detection_enabled !== false;
                if (get('set-skip-2fa-google')) get('set-skip-2fa-google').checked = !!d.skip_2fa_on_google_login;
                if (get('set-default-2fa-method')) get('set-default-2fa-method').value = d.default_2fa_method || 'totp';
                // theme etc handled elsewhere if needed
            } catch (e) { /* element missing ok */ }
        }

        if (get('model-search')) {
            get('model-search').addEventListener('input', (e) => scheduleModelListRender(e.target.value));
        }
        if (get('model-tag-bar')) {
            get('model-tag-bar').addEventListener('click', (e) => {
                const btn = e.target.closest('.model-tag-btn');
                if (!btn) return;
                const t = btn.innerText.trim().toLowerCase();
                activeModelTag = MODEL_TAGS.includes(t) ? t : 'all';
                updateModelTagUi();
                renderModelList(get('model-search').value);
            });
            updateModelTagUi();
        }

        window.quickStart = (m) => {
            selectModelById(m);
            get('welcome-screen').classList.add('hidden');
        };

        const BROWSER_FAST_DISABLED_OPTIONS = [
            ['enable-search', 'search-container'],
            ['enable-url-context', 'url-context-container'],
            ['enable-maps', 'maps-grounding-container'],
            ['enable-sys-prompt', 'sys-prompt-option'],
            ['enable-prompt-cache', 'prompt-cache-container'],
        ];

        function applyBrowserFastModeRestrictions() {
            if (!browserFastModeEnabled) return;
            if (!browserFastPreviousOptions) {
                browserFastPreviousOptions = {
                    checks: Object.fromEntries(BROWSER_FAST_DISABLED_OPTIONS.map(([id]) => [id, !!(get(id) && get(id).checked)])),
                    coding: !!codingModeEnabled,
                };
            }
            BROWSER_FAST_DISABLED_OPTIONS.forEach(([id, containerId]) => {
                const checkbox = get(id);
                const container = get(containerId);
                if (checkbox) {
                    checkbox.checked = false;
                    checkbox.disabled = true;
                }
                if (container) container.classList.add('opacity-50', 'pointer-events-none');
            });
            if (codingModeEnabled) syncCodingModeUi(false, { persist: false });
            const codingCheckbox = get('enable-coding-mode');
            const codingContainer = get('coding-mode-container');
            if (codingCheckbox) codingCheckbox.disabled = true;
            if (codingContainer) codingContainer.classList.add('opacity-50', 'pointer-events-none');
            refreshMinimalOptionsIfOpen();
        }

        function restoreBrowserFastModeOptions() {
            const previous = browserFastPreviousOptions;
            if (!previous) return;
            BROWSER_FAST_DISABLED_OPTIONS.forEach(([id, containerId]) => {
                const checkbox = get(id);
                const container = get(containerId);
                if (checkbox) {
                    checkbox.disabled = false;
                    if (previous && previous.checks && Object.prototype.hasOwnProperty.call(previous.checks, id)) {
                        checkbox.checked = !!previous.checks[id];
                    }
                }
                if (container) container.classList.remove('opacity-50', 'pointer-events-none');
            });
            const codingCheckbox = get('enable-coding-mode');
            const codingContainer = get('coding-mode-container');
            if (codingCheckbox) codingCheckbox.disabled = false;
            if (codingContainer) codingContainer.classList.remove('opacity-50', 'pointer-events-none');
            if (previous && previous.coding) syncCodingModeUi(true, { persist: false });
            browserFastPreviousOptions = null;
            if (typeof updatePromptCacheUi === 'function') updatePromptCacheUi();
            refreshMinimalOptionsIfOpen();
        }

        function setBrowserFastModeEnabled(enabled, opts = {}) {
            browserFastModeEnabled = !!enabled;
            const toggle = get('enable-browser-fast-mode');
            if (toggle) toggle.checked = browserFastModeEnabled;
            const container = get('browser-fast-mode-container');
            if (container) {
                container.classList.toggle('ring-1', browserFastModeEnabled);
                container.classList.toggle('ring-amber-300', browserFastModeEnabled);
            }
            if (!browserFastModeEnabled && opts.clearKey !== false) {
                browserFastApiKey = '';
                browserFastApiKeyModel = '';
                browserFastBootstrap = null;
            }
            if (browserFastModeEnabled) applyBrowserFastModeRestrictions();
            else if (opts.restoreOptions !== false) restoreBrowserFastModeOptions();
        }

        function openBrowserFastModeModal(showWarning = true) {
            const warning = get('browser-fast-mode-warning');
            const ignoreRow = get('browser-fast-mode-ignore-row');
            if (warning) warning.classList.toggle('hidden', !showWarning);
            if (ignoreRow) ignoreRow.classList.toggle('hidden', !showWarning);
            const description = get('browser-fast-mode-key-description');
            const model = String(get('model-select') ? get('model-select').value : 'Gemini');
            if (description) description.textContent = `${model} のモデル別キー → 共通Geminiキーの順に、サーバーから自動取得します。`;
            showModal('browser-fast-mode-modal');
        }

        function browserFastBootstrapMatches(data, model, threadId, parentId) {
            if (!data || data.model !== model) return false;
            if (String(data.thread_id || '') !== String(threadId || '')) return false;
            return String(data.parent_id || '') === String(parentId || '');
        }

        async function fetchBrowserFastBootstrap(force = false) {
            const model = String(get('model-select') ? get('model-select').value : '').trim();
            const threadId = currentThreadId || null;
            const parentId = threadId ? (currentParentId || null) : null;
            if (!force && browserFastBootstrapMatches(browserFastBootstrap, model, threadId, parentId) && browserFastApiKey) {
                return browserFastBootstrap;
            }
            const response = await apiFetch('/api/browser_fast_mode/bootstrap', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model, thread_id: threadId, parent_id: parentId }),
            });
            const data = await response.json().catch(() => ({}));
            if (!response.ok || !data.api_key) {
                throw new Error(data.error || 'サーバー保存済みのGemini APIキーを取得できませんでした');
            }
            browserFastApiKey = String(data.api_key);
            browserFastApiKeyModel = model;
            browserFastBootstrap = data;
            return data;
        }

        async function requestBrowserFastModeEnable() {
            const model = String(get('model-select') ? get('model-select').value : '').toLowerCase();
            if (!model.startsWith('gemini-') || /(image|native-audio|tts|live)/.test(model)) {
                showToast('高速モードはGeminiテキストモデル専用です', 'warning', true);
                setBrowserFastModeEnabled(false);
                return;
            }
            if (currentImageUrls.length || uploadProgressState.active > 0 || browserFastLocalFiles.size) {
                showToast('高速モードへ切り替える前に添付ファイルをクリアしてください', 'warning', true);
                setBrowserFastModeEnabled(false);
                return;
            }
            const warningIgnored = (() => {
                try { return localStorage.getItem(BROWSER_FAST_IGNORE_WARNING_STORAGE) === '1'; } catch (e) { return false; }
            })();
            if (warningIgnored) {
                try {
                    await fetchBrowserFastBootstrap(true);
                    setBrowserFastModeEnabled(true, { clearKey: false });
                    showToast('高速モードを有効にしました', 'warning', false);
                } catch (error) {
                    setBrowserFastModeEnabled(false);
                    showToast(error.message || '高速モードを有効化できませんでした', 'error', true);
                }
                return;
            }
            openBrowserFastModeModal(!warningIgnored);
        }

        // Critical UI initializations - independent listener to survive errors in main init
        document.addEventListener('DOMContentLoaded', () => {
            if (get('menu-btn')) {
                get('menu-btn').onclick = () => { get('sidebar').classList.toggle('open'); get('overlay').classList.toggle('active'); };
            }
            if (get('overlay')) {
                get('overlay').onclick = () => { get('sidebar').classList.remove('open'); get('overlay').classList.remove('active'); };
            }
        });
