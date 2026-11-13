        function getModelCapabilitySearchTerms(m) {
            const model = String(m.id || '').toLowerCase();
            const terms = [];
            const isDeepSeek = model.includes('deepseek');
            const isTts = model.includes('tts');
            const isOcr = model.startsWith('mistral-ocr');
            const isNonTextModel = isTts || isOcr || model.startsWith('ideogram-') || model.includes('transcribe') || model.includes('realtime')
                || model.includes('voice-agent') || model.includes('native-audio') || model.includes('live')
                || model.includes('image') || model.includes('video') || (model.includes('gemini') && model.includes('nano')) || model.startsWith('veo-')
                || model.includes('omni-flash') || model.startsWith('lyria-') || model.includes('embedding');
            const isLlm = !isNonTextModel && (
                model.includes('gpt') || model.includes('gemini') || model.includes('grok') || isDeepSeek || model.startsWith('glm-')
                || model.startsWith('deep-research-') || model.startsWith('antigravity-')
            );
            const add = (...values) => values.forEach(value => terms.push(value, value.replace(/-/g, ' ')));

            if (model === 'gemini-nano-banana-2.1' || model.includes('gemini-3.1-flash-image') || model.includes('gemini-3-pro-image') || model.includes('gemini-2.5-flash-image')) {
                add('image generation', 'image editing');
            }
            if (model === 'gemini-nano-banana-2.1') {
                add('thinking', '思考', 'minimal', 'medium', 'high', 'thinking level', 'video input');
            } else if (model === 'gemini-3.1-flash-lite-image') {
                add('thinking', '思考', 'minimal', 'high', 'thinking level');
            } else if (model === 'gemini-3.1-flash-image') {
                add('thinking', '思考', 'minimal', 'high', 'thinking level');
            } else if (model.includes('gemini') && !isNonTextModel) {
                add('thinking', '思考', 'thinking level');
                if (model === 'gemini-4-hydrogen') {
                    add('low', 'medium', 'high', 'xhigh');
                } else if (model === 'gemini-4-carbon' || model === 'gemini-4-argon' || model === 'gemini-3.8-flash' || model === 'gemini-3.7-flash') {
                    add('low', 'medium', 'high');
                } else if (model === 'gemini-3.6-flash') {
                    add('medium', 'high');
                } else if (model === 'gemini-3.5-flash-lite') {
                    add('minimal', 'medium', 'high');
                } else if (model.includes('flash')) {
                    add('minimal', 'low', 'medium', 'high');
                } else {
                    add('low', 'high');
                }
            }

            if (isDeepSeek) {
                add('thinking', '思考', 'reasoning', '推論', 'reasoning effort', 'high');
                if (model !== 'deepseek-v4-pro') add('low');
                if (model.includes('v4-flash')) add('none', 'max');
            }

            const supportsReasoningEffort = isLlm && (
                model.includes('gpt-5') || model.includes('o1') || model.includes('o3')
                || model.includes('grok-4.3') || model.includes('grok-4.5') || model.includes('grok-4.6')
                || model.includes('grok-4.20-0309-reasoning') || model.includes('grok-build')
                || model.includes('multi-agent') || (model.includes('gpt') && !isTts)
            );
            if (supportsReasoningEffort) {
                add('reasoning', '推論', 'reasoning effort', 'low', 'high');
                const isGpt56Model = model === 'gpt-5.6' || model.startsWith('gpt-5.6-');
                const isGpt6Model = model.startsWith('gpt-6');
                const isGrok46 = model.includes('grok-4.6');
                const supportsMedium = model.includes('grok-4.3') || model.includes('grok-4.5') || isGrok46
                    || model.includes('grok-4.20-0309-reasoning') || model.includes('grok-build')
                    || model.includes('multi-agent') || model.includes('gpt-5') || isGpt6Model || model.includes('o1') || model.includes('o3');
                const supportsNone = model.includes('grok-4.3') || model.includes('grok-build')
                    || model.includes('gpt-5') || model === 'gpt-6-sol' || model === 'gpt-6-luna' || model === 'gpt-6.1-terra' || isDeepSeek;
                if (supportsMedium) add('medium');
                if (supportsNone) add('none');
                if (isGpt56Model || isGpt6Model || isDeepSeek) add('max');
                if (isGrok46 || model.includes('multi-agent') || isGpt56Model || isGpt6Model) add('xhigh');
            }

            if (model.includes('claude')) add('thinking', '思考', 'thinking budget', 'budget');
            if (m.agenticView) add('agentic view');
            return [...new Set(terms)];
        }

        // The settings form owns its draft. Reuse the picker without changing chat state.
        let settingsModelPickerTarget = null;
        const settingsModelPickerTargets = ['set-default-model', 'set-default-vision-model'];
        function syncSettingsModelPickerLabels() {
            settingsModelPickerTargets.forEach(target => {
                const select = get(target);
                const label = get(target + '-label');
                if (select && label) label.textContent = select.selectedOptions[0]?.textContent || select.value || 'モデルを選択';
            });
        }
        function modelPickerSelectedId() {
            return get(settingsModelPickerTarget || 'model-select')?.value || '';
        }
        function settingsModelPickerAllows(id) {
            if (!settingsModelPickerTarget) return true;
            return Array.from(get(settingsModelPickerTarget)?.options || []).some(option => option.value === id);
        }
        function openSettingsModelPicker(target, skipHistory = false) {
            const settings = get('settings-modal');
            if (!settingsModalLoaded || !settingsModelPickerTargets.includes(target)
                || !(settings?.classList.contains('modal-open') || settings?.classList.contains('modal-prep'))) return;
            const select = get(target);
            if (!select || select.disabled || !select.options.length) return;
            settingsModelPickerTarget = target;
            window._visionPickerActive = false;
            if (!skipHistory) history.pushState({ ...history.state, settingsModelPicker: target }, '', '/settings');
            get('settings-modal').inert = true;
            activeModelTag = 'all';
            openModelModal();
        }
        window.openSettingsModelPicker = openSettingsModelPicker;

        const modelListGroups = [];
        let modelListBanner = null;
        let modelListEmpty = null;
        let modelListBuilt = false;
        let modelListAnimated = false;
        let modelListRenderFrame = 0;
        function isLlmModelForDisplay(model) {
            const id = String(model && model.id || '').toLowerCase();
            return !/(ocr|tts|transcri|realtime|voice-agent|native-audio|\blive\b|image|video|embedding|lyria|veo|omni-|ideogram|deep-research|antigravity|computer-use|robotics)/.test(id);
        }
        function buildModelList() {
            const container = get('model-list-container');
            if (!container || modelListBuilt) return;
            container.innerHTML = '';
            modelListBanner = document.createElement('div');
            modelListBanner.className = 'model-banner hidden';
            container.appendChild(modelListBanner);

            const providerGroups = new Map();
            MODELS.forEach(group => {
                group.items.filter(m => !m.deprecated).forEach(model => {
                    const provider = getModelApiProvider(model.id) || 'other';
                    if (!providerGroups.has(provider)) providerGroups.set(provider, []);
                    providerGroups.get(provider).push({ model, category: group.category });
                });
            });
            providerGroups.forEach((providerItems, provider) => {
                const groupEl = document.createElement('section');
                groupEl.className = 'model-list-group';
                const providerLabel = PROVIDER_LABELS[provider] || provider;
                groupEl.innerHTML = `
                    <div class="model-group-header">
                        <i class="fas fa-brain text-green-400"></i>
                        <div>
                            <h3 class="model-group-title">${providerLabel}</h3>
                            <p class="model-group-desc">LLM models first</p>
                        </div>
                    </div>
                    <div class="model-group-grid"></div>
                `;
                const grid = groupEl.querySelector('.model-group-grid');
                const entries = providerItems.map(itemData => ({ ...itemData, provider }))
                    .sort((a, b) => Number(isLlmModelForDisplay(b.model)) - Number(isLlmModelForDisplay(a.model)))
                    .map(({ model: m, category, provider }) => {
                        const item = document.createElement('button');
                        const apiModelName = String(m.apiId || m.id || '').trim();
                        const agenticViewHtml = m.agenticView ? `<span class="inline-flex items-center gap-1 rounded-full border border-teal-500/40 bg-teal-900/20 px-2 py-0.5 text-[9px] font-semibold text-teal-200 whitespace-nowrap" title="Agentic View対応：画像をクロップして再観察しながら推論を継続できます"><i class="fas fa-eye" aria-hidden="true"></i>Agentic View</span>` : '';
                        const apiModelHtml = apiModelName ? `<div class="text-[10px] text-cyan-300/90 mt-1.5 font-mono break-all"><span class="font-sans text-gray-500 mr-1">API model:</span>${escapeHtml(apiModelName)}</div>` : '';
                        const priceHtml = m.price ? `<div class="text-[10px] text-amber-400/90 mt-1.5 font-mono flex items-start gap-1"><i class="fas fa-tag text-[9px] mt-0.5 opacity-70 shrink-0"></i><span>${m.price}</span></div>` : '';
                        item.type = 'button';
                        item.className = 'model-card';
                        item.dataset.selected = '0';
                        item.onclick = () => selectModelByUser(m.id, m.name);
                        item.innerHTML = `
                            <div class="flex justify-between items-start gap-2 w-full mb-1">
                                <div class="flex flex-wrap items-center gap-2 min-w-0">
                                    <span class="model-name font-bold text-sm">${m.name}</span>
                                    ${agenticViewHtml}
                                </div>
                                <i class="model-selected-icon fas fa-check-circle hidden shrink-0 mt-0.5"></i>
                            </div>
                            <span class="model-desc text-[10px]">${m.desc}</span>
                            ${apiModelHtml}
                            ${priceHtml}
                        `;
                        grid.appendChild(item);
                        return {
                            model: m,
                            button: item,
                            searchText: `${m.name} ${m.id} ${apiModelName} ${m.agenticView ? 'agentic view' : ''} ${category} ${getModelTags(m, { category }).join(' ')} ${getModelCapabilitySearchTerms(m).join(' ')}`.toLowerCase(),
                            provider,
                            tags: new Set(getModelTags(m, { category })),
                        };
                    });
                modelListGroups.push({ element: groupEl, entries });
                container.appendChild(groupEl);
            });

            modelListEmpty = document.createElement('div');
            modelListEmpty.className = 'model-list-empty hidden';
            container.appendChild(modelListEmpty);
            modelListBuilt = true;
        }

        function updateModelButtonSelection(entry, selectedModel) {
            const isSelected = selectedModel === entry.model.id;
            if (entry.button.dataset.selected === (isSelected ? '1' : '0')) return;
            entry.button.dataset.selected = isSelected ? '1' : '0';
            entry.button.classList.toggle('is-selected', isSelected);
            const icon = entry.button.querySelector('.model-selected-icon');
            if (icon) icon.classList.toggle('hidden', !isSelected);
        }

        function renderModelList(filter = "", options = {}) {
            const container = get('model-list-container');
            if (!container) return;
            buildModelList();
            const f = filter.toLowerCase();
            const lockedProvider = settingsModelPickerTarget || window._visionPickerActive ? null : getPromptCacheLockedProvider();
            const lockedLabel = lockedProvider ? (PROVIDER_LABELS[lockedProvider] || lockedProvider) : '';
            const selectedModel = modelPickerSelectedId();
            let visibleCount = 0;

            modelListBanner.classList.toggle('hidden', !lockedProvider);
            if (lockedProvider) {
                modelListBanner.innerHTML = `<i class="fas fa-database mr-1.5"></i>PromptCache 有効中: <strong>${lockedLabel}</strong> のモデルのみ選択できます（他APIへの切替は不可）`;
            }

            modelListGroups.forEach(group => {
                let groupVisibleCount = 0;
                group.entries.forEach(entry => {
                    const visible = settingsModelPickerAllows(entry.model.id) && entry.searchText.includes(f)
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

        function animateModelCategoryChange() {
            const container = get('model-list-container');
            if (!container) return;
            container.classList.remove('model-category-enter');
            void container.offsetWidth;
            container.classList.add('model-category-enter');
        }

        let modelListScrollFrame = 0;
        function scrollSelectedModelIntoView() {
            const container = get('model-list-container');
            const selectedModel = modelPickerSelectedId();
            const entry = modelListGroups.flatMap(group => group.entries)
                .find(item => item.model.id === selectedModel);
            if (!container || !entry || entry.button.classList.contains('hidden')) return;

            const containerRect = container.getBoundingClientRect();
            const cardRect = entry.button.getBoundingClientRect();
            const edgePadding = 12;
            let targetTop = container.scrollTop;
            if (cardRect.top < containerRect.top + edgePadding) {
                targetTop += cardRect.top - containerRect.top - edgePadding;
            } else if (cardRect.bottom > containerRect.bottom - edgePadding) {
                targetTop += cardRect.bottom - containerRect.bottom + edgePadding;
            }
            targetTop = Math.max(0, Math.min(targetTop, container.scrollHeight - container.clientHeight));
            if (Math.abs(targetTop - container.scrollTop) < 1) return;

            if (modelListScrollFrame) cancelAnimationFrame(modelListScrollFrame);
            if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
                container.scrollTop = targetTop;
                return;
            }

            const startTop = container.scrollTop;
            const distance = targetTop - startTop;
            const startTime = performance.now();
            const duration = 160;
            const step = (now) => {
                const progress = Math.min(1, (now - startTime) / duration);
                const eased = 1 - Math.pow(1 - progress, 3);
                container.scrollTop = startTop + distance * eased;
                if (progress < 1) {
                    modelListScrollFrame = requestAnimationFrame(step);
                } else {
                    modelListScrollFrame = 0;
                }
            };
            modelListScrollFrame = requestAnimationFrame(step);
        }

        function openModelModal() {
            if (!settingsModelPickerTarget && location.pathname !== '/model') {
                history.pushState({ modal: 'model' }, '', '/model');
            }
            const search = get('model-search');
            if (search) search.value = '';
            updateModelTagUi();
            syncModelSearchClear();
            // Build/update while hidden so opening animation never competes with DOM construction.
            renderModelList('', { animate: true });
            showModal('model-modal');
            requestAnimationFrame(() => requestAnimationFrame(scrollSelectedModelIntoView));
            // Prevent auto-focus on mobile to avoid keyboard popup
            if (search && window.innerWidth > 768) {
                requestAnimationFrame(() => search.focus({ preventScroll: true }));
            }
        }
        window.closeModelModal = (skipHistory = false) => {
            const settingsTarget = settingsModelPickerTarget;
            settingsModelPickerTarget = null;
            window._visionPickerActive = false;
            hideModal('model-modal');
            if (settingsTarget) {
                get('settings-modal').inert = false;
                if (!skipHistory && history.state?.settingsModelPicker) history.back();
                get(settingsTarget + '-picker')?.focus({ preventScroll: true });
                return;
            }
            if (!skipHistory && location.pathname === '/model') {
                history.back();
            }
        };

        function selectModel(id, name) {
            if (settingsModelPickerTarget) {
                if (!settingsModelPickerAllows(id)) return;
                const select = get(settingsModelPickerTarget);
                select.value = id;
                select.dispatchEvent(new Event('change', { bubbles: true }));
                syncSettingsModelPickerLabels();
                window.closeModelModal();
                return;
            }
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
        function selectModelByUser(id, name) {
            const forVision = !!window._visionPickerActive || !!settingsModelPickerTarget;
            selectModel(id, name);
            const el = get('model-select');
            // PromptCache can refuse a switch to another provider; only a real change counts.
            if (!forVision && el && el.value === id) modelChosenManually = true;
        }

        function populateAiSafeFormFields(d) {
            // Mirror a subset of the population in openSettingsModal for live update after AI apply
            if (!d) return;
            try {
                if (get('set-default-model')) get('set-default-model').value = d.default_model || get('set-default-model').value;
                if (get('set-default-vision-model')) get('set-default-vision-model').value = d.default_vision_model || 'gemini-3-flash-preview';
                syncSettingsModelPickerLabels();
                if (get('set-default-search')) get('set-default-search').checked = !!d.default_enable_search;
                if (get('set-default-url-context')) get('set-default-url-context').checked = !!d.default_enable_url_context;
                if (get('set-default-maps')) get('set-default-maps').checked = !!d.default_enable_maps;
                if (get('set-default-python')) get('set-default-python').checked = !!d.default_enable_python;
                if (get('set-default-file-creation')) get('set-default-file-creation').checked = !!d.default_enable_file_creation;
                if (get('set-default-thinking')) get('set-default-thinking').checked = !!d.default_enable_thinking;
                if (get('set-default-sys-prompt')) get('set-default-sys-prompt').checked = !!d.default_enable_system_prompt;
                if (get('set-default-mcp')) get('set-default-mcp').checked = d.default_enable_mcp !== false;
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

        function syncModelSearchClear() {
            const input = get('model-search');
            const clearBtn = get('model-search-clear');
            if (clearBtn) clearBtn.classList.toggle('hidden', !input || !input.value);
        }

        if (get('model-search')) {
            get('model-search').addEventListener('input', (e) => {
                scheduleModelListRender(e.target.value);
                syncModelSearchClear();
            });
        }
        if (get('model-search-clear')) {
            get('model-search-clear').addEventListener('click', () => {
                const input = get('model-search');
                if (!input) return;
                input.value = '';
                syncModelSearchClear();
                scheduleModelListRender('');
                input.focus();
            });
        }
        if (get('model-tag-bar')) {
            get('model-tag-bar').addEventListener('click', (e) => {
                const btn = e.target.closest('.model-tag-btn');
                if (!btn) return;
                const t = btn.innerText.trim().toLowerCase();
                const nextTag = MODEL_TAGS.includes(t) ? t : 'all';
                if (nextTag === activeModelTag) return;
                activeModelTag = nextTag;
                updateModelTagUi();
                const search = get('model-search');
                renderModelList(search ? search.value : '');
                animateModelCategoryChange();
            });
            updateModelTagUi();
        }

        window.quickStart = (m) => {
            const found = MODELS.flatMap(g => g.items).find(i => i.id === m);
            selectModelByUser(m, found ? found.name : m);
            get('welcome-screen').classList.add('hidden');
        };

        const BROWSER_FAST_DISABLED_OPTIONS = [
            ['enable-search', 'search-container'],
            ['enable-url-context', 'url-context-container'],
            ['enable-maps', 'maps-grounding-container'],
            ['enable-sys-prompt', 'sys-prompt-option'],
            ['enable-prompt-cache', 'prompt-cache-container'],
            ['enable-mcp', 'mcp-container'],
            ['enable-skills', 'skills-container'],
            ['enable-file-creation', 'file-creation-container'],
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
            if (typeof syncMcpAutoSysRows === 'function') syncMcpAutoSysRows();
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
            if (typeof syncMcpAutoSysRows === 'function') syncMcpAutoSysRows();
            refreshMinimalOptionsIfOpen();
        }

        function isBatchModelKey(model) {
            const m = String(model || '').trim().toLowerCase();
            if (m.startsWith('gpt-')) return !/(image|audio|tts|transcribe|realtime|search)/.test(m);
            if (m.startsWith('grok-')) return !/(image|video|voice|audio|tts|realtime)/.test(m);
            if (m.startsWith('glm-')) return !/(image|audio|tts|transcribe|realtime|video|embedding|ocr)/.test(m);
            if (!m.startsWith('gemini-')) return false;
            return !/(embedding|video|veo|music|lyria|native-audio|tts|live|transcribe|agent|deep-research|robotics|computer-use)/.test(m);
        }

        function updateBatchUi(model) {
            const container = get('batch-mode-container');
            const checkbox = get('enable-batch-mode');
            if (!container || !checkbox) return;
            const supported = isBatchModelKey(model);
            container.classList.toggle('hidden', !supported);
            checkbox.disabled = !supported || browserFastModeEnabled;
            if (!supported) checkbox.checked = false;
            container.classList.toggle('ring-1', supported && checkbox.checked);
            container.classList.toggle('ring-violet-300', supported && checkbox.checked);
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
            updateBatchUi(get('model-select') ? get('model-select').value : '');
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
        //! @chat-core-bundle-split: scripts/build_frontend.sh starts the next browser file here (see README.md).

        document.addEventListener('DOMContentLoaded', () => {
            initThemeFromServer();
            applyLiquidGlassMode(INITIAL_LIQUID_GLASS_ENABLED);
            updateCurrentChatHeaderUi();
            ensureCurrentChatHeaderTicker();
            try { sessionStorage.removeItem('browser_fast_mode_gemini_key'); } catch (e) {}
            const fastToggle = get('enable-browser-fast-mode');
            if (fastToggle) {
                fastToggle.checked = false;
                fastToggle.onchange = () => {
                    if (fastToggle.checked) {
                        const batchToggle = get('enable-batch-mode');
                        if (batchToggle && batchToggle.checked) batchToggle.checked = false;
                        requestBrowserFastModeEnable();
                    }
                    else setBrowserFastModeEnabled(false);
                };
            }
            const batchToggle = get('enable-batch-mode');
            if (batchToggle) {
                batchToggle.onchange = () => {
                    if (batchToggle.checked) {
                        if (browserFastModeEnabled) setBrowserFastModeEnabled(false);
                        const codingToggle = get('enable-coding-mode');
                        if (codingToggle && codingToggle.checked) {
                            codingToggle.checked = false;
                            if (typeof syncCodingModeUi === 'function') syncCodingModeUi(false);
                            showToast('Batch APIではCoding Modeを利用できないため解除しました', 'warning', true);
                        }
                    }
                    updateBatchUi(get('model-select') ? get('model-select').value : '');
                };
            }
            const fastModelSelect = get('model-select');
            if (fastModelSelect) fastModelSelect.addEventListener('change', () => {
                setTimeout(() => {
                    if (!browserFastModeEnabled) return;
                    const model = String(fastModelSelect.value || '').toLowerCase();
                    browserFastApiKey = '';
                    browserFastApiKeyModel = '';
                    browserFastBootstrap = null;
                    if (!model.startsWith('gemini-') || /(image|native-audio|tts|live|flash-cyber)/.test(model)) {
                        setBrowserFastModeEnabled(false);
                        fastModelSelect.dispatchEvent(new Event('change'));
                        showToast('対象外モデルを選択したため高速モードを解除しました', 'warning', true);
                    } else {
                        applyBrowserFastModeRestrictions();
                    }
                }, 0);
            });
            const fastEnableBtn = get('browser-fast-mode-enable-btn');
            if (fastEnableBtn) fastEnableBtn.onclick = async () => {
                const originalHtml = fastEnableBtn.innerHTML;
                fastEnableBtn.disabled = true;
                fastEnableBtn.innerHTML = '<i class="fas fa-spinner fa-spin mr-1"></i>保存済みキーを取得中...';
                try {
                    await fetchBrowserFastBootstrap(true);
                    const ignore = get('browser-fast-mode-ignore-warning');
                    if (ignore && ignore.checked) {
                        try { localStorage.setItem(BROWSER_FAST_IGNORE_WARNING_STORAGE, '1'); } catch (e) {}
                    }
                    hideModal('browser-fast-mode-modal');
                    setBrowserFastModeEnabled(true, { clearKey: false });
                    showToast('高速モードを有効にしました。生成中は再読み込みしないでください。', 'warning', true);
                } catch (error) {
                    showToast(error.message || '保存済みGemini APIキーを取得できませんでした', 'error', true);
                } finally {
                    fastEnableBtn.disabled = false;
                    fastEnableBtn.innerHTML = originalHtml;
                }
            };
            const fastCancelBtn = get('browser-fast-mode-cancel-btn');
            if (fastCancelBtn) fastCancelBtn.onclick = () => {
                hideModal('browser-fast-mode-modal');
                setBrowserFastModeEnabled(false);
            };
            const bar = document.getElementById('alpha-bar'); setTimeout(() => { if(bar) { const target = document.getElementById('version-display'); if(target) { const barRect = bar.getBoundingClientRect(); const targetRect = target.getBoundingClientRect(); const tx = targetRect.left + (targetRect.width/2) - (barRect.left + barRect.width/2); const ty = targetRect.top + (targetRect.height/2) - (barRect.top + barRect.height/2); bar.style.transform = `translate(${tx}px, ${ty}px) scale(0.1)`; bar.style.opacity = '0'; setTimeout(() => { target.classList.add('pulse-target'); setTimeout(() => target.classList.remove('pulse-target'), 2000); bar.remove(); }, 800); } else { bar.style.opacity = '0'; setTimeout(() => bar.remove(), 1000); } } }, 3000);
            function updateGptImageUi() {
                const wrap = get('gpt-image-options');
                if (!wrap) return;
                if (isGptImageModel()) {
                    wrap.classList.remove('hidden');
                } else {
                    wrap.classList.add('hidden');
                }
                const model = (get('model-select').value || '').toLowerCase();
                const is26 = model.includes('gpt-image-2.6');
                const fmt = get('gpt-image-format');
                if (get('gpt-image-complexity-wrap')) get('gpt-image-complexity-wrap').classList.toggle('hidden', !is26);
                if (fmt) {
                    Array.from(fmt.options).forEach(opt => {
                        if (opt.hasAttribute('data-gpt-image-26')) opt.hidden = !is26;
                    });
                    if (!is26 && fmt.value === 'gif') fmt.value = 'jpeg';
                }
                const isGif = is26 && fmt && fmt.value === 'gif';
                const compWrap = get('gpt-image-compression-wrap');
                if (fmt && compWrap) {
                    if (fmt.value === 'png' || fmt.value === 'gif') {
                        compWrap.classList.add('hidden');
                    } else {
                        compWrap.classList.remove('hidden');
                    }
                }
                if (get('gpt-image-gif-wrap')) get('gpt-image-gif-wrap').classList.toggle('hidden', !isGif);
            }
            function updateGeminiImageUi() {
                const wrap = get('gemini-image-options');
                if (!wrap) return;
                if (isGeminiImageModel()) {
                    wrap.classList.remove('hidden');
                } else {
                    wrap.classList.add('hidden');
                }
                const model = (get('model-select').value || '').toLowerCase();
                const isLiteImage = model.includes('gemini-3.1-flash-lite-image');
                [get('gemini-image-size'), get('modal-gemini-image-size')].forEach(sizeEl => {
                    if (!sizeEl) return;
                    Array.from(sizeEl.options).forEach(opt => {
                        if (opt.value !== '1K') opt.disabled = isLiteImage;
                    });
                    if (isLiteImage && sizeEl.value !== '1K') sizeEl.value = '1K';
                });
            }
        function updateGrokImageUi() {
            const wrap = get('grok-image-options');
            if (!wrap) return;
            const model = (get('model-select').value || '').toLowerCase();
            const isGrokImg = isGrokImageModel();
            const showResolution = model === 'grok-imagine-image-quality' || model === 'grok-imagine-image-2.0';
            const showQuality = model === 'grok-imagine-image-2.0';
            if (isGrokImg) {
                wrap.classList.remove('hidden');
                const resWrap = get('grok-image-resolution') ? get('grok-image-resolution').parentElement : null;
                if (resWrap) {
                    resWrap.classList.toggle('hidden', !showResolution);
                }
                const qualityWrap = get('grok-image-quality') ? get('grok-image-quality').parentElement : null;
                if (qualityWrap) {
                    qualityWrap.classList.toggle('hidden', !showQuality);
                }
            } else {
                wrap.classList.add('hidden');
            }

            // Modal part
            const modalWrap = get('modal-grok-image-options');
            if (modalWrap) {
                const resWrapModal = get('modal-grok-image-resolution') ? get('modal-grok-image-resolution').parentElement : null;
                if (resWrapModal) {
                    resWrapModal.classList.toggle('hidden', !showResolution);
                }
                const qualityWrapModal = get('modal-grok-image-quality') ? get('modal-grok-image-quality').parentElement : null;
                if (qualityWrapModal) {
                    qualityWrapModal.classList.toggle('hidden', !showQuality);
                }
            }
        }
        function updateIdeogramImageUi() {
            const wrap = get('ideogram-image-options');
            const modalWrap = get('modal-ideogram-image-options');
            const model = (get('model-select') && get('model-select').value) || '';
            const active = isIdeogramModel(model);
            const traits = ideogramModelTraits(model);
            const visibleByField = {
                resolution: traits.resolution, quality: traits.quality, speed: traits.speed,
                style: traits.style, negative: traits.negative
            };
            if (wrap) wrap.classList.toggle('hidden', !active);
            if (modalWrap) modalWrap.classList.toggle('hidden', !active);
            updateIdeogramPromptInputUi();
            if (!active) return;
            const removeBackground = get('ideogram-image-operation')?.value === 'remove_background';
            const removeObject = get('ideogram-image-operation')?.value === 'remove_object';
            IDEOGRAM_IMAGE_FIELDS.forEach((field) => {
                const show = field === 'operation' || (removeObject && field === 'seed')
                    || (!(removeBackground || removeObject) && visibleByField[field] !== false);
                ['ideogram-image-', 'modal-ideogram-image-'].forEach((prefix) => {
                    const el = get(prefix + field);
                    if (!el || !el.parentElement) return;
                    // The modal's negative prompt row sits outside the grid, so hide that row itself.
                    el.parentElement.classList.toggle('hidden', !show);
                });
            });
            ['ideogram-image-style', 'modal-ideogram-image-style'].forEach((id) => {
                const el = get(id);
                if (!el) return;
                const wanted = traits.styleOptions.join('|');
                if (el.dataset.options !== wanted) {
                    const prev = el.value;
                    el.innerHTML = traits.styleOptions.map((v) => `<option value="${v}">${v === 'auto' ? 'Auto' : v}</option>`).join('');
                    el.dataset.options = wanted;
                    el.value = traits.styleOptions.includes(prev) ? prev : 'auto';
                }
            });
            const note = get('ideogram-image-note');
            if (note) {
                note.textContent = removeBackground
                    ? '※ 背景を削除: 画像1枚 / 25MBまで / PNG・JPEG・WEBP。添付がない場合は表示中のブランチの最後の画像。送信ボタンだけで実行し、透過PNGで出力'
                    : removeObject
                    ? '※ Object Remover: 画像1枚 + 同じサイズの白黒マスク画像が必要です。マスクの白い部分を削除します'
                    : traits.edit
                    ? '※ Ideogram 4.5: 画像を添付（または直前の画像がある場合）は Precise Edit。出力は元画像と同サイズ'
                    : '※ Ideogram: テキストからの生成のみ（画像編集は Ideogram 4.5 のみ）';
            }
        }
        ['ideogram-image-operation', 'modal-ideogram-image-operation'].forEach((id) => {
            get(id)?.addEventListener('change', () => {
                const other = get(id.startsWith('modal-') ? 'ideogram-image-operation' : 'modal-ideogram-image-operation');
                if (other) other.value = get(id).value;
                updateIdeogramImageUi();
                toggleOptions();
            });
        });
        function updateGrokVideoUi() {
            const wrap = get('grok-video-options');
            if (!wrap) return;
            const model = String(get('model-select')?.value || '').toLowerCase();
            if (isGrokVideoModel()) {
                wrap.classList.remove('hidden');
            } else {
                wrap.classList.add('hidden');
            }
            const resolution = get('grok-video-resolution');
            if (resolution) {
                const fullHd = Array.from(resolution.options).find(opt => opt.value === '1080p');
                const supportsFullHd = ['grok-imagine-video-1.5', 'grok-imagine-video-1.5-lite'].includes(model);
                if (fullHd) fullHd.disabled = !supportsFullHd;
                if (!supportsFullHd && resolution.value === '1080p') resolution.value = '720p';
            }
        }
        function updateGeminiVideoUi() {
            const wrap = get('gemini-video-options');
            if (!wrap) return;
            const model = String(get('model-select')?.value || '').toLowerCase();
            if (isGeminiVideoModel()) {
                wrap.classList.remove('hidden');
            } else {
                wrap.classList.add('hidden');
            }
            const resolution = get('gemini-video-resolution');
            if (resolution) {
                const fourK = Array.from(resolution.options).find(opt => opt.value === '4K');
                const no4k = model === 'veo-3.1-lite-generate-preview' || model === 'veo-3.1-fast-generate-preview' || model === 'gemini-omni-flash';
                if (fourK) fourK.disabled = no4k;
                if (no4k && resolution.value === '4K') resolution.value = '1080p';
            }
            // Gemini Omni 1.1 Flash generates fixed-length clips and has no
            // output duration parameter, so hide the duration selector for it.
            const durationWrap = get('gemini-video-duration-wrap');
            if (durationWrap) durationWrap.classList.toggle('hidden', model === 'gemini-omni-1.1-flash');
        }
        function updateGeminiMusicUi() {
            const wrap = get('gemini-music-options');
            if (!wrap) return;
            const realtime = isGeminiRealtimeMusicModel();
            const music = isGeminiMusicModel() && !realtime;
            wrap.classList.toggle('hidden', !music);
            const studioBar = get('lyria-realtime-studio-bar');
            if (studioBar) studioBar.classList.toggle('hidden', !realtime);
        }
        function updateXaiChatUi() {
            const wrap = get('xai-chat-options');
            if (!wrap) return;
            const model = String(get('model-select')?.value || '').toLowerCase();
            const show = model.startsWith('grok-') && !isGrokImageModel(model) && !isGrokVideoModel(model) && !model.includes('voice');
            wrap.classList.toggle('hidden', !show);
            const logprobs = get('xai-logprobs');
            const topLogprobs = get('xai-top-logprobs');
            const unsupported = model.includes('grok-4.20');
            if (logprobs) {
                logprobs.disabled = unsupported;
                if (unsupported) logprobs.checked = false;
            }
            if (topLogprobs) {
                topLogprobs.disabled = unsupported;
                if (unsupported) topLogprobs.value = '';
            }
        }
        function updateMistralOcrUi() {
            const ocr = isMistralOcrModel();
            const wrap = get('mistral-ocr-options');
            if (wrap) wrap.classList.toggle('hidden', !ocr);
            const modalWrap = get('modal-mistral-ocr-options');
            if (modalWrap) modalWrap.classList.toggle('hidden', !ocr);
            ['canvas-mode-container', 'coding-mode-container', 'browser-fast-mode-container'].forEach((id) => {
                const el = get(id);
                if (!el) return;
                el.classList.toggle('opacity-50', ocr);
                el.classList.toggle('pointer-events-none', ocr);
            });
            if (ocr) {
                if (canvasModeEnabled) syncCanvasModeUi(false, { persist: false });
                if (codingModeEnabled) syncCodingModeUi(false, { persist: false });
                if (typeof browserFastModeEnabled !== 'undefined' && browserFastModeEnabled) {
                    setBrowserFastModeEnabled(false);
                }
            }
        }
        function updateImageInputLimits() {
            const el = get('image-input-limits');
            if (!el) return;
            const model = (get('model-select').value || '').toLowerCase();
            let html = '';
            let show = false;
            if (model.includes('gpt-image')) {
                show = true;
                html = [
                    '<div class="font-bold text-gray-300 mb-1">GPT-Image 入力制限</div>',
                    '<div>最大 16 枚 / 画像1枚あたり 50MB 未満 / PNG・JPG・WEBP</div>',
                    '<div>マスク使用時: PNGのみ、4MB未満、元画像と同サイズ</div>'
                ].join('');
            } else if (model === 'deepseek-v4.1-flash' || model === 'deepseek-v4-flash-vision-exp') {
                show = true;
                html = [
                    '<div class="font-bold text-gray-300 mb-1">DeepSeek V4.1 Flash 入力制限</div>',
                    '<div>JPEG・PNG・GIF・WebP / 画像1枚あたり最大32MB / リクエスト合計48MB</div>',
                    '<div>画像は約800×800相当へ自動リサイズ（1枚あたり最大384トークン）</div>'
                ].join('');
            } else if (model.includes('deepseek')) {
            } else if (isGeminiImageModelKey(model)) {
                show = true;
                if (model.includes('gemini-3.1-flash-lite-image')) {
                    html = [
                        '<div class="font-bold text-gray-300 mb-1">Nano Banana 2 Lite 入力目安</div>',
                        '<div>画像生成・編集 / 1K出力 / 最大14枚の参照画像に対応</div>',
                        '<div>複数参照や連続編集より、低遅延・大量生成向けです</div>'
                    ].join('');
                } else if (model === 'gemini-nano-banana-2.1') {
                    html = [
                        '<div class="font-bold text-gray-300 mb-1">Nano Banana 2.1 入力目安</div>',
                        '<div>画像生成・編集 / 1K・2K・4K出力 / 最大14枚の参照画像に対応</div>',
                        '<div>動画を参考にした画像生成にも対応します</div>'
                    ].join('');
                } else if (model.includes('gemini-3.1-flash-image')) {
                    html = [
                        '<div class="font-bold text-gray-300 mb-1">Nano Banana 2 入力目安</div>',
                        '<div>画像入力は最大3枚程度を推奨（Gemini 3.1 Flash Image）</div>'
                    ].join('');
                } else if (model.includes('gemini-2.5') && model.includes('image')) {
                    html = [
                        '<div class="font-bold text-gray-300 mb-1">Nano Banana 入力目安</div>',
                        '<div>画像入力は最大3枚までが推奨</div>'
                    ].join('');
                } else {
                    html = [
                        '<div class="font-bold text-gray-300 mb-1">Nano Banana Pro 入力目安</div>',
                        '<div>高精度は最大5枚 / 合計14枚まで対応</div>'
                    ].join('');
                }
            } else if (isMistralOcrModel(model)) {
                show = true;
                html = [
                    '<div class="font-bold text-gray-300 mb-1">Mistral OCR 4 入力</div>',
                    '<div>PDF / PNG / JPEG / TIFF / BMP / GIF / WEBP / DOCX / PPTX、または公開URL</div>',
                    '<div>最大 512MB / 会話履歴は送信しません / チャット補完・Search・Python・Canvas 非対応</div>'
                ].join('');
            } else if (model.includes('grok')) {
                show = true;
                html = [
                    '<div class="font-bold text-gray-300 mb-1">Grok 画像入力制限</div>',
                    '<div>最大 20MiB / PNG・JPG のみ / 枚数制限なし</div>'
                ].join('');
            } else if (isIdeogramModel(model)) {
                show = true;
                html = ideogramModelTraits(model).edit ? [
                    '<div class="font-bold text-gray-300 mb-1">Ideogram 4.5 入力制限</div>',
                    '<div>編集する画像1枚 + 参考画像は最大4枚 / 1枚あたり25MBまで / PNG・JPEG・WEBP</div>',
                    '<div>画像がない場合は直前の画像を編集します。背景削除と Object Remover も利用できます</div>'
                ].join('') : [
                    '<div class="font-bold text-gray-300 mb-1">Ideogram 入力</div>',
                    '<div>生成・編集では画像入力は Ideogram 4.5 のみ。背景削除と Object Remover は全モデルで利用できます</div>'
                ].join('');
            } else if (model.includes('grok') && model.includes('video')) {
                show = true;
                html = [
                    '<div class="font-bold text-gray-300 mb-1">Grok 動画生成制限</div>',
                    '<div>Duration: 1-15s / Resolution: 720p, 480p</div>',
                    '<div>画像からの動画生成に対応 (PNG・JPG)</div>'
                ].join('');
            }
            if (show) {
                el.innerHTML = html;
                el.classList.remove('hidden');
            } else {
                el.classList.add('hidden');
                el.innerHTML = '';
            }
        }
