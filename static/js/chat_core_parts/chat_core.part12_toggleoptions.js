            // 画像・OCR などでSysPromptが使えないモデルへ切り替えると、スイッチは強制的にOFFになる。
            // 使えるモデルへ戻したときに、切り替え前の選択を復元する。
            function toggleOptions() {
                const sysSwitch = get('enable-sys-prompt');
                const sysWasDisabled = !!(sysSwitch && sysSwitch.disabled);
                const sysWasChecked = !!(sysSwitch && sysSwitch.checked);
                toggleOptionsForModel();
                if (!sysSwitch) return;
                if (sysSwitch.disabled) {
                    if (!sysWasDisabled && sysWasChecked) sysSwitch.dataset.restoreChecked = '1';
                    else if (!sysWasDisabled) delete sysSwitch.dataset.restoreChecked;
                } else {
                    if (sysWasDisabled && sysSwitch.dataset.restoreChecked === '1') sysSwitch.checked = true;
                    delete sysSwitch.dataset.restoreChecked;
                }
            }
            // applyChatDefaults（part04、DOMContentLoaded の外）からも呼ぶため公開する。
            window.toggleOptions = toggleOptions;
            // Reasoning effort を使うモデルも、Gemini と同じ「Thinking」ボタン
            // （ON/OFF ＋ レベル選択）で操作する。reasoning-effort は送信値の
            // 正本として残し、表示だけを Thinking 側へ寄せる。
            function setComposerThinkingMode(mode) {
                const thinkOpts = get('thinking-options');
                const thinkLvl = get('thinking-level');
                const effortSel = get('reasoning-effort');
                const thinkChk = get('enable-thinking');
                const budgetLbl = get('thinking-budget-label');
                const thinkBudget = get('thinking-budget');
                const isEffort = mode === 'effort';
                if (thinkOpts) thinkOpts.dataset.thinkingMode = isEffort ? 'effort' : 'thinking';
                if (thinkLvl) thinkLvl.classList.toggle('hidden', isEffort);
                if (effortSel) effortSel.classList.toggle('hidden', !isEffort);
                if (budgetLbl) budgetLbl.classList.toggle('hidden', isEffort);
                if (thinkBudget) thinkBudget.classList.toggle('hidden', isEffort);
                if (!thinkChk) return;
                if (isEffort) {
                    // Gemini 用のON/OFF状態を退避し、モデルを戻したときに復元する。
                    if (thinkChk.dataset.thinkingChecked === undefined) {
                        thinkChk.dataset.thinkingChecked = thinkChk.checked ? '1' : '0';
                    }
                    const noneOpt = effortSel ? Array.from(effortSel.options).find(opt => opt.value === 'none') : null;
                    const noneAvailable = !!(noneOpt && !noneOpt.disabled && !noneOpt.classList.contains('hidden'));
                    thinkChk.checked = noneAvailable ? !!(effortSel && String(effortSel.value || '').toLowerCase() !== 'none') : true;
                    thinkChk.disabled = !noneAvailable;
                } else if (thinkChk.dataset.thinkingChecked !== undefined) {
                    thinkChk.checked = thinkChk.dataset.thinkingChecked === '1';
                    delete thinkChk.dataset.thinkingChecked;
                }
            }
            function toggleOptionsForModel() {
                const modelEl = get('model-select');
                if (!modelEl) return;
                const model = modelEl.value;
                const modelLower = String(model || '').toLowerCase();
                const isDeepSeek = modelLower.includes('deepseek');
                const isZai = modelLower.startsWith('glm-');
                const thinkOpts = get('thinking-options');
                const reasonOpts = get('reasoning-effort-container');
                const thinkChk = get('enable-thinking');
                const thinkLvl = get('thinking-level');
                const thinkBudget = get('thinking-budget');
                const searchChk = get('enable-search');
                const searchCont = get('search-container');
                const urlCont = get('url-context-container');
                const mapsChk = get('enable-maps');
                const mapsCont = get('maps-grounding-container');
                const sysChk = get('enable-sys-prompt'); const sysLbl = get('sys-prompt-option');
                const pyChk = get('enable-python'); const pyCont = get('python-container');
                const cacheCont = get('prompt-cache-container');
                const cacheChk = get('enable-prompt-cache');
                const isSearchModel = model === 'gpt-5-search-api';
                const isTts = model.includes('tts');
                const isOcr = isMistralOcrModel(model);
                const isNanoBanana2Lite = modelLower.includes('gemini-3.1-flash-lite-image');
                const isNanoBanana21 = modelLower === 'gemini-nano-banana-2.1';
                const isNanoBanana2 = (modelLower.includes('gemini-3.1-flash-image') && !isNanoBanana2Lite) || isNanoBanana21;
                const isClaude = isClaudeModelKey(model);
                const isGeminiCyber = modelLower === 'gemini-3.8-flash-cyber';
                // DeepSeek KV cache and Z.AI context caching are automatic; neither uses an app-supplied prompt_cache_key.
                const promptCacheSupported = isLlmModel() && !isDeepSeek && !isTts && !modelLower.includes('realtime') && !modelLower.includes('native-audio') && !modelLower.includes('live');
                if (cacheCont) {
                    if (promptCacheSupported) {
                        cacheCont.classList.remove('hidden', 'opacity-50', 'pointer-events-none');
                        if (cacheChk) cacheChk.disabled = false;
                    } else {
                        if (cacheChk) { cacheChk.checked = false; cacheChk.disabled = true; }
                        cacheCont.classList.add('opacity-50', 'pointer-events-none');
                    }
                }
                updatePromptCacheUi();

                if (thinkOpts) thinkOpts.classList.add('hidden');
                if (reasonOpts) reasonOpts.classList.add('hidden');
                // 基本は Gemini と同じ Thinking（thinking-level）表示に戻し、
                // 対象モデルのときだけ下で effort 表示へ切り替える。
                setComposerThinkingMode('thinking');
                const vmi = get('vision-model-info');
                if (vmi) vmi.classList.add('hidden');
                if (reasonOpts) {
                    const effortSel = get('reasoning-effort');
                    if (effortSel) {
                        Array.from(effortSel.options).forEach(opt => {
                            const isGpt56Model = modelLower === 'gpt-5.6' || modelLower.startsWith('gpt-5.6-');
                            const isGpt6Model = modelLower.startsWith('gpt-6');
                            const gpt6AllowsNone = modelLower === 'gpt-6-sol' || modelLower === 'gpt-6-luna' || modelLower === 'gpt-6.1-terra';
                            // DeepSeek Flash family shares the low/high/max effort mapping.
                            const isDeepSeekFlash = modelLower === 'deepseek-v4.1-flash' || modelLower === 'deepseek-v4-flash-0731' || modelLower === 'deepseek-v4-flash' || modelLower === 'deepseek-v4-flash-vision-exp';
                            const isDeepSeekPro = modelLower === 'deepseek-v4-pro';
                            const isGrok45 = modelLower.includes('grok-4.5');
                            const isGrok46 = modelLower.includes('grok-4.6');
                            if (opt.value === 'max') {
                                opt.classList.toggle('hidden', !isGpt56Model && !isGpt6Model && !isDeepSeekFlash && !isDeepSeekPro);
                            } else if (opt.value === 'xhigh') {
                                opt.classList.toggle('hidden', !isGrok46 && !modelLower.includes('multi-agent') && !isGpt56Model && !isGpt6Model);
                            } else if (opt.value === 'medium') {
                                opt.classList.toggle('hidden', !(modelLower.includes('grok-4.3') || isGrok45 || isGrok46 || modelLower.includes('grok-4.20-0309-reasoning') || modelLower.includes('grok-build') || modelLower.includes('multi-agent') || modelLower.includes('gpt-5') || isGpt6Model || modelLower.includes('o1') || modelLower.includes('o3')));
                            } else if (opt.value === 'none') {
                                opt.classList.toggle('hidden', !modelLower.includes('grok-4.3') && !modelLower.includes('grok-build') && !modelLower.includes('gpt-5') && !gpt6AllowsNone && !isDeepSeekFlash && !isDeepSeekPro);
                            } else if (opt.value === 'low') {
                                opt.classList.toggle('hidden', isDeepSeekPro);
                            }
                        });
                        const selectedEffort = effortSel.selectedOptions && effortSel.selectedOptions[0];
                        if (selectedEffort && selectedEffort.classList.contains('hidden')) {
                            effortSel.value = isDeepSeek ? 'high' : 'medium';
                        }
                    }
                }
                if(urlCont) urlCont.classList.add('hidden');
                if(mapsCont) mapsCont.classList.add('hidden');
                if(thinkChk) thinkChk.disabled = false;

                if(thinkBudget) {
                    thinkBudget.disabled = true;
                    thinkBudget.classList.add('opacity-50');
                }
                const isGeminiImage = isGeminiImageModelKey(model);
                if(isTts || isOcr) {
                    if(searchCont) { get('enable-search').checked = false; searchCont.classList.add('opacity-50', 'pointer-events-none'); }
                    if(urlCont) { get('enable-url-context').checked = false; urlCont.classList.add('opacity-50', 'pointer-events-none'); }
                    if(mapsCont && mapsChk) { mapsChk.checked = false; mapsCont.classList.add('opacity-50', 'pointer-events-none'); }
                    if(pyCont) { pyChk.checked = false; pyCont.classList.add('opacity-50', 'pointer-events-none'); }
                    if (sysChk && sysLbl) { sysChk.checked = false; sysChk.disabled = true; sysLbl.classList.add('opacity-50'); }
                } else if (isNanoBanana2 || isNanoBanana2Lite) {
                    if (mapsCont && mapsChk) {
                        mapsChk.checked = false;
                        mapsCont.classList.add('hidden', 'opacity-50', 'pointer-events-none');
                    }
                    thinkOpts.classList.remove('hidden');
                    if (isNanoBanana21) {
                        Array.from(thinkLvl.options).forEach(opt => {
                            opt.disabled = opt.value === 'low';
                        });
                        if (!['minimal', 'medium', 'high'].includes(thinkLvl.value)) thinkLvl.value = 'medium';
                    } else {
                        Array.from(thinkLvl.options).forEach(opt => {
                            if (['low', 'medium'].includes(opt.value)) opt.disabled = true;
                            if (['minimal', 'high'].includes(opt.value)) opt.disabled = false;
                        });
                        if (!['minimal', 'high'].includes(thinkLvl.value)) {
                            thinkLvl.value = isNanoBanana2Lite ? 'minimal' : 'high';
                        }
                    }
                    if (thinkChk) thinkChk.disabled = false;
                    if (isNanoBanana2Lite) {
                        if (searchChk) {
                            searchChk.checked = false;
                            searchChk.disabled = true;
                        }
                        if (searchCont) searchCont.classList.add('opacity-50', 'pointer-events-none');
                    }
                } else if (isGeminiImage) {
                    if (mapsCont && mapsChk) {
                        mapsChk.checked = false;
                        mapsCont.classList.add('hidden', 'opacity-50', 'pointer-events-none');
                    }
                } else if (isClaude) {
                    thinkOpts.classList.remove('hidden');
                    if (thinkBudget) {
                        thinkBudget.disabled = false;
                        thinkBudget.classList.remove('opacity-50');
                    }
                    Array.from(thinkLvl.options).forEach(opt => {
                        opt.disabled = true; // Claude thinking only uses budget and enabled flag
                    });
                    if (pyCont) { pyChk.checked = false; pyCont.classList.add('opacity-50', 'pointer-events-none'); }
                } else if (isGeminiCyber) {
                    if (thinkOpts) thinkOpts.classList.remove('hidden');
                    if (thinkChk) { thinkChk.checked = true; thinkChk.disabled = true; }
                    Array.from(thinkLvl.options).forEach(opt => { opt.disabled = !['low', 'medium', 'high'].includes(opt.value); });
                    if (!['low', 'medium', 'high'].includes(thinkLvl.value)) thinkLvl.value = 'medium';
                    [searchCont, urlCont, mapsCont, pyCont].forEach(cont => { if (cont) cont.classList.add('opacity-50', 'pointer-events-none'); });
                    [searchChk, mapsChk, pyChk].forEach(chk => { if (chk) { chk.checked = false; chk.disabled = true; } });
                    const urlChk = get('enable-url-context');
                    if (urlChk) { urlChk.checked = false; urlChk.disabled = true; }
                    if (sysChk && sysLbl) { sysChk.disabled = false; sysLbl.classList.remove('opacity-50'); }
                } else if(model.includes('gemini') && !isGeminiImage) {
                    thinkOpts.classList.remove('hidden');
                    if(urlCont) {
                        urlCont.classList.remove('hidden', 'opacity-50', 'pointer-events-none');
                    }
                    const isGemini3 = model.includes('gemini-3') || model.includes('gemini-4');
                    if (mapsCont) {
                        if (isGemini3) {
                            mapsCont.classList.remove('hidden', 'opacity-50', 'pointer-events-none');
                        } else {
                            if (mapsChk) mapsChk.checked = false;
                            mapsCont.classList.add('hidden', 'opacity-50', 'pointer-events-none');
                        }
                    }
                    const isFlash = model.includes('flash');
                    const isHydrogen = model === 'gemini-4-hydrogen';
                    Array.from(thinkLvl.options).forEach(opt => {
                        if (isHydrogen) {
                            opt.disabled = !['low', 'medium', 'high', 'xhigh'].includes(opt.value);
                        } else if (model === 'gemini-4-carbon' || model === 'gemini-4-argon' || model === 'gemini-3.8-flash' || model === 'gemini-3.7-flash') {
                            opt.disabled = !['low', 'medium', 'high'].includes(opt.value);
                        } else if (model === 'gemini-3.6-flash') {
                            opt.disabled = !['medium', 'high'].includes(opt.value);
                        } else if (model === 'gemini-3.5-flash-lite') {
                            opt.disabled = !['minimal', 'medium', 'high'].includes(opt.value);
                        } else if(opt.value === 'xhigh') {
                            opt.disabled = true;
                        } else if(['minimal', 'medium'].includes(opt.value)) {
                            opt.disabled = !isFlash;
                        } else {
                            opt.disabled = false;
                        }
                    });
                    if (isHydrogen && !['low', 'medium', 'high', 'xhigh'].includes(thinkLvl.value)) {
                        thinkLvl.value = 'high';
                    } else if ((model === 'gemini-4-carbon' || model === 'gemini-4-argon' || model === 'gemini-3.8-flash' || model === 'gemini-3.7-flash') && !['low', 'medium', 'high'].includes(thinkLvl.value)) {
                        thinkLvl.value = 'medium';
                    } else if (model === 'gemini-3.6-flash' && !['medium', 'high'].includes(thinkLvl.value)) {
                        thinkLvl.value = 'medium';
                    } else if (model === 'gemini-3.5-flash-lite' && !['minimal', 'medium', 'high'].includes(thinkLvl.value)) {
                        thinkLvl.value = 'minimal';
                    } else if(!isFlash && ['minimal', 'medium'].includes(thinkLvl.value)) {
                        thinkLvl.value = 'high';
                    }
                    if(isGemini3) {
                        if(thinkChk) { thinkChk.checked = true; thinkChk.disabled = true; }
                    } else if(thinkChk) {
                        thinkChk.disabled = false;
                    }
                    if(thinkBudget && model.includes('gemini-2.5')) {
                        thinkBudget.disabled = false;
                        thinkBudget.classList.remove('opacity-50');
                    }
                    if(thinkBudget && !model.includes('gemini-2.5')) {
                        thinkBudget.disabled = true;
                        thinkBudget.classList.add('opacity-50');
                    }
                }
                const supportsReasoningEffort = isLlmModel() && (modelLower.includes('gpt-5') || modelLower.includes('o1') || modelLower.includes('o3') || modelLower.includes('grok-4.3') || modelLower.includes('grok-4.5') || modelLower.includes('grok-4.6') || modelLower.includes('grok-4.20-0309-reasoning') || modelLower.includes('grok-build') || modelLower.includes('multi-agent') || (modelLower.includes('gpt') && !modelLower.includes('tts')));
                if (supportsReasoningEffort) {
                    if (thinkOpts) thinkOpts.classList.remove('hidden');
                    setComposerThinkingMode('effort');
                    if (searchCont) searchCont.classList.remove('opacity-50', 'pointer-events-none');
                } else if (isDeepSeek) {
                    if (thinkOpts) thinkOpts.classList.remove('hidden');
                    setComposerThinkingMode('effort');
                    // V4.1 Flash handles images natively; only text-only DeepSeek models need the vision-model notice.
                    const vmi = get('vision-model-info');
                    if (vmi) vmi.classList.toggle('hidden', modelLower === 'deepseek-v4.1-flash' || modelLower === 'deepseek-v4-flash-vision-exp');
                    if (searchChk) searchChk.disabled = false;
                    if (searchCont) searchCont.classList.remove('opacity-50', 'pointer-events-none');
                    if (urlCont) {
                        const urlChk = get('enable-url-context');
                        if (urlChk) urlChk.checked = false;
                        urlCont.classList.add('opacity-50', 'pointer-events-none');
                    }
                    if (mapsCont && mapsChk) {
                        mapsChk.checked = false;
                        mapsCont.classList.add('opacity-50', 'pointer-events-none');
                    }
                }
                else if (!isOcr) {
                    if(searchCont) searchCont.classList.remove('opacity-50', 'pointer-events-none');
                    if(mapsCont && mapsChk) {
                        mapsChk.checked = false;
                        mapsCont.classList.add('hidden', 'opacity-50', 'pointer-events-none');
                    }
                }

                // TTS Special Handling
                if(isTts) {
                    if(pyCont) pyCont.classList.add('opacity-50', 'pointer-events-none');
                } else {
                    if(pyCont) pyCont.classList.remove('opacity-50', 'pointer-events-none');
                    if((!isGeminiImage || isNanoBanana2) && !model.includes('gpt-image')) {
                         sysChk.disabled = false; sysLbl.classList.remove('opacity-50');
                    }
                }

                if(((isGeminiImage && !isNanoBanana2) || model.includes('gpt-image') || isGrokImageModel() || isIdeogramModel(model) || isGrokVideoModel() || isOcr)) { if (sysChk && sysLbl) { sysChk.checked = false; sysChk.disabled = true; sysLbl.classList.add('opacity-50'); } }
                if (pyCont) {
                    if (isLlmModel()) {
                        pyCont.classList.remove('hidden');
                        pyChk.disabled = false;
                    } else {
                        pyChk.checked = false;
                        pyChk.disabled = true;
                        pyCont.classList.add('hidden');
                    }
                }
                if (isSearchModel) {
                    if (searchChk) {
                        searchChk.checked = true;
                        searchChk.disabled = true;
                    }
                    if (searchCont) searchCont.classList.add('opacity-50', 'pointer-events-none');
                    if (pyCont) {
                        pyChk.checked = false;
                        pyChk.disabled = true;
                        pyCont.classList.add('opacity-50', 'pointer-events-none');
                    }
                } else if (searchChk && !model.includes('tts') && !isOcr && !isNanoBanana2Lite) {
                    searchChk.disabled = false;
                }
                if (isGeminiCyber) {
                    [searchChk, mapsChk, pyChk].forEach(chk => { if (chk) { chk.checked = false; chk.disabled = true; } });
                    const cyberUrlChk = get('enable-url-context');
                    if (cyberUrlChk) { cyberUrlChk.checked = false; cyberUrlChk.disabled = true; }
                    [searchCont, urlCont, mapsCont, pyCont].forEach(cont => { if (cont) cont.classList.add('opacity-50', 'pointer-events-none'); });
                }
                if (isZai) {
                    if (vmi) vmi.classList.toggle('hidden', ['glm-5.3-flash', 'glm-5.3-flashx', 'glm-4.6v', 'glm-4.6v-flashx', 'glm-4.6v-flash', 'glm-4.5v'].includes(modelLower));
                    [searchChk, pyChk].forEach(chk => { if (chk) { chk.checked = false; chk.disabled = true; } });
                    [searchCont, pyCont].forEach(cont => { if (cont) cont.classList.add('opacity-50', 'pointer-events-none'); });
                }
                const maskBtn = get('mask-btn');
                if (maskBtn) {
                    if (isGptImageModel() || isIdeogramObjectRemoval()) {
                        maskBtn.classList.remove('hidden');
                    } else {
                        maskBtn.classList.add('hidden');
                        currentMaskImage = null;
                        updateMaskPreview();
                    }
                }
                updateTtsUi();
                updateStsUi();
                updateStsOptions();
                updateGptImageUi();
                updateGeminiImageUi();
                updateGrokImageUi();
                updateIdeogramImageUi();
                updateGrokVideoUi();
                updateGeminiVideoUi();
                updateGeminiMusicUi();
                updateBatchUi(model);
                const freeCont = get('gemini-free-tier-container');
                if (freeCont) {
                    const showFree = typeof isGeminiApiModel === 'function' && isGeminiApiModel(model)
                        && (typeof currentGeminiBackend !== 'function' || currentGeminiBackend() !== 'vertex_ai');
                    freeCont.classList.toggle('hidden', !showFree);
                }
                updateXaiChatUi();
                updateMistralOcrUi();
                updateImageInputLimits();
                purgeUnsupportedAttachments(true);
                refreshMinimalOptionsIfOpen();
                applyMcpPromptChipUi();
            }
            if (get('model-select')) {
                get('model-select').addEventListener('change', toggleOptions);
                get('model-select').addEventListener('change', () => schedulePromptTokenEstimate(true));
            }
            bindPromptCacheControls();
            toggleOptions();
            if (minimalPromptMode) setMinimalPromptMode(true);
            else setCompactPromptMode(compactPromptMode, true);
            renderWelcomeQuickStart();
            const canvasModeCheckbox = get('enable-canvas-mode');
            if (canvasModeCheckbox) {
                canvasModeCheckbox.checked = canvasModeEnabled;
                canvasModeCheckbox.addEventListener('change', () => syncCanvasModeUi(canvasModeCheckbox.checked));
            }
            syncCanvasModeUi(canvasModeEnabled, { persist: false, skipReset: false });
            const codingModeCheckbox = get('enable-coding-mode');
            if (codingModeCheckbox) {
                codingModeCheckbox.checked = codingModeEnabled;
                codingModeCheckbox.addEventListener('change', () => syncCodingModeUi(codingModeCheckbox.checked));
            }
            if (get('clear-coding-target-btn')) {
                get('clear-coding-target-btn').addEventListener('click', () => {
                    codingTargetSelection = null;
                    syncCodingModeUi(codingModeEnabled, { persist: false });
                    showToast('最新のコードブロックを自動選択します', 'info', false);
                });
            }
            syncCodingModeUi(codingModeEnabled, { persist: false });
            if (get('canvas-panel-close-btn')) {
                get('canvas-panel-close-btn').addEventListener('click', () => syncCanvasModeUi(false));
            }
            if (get('canvas-panel-clear-btn')) {
                get('canvas-panel-clear-btn').addEventListener('click', () => {
                    if (!canvasModeEnabled) return;
                    resetCanvasPreviewPanel();
                    showToast('Canvasプレビューをクリアしました', 'info', false);
                });
            }
            if (get('canvas-block-list')) {
                get('canvas-block-list').addEventListener('click', (e) => {
                    const btn = e.target.closest('[data-canvas-block-index]');
                    if (!btn) return;
                    const index = Number(btn.getAttribute('data-canvas-block-index'));
                    applyCanvasSelection(index, {
                        view: 'preview',
                        animateView: true,
                        transitionFrom: 'blocks'
                    });
                });
            }
            if (get('canvas-source-select')) {
                get('canvas-source-select').addEventListener('change', (e) => {
                    if (e.target.value === '') return;
                    const index = Number(e.target.value);
                    if (!Number.isInteger(index)) return;
                    applyCanvasSelection(index, { view: 'source' });
                });
            }
            if (get('canvas-panel-tabs')) {
                get('canvas-panel-tabs').addEventListener('click', (e) => {
                    const btn = e.target.closest('[data-canvas-panel-view]');
                    if (!btn) return;
                    const view = btn.getAttribute('data-canvas-panel-view');
                    syncCanvasPanelViewUi(view, { focus: false });
                });
            }
            if (get('canvas-panel-copy-btn')) {
                get('canvas-panel-copy-btn').addEventListener('click', () => {
                    const els = getCanvasModeElements();
                    const text = els && els.code ? (els.code.textContent || '') : '';
                    if (!text.trim()) {
                        showToast('コピーするコードがありません', 'info', false);
                        return;
                    }
                    copyToClipboard(text, () => showToast('Canvasコードをコピーしました', 'success'), () => showToast('コピーに失敗しました', 'error', true));
                });
            }
            const promptControlsToggleBtn = get('prompt-controls-toggle-btn');
            if (promptControlsToggleBtn) {
                promptControlsToggleBtn.onclick = () => togglePromptControlDetails();
            }
            if (get('tts-voice')) get('tts-voice').addEventListener('change', updateTtsUi);
            if (get('gpt-image-format')) get('gpt-image-format').addEventListener('change', () => updateGptImageUi());
            if (get('gemini-image-size')) get('gemini-image-size').addEventListener('change', () => updateGeminiImageUi());
            if (get('tts-speed') && get('tts-speed-label')) {
                get('tts-speed').addEventListener('input', () => {
                    get('tts-speed-label').textContent = `${Number(get('tts-speed').value || 1).toFixed(2)}x`;
                });
            }
            if (get('sts-speed') && get('sts-speed-label')) {
                get('sts-speed').addEventListener('input', () => {
                    get('sts-speed-label').textContent = `${Number(get('sts-speed').value || 1).toFixed(2)}x`;
                });
            }
            if (window.marked && typeof window.marked.use === 'function') window.marked.use({
                renderer: {
                    code(c, i, e) {
                        const l = (i || '').match(/\S*/)[0];
                        if (l === 'pyexec') {
                            // Completed answers surface Python runs via the bubble footer button.
                            // Keep the fence out of the inline answer body.
                            return '';
                        }
                        if (l === 'chat_error') {
                            // Persisted generation errors (saved to DB so reloads still show them).
                            return buildChatErrorBubbleHtml(c || '');
                        }
                        const raw = c || '';
                        const lowerLang = (l || '').toLowerCase();
                        if (lowerLang === 'mathsvg') {
                            // 数学ステップの図: 画像だけを描き、コード用のUIは出さない。
                            const mathSvgCode = getRenderableSvgCode('mathsvg', raw);
                            return mathSvgCode && typeof window.buildMathSvgRenderHtml === 'function'
                                ? window.buildMathSvgRenderHtml(mathSvgCode)
                                : '';
                        }
                        let h = '';
                        try {
                            const lang = hljs.getLanguage(l) ? l : 'plaintext';
                            // Optimization: Skip heavy highlighting for very large blocks during streaming
                            if (activeStreamingBubbleId && raw.length > 20000) {
                                h = escapeHtml(raw);
                            } else {
                                h = hljs.highlight(raw, { language: lang }).value;
                            }
                        } catch (err) {
                            h = escapeHtml(raw);
                        }
                        const enc = encodeURIComponent(raw).replace(/'/g, "%27");
                        const isSuspicious = detectBlockedScriptsInCode(raw);
                        const codeKey = hashString(`${l || 'TEXT'}\n${raw || ''}`);
                        let previewBtn = '';
                        if (canvasModeEnabled) {
                            const isActiveCanvasBlock = String(canvasPreviewState.selectedKey || '') === codeKey;
                            const canvasTitle = isActiveCanvasBlock ? 'Canvasで表示中' : 'Canvasでプレビューする';
                            previewBtn = `<button class="canvas-preview-btn${isActiveCanvasBlock ? ' canvas-active' : ''}" data-code="${enc}" data-code-key="${codeKey}" data-canvas-lang="${escapeHtml(l || 'txt')}" title="${canvasTitle}" aria-label="${canvasTitle}" aria-pressed="${isActiveCanvasBlock ? 'true' : 'false'}"><i class="fas ${isActiveCanvasBlock ? 'fa-layer-group' : 'fa-window-restore'}"></i></button>`;
                        } else if (isHtmlPreviewCandidate(lowerLang, raw)) {
                            const label = isSuspicious ? 'セーフプレビュー' : 'プレビュー';
                            const icon = isSuspicious ? 'fa-shield-halved' : 'fa-up-right-from-square';
                            previewBtn = `<button class="html-preview-btn" data-code="${enc}" ${isSuspicious ? 'data-suspicious="1"' : ''} title="${label}" aria-label="${label}"><i class="fas ${icon}"></i></button>`;
                        }
                        const downloadBtn = `<button class="download-btn" data-code="${enc}" data-lang="${l || 'txt'}" title="ダウンロード" aria-label="ダウンロード"><i class="fas fa-download"></i></button>`;
                        const codingBtn = lowerLang === 'diff'
                            ? ''
                            : `<button class="coding-target-btn" data-code="${enc}" data-code-key="${codeKey}" data-coding-lang="${escapeHtml(l || 'text')}" aria-pressed="false" title="Coding Modeの編集対象に指定" aria-label="編集対象に指定"><i class="fas fa-quote-right"></i></button>`;
                        const langLabel = (l || 'TEXT') + (isSuspicious ? ' <span class="suspicious-badge" title="polyfill.io などの危険スクリプトURLを検出しました">⚠</span>' : '');
                        const svgCode = getRenderableSvgCode(lowerLang, raw);
                        const svgRender = svgCode ? buildSvgCodeRenderHtml(svgCode, codeKey) : '';
                        return `${svgRender}<div class="code-wrapper collapsed" data-collapsed="true" data-code-key="${codeKey}"><div class="code-header"><span class="code-lang">${langLabel}</span><div class="code-actions"><button class="code-toggle" aria-expanded="false" title="展開" aria-label="展開"><i class="fas fa-chevron-down"></i></button>${codingBtn}${previewBtn}${downloadBtn}<button class="copy-btn" data-code="${enc}" title="コピー" aria-label="コピー"><i class="fas fa-copy"></i></button></div></div><div class="code-body"><pre><code class="hljs language-${l}">${h}</code></pre></div></div>`;
                    },
                    link(h, t, x) { return `<a href="${h}" title="${t || ''}" target="_blank">${x}</a>`; },
                    image(h, t, x) { return buildChatImageHtml(h, { alt: x, title: t }); }
                },
                breaks: true,
                gfm: true
            });

            // Infinite Scroll Observer
            threadObserver = new IntersectionObserver((entries) => {
                if(entries[0].isIntersecting && hasMoreThreads) loadThreads(true);
            }, { root: get('thread-list'), threshold: 0.1 });
            threadObserver.observe(get('scroll-sentinel'));

            initLowBandwidthMode();
            checkVersion();
            get('version-update-dismiss')?.addEventListener('click', () => {
                const latest = localStorage.getItem("app_version") || "";
                if (latest) localStorage.setItem("version_notified", latest);
                hideModal('version-update-modal');
            });
            const versionUpdateClearCacheSetting = get('version-update-clear-cache');
            if (versionUpdateClearCacheSetting) {
                versionUpdateClearCacheSetting.checked = !!(window.CHAT_CONFIG && window.CHAT_CONFIG.clearCacheOnVersionUpdate);
                versionUpdateClearCacheSetting.addEventListener('change', () => {
                    versionUpdateCachePreferenceSavePromise = saveVersionUpdateCachePreference(versionUpdateClearCacheSetting.checked);
                });
            }
            get('version-update-reload')?.addEventListener('click', async () => {
                await versionUpdateCachePreferenceSavePromise.catch(() => {});
                const clearCacheEnabled = !!get('version-update-clear-cache')?.checked;
                if (clearCacheEnabled) {
                    await clearSiteCacheAndReload(get('version-update-reload'), { scanFirst: true });
                } else {
                    location.reload();
                }
            });
            if (window.ConnectionMonitor) {
                window.ConnectionMonitor.setVersionChangeHandler((hbVersion) => {
                    if (hbVersion && hbVersion !== appVersion) {
                        const hbNotified = localStorage.getItem("version_notified") || "";
                        if (hbNotified !== hbVersion) {
                            localStorage.setItem("app_version", hbVersion);
                            purgeCaches().then(() => checkAndNotifyVersion(hbVersion));
                        }
                    }
                });
                window.ConnectionMonitor.start();
                window.addEventListener('online', () => window.ConnectionMonitor.probeNow());
                window.addEventListener('offline', () => {
                    window.ConnectionMonitor.cancelProbe();
                    window.ConnectionMonitor.setUnavailable('offline');
                });
                window.addEventListener('focus', () => window.ConnectionMonitor.probeNow());
                document.addEventListener('visibilitychange', () => {
                    if (!document.hidden) window.ConnectionMonitor.probeNow();
                });
                window.addEventListener('pagehide', () => window.ConnectionMonitor.stop());
            }
            applyCacheMode(useSwCache);
            // If the account is temporarily locked, show the lock screen before
            // doing anything else so the user sees the reason and remaining time.
            // Admins are never locked (ban/lock monitoring exemption).
            if (botConfig && botConfig.lock && botConfig.lock.active && !isAdminUser) {
                showBotLockOverlay(botConfig.lock.message, botConfig.lock.remaining_seconds);
            }
            if (window.__turnstileApiLoaded && window.initTurnstileWidget) window.initTurnstileWidget();
            if (botConfig && botConfig.globalEnabled && botConfig.accountEnabled && !isAdminUser) {
                if (botConfig.turnstileVerified) botDetectionVerified = true;
                try { botTelemetry.start(); } catch (e) { console.error(e); }
                try { runBotDetectionGate(); } catch (e) { console.error(e); }
            } else {
                const container = get('turnstile-container');
                if (container) container.classList.add('hidden');
            }
            const formatSessionTime = (val) => {
                if (!val) return '不明';
                const d = new Date(val);
                if (Number.isNaN(d.getTime())) return val;
                return d.toLocaleString();
            };
            const renderPasskeyList = (items) => {
                const list = Array.isArray(items) ? items : [];
                const wrap = get('passkey-list');
                const countEl = get('passkey-count');
                if (countEl) countEl.innerText = String(list.length);
                if (!wrap) return;
                if (!list.length) {
                    wrap.innerHTML = '<div class="text-[11px] text-gray-500">登録済みのパスキーはありません。</div>';
                    return;
                }
                wrap.innerHTML = '';
                list.forEach((item, idx) => {
                    const credId = item && item.id ? String(item.id) : '';
                    const row = document.createElement('div');
                    row.className = 'bg-gray-800/60 border border-gray-700 rounded p-2 flex items-center justify-between gap-2';
                    const left = document.createElement('div');
                    left.className = 'min-w-0';
                    const nameEl = document.createElement('div');
                    nameEl.className = 'text-xs text-gray-200 truncate';
                    nameEl.innerText = (item && item.name) ? String(item.name) : `Security Key ${idx + 1}`;
                    const metaEl = document.createElement('div');
                    metaEl.className = 'text-[10px] text-gray-500 mt-1';
                    metaEl.innerText = item && item.created_at ? `登録日時: ${formatSessionTime(item.created_at)}` : '登録日時: 不明';
                    left.appendChild(nameEl);
                    left.appendChild(metaEl);
                    row.appendChild(left);
                    const btn = document.createElement('button');
                    btn.type = 'button';
                    btn.className = 'bg-red-700 hover:bg-red-600 text-white px-2 py-1 rounded text-[10px] font-bold btn-hover shrink-0';
                    btn.innerText = '削除';
                    btn.disabled = !credId;
                    if (credId) {
                        btn.onclick = () => window.removeWebAuthnCredential(credId);
                    }
                    row.appendChild(btn);
                    wrap.appendChild(row);
                });
            };
            const renderSessions = (sessions) => {
                const list = get('session-list');
                if (!list) return;
                if (!sessions || !sessions.length) {
                    list.innerHTML = '<div class="text-xs text-gray-500">アクティブなセッションはありません。</div>';
                    return;
                }
                list.innerHTML = sessions.map((s) => {
                    const currentBadge = s.is_current ? '<span class="text-[10px] bg-blue-600 text-white px-1.5 py-0.5 rounded">現在</span>' : '';
                    const revokedBadge = s.is_revoked ? '<span class="text-[10px] bg-gray-700 text-gray-300 px-1.5 py-0.5 rounded">失効</span>' : '';
                    const actionBtn = (!s.is_current && !s.is_revoked) ? `<button data-session-id="${escapeHtml(s.id)}" class="session-revoke-btn bg-gray-700 hover:bg-gray-600 text-white px-3 py-1 rounded text-[11px] font-bold btn-hover">ログアウト</button>` : '';
                    const ua = (s.user_agent || 'Unknown').slice(0, 120);
                    const ip = s.ip_address || 'Unknown';
                    return `<div class="ui-enter-item bg-gray-800/60 border border-gray-700 rounded p-3 flex items-center justify-between gap-3"><div class="min-w-0"><div class="flex items-center gap-2 mb-1">${currentBadge}${revokedBadge}<div class="text-xs text-gray-200">${escapeHtml(ip)}</div></div><div class="text-[11px] text-gray-400 truncate">${escapeHtml(ua)}</div><div class="text-[10px] text-gray-500 mt-1">最終アクセス: ${escapeHtml(formatSessionTime(s.last_seen_at))} / 作成: ${escapeHtml(formatSessionTime(s.created_at))}</div></div>${actionBtn}</div>`;
                }).join('');
                list.querySelectorAll('.session-revoke-btn').forEach((btn) => {
                    btn.onclick = async () => {
                        const id = btn.getAttribute('data-session-id');
                        if (!id) return;
                        if (!confirm('このセッションをログアウトしますか？')) return;
                        const res = await apiFetch('/api/sessions/revoke', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({id})});
                        let data = {};
                        try { data = await res.json(); } catch (e) {}
                        if (res.ok) {
                            if (data.logged_out) {
                                location.href = '/login';
                                return;
                            }
                            await loadSessions();
                        } else {
                            showToast((data && data.error) || 'ログアウトに失敗しました', 'error', true);
                        }
                    };
                });
            };
            const loadSessions = async () => {
                const list = get('session-list');
                if (list) list.innerHTML = '<div class="text-xs text-gray-500">読み込み中...</div>';
                const res = await apiFetch('/api/sessions');
                let data = {};
                try { data = await res.json(); } catch (e) {}
                if (!res.ok) {
                    if (data && data.error === 'session_revoked') {
                        location.href = '/login';
                        return;
                    }
                    if (list) list.innerHTML = '<div class="text-xs text-red-400">セッションの取得に失敗しました。</div>';
                    return;
                }
                const sessions = (data.sessions || []).filter(s => !s.is_revoked);
                renderSessions(sessions);
            };
            const bindSessionButtons = () => {
                const refreshBtn = get('session-refresh-btn');
                if (refreshBtn) refreshBtn.onclick = () => loadSessions();
                const revokeOthersBtn = get('session-revoke-others-btn');
                if (revokeOthersBtn) revokeOthersBtn.onclick = async () => {
                    if (!confirm('現在の端末以外をログアウトしますか？')) return;
                    const res = await apiFetch('/api/sessions/revoke_others', {method:'POST'});
                    if (res.ok) {
                        await loadSessions();
                    } else {
                        showToast('操作に失敗しました', 'error', true);
                    }
                };
                const revokeAllBtn = get('session-revoke-all-btn');
                if (revokeAllBtn) revokeAllBtn.onclick = async () => {
                    if (!confirm('全セッションを強制ログアウトします。よろしいですか？')) return;
                    const res = await apiFetch('/api/sessions/revoke_all', {method:'POST'});
                    if (res.ok) {
                        location.href = '/login';
                    } else {
                        showToast('操作に失敗しました', 'error', true);
                    }
                };
            };
            ensureUserSettingsSnapshot().then(d => {
                if (d) {
                    currentVisionModel = d.default_vision_model || 'gemini-3-flash-preview';
                }
                applyChatDefaults(d);
                try { loadMcpServers(); } catch (e) {}
                if (d && d.theme_color) {
                    applyThemeColor(d.theme_color, true);
                }
                if (d && Object.prototype.hasOwnProperty.call(d, 'minimal_prompt_mode') && d.minimal_prompt_mode) {
                    setMinimalPromptMode(true);
                } else if (d && Object.prototype.hasOwnProperty.call(d, 'compact_prompt_mode')) {
                    setCompactPromptMode(!!d.compact_prompt_mode);
                }
                if (get('set-server-diagnostic-log-retention-hours')) get('set-server-diagnostic-log-retention-hours').value = d.server_diagnostic_log_retention_hours || 1;
                if (get('set-server-diagnostic-log')) get('set-server-diagnostic-log').checked = d.enable_server_diagnostic_log === true;
                if (get('set-client-debug-log')) {
                    syncClientDebugLogToggle(d.enable_client_debug_log === true, 'settings sync');
                }
                const sysChk = get('enable-sys-prompt');
                if (sysChk && d && d.system_prompt && String(d.system_prompt).trim()) {
                    if (!sysChk.disabled && !d.default_enable_system_prompt && !d.use_last_chat_settings) sysChk.checked = true;
                    toggleOptions();
                }
            }).catch(() => {});
            installAdminSidebarDebugObserver();
            if (isAdminSidebarDebugEnabled()) {
                try {
                    nativeConsoleInfo(ADMIN_SIDEBAR_DEBUG_PREFIX, 'enabled. Open the browser DevTools Console (F12). After reproducing, run copyAdminSidebarDebug() and paste the result.');
                } catch (error) {}
            }
            snapshotSidebarHistory('page-init');
            loadThreads(); loadSkills();

            get('send-btn').onclick = () => { if (isStopMode) stopGeneration(); else sendMessage(); };
            get('new-chat-btn').onclick = () => startNewChat();
            bindUploadButton();
            bindThinkingEffortSync();
            bindMinimalOptionsEvents();
            const vmBtn = get('vision-model-change-btn');
            if (vmBtn) vmBtn.onclick = () => _openVisionModelSelector();
            const onlyEl = get('compression-format-only');
            if (onlyEl) {
                onlyEl.onchange = () => {
                    const disabled = onlyEl.checked;
                    const sizeEl = get('compression-max-size');
                    const dimEl = get('compression-max-dim');
                    if (sizeEl) sizeEl.disabled = disabled;
                    if (dimEl) dimEl.disabled = disabled;
                    const sizeWrap = get('compression-size-wrap');
                    const dimWrap = get('compression-dim-wrap');
                    if (sizeWrap) sizeWrap.style.opacity = disabled ? '0.4' : '1';
                    if (dimWrap) dimWrap.style.opacity = disabled ? '0.4' : '1';
                };
            }
            const bindTemporaryChatToggle = () => {
                const temporaryChatChk = get('enable-temporary-chat');
                if (!temporaryChatChk || temporaryChatChk.dataset.bound === '1') return;
                temporaryChatChk.dataset.bound = '1';
                temporaryChatChk.checked = !!temporaryChatEnabled;
                temporaryChatChk.onchange = async () => {
                    const prev = temporaryChatEnabled;
                    const ok = await applyTemporaryChatSetting(temporaryChatChk.checked);
                    if (!ok) {
                        setTemporaryChatUiState(prev);
                        ensureTemporaryChatHeartbeat(false);
                    }
                };
            };
            bindTemporaryChatToggle();
            document.addEventListener('visibilitychange', () => {
                if (document.visibilityState === 'visible') {
                    ensureTemporaryChatHeartbeat(true);
                }
            });
            window.addEventListener('focus', () => {
                ensureTemporaryChatHeartbeat(true);
            });
            window.addEventListener('beforeunload', () => {
                stopTemporaryChatHeartbeat();
                stopCameraCaptureStream();
            });
            const storageRefreshBtn = get('storage-usage-refresh');
            if (storageRefreshBtn) storageRefreshBtn.onclick = () => loadStorageUsage();
            let activeAccountTransfer = null;
            const createAccountTransferId = () => {
                const bytes = new Uint8Array(16);
                window.crypto.getRandomValues(bytes);
                return Array.from(bytes, byte => byte.toString(16).padStart(2, '0')).join('');
            };
            const renderAccountTransferProgress = (payload = {}) => {
                const wrap = get('account-transfer-progress');
                const bar = get('account-transfer-progress-bar');
                const percent = get('account-transfer-progress-percent');
                const textEl = get('account-transfer-progress-text');
                const detail = get('account-transfer-progress-detail');
                const value = Math.max(0, Math.min(100, Number(payload.progress) || 0));
                if (wrap) wrap.classList.remove('hidden');
                if (bar) bar.style.width = `${value}%`;
                if (percent) percent.textContent = `${Math.round(value)}%`;
                if (textEl) textEl.textContent = payload.message || '処理状況を確認しています';
                if (detail) {
                    const labels = {
                        queued: '順番待ち', preparing: 'データを準備中', exporting_files: 'ファイルを書き出し中',
                        finalizing: '最終処理中', ready: 'ダウンロード準備完了', downloading: 'ダウンロード中',
                        uploading: 'ZIPをアップロード中', validating: 'ZIPを検証中', validating_files: 'ファイル情報を検証中',
                        reading_files: 'ファイルを読み込み中', importing_settings: '設定を反映中',
                        importing_credentials: '認証情報を反映中', importing_skills: 'Skillを追加中',
                        saving_files: 'ファイルを保存中', importing_chats: 'チャット履歴を追加中',
                        importing_feedback: 'フィードバックを追加中', importing_diagnostics: '診断データを追加中',
                        cancelling: 'キャンセル処理中', cancelled: 'キャンセル済み', expired: '保存期限切れ',
                        completed: '完了', failed: '失敗'
                    };
                    detail.textContent = labels[payload.phase] || '処理状況を確認しています。';
                }
                const cancelBtn = get('account-transfer-cancel-btn');
                if (cancelBtn) {
                    cancelBtn.classList.toggle('hidden', ['ready', 'completed', 'failed', 'cancelled', 'expired'].includes(payload.phase));
                }
            };
            const setAccountTransferControls = (running) => {
                if (accountExportBtn) accountExportBtn.disabled = !!running;
                const importBtn = get('account-import-btn');
                if (importBtn) importBtn.disabled = !!running;
                const cancelBtn = get('account-transfer-cancel-btn');
                if (cancelBtn) cancelBtn.disabled = !running;
            };
            const renderAccountExportAvailability = (payload = {}) => {
                const wrap = get('account-export-ready');
                const textEl = get('account-export-ready-text');
                const expiryEl = get('account-export-expiry');
                const downloadBtn = get('account-export-download-btn');
                const available = !!(payload.available && payload.download_url);
                if (wrap) wrap.classList.toggle('hidden', !available);
                if (!available) {
                    if (downloadBtn) downloadBtn.removeAttribute('href');
                    return;
                }
                const size = Math.max(0, Number(payload.size_bytes) || 0);
                const sizeLabel = size >= 1024 * 1024 * 1024
                    ? `${(size / (1024 * 1024 * 1024)).toFixed(2)} GB`
                    : `${(size / (1024 * 1024)).toFixed(1)} MB`;
                if (textEl) {
                    const warning = Number(payload.unreadable_count) > 0
                        ? `（読取不能 ${Number(payload.unreadable_count)}件を復旧用として収録）`
                        : '';
                    textEl.textContent = `エクスポートZIPをダウンロードできます：${sizeLabel}${warning}`;
                }
                if (expiryEl) {
                    const expiry = payload.expires_at ? new Date(payload.expires_at) : null;
                    expiryEl.textContent = expiry && !Number.isNaN(expiry.getTime())
                        ? `保存期限：${expiry.toLocaleString()}（期限後に自動削除）`
                        : '完成から1時間後に自動削除されます。';
                }
                if (downloadBtn) downloadBtn.href = payload.download_url;
            };
            const pollAccountTransfer = async (transfer) => {
                while (activeAccountTransfer === transfer && !transfer.stopped) {
                    try {
                        const res = await apiFetch(`/api/account/transfer/${transfer.id}`, manualSpinnerRequestOptions({
                            cache: 'no-store',
                        }));
                        const data = await res.json().catch(() => ({}));
                        if (res.ok) {
                            // A missing Redis status is reported as "pending"; do not
                            // overwrite client-side upload progress while the server
                            // has not yet begun the actual import.
                            if (data.state !== 'pending') renderAccountTransferProgress(data);
                            if (['ready', 'completed', 'failed', 'cancelled', 'expired'].includes(data.state)) return data;
                        }
                    } catch (_) {}
                    await new Promise(resolve => setTimeout(resolve, 700));
                }
                return null;
            };
            const handleFinishedAccountExport = (transfer, data, notify = true) => {
                if (!data) return;
                renderAccountTransferProgress(data);
                renderAccountExportAvailability(data);
                if (notify && data.state === 'ready') {
                    showToast(data.message || 'エクスポートZIPの準備が完了しました', Number(data.unreadable_count) > 0 ? 'warning' : 'success', Number(data.unreadable_count) > 0);
                } else if (notify && data.state === 'failed') {
                    showToast(data.message || 'エクスポートに失敗しました', 'error', true);
                }
                finishAccountTransfer(transfer);
            };
            const refreshLatestAccountExport = async () => {
                try {
                    const res = await apiFetch('/api/account/export/latest', manualSpinnerRequestOptions({cache: 'no-store'}));
                    const data = await res.json().catch(() => ({}));
                    if (!res.ok) return;
                    renderAccountExportAvailability(data);
                    if (data.state === 'ready') {
                        renderAccountTransferProgress(data);
                        return;
                    }
                    if (['failed', 'cancelled', 'expired'].includes(data.state)) {
                        // Show a terminal state left over from a previous session
                        // (e.g. the background job was interrupted) instead of
                        // silently showing nothing after the user returns.
                        renderAccountTransferProgress(data);
                        return;
                    }
                    if (!['queued', 'running', 'cancelling'].includes(data.state) || !data.job_id) return;
                    if (activeAccountTransfer && activeAccountTransfer.id === data.job_id) return;
                    if (activeAccountTransfer) return;
                    const transfer = {id: data.job_id, type: 'export', stopped: false, restored: true};
                    activeAccountTransfer = transfer;
                    setAccountTransferControls(true);
                    renderAccountTransferProgress(data);
                    const finished = await pollAccountTransfer(transfer);
                    if (finished) handleFinishedAccountExport(transfer, finished, true);
                } catch (_) {}
            };
            const finishAccountTransfer = (transfer) => {
                if (activeAccountTransfer === transfer) activeAccountTransfer = null;
                transfer.stopped = true;
                setAccountTransferControls(false);
            };
            const accountTransferCancelBtn = get('account-transfer-cancel-btn');
            if (accountTransferCancelBtn) {
                accountTransferCancelBtn.onclick = async () => {
                    const transfer = activeAccountTransfer;
                    if (!transfer || transfer.stopped) return;
                    transfer.cancelRequested = true;
                    accountTransferCancelBtn.disabled = true;
                    renderAccountTransferProgress({progress: 0, phase: 'cancelling', message: 'キャンセルしています'});
                    try {
                        await apiFetch(`/api/account/transfer/${transfer.id}/cancel`, manualSpinnerRequestOptions({
                            method: 'POST',
                        }));
                    } catch (_) {}
                    if (transfer.controller) transfer.controller.abort();
                    renderAccountTransferProgress({progress: 0, phase: 'cancelled', message: 'キャンセルしました'});
                    if (transfer.type === 'export') renderAccountExportAvailability({available: false});
                    finishAccountTransfer(transfer);
                    showToast('処理をキャンセルしました', 'info');
                };
            }
            const accountExportBtn = get('account-export-btn');
            if (accountExportBtn) {
                accountExportBtn.onclick = async () => {
                    if (activeAccountTransfer) return;
                    const transfer = {id: createAccountTransferId(), type: 'export', stopped: false};
                    activeAccountTransfer = transfer;
                    setAccountTransferControls(true);
                    renderAccountExportAvailability({available: false});
                    renderAccountTransferProgress({progress: 0, phase: 'queued', message: 'エクスポートを受け付けています'});
                    try {
                        const res = await apiFetch('/api/account/export', manualSpinnerRequestOptions({
                            method: 'POST', headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({job_id: transfer.id}), keepalive: true,
                        }));
                        const data = await res.json().catch(() => ({}));
                        if (res.status === 409 && data.error === 'export_in_progress' && data.job_id) {
                            transfer.id = data.job_id;
                        } else if (!res.ok) {
                            throw new Error(data.error === 'rate_limit' ? 'エクスポート回数の上限に達しました' : (data.error || 'エクスポートを開始できませんでした'));
                        }
                        renderAccountTransferProgress({progress: 0, phase: 'queued', message: 'バックグラウンドでエクスポートしています'});
                        const finished = await pollAccountTransfer(transfer);
                        if (!transfer.cancelRequested && finished) handleFinishedAccountExport(transfer, finished, true);
                    } catch (error) {
                        const message = error && error.message ? error.message : 'エクスポートを開始できませんでした';
                        renderAccountTransferProgress({progress: 0, phase: 'failed', message});
                        showToast(message, 'error', true);
                        finishAccountTransfer(transfer);
                    }
                };
            }
            const accountExportDownloadBtn = get('account-export-download-btn');
            if (accountExportDownloadBtn) {
                accountExportDownloadBtn.addEventListener('click', async (event) => {
                    const href = accountExportDownloadBtn.getAttribute('href');
                    if (!href || href === '#') return;
                    // Re-verify availability immediately before navigating so a
                    // stale/expired download URL never sends the user to a
                    // full-page 404. The archive may have been auto-deleted
                    // between the status check and the click.
                    event.preventDefault();
                    try {
                        const res = await apiFetch('/api/account/export/latest', manualSpinnerRequestOptions({cache: 'no-store'}));
                        const data = await res.json().catch(() => ({}));
                        if (res.ok && data.available && data.download_url) {
                            accountExportDownloadBtn.href = data.download_url;
                            window.location.assign(data.download_url);
                        } else {
                            renderAccountExportAvailability(data);
                            renderAccountTransferProgress(data);
                            showToast('エクスポートZIPをダウンロードできません。最新の状態を確認してください。', 'warning', true);
                            refreshLatestAccountExport();
                        }
                    } catch (_) {
                        // Network hiccup: fall back to the anchor navigation so
                        // the click still behaves like a normal download link.
                        window.location.assign(href);
                    }
                });
            }
            setAccountTransferControls(false);
            refreshLatestAccountExport();
            // ---- Import file selection modal (storage limit exceeded) ----
            const importFileGrid = get('import-files-grid');
            const importFileInfo = get('import-files-info');
            const importFileSummary = get('import-files-summary');
            const importFormatBytes = (bytes) => {
                const n = Math.max(0, Number(bytes) || 0);
                if (n >= 1024 * 1024 * 1024) return `${(n / (1024 * 1024 * 1024)).toFixed(2)} GB`;
                if (n >= 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(1)} MB`;
                if (n >= 1024) return `${Math.round(n / 1024)} KB`;
                return `${n} B`;
            };
