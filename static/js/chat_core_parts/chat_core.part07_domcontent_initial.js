
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
                    if (fastToggle.checked) requestBrowserFastModeEnable();
                    else setBrowserFastModeEnabled(false);
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
                    if (!model.startsWith('gemini-') || /(image|native-audio|tts|live)/.test(model)) {
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
                const fmt = get('gpt-image-format');
                const compWrap = get('gpt-image-compression-wrap');
                if (fmt && compWrap) {
                    if (fmt.value === 'png') {
                        compWrap.classList.add('hidden');
                    } else {
                        compWrap.classList.remove('hidden');
                    }
                }
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
                if (fullHd) fullHd.disabled = model !== 'grok-imagine-video-1.5';
                if (model !== 'grok-imagine-video-1.5' && resolution.value === '1080p') resolution.value = '720p';
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
            } else if (model === 'deepseek-v4-flash-vision-exp') {
                show = true;
                html = [
                    '<div class="font-bold text-gray-300 mb-1">DeepSeek V4 Flash Vision Exp 入力制限</div>',
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
            function toggleOptions() {
                const modelEl = get('model-select');
                if (!modelEl) return;
                const model = modelEl.value;
                const modelLower = String(model || '').toLowerCase();
                const isDeepSeek = modelLower.includes('deepseek');
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
                const isNanoBanana2 = modelLower.includes('gemini-3.1-flash-image') && !isNanoBanana2Lite;
                const isClaude = isClaudeModelKey(model);
                // DeepSeek KV cache is automatic; it does not use an app-supplied prompt_cache_key.
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
                const vmi = get('vision-model-info');
                if (vmi) vmi.classList.add('hidden');
                if (reasonOpts) {
                    const effortSel = get('reasoning-effort');
                    if (effortSel) {
                        Array.from(effortSel.options).forEach(opt => {
                            const isGpt56Model = modelLower === 'gpt-5.6' || modelLower.startsWith('gpt-5.6-');
                            // DeepSeek V4 Flash family (incl. Vision Exp) shares the low/high/max effort mapping.
                            const isDeepSeekFlash0731 = modelLower === 'deepseek-v4-flash-0731' || modelLower === 'deepseek-v4-flash' || modelLower === 'deepseek-v4-flash-vision-exp';
                            const isDeepSeekPro = modelLower === 'deepseek-v4-pro';
                            const isGrok45 = modelLower.includes('grok-4.5');
                            const isGrok46 = modelLower.includes('grok-4.6');
                            if (opt.value === 'max') {
                                opt.classList.toggle('hidden', !isGpt56Model && !isDeepSeekFlash0731 && !isDeepSeekPro);
                            } else if (opt.value === 'xhigh') {
                                opt.classList.toggle('hidden', !isGrok46 && !modelLower.includes('multi-agent') && !isGpt56Model);
                            } else if (opt.value === 'medium') {
                                opt.classList.toggle('hidden', !(modelLower.includes('grok-4.3') || isGrok45 || isGrok46 || modelLower.includes('grok-4.20-0309-reasoning') || modelLower.includes('grok-build') || modelLower.includes('multi-agent') || modelLower.includes('gpt-5') || modelLower.includes('o1') || modelLower.includes('o3')));
                            } else if (opt.value === 'none') {
                                opt.classList.toggle('hidden', !modelLower.includes('grok-4.3') && !modelLower.includes('grok-build') && !modelLower.includes('gpt-5') && !isDeepSeekFlash0731 && !isDeepSeekPro);
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
                    Array.from(thinkLvl.options).forEach(opt => {
                        if (['low', 'medium'].includes(opt.value)) opt.disabled = true;
                        if (['minimal', 'high'].includes(opt.value)) opt.disabled = false;
                    });
                    if (!['minimal', 'high'].includes(thinkLvl.value)) {
                        thinkLvl.value = isNanoBanana2Lite ? 'minimal' : 'high';
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
                } else if(model.includes('gemini') && !isGeminiImage) {
                    thinkOpts.classList.remove('hidden');
                    if(urlCont) {
                        urlCont.classList.remove('hidden', 'opacity-50', 'pointer-events-none');
                    }
                    const isGemini3 = model.includes('gemini-3');
                    if (mapsCont) {
                        if (isGemini3) {
                            mapsCont.classList.remove('hidden', 'opacity-50', 'pointer-events-none');
                        } else {
                            if (mapsChk) mapsChk.checked = false;
                            mapsCont.classList.add('hidden', 'opacity-50', 'pointer-events-none');
                        }
                    }
                    const isFlash = model.includes('flash');
                    Array.from(thinkLvl.options).forEach(opt => {
                        if (model === 'gemini-3.7-flash') {
                            opt.disabled = !['low', 'medium', 'high'].includes(opt.value);
                        } else if (model === 'gemini-3.6-flash') {
                            opt.disabled = !['medium', 'high'].includes(opt.value);
                        } else if (model === 'gemini-3.5-flash-lite') {
                            opt.disabled = !['minimal', 'medium', 'high'].includes(opt.value);
                        } else if(['minimal', 'medium'].includes(opt.value)) {
                            opt.disabled = !isFlash;
                        } else {
                            opt.disabled = false;
                        }
                    });
                    if (model === 'gemini-3.7-flash' && !['low', 'medium', 'high'].includes(thinkLvl.value)) {
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
                    reasonOpts.classList.remove('hidden');
                    if(searchCont) searchCont.classList.remove('opacity-50', 'pointer-events-none');
                } else if (isDeepSeek) {
                    reasonOpts.classList.remove('hidden');
                    // Vision Exp handles images natively; only text-only DeepSeek models need the vision-model notice.
                    const vmi = get('vision-model-info');
                    if (vmi) vmi.classList.toggle('hidden', modelLower === 'deepseek-v4-flash-vision-exp');
                    if (searchChk) {
                        searchChk.checked = false;
                        searchChk.disabled = true;
                    }
                    if (searchCont) searchCont.classList.add('opacity-50', 'pointer-events-none');
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

                if(((isGeminiImage && !isNanoBanana2) || model.includes('gpt-image') || isGrokImageModel() || isGrokVideoModel() || isOcr)) { if (sysChk && sysLbl) { sysChk.checked = false; sysChk.disabled = true; sysLbl.classList.add('opacity-50'); } }
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
                } else if (searchChk && !model.includes('tts') && !isOcr && !isDeepSeek && !isNanoBanana2Lite) {
                    searchChk.disabled = false;
                }
                const maskBtn = get('mask-btn');
                if (maskBtn) {
                    if (isGptImageModel()) {
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
                updateGrokVideoUi();
                updateGeminiVideoUi();
                updateGeminiMusicUi();
                updateXaiChatUi();
                updateMistralOcrUi();
                updateImageInputLimits();
                purgeUnsupportedAttachments(true);
                refreshMinimalOptionsIfOpen();
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
                        return `<div class="code-wrapper collapsed" data-collapsed="true" data-code-key="${codeKey}"><div class="code-header"><span class="code-lang">${langLabel}</span><div class="code-actions"><button class="code-toggle" aria-expanded="false" title="展開" aria-label="展開"><i class="fas fa-chevron-down"></i></button>${codingBtn}${previewBtn}${downloadBtn}<button class="copy-btn" data-code="${enc}" title="コピー" aria-label="コピー"><i class="fas fa-copy"></i></button></div></div><div class="code-body"><pre><code class="hljs language-${l}">${h}</code></pre></div></div>`;
                    },
                    link(h, t, x) { return `<a href="${h}" title="${t || ''}" target="_blank">${x}</a>`; },
                    image(h, t, x) { const alt = escapeHtml(x || ''); const title = t ? ` title="${escapeHtml(t)}"` : ''; if (String(h || '').startsWith('sandbox:')) { return `<span class="text-xs text-gray-500" title="${escapeHtml(h)}">${alt || '（画像データは取得できませんでした）'}</span>`; } const viewerSrc = escapeHtml(h || ''); return `<img src="${h}" data-viewer-src="${viewerSrc}" alt="${alt}"${title} class="chat-image" loading="lazy" width="320" height="320">`; }
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
            apiFetch(CHAT_CONFIG.urls.handleSettingsQuery).then(r => r.json()).then(d => {
                cacheUserSettings(d);
                if (d) {
                    currentVisionModel = d.default_vision_model || 'gemini-3-flash-preview';
                }
                applyChatDefaults(d);
                if (d && d.theme_color) {
                    applyThemeColor(d.theme_color, true);
                }
                if (d && Object.prototype.hasOwnProperty.call(d, 'minimal_prompt_mode') && d.minimal_prompt_mode) {
                    setMinimalPromptMode(true);
                } else if (d && Object.prototype.hasOwnProperty.call(d, 'compact_prompt_mode')) {
                    setCompactPromptMode(!!d.compact_prompt_mode);
                }
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
            loadThreads(); loadGems();

            get('send-btn').onclick = () => { if (isStopMode) stopGeneration(); else sendMessage(); };
            get('new-chat-btn').onclick = () => startNewChat();
            bindUploadButton();
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
                        importing_credentials: '認証情報を反映中', importing_gems: 'Gemを追加中',
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
            let importFileSelection = null;
            const updateImportFileSelectionUi = () => {
                if (!importFileSelection) return;
                const files = importFileSelection.files;
                const selection = importFileSelection.selection;
                let total = 0;
                files.forEach((f) => { if (selection.has(f.archive_path)) total += (Number(f.size_bytes) || 0); });
                const available = Number(importFileSelection.available_bytes) || 0;
                const over = total > available;
                if (importFileSummary) {
                    importFileSummary.textContent = `選択中: ${importFormatBytes(total)} / 利用可能: ${importFormatBytes(available)}${over ? ' （容量超過）' : ''}`;
                    importFileSummary.classList.toggle('text-red-300', over);
                }
                if (importFileInfo) importFileInfo.textContent = `${files.length} files`;
            };
            const renderImportFileItems = () => {
                if (!importFileGrid || !importFileSelection) return;
                importFileGrid.innerHTML = '';
                const files = importFileSelection.files;
                if (!files.length) {
                    importFileGrid.innerHTML = '<div class="text-xs text-gray-500">インポート可能なファイルがありません。</div>';
                    updateImportFileSelectionUi();
                    return;
                }
                files.forEach((f) => {
                    const label = document.createElement('label');
                    const checked = importFileSelection.selection.has(f.archive_path);
                    label.className = `relative bg-gray-800 border rounded flex items-center gap-2 p-2 cursor-pointer transition hover:border-blue-500 ${checked ? 'border-blue-500' : 'border-gray-600'}`;
                    label.innerHTML =
                        `<input type="checkbox" class="import-file-check accent-blue-500 w-4 h-4 shrink-0"${checked ? ' checked' : ''}>` +
                        `<div class="min-w-0 flex-1">` +
                            `<div class="text-xs text-gray-200 truncate" title="${escapeHtml(f.display_name)}">${escapeHtml(f.display_name)}</div>` +
                            `<div class="text-[10px] text-gray-500">${importFormatBytes(f.size_bytes)}</div>` +
                        `</div>`;
                    const checkbox = label.querySelector('.import-file-check');
                    checkbox.addEventListener('change', () => {
                        if (checkbox.checked) importFileSelection.selection.add(f.archive_path);
                        else importFileSelection.selection.delete(f.archive_path);
                        label.classList.toggle('border-blue-500', checkbox.checked);
                        label.classList.toggle('border-gray-600', !checkbox.checked);
                        updateImportFileSelectionUi();
                    });
                    importFileGrid.appendChild(label);
                });
                updateImportFileSelectionUi();
            };
            const showImportFileSelection = (payload) => new Promise((resolve) => {
                importFileSelection = {
                    files: payload.files || [],
                    selection: new Set((payload.files || []).map(f => f.archive_path)),
                    available_bytes: payload.available_bytes,
                    resolve,
                };
                renderImportFileItems();
                if (!get('import-files-modal')) {
                    resolve(null);
                    return;
                }
                showModal('import-files-modal');
            });
            const closeImportFileSelection = (result) => {
                hideModal('import-files-modal');
                if (importFileSelection) {
                    const resolver = importFileSelection.resolve;
                    importFileSelection = null;
                    resolver(result);
                }
            };
            const importFileCloseBtn = get('import-files-close');
            if (importFileCloseBtn) importFileCloseBtn.onclick = () => closeImportFileSelection(null);
            const importFileCancelBtn = get('import-files-cancel');
            if (importFileCancelBtn) importFileCancelBtn.onclick = () => closeImportFileSelection(null);
            const importFileConfirmBtn = get('import-files-confirm');
            if (importFileConfirmBtn) {
                importFileConfirmBtn.onclick = () => {
                    if (!importFileSelection) return;
                    const selected = Array.from(importFileSelection.selection);
                    closeImportFileSelection(selected.length ? selected.join(',') : '__none__');
                };
            }
            const importFileSelectAllBtn = get('import-files-select-all');
            if (importFileSelectAllBtn) {
                importFileSelectAllBtn.onclick = () => {
                    if (!importFileSelection) return;
                    importFileSelection.files.forEach((f) => importFileSelection.selection.add(f.archive_path));
                    renderImportFileItems();
                };
            }
            const importFileNoneBtn = get('import-files-none');
            if (importFileNoneBtn) {
                importFileNoneBtn.onclick = () => {
                    if (!importFileSelection) return;
                    importFileSelection.selection.clear();
                    renderImportFileItems();
                };
            }

            // --- Settings import confirmation --------------------------------
            const ACCOUNT_SETTING_LABELS = {
                system_prompt: 'システムプロンプト', system_prompt_enabled: 'システムプロンプトを使用',
                apply_global_system_prompt: '全体システムプロンプトを適用', apply_auto_system_prompt_notices: '自動注入システムプロンプトを適用',
                auto_system_prompt_notices_config: '自動注入システムプロンプトの種類別設定',
                gemini_backend: 'Gemini バックエンド', gemini_vertex_location: 'Vertex AI ロケーション',
                mic_transcribe_mode: 'マイク文字起こし方式', stt_model: 'STTモデル', llm_transcribe_prompt: 'LLM文字起こしプロンプト',
                enter_to_send: 'Enterキーで送信', use_sw_cache: 'Service Workerキャッシュ',
                clear_cache_on_version_update: 'バージョン更新時キャッシュ削除', theme_color: 'テーマカラー',
                liquid_glass_enabled: 'Liquid Glass', auto_search_on_links: 'リンクで自動検索',
                compact_prompt_mode: 'プロンプトバー表示（コンパクト）', minimal_prompt_mode: 'プロンプトバー表示（ミニマル）',
                use_last_chat_settings: '直前のチャット設定を使用', voice_studio_ui: '音声スタジオUI',
                temp_chat_timeout_seconds: '一時チャットの有効時間（秒）', default_model: '既定のモデル',
                default_enable_search: '既定: Search', default_enable_url_context: '既定: URLコンテキスト',
                default_enable_maps: '既定: Maps', default_enable_python: '既定: Python',
                default_enable_file_creation: '既定: File',
                default_enable_thinking: '既定: Thinking', default_thinking_level: '既定: Thinkingレベル',
                default_thinking_budget: '既定: Thinking budget', default_reasoning_effort: '既定: Reasoning effort',
                default_enable_system_prompt: '既定: システムプロンプト', default_safety_setting: '既定: 安全設定',
                default_vision_model: 'Vision Model', rich_paste_prompt_default: 'リッチ貼り付けプロンプト',
                rich_paste_prompt_use_custom_default: 'リッチ貼り付けカスタムプロンプト既定',
                last_model: '直前のモデル', last_enable_search: '直前: Search', last_enable_url_context: '直前: URLコンテキスト',
                last_enable_maps: '直前: Maps', last_enable_python: '直前: Python', last_enable_file_creation: '直前: File', last_enable_thinking: '直前: Thinking',
                last_thinking_level: '直前: Thinkingレベル', last_thinking_budget: '直前: Thinking budget',
                last_reasoning_effort: '直前: Reasoning effort', last_enable_system_prompt: '直前: システムプロンプト',
                last_safety_setting: '直前: 安全設定', enable_latency_metrics: 'レスポンス速度の計測',
                enable_client_debug_log: 'デバッグログの拡張送信',
            };
            const formatAccountSettingValue = (value) => {
                if (value === true) return 'ON';
                if (value === false) return 'OFF';
                if (value === null || value === undefined || value === '') return '未設定';
                const text = String(value);
                return text.length > 60 ? text.slice(0, 60) + '…' : text;
            };
            let settingsImportConfirmationResolver = null;
            const resolveSettingsImportConfirmation = (result) => {
                if (settingsImportConfirmationResolver) {
                    const resolver = settingsImportConfirmationResolver;
                    settingsImportConfirmationResolver = null;
                    hideModal('settings-confirmation-modal');
                    resolver(result);
                }
            };
            const showSettingsImportConfirmation = (payload) => new Promise((resolve) => {
                const modal = get('settings-confirmation-modal');
                if (!modal) {
                    // No modal markup (unexpected).  Proceed so the import is not blocked.
                    resolve(true);
                    return;
                }
                settingsImportConfirmationResolver = resolve;
                const changes = Array.isArray(payload && payload.settings_changes) ? payload.settings_changes : [];
                const listEl = get('settings-confirmation-list');
                if (listEl) {
                    if (!changes.length) {
                        listEl.innerHTML = '<div class="text-xs text-gray-400">変更される設定はありませんでした。</div>';
                    } else {
                        listEl.innerHTML = changes.map((c) => {
                            const label = ACCOUNT_SETTING_LABELS[c.field] || c.field;
                            const current = formatAccountSettingValue(c.current);
                            const incoming = formatAccountSettingValue(c.incoming);
                            return `<div class="rounded border border-gray-700 bg-gray-800/60 p-2">
                                <div class="text-xs font-bold text-gray-100">${escapeHtml(label)}</div>
                                <div class="text-[11px] text-gray-400 mt-1">現在: ${escapeHtml(current)}</div>
                                <div class="text-[11px] text-emerald-300">→ ${escapeHtml(incoming)}</div>
                            </div>`;
                        }).join('');
                    }
                }
                const countEl = get('settings-confirmation-count');
                if (countEl) countEl.textContent = `${changes.length}件の設定が変更されます`;
                showModal('settings-confirmation-modal');
            });
            const settingsConfirmationModal = get('settings-confirmation-modal');
            if (settingsConfirmationModal) {
                settingsConfirmationModal.addEventListener('click', (e) => {
                    if (e.target === settingsConfirmationModal) resolveSettingsImportConfirmation(false);
                });
            }
            const settingsConfirmationCloseBtn = get('settings-confirmation-close');
            if (settingsConfirmationCloseBtn) settingsConfirmationCloseBtn.onclick = () => resolveSettingsImportConfirmation(false);
            const settingsConfirmationCancelBtn = get('settings-confirmation-cancel');
            if (settingsConfirmationCancelBtn) settingsConfirmationCancelBtn.onclick = () => resolveSettingsImportConfirmation(false);
            const settingsConfirmationConfirmBtn = get('settings-confirmation-confirm');
            if (settingsConfirmationConfirmBtn) settingsConfirmationConfirmBtn.onclick = () => resolveSettingsImportConfirmation(true);

            const accountImportBtn = get('account-import-btn');
            const inplaceToggle = get('account-import-inplace');
            const inplaceWarn = get('account-import-inplace-warning');
            if (inplaceToggle && inplaceWarn) {
                const syncInplaceWarn = () => inplaceWarn.classList.toggle('hidden', !inplaceToggle.checked);
                inplaceToggle.addEventListener('change', syncInplaceWarn);
                syncInplaceWarn();
            }
            if (accountImportBtn) {
                accountImportBtn.onclick = async () => {
                    const input = get('account-import-file');
                    const file = input && input.files ? input.files[0] : null;
                    const categoryBox = get('account-import-categories');
                    const categories = categoryBox
                        ? Array.from(categoryBox.querySelectorAll('input[type="checkbox"]:checked')).map(el => el.value)
                        : [];
                    const inplaceBox = get('account-import-inplace');
                    const restoreInplace = !!(inplaceBox && inplaceBox.checked);
                    const settingsBypassBox = get('account-import-settings-bypass');
                    const settingsBypass = !!(settingsBypassBox && settingsBypassBox.checked);
                    let settingsConfirmed = false;
                    if (!file) {
                        showToast('インポートするZIPファイルを選択してください', 'error', true);
                        return;
                    }
                    if (!categories.length) {
                        showToast('インポートするデータを1つ以上選択してください', 'error', true);
                        return;
                    }
                    const selectedLabels = categoryBox
                        ? Array.from(categoryBox.querySelectorAll('input[type="checkbox"]:checked')).map(el => (el.closest('label') && el.closest('label').textContent || el.value).trim())
                        : categories;
                    if (!confirm(`次のデータをインポートします。既存データは削除されません。すでに同じ内容のデータがある場合はスキップされます。\n\n${selectedLabels.join('、')}${restoreInplace ? '\n※「元の場所へ復元」: このアカウントの同名ファイルを上書きします' : ''}\n\n続行しますか？`)) return;
                    const transfer = {
                        id: createAccountTransferId(), type: 'import', stopped: false,
                        controller: new AbortController()
                    };
                    activeAccountTransfer = transfer;
                    setAccountTransferControls(true);
                    renderAccountTransferProgress({progress: 0, phase: 'uploading', message: 'アップロードを準備しています'});
                    const resultBox = get('account-import-result');
                    let pollPromise = Promise.resolve(null);
                    try {
                        const chunkSize = 10 * 1024 * 1024;
                        const totalChunks = Math.max(1, Math.ceil(file.size / chunkSize));
                        const startRes = await apiFetch('/api/account/import/upload/start', manualSpinnerRequestOptions({
                            method: 'POST', headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({size: file.size}), signal: transfer.controller.signal,
                        }));
                        const startData = await startRes.json().catch(() => ({}));
                        if (!startRes.ok) throw new Error(startData.error || 'アップロードを開始できません');
                        transfer.uploadId = startData.upload_id;
                        const actualChunkSize = startData.chunk_size || chunkSize;
                        let uploadedChunks = 0;
                        let nextChunk = 0;
                        const uploadWorker = async () => {
                            while (true) {
                                const index = nextChunk++;
                                if (index >= totalChunks) return;
                                const chunk = file.slice(index * actualChunkSize, Math.min(file.size, (index + 1) * actualChunkSize));
                                const chunkForm = new FormData();
                                chunkForm.append('chunk', chunk, file.name);
                                chunkForm.append('index', String(index));
                                const chunkRes = await apiFetch(`/api/account/import/upload/${encodeURIComponent(transfer.uploadId)}/chunk`, manualSpinnerRequestOptions({
                                    method: 'POST', body: chunkForm, signal: transfer.controller.signal,
                                }));
                                const chunkData = await chunkRes.json().catch(() => ({}));
                                if (!chunkRes.ok) throw new Error(chunkData.error || 'アップロードに失敗しました');
                                uploadedChunks++;
                                renderAccountTransferProgress({progress: Math.min(35, Math.round((uploadedChunks / totalChunks) * 35)), phase: 'uploading', message: `ZIPを並列アップロードしています（${uploadedChunks}/${totalChunks}）`});
                                if (window.ConnectionMonitor) window.ConnectionMonitor.reportActivity();
                            }
                        };
                        let importUploadOpStarted = false;
                        if (window.ConnectionMonitor) {
                            window.ConnectionMonitor.operationStarted();
                            importUploadOpStarted = true;
                        }
                        try {
                            await Promise.all([uploadWorker(), uploadWorker(), uploadWorker()]);
                            const completeRes = await apiFetch(`/api/account/import/upload/${encodeURIComponent(transfer.uploadId)}/complete`, manualSpinnerRequestOptions({method: 'POST', signal: transfer.controller.signal}));
                            const completeData = await completeRes.json().catch(() => ({}));
                            if (!completeRes.ok) throw new Error(completeData.error || 'アップロードを完了できません');
                            renderAccountTransferProgress({progress: 35, phase: 'validating', message: 'ZIPを検証しています'});
                        } finally {
                            if (importUploadOpStarted && window.ConnectionMonitor) window.ConnectionMonitor.operationEnded();
                        }
                        let selectedFiles = '';
                        let importDone = false;
                        let parseFailures = 0;
                        // After a settings / API-credentials import, re-fetch the saved
                        // settings and repopulate the (possibly still-open) settings modal
                        // so imported values are reflected instead of stale pre-import
                        // form values that a later save would overwrite.  The import has
                        // already persisted the values server-side, so the page is then
                        // reloaded: that rebuilds every client-side setting (the modal
                        // form, cached settings, theme, prompt-bar mode, model list, API
                        // keys) from the freshly imported state and is guaranteed to
                        // reflect the import.
                        const refreshSettingsFormAfterImport = async () => {
                            let reloadScheduled = false;
                            const scheduleReload = () => {
                                if (reloadScheduled) return;
                                reloadScheduled = true;
                                setTimeout(() => { location.reload(); }, 1100);
                            };
                            try {
                                const res = await apiFetch(CHAT_CONFIG.urls.handleSettingsQuery, {cache: 'no-store'});
                                const data = await res.json().catch(() => null);
                                if (!res.ok || !data) {
                                    scheduleReload();
                                    return;
                                }
                                cacheUserSettings(data);
                                const settingsModalEl = get('settings-modal');
                                if (settingsModalEl && settingsModalEl.classList.contains('modal-open')) {
                                    try { populateSettingsFormFromData(data); } catch (_) {}
                                }
                                // Apply UI-affecting settings immediately (theme / prompt-bar mode).
                                if (data.theme_color) applyThemeColor(data.theme_color, true);
                                if (Object.prototype.hasOwnProperty.call(data, 'minimal_prompt_mode') && data.minimal_prompt_mode) {
                                    setMinimalPromptMode(true);
                                } else if (Object.prototype.hasOwnProperty.call(data, 'compact_prompt_mode')) {
                                    setCompactPromptMode(!!data.compact_prompt_mode);
                                }
                            } catch (_) {}
                            scheduleReload();
                        };
                        const finishImportSuccess = (terminal) => {
                            const message = (terminal && terminal.message) || 'インポートが完了しました';
                            if (resultBox) {
                                resultBox.textContent = `完了: ${message}`;
                                resultBox.classList.remove('hidden', 'text-red-300');
                                resultBox.classList.add('text-emerald-300');
                            }
                            renderAccountTransferProgress({progress: 100, phase: 'completed', message});
                            showToast('選択したアカウントデータをインポートしました', 'success');
                            if (categories.includes('chats')) loadThreads();
                            if (categories.includes('gems')) loadGems();
                            if (categories.includes('files')) loadStorageUsage();
                            if (categories.includes('settings') || categories.includes('api_credentials')) refreshSettingsFormAfterImport();
                        };
                        const fetchImportStatus = async () => {
                            try {
                                const res = await apiFetch(`/api/account/transfer/${transfer.id}`, manualSpinnerRequestOptions({cache: 'no-store'}));
                                const data = await res.json().catch(() => null);
                                return data && data.state ? data : null;
                            } catch (_) {
                                return null;
                            }
                        };
                        const settleUnreadableImport = async () => {
                            // The import response could not be read. The authoritative
                            // transfer status tells us what actually happened server-side:
                            //   completed -> import really finished (response was lost)
                            //   failed/cancelled/expired -> real server outcome
                            //   needs_selection -> the picker data survived in the status
                            //   running -> server still importing; wait for the terminal state
                            const outcome = await fetchImportStatus();
                            if (!outcome) return {status: 'unknown'};
                            if (outcome.state === 'completed') {
                                finishImportSuccess(outcome);
                                return {status: 'done'};
                            }
                            if (['failed', 'cancelled', 'expired'].includes(outcome.state)) {
                                throw new Error(outcome.message || 'インポートに失敗しました');
                            }
                            if (outcome.state === 'needs_selection' && Array.isArray(outcome.files)) {
                                const chosen = await showImportFileSelection({
                                    files: outcome.files,
                                    available_bytes: outcome.available_bytes,
                                });
                                if (chosen === null) {
                                    renderAccountTransferProgress({progress: 0, phase: 'cancelled', message: 'ファイル選択をキャンセルしました'});
                                    if (transfer.uploadId) {
                                        apiFetch(`/api/account/import/upload/${encodeURIComponent(transfer.uploadId)}`, manualSpinnerRequestOptions({method: 'DELETE'})).catch(() => null);
                                    }
                                    return {status: 'cancelled'};
                                }
                                selectedFiles = chosen;
                                return {status: 'reselect'};
                            }
                            if (outcome.state === 'needs_settings_confirmation' && Array.isArray(outcome.settings_changes)) {
                                const ok = await showSettingsImportConfirmation({ settings_changes: outcome.settings_changes });
                                if (!ok) {
                                    renderAccountTransferProgress({progress: 0, phase: 'cancelled', message: '設定のインポートをキャンセルしました'});
                                    if (transfer.uploadId) {
                                        apiFetch(`/api/account/import/upload/${encodeURIComponent(transfer.uploadId)}`, manualSpinnerRequestOptions({method: 'DELETE'})).catch(() => null);
                                    }
                                    return {status: 'cancelled'};
                                }
                                settingsConfirmed = true;
                                return {status: 'reselect'};
                            }
                            if (outcome.state === 'running') {
                                // The connection dropped but the server is still importing.
                                // The ongoing poller resolves once a terminal state is reached.
                                const terminal = await Promise.race([
                                    pollPromise.catch(() => null),
                                    new Promise(resolve => setTimeout(() => resolve(null), 60000)),
                                ]);
                                if (terminal && terminal.state === 'completed') {
                                    finishImportSuccess(terminal);
                                    return {status: 'done'};
                                }
                                if (terminal && ['failed', 'cancelled', 'expired'].includes(terminal.state)) {
                                    throw new Error(terminal.message || 'インポートに失敗しました');
                                }
                                throw new Error('インポート処理がサーバー側で継続中です。しばらくしてからページを再読み込みして確認してください');
                            }
                            // pending / unknown: the request never reached the server.
                            return {status: 'unknown'};
                        };
                        while (!importDone) {
                            transfer.stopped = true;
                            await pollPromise.catch(() => null);
                            transfer.stopped = false;
                            pollPromise = pollAccountTransfer(transfer);
                            let res;
                            try {
                                res = await apiFetch('/api/account/import', manualSpinnerRequestOptions({
                                    method: 'POST', headers: {'Content-Type': 'application/json'},
                                    body: JSON.stringify({upload_id: transfer.uploadId, categories: categories.join(','), job_id: transfer.id, selected_files: selectedFiles, restore_inplace: restoreInplace, confirm_settings: (settingsConfirmed || settingsBypass)}),
                                    signal: transfer.controller.signal,
                                }));
                            } catch (error) {
                                if (transfer.cancelRequested || (error && error.name === 'AbortError')) throw error;
                                const settled = await settleUnreadableImport();
                                if (settled.status === 'done') { importDone = true; break; }
                                if (settled.status === 'cancelled') return;
                                if (settled.status === 'reselect') continue;
                                if (parseFailures < 2) { parseFailures++; continue; }
                                throw new Error('インポート応答を取得できませんでした。通信環境をご確認のうえ、もう一度お試しください');
                            }
                            let data = null;
                            try {
                                data = await res.json();
                            } catch (_) {
                                data = null;
                            }
                            if (data === null) {
                                const settled = await settleUnreadableImport();
                                if (settled.status === 'done') { importDone = true; break; }
                                if (settled.status === 'cancelled') return;
                                if (settled.status === 'reselect') continue;
                                if (res.ok) throw new Error('インポート結果を確認できませんでした。ページを再読み込みして確認してください');
                                if (parseFailures < 2) { parseFailures++; continue; }
                                throw new Error('インポート応答を取得できませんでした。通信環境をご確認のうえ、もう一度お試しください');
                            }
                            if (!res.ok && data.error === 'storage_limit_files' && data.files) {
                                const chosen = await showImportFileSelection(data);
                                if (chosen === null) {
                                    renderAccountTransferProgress({progress: 0, phase: 'cancelled', message: 'ファイル選択をキャンセルしました'});
                                    if (transfer.uploadId) {
                                        apiFetch(`/api/account/import/upload/${encodeURIComponent(transfer.uploadId)}`, manualSpinnerRequestOptions({method: 'DELETE'})).catch(() => null);
                                    }
                                    return;
                                }
                                selectedFiles = chosen;
                                continue;
                            }
                            if (data && data.status === 'settings_confirmation' && Array.isArray(data.settings_changes)) {
                                const ok = await showSettingsImportConfirmation(data);
                                if (!ok) {
                                    renderAccountTransferProgress({progress: 0, phase: 'cancelled', message: '設定のインポートをキャンセルしました'});
                                    if (transfer.uploadId) {
                                        apiFetch(`/api/account/import/upload/${encodeURIComponent(transfer.uploadId)}`, manualSpinnerRequestOptions({method: 'DELETE'})).catch(() => null);
                                    }
                                    return;
                                }
                                settingsConfirmed = true;
                                continue;
                            }
                            if (!res.ok) throw new Error(data.error || 'インポートに失敗しました');
                            const imported = data.imported || {};
                            const detail = [
                                `設定 ${imported.settings || 0}件`, `API認証 ${imported.api_credentials || 0}件`,
                                `チャット ${imported.chats || 0}件`, `Gem ${imported.gems || 0}件`,
                                `ファイル ${imported.files || 0}件`, `フィードバック ${imported.feedback || 0}件`,
                                `診断データ ${imported.diagnostics || 0}件`,
                            ].join(' / ');
                            const duplicated = data.duplicates || {};
                            const dupLabels = {chats: 'チャット', gems: 'Gem', files: 'ファイル', feedback: 'フィードバック', diagnostics: '診断データ'};
                            const dupParts = [];
                            for (const key of Object.keys(dupLabels)) {
                                const count = Number(duplicated[key]) || 0;
                                if (count > 0) dupParts.push(`${dupLabels[key]} ${count}件`);
                            }
                            const dupNote = dupParts.length ? `（重複をスキップ: ${dupParts.join('、')}）` : '';
                            if (resultBox) {
                                resultBox.textContent = `完了: ${detail}${dupNote}`;
                                resultBox.classList.remove('hidden', 'text-red-300');
                                resultBox.classList.add('text-emerald-300');
                            }
                            renderAccountTransferProgress({progress: 100, phase: 'completed', message: 'インポートが完了しました'});
                            showToast('選択したアカウントデータをインポートしました', 'success');
                            if (categories.includes('chats')) loadThreads();
                            if (categories.includes('gems')) loadGems();
                            if (categories.includes('files')) loadStorageUsage();
                            if (categories.includes('settings') || categories.includes('api_credentials')) refreshSettingsFormAfterImport();
                            importDone = true;
                        }
                    } catch (error) {
                        if (transfer.uploadId) {
                            apiFetch(`/api/account/import/upload/${encodeURIComponent(transfer.uploadId)}`, manualSpinnerRequestOptions({method: 'DELETE'})).catch(() => null);
                        }
                        if (transfer.cancelRequested || (error && error.name === 'AbortError')) return;
                        const rawMessage = error && error.message ? error.message : '';
                        const friendlyMessage = rawMessage === 'storage_limit_exceeded'
                            ? 'ストレージ上限を超えるためインポートできません'
                            : (rawMessage || 'インポートに失敗しました');
                        renderAccountTransferProgress({progress: 0, phase: 'failed', message: friendlyMessage});
                        if (resultBox) {
                            resultBox.textContent = friendlyMessage;
                            resultBox.classList.remove('hidden', 'text-emerald-300');
                            resultBox.classList.add('text-red-300');
                        }
                        showToast(friendlyMessage, 'error', true);
                    } finally {
                        transfer.stopped = true;
                        await pollPromise.catch(() => null);
                        finishAccountTransfer(transfer);
                    }
                };
            }
