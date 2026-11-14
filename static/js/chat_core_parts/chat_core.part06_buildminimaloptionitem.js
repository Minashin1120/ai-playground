        function buildMinimalOptionItem(item) {
            const row = document.createElement('div');
            row.className = 'minimal-option-item';
            row.dataset.key = item.key;
            if (item.action) row.classList.add('action-' + item.action);
            if (minimalOptionChecked(item)) row.classList.add('on');
            else row.classList.add('off');
            if (minimalOptionDisabled(item)) row.classList.add('disabled');
            const icon = document.createElement('i');
            icon.className = 'fas ' + item.icon + ' minimal-option-icon';
            row.appendChild(icon);
            const label = document.createElement('span');
            label.className = 'minimal-option-label';
            label.textContent = item.label;
            row.appendChild(label);
            if (item.special === 'thinking') {
                const level = document.createElement('span');
                level.className = 'thinking-slide-value minimal-option-thinking-level';
                level.textContent = currentThinkingLevelLabel();
                row.appendChild(level);
            }
            if (item.selectId) {
                const src = get(item.selectId);
                if (src) {
                    const clone = src.cloneNode(true);
                    clone.removeAttribute('id');
                    clone.className = 'minimal-option-select';
                    clone.addEventListener('change', () => {
                        src.value = clone.value;
                        src.dispatchEvent(new Event('change', { bubbles: true }));
                        refreshMinimalOptionItems();
                    });
                    row.appendChild(clone);
                }
            }
            if (item.gear) {
                const gear = document.createElement('button');
                gear.type = 'button';
                gear.className = 'minimal-option-gear';
                gear.title = item.label + '設定';
                const gicon = document.createElement('i');
                gicon.className = 'fas fa-cog';
                gear.appendChild(gicon);
                gear.addEventListener('click', (e) => {
                    e.stopPropagation();
                    closeMinimalOptions();
                    if (typeof item.gearAction === 'function') item.gearAction();
                });
                row.appendChild(gear);
            }
            row.addEventListener('click', () => handleMinimalOptionClick(item));
            return row;
        }

        function renderMinimalOptionItems() {
            const wrap = get('minimal-options-items');
            if (!wrap) return;
            const frag = document.createDocumentFragment();
            MINIMAL_POPUP_ITEMS.forEach((item) => {
                if (!minimalOptionVisible(item)) return;
                frag.appendChild(buildMinimalOptionItem(item));
            });
            wrap.innerHTML = '';
            wrap.appendChild(frag);
        }

        function refreshMinimalOptionItems() {
            const wrap = get('minimal-options-items');
            if (!wrap || !minimalOptionsOpen) return;
            const rows = wrap.querySelectorAll('.minimal-option-item');
            const rowByKey = {};
            rows.forEach((row) => { rowByKey[row.dataset.key] = row; });
            MINIMAL_POPUP_ITEMS.forEach((item) => {
                const row = rowByKey[item.key];
                if (!row) return;
                if (!minimalOptionVisible(item)) { row.classList.add('hidden'); return; }
                row.classList.remove('hidden');
                row.classList.toggle('on', minimalOptionChecked(item));
                row.classList.toggle('off', !minimalOptionChecked(item));
                row.classList.toggle('disabled', minimalOptionDisabled(item));
                if (item.special === 'thinking') {
                    const level = row.querySelector('.minimal-option-thinking-level');
                    if (level) level.textContent = currentThinkingLevelLabel();
                }
                if (item.selectId) {
                    const src = get(item.selectId);
                    const clone = row.querySelector('.minimal-option-select');
                    if (src && clone && document.activeElement !== clone && clone.value !== src.value) {
                        clone.value = src.value;
                    }
                }
            });
        }

        function handleMinimalOptionClick(item) {
            if (item.action === 'upload') {
                closeMinimalOptions();
                openUploadModal();
                return;
            }
            if (item.action === 'button') {
                const button = get(item.buttonId);
                closeMinimalOptions();
                if (button) button.click();
                return;
            }
            if (item.special === 'thinking') {
                if (thinkingUsesLevelSlider()) {
                    closeMinimalOptions();
                    showThinkingSlider();
                    return;
                }
                // Thinking needs special handling: for models where thinking is
                // forced on (e.g. Gemini 3.x) the checkbox is disabled, but the
                // row must still open the level slider.
                const chk = get(item.checkboxId);
                if (chk && !chk.disabled) {
                    const turningOn = !chk.checked;
                    chk.checked = turningOn;
                    chk.dispatchEvent(new Event('change', { bubbles: true }));
                    if (turningOn && thinkingOptionsList().length) {
                        closeMinimalOptions();
                        showThinkingSlider();
                    } else {
                        hideThinkingSlider();
                    }
                    refreshMinimalOptionItems();
                } else {
                    // Thinking is forced on (or the checkbox is unavailable):
                    // open the slider so the level can still be adjusted.
                    closeMinimalOptions();
                    showThinkingSlider();
                }
                return;
            }
            if (minimalOptionDisabled(item)) return;
            if (item.selectId) return; // select rows change via their own <select>
            const chk = get(item.checkboxId);
            if (!chk) return;
            if (chk.disabled) return;
            chk.checked = !chk.checked;
            chk.dispatchEvent(new Event('change', { bubbles: true }));
            refreshMinimalOptionItems();
            // Fast mode may open a modal and settles asynchronously; temporary
            // chat confirmation is async too, so refresh the rows afterwards.
            if (item.key === 'fast') {
                closeMinimalOptions();
                setTimeout(() => refreshMinimalOptionItems(), 350);
            } else if (item.key === 'tempchat') {
                setTimeout(() => refreshMinimalOptionItems(), 350);
            }
        }

        function moveModelPanelsIntoPopup() {
            const body = get('minimal-options-model-body');
            if (!body) return;
            let anyVisible = false;
            MINIMAL_MODEL_PANEL_IDS.forEach((id) => {
                const el = get(id);
                if (!el) return;
                if (el.parentElement === body) {
                    if (!el.classList.contains('hidden')) anyVisible = true;
                    return;
                }
                if (minimalPanelOrigins.has(el)) return;
                minimalPanelOrigins.set(el, { parent: el.parentElement, next: el.nextSibling });
                body.appendChild(el);
                if (!el.classList.contains('hidden')) anyVisible = true;
            });
            refreshMinimalModelSection();
        }
        function restoreModelPanelsFromPopup() {
            const body = get('minimal-options-model-body');
            if (!body) return;
            minimalPanelOrigins.forEach((origin, el) => {
                if (origin.parent && origin.parent.contains(el)) {
                    if (origin.next && origin.next.parentNode === origin.parent) origin.parent.insertBefore(el, origin.next);
                    else origin.parent.appendChild(el);
                }
            });
            minimalPanelOrigins.clear();
        }
        function refreshMinimalModelSection() {
            const body = get('minimal-options-model-body');
            const section = get('minimal-options-model-section');
            if (!body || !section) return;
            let anyVisible = false;
            Array.from(body.children).forEach((child) => {
                if (!child.classList.contains('hidden')) anyVisible = true;
            });
            section.classList.toggle('hidden', !anyVisible);
        }

        function openMinimalOptions() {
            if (minimalOptionsOpen || !minimalPromptMode) return;
            hideThinkingSlider();
            minimalOptionsOpen = true;
            renderMinimalOptionItems();
            moveModelPanelsIntoPopup();
            const popup = get('minimal-options-popup');
            if (!popup) return;
            // Clear any inline transform/opacity left by a swipe-to-close so a
            // quick reopen starts from the clean open state.
            const panel = get('minimal-options-panel');
            if (panel) panel.style.cssText = '';
            popup.classList.remove('minimal-options-closing', 'minimal-options-open');
            popup.classList.remove('hidden');
            popup.setAttribute('aria-hidden', 'false');
            void popup.offsetWidth;
            popup.classList.add('minimal-options-open');
        }
        function closeMinimalOptions() {
            if (!minimalOptionsOpen) return;
            minimalOptionsOpen = false;
            const popup = get('minimal-options-popup');
            if (popup) {
                popup.classList.add('minimal-options-closing');
                popup.setAttribute('aria-hidden', 'true');
                setTimeout(() => {
                    if (!minimalOptionsOpen) {
                    popup.classList.remove('minimal-options-open', 'minimal-options-closing');
                    popup.classList.add('hidden');
                    }
                }, 560);
            }
            restoreModelPanelsFromPopup();
            hideThinkingSlider();
        }
        function toggleMinimalOptions() {
            if (minimalOptionsOpen) closeMinimalOptions();
            else openMinimalOptions();
        }
        function refreshMinimalOptionsIfOpen() {
            if (!minimalOptionsOpen) return;
            renderMinimalOptionItems();
            refreshMinimalModelSection();
        }

        // --- Thinking level slide bar ---
        function allowedThinkingValues() {
            return thinkingSliderOptions().map((o) => o.value);
            // All options disabled (e.g. Claude uses budget only) -> nothing to pick.
        }
        function thinkingIndexFromValue(value) {
            const allowed = allowedThinkingValues();
            const idx = allowed.indexOf(String(value));
            return idx < 0 ? Math.max(0, allowed.length - 1) : idx;
        }
        function syncThinkingSliderUi() {
            const slider = get('thinking-slider');
            const label = get('thinking-slide-value');
            const scale = get('thinking-slide-scale');
            const opts = thinkingSliderOptions();
            let idx = opts.findIndex((o) => o.value === thinkingSliderValue());
            if (idx < 0) idx = Math.max(0, opts.length - 1);
            if (slider) {
                slider.min = '0';
                slider.max = String(Math.max(0, opts.length - 1));
                slider.value = String(idx);
            }
            if (label) label.textContent = opts[idx] ? opts[idx].textContent.trim() : THINKING_LEVELS[Math.min(idx, THINKING_LEVELS.length - 1)].label;
            if (scale) {
                scale.innerHTML = '';
                opts.forEach((opt) => {
                    const span = document.createElement('span');
                    span.textContent = opt.textContent.trim();
                    scale.appendChild(span);
                });
            }
        }
        function scheduleThinkingSliderHide() {
            if (thinkingSliderTimer) clearTimeout(thinkingSliderTimer);
            thinkingSliderTimer = setTimeout(() => {
                thinkingSliderTimer = null;
                hideThinkingSlider();
            }, 2500);
        }
        function showThinkingSlider() {
            if (thinkingSliderOpen) {
                scheduleThinkingSliderHide();
                return;
            }
            const bar = get('thinking-slide-bar');
            if (!bar) return;
            // Clear any leftover drag transform from a previous swipe-close so
            // the slider always reappears in the open position.
            const inner = get('thinking-slide-inner');
            if (inner) inner.style.transform = '';
            thinkingSliderOpen = true;
            bar.classList.remove('hidden');
            bar.setAttribute('aria-hidden', 'false');
            syncThinkingSliderUi();
            void bar.offsetWidth;
            bar.classList.add('thinking-slide-open');
            scheduleThinkingSliderHide();
        }
        function hideThinkingSlider() {
            if (thinkingSliderTimer) { clearTimeout(thinkingSliderTimer); thinkingSliderTimer = null; }
            const bar = get('thinking-slide-bar');
            if (!bar) return;
            thinkingSliderOpen = false;
            bar.classList.remove('thinking-slide-open');
            bar.setAttribute('aria-hidden', 'true');
            setTimeout(() => {
                if (!thinkingSliderOpen) bar.classList.add('hidden');
                // Clear any leftover drag transform only after the close
                // transition has finished, so a swipe-to-close fades out from
                // the released position instead of bouncing back to translateY(0).
                const inner = get('thinking-slide-inner');
                if (inner) inner.style.transform = '';
            }, 360);
        }
        // effort モデルでも Enable Thinking と reasoning-effort の表示を
        // 一つの Thinking ボタンとして同期させる。
        function bindThinkingEffortSync() {
            const effortSel = get('reasoning-effort');
            const thinkChk = get('enable-thinking');
            const thinkOpts = get('thinking-options');
            const isEffortMode = () => !!(thinkOpts && thinkOpts.dataset.thinkingMode === 'effort');
            const noneAvailable = () => {
                if (!effortSel) return false;
                const noneOpt = Array.from(effortSel.options).find((opt) => opt.value === 'none');
                return !!(noneOpt && !noneOpt.disabled && !noneOpt.classList.contains('hidden'));
            };
            if (effortSel && effortSel.dataset.thinkingSyncBound !== '1') {
                effortSel.dataset.thinkingSyncBound = '1';
                effortSel.addEventListener('change', () => {
                    if (!isEffortMode() || !thinkChk) return;
                    thinkChk.checked = noneAvailable() ? String(effortSel.value || '').toLowerCase() !== 'none' : true;
                    thinkChk.disabled = !noneAvailable();
                    refreshMinimalOptionItems();
                });
            }
            if (thinkChk && thinkChk.dataset.thinkingEffortBound !== '1') {
                thinkChk.dataset.thinkingEffortBound = '1';
                thinkChk.addEventListener('change', () => {
                    if (!isEffortMode() || !effortSel) return;
                    if (!thinkChk.checked) {
                        effortSel.value = 'none';
                    } else if (!noneAvailable() || String(effortSel.value || '').toLowerCase() === 'none') {
                        const options = Array.from(effortSel.options).filter((opt) => !opt.disabled && !opt.classList.contains('hidden'));
                        const preferred = options.find((opt) => opt.value === 'medium')
                            || options.find((opt) => opt.value === 'high')
                            || options[0];
                        if (preferred) effortSel.value = preferred.value;
                    }
                    effortSel.dispatchEvent(new Event('change', { bubbles: true }));
                    refreshMinimalOptionItems();
                });
            }
        }

        function bindMinimalOptionsEvents() {
            const backdrop = get('minimal-options-backdrop');
            const closeBtn = get('minimal-options-close-btn');
            // Portal the fixed popup before it can receive pointer input.  A
            // transformed composer would otherwise change its containing block.
            const popup = get('minimal-options-popup');
            if (popup && popup.parentNode !== document.body) document.body.appendChild(popup);
            if (backdrop) backdrop.addEventListener('click', () => closeMinimalOptions());
            if (closeBtn) closeBtn.addEventListener('click', () => closeMinimalOptions());
            document.addEventListener('keydown', (e) => {
                if (e.key !== 'Escape') return;
                if (minimalOptionsOpen) {
                    closeMinimalOptions();
                    return;
                }
                if (thinkingSliderOpen) hideThinkingSlider();
            });
            const slider = get('thinking-slider');
            if (slider) {
                slider.addEventListener('input', () => {
                    const idx = Number(slider.value);
                    const allowed = allowedThinkingValues();
                    const sel = activeThinkingSelect();
                    if (allowed.length && sel) {
                        const targetIdx = Math.max(0, Math.min(idx, allowed.length - 1));
                        const value = allowed[targetIdx];
                        const chk = get('enable-thinking');
                        if (value === 'off') {
                            if (chk) { chk.checked = false; chk.dispatchEvent(new Event('change', { bubbles: true })); }
                        } else {
                            if (get('thinking-options')?.dataset.thinkingMode !== 'effort' && chk && !chk.checked) {
                                chk.checked = true;
                                chk.dispatchEvent(new Event('change', { bubbles: true }));
                            }
                            sel.value = value;
                            sel.dispatchEvent(new Event('change', { bubbles: true }));
                        }
                    }
                    syncThinkingSliderUi();
                    scheduleThinkingSliderHide();
                });
            }
            const closeSlider = get('thinking-slide-close-btn');
            if (closeSlider) closeSlider.addEventListener('click', (e) => {
                e.stopPropagation();
                hideThinkingSlider();
            });
            const slideBar = get('thinking-slide-bar');
            if (slideBar) {
                const slideInner = get('thinking-slide-inner');
                slideBar.addEventListener('touchstart', (e) => {
                    if (!thinkingSliderOpen) return;
                    thinkingSliderDragging = true;
                    thinkingSliderStartY = e.touches[0].clientY;
                    thinkingSliderStartX = e.touches[0].clientX;
                    thinkingSliderAxis = null;
                    if (slideInner) slideInner.classList.add('dragging');
                }, { passive: true });
                slideBar.addEventListener('touchmove', (e) => {
                    if (!thinkingSliderDragging) return;
                    const dx = e.touches[0].clientX - thinkingSliderStartX;
                    const dy = e.touches[0].clientY - thinkingSliderStartY;
                    // Lock the gesture axis from the first substantial movement so
                    // a horizontal swipe never moves or dismisses the slider.
                    if (thinkingSliderAxis === null) {
                        if (Math.abs(dx) > 8 || Math.abs(dy) > 8) {
                            thinkingSliderAxis = Math.abs(dy) > Math.abs(dx) ? 'v' : 'h';
                        }
                    }
                    if (thinkingSliderAxis !== 'v') return;
                    if (dy > 0) {
                        if (e.cancelable) e.preventDefault();
                        // Dead zone + dampened travel keeps the drag stable.
                        const travel = Math.min((dy - 8) * 0.5, 120);
                        if (slideInner) slideInner.style.transform = travel > 0 ? `translateY(${travel}px)` : '';
                    } else if (slideInner) {
                        slideInner.style.transform = '';
                    }
                }, { passive: false });
                slideBar.addEventListener('touchend', (e) => {
                    if (!thinkingSliderDragging) return;
                    thinkingSliderDragging = false;
                    const dy = e.changedTouches[0].clientY - thinkingSliderStartY;
                    if (slideInner) slideInner.classList.remove('dragging');
                    if (thinkingSliderAxis === 'v' && dy > 100) {
                        // Close from the released position: keep the inner below
                        // the open state so the bar fades out without bouncing
                        // back up to translateY(0) first.
                        if (slideInner) slideInner.style.transform = `translateY(${Math.max(dy * 0.5, 60)}px)`;
                        hideThinkingSlider();
                    } else {
                        if (slideInner) slideInner.style.transform = '';
                        scheduleThinkingSliderHide();
                    }
                }, { passive: true });
                slideBar.addEventListener('touchcancel', () => {
                    thinkingSliderDragging = false;
                    if (slideInner) {
                        slideInner.classList.remove('dragging');
                        slideInner.style.transform = '';
                    }
                    scheduleThinkingSliderHide();
                }, { passive: true });
            }
            // Swipe down on the popup panel to close it (bottom-sheet gesture).
            const popupPanel = get('minimal-options-panel');
            if (popupPanel) {
                popupPanel.addEventListener('touchstart', (e) => {
                    if (!minimalOptionsOpen) return;
                    popupSwipeDragging = true;
                    popupSwipeStartY = e.touches[0].clientY;
                    popupSwipeStartX = e.touches[0].clientX;
                    popupSwipeAxis = null;
                    let node = e.target instanceof Element ? e.target : null;
                    let atTop = true;
                    while (node && node !== popupPanel) {
                        if (node.scrollTop > 0) { atTop = false; break; }
                        node = node.parentElement;
                    }
                    popupSwipeAtTop = atTop;
                    if (atTop) popupPanel.classList.add('dragging');
                }, { passive: true });
                popupPanel.addEventListener('touchmove', (e) => {
                    if (!popupSwipeDragging || !popupSwipeAtTop || !minimalOptionsOpen) return;
                    const dx = e.touches[0].clientX - popupSwipeStartX;
                    const dy = e.touches[0].clientY - popupSwipeStartY;
                    // Lock the gesture axis from the first substantial movement so
                    // a horizontal swipe never drags or closes the popup.
                    if (popupSwipeAxis === null) {
                        if (Math.abs(dx) > 8 || Math.abs(dy) > 8) {
                            popupSwipeAxis = Math.abs(dy) > Math.abs(dx) ? 'v' : 'h';
                        }
                    }
                    if (popupSwipeAxis !== 'v') return;
                    if (dy > 0) {
                        if (e.cancelable) e.preventDefault();
                        popupPanel.style.transform = `translateY(${Math.min(dy * 0.6, 140)}px)`;
                    }
                }, { passive: false });
                popupPanel.addEventListener('touchend', (e) => {
                    if (!popupSwipeDragging) return;
                    popupSwipeDragging = false;
                    const dy = e.changedTouches[0].clientY - popupSwipeStartY;
                    popupPanel.classList.remove('dragging');
                    if (popupSwipeAtTop && popupSwipeAxis !== 'h' && dy > 70) {
                        // Close from the released position: keep the panel below
                        // the open state and fade it out so it never bounces back
                        // up to translateY(0) before closing.
                        popupPanel.style.transform = `translateY(${Math.max(dy * 0.6, 100)}px)`;
                        popupPanel.style.opacity = '0';
                        closeMinimalOptions();
                    } else {
                        // Snap back to the open position.
                        popupPanel.style.transform = '';
                    }
                }, { passive: true });
                popupPanel.addEventListener('touchcancel', () => {
                    popupSwipeDragging = false;
                    popupPanel.classList.remove('dragging');
                    popupPanel.style.transform = '';
                    popupPanel.style.opacity = '';
                }, { passive: true });
            }
        }
        function bindUploadButton() {
            const btn = get('upload-btn');
            if (!btn) return;
            btn.onclick = () => {
                if (minimalPromptMode) toggleMinimalOptions();
                else openUploadModal();
            };
        }

        function applyChatDefaults(d) {
            if (!d) return;
            if (Object.prototype.hasOwnProperty.call(d, 'voice_studio_ui')) {
                voiceStudioUiEnabled = d.voice_studio_ui !== false;
            }
            applyTemporaryChatTimeoutSeconds(d.temp_chat_timeout_seconds);
            if (chatDefaultsLoaded) return;
            const useLast = !!d.use_last_chat_settings;
            const src = useLast ? {
                model: d.last_model,
                enable_search: d.last_enable_search,
                gemini_free_tier: d.last_gemini_free_tier,
                gemini_flex: d.last_gemini_flex,
                enable_url_context: d.last_enable_url_context,
                enable_maps: d.last_enable_maps,
                enable_python: d.last_enable_python,
                enable_file_creation: d.last_enable_file_creation,
                enable_thinking: d.last_enable_thinking,
                thinking_level: d.last_thinking_level,
                thinking_budget: d.last_thinking_budget,
                reasoning_effort: d.last_reasoning_effort,
                enable_system_prompt: d.last_enable_system_prompt,
                enable_mcp: d.last_enable_mcp,
                safety_setting: d.last_safety_setting
            } : {
                model: d.default_model,
                enable_search: d.default_enable_search,
                gemini_free_tier: d.default_gemini_free_tier,
                gemini_flex: d.default_gemini_flex,
                enable_url_context: d.default_enable_url_context,
                enable_maps: d.default_enable_maps,
                enable_python: d.default_enable_python,
                enable_file_creation: d.default_enable_file_creation,
                enable_thinking: d.default_enable_thinking,
                thinking_level: d.default_thinking_level,
                thinking_budget: d.default_thinking_budget,
                reasoning_effort: d.default_reasoning_effort,
                enable_system_prompt: d.default_enable_system_prompt,
                enable_mcp: d.default_enable_mcp,
                safety_setting: d.default_safety_setting
            };
            const s = (v, fallback) => (v === undefined || v === null || v === "") ? fallback : v;
            if (src.model) selectModelById(src.model);
            if (get('enable-search')) get('enable-search').checked = !!s(src.enable_search, get('enable-search').checked);
            if (get('enable-gemini-free-tier')) get('enable-gemini-free-tier').checked = !!s(src.gemini_free_tier, get('enable-gemini-free-tier').checked);
            if (get('enable-gemini-flex')) get('enable-gemini-flex').checked = !!s(src.gemini_flex, get('enable-gemini-flex').checked);
            if (get('enable-url-context')) get('enable-url-context').checked = !!s(src.enable_url_context, get('enable-url-context').checked);
            if (get('enable-maps')) get('enable-maps').checked = !!s(src.enable_maps, get('enable-maps').checked);
            if (get('enable-python')) get('enable-python').checked = !!s(src.enable_python, get('enable-python').checked);
            if (get('enable-file-creation')) get('enable-file-creation').checked = !!s(src.enable_file_creation, get('enable-file-creation').checked);
            if (get('enable-thinking')) get('enable-thinking').checked = !!s(src.enable_thinking, get('enable-thinking').checked);
            if (get('thinking-level')) get('thinking-level').value = s(src.thinking_level, get('thinking-level').value || "high");
            if (get('thinking-budget')) get('thinking-budget').value = s(src.thinking_budget, get('thinking-budget').value || 4096);
            if (get('reasoning-effort')) get('reasoning-effort').value = s(src.reasoning_effort, get('reasoning-effort').value || "medium");
            if (get('enable-sys-prompt')) get('enable-sys-prompt').checked = !!s(src.enable_system_prompt, get('enable-sys-prompt').checked);
            if (get('enable-mcp')) get('enable-mcp').checked = !!s(src.enable_mcp, get('enable-mcp').checked);
            if (get('safety-setting')) get('safety-setting').value = s(src.safety_setting, get('safety-setting').value || "default");
            chatDefaultsLoaded = true;
            // toggleOptions は DOMContentLoaded 内（part08）で定義され、window 経由で公開される。
            if (typeof window.toggleOptions === 'function') window.toggleOptions();
            if (typeof window.applyMathStepsChatDefaults === 'function') window.applyMathStepsChatDefaults(d, useLast);
            applyMcpPromptChipUi();
        }
        function setEditUi(active) {
            const bar = get('edit-bar');
            if (!bar) return;
            if (active) {
                bar.classList.remove('hidden');
                bar.classList.add('flex');
            } else {
                bar.classList.add('hidden');
                bar.classList.remove('flex');
            }
            updatePromptPlaceholder();
            syncCodingModeUi(codingModeEnabled, { persist: false });
        }
        const PROMPT_OPTION_INPUT_IDS = {
            enable_search: 'enable-search', gemini_free_tier: 'enable-gemini-free-tier', gemini_flex: 'enable-gemini-flex', enable_url_context: 'enable-url-context',
            enable_maps: 'enable-maps', enable_python: 'enable-python',
            enable_file_creation: 'enable-file-creation', enable_mcp: 'enable-mcp',
            enable_skills: 'enable-skills',
            enable_thinking: 'enable-thinking', thinking_level: 'thinking-level',
            thinking_budget: 'thinking-budget', reasoning_effort: 'reasoning-effort',
            safety_setting: 'safety-setting', enable_prompt_caching: 'enable-prompt-cache',
            image_vision_model: 'image-vision-model',
            tts_voice: 'tts-voice', tts_voice_custom: 'tts-voice-custom',
            tts_language: 'tts-language', tts_speed: 'tts-speed',
            image_size: 'gpt-image-size', image_quality: 'gpt-image-quality',
            image_complexity: 'gpt-image-complexity',
            image_format: 'gpt-image-format', image_compression: 'gpt-image-compression',
            image_frame_count: 'gpt-image-frame-count', image_frame_duration_ms: 'gpt-image-frame-duration',
            gemini_image_aspect: 'gemini-image-aspect', gemini_image_size: 'gemini-image-size',
            grok_image_aspect: 'grok-image-aspect', grok_image_resolution: 'grok-image-resolution',
            grok_image_quality: 'grok-image-quality', grok_image_format: 'grok-image-format',
            grok_image_count: 'grok-image-count', ideogram_operation: 'ideogram-image-operation', ideogram_aspect: 'ideogram-image-aspect',
            ideogram_resolution: 'ideogram-image-resolution', ideogram_quality: 'ideogram-image-quality',
            ideogram_speed: 'ideogram-image-speed', ideogram_magic_prompt: 'ideogram-image-magic',
            ideogram_style_type: 'ideogram-image-style', ideogram_count: 'ideogram-image-count',
            ideogram_seed: 'ideogram-image-seed', xai_temperature: 'xai-temperature',
            xai_top_p: 'xai-top-p', xai_max_completion_tokens: 'xai-max-completion-tokens',
            xai_seed: 'xai-seed', xai_presence_penalty: 'xai-presence-penalty',
            xai_frequency_penalty: 'xai-frequency-penalty', xai_response_format: 'xai-response-format',
            xai_tool_choice: 'xai-tool-choice', xai_parallel_tool_calls: 'xai-parallel-tool-calls',
            xai_logprobs: 'xai-logprobs', xai_top_logprobs: 'xai-top-logprobs',
            grok_video_duration: 'grok-video-duration', grok_video_aspect: 'grok-video-aspect',
            grok_video_resolution: 'grok-video-resolution', gemini_video_duration: 'gemini-video-duration',
            gemini_video_aspect: 'gemini-video-aspect', gemini_video_resolution: 'gemini-video-resolution',
            music_instrumental: 'music-instrumental', ocr_table_format: 'ocr-table-format',
            ocr_extract_header: 'ocr-extract-header', ocr_extract_footer: 'ocr-extract-footer',
            ocr_include_blocks: 'ocr-include-blocks', ocr_include_image_base64: 'ocr-include-images',
            ocr_pages: 'ocr-pages',
            // Free-text fields are included in ephemeral edit snapshots, never in saved prompt options.
            tts_style: 'tts-style', ideogram_negative_prompt: 'ideogram-image-negative', xai_stop: 'xai-stop'
        };
        // Kept in memory for this edit only, never persisted or sent with the prompt.
        let promptSettingsBeforeEdit = null;
        function capturePromptSettingsBeforeEdit() {
            const options = {};
            for (const [key, id] of Object.entries(PROMPT_OPTION_INPUT_IDS)) {
                const el = get(id);
                if (el) options[key] = el.type === 'checkbox' ? !!el.checked : el.value;
            }
            Object.assign(options, { canvas_mode: !!canvasModeEnabled, coding_mode: !!codingModeEnabled,
                browser_fast_mode: !!browserFastModeEnabled, batch_mode: !!get('enable-batch-mode')?.checked,
                math_steps: (typeof window.collectMathStepsSendOption === 'function' ? window.collectMathStepsSendOption() : false) });
            const sys = get('enable-sys-prompt');
            return { model: get('model-select')?.value, prompt_options: options,
                modelChosenManually, visionModel: currentVisionModel, mask: currentMaskImage,
                codingTarget: codingTargetSelection,
                systemPrompt: sys ? { checked: sys.checked, restoreChecked: sys.dataset.restoreChecked } : null };
        }
        function restorePromptSettingsBeforeEdit(saved) {
            // A cancel restores the selection from before editing, including its model and cache flag.
            modelChosenManually = false;
            try { restorePromptOptionsForEdit(saved); }
            finally { modelChosenManually = saved.modelChosenManually; }
            currentVisionModel = saved.visionModel;
            currentMaskImage = saved.mask;
            codingTargetSelection = saved.codingTarget;
            const sys = get('enable-sys-prompt');
            if (sys && saved.systemPrompt) {
                sys.checked = saved.systemPrompt.checked;
                if (saved.systemPrompt.restoreChecked === undefined) delete sys.dataset.restoreChecked;
                else sys.dataset.restoreChecked = saved.systemPrompt.restoreChecked;
                sys.dispatchEvent(new Event('change', { bubbles: true }));
            }
            updateMaskPreview();
            syncCodingModeUi(codingModeEnabled, { persist: false });
            schedulePromptTokenEstimate(true);
        }
        function cancelEdit() {
            const wasEditing = !!editingMessageId;
            const savedSettings = promptSettingsBeforeEdit;
            promptSettingsBeforeEdit = null;
            editingMessageId = null;
            currentParentId = currentLeafId || null;
            const input = get('prompt-input');
            if (input) {
                input.value = '';
                input.style.height = 'auto';
            }
            currentImageUrls = [];
            get('file-preview').classList.add('hidden');
            get('file-input').value = '';
            clearQuote();
            setEditUi(false);
            if (wasEditing && savedSettings) restorePromptSettingsBeforeEdit(savedSettings);
        }
        function restoreLatestPromptOptions() {
            if (editingMessageId || !currentLeafId || !allMessages.length) return;
            const byId = new Map(allMessages.map(message => [String(message.id), message]));
            let message = byId.get(String(currentLeafId));
            while (message && message.role !== 'user') {
                message = byId.get(String(message.parent_id));
            }
            if (message && message.prompt_options) restorePromptOptionsForEdit(message);
        }
        function restorePromptOptionsForEdit(message) {
            const options = message && message.prompt_options;
            // The message model belongs to this prompt; the thread's last model may differ.
            // A model the user picked in this chat stays selected; the prompt's options still come back.
            if (!modelChosenManually && message.model && get('model-select') && get('model-select').value !== message.model) {
                const cache = get('enable-prompt-cache');
                if (cache) cache.checked = false;
                selectModelById(message.model);
            }
            if (!options || typeof options !== 'object' || Array.isArray(options)) return;
            const ids = PROMPT_OPTION_INPUT_IDS;
            for (const [key, id] of Object.entries(ids)) {
                if (!Object.prototype.hasOwnProperty.call(options, key)) continue;
                // PromptCache follows the model's provider, so it stays as is with a kept model.
                if (key === 'enable_prompt_caching' && modelChosenManually) continue;
                const el = get(id);
                if (!el) continue;
                if (el.type === 'checkbox') el.checked = !!options[key];
                else el.value = String(options[key]);
                if (key === 'enable_prompt_caching') updatePromptCacheUi();
                else el.dispatchEvent(new Event('change', { bubbles: true }));
            }
            if (Object.prototype.hasOwnProperty.call(options, 'image_vision_model')) {
                currentVisionModel = options.image_vision_model;
            }
            if (Object.prototype.hasOwnProperty.call(options, 'canvas_mode')) {
                syncCanvasModeUi(!!options.canvas_mode, { persist: false });
            }
            if (Object.prototype.hasOwnProperty.call(options, 'math_steps')) {
                if (typeof window.setMathStepsEnabled === 'function') window.setMathStepsEnabled(!!options.math_steps, { persist: false });
            }
            if (Object.prototype.hasOwnProperty.call(options, 'coding_mode')) {
                syncCodingModeUi(!!options.coding_mode, { persist: false });
            }
            if (Object.prototype.hasOwnProperty.call(options, 'browser_fast_mode')) {
                setBrowserFastModeEnabled(!!options.browser_fast_mode);
            }
            if (Object.prototype.hasOwnProperty.call(options, 'batch_mode')) {
                const batch = get('enable-batch-mode');
                if (batch) { batch.checked = !!options.batch_mode; batch.dispatchEvent(new Event('change', { bubbles: true })); }
            }
            if (typeof window.toggleOptions === 'function') window.toggleOptions();
            applyMcpPromptChipUi();
        }
        function beginEditMessage(id, autoSend = false) {
            const text = messageStore[id];
            if (text === undefined || text === null) return;
            if (!editingMessageId || !promptSettingsBeforeEdit) promptSettingsBeforeEdit = capturePromptSettingsBeforeEdit();
            const input = get('prompt-input');
            input.value = text || '';
            input.focus();
            input.style.height = 'auto';
            input.style.height = input.scrollHeight + 'px';

            const msg = allMessages.find(m => m.id == id);
            const meta = messageMeta[id] || {};
            restorePromptOptionsForEdit(msg);
            currentMaskImage = (msg && msg.prompt_options && msg.prompt_options.image_mask) || meta.image_mask || null;
            updateMaskPreview();

            // Set parent_id for branching
            if (msg) {
                currentParentId = (msg.parent_id === undefined ? null : msg.parent_id);
            } else if (meta.parent_id !== undefined) {
                currentParentId = meta.parent_id;
            }

            editingMessageId = id;
            setEditUi(true);

            const imageUrl = msg ? msg.image_url : meta.image_url;
            if (imageUrl) {
                try {
                    const imgs = JSON.parse(imageUrl);
                    if (Array.isArray(imgs) && imgs.length) {
                        currentImageUrls = imgs.map((u) => {
                            let src = 'unknown';
                            let path = u;
                            if (u && typeof u === 'object') {
                                src = normalizeAttachmentSource(u.source);
                                path = u.filepath || u.path || u.url || u.file || '';
                            }
                            const norm = normalizeAttachmentPath(path);
                            if (norm) setAttachmentSourceForPath(norm, src);
                            return norm;
                        }).filter(Boolean);
                        get('file-preview').classList.remove('hidden');
                        get('file-name').innerText = `${currentImageUrls.length} files ready`;
                    } else {
                        currentImageUrls = [];
                        get('file-preview').classList.add('hidden');
                        get('file-input').value = '';
                    }
                } catch (e) {
                    currentImageUrls = [];
                    get('file-preview').classList.add('hidden');
                    get('file-input').value = '';
                }
            } else {
                currentImageUrls = [];
                get('file-preview').classList.add('hidden');
                get('file-input').value = '';
            }

            const quoteText = msg ? msg.quote_text : meta.quote_text;
            if (quoteText) {
                currentQuote = quoteText;
                currentQuoteMessageId = null;
                renderQuoteBarContent(currentQuote);
                get('quote-bar').classList.add('visible');
            } else {
                clearQuote();
            }
            schedulePromptTokenEstimate(true);
            if (autoSend) sendMessage();
        }
        function playSendAnimation() {
            const btn = get('send-btn');
            if (!btn) return;
            btn.classList.remove('fly');
            void btn.offsetWidth;
            btn.classList.add('fly');
        }
        function setSendBtnToStopMode() {
            const btn = get('send-btn');
            if (!btn) return;
            btn.onclick = stopGeneration;
            isStopMode = true;
            btn.disabled = false;
            const applyStopUi = () => {
                if (!btn || !isStopMode) return;
                btn.classList.add('stop-mode');
                btn.innerHTML = '<span style="font-size:20px;line-height:1;color:#fff;">■</span>';
                btn.classList.add('btn-swap');
                setTimeout(() => btn.classList.remove('btn-swap'), 300);
            };
            if (btn.classList.contains('fly')) {
                const onEnd = (e) => {
                    if (e.animationName !== 'sendBtnPop') return;
                    btn.removeEventListener('animationend', onEnd);
                    applyStopUi();
                };
                btn.addEventListener('animationend', onEnd);
                setTimeout(applyStopUi, 700);
            } else {
                applyStopUi();
            }
        }
        function setSendBtnToSendMode() {
            const btn = get('send-btn');
            if (!btn) return;
            btn.classList.remove('stop-mode', 'fly', 'btn-swap');
            btn.innerHTML = '<i class="fas fa-paper-plane"></i>';
            btn.classList.add('btn-swap');
            setTimeout(() => btn.classList.remove('btn-swap'), 300);
            btn.onclick = sendMessage;
            isStopMode = false;
        }
        async function stopGeneration() {
            const stopThreadId = (currentThreadId !== null && currentThreadId !== undefined && currentThreadId !== '') ? String(currentThreadId) : null;
            const stopJobId = normalizeJobIdForUi(currentJobId);
            const stopSeq = ++manualStopSeq;
            const partialSnapshot = captureStoppedPartialBubbleSnapshot(getActiveStreamingBubbleElement());
            manualStopContext = { seq: stopSeq, threadId: stopThreadId, jobId: stopJobId, partialSnapshot };
            if (stopJobId) suppressPendingJob(stopJobId);
            if(abortController) abortController.abort();
            try {
                if(stopJobId || stopThreadId) {
                    const stopPayload = {};
                    if (stopJobId) stopPayload.job_id = stopJobId;
                    if (stopThreadId) stopPayload.thread_id = stopThreadId;
                    const stopRes = await apiFetch("/api/stop_chat", {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(stopPayload)});
                    const stopData = await stopRes.json().catch(() => ({}));
                    const resolvedStopJobId = normalizeJobIdForUi(stopData && stopData.job_id);
                    if (resolvedStopJobId) {
                        suppressPendingJob(resolvedStopJobId);
                        if (manualStopContext && manualStopContext.seq === stopSeq) {
                            manualStopContext.jobId = resolvedStopJobId;
                        }
                    }
                }
                if (manualStopContext && manualStopContext.seq === stopSeq) {
                    const synced = await syncThreadAfterAbortedStream(stopThreadId, { retries: 2, retryDelayMs: 180, notifyOnFailure: true });
                    if (synced && manualStopContext.partialSnapshot) {
                        appendStoppedPartialBubbleSnapshot(manualStopContext.partialSnapshot, stopThreadId);
                    }
                }
            } finally {
                if (manualStopContext && manualStopContext.seq === stopSeq) {
                    manualStopContext = null;
                }
                setSendBtnToSendMode();
                updateFilePreview();
            }
        }
        async function purgeCaches() {
            if ('caches' in window) {
                const keys = await caches.keys();
                await Promise.all(keys.map(k => caches.delete(k)));
            }
            if (navigator.serviceWorker) {
                const regs = await navigator.serviceWorker.getRegistrations();
                await Promise.all(regs.map(r => r.unregister()));
            }
        }
        const SW_CACHE_MODE_STORAGE_KEY = 'ai_sw_cache_mode_v2';
        async function applyCacheMode(enable, options = {}) {
            if (!('serviceWorker' in navigator)) return;
            if (enable) {
                try {
                    await navigator.serviceWorker.register(`/sw.js?v=${encodeURIComponent(appVersion)}`);
                    localStorage.setItem(SW_CACHE_MODE_STORAGE_KEY, 'enabled');
                } catch (e) {}
            } else {
                // Do not scan Cache Storage and unregister workers on every page load.
                // Purge once when migrating from the old behaviour or when the user
                // explicitly changes the setting from enabled to disabled.
                const previousMode = localStorage.getItem(SW_CACHE_MODE_STORAGE_KEY);
                const needsCleanup = !!options.forceCleanup || previousMode !== 'disabled';
                if (needsCleanup) await purgeCaches();
                localStorage.setItem(SW_CACHE_MODE_STORAGE_KEY, 'disabled');
            }
        }
        function checkAndNotifyVersion(latest) {
            if (!latest || !appVersion) return;
            if (latest === appVersion) return;
            const notified = localStorage.getItem("version_notified") || "";
            if (notified === latest) return;
            localStorage.setItem("app_version", latest);
            syncVersionUpdateCachePreferenceUi();
            showModal("version-update-modal");
        }
        async function checkVersion() {
            try {
                const res = await fetch("/api/version", { cache: "no-store" });
                if (!res.ok) return;
                const data = await res.json();
                const latest = data.version || "";
                const stored = localStorage.getItem("app_version") || "";
                if (latest && !stored) localStorage.setItem("app_version", latest);
                if (latest && stored && latest !== stored) {
                    await purgeCaches();
                    checkAndNotifyVersion(latest);
                }
            } catch (e) {}
        }
        async function fetchChatStreamWithUnavailableRetry(url, options, pendingBubble) {
            let retryCount = 0;
            while (true) {
                if (options.signal && options.signal.aborted) throw new DOMException('Aborted', 'AbortError');
                try {
                    const response = await apiFetch(url, options);
                    const unavailableMode = window.ConnectionMonitor.retryModeForResponse(response);
                    let submissionProcessing = false;
                    if (response.status === 425) {
                        const pendingPayload = await response.clone().json().catch(() => ({}));
                        submissionProcessing = pendingPayload.code === 'submission_in_progress';
                    }
                    if (!unavailableMode && !submissionProcessing) {
                        window.ConnectionMonitor.markReachable();
                        return response;
                    }
                    retryCount += 1;
                    if (unavailableMode) window.ConnectionMonitor.setUnavailable(unavailableMode);
                    updatePendingSkeletonStatus(
                        pendingBubble,
                        unavailableMode === 'maintenance' ? 'メンテナンス終了を待っています...' : 'サーバーの復帰を待っています...',
                        `送信内容を保持して自動再試行中（${retryCount}回目）`
                    );
                } catch (error) {
                    if ((options.signal && options.signal.aborted) || error.name === 'AbortError') throw error;
                    retryCount += 1;
                    const unavailableMode = 'offline';
                    window.ConnectionMonitor.setUnavailable(unavailableMode);
                    updatePendingSkeletonStatus(
                        pendingBubble,
                        'インターネット接続の復帰を待っています...',
                        `送信内容を保持して自動再試行中（${retryCount}回目）`
                    );
                }
                await window.ConnectionMonitor.waitForRetry(options.signal);
            }
        }
        function createClientRequestId() {
            if (window.crypto && typeof window.crypto.randomUUID === 'function') return window.crypto.randomUUID();
            const randomPart = window.crypto && typeof window.crypto.getRandomValues === 'function'
                ? Array.from(window.crypto.getRandomValues(new Uint32Array(4))).map((value) => value.toString(16)).join('')
                : `${Date.now().toString(16)}${Math.random().toString(16).slice(2)}`;
            return `req-${randomPart}`.slice(0, 64);
        }
