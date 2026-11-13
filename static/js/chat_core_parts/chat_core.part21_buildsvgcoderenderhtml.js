        //! @chat-core-bundle-split: scripts/build_frontend.sh starts the next browser file here (see README.md).
        function buildSvgCodeRenderHtml(code, key) {
            let fill = true;
            const markup = String(code || '').replace(/<svg\b([^>]*)>/i, (tag, attrs) => {
                let next = attrs;
                // An <img> parses SVG as XML: the root needs the SVG namespace (and xlink when used).
                if (!/\sxmlns\s*=/i.test(next)) next = ` xmlns="http://www.w3.org/2000/svg"${next}`;
                if (/\bxlink:/i.test(code) && !/\sxmlns:xlink\s*=/i.test(next)) next = ` xmlns:xlink="http://www.w3.org/1999/xlink"${next}`;
                const width = next.match(/\swidth\s*=\s*["']?\s*([\d.]+)\s*(px)?\s*(?:["'\s/]|$)/i);
                if (width && Number(width[1]) > 0) fill = false;
                return `<svg${next}>`;
            });
            const src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(markup)}`;
            return `<div class="svg-render-box svg-code-render" data-svg-key="${escapeHtml(String(key || ''))}"><img src="${src}" alt="SVG"${fill ? ' class="svg-code-fill"' : ''} decoding="async"></div>`;
        }
        function getCanvasModeElements() {
            const panel = get('canvas-panel');
            if (!panel) return null;
            return {
                panel,
                stage: get('conversation-stage'),
                title: get('canvas-panel-title'),
                status: get('canvas-panel-status'),
                blockCount: get('canvas-block-count'),
                blockList: get('canvas-block-list'),
                panelTabs: get('canvas-panel-tabs'),
                previewLang: get('canvas-preview-lang'),
                sourceSelect: get('canvas-source-select'),
                frame: get('canvas-preview-frame'),
                empty: get('canvas-preview-empty'),
                sourceScroll: get('canvas-source-scroll'),
                code: get('canvas-code-text'),
                copyBtn: get('canvas-panel-copy-btn'),
                clearBtn: get('canvas-panel-clear-btn'),
                closeBtn: get('canvas-panel-close-btn')
            };
        }
        function isCanvasHtmlPreviewCandidate(lang, code) {
            const token = String(lang || '').trim().toLowerCase();
            if (token === 'html' || token === 'htm' || token === 'xhtml') return true;
            if (token) return false;
            const raw = String(code || '');
            return /<!doctype\s+html/i.test(raw) || /<html[\s>]/i.test(raw);
        }
        function normalizeCanvasBlock(block, index) {
            const lang = String(block && block.lang ? block.lang : '').trim();
            const code = String(block && block.code !== undefined && block.code !== null ? block.code : '');
            const open = !!(block && block.open);
            return {
                ...block,
                index,
                lang,
                code,
                open,
                key: hashString(`${lang || 'TEXT'}\n${code || ''}`)
            };
        }
        function parseCanvasMarkdown(text) {
            const rawText = String(text || '');
            const lines = rawText.split(/\r?\n/);
            const blocks = [];
            const output = [];
            const fenceStartRe = /^(\s*)(`{3,}|~{3,})(.*)$/;
            let activeFence = null;
            let activeLang = '';
            let activeBuffer = [];
            for (const line of lines) {
                if (!activeFence) {
                    const match = line.match(fenceStartRe);
                    if (match) {
                        activeFence = match[2];
                        activeLang = String(match[3] || '').trim();
                        activeBuffer = [];
                        blocks.push({ lang: activeLang, code: '', open: true });
                        output.push('<div class="canvas-code-placeholder">Canvasで表示中</div>');
                        continue;
                    }
                    output.push(line);
                    continue;
                }
                const trimmed = String(line || '').trim();
                if (trimmed && trimmed.replace(/\s+/g, '') === activeFence) {
                    const block = blocks[blocks.length - 1];
                    if (block) {
                        block.code = activeBuffer.join('\n');
                        block.open = false;
                    }
                    activeFence = null;
                    activeLang = '';
                    activeBuffer = [];
                    continue;
                }
                activeBuffer.push(line);
                const block = blocks[blocks.length - 1];
                if (block) block.code = activeBuffer.join('\n');
            }
            if (activeFence && blocks.length) {
                const block = blocks[blocks.length - 1];
                if (block) {
                    block.code = activeBuffer.join('\n');
                    block.open = true;
                }
            }
            const normalizedBlocks = blocks.map((block, index) => normalizeCanvasBlock(block, index));
            const primarySelection = selectCanvasPreviewBlock(normalizedBlocks, rawText);
            return {
                renderText: output.join('\n'),
                blocks: normalizedBlocks,
                primaryBlock: primarySelection ? primarySelection.block : null,
                primaryIndex: primarySelection ? primarySelection.index : -1,
                rawText
            };
        }
        function selectCanvasPreviewBlock(blocks, rawText = '', preferredIndex = -1) {
            const list = Array.isArray(blocks) ? blocks : [];
            if (Number.isInteger(preferredIndex) && preferredIndex >= 0 && preferredIndex < list.length) {
                const preferred = list[preferredIndex];
                return {
                    block: preferred,
                    index: preferredIndex,
                    previewType: isCanvasHtmlPreviewCandidate(preferred.lang, preferred.code) ? 'html' : 'code'
                };
            }
            if (list.length > 0) {
                const index = list.length - 1;
                const block = list[index];
                return {
                    block,
                    index,
                    previewType: isCanvasHtmlPreviewCandidate(block.lang, block.code) ? 'html' : 'code'
                };
            }
            const raw = String(rawText || '');
            if (isCanvasHtmlPreviewCandidate('', raw)) {
                const fallback = normalizeCanvasBlock({ lang: 'html', code: raw, open: true, fallback: true }, 0);
                return { block: fallback, index: -1, previewType: 'html' };
            }
            return null;
        }
        function getCanvasSelectedBlock() {
            const blocks = Array.isArray(canvasPreviewState.blocks) ? canvasPreviewState.blocks : [];
            if (!blocks.length) {
                const raw = String(canvasPreviewState.rawText || '');
                if (isCanvasHtmlPreviewCandidate('', raw)) {
                    return { block: normalizeCanvasBlock({ lang: 'html', code: raw, open: true, fallback: true }, 0), index: -1 };
                }
                return null;
            }
            const preferredIndex = Number.isInteger(canvasPreviewState.selectedIndex) ? canvasPreviewState.selectedIndex : -1;
            const selection = selectCanvasPreviewBlock(blocks, canvasPreviewState.rawText, preferredIndex);
            if (!selection || !selection.block) return null;
            return selection;
        }
        function syncCanvasPreviewButtons(root = document) {
            if (!root || typeof root.querySelectorAll !== 'function') return;
            const selectedKey = String(canvasPreviewState.selectedKey || '');
            root.querySelectorAll('.canvas-preview-btn').forEach((btn) => {
                const codeKey = String(btn.getAttribute('data-code-key') || '');
                const isActive = !!selectedKey && selectedKey === codeKey;
                btn.classList.toggle('canvas-active', isActive);
                btn.setAttribute('aria-pressed', isActive ? 'true' : 'false');
                btn.setAttribute('data-canvas-active', isActive ? '1' : '0');
                btn.innerHTML = isActive
                    ? '<i class="fas fa-layer-group"></i>'
                    : '<i class="fas fa-window-restore"></i>';
                btn.title = isActive ? 'Canvasで表示中' : 'Canvasでプレビューする';
                btn.setAttribute('aria-label', isActive ? 'Canvasで表示中' : 'Canvasでプレビューする');
            });
        }
        function isCanvasMobileLayout() {
            try {
                return window.matchMedia('(max-width: 1023px)').matches;
            } catch (e) {
                return false;
            }
        }
        function animateCanvasMobileViewEntry(els, previousView, nextView) {
            if (!els || !isCanvasMobileLayout() || previousView === nextView) return;
            const sectionByView = {
                preview: get('canvas-preview-shell'),
                blocks: get('canvas-block-shell'),
                source: get('canvas-source-shell')
            };
            const viewOrder = { preview: 0, blocks: 1, source: 2 };
            const section = sectionByView[nextView];
            if (!section || !(previousView in viewOrder) || !(nextView in viewOrder)) return;
            canvasPreviewState.viewAnimationToken += 1;
            const animationToken = canvasPreviewState.viewAnimationToken;
            if (canvasPreviewState.viewAnimationTimer) {
                clearTimeout(canvasPreviewState.viewAnimationTimer);
                canvasPreviewState.viewAnimationTimer = null;
            }
            Object.values(sectionByView).forEach((item) => {
                if (!item) return;
                item.classList.remove('canvas-view-enter-from-left', 'canvas-view-enter-from-right');
            });
            void section.offsetWidth;
            const directionClass = viewOrder[nextView] < viewOrder[previousView]
                ? 'canvas-view-enter-from-left'
                : 'canvas-view-enter-from-right';
            section.classList.add(directionClass);
            canvasPreviewState.viewAnimationTimer = setTimeout(() => {
                if (animationToken !== canvasPreviewState.viewAnimationToken) return;
                section.classList.remove(directionClass);
                canvasPreviewState.viewAnimationTimer = null;
            }, 340);
        }
        function syncCanvasPanelViewUi(view = canvasPreviewState.mobileView, options = {}) {
            const els = getCanvasModeElements();
            if (!els || !els.panel) return;
            const nextView = ['preview', 'blocks', 'source'].includes(view) ? view : 'preview';
            const previousView = ['preview', 'blocks', 'source'].includes(options.fromView)
                ? options.fromView
                : canvasPreviewState.mobileView;
            canvasPreviewState.mobileView = nextView;
            els.panel.dataset.canvasMobileView = nextView;
            const tabs = els.panelTabs ? Array.from(els.panelTabs.querySelectorAll('[data-canvas-panel-view]')) : [];
            tabs.forEach((btn) => {
                const active = btn.getAttribute('data-canvas-panel-view') === nextView;
                btn.classList.toggle('active', active);
                btn.setAttribute('aria-pressed', active ? 'true' : 'false');
            });
            if (options.animate === true) {
                animateCanvasMobileViewEntry(els, previousView, nextView);
            }
            if (options.focus !== false && isCanvasMobileLayout()) {
                if (nextView === 'preview' && els.frame && !els.frame.classList.contains('hidden')) {
                    els.frame.focus({ preventScroll: true });
                } else if (nextView === 'source' && els.sourceScroll) {
                    els.sourceScroll.focus({ preventScroll: true });
                } else if (nextView === 'blocks' && els.blockList) {
                    els.blockList.focus?.({ preventScroll: true });
                }
            }
        }
        function renderCanvasBlockChips() {
            const els = getCanvasModeElements();
            if (!els || !els.blockList) return;
            const blocks = Array.isArray(canvasPreviewState.blocks) ? canvasPreviewState.blocks : [];
            if (els.blockCount) els.blockCount.textContent = String(blocks.length);
            if (!blocks.length) {
                els.blockList.innerHTML = '<div class="px-2 py-3 text-xs text-gray-500">コードブロックを待機中</div>';
                return;
            }
            const selectedIndex = Number.isInteger(canvasPreviewState.selectedIndex) ? canvasPreviewState.selectedIndex : -1;
            els.blockList.innerHTML = blocks.map((block, index) => {
                const lang = String(block && block.lang ? block.lang : 'text').trim() || 'text';
                const selected = index === selectedIndex;
                const stateLabel = block && block.open ? '生成中' : '表示';
                const codeLines = String(block && block.code ? block.code : '').split(/\r?\n/);
                const firstCodeLine = codeLines.find((line) => line.trim()) || '空のコードブロック';
                const preview = firstCodeLine.trim().replace(/\s+/g, ' ').slice(0, 120);
                const title = `${selected ? '現在表示中' : '切り替え'}: ${lang}`;
                const ariaLabel = `${title}、${preview}`;
                return `<button type="button" class="canvas-block-chip${selected ? ' active' : ''}" data-canvas-block-index="${index}" title="${escapeHtml(title)}" aria-label="${escapeHtml(ariaLabel)}" aria-pressed="${selected ? 'true' : 'false'}"><span class="canvas-block-chip-index">#${index + 1}</span><span class="canvas-block-chip-main"><span class="canvas-block-chip-lang">${escapeHtml(lang)}</span><span class="canvas-block-chip-preview">${escapeHtml(preview)}</span></span><span class="canvas-block-chip-state">${selected ? '表示中' : stateLabel}</span></button>`;
            }).join('');
        }
        function renderCanvasSourceOptions() {
            const els = getCanvasModeElements();
            if (!els || !els.sourceSelect) return;
            const blocks = Array.isArray(canvasPreviewState.blocks) ? canvasPreviewState.blocks : [];
            if (!blocks.length) {
                els.sourceSelect.innerHTML = '<option value="">-</option>';
                els.sourceSelect.disabled = true;
                els.sourceSelect.dataset.canvasOptionsSignature = '';
                return;
            }
            const selectedIndex = Number.isInteger(canvasPreviewState.selectedIndex)
                ? canvasPreviewState.selectedIndex
                : blocks.length - 1;
            els.sourceSelect.disabled = false;
            const optionLabels = blocks.map((block, index) => {
                const lang = String(block && block.lang ? block.lang : 'text').trim() || 'text';
                return `#${index + 1} ${lang}`;
            });
            const signature = JSON.stringify(optionLabels);
            if (els.sourceSelect.dataset.canvasOptionsSignature !== signature) {
                els.sourceSelect.innerHTML = optionLabels.map((label, index) => (
                    `<option value="${index}">${escapeHtml(label)}</option>`
                )).join('');
                els.sourceSelect.dataset.canvasOptionsSignature = signature;
            }
            els.sourceSelect.value = String(selectedIndex);
        }
        function resetCanvasScrollState() {
            canvasPreviewState.sourceScrollTop = 0;
            canvasPreviewState.sourceScrollLeft = 0;
            canvasPreviewState.frameScrollX = 0;
            canvasPreviewState.frameScrollY = 0;
            const els = getCanvasModeElements();
            if (els && els.sourceScroll) {
                els.sourceScroll.scrollTop = 0;
                els.sourceScroll.scrollLeft = 0;
            }
        }
        function instrumentCanvasPreviewDocument(html, token) {
            const initialX = Math.max(0, Number(canvasPreviewState.frameScrollX) || 0);
            const initialY = Math.max(0, Number(canvasPreviewState.frameScrollY) || 0);
            const source = String(html || '');
            const bridgeCode = `(function(){const token=${JSON.stringify(token)};let timer=0;function report(){parent.postMessage({type:'canvas-preview-scroll',token:token,x:window.scrollX||0,y:window.scrollY||0},'*')}addEventListener('scroll',function(){clearTimeout(timer);timer=setTimeout(report,40)},{passive:true});addEventListener('message',function(event){const data=event.data||{};if(data.type==='canvas-preview-restore-scroll'&&data.token===token){requestAnimationFrame(function(){scrollTo(Number(data.x)||0,Number(data.y)||0);report()})}});requestAnimationFrame(function(){scrollTo(${initialX},${initialY});report()})})();`;
            try {
                const doc = new DOMParser().parseFromString(source, 'text/html');
                const script = doc.createElement('script');
                script.setAttribute('data-canvas-scroll-bridge', 'true');
                script.textContent = bridgeCode;
                (doc.body || doc.documentElement).appendChild(script);
                return '<!DOCTYPE html>\n' + doc.documentElement.outerHTML;
            } catch (e) {
                return `${source}<script data-canvas-scroll-bridge>${bridgeCode}<\/script>`;
            }
        }
        window.addEventListener('message', (event) => {
            const data = event && event.data ? event.data : null;
            if (!data || data.type !== 'canvas-preview-scroll') return;
            const els = getCanvasModeElements();
            if (!els || !els.frame || event.source !== els.frame.contentWindow) return;
            if (data.token !== canvasPreviewState.frameRenderToken) return;
            canvasPreviewState.frameScrollX = Math.max(0, Number(data.x) || 0);
            canvasPreviewState.frameScrollY = Math.max(0, Number(data.y) || 0);
        });
        function showCanvasPreviewPanel() {
            const els = getCanvasModeElements();
            if (!els) return;
            canvasPreviewState.panelAnimationToken += 1;
            const token = canvasPreviewState.panelAnimationToken;
            if (canvasPreviewState.panelHideTimer) {
                clearTimeout(canvasPreviewState.panelHideTimer);
                canvasPreviewState.panelHideTimer = null;
            }
            els.panel.classList.remove('hidden', 'canvas-closing');
            if (els.stage) els.stage.classList.add('canvas-enabled');
            requestAnimationFrame(() => {
                if (token !== canvasPreviewState.panelAnimationToken) return;
                els.panel.classList.add('canvas-panel-open');
            });
        }
        function hideCanvasPreviewPanel(animate = true) {
            const els = getCanvasModeElements();
            if (!els) return;
            canvasPreviewState.panelAnimationToken += 1;
            if (canvasPreviewState.panelHideTimer) {
                clearTimeout(canvasPreviewState.panelHideTimer);
                canvasPreviewState.panelHideTimer = null;
            }
            if (!animate) {
                els.panel.classList.add('hidden');
                els.panel.classList.remove('canvas-panel-open', 'canvas-closing');
                if (els.stage) els.stage.classList.remove('canvas-enabled');
                return;
            }
            els.panel.classList.remove('canvas-panel-open');
            els.panel.classList.add('canvas-closing');
            canvasPreviewState.panelHideTimer = window.setTimeout(() => {
                els.panel.classList.add('hidden');
                els.panel.classList.remove('canvas-closing');
                if (els.stage) els.stage.classList.remove('canvas-enabled');
                canvasPreviewState.panelHideTimer = null;
            }, 220);
        }
        function resetCanvasPreviewPanel(message = 'Canvasで表示中') {
            const els = getCanvasModeElements();
            if (!els) return;
            canvasPreviewState.blocks = [];
            canvasPreviewState.rawText = '';
            canvasPreviewState.renderText = '';
            canvasPreviewState.selectedIndex = -1;
            canvasPreviewState.selectedKey = '';
            canvasPreviewState.selectionMode = 'auto';
            canvasPreviewState.mobileView = 'preview';
            canvasPreviewState.lastCanvasData = null;
            resetCanvasScrollState();
            showCanvasPreviewPanel();
            syncCanvasPanelViewUi('preview', { focus: false });
            if (els.title) els.title.textContent = message;
            if (els.status) els.status.textContent = 'コードブロックを待機中';
            if (els.previewLang) els.previewLang.textContent = 'idle';
            if (els.sourceSelect) {
                els.sourceSelect.innerHTML = '<option value="">-</option>';
                els.sourceSelect.disabled = true;
                els.sourceSelect.dataset.canvasOptionsSignature = '';
            }
            if (els.code) els.code.textContent = '';
            if (els.blockCount) els.blockCount.textContent = '0';
            if (els.blockList) els.blockList.innerHTML = '<div class="px-2 py-3 text-xs text-gray-500">コードブロックを待機中</div>';
            if (els.sourceScroll) els.sourceScroll.scrollTop = 0;
            if (els.frame) {
                els.frame.removeAttribute('srcdoc');
                els.frame.classList.add('hidden');
            }
            if (els.empty) els.empty.classList.remove('hidden');
            syncCanvasPreviewButtons();
        }
        function updateCanvasPreviewState(canvasData = null) {
            const data = canvasData || canvasPreviewState.lastCanvasData;
            if (!data) return null;
            canvasPreviewState.lastCanvasData = data;
            canvasPreviewState.blocks = Array.isArray(data.blocks) ? data.blocks.slice() : [];
            canvasPreviewState.rawText = String(data.rawText || '');
            canvasPreviewState.renderText = String(data.renderText || '');
            const blocks = canvasPreviewState.blocks;
            const previousIndex = Number.isInteger(canvasPreviewState.selectedIndex)
                ? canvasPreviewState.selectedIndex
                : -1;
            if (!blocks.length) {
                const fallbackSelection = selectCanvasPreviewBlock([], canvasPreviewState.rawText);
                if (fallbackSelection && fallbackSelection.block) {
                    canvasPreviewState.selectedIndex = -1;
                    canvasPreviewState.selectedKey = fallbackSelection.block.key || '';
                    return fallbackSelection.block;
                }
                canvasPreviewState.selectedIndex = -1;
                canvasPreviewState.selectedKey = '';
                canvasPreviewState.selectionMode = 'auto';
                if (previousIndex !== -1) resetCanvasScrollState();
                return null;
            }
            let nextIndex = blocks.length - 1;
            if (canvasPreviewState.selectionMode === 'manual' && previousIndex >= 0 && previousIndex < blocks.length) {
                nextIndex = previousIndex;
            } else {
                canvasPreviewState.selectionMode = 'auto';
            }
            const nextBlock = blocks[nextIndex] || null;
            canvasPreviewState.selectedIndex = nextBlock ? nextIndex : -1;
            canvasPreviewState.selectedKey = nextBlock && nextBlock.key ? nextBlock.key : '';
            if (previousIndex !== canvasPreviewState.selectedIndex) resetCanvasScrollState();
            return nextBlock;
        }
        function refreshCanvasPreviewPanel() {
            const els = getCanvasModeElements();
            if (!els || !canvasModeEnabled) return;
            showCanvasPreviewPanel();
            syncCanvasPanelViewUi(canvasPreviewState.mobileView || 'preview', { focus: false });
            const blocks = Array.isArray(canvasPreviewState.blocks) ? canvasPreviewState.blocks : [];
            const selected = getCanvasSelectedBlock();
            const block = selected && selected.block ? selected.block : null;
            const index = selected && Number.isInteger(selected.index) ? selected.index : -1;
            const hasBlock = !!block;
            const blockLang = String(block && block.lang ? block.lang : '').trim();
            const code = String(block && block.code !== undefined && block.code !== null ? block.code : '');
            const isHtml = hasBlock ? isCanvasHtmlPreviewCandidate(blockLang, code) : false;
            const statusText = !hasBlock
                ? 'コードブロックを待機中'
                : (isHtml ? 'HTML をリアルタイムでプレビューしています' : (block && block.open ? 'コードブロックを生成中' : 'コードブロックをプレビューしています'));
            const titleText = !hasBlock
                ? 'Canvasで表示中'
                : (isHtml ? `HTML Canvas Preview${blocks.length > 1 && index >= 0 ? ` #${index + 1}/${blocks.length}` : ''}` : `Canvas Preview: ${blockLang || 'text'}${blocks.length > 1 && index >= 0 ? ` #${index + 1}/${blocks.length}` : ''}`);
            if (els.title) els.title.textContent = titleText;
            if (els.status) els.status.textContent = statusText;
            if (els.previewLang) els.previewLang.textContent = hasBlock ? (blockLang || 'text') : 'idle';
            const sourceScrollTop = els.sourceScroll ? els.sourceScroll.scrollTop : canvasPreviewState.sourceScrollTop;
            const sourceScrollLeft = els.sourceScroll ? els.sourceScroll.scrollLeft : canvasPreviewState.sourceScrollLeft;
            if (els.code) els.code.textContent = code;
            if (els.sourceScroll) {
                els.sourceScroll.scrollTop = sourceScrollTop;
                els.sourceScroll.scrollLeft = sourceScrollLeft;
                canvasPreviewState.sourceScrollTop = els.sourceScroll.scrollTop;
                canvasPreviewState.sourceScrollLeft = els.sourceScroll.scrollLeft;
            }
            if (els.blockCount) els.blockCount.textContent = String(blocks.length);
            renderCanvasBlockChips();
            renderCanvasSourceOptions();

            if (hasBlock) {
                canvasPreviewState.frameRenderToken += 1;
                const frameRenderToken = canvasPreviewState.frameRenderToken;
                const previewDoc = instrumentCanvasPreviewDocument(buildCanvasPreviewDocument(block), frameRenderToken);
                if (els.frame) {
                    els.frame.srcdoc = previewDoc;
                    els.frame.classList.remove('hidden');
                    els.frame.addEventListener('load', () => {
                        if (frameRenderToken !== canvasPreviewState.frameRenderToken || !els.frame.contentWindow) return;
                        els.frame.contentWindow.postMessage({
                            type: 'canvas-preview-restore-scroll',
                            token: frameRenderToken,
                            x: canvasPreviewState.frameScrollX,
                            y: canvasPreviewState.frameScrollY
                        }, '*');
                    }, { once: true });
                }
                if (els.empty) els.empty.classList.add('hidden');
            } else {
                if (els.frame) {
                    els.frame.removeAttribute('srcdoc');
                    els.frame.classList.add('hidden');
                }
                if (els.empty) els.empty.classList.remove('hidden');
            }
            syncCanvasPreviewButtons();
        }
        function applyCanvasSelection(index, options = {}) {
            const blocks = Array.isArray(canvasPreviewState.blocks) ? canvasPreviewState.blocks : [];
            if (!blocks.length) return false;
            const nextIndex = Number(index);
            if (!Number.isInteger(nextIndex) || nextIndex < 0 || nextIndex >= blocks.length) return false;
            const changed = canvasPreviewState.selectedIndex !== nextIndex;
            canvasPreviewState.selectedIndex = nextIndex;
            canvasPreviewState.selectedKey = blocks[nextIndex] && blocks[nextIndex].key ? blocks[nextIndex].key : '';
            canvasPreviewState.selectionMode = 'manual';
            if (changed) resetCanvasScrollState();
            syncCanvasPanelViewUi(options.view || 'preview', {
                focus: false,
                animate: options.animateView === true,
                fromView: options.transitionFrom
            });
            renderCanvasBlockChips();
            syncCanvasPreviewButtons();
            refreshCanvasPreviewPanel();
            return true;
        }
        function applyCanvasSelectionByKey(codeKey) {
            const blocks = Array.isArray(canvasPreviewState.blocks) ? canvasPreviewState.blocks : [];
            if (!blocks.length) return false;
            const targetKey = String(codeKey || '');
            if (!targetKey) return false;
            const index = blocks.findIndex((block) => block && block.key === targetKey);
            if (index === -1) return false;
            return applyCanvasSelection(index);
        }
        function decodeCanvasPreviewButtonCode(btn) {
            if (!btn) return null;
            const codeEnc = btn.getAttribute('data-code') || '';
            if (!codeEnc) return null;
            let code = '';
            try {
                code = decodeURIComponent(codeEnc);
            } catch (e) {
                code = codeEnc;
            }
            const lang = String(btn.getAttribute('data-canvas-lang') || btn.getAttribute('data-lang') || '').trim();
            const codeKey = String(btn.getAttribute('data-code-key') || hashString(`${lang || 'TEXT'}\n${code || ''}`));
            return { code, lang, codeKey };
        }
        function collectCanvasBlocksFromButton(btn) {
            const payload = decodeCanvasPreviewButtonCode(btn);
            if (!payload) return null;
            const group = btn && typeof btn.closest === 'function' ? btn.closest('.message-group') : null;
            const sourceBtns = group ? Array.from(group.querySelectorAll('.canvas-preview-btn')) : [];
            if (!sourceBtns.length) {
                const block = normalizeCanvasBlock({ lang: payload.lang, code: payload.code, open: false }, 0);
                return { blocks: [block], selectedIndex: 0, selectedKey: block.key || payload.codeKey || '' };
            }
            const blocks = [];
            let selectedIndex = -1;
            sourceBtns.forEach((sourceBtn, index) => {
                const sourcePayload = decodeCanvasPreviewButtonCode(sourceBtn);
                if (!sourcePayload) return;
                const block = normalizeCanvasBlock({ lang: sourcePayload.lang, code: sourcePayload.code, open: false }, index);
                blocks.push(block);
                if (selectedIndex === -1 && sourcePayload.codeKey === payload.codeKey) {
                    selectedIndex = blocks.length - 1;
                }
            });
            if (!blocks.length) return null;
            if (selectedIndex === -1) selectedIndex = 0;
            const selected = blocks[selectedIndex] || blocks[0] || null;
            return {
                blocks,
                selectedIndex,
                selectedKey: selected && selected.key ? selected.key : payload.codeKey || ''
            };
        }
        function previewCanvasCodeFromButton(btn) {
            window.ActivityLog?.log('html_preview.start');
            if (!btn) return false;
            const payload = collectCanvasBlocksFromButton(btn);
            if (!payload || !payload.blocks || !payload.blocks.length) return false;
            const currentBlocks = Array.isArray(canvasPreviewState.blocks) ? canvasPreviewState.blocks : [];
            const matchedIndex = currentBlocks.findIndex((block) => block && block.key === payload.selectedKey);
            if (matchedIndex !== -1 && currentBlocks.length > 1) {
                return applyCanvasSelection(matchedIndex);
            }
            const selectedBlock = payload.blocks[payload.selectedIndex] || payload.blocks[0] || null;
            canvasPreviewState.blocks = payload.blocks;
            canvasPreviewState.rawText = selectedBlock && selectedBlock.code !== undefined && selectedBlock.code !== null ? String(selectedBlock.code) : '';
            canvasPreviewState.renderText = canvasPreviewState.rawText;
            canvasPreviewState.selectedIndex = Number.isInteger(payload.selectedIndex) ? payload.selectedIndex : 0;
            canvasPreviewState.selectedKey = payload.selectedKey || (selectedBlock && selectedBlock.key) || '';
            canvasPreviewState.selectionMode = 'manual';
            resetCanvasScrollState();
            canvasPreviewState.lastCanvasData = {
                renderText: canvasPreviewState.renderText,
                blocks: payload.blocks,
                primaryBlock: selectedBlock,
                primaryIndex: canvasPreviewState.selectedIndex,
                rawText: canvasPreviewState.rawText
            };
            canvasPreviewState.mobileView = 'preview';
            syncCanvasPanelViewUi('preview', { focus: false });
            refreshCanvasPreviewPanel();
            return true;
        }
        function buildCanvasPreviewDocument(block) {
            const code = String(block && block.code !== undefined && block.code !== null ? block.code : '');
            const lang = String(block && block.lang ? block.lang : '').trim().toLowerCase();
            const isHtml = isCanvasHtmlPreviewCandidate(lang, code);
            if (isHtml) {
                return sanitizeHtmlForPreview(code);
            }
            const title = lang ? `Canvas Preview: ${lang}` : 'Canvas Preview';
            const safeCode = escapeHtml(code || '');
            return `<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>${escapeHtml(title)}</title><style>
                :root { color-scheme: dark; }
                html, body { margin: 0; min-height: 100%; background: #0b1220; color: #e5e7eb; font-family: "Noto Sans JP", system-ui, -apple-system, "Segoe UI", sans-serif; }
                body { box-sizing: border-box; padding: 16px; }
                .frame {
                    background: linear-gradient(180deg, rgba(15, 23, 42, 0.92), rgba(2, 6, 23, 0.94));
                    border: 1px solid rgba(148, 163, 184, 0.18);
                    border-radius: 14px;
                    padding: 14px;
                    box-shadow: 0 20px 48px rgba(0, 0, 0, 0.34);
                }
                .label {
                    font-size: 11px;
                    text-transform: uppercase;
                    letter-spacing: 0.14em;
                    color: #67e8f9;
                    margin-bottom: 10px;
                }
                pre {
                    margin: 0;
                    white-space: pre-wrap;
                    word-break: break-word;
                    overflow-wrap: anywhere;
                    font-family: "JetBrains Mono", "Noto Sans Mono", ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
                    font-size: 13px;
                    line-height: 1.6;
                    color: #e2e8f0;
                }
                .muted { color: #94a3b8; }
            </style></head><body><div class="frame"><div class="label">${escapeHtml(title)}</div><pre>${safeCode || '<span class="muted">Canvasで表示中</span>'}</pre></div></body></html>`;
        }
        function syncCanvasModeUi(enabled = canvasModeEnabled, options = {}) {
            const persist = options.persist !== false;
            canvasModeEnabled = !!enabled;
            if (persist) {
                try {
                    localStorage.setItem(CANVAS_MODE_STORAGE_KEY, canvasModeEnabled ? 'true' : 'false');
                } catch (e) {}
            }
            const checkbox = get('enable-canvas-mode');
            if (checkbox && checkbox.checked !== canvasModeEnabled) checkbox.checked = canvasModeEnabled;
            if (!canvasModeEnabled) {
                hideCanvasPreviewPanel(options.animate !== false);
                if (!activeStreamingBubbleId && currentThreadId) {
                    try {
                        renderThreadTree({ silent: true, keepScroll: true });
                    } catch (e) {}
                }
                return;
            }
            showCanvasPreviewPanel();
            if (isCanvasMobileLayout()) {
                syncCanvasPanelViewUi('preview', { focus: false });
            }
            syncCanvasPanelViewUi(canvasPreviewState.mobileView || 'preview', { focus: false });
            if (options.skipReset) return;
            if (!activeStreamingBubbleId) {
                resetCanvasPreviewPanel();
                if (currentThreadId) {
                    try {
                        renderThreadTree({ silent: true, keepScroll: true });
                    } catch (e) {}
                }
            } else {
                refreshCanvasPreviewPanel();
            }
        }

        function normalizeMarkdownNewlines(text) {
            return String(text || '').replace(/\r\n/g, '\n').replace(/\r/g, '\n');
        }

        function stripExactFencedBlock(text, language, body) {
            let result = normalizeMarkdownNewlines(text);
            const bodyStr = normalizeMarkdownNewlines(body);
            if (!bodyStr && bodyStr !== '') return result;
            const langs = language ? [String(language), ''] : [''];
            for (const fenceChar of ['`', '~']) {
                for (let n = 3; n <= 10; n++) {
                    const fence = fenceChar.repeat(n);
                    for (const lang of langs) {
                        const open = `${fence}${lang}\n`;
                        const close = `\n${fence}`;
                        const candidate = open + bodyStr + close;
                        if (!result.includes(candidate)) continue;
                        result = result.split(candidate).join('');
                    }
                }
            }
            return result;
        }

        function stripVisiblePythonOutputBlock(text, output) {
            let result = normalizeMarkdownNewlines(text);
            const outStr = normalizeMarkdownNewlines(output == null ? '' : String(output));
            const prefixes = ['**Output:**\n', '**Output:** \n', '**Output:**'];
            for (const prefix of prefixes) {
                for (const fenceChar of ['`', '~']) {
                    for (let n = 3; n <= 10; n++) {
                        const fence = fenceChar.repeat(n);
                        const candidates = [
                            `${prefix}${fence}\n${outStr}\n${fence}`,
                            `${prefix}\n${fence}\n${outStr}\n${fence}`,
                            `\n${prefix}${fence}\n${outStr}\n${fence}`,
                            `\n${prefix}\n${fence}\n${outStr}\n${fence}`,
                        ];
                        candidates.forEach((candidate) => {
                            if (result.includes(candidate)) {
                                result = result.split(candidate).join('\n');
                            }
                        });
                    }
                }
            }
            return result;
        }

        function buildChatErrorBubbleHtml(errorText) {
            const msg = String(errorText == null ? '' : errorText).trim() || 'Unknown error';
            return `<div class="text-red-400 text-xs mt-2 border border-red-500 p-2 rounded chat-error-box" role="alert"><i class="fas fa-triangle-exclamation mr-1"></i>Error: ${escapeHtml(msg)}</div>`;
        }

        function buildChatErrorMarkdown(errorText, partialContent = '') {
            let err = String(errorText == null ? '' : errorText).trim() || 'Unknown error';
            // Keep the fence well-formed even if the message contains backticks.
            err = err.replace(/```/g, "'''");
            if (err.length > 50000) err = err.slice(0, 50000) + '…';
            const fence = '```chat_error\n' + err + '\n```';
            const partial = String(partialContent == null ? '' : partialContent).replace(/\s+$/, '');
            return partial ? (partial + '\n\n' + fence) : fence;
        }

        function extractPythonExecutionsFromContent(rawText) {
            const source = normalizeMarkdownNewlines(rawText);
            const executions = [];
            const runOffsets = [];
            if (!source) return { text: '', executions };

            // ```pyexec ... ``` (fences of length >= 3)
            const pyexecRe = /(?:^|\n)(`{3,}|~{3,})pyexec[ \t]*\n([\s\S]*?)\n\1[ \t]*(?=\n|$)/g;
            let cleaned = source.replace(pyexecRe, (match, fence, body, offset) => {
                runOffsets.push(offset);
                const raw = String(body || '').trim();
                try {
                    const obj = JSON.parse(raw);
                    executions.push({
                        code: obj && obj.code != null ? String(obj.code) : '',
                        output: obj && obj.output != null ? String(obj.output) : '',
                        images: obj && Array.isArray(obj.images) ? obj.images.filter(u => typeof u === 'string') : []
                    });
                } catch (e) {
                    executions.push({ code: raw, output: '' });
                }
                return '\n';
            });

            const agenticRe = /!\[Agentic View\]\(([^)]+)\)/g;
            if (!executions.length && /!\[Agentic View\]\(/.test(source)) {
                cleaned = source.replace(/(?:^|\n)```python\n([\s\S]*?)\n```(?:\n+\*\*Output:\*\*\n```\n([\s\S]*?)\n```)?/g,
                    (match, code, output, offset) => {
                        runOffsets.push(offset);
                        executions.push({ code, output: output || '', images: [] }); return '\n';
                    });
            }
            if (executions.length) {
                const known = new Set(executions.flatMap(ex => ex.images || []));
                for (const match of source.matchAll(agenticRe)) {
                    if (known.has(match[1])) continue;
                    let index = runOffsets.length - 1;
                    while (index > 0 && runOffsets[index] > match.index) index--;
                    const run = executions[index];
                    run.images = Array.from(new Set([...(run.images || []), match[1]]));
                }
                cleaned = cleaned.replace(agenticRe, '');
            }

            executions.forEach((ex) => {
                if (ex.code) {
                    cleaned = stripExactFencedBlock(cleaned, 'python', ex.code);
                    cleaned = stripExactFencedBlock(cleaned, 'py', ex.code);
                }
                cleaned = stripVisiblePythonOutputBlock(cleaned, ex.output);
            });

            const generatedImages = new Set(executions.flatMap(ex => ex.images || []));
            cleaned = cleaned.replace(/(`{3,}[\s\S]*?`{3,}|~{3,}[\s\S]*?~{3,}|`[^`\n]*`)|(?<!!)\[([^\]\n]*)\]\(([^)\s]+)\)/g,
                (match, code, label, url) => !code && generatedImages.has(url) && /\.(?:png|jpe?g|webp|gif|bmp)(?:[?#].*)?$/i.test(url)
                    ? `![${label}](${url})` : match);

            const lastRun = executions.slice().reverse().find(ex => ex.images && ex.images.length);
            const visibleText = cleaned.replace(/`{3,}[\s\S]*?`{3,}|~{3,}[\s\S]*?~{3,}|`[^`\n]*`/g, '');
            const shownImages = new Set(Array.from(visibleText.matchAll(/!\[[^\]\n]*\]\(([^)\s]+)\)/g), m => m[1]));
            const finalImages = (lastRun ? lastRun.images : []).filter(u => /^\/files\/[A-Za-z0-9_./%-]+$/.test(u) && !shownImages.has(u));
            if (finalImages.length) {
                cleaned += '\n\n**Pythonの最後の実行画像**\n\n' + finalImages.map(u => `![Pythonの最後の実行画像](${u})`).join('\n\n');
            }

            cleaned = cleaned
                .replace(/[ \t]+\n/g, '\n')
                .replace(/\n{3,}/g, '\n\n')
                .replace(/^\n+/, '')
                .replace(/\n+$/, '');

            return { text: cleaned, executions };
        }

        function extractMcpExecutionNotesFromContent(rawText) {
            const source = normalizeMarkdownNewlines(rawText);
            const notes = [];
            if (!source) return { text: '', notes };

            const keptLines = [];
            source.split('\n').forEach((line) => {
                // MCP execution notices are emitted as one Markdown blockquote
                // line. Keep them out of the streamed prose so they can be
                // rendered together after the answer text.
                if (/^>\s*(?:🔧|🚫)\s*\*\*MCPツール実行(?:[:：]|は|（)/.test(line)) {
                    notes.push(line.trim());
                } else {
                    keptLines.push(line);
                }
            });

            const cleaned = keptLines.join('\n')
                .replace(/[ \t]+\n/g, '\n')
                .replace(/\n{3,}/g, '\n\n')
                .replace(/^\n+/, '')
                .replace(/\n+$/, '');
            return { text: cleaned, notes };
        }

        function appendMcpExecutionNotes(text, notes) {
            const body = String(text || '').trim();
            const items = Array.isArray(notes) ? notes.filter(Boolean) : [];
            if (!items.length) return body;
            return body ? `${body}\n\n${items.join('\n')}` : items.join('\n');
        }

        function buildPythonExecDetailBoxHtml(ex, index, total) {
            const codeRaw = ex && ex.code != null ? String(ex.code) : '';
            const outputRaw = ex && ex.output != null ? String(ex.output) : '';
            let codeHtml = '';
            try {
                if (window.hljs && typeof window.hljs.highlight === 'function') {
                    codeHtml = window.hljs.highlight(codeRaw, { language: 'python' }).value;
                } else {
                    codeHtml = escapeHtml(codeRaw);
                }
            } catch (e) {
                codeHtml = escapeHtml(codeRaw);
            }
            const imageMarkdown = (ex.images || []).filter(u => /^\/files\/[A-Za-z0-9_./%-]+$/.test(u))
                .map(u => `![Python output](${u})`).join('\n');
            const imageHtml = imageMarkdown ? `<div class="python-section"><div class="python-label">Images</div>${sanitizeMarkdownHtml(marked.parse(imageMarkdown))}</div>` : '';
            const outputHtml = escapeHtml(outputRaw);
            const encCode = encodeURIComponent(codeRaw).replace(/'/g, '%27');
            const encOut = encodeURIComponent(outputRaw).replace(/'/g, '%27');
            const codeKey = hashString(`pyexec-detail\n${codeRaw}\n${outputRaw}\n${index}`);
            const label = total > 1 ? `Python Execution ${index + 1}/${total}` : 'Python Execution';
            const downloadBtn = `<button class="download-btn" data-code="${encCode}" data-lang="python" title="コードをダウンロード" aria-label="コードをダウンロード"><i class="fas fa-download"></i></button>`;
            const codingBtn = `<button class="coding-target-btn" data-code="${encCode}" data-code-key="${codeKey}" data-coding-lang="python" aria-pressed="false" title="Coding Modeの編集対象に指定" aria-label="編集対象に指定"><i class="fas fa-quote-right"></i></button>`;
            return `<div class="code-wrapper python-box" data-collapsed="false" data-code-key="${codeKey}"><div class="code-header"><span class="code-lang"><i class="fas fa-terminal"></i> ${escapeHtml(label)}</span><div class="code-actions">${codingBtn}${downloadBtn}<button class="copy-btn" data-copy="code" data-code="${encCode}" title="コードをコピー" aria-label="コードをコピー"><i class="fas fa-copy"></i></button><button class="copy-btn" data-copy="output" data-code="${encOut}" title="出力をコピー" aria-label="出力をコピー"><i class="fas fa-align-left"></i></button></div></div><div class="code-body"><div class="python-section"><div class="python-label">Code</div><pre><code class="hljs language-python python-code">${codeHtml}</code></pre></div><div class="python-section"><div class="python-label">Output</div><pre><code class="hljs language-plaintext python-output">${outputHtml}</code></pre></div>${imageHtml}</div></div>`;
        }

        function showPythonExecDetailModal(messageId = null) {
            if (location.pathname !== '/python-execution') {
                const state = { modal: 'python-execution' };
                if (messageId !== null) state.messageId = messageId;
                history.pushState(state, '', '/python-execution');
            }
            showModal('python-exec-modal');
        }

        function openPythonExecDetail(id) {
            const meta = messageMeta[id];
            const modal = get('python-exec-modal');
            const body = get('python-exec-modal-body');
            const title = get('python-exec-modal-title');
            if (!modal || !body) return;
            const executions = (meta && Array.isArray(meta.python_executions)) ? meta.python_executions : [];
            if (!executions.length) {
                showToast('Python実行結果がありません', 'info', false);
                return;
            }
            if (title) {
                const countLabel = executions.length > 1 ? `（${executions.length}件）` : '';
                title.textContent = `Python 実行結果${countLabel}`;
            }
            body.innerHTML = executions.map((ex, i) => buildPythonExecDetailBoxHtml(ex, i, executions.length)).join('');
            body.querySelectorAll('img').forEach(img => {
                img.addEventListener('click', () => openImageViewer(img.dataset.viewerSrc || img.src, '#python-exec-modal-body img'));
            });
            if (codingModeEnabled) {
                syncCodingTargetButtons(body);
                syncCodingModeUi(true, { persist: false });
            }
            showPythonExecDetailModal(id);
        }
        window.openPythonExecDetail = openPythonExecDetail;

        function closePythonExecDetail(skipHistory = false) {
            const modal = get('python-exec-modal');
            if (!modal) return;
            hideModal('python-exec-modal');
            if (!skipHistory && location.pathname === '/python-execution') {
                history.back();
            }
        }
        window.closePythonExecDetail = closePythonExecDetail;

        function buildAiMarkdownHtml(text, opts = {}) {
            const mcpExtract = extractMcpExecutionNotesFromContent(text);
            const displayText = appendMcpExecutionNotes(mcpExtract.text, mcpExtract.notes);
            // 数学ステップ: 見出しがあれば1ステップずつ表示する（なければ通常のMarkdown）。
            if (typeof window.buildMathStepsBubbleHtml === 'function') {
                const stepsHtml = window.buildMathStepsBubbleHtml(displayText, opts);
                if (stepsHtml) {
                    if (!lowBandwidthMode && maybeNeedsMathJax(displayText)) ensureMathJaxLoaded().catch(() => {});
                    return stepsHtml;
                }
            }
            const canvasData = canvasModeEnabled ? parseCanvasMarkdown(displayText) : { renderText: displayText, blocks: [], primaryBlock: null, rawText: displayText };
            if (canvasModeEnabled) {
                updateCanvasPreviewState(canvasData);
                refreshCanvasPreviewPanel();
            }
            const wrap = document.createElement('div');
            wrap.className = 'prose prose-invert text-sm break-words';
            wrap.innerHTML = sanitizeMarkdownHtml(canvasData.renderText);
            wrapRenderedSvgBoxes(wrap);
            if (!lowBandwidthMode) {
                if (maybeNeedsHighlight(canvasData.renderText, wrap)) ensureHighlightLoaded().catch(() => {});
                if (maybeNeedsMathJax(canvasData.renderText)) ensureMathJaxLoaded().catch(() => {});
            }
            return wrap.outerHTML;
        }
        function renderAiMarkdownInto(container, text, opts = {}) {
            if (!container) return;
            const mcpExtract = extractMcpExecutionNotesFromContent(text);
            const displayText = appendMcpExecutionNotes(mcpExtract.text, mcpExtract.notes);
            if (typeof window.renderMathStepsInto === 'function' && window.renderMathStepsInto(container, displayText, opts)) {
                if (!lowBandwidthMode) {
                    if (maybeNeedsMathJax(displayText)) ensureMathJaxLoaded().catch(() => {});
                }
                return;
            }
            const canvasData = canvasModeEnabled ? parseCanvasMarkdown(displayText) : { renderText: displayText, blocks: [], primaryBlock: null, rawText: displayText };
            if (canvasModeEnabled) {
                updateCanvasPreviewState(canvasData);
                refreshCanvasPreviewPanel();
            }
            if (opts.incrementalMath) {
                const template = document.createElement('template');
                template.innerHTML = sanitizeMarkdownHtml(canvasData.renderText, { streamMathSegments: true });
                const preserved = new Map();
                container.querySelectorAll('.stream-math-segment[data-stream-math-key]').forEach((el) => {
                    const key = el.getAttribute('data-stream-math-key');
                    if (key) preserved.set(key, el);
                });
                const newMathSegments = [];
                template.content.querySelectorAll('.stream-math-segment[data-stream-math-key]').forEach((fresh) => {
                    const old = preserved.get(fresh.getAttribute('data-stream-math-key'));
                    if (old) fresh.replaceWith(old);
                    else newMathSegments.push(fresh);
                });
                restoreSvgCodeRenders(template.content, collectSvgCodeRenders(container));
                container.replaceChildren(template.content);
                wrapRenderedSvgBoxes(container);
                queueHighlight(container, canvasData.renderText);
                queueIncrementalMathTypeset(newMathSegments);
                return;
            }
            const preservedSvgRenders = collectSvgCodeRenders(container);
            container.innerHTML = sanitizeMarkdownHtml(canvasData.renderText);
            restoreSvgCodeRenders(container, preservedSvgRenders);
            wrapRenderedSvgBoxes(container);
            queueMessageDecorations(container, canvasData.renderText);
        }
        // Drawn SVG code blocks keep their <img> across stream re-renders so the drawing does not flicker.
        function collectSvgCodeRenders(root) {
            const map = new Map();
            if (!root || typeof root.querySelectorAll !== 'function') return map;
            root.querySelectorAll('.svg-code-render[data-svg-key]').forEach((el) => {
                const key = el.getAttribute('data-svg-key');
                if (key && !map.has(key)) map.set(key, el);
            });
            return map;
        }
        function restoreSvgCodeRenders(root, preserved) {
            if (!root || !preserved || !preserved.size || typeof root.querySelectorAll !== 'function') return;
            root.querySelectorAll('.svg-code-render[data-svg-key]').forEach((fresh) => {
                const key = fresh.getAttribute('data-svg-key');
                const old = key ? preserved.get(key) : null;
                if (!old) return;
                preserved.delete(key);
                fresh.replaceWith(old);
            });
        }
        function wrapRenderedSvgBoxes(root) {
            if (!root || typeof root.querySelectorAll !== 'function') return;
            root.querySelectorAll('svg').forEach((svg) => {
                if (!svg || !svg.parentNode) return;
                if (svg.closest('.svg-render-box')) return;
                if (svg.closest('pre, code, .code-wrapper, .thought-container')) return;
                const frame = document.createElement('span');
                frame.className = 'svg-render-box';
                svg.parentNode.insertBefore(frame, svg);
                frame.appendChild(svg);
            });
        }
        const SEARCH_COMPLETE_HTML = '<div class="search-box visible mb-2"><i class="fas fa-check-circle text-green-400"></i> Search complete</div>';
