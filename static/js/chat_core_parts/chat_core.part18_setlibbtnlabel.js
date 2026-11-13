
        const setLibBtnLabel = (btn, label) => {
            if (!btn) return;
            const span = btn.querySelector('span');
            if (span) span.textContent = label;
            else btn.textContent = label;
        };
        window.updateLibSelectionUi = function () {
            if (!lib.selected) lib.selected = new Set();
            const count = lib.selected.size;
            const delBtn = get('lib-del-btn');
            const downloadBtn = get('lib-download-btn');
            const attachBtn = get('lib-attach-btn');
            const renameBtn = get('lib-rename-btn');
            const usageBtn = get('lib-usage-btn');
            if (delBtn) {
                delBtn.disabled = count === 0;
                setLibBtnLabel(delBtn, count ? `削除 (${count})` : "削除");
            }
            if (downloadBtn) {
                downloadBtn.disabled = count === 0;
                setLibBtnLabel(downloadBtn, count ? `ダウンロード (${count})` : "ダウンロード");
            }
            if (attachBtn) {
                attachBtn.disabled = count === 0;
                setLibBtnLabel(attachBtn, count ? `添付 (${count})` : "添付");
            }
            if (renameBtn) {
                renameBtn.disabled = count !== 1;
                setLibBtnLabel(renameBtn, "名前変更");
            }
            if (usageBtn) {
                usageBtn.disabled = count !== 1;
                setLibBtnLabel(usageBtn, "使用チャット");
            }
            if (lib.modal) {
                const isMobile = window.matchMedia('(max-width: 768px)').matches;
                lib.modal.classList.toggle('lib-selecting', isMobile && count > 0);
            }
        };
        function setLibAttachMode(flag) {
            lib.attachMode = !!flag;
        }
        const openLibModal = (attachMode = false) => {
            setLibAttachMode(attachMode);
            showModal('lib-modal');
            loadLibraryFiles();
            if (location.pathname !== '/library') {
                history.pushState({ modal: 'library' }, '', '/library');
            }
        };
        window.closeLibModal = (skipHistory = false) => {
            hideModal('lib-modal');
            if (!skipHistory && location.pathname === '/library') {
                history.back();
            }
        };
        get('lib-btn').onclick = () => openLibModal(false);
        get('lib-del-btn').onclick = deleteSelectedFiles;
        if (get('lib-download-btn')) get('lib-download-btn').onclick = () => downloadSelectedLibraryFiles();
        if (get('lib-attach-btn')) get('lib-attach-btn').onclick = () => attachSelectedLibraryFiles();
        if (get('lib-rename-btn')) get('lib-rename-btn').onclick = () => renameSelectedLibraryFile();
        if (get('lib-usage-btn')) get('lib-usage-btn').onclick = () => showSelectedFileUsage();
        if (get('upload-lib-btn')) get('upload-lib-btn').onclick = () => openLibModal(true);
        if (get('lib-search')) {
            let librarySearchTimer = null;
            get('lib-search').oninput = () => {
                lib.searchQuery = (get('lib-search').value || '').trim();
                if (librarySearchTimer) clearTimeout(librarySearchTimer);
                librarySearchTimer = setTimeout(() => loadLibraryFiles(), 250);
            };
        }
        if (get('lib-sort')) {
            const storedSort = localStorage.getItem(LIB_SORT_KEY) || 'newest';
            get('lib-sort').value = storedSort;
            get('lib-sort').onchange = () => {
                const v = get('lib-sort').value || 'newest';
                localStorage.setItem(LIB_SORT_KEY, v);
                loadLibraryFiles();
            };
        }
        if (get('lib-favorite-filter-btn')) {
            lib.favoritesOnly = localStorage.getItem(LIB_FAVORITES_ONLY_KEY) === 'true';
            get('lib-favorite-filter-btn').onclick = () => {
                lib.favoritesOnly = !lib.favoritesOnly;
                localStorage.setItem(LIB_FAVORITES_ONLY_KEY, String(lib.favoritesOnly));
                loadLibraryFiles();
            };
        }
        if (get('lib-load-more-btn')) get('lib-load-more-btn').onclick = () => loadLibraryFiles(true);

        const openSkillModal = () => {
            editingSkillUuid = null;
            get('skill-modal-title').textContent = 'Create New Skill';
            get('save-skill-btn').innerText = "Create Skill";
            showModal('skill-modal');
            get('skill-name').value=''; get('skill-desc').value=''; get('skill-inst').value='';
            if (location.pathname !== '/skill') {
                history.pushState({ modal: 'skill' }, '', '/skill');
            }
        };
        window.closeSkillModal = (skipHistory = false) => {
            hideModal('skill-modal');
            if (!skipHistory && location.pathname === '/skill') {
                history.back();
            }
        };
        get('add-skill-btn').onclick = () => openSkillModal();
            get('save-skill-btn').onclick = async () => {
                const name = get('skill-name').value;
                const desc = get('skill-desc').value;
                const inst = get('skill-inst').value;

                if(name && inst) {
                    const method = editingSkillUuid ? 'PUT' : 'POST';
                    const url = editingSkillUuid ? `/api/skills/${editingSkillUuid}` : CHAT_CONFIG.urls.handleSkills;
                    await apiFetch(url, {
                        method: method,
                        headers:{'Content-Type':'application/json'},
                        body:JSON.stringify({name, description:desc, instruction:inst})
                    });
                    window.closeSkillModal();
                    loadSkills();
                    if (editingSkillUuid && activeSkill && activeSkill.uuid === editingSkillUuid) {
                       activeSkill.name = name;
                       activeSkill.instruction = inst;
                       applyActiveSkill(activeSkill);
                    }
                } else alert("Name and Instruction are required.");
            };
            document.addEventListener('click', function(e) {
                if (e.target.closest('.edit-btn')) {
                    const btn = e.target.closest('.edit-btn');
                    const id = btn.getAttribute('data-id');
                    beginEditMessage(id);
                }
                if (e.target.closest('.code-toggle')) {
                    const btn = e.target.closest('.code-toggle');
                    const wrapper = btn.closest('.code-wrapper');
                    if (!wrapper) return;
                    const isCollapsed = wrapper.classList.toggle('collapsed');
                    wrapper.setAttribute('data-collapsed', isCollapsed ? 'true' : 'false');
                    btn.setAttribute('aria-expanded', isCollapsed ? 'false' : 'true');
                    btn.innerHTML = isCollapsed
                        ? '<i class="fas fa-chevron-down"></i>'
                        : '<i class="fas fa-chevron-up"></i>';
                    btn.title = isCollapsed ? '展開' : '折りたたむ';
                    btn.setAttribute('aria-label', isCollapsed ? '展開' : '折りたたむ');
                }
                if (e.target.closest('.download-btn')) {
                    const btn = e.target.closest('.download-btn');
                    const codeEnc = btn.getAttribute('data-code');
                    const lang = (btn.getAttribute('data-lang') || 'txt').toLowerCase();
                    if (codeEnc) {
                        try {
                            const code = decodeURIComponent(codeEnc);
                            const blob = new Blob([code], { type: 'text/plain' });
                            const url = URL.createObjectURL(blob);
                            const a = document.createElement('a');
                            a.href = url;
                            const map = {
                                python: 'py', javascript: 'js', typescript: 'ts', markdown: 'md',
                                html: 'html', css: 'css', json: 'json', xml: 'xml', sql: 'sql',
                                bash: 'sh', sh: 'sh', shell: 'sh', zsh: 'sh',
                                c: 'c', cpp: 'cpp', csharp: 'cs', cs: 'cs',
                                java: 'java', kotlin: 'kt', swift: 'swift',
                                go: 'go', rust: 'rs', ruby: 'rb', php: 'php',
                                perl: 'pl', lua: 'lua', r: 'r', matlab: 'm',
                                yaml: 'yaml', yml: 'yaml', toml: 'toml', ini: 'ini',
                                plaintext: 'txt', text: 'txt'
                            };
                            let ext = map[lang] || lang;
                            if (lang.length > 8 || /[^a-z0-9]/.test(lang)) ext = 'txt';
                            let filename = `code.${ext}`;
                            if (lang === 'dockerfile') filename = 'Dockerfile';
                            if (lang === 'makefile') filename = 'Makefile';
                            a.download = filename;
                            document.body.appendChild(a);
                            a.click();
                            document.body.removeChild(a);
                            URL.revokeObjectURL(url);
                        } catch (err) {
                            console.error('Download failed', err);
                        }
                    }
                }
                if (e.target.closest('.coding-target-btn')) {
                    selectCodingTargetFromButton(e.target.closest('.coding-target-btn'));
                }
                if (e.target.closest('.copy-btn')) {
                    const btn = e.target.closest('.copy-btn');
                    const code = btn.getAttribute('data-code');
                    if (code) window.copyCode(btn, code);
                }
                if (e.target.closest('.html-preview-btn')) {
                    const btn = e.target.closest('.html-preview-btn');
                    const code = btn.getAttribute('data-code');
                    if (code) openHtmlCodePreview(code);
                }
                if (e.target.closest('.canvas-preview-btn')) {
                    const btn = e.target.closest('.canvas-preview-btn');
                    previewCanvasCodeFromButton(btn);
                }
            });
            document.querySelectorAll('.modal-overlay').forEach((overlay) => {
                overlay.addEventListener('click', (e) => {
                    if (e.target === overlay) closeModalById(overlay.id);
                });
            });

            // Initial Load Logic for Permalink
            if(currentThreadId) { loadMessages(currentThreadId); }
            else { schedulePromptTokenEstimate(true); }
        });

        function updateFilePreview() {
            const preview = get('file-preview');
            const nameEl = get('file-name');
            const progressContainer = get('upload-total-progress');
            const progressBar = get('upload-total-progress-bar');
            const thumbContainer = get('file-preview-thumbs');

            const modalStatusText = get('upload-modal-status-text');
            const modalProgressContainer = get('upload-modal-total-progress');
            const modalProgressBar = get('upload-modal-total-progress-bar');

            if (!preview || !nameEl) return;

            // Sync Thumbnails from the upload list (modal)
            if (thumbContainer) {
                const uploadRows = document.querySelectorAll('#upload-list .upload-row');

                // Rebuild thumbnail list
                thumbContainer.innerHTML = '';
                uploadRows.forEach((row, index) => {
                    const localUrl = row.getAttribute('data-local-url');
                    const filename = row.getAttribute('data-filename');
                    // Check if it's meant to be an image
                    const isImage = row.querySelector('img.upload-preview') !== null;

                    let el;
                    if (isImage) {
                        let src = localUrl;
                        if (!src && filename) {
                            const displayPath = filename.replace(/^\d+\//, '');
                            src = buildAttachmentPreviewUrl(displayPath);
                        }

                        if (src) {
                            el = document.createElement('img');
                            el.src = src;
                            el.className = 'thumb-item shadow-sm';
                            el.dataset.viewerSrc = src;
                            el.dataset.viewerFilename = filename || src.split('/').pop();
                            el.uploadRow = row;
                            el.onclick = function(e) {
                                e.preventDefault();
                                openImageViewer(this.dataset.viewerSrc, '.thumb-item');
                            };
                            el.onerror = function() {
                                // Fallback if image fails to load (e.g. revoked URL)
                                this.parentElement.replaceChild(createFileThumb('ERR'), this);
                            };
                        }
                    }

                    if (!el) {
                        el = createFileThumb('FILE');
                    }

                    el.style.animationDelay = `${index * 32}ms`;
                    thumbContainer.appendChild(el);
                });

                if (uploadRows.length > 0) thumbContainer.classList.remove('hidden');
                else thumbContainer.classList.add('hidden');
            }

            function createFileThumb(text) {
                const el = document.createElement('div');
                el.className = 'thumb-item bg-gray-800 flex items-center justify-center text-gray-500 text-[9px] shadow-sm font-bold';
                el.innerText = text;
                return el;
            }

            const pendingUrls = collectImageUrlsForSend();
            const total = uploadProgressState.total;
            const completed = uploadProgressState.completed;
            const active = uploadProgressState.active;

            if (total === 0) {
                preview.classList.add('hidden');
                if (progressContainer) progressContainer.classList.add('hidden');
                if (modalProgressContainer) modalProgressContainer.classList.add('hidden');
                if (thumbContainer) thumbContainer.classList.add('hidden');
            }

            const sendBtn = get('send-btn');
            const micBtn = get('mic-btn');
            const maskBtn = get('mask-btn');
            const isAiThinking = isStopMode;
            if (active > 0) {
                if (sendBtn) sendBtn.disabled = true;
                if (micBtn) micBtn.disabled = true;
                if (maskBtn) maskBtn.disabled = true;
            } else if (!isAiThinking) {
                if (sendBtn) sendBtn.disabled = false;
                if (micBtn) micBtn.disabled = false;
                if (maskBtn) maskBtn.disabled = false;
            }

            if (active > 0) {
                const statusStr = `Preparing... (${completed}/${total})`;
                preview.classList.remove('hidden');
                nameEl.innerText = statusStr;
                if (modalStatusText) modalStatusText.innerText = `(${completed}/${total})`;

                let sumPct = completed * 100;
                let countActive = 0;
                for (let id in uploadProgressState.perFilePct) {
                    sumPct += uploadProgressState.perFilePct[id];
                    countActive++;
                }
                const pct = total > 0 ? (sumPct / (total * 100)) * 100 : 0;
                const pctStr = `${Math.min(100, pct)}%`;

                if (progressContainer && progressBar) {
                    progressContainer.classList.remove('hidden');
                    progressBar.style.width = pctStr;
                }
                if (modalProgressContainer && modalProgressBar) {
                    modalProgressContainer.classList.remove('hidden');
                    modalProgressBar.style.width = pctStr;
                }
            } else {
                if (modalStatusText) modalStatusText.innerText = '';
                if (modalProgressContainer) modalProgressContainer.classList.add('hidden');

                if (pendingUrls.length > 0) {
                    preview.classList.remove('hidden');
                    nameEl.innerText = `${pendingUrls.length} files ready`;
                    if (progressContainer) progressContainer.classList.add('hidden');
                } else {
                    preview.classList.add('hidden');
                    nameEl.innerText = '';
                    if (progressContainer) progressContainer.classList.add('hidden');
                }
            }
            schedulePromptTokenEstimate();
        }
        function updateMaskPreview() {
            const preview = get('mask-preview');
            const nameEl = get('mask-name');
            if (!preview || !nameEl) return;
            if (currentMaskImage) {
                preview.classList.remove('hidden');
                nameEl.innerText = `Mask: ${currentMaskImage.split('/').pop()}`;
            } else {
                preview.classList.add('hidden');
                nameEl.innerText = '';
            }
        }
        const markerToolHints = {
            draw: 'マーカー（色・透明度変更可） / 二本指で拡大',
            mosaic: 'ドラッグで範囲モザイク（複数追加可） / 二本指で拡大',
            crop: '外側をドラッグして切り取り / 二本指で拡大'
        };
        function normalizeMarkerHexColor(color) {
            const v = String(color || '').trim().toLowerCase();
            if (/^#[0-9a-f]{6}$/.test(v)) return v;
            if (/^#[0-9a-f]{3}$/.test(v)) {
                const r = v[1], g = v[2], b = v[3];
                return `#${r}${r}${g}${g}${b}${b}`;
            }
            return '#facc15';
        }
        function markerHexToRgb(hex) {
            const normalized = normalizeMarkerHexColor(hex);
            return {
                r: parseInt(normalized.slice(1, 3), 16),
                g: parseInt(normalized.slice(3, 5), 16),
                b: parseInt(normalized.slice(5, 7), 16)
            };
        }
        function clampMarkerOpacityPct(value, fallbackPct = 60) {
            const parsed = Number(value);
            const pct = Number.isFinite(parsed) ? parsed : fallbackPct;
            return Math.max(MARKER_OPACITY_MIN_PCT, Math.min(MARKER_OPACITY_MAX_PCT, pct));
        }
        function formatMarkerOpacityPct(pctValue) {
            const rounded = Math.round(clampMarkerOpacityPct(pctValue) * 10) / 10;
            return Number.isInteger(rounded) ? String(rounded) : String(rounded).replace(/\.0$/, '');
        }
        function getMarkerStrokeStyle() {
            const rgb = markerHexToRgb(markerState.colorHex);
            const alpha = Math.max(MARKER_OPACITY_MIN_ALPHA, Math.min(1, Number(markerState.opacity) || 0.6));
            return `rgba(${rgb.r},${rgb.g},${rgb.b},${alpha})`;
        }
        function syncMarkerColorControls() {
            const colorHex = normalizeMarkerHexColor(markerState.colorHex);
            markerState.colorHex = colorHex;
            const alpha = Math.max(MARKER_OPACITY_MIN_ALPHA, Math.min(1, Number(markerState.opacity) || 0.6));
            markerState.opacity = alpha;
            const opacityPct = alpha * 100;
            const opacityPctText = formatMarkerOpacityPct(opacityPct);
            const colorPicker = get('marker-color-picker');
            if (colorPicker && colorPicker.value !== colorHex) colorPicker.value = colorHex;
            const opacityInput = get('marker-opacity');
            if (opacityInput && opacityInput.value !== opacityPctText) opacityInput.value = opacityPctText;
            const opacityNumber = get('marker-opacity-number');
            if (opacityNumber && opacityNumber.value !== opacityPctText) opacityNumber.value = opacityPctText;
            const opacityText = get('marker-opacity-value');
            if (opacityText) opacityText.textContent = `${opacityPctText}%`;
            const chips = document.querySelectorAll('#marker-toolbar .marker-color-chip[data-marker-color]');
            chips.forEach((chip) => {
                const v = normalizeMarkerHexColor(chip.getAttribute('data-marker-color'));
                chip.classList.toggle('active', v === colorHex);
            });
        }
        function setMarkerColor(colorHex) {
            markerState.colorHex = normalizeMarkerHexColor(colorHex);
            syncMarkerColorControls();
        }
        function setMarkerOpacity(pctValue) {
            const pct = clampMarkerOpacityPct(pctValue, 60);
            markerState.opacity = pct / 100;
            syncMarkerColorControls();
        }
        function setMarkerMode(mode) {
            markerState.mode = mode;
            if (mode !== 'mosaic') markerState.mosaicPreviewRect = null;
            const btnDraw = get('marker-tool-draw');
            const btnMosaic = get('marker-tool-mosaic');
            const btnCrop = get('marker-tool-crop');
            if (btnDraw) btnDraw.classList.toggle('active', mode === 'draw');
            if (btnMosaic) btnMosaic.classList.toggle('active', mode === 'mosaic');
            if (btnCrop) btnCrop.classList.toggle('active', mode === 'crop');
            const hint = get('marker-tool-hint');
            if (hint) hint.textContent = markerState.maskMode
                ? (mode === 'auto' ? '画像をタップすると近似色のつながった範囲を塗ります' : '白く塗った部分が削除対象になります / 二本指で拡大')
                : (markerToolHints[mode] || '');
            const cropReset = get('marker-crop-reset');
            if (cropReset) cropReset.classList.toggle('hidden', mode !== 'crop');
            document.querySelectorAll('.marker-edit-only').forEach((el) => el.classList.toggle('hidden', markerState.maskMode));
            const mosaic = get('marker-tool-mosaic');
            const crop = get('marker-tool-crop');
            if (mosaic) mosaic.classList.toggle('hidden', markerState.maskMode);
            if (crop) crop.classList.toggle('hidden', markerState.maskMode);
            const draw = get('marker-tool-draw');
            const auto = get('marker-tool-auto');
            if (draw) draw.textContent = markerState.maskMode ? 'ブラシ' : 'マーカー';
            if (auto) {
                auto.classList.toggle('hidden', !markerState.maskMode);
                auto.classList.toggle('active', mode === 'auto');
            }
            if (draw) draw.classList.toggle('active', mode === 'draw');
            const brushControl = get('marker-brush-control');
            if (brushControl) brushControl.classList.toggle('hidden', markerState.maskMode && mode === 'auto');
            const toleranceControl = get('marker-auto-tolerance-control');
            if (toleranceControl) toleranceControl.classList.toggle('hidden', !markerState.maskMode || mode !== 'auto');
            const title = get('marker-modal-title');
            const description = get('marker-modal-description');
            const save = get('marker-save');
            if (title) title.textContent = markerState.maskMode ? 'マスク作成' : '画像編集';
            if (description) description.textContent = markerState.maskMode
                ? '白く塗った領域が削除対象です。ブラシまたは自動塗りを使えます。マスクは対象画像と同じサイズで作成されます。'
                : 'マーカー・モザイク・トリミングが可能です。送信時に編集済み画像の確認を促します。';
            if (save) save.textContent = markerState.maskMode ? 'マスクを作成' : '保存して反映';
            const drawCanvas = get('marker-canvas');
            if (drawCanvas) drawCanvas.style.pointerEvents = (mode === 'crop') ? 'none' : 'auto';
            const cropCanvas = get('marker-crop-canvas');
            if (cropCanvas) cropCanvas.style.pointerEvents = (mode === 'crop') ? 'auto' : 'none';
            if (mode === 'crop' && (!markerState.cropRect || markerState.cropRect.w <= 1 || markerState.cropRect.h <= 1)) {
                resetCropRectToFull();
            }
            renderCropOverlay();
        }
        function clearCropRect() {
            resetCropRectToFull();
            renderCropOverlay();
        }
        function resetCropRectToFull() {
            const cropCanvas = get('marker-crop-canvas');
            if (!cropCanvas) return;
            const w = Math.max(1, cropCanvas.width || 0);
            const h = Math.max(1, cropCanvas.height || 0);
            if (w <= 1 || h <= 1) return;
            markerState.cropRect = { x: 0, y: 0, w, h };
        }
        function clampMarkerViewOffset() {
            markerView.scale = Math.min(markerView.maxScale, Math.max(markerView.minScale, Number(markerView.scale) || 1));
            if (markerView.scale <= markerView.minScale + 0.0001) {
                markerView.offsetX = 0;
                markerView.offsetY = 0;
                return;
            }
            const stage = get('marker-stage');
            const viewport = get('marker-viewport');
            if (!stage || !viewport) return;
            const stageW = Math.max(1, stage.clientWidth || 0);
            const stageH = Math.max(1, stage.clientHeight || 0);
            const baseW = Math.max(1, viewport.offsetWidth || viewport.clientWidth || 0);
            const baseH = Math.max(1, viewport.offsetHeight || viewport.clientHeight || 0);
            if (stageW <= 1 || stageH <= 1 || baseW <= 1 || baseH <= 1) return;
            const baseLeft = (stageW - baseW) / 2;
            const baseTop = (stageH - baseH) / 2;
            const scaledW = baseW * markerView.scale;
            const scaledH = baseH * markerView.scale;
            const minVisibleX = Math.min(stageW * 0.45, Math.max(24, stageW * 0.12));
            const minVisibleY = Math.min(stageH * 0.45, Math.max(24, stageH * 0.12));
            const minOffsetX = minVisibleX - baseLeft - scaledW;
            const maxOffsetX = stageW - minVisibleX - baseLeft;
            const minOffsetY = minVisibleY - baseTop - scaledH;
            const maxOffsetY = stageH - minVisibleY - baseTop;
            const clampOffset = (value, min, max) => {
                if (!Number.isFinite(value)) return 0;
                if (min > max) return (min + max) / 2;
                return Math.min(max, Math.max(min, value));
            };
            markerView.offsetX = clampOffset(markerView.offsetX, minOffsetX, maxOffsetX);
            markerView.offsetY = clampOffset(markerView.offsetY, minOffsetY, maxOffsetY);
        }
        function applyMarkerTransform() {
            const viewport = get('marker-viewport');
            if (!viewport) return;
            clampMarkerViewOffset();
            viewport.style.transform = `translate(${markerView.offsetX}px, ${markerView.offsetY}px) scale(${markerView.scale})`;
        }
        function resetMarkerTransform() {
            markerView.scale = 1;
            markerView.offsetX = 0;
            markerView.offsetY = 0;
            applyMarkerTransform();
        }
        function getRowMarkerKey(row) {
            if (!row) return null;
            return row.dataset.uploadId || row.getAttribute('data-filename') || null;
        }
        function setRowMarkerState(row, applied) {
            const key = getRowMarkerKey(row);
            if (key) {
                if (applied) markerAppliedUploads.add(key);
                else markerAppliedUploads.delete(key);
            }
            const tag = row ? row.querySelector('.upload-marker-tag') : null;
            if (tag) tag.classList.toggle('hidden', !applied);
        }
        function hasMarkerHint() {
            return markerAppliedUploads.size > 0;
        }
        function normalizeAttachmentSource(source) {
            const raw = String(source || '').trim().toLowerCase();
            if (raw === 'library' || raw === 'lib') return 'library';
            if (raw === 'upload' || raw === 'uploaded') return 'upload';
            return 'unknown';
        }
        function normalizeAttachmentDisplayName(name) {
            if (name === null || name === undefined) return '';
            let v = String(name).replace(/\u0000/g, '');
            v = v.replace(/\r/g, ' ').replace(/\n/g, ' ').replace(/\t/g, ' ');
            v = v.trim();
            if (!v) return '';
            v = v.split('/').pop().split('\\').pop().trim();
            v = v.replace(/\s{2,}/g, ' ');
            v = v.replace(/[<>:"/\\|?*]+/g, '_');
            if (!v || v === '.' || v === '..') return '';
            if (v.length > 180) v = v.slice(0, 180).trim();
            return v;
        }
        function defaultAttachmentDisplayName(path) {
            const norm = normalizeAttachmentPath(path);
            if (!norm) return '';
            return norm.split('/').pop() || norm;
        }
        function setAttachmentNameForPath(path, displayName) {
            const norm = normalizeAttachmentPath(path);
            if (!norm) return;
            const next = normalizeAttachmentDisplayName(displayName) || defaultAttachmentDisplayName(norm);
            if (!next) return;
            attachmentNameByPath.set(norm, next);
        }
        function getAttachmentNameForPath(path) {
            const norm = normalizeAttachmentPath(path);
            if (!norm) return '';
            const named = normalizeAttachmentDisplayName(attachmentNameByPath.get(norm));
            if (named) return named;
            return defaultAttachmentDisplayName(norm);
        }
        function setRowAttachmentName(row, displayName) {
            if (!row) return;
            const next = normalizeAttachmentDisplayName(displayName) || getAttachmentNameForPath(row.getAttribute('data-filename')) || 'file';
            row.dataset.displayName = next;
            const nameEl = row.querySelector('.truncate');
            if (nameEl) nameEl.textContent = next;
            const path = row.getAttribute('data-filename');
            if (path) setAttachmentNameForPath(path, next);
        }
        function isRowAttachmentNameCustomized(row) {
            return !!(row && row.dataset.sendNameCustomized === '1');
        }
        function setRowAttachmentNameCustomized(row, customized) {
            if (!row) return;
            row.dataset.sendNameCustomized = customized ? '1' : '';
        }
        function getRowDefaultAttachmentName(row) {
            if (!row) return 'file';
            const path = row.getAttribute('data-filename');
            if (path) return defaultAttachmentDisplayName(path) || 'file';
            const localDefault = normalizeAttachmentDisplayName(row.dataset.defaultDisplayName);
            if (localDefault) return localDefault;
            return normalizeAttachmentDisplayName(row.dataset.displayName) || 'file';
        }
        function promptRowAttachmentName(row) {
            if (!row) return;
            const currentName = getRowAttachmentName(row) || getRowDefaultAttachmentName(row) || 'file';
            const input = prompt('送信時のファイル名を入力してください（空欄でデフォルトに戻す）', currentName);
            if (input === null) return;
            const next = normalizeAttachmentDisplayName(input);
            if (!next) {
                const fallback = getRowDefaultAttachmentName(row);
                setRowAttachmentName(row, fallback);
                setRowAttachmentNameCustomized(row, false);
                showToast('送信名をデフォルトに戻しました', 'success');
                return;
            }
            setRowAttachmentName(row, next);
            setRowAttachmentNameCustomized(row, true);
            showToast('送信名を更新しました', 'success');
        }
        function getRowAttachmentName(row) {
            if (!row) return '';
            const path = row.getAttribute('data-filename');
            const fromPath = getAttachmentNameForPath(path);
            if (fromPath) return fromPath;
            const fromRow = normalizeAttachmentDisplayName(row.dataset.displayName);
            if (fromRow) return fromRow;
            const nameEl = row.querySelector('.truncate');
            const fromText = normalizeAttachmentDisplayName(nameEl ? nameEl.textContent : '');
            if (fromText) return fromText;
            return getAttachmentNameForPath(path);
        }
        function setAttachmentSourceForPath(path, source) {
            const norm = normalizeAttachmentPath(path);
            if (!norm) return;
            const src = normalizeAttachmentSource(source);
            if (src === 'unknown') return;
            attachmentSourceByPath.set(norm, src);
        }
        function getAttachmentSourceForPath(path) {
            const norm = normalizeAttachmentPath(path);
            if (!norm) return 'unknown';
            return normalizeAttachmentSource(attachmentSourceByPath.get(norm));
        }
        function setRowAttachmentSource(row, source) {
            if (!row) return;
            const src = normalizeAttachmentSource(source);
            row.dataset.fileSource = src;
            const path = row.getAttribute('data-filename');
            if (path) setAttachmentSourceForPath(path, src);
        }
        function getRowAttachmentSource(row) {
            if (!row) return 'unknown';
            const fromRow = normalizeAttachmentSource(row.dataset.fileSource);
            if (fromRow !== 'unknown') return fromRow;
            const path = row.getAttribute('data-filename');
            return getAttachmentSourceForPath(path);
        }
        function getRowOriginalAttachmentSource(row) {
            if (!row) return 'unknown';
            const fromRow = normalizeAttachmentSource(row.dataset.originalSource);
            if (fromRow !== 'unknown') return fromRow;
            const path = row.getAttribute('data-original-filename');
            return getAttachmentSourceForPath(path);
        }
        function prepareMarkerBaseCanvas(img, width, height) {
            const base = document.createElement('canvas');
            base.width = width;
            base.height = height;
            const bctx = base.getContext('2d');
            if (bctx) {
                bctx.drawImage(img, 0, 0, width, height);
                markerState.baseImageData = bctx.getImageData(0, 0, width, height);
                markerState.baseCanvas = base;
            } else {
                markerState.baseImageData = null;
                markerState.baseCanvas = null;
            }
        }
        function renderCropOverlay() {
            const cropCanvas = get('marker-crop-canvas');
            if (!cropCanvas) return;
            const ctx = cropCanvas.getContext('2d');
            if (!ctx) return;
            ctx.clearRect(0, 0, cropCanvas.width, cropCanvas.height);
            const drawRect = (rect, stroke, fill = null, dashed = false) => {
                if (!rect) return;
                const x = Math.max(0, rect.x);
                const y = Math.max(0, rect.y);
                const w = Math.max(1, rect.w);
                const h = Math.max(1, rect.h);
                if (fill) {
                    ctx.fillStyle = fill;
                    ctx.fillRect(x, y, w, h);
                }
                ctx.save();
                if (dashed) ctx.setLineDash([6, 4]);
                ctx.strokeStyle = stroke;
                ctx.lineWidth = 2;
                ctx.strokeRect(x + 0.5, y + 0.5, Math.max(1, w - 1), Math.max(1, h - 1));
                ctx.restore();
            };

            const rect = markerState.cropRect;
            const isFull = rect && rect.x === 0 && rect.y === 0 && Math.abs(rect.w - cropCanvas.width) < 1 && Math.abs(rect.h - cropCanvas.height) < 1;

            if (rect && (markerState.mode === 'crop' || !isFull)) {
                ctx.fillStyle = 'rgba(0,0,0,0.35)';
                ctx.fillRect(0, 0, cropCanvas.width, cropCanvas.height);
                const x = Math.max(0, rect.x);
                const y = Math.max(0, rect.y);
                const w = Math.max(1, rect.w);
                const h = Math.max(1, rect.h);
                ctx.clearRect(x, y, w, h);
                if (markerState.mode === 'crop') {
                    drawRect(rect, 'rgba(250,204,21,0.9)');
                } else {
                    drawRect(rect, 'rgba(250,204,21,0.4)');
                }
            }

            if (markerState.mode === 'crop') {
                return;
            }
            if (markerState.mode !== 'mosaic') return;
            const rects = Array.isArray(markerState.mosaicRects) ? markerState.mosaicRects : [];
            rects.forEach((rect) => drawRect(rect, 'rgba(250,204,21,0.9)', 'rgba(250,204,21,0.10)'));
            if (markerState.mosaicPreviewRect) {
                drawRect(markerState.mosaicPreviewRect, 'rgba(56,189,248,0.95)', 'rgba(56,189,248,0.14)', true);
            }
        }
        function collectImageUrlsForSend() {
            return collectAttachmentItemsForSend().map((it) => it.path);
        }
        function collectAttachmentItemsForSend() {
            const items = [];
            const indexByPath = new Map();
            const pushItem = (path, source, displayName) => {
                const norm = normalizeAttachmentPath(path);
                if (!norm) return;
                const src = normalizeAttachmentSource(source);
                const name = normalizeAttachmentDisplayName(displayName) || getAttachmentNameForPath(norm);
                const idx = indexByPath.get(norm);
                if (idx === undefined) {
                    const nextIndex = items.length;
                    indexByPath.set(norm, nextIndex);
                    items.push({ path: norm, source: src, name });
                    return;
                }
                const prev = items[idx];
                if (!prev) return;
                const prevSrc = normalizeAttachmentSource(prev.source);
                if (prevSrc === 'unknown' && src !== 'unknown') {
                    prev.source = src;
                } else if (prevSrc === 'library' && src === 'upload') {
                    prev.source = src;
                }
                if (!normalizeAttachmentDisplayName(prev.name) && name) {
                    prev.name = name;
                }
            };

            const list = get('upload-list');
            if (list) {
                list.querySelectorAll('[data-filename]').forEach((row) => {
                    const main = row.getAttribute('data-filename');
                    pushItem(main, getRowAttachmentSource(row), getRowAttachmentName(row));
                    const original = row.getAttribute('data-original-filename');
                    const attachOriginal = row.dataset.attachOriginal === '1';
                    if (attachOriginal) {
                        pushItem(original, getRowOriginalAttachmentSource(row), getAttachmentNameForPath(original));
                    }
                });
            }
            if (currentImageUrls && currentImageUrls.length) {
                currentImageUrls.forEach((u) => {
                    pushItem(u, getAttachmentSourceForPath(u), getAttachmentNameForPath(u));
                });
            }
            return items;
        }
        function collectUploadedImageUrlsForSend() {
            return collectAttachmentItemsForSend()
                .filter((it) => normalizeAttachmentSource(it.source) === 'upload')
                .map((it) => it.path);
        }
        function purgeUnsupportedAttachments(notify = true) {
            const support = getModelMediaSupport(get('model-select').value);
            let removedAudio = 0;
            let removedVideo = 0;
            if (Array.isArray(currentImageUrls) && currentImageUrls.length) {
                const filtered = [];
                currentImageUrls.forEach((fp) => {
                    const normalized = normalizeAttachmentPath(fp);
                    if (!normalized) return;
                    const isAudio = isAudioPath(normalized);
                    const isVideo = isVideoPath(normalized);
                    if ((isAudio && !support.audio) || (isVideo && !support.video)) {
                        if (isAudio) removedAudio += 1;
                        if (isVideo) removedVideo += 1;
                        return;
                    }
                    filtered.push(normalized);
                });
                if (filtered.length !== currentImageUrls.length) {
                    currentImageUrls = filtered;
                }
            }
            const list = get('upload-list');
            if (list) {
                list.querySelectorAll('[data-filename]').forEach((row) => {
                    const fp = row.getAttribute('data-filename');
                    if (!fp) return;
                    if (!currentImageUrls.includes(fp) && (isAudioPath(fp) || isVideoPath(fp))) {
                        setRowMarkerState(row, false);
                        row.remove();
                    }
                });
                if (list.children.length === 0) {
                    list.innerHTML = '<div class="text-xs text-gray-500">まだアップロードがありません。</div>';
                }
            }
            updateFilePreview();
            if (notify && (removedAudio || removedVideo)) {
                const parts = [];
                if (removedAudio) parts.push(`${removedAudio}件の音声`);
                if (removedVideo) parts.push(`${removedVideo}件の動画`);
                showToast(`このモデルは${parts.join('・')}入力に非対応のため削除しました`, "error", true);
            }
        }
        function getRowImageSource(row) {
            if (!row) return '';
            const localUrl = row.getAttribute('data-local-url');
            if (localUrl) return localUrl;
            const filepath = row.getAttribute('data-filename');
            if (filepath) return buildFileUrl(filepath);
            return '';
        }
        function buildFileUrl(path) {
            const norm = normalizeAttachmentPath(path);
            if (!norm) return '';
            return FILE_BASE_URL + norm;
        }
        function buildAttachmentPreviewUrl(path) {
            const norm = normalizeAttachmentPath(path);
            if (!norm) return '';
            if (isImagePath(norm)) return FILE_THUMB_BASE_URL + norm;
            return FILE_BASE_URL + norm;
        }
        window.closeMarkerModal = (skipHistory = false) => {
            hideModal('marker-modal');
            if (!skipHistory && location.pathname === '/edit-image') {
                history.back();
            }
        };
        function openMarkerModalForRow(row, maskMode = false) {
            const src = getRowImageSource(row);
            if (!src) {
                showToast("画像が読み込めませんでした", "error", true);
                return;
            }
            markerState.row = row;
            markerState.maskMode = maskMode;
            const nameEl = row ? row.querySelector('.truncate') : null;
            markerState.filename = nameEl ? nameEl.textContent.trim() : 'image.png';
            markerState.hasStroke = false;
            markerState.history = [];
            markerState.naturalWidth = 0;
            markerState.naturalHeight = 0;
            markerState.cropRect = null;
            markerState.mosaicRects = [];
            markerState.mosaicPreviewRect = null;
            markerState.baseCanvas = null;
            markerState.baseImageData = null;
            setMarkerMode('draw');
            const attachOriginal = get('marker-attach-original');
            if (attachOriginal) {
                attachOriginal.checked = row.dataset.attachOriginal === '1';
            }
            const img = get('marker-image');
            const canvas = get('marker-canvas');
            const cropCanvas = get('marker-crop-canvas');
            if (canvas) {
                const ctx = canvas.getContext('2d');
                if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
            }
            if (cropCanvas) {
                const cctx = cropCanvas.getContext('2d');
                if (cctx) cctx.clearRect(0, 0, cropCanvas.width, cropCanvas.height);
            }
            resetMarkerTransform();
            showModal('marker-modal');
            if (location.pathname !== '/edit-image') {
                history.pushState({ modal: 'marker' }, '', '/edit-image');
            }
            if (img) {
                img.onload = () => {
                    const stage = get('marker-stage');
                    if (!stage || !canvas) return;
                    const width = Math.max(1, Math.floor(img.clientWidth));
                    const height = Math.max(1, Math.floor(img.clientHeight));
                    canvas.width = width;
                    canvas.height = height;
                    canvas.style.width = `${width}px`;
                    canvas.style.height = `${height}px`;
                    canvas.style.left = '0px';
                    canvas.style.top = '0px';
                    if (cropCanvas) {
                        cropCanvas.width = width;
                        cropCanvas.height = height;
                        cropCanvas.style.width = `${width}px`;
                        cropCanvas.style.height = `${height}px`;
                        cropCanvas.style.left = '0px';
                        cropCanvas.style.top = '0px';
                    }
                    markerState.naturalWidth = img.naturalWidth || width;
                    markerState.naturalHeight = img.naturalHeight || height;
                    const ctx = canvas.getContext('2d');
                    if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
                    prepareMarkerBaseCanvas(img, width, height);
                    saveMarkerHistory();
                    if (markerState.mode === 'crop' && !markerState.cropRect) {
                        resetCropRectToFull();
                    }
                    renderCropOverlay();
                    resetMarkerTransform();
                };
                img.src = src;
            }
        }
        let uploadProgressState = { total: 0, completed: 0, active: 0, perFilePct: {} };
        const uploadCancelTokens = new Set();
        let uploadGeneration = 0;
        function isUploadCancelled(row) {
            return !!(row && (row.generation !== undefined && row.generation !== uploadGeneration
                || row.uploadId && uploadCancelTokens.has(row.uploadId)
                || row.row && !row.row.isConnected));
        }

        function updateGlobalUploadProgress(uploadId, pct) {
            if (uploadProgressState.perFilePct.hasOwnProperty(uploadId)) {
                uploadProgressState.perFilePct[uploadId] = pct;
                updateFilePreview();
            }
        }
        function resetUploadState() {
            uploadGeneration++;
            browserFastLocalFiles.forEach((entry) => {
                const row = entry && entry.rowObj ? entry.rowObj.row : null;
                const localUrl = row ? row.getAttribute('data-local-url') : null;
                if (localUrl) URL.revokeObjectURL(localUrl);
            });
            browserFastLocalFiles.clear();
            currentImageUrls = [];
            currentMaskImage = null;
            uploadProgressState = { total: 0, completed: 0, active: 0, perFilePct: {} };
            uploadCancelTokens.clear();
            markerAppliedUploads.clear();
            const previewBar = get('file-preview');
            if (previewBar) previewBar.classList.add('hidden');
            const thumbContainer = get('file-preview-thumbs');
            if (thumbContainer) {
                thumbContainer.innerHTML = '';
                thumbContainer.classList.add('hidden');
            }
            const list = get('upload-list');
            if (list) list.innerHTML = '<div class="text-xs text-gray-500">まだアップロードがありません。</div>';
            updateFilePreview();
            updateMaskPreview();

            const input = get('file-input');
            if (input) input.value = '';
            const photoInput = get('photo-input');
            if (photoInput) photoInput.value = '';
            const maskInput = get('mask-input');
            if (maskInput) maskInput.value = '';
        }

        async function uploadMaskFile(file) {
            if (!file) return false;
            const fd = new FormData();
            fd.append('file', file);
            try {
                const r = await fetch(CHAT_CONFIG.urls.upload, { method: 'POST', body: fd });
                const d = await r.json();
                if (r.ok && d.filename) {
                    currentMaskImage = d.filename;
                    updateMaskPreview();
                    return true;
                } else {
                    showToast(d.error || "Mask upload failed", "error", true);
                }
            } catch (e) {
                showToast("Mask upload failed", "error", true);
            }
            return false;
        }
