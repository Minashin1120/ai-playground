
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
            if (hint) hint.textContent = markerToolHints[mode] || '';
            const cropReset = get('marker-crop-reset');
            if (cropReset) cropReset.classList.toggle('hidden', mode !== 'crop');
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
        function openMarkerModalForRow(row) {
            const src = getRowImageSource(row);
            if (!src) {
                showToast("画像が読み込めませんでした", "error", true);
                return;
            }
            markerState.row = row;
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

        function updateGlobalUploadProgress(uploadId, pct) {
            if (uploadProgressState.perFilePct.hasOwnProperty(uploadId)) {
                uploadProgressState.perFilePct[uploadId] = pct;
                updateFilePreview();
            }
        }
        function resetUploadState() {
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
            updateFilePreview();
            updateMaskPreview();
            const list = get('upload-list');
            if (list) list.innerHTML = '<div class="text-xs text-gray-500">まだアップロードがありません。</div>';

            const input = get('file-input');
            if (input) input.value = '';
            const photoInput = get('photo-input');
            if (photoInput) photoInput.value = '';
            const maskInput = get('mask-input');
            if (maskInput) maskInput.value = '';
        }

        async function uploadMaskFile(file) {
            if (!file) return;
            const fd = new FormData();
            fd.append('file', file);
            try {
                const r = await fetch(CHAT_CONFIG.urls.upload, { method: 'POST', body: fd });
                const d = await r.json();
                if (r.ok && d.filename) {
                    currentMaskImage = d.filename;
                    updateMaskPreview();
                } else {
                    showToast(d.error || "Mask upload failed", "error", true);
                }
            } catch (e) {
                showToast("Mask upload failed", "error", true);
            }
        }
        function setCameraCaptureStatus(text, isError = false) {
            const el = get('camera-status');
            if (!el) return;
            el.textContent = text || '';
            el.classList.toggle('text-red-300', !!isError);
            el.classList.toggle('text-gray-400', !isError);
        }
        function updateCameraCapturePendingUi() {
            const count = cameraCapturePendingFiles.length;
            const attachBtn = get('camera-attach-btn');
            if (attachBtn) {
                attachBtn.disabled = count === 0 || cameraCaptureBusy;
                attachBtn.textContent = count ? `添付 (${count})` : '添付 (0)';
            }
            const clearBtn = get('camera-clear-btn');
            if (clearBtn) clearBtn.disabled = count === 0 || cameraCaptureBusy;
            const previewList = get('camera-capture-preview-list');
            if (previewList) {
                previewList.innerHTML = '';
                cameraCapturePendingPreviewUrls.forEach((url, index) => {
                    const item = document.createElement('div');
                    item.className = 'relative rounded overflow-hidden border border-gray-700 bg-black aspect-square';
                    item.innerHTML = `
                        <img src="${url}" alt="capture ${index + 1}" class="w-full h-full object-cover block">
                        <div class="absolute bottom-0 right-0 text-[10px] px-1 py-0.5 bg-black/70 text-white">${index + 1}</div>
                    `;
                    previewList.appendChild(item);
                });
                previewList.classList.toggle('hidden', count === 0);
            }
        }
        function resetCameraCapturePending(opts = {}) {
            while (cameraCapturePendingPreviewUrls.length) {
                const url = cameraCapturePendingPreviewUrls.pop();
                try { URL.revokeObjectURL(url); } catch (e) {}
            }
            cameraCapturePendingFiles.length = 0;
            updateCameraCapturePendingUi();
            if (!opts.keepStatus) {
                if (cameraCaptureStream) setCameraCaptureStatus('撮影して追加できます。最後に「添付」を押してください。');
                else setCameraCaptureStatus('カメラを起動中...');
            }
        }
        function stopCameraCaptureStream() {
            const video = get('camera-video');
            if (video && video.srcObject) {
                try { video.pause(); } catch (e) {}
                video.srcObject = null;
            }
            if (cameraCaptureStream) {
                try {
                    cameraCaptureStream.getTracks().forEach((track) => {
                        try { track.stop(); } catch (e) {}
                    });
                } catch (e) {}
            }
            cameraCaptureStream = null;
            cameraCaptureBusy = false;
            const captureBtn = get('camera-capture-btn');
            if (captureBtn) captureBtn.disabled = true;
            const switchBtn = get('camera-switch-btn');
            if (switchBtn) switchBtn.disabled = true;
        }
        async function startCameraCaptureStream(preferredFacingMode = 'environment') {
            const video = get('camera-video');
            if (!video) throw new Error('camera video element not found');
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                throw new Error('このブラウザはカメラAPIに対応していません');
            }
            stopCameraCaptureStream();
            setCameraCaptureStatus('カメラを起動中...');
            const switchBtn = get('camera-switch-btn');
            if (switchBtn) switchBtn.disabled = true;
            const candidates = [
                { video: { facingMode: { ideal: preferredFacingMode }, width: { ideal: 1920 }, height: { ideal: 1080 } }, audio: false },
                { video: { facingMode: preferredFacingMode }, audio: false },
                { video: true, audio: false }
            ];
            let lastErr = null;
            for (const constraints of candidates) {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia(constraints);
                    cameraCaptureStream = stream;
                    video.srcObject = stream;
                    try { await video.play(); } catch (e) {}
                    const track = stream.getVideoTracks && stream.getVideoTracks()[0];
                    const settings = track && track.getSettings ? track.getSettings() : {};
                    const actualFacing = String(settings.facingMode || '').toLowerCase();
                    if (actualFacing === 'user' || actualFacing === 'environment') {
                        cameraCaptureFacingMode = actualFacing;
                    } else {
                        cameraCaptureFacingMode = preferredFacingMode;
                    }
                    const captureBtn = get('camera-capture-btn');
                    if (captureBtn) captureBtn.disabled = false;
                    if (switchBtn) switchBtn.disabled = false;
                    setCameraCaptureStatus(cameraCapturePendingFiles.length > 0
                        ? `${cameraCapturePendingFiles.length}枚撮影済み。続けて撮影するか「添付」を押してください。`
                        : '撮影して追加できます。最後に「添付」を押してください。');
                    updateCameraCapturePendingUi();
                    return stream;
                } catch (e) {
                    lastErr = e;
                }
            }
            throw lastErr || new Error('カメラを起動できませんでした');
        }
        async function openCameraCaptureModal() {
            if (!window.isSecureContext && location.hostname !== 'localhost' && location.hostname !== '127.0.0.1') {
                showToast('カメラ起動は HTTPS / localhost 環境で利用できます。写真選択に切り替えます。', 'warning', true);
                const photoInput = get('photo-input');
                if (photoInput) photoInput.click();
                return;
            }
            resetCameraCapturePending({ keepStatus: true });
            updateCameraCapturePendingUi();
            showModal('camera-capture-modal');
            if (location.pathname !== '/camera') {
                history.pushState({ modal: 'camera' }, '', '/camera');
            }
            try {
                await startCameraCaptureStream(cameraCaptureFacingMode || 'environment');
            } catch (e) {
                const msg = (e && e.message) ? e.message : 'カメラを起動できませんでした';
                setCameraCaptureStatus(msg, true);
                showToast(msg, 'error', true);
                const captureBtn = get('camera-capture-btn');
                if (captureBtn) captureBtn.disabled = true;
                const attachBtn = get('camera-attach-btn');
                if (attachBtn) attachBtn.disabled = true;
            }
        }
        function closeCameraCaptureModal(options = {}) {
            const skipHistory = options.skipHistory || false;
            hideModal('camera-capture-modal', options);
            if (!skipHistory && location.pathname === '/camera') {
                history.back();
            }
        }
        async function toggleCameraCaptureFacing() {
            if (cameraCaptureBusy) return;
            const switchBtn = get('camera-switch-btn');
            if (switchBtn) switchBtn.disabled = true;
            const next = String(cameraCaptureFacingMode || '').toLowerCase() === 'user' ? 'environment' : 'user';
            cameraCaptureFacingMode = next;
            try {
                await startCameraCaptureStream(next);
            } catch (e) {
                const msg = (e && e.message) ? e.message : 'カメラ切替に失敗しました';
                setCameraCaptureStatus(msg, true);
                showToast(msg, 'error', true);
            } finally {
                if (switchBtn && get('camera-capture-modal') && !get('camera-capture-modal').classList.contains('hidden')) {
                    switchBtn.disabled = false;
                }
            }
        }
        function buildCameraCaptureFilename() {
            const now = new Date();
            const pad = (n) => String(n).padStart(2, '0');
            const ms = String(now.getMilliseconds()).padStart(3, '0');
            cameraCaptureSequence = (cameraCaptureSequence + 1) % 1000;
            const seq = String(cameraCaptureSequence).padStart(3, '0');
            return `camera_${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}_${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}_${ms}_${seq}.jpg`;
        }
        async function captureCameraShot() {
            if (cameraCaptureBusy) return;
            const video = get('camera-video');
            const canvas = get('camera-canvas');
            const modal = get('camera-capture-modal');
            if (!video || !canvas || !modal) return;
            if (!video.videoWidth || !video.videoHeight) {
                showToast('カメラ映像の準備中です。少し待ってから再度お試しください。', 'warning', true);
                return;
            }
            cameraCaptureBusy = true;
            const captureBtn = get('camera-capture-btn');
            if (captureBtn) captureBtn.disabled = true;
            const attachBtn = get('camera-attach-btn');
            if (attachBtn) attachBtn.disabled = true;
            setCameraCaptureStatus('撮影中...');
            try {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                const ctx = canvas.getContext('2d');
                if (!ctx) throw new Error('撮影処理に失敗しました');
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                const blob = await new Promise((resolve, reject) => {
                    canvas.toBlob((b) => {
                        if (b) resolve(b);
                        else reject(new Error('画像の生成に失敗しました'));
                    }, 'image/jpeg', 0.92);
                });
                const file = new File([blob], buildCameraCaptureFilename(), { type: 'image/jpeg', lastModified: Date.now() });
                cameraCapturePendingFiles.push(file);
                cameraCapturePendingPreviewUrls.push(URL.createObjectURL(blob));
                updateCameraCapturePendingUi();
                setCameraCaptureStatus(`${cameraCapturePendingFiles.length}枚撮影済み。続けて撮影するか「添付」を押してください。`);
            } catch (e) {
                const msg = (e && e.message) ? e.message : '撮影に失敗しました';
                setCameraCaptureStatus(msg, true);
                showToast(msg, 'error', true);
            } finally {
                cameraCaptureBusy = false;
                if (captureBtn && modal && !modal.classList.contains('hidden')) {
                    captureBtn.disabled = false;
                }
                updateCameraCapturePendingUi();
            }
        }
        async function attachCameraCapturedFiles() {
            if (cameraCaptureBusy) return;
            if (!cameraCapturePendingFiles.length) {
                showToast('先に撮影してください', 'warning', true);
                return;
            }
            const modal = get('camera-capture-modal');
            cameraCaptureBusy = true;
            const captureBtn = get('camera-capture-btn');
            const switchBtn = get('camera-switch-btn');
            const attachBtn = get('camera-attach-btn');
            const clearBtn = get('camera-clear-btn');
            if (captureBtn) captureBtn.disabled = true;
            if (switchBtn) switchBtn.disabled = true;
            if (attachBtn) attachBtn.disabled = true;
            if (clearBtn) clearBtn.disabled = true;
            const filesToUpload = Array.from(cameraCapturePendingFiles).reverse();

            // モーダルを即座に閉じて他の操作を可能にする
            // skipReset: true を指定して、アップロード中にファイルが消されないようにする
            closeCameraCaptureModal({ skipReset: true });
            // closeUploadModal(); // Prevent history conflict

            // stopCameraCaptureStream が busy = false にしてしまうため、戻す
            cameraCaptureBusy = true;

            setCameraCaptureStatus(`${filesToUpload.length}枚を添付中...`);
            try {
                await handleFiles(filesToUpload, { openModal: false });
                showToast(`${filesToUpload.length}枚の画像を添付しました`, 'success');
            } catch (e) {
                const msg = (e && e.message) ? e.message : '撮影画像の添付に失敗しました';
                showToast(msg, 'error', true);
            } finally {
                cameraCaptureBusy = false;
                // 成功・失敗に関わらず最後は撮影キューを空にする
                resetCameraCapturePending({ keepStatus: true });
                if (modal && !modal.classList.contains('hidden')) {
                    if (captureBtn) captureBtn.disabled = false;
                    if (switchBtn) switchBtn.disabled = false;
                    updateCameraCapturePendingUi();
                }
            }
        }
        function openUploadModal() {
            syncUploadRowsFromCurrent();
            showModal('upload-modal');
            if (location.pathname !== '/upload') {
                history.pushState({ modal: 'upload' }, '', '/upload');
            }
            const vmi = get('vision-model-info');
            if (vmi) {
                const model = get('model-select') ? get('model-select').value : '';
                const uploadModelLower = model.toLowerCase();
                // Vision Exp accepts images natively, so the vision-model notice is not needed.
                const needsVisionNotice = uploadModelLower.includes('deepseek') && uploadModelLower !== 'deepseek-v4-flash-vision-exp';
                vmi.classList.toggle('hidden', !needsVisionNotice);
            }
            _syncVisionModelDisplay();
        }
        function _syncVisionModelDisplay() {
            const display = get('vision-model-display');
            if (!display) return;
            const vm = currentVisionModel;
            if (vm) {
                let name = vm;
                MODELS.forEach(g => (g.items || []).forEach(m => { if (m.id === vm) name = m.name; }));
                display.textContent = name;
            } else {
                display.textContent = '設定から選択';
            }
        }
        function _openVisionModelSelector() {
            window._visionPickerActive = true;
            openModelModal();
            setTimeout(() => {
                const searchEl = get('model-search');
                if (searchEl) searchEl.value = '';
                renderModelList('');
            }, 50);
        }
        function closeUploadModal(skipHistory = false) {
            hideModal('upload-modal');
            if (!skipHistory && location.pathname === '/upload') {
                history.back();
            }
        }
        function syncUploadRowsFromCurrent() {
            const list = get('upload-list');
            if (!list) return;
            const existing = new Set();
            list.querySelectorAll('[data-filename]').forEach((el) => {
                const fp = el.getAttribute('data-filename');
                if (fp) existing.add(fp);
            });
            currentImageUrls.forEach((fp) => {
                if (!existing.has(fp)) {
                    addStoredUploadRow(fp, {
                        source: getAttachmentSourceForPath(fp),
                        displayName: getAttachmentNameForPath(fp)
                    });
                }
            });
            if (list.children.length === 0) {
                list.innerHTML = '<div class="text-xs text-gray-500">まだアップロードがありません。</div>';
            }
        }
        function decrementUploadTotal(uploadId) {
            if (uploadProgressState.total > 0) uploadProgressState.total--;
            if (uploadProgressState.perFilePct.hasOwnProperty(uploadId)) {
                delete uploadProgressState.perFilePct[uploadId];
                if (uploadProgressState.active > 0) uploadProgressState.active--;
            }
            if (uploadProgressState.active <= 0) {
                uploadProgressState.total = 0;
                uploadProgressState.completed = 0;
                uploadProgressState.active = 0;
                uploadProgressState.perFilePct = {};
            }
            updateFilePreview();
        }
        function addStoredUploadRow(filepath, opts = {}) {
            if (!filepath) return null;
            filepath = normalizeAttachmentPath(filepath);
            if (!filepath) return null;
            const source = normalizeAttachmentSource(opts.source);
            const list = get('upload-list');
            if (!list) return null;
            if (list.children.length === 1 && list.children[0].classList.contains('text-gray-500')) {
                list.innerHTML = '';
            }
            const filename = filepath.split('/').pop() || filepath;
            const displayName = normalizeAttachmentDisplayName(opts.displayName) || getAttachmentNameForPath(filepath) || filename;
            const ext = (filename.split('.').pop() || '').toLowerCase();
            const isImage = ['png','jpg','jpeg','webp','gif'].includes(ext);
            const fileUrl = buildFileUrl(filepath);
            const previewUrl = isImage ? buildAttachmentPreviewUrl(filepath) : fileUrl;
            const uploadId = `lib_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
            const row = document.createElement('div');
            row.className = 'upload-row ui-enter bg-gray-900/60 rounded p-2';
            row.dataset.uploadId = uploadId;
            row.setAttribute('data-filename', filepath);
            row.dataset.fileSource = source;
            row.dataset.displayName = displayName;
            row.dataset.defaultDisplayName = displayName;
            row.dataset.sendNameCustomized = '';
            const safeName = escapeHtml(displayName);
            const markerBtnHtml = isImage && !browserFastModeEnabled
                ? `<button class="upload-marker text-[10px] border rounded px-2 py-1">画像編集</button>`
                : '';
            const previewHtml = isImage
                ? `<img src="${previewUrl}" loading="lazy" decoding="async" class="upload-preview w-12 h-12 object-cover rounded border border-gray-700 cursor-pointer" alt="${safeName}">`
                : `<div class="upload-preview w-12 h-12 bg-gray-800 rounded border border-gray-700 flex items-center justify-center text-gray-400 text-sm cursor-pointer">FILE</div>`;
            row.innerHTML = `
                <div class="flex items-center gap-3">
                    ${previewHtml}
                    <div class="flex-1 min-w-0">
                        <div class="truncate text-xs text-gray-200">${safeName}</div>
                        <div class="flex items-center gap-2">
                            <div class="upload-status text-[10px] text-gray-400">ready</div>
                            <span class="upload-marker-tag hidden">編集済み</span>
                        </div>
                    </div>
                    <div class="flex items-center gap-1">
                        ${markerBtnHtml}
                        <button class="upload-send-name text-[10px] text-gray-300 hover:text-white border border-gray-700 rounded px-2 py-1">送信名</button>
                        <button class="upload-remove text-[10px] text-gray-400 hover:text-red-400 border border-gray-700 rounded px-2 py-1">削除</button>
                    </div>
                </div>
                <div class="upload-progress h-2 rounded mt-2 overflow-hidden">
                    <div style="width:100%"></div>
                </div>
            `;
            const previewEl = row.querySelector('.upload-preview');
            if (previewEl) {
                previewEl.onclick = () => openFileViewer(fileUrl, getRowAttachmentName(row) || displayName);
            }
            const sendNameBtn = row.querySelector('.upload-send-name');
            if (sendNameBtn) {
                sendNameBtn.onclick = () => promptRowAttachmentName(row);
            }
            const removeBtn = row.querySelector('.upload-remove');
            if (removeBtn) {
                removeBtn.onclick = () => {
                    uploadCancelTokens.add(uploadId);
                    browserFastLocalFiles.delete(uploadId);
                    decrementUploadTotal(uploadId);
                    const stored = row.getAttribute('data-filename');
                    if (stored) currentImageUrls = currentImageUrls.filter(x => x !== stored);
                    setRowMarkerState(row, false);
                    row.remove();
                    updateFilePreview();
                    if (list.children.length === 0) {
                        list.innerHTML = '<div class="text-xs text-gray-500">まだアップロードがありません。</div>';
                    }
                };
            }
            const markerBtn = row.querySelector('.upload-marker');
            if (markerBtn) {
                markerBtn.onclick = () => openMarkerModalForRow(row);
            }
            setAttachmentSourceForPath(filepath, source);
            setAttachmentNameForPath(filepath, displayName);
            list.prepend(row);
            return {
                row: row,
                bar: row.querySelector('.upload-progress > div'),
                status: row.querySelector('.upload-status'),
                uploadId: uploadId
            };
        }
        function addUploadRow(file) {
            const list = get('upload-list');
            if (!list) return null;
            if (list.children.length === 1 && list.children[0].classList.contains('text-gray-500')) {
                list.innerHTML = '';
            }
            const uploadId = `up_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
            const row = document.createElement('div');
            row.className = 'upload-row ui-enter bg-gray-900/60 rounded p-2';
            row.dataset.uploadId = uploadId;
            row.dataset.fileSource = 'upload';
            const displayName = normalizeAttachmentDisplayName(file.name || 'file') || 'file';
            row.dataset.displayName = displayName;
            row.dataset.defaultDisplayName = displayName;
            row.dataset.sendNameCustomized = '';
            const safeName = escapeHtml(displayName);
            const isImage = file && file.type && file.type.startsWith('image/');
            let previewHtml = '<div class="upload-preview w-12 h-12 bg-gray-800 rounded border border-gray-700 flex items-center justify-center text-gray-400 text-sm">FILE</div>';
            const markerBtnHtml = isImage && !browserFastModeEnabled
                ? `<button class="upload-marker text-[10px] border rounded px-2 py-1">画像編集</button>`
                : '';
            let previewUrl = '';
            if (isImage) {
                previewUrl = URL.createObjectURL(file);
                previewHtml = `<img src="${previewUrl}" class="upload-preview w-12 h-12 object-cover rounded border border-gray-700 cursor-pointer" alt="${safeName}">`;
            } else {
                previewUrl = URL.createObjectURL(file);
                previewHtml = `<div class="upload-preview w-12 h-12 bg-gray-800 rounded border border-gray-700 flex items-center justify-center text-gray-400 text-sm cursor-pointer">FILE</div>`;
            }
            row.innerHTML = `
                <div class="flex items-center gap-3">
                    ${previewHtml}
                    <div class="flex-1 min-w-0">
                        <div class="truncate text-xs text-gray-200">${safeName}</div>
                        <div class="flex items-center gap-2">
                            <div class="upload-status text-[10px] text-gray-400">待機中</div>
                            <span class="upload-marker-tag hidden">編集済み</span>
                        </div>
                    </div>
                    <div class="flex items-center gap-1">
                        ${markerBtnHtml}
                        <button class="upload-send-name text-[10px] text-gray-300 hover:text-white border border-gray-700 rounded px-2 py-1">送信名</button>
                        <button class="upload-remove text-[10px] text-gray-400 hover:text-red-400 border border-gray-700 rounded px-2 py-1">削除</button>
                    </div>
                </div>
                <div class="upload-progress h-2 rounded mt-2 overflow-hidden">
                    <div style="width:0%"></div>
                </div>
            `;
            if (previewUrl) row.setAttribute('data-local-url', previewUrl);
            const previewEl = row.querySelector('.upload-preview');
            if (previewEl) {
                previewEl.onclick = () => {
                    const filepath = row.getAttribute('data-filename');
                    const url = filepath ? buildFileUrl(filepath) : row.getAttribute('data-local-url');
                    const openName = normalizeAttachmentDisplayName(row.dataset.displayName) || file.name || filepath || '';
                    openFileViewer(url, openName);
                };
            }
            const removeBtn = row.querySelector('.upload-remove');
            if (removeBtn) {
                removeBtn.onclick = () => {
                    uploadCancelTokens.add(uploadId);
                    browserFastLocalFiles.delete(uploadId);
                    decrementUploadTotal(uploadId);
                    const localUrl = row.getAttribute('data-local-url');
                    if (localUrl) URL.revokeObjectURL(localUrl);
                    const stored = row.getAttribute('data-filename');
                    if (stored) currentImageUrls = currentImageUrls.filter(x => x !== stored);
                    setRowMarkerState(row, false);
                    row.remove();
                    updateFilePreview();
                    if (list.children.length === 0) {
                        list.innerHTML = '<div class="text-xs text-gray-500">まだアップロードがありません。</div>';
                    }
                };
            }
            const markerBtn = row.querySelector('.upload-marker');
            if (markerBtn) {
                markerBtn.onclick = () => openMarkerModalForRow(row);
            }
            const sendNameBtn = row.querySelector('.upload-send-name');
            if (sendNameBtn) {
                sendNameBtn.onclick = () => promptRowAttachmentName(row);
            }
            list.prepend(row);
            return {
                uploadId,
                row,
                status: row.querySelector('.upload-status'),
                bar: row.querySelector('.upload-progress > div')
            };
        }
        // Use chunked upload for medium/large files to avoid proxy/body size limits.
        const CHUNK_THRESHOLD_BYTES = 20 * 1024 * 1024;
        async function uploadFileChunked(file, row) {
            if (!file) return false;
            let uploadOpStarted = false;
            if (window.ConnectionMonitor) {
                window.ConnectionMonitor.operationStarted();
                uploadOpStarted = true;
            }
            try {
                const initRes = await apiFetch("/upload/init", {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json', 'X-CSRF-Token': csrfToken},
                    body: JSON.stringify({ filename: file.name, size: file.size })
                });
                const initData = await initRes.json();
                if (!initRes.ok) {
                    const msg = initData && initData.error ? initData.error : "アップロードに失敗しました";
                    if (row && row.status) row.status.textContent = '失敗';
                    showToast(msg, "error", true);
                    return false;
                }
                const uploadId = initData.upload_id;
                const chunkSize = initData.chunk_size || (10 * 1024 * 1024);
                const totalChunks = Math.ceil(file.size / chunkSize);
                for (let i = 0; i < totalChunks; i++) {
                    const start = i * chunkSize;
                    const end = Math.min(file.size, start + chunkSize);
                    const chunk = file.slice(start, end);
                    const ok = await new Promise((resolve) => {
                        const xhr = new XMLHttpRequest();
                        xhr.open('POST', "/upload/chunk", true);
                        xhr.setRequestHeader('X-CSRF-Token', csrfToken);
                        xhr.upload.onprogress = (e) => {
                            if (e.lengthComputable && row && row.bar) {
                                const done = start + e.loaded;
                                const pct = Math.min(100, Math.floor((done / file.size) * 100));
                                row.bar.style.width = `${pct}%`;
                                if (row.status) row.status.textContent = `${pct}%`;
                                if (row.uploadId) updateGlobalUploadProgress(row.uploadId, pct);
                            }
                            if (window.ConnectionMonitor) window.ConnectionMonitor.reportActivity();
                        };
                        xhr.onload = () => {
                            if (xhr.status >= 200 && xhr.status < 300) resolve(true);
                            else resolve(false);
                        };
                        xhr.onerror = () => resolve(false);
                        const fd = new FormData();
                        fd.append('upload_id', uploadId);
                        fd.append('index', String(i));
                        fd.append('total', String(totalChunks));
                        fd.append('chunk', chunk, file.name);
                        xhr.send(fd);
                    });
                    if (!ok) {
                        if (row && row.status) row.status.textContent = '失敗';
                        showToast("アップロードに失敗しました", "error", true);
                        return false;
                    }
                }
                if (row && row.status) row.status.textContent = '処理中...';
                const doneRes = await apiFetch("/upload/complete", {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json', 'X-CSRF-Token': csrfToken},
                    body: JSON.stringify({ upload_id: uploadId })
                });
                const doneData = await doneRes.json();
                if (doneRes.ok && doneData && doneData.filename) {
                    if (row && row.row && row.uploadId && uploadCancelTokens.has(row.uploadId)) {
                        if (row.row && row.row.parentNode) row.row.remove();
                        return false;
                    }
                    if (row && row.row) {
                        const localUrl = row.row.getAttribute('data-local-url');
                        if (localUrl) URL.revokeObjectURL(localUrl);
                        row.row.removeAttribute('data-local-url');

                        // Update source to server URL
                        const img = row.row.querySelector('img.upload-preview');
                        if (img) {
                            const displayPath = doneData.filename.replace(/^\d+\//, '');
                            img.src = buildAttachmentPreviewUrl(displayPath);
                        }
                    }
                    const normPath = normalizeAttachmentPath(doneData.filename);
                    if (normPath) currentImageUrls.push(normPath);
                    if (row && row.row) {
                        row.row.setAttribute('data-filename', normPath || doneData.filename);
                        setRowAttachmentSource(row.row, 'upload');
                        if (normPath) {
                            const hasCustom = isRowAttachmentNameCustomized(row.row);
                            const fallbackName = defaultAttachmentDisplayName(normPath);
                            const nextName = hasCustom ? (normalizeAttachmentDisplayName(row.row.dataset.displayName) || fallbackName) : fallbackName;
                            row.row.dataset.defaultDisplayName = fallbackName;
                            setRowAttachmentName(row.row, nextName);
                        }
                    }
                    if (normPath) setAttachmentSourceForPath(normPath, 'upload');
                    if (row && row.status) row.status.textContent = '完了';
                    updateFilePreview();
                    const filenames = Array.isArray(doneData.filenames) && doneData.filenames.length ? doneData.filenames : [doneData.filename];
                    filenames.forEach((fp) => addLibraryFileFromPath(fp));
                    return true;
                }
                const msg = (doneData && doneData.error) ? doneData.error : "アップロードに失敗しました";
                if (row && row.status) row.status.textContent = '失敗';
                showToast(msg, "error", true);
                return false;
            } catch (e) {
                if (row && row.status) row.status.textContent = '失敗';
                showToast("アップロード中にエラーが発生しました", "error", true);
                return false;
            } finally {
                if (uploadOpStarted && window.ConnectionMonitor) window.ConnectionMonitor.operationEnded();
            }
        }
