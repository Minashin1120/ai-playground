        function uploadFileWithProgress(file, row) {
            return new Promise((resolve) => {
                if (file && file.size > CHUNK_THRESHOLD_BYTES) {
                    uploadFileChunked(file, row).then(resolve);
                    return;
                }
                let uploadOpStarted = false;
                if (window.ConnectionMonitor) {
                    window.ConnectionMonitor.operationStarted();
                    uploadOpStarted = true;
                }
                const finishUploadOp = () => {
                    if (uploadOpStarted && window.ConnectionMonitor) {
                        window.ConnectionMonitor.operationEnded();
                        uploadOpStarted = false;
                    }
                };
                const xhr = new XMLHttpRequest();
                xhr.open('POST', CHAT_CONFIG.urls.upload, true);
                xhr.setRequestHeader('X-CSRF-Token', csrfToken);
                xhr.upload.onprogress = (e) => {
                    if (e.lengthComputable && row && row.bar) {
                        const pct = Math.min(100, Math.floor((e.loaded / e.total) * 100));
                        row.bar.style.width = `${pct}%`;
                        if (row.status) row.status.textContent = `${pct}%`;
                        if (row.uploadId) updateGlobalUploadProgress(row.uploadId, pct);
                    }
                    if (window.ConnectionMonitor) window.ConnectionMonitor.reportActivity();
                };
                xhr.onload = () => {
                    let d = {};
                    try { d = JSON.parse(xhr.responseText || '{}'); } catch (e) {}
                    if (xhr.status >= 200 && xhr.status < 300 && d && d.filename) {
                        if (row && row.row && row.uploadId && uploadCancelTokens.has(row.uploadId)) {
                            if (row.row && row.row.parentNode) row.row.remove();
                            finishUploadOp();
                            resolve(false);
                            return;
                        }
                        if (row && row.row) {
                            const localUrl = row.row.getAttribute('data-local-url');
                            if (localUrl) URL.revokeObjectURL(localUrl);
                            row.row.removeAttribute('data-local-url');

                            // Update source to server URL
                            const img = row.row.querySelector('img.upload-preview');
                            if (img) {
                                const displayPath = d.filename.replace(/^\d+\//, '');
                                img.src = buildAttachmentPreviewUrl(displayPath);
                            }
                        }
                        const normPath = normalizeAttachmentPath(d.filename);
                        if (normPath) currentImageUrls.push(normPath);
                        if (row && row.row) {
                            row.row.setAttribute('data-filename', normPath || d.filename);
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
                        const filenames = Array.isArray(d.filenames) && d.filenames.length ? d.filenames : [d.filename];
                        filenames.forEach((fp) => addLibraryFileFromPath(fp));
                        finishUploadOp();
                        resolve(true);
                    } else {
                        const msg = (d && d.error) ? d.error : "アップロードに失敗しました";
                        if (row && row.status) row.status.textContent = '失敗';
                        showToast(msg, "error", true);
                        finishUploadOp();
                        resolve(false);
                    }
                };
                xhr.onerror = () => {
                    if (row && row.status) row.status.textContent = '失敗';
                    showToast("アップロード中にエラーが発生しました", "error", true);
                    finishUploadOp();
                    resolve(false);
                };
                const fd = new FormData();
                fd.append('file', file);
                xhr.send(fd);
            });
        }
        function isVideoFile(file) {
            if (!file) return false;
            if (file.type && file.type.startsWith('video/')) return true;
            return VIDEO_EXTS.includes(getFileExt(file.name || ''));
        }
        function isAudioFile(file) {
            if (!file) return false;
            if (file.type && file.type.startsWith('audio/')) return true;
            return AUDIO_EXTS.includes(getFileExt(file.name || ''));
        }
        function encodeWav(buffers, sampleRate) {
            let length = 0;
            buffers.forEach(b => { length += b.length; });
            const pcm = new Float32Array(length);
            let offset = 0;
            buffers.forEach(b => {
                pcm.set(b, offset);
                offset += b.length;
            });
            const bytes = new ArrayBuffer(44 + pcm.length * 2);
            const view = new DataView(bytes);
            const writeString = (o, s) => { for (let i = 0; i < s.length; i++) view.setUint8(o + i, s.charCodeAt(i)); };
            writeString(0, 'RIFF');
            view.setUint32(4, 36 + pcm.length * 2, true);
            writeString(8, 'WAVE');
            writeString(12, 'fmt ');
            view.setUint32(16, 16, true);
            view.setUint16(20, 1, true);
            view.setUint16(22, 1, true);
            view.setUint32(24, sampleRate, true);
            view.setUint32(28, sampleRate * 2, true);
            view.setUint16(32, 2, true);
            view.setUint16(34, 16, true);
            writeString(36, 'data');
            view.setUint32(40, pcm.length * 2, true);
            let idx = 44;
            for (let i = 0; i < pcm.length; i++) {
                const s = Math.max(-1, Math.min(1, pcm[i]));
                view.setInt16(idx, s < 0 ? s * 0x8000 : s * 0x7fff, true);
                idx += 2;
            }
            return new Blob([view], { type: 'audio/wav' });
        }
        function pickAudioRecorderType() {
            if (typeof MediaRecorder === 'undefined') return '';
            const candidates = [
                'audio/webm;codecs=opus',
                'audio/webm',
                'audio/ogg;codecs=opus',
                'audio/ogg'
            ];
            for (const c of candidates) {
                if (MediaRecorder.isTypeSupported(c)) return c;
            }
            return '';
        }
        function updateUploadRowFile(row, file) {
            if (!row || !row.row || !file) return;
            const nameEl = row.row.querySelector('.truncate');
            const hasCustomName = isRowAttachmentNameCustomized(row.row);
            const displayName = hasCustomName
                ? (normalizeAttachmentDisplayName(row.row.dataset.displayName) || 'file')
                : (normalizeAttachmentDisplayName(file.name || 'file') || 'file');
            if (nameEl) nameEl.textContent = displayName;
            row.row.dataset.displayName = displayName;
            if (!hasCustomName) row.row.dataset.defaultDisplayName = displayName;
            const localUrl = row.row.getAttribute('data-local-url');
            if (localUrl) URL.revokeObjectURL(localUrl);
            const newUrl = URL.createObjectURL(file);
            row.row.setAttribute('data-local-url', newUrl);
            const isImage = file.type && file.type.startsWith('image/');
            const safeName = escapeHtml(displayName);
            const previewHtml = isImage
                ? `<img src="${newUrl}" class="upload-preview w-12 h-12 object-cover rounded border border-gray-700 cursor-pointer" alt="${safeName}">`
                : `<div class="upload-preview w-12 h-12 bg-gray-800 rounded border border-gray-700 flex items-center justify-center text-gray-400 text-sm cursor-pointer">FILE</div>`;
            const previewOld = row.row.querySelector('.upload-preview');
            if (previewOld) previewOld.outerHTML = previewHtml;
            const previewEl = row.row.querySelector('.upload-preview');
            if (previewEl) {
                previewEl.onclick = () => {
                    const filepath = row.row.getAttribute('data-filename');
                    const url = filepath ? buildFileUrl(filepath) : row.row.getAttribute('data-local-url');
                    openFileViewer(url, getRowAttachmentName(row.row) || displayName || filepath || '');
                };
            }
            const markerBtn = row.row.querySelector('.upload-marker');
            if (markerBtn) {
                markerBtn.classList.toggle('hidden', !isImage);
            }
            if (!isImage) {
                setRowMarkerState(row.row, false);
                row.row.dataset.originalFilename = '';
                row.row.dataset.originalSource = '';
                row.row.dataset.attachOriginal = '';
            }
        }
        function saveMarkerHistory() {
            const canvas = get('marker-canvas');
            if (!canvas) return;
            const ctx = canvas.getContext('2d');
            if (!ctx) return;
            const mosaicRects = Array.isArray(markerState.mosaicRects)
                ? markerState.mosaicRects.map((r) => ({ x: r.x, y: r.y, w: r.w, h: r.h }))
                : [];
            markerState.history.push({
                imageData: ctx.getImageData(0, 0, canvas.width, canvas.height),
                mosaicRects
            });
            if (markerState.history.length > 40) markerState.history.shift();
        }
        function undoMarkerCanvas() {
            if (markerState.history.length <= 1) return; // Only initial state left or empty
            markerState.history.pop(); // Remove current
            const canvas = get('marker-canvas');
            if (!canvas) return;
            const ctx = canvas.getContext('2d');
            if (!ctx) return;
            const last = markerState.history[markerState.history.length - 1];
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            if (last && last.imageData) {
                ctx.putImageData(last.imageData, 0, 0);
                markerState.mosaicRects = Array.isArray(last.mosaicRects)
                    ? last.mosaicRects.map((r) => ({ x: r.x, y: r.y, w: r.w, h: r.h }))
                    : [];
            } else if (last) {
                // Backward compatible fallback for old history snapshots.
                ctx.putImageData(last, 0, 0);
                markerState.mosaicRects = [];
            } else {
                markerState.mosaicRects = [];
            }
            markerState.mosaicPreviewRect = null;
            markerState.hasStroke = markerState.history.length > 1;
            renderCropOverlay();
        }
        function clearMarkerCanvas() {
            const canvas = get('marker-canvas');
            if (!canvas) return;
            const ctx = canvas.getContext('2d');
            if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
            markerState.hasStroke = false;
            markerState.mosaicRects = [];
            markerState.mosaicPreviewRect = null;
            renderCropOverlay();
            saveMarkerHistory();
        }
        function initMarkerCanvas() {
            const canvas = get('marker-canvas');
            if (!canvas) return;
            const ctx = canvas.getContext('2d');
            const sizeInput = get('marker-size');
            const activePointers = new Map();
            let pinching = false;
            let pinchStartDist = 0;
            let pinchStartScale = markerView.scale;
            let pinchStartOffset = { x: 0, y: 0 };
            let pinchStartMid = { x: 0, y: 0 };
            let strokePoints = [];
            let strokeSize = 16;
            let strokeStyle = '';
            let drawBaseCanvas = null;
            let drawBaseCtx = null;
            let drawLayerCanvas = null;
            let drawLayerCtx = null;
            let mosaicSelecting = false;
            let mosaicStartPoint = null;
            const getPoint = (e) => {
                const rect = canvas.getBoundingClientRect();
                const x = (e.clientX - rect.left) * (canvas.width / rect.width);
                const y = (e.clientY - rect.top) * (canvas.height / rect.height);
                return { x, y };
            };
            const getMid = (a, b) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });
            const getDist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
            let drawing = false;
            const ensureDrawBuffers = () => {
                if (!drawBaseCanvas) {
                    drawBaseCanvas = document.createElement('canvas');
                    drawBaseCtx = drawBaseCanvas.getContext('2d');
                }
                if (!drawLayerCanvas) {
                    drawLayerCanvas = document.createElement('canvas');
                    drawLayerCtx = drawLayerCanvas.getContext('2d');
                }
                if (drawBaseCanvas.width !== canvas.width || drawBaseCanvas.height !== canvas.height) {
                    drawBaseCanvas.width = canvas.width;
                    drawBaseCanvas.height = canvas.height;
                }
                if (drawLayerCanvas.width !== canvas.width || drawLayerCanvas.height !== canvas.height) {
                    drawLayerCanvas.width = canvas.width;
                    drawLayerCanvas.height = canvas.height;
                }
            };
            const renderDrawPreview = () => {
                if (!ctx || !drawBaseCanvas || !drawLayerCanvas) return;
                const alpha = Math.max(MARKER_OPACITY_MIN_ALPHA, Math.min(1, Number(markerState.opacity) || 0.6));
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(drawBaseCanvas, 0, 0);
                ctx.save();
                ctx.globalAlpha = alpha;
                ctx.drawImage(drawLayerCanvas, 0, 0);
                ctx.restore();
            };
            const applyMarkerBrush = () => {
                if (!drawLayerCtx) return;
                drawLayerCtx.strokeStyle = strokeStyle;
                drawLayerCtx.fillStyle = strokeStyle;
                drawLayerCtx.lineWidth = strokeSize;
                drawLayerCtx.lineCap = 'round';
                drawLayerCtx.lineJoin = 'round';
            };
            const appendStrokePoint = (point) => {
                if (!point) return false;
                if (strokePoints.length === 0) {
                    strokePoints.push(point);
                    return true;
                }
                const from = strokePoints[strokePoints.length - 1];
                const dx = point.x - from.x;
                const dy = point.y - from.y;
                const dist = Math.hypot(dx, dy);
                const minStep = Math.max(0.35, strokeSize * 0.04);
                if (dist < minStep) return false;
                const maxStep = Math.max(1, strokeSize * 0.25);
                const steps = Math.max(1, Math.ceil(dist / maxStep));
                for (let i = 1; i <= steps; i++) {
                    const t = i / steps;
                    strokePoints.push({
                        x: from.x + dx * t,
                        y: from.y + dy * t
                    });
                }
                return true;
            };
            const renderStrokeLayer = () => {
                if (!drawLayerCtx) return;
                drawLayerCtx.clearRect(0, 0, drawLayerCanvas.width, drawLayerCanvas.height);
                if (strokePoints.length === 0) return;
                applyMarkerBrush();
                if (strokePoints.length === 1) {
                    const p = strokePoints[0];
                    drawLayerCtx.beginPath();
                    drawLayerCtx.arc(p.x, p.y, strokeSize / 2, 0, Math.PI * 2);
                    drawLayerCtx.fill();
                    return;
                }
                drawLayerCtx.beginPath();
                drawLayerCtx.moveTo(strokePoints[0].x, strokePoints[0].y);
                if (strokePoints.length === 2) {
                    drawLayerCtx.lineTo(strokePoints[1].x, strokePoints[1].y);
                } else {
                    for (let i = 1; i < strokePoints.length - 2; i++) {
                        const p = strokePoints[i];
                        const n = strokePoints[i + 1];
                        const mid = getMid(p, n);
                        drawLayerCtx.quadraticCurveTo(p.x, p.y, mid.x, mid.y);
                    }
                    const secondLast = strokePoints[strokePoints.length - 2];
                    const last = strokePoints[strokePoints.length - 1];
                    drawLayerCtx.quadraticCurveTo(secondLast.x, secondLast.y, last.x, last.y);
                }
                drawLayerCtx.stroke();
            };
            const normalizeMosaicRect = (a, b) => {
                if (!a || !b) return null;
                const x = Math.min(a.x, b.x);
                const y = Math.min(a.y, b.y);
                const w = Math.abs(a.x - b.x);
                const h = Math.abs(a.y - b.y);
                return { x, y, w, h };
            };
            const buildMosaicRectFromPoint = (p) => {
                const size = sizeInput ? Number(sizeInput.value || 16) : 16;
                const side = Math.max(6, Math.floor(size));
                const half = Math.floor(side / 2);
                return { x: p.x - half, y: p.y - half, w: side, h: side };
            };
            const getMosaicSourceImageData = () => {
                const sourceCanvas = document.createElement('canvas');
                sourceCanvas.width = canvas.width;
                sourceCanvas.height = canvas.height;
                const sourceCtx = sourceCanvas.getContext('2d');
                if (!sourceCtx) return null;
                if (markerState.baseCanvas) {
                    sourceCtx.drawImage(markerState.baseCanvas, 0, 0);
                }
                sourceCtx.drawImage(canvas, 0, 0);
                try {
                    return sourceCtx.getImageData(0, 0, canvas.width, canvas.height);
                } catch (e) {
                    return null;
                }
            };
            const applyMosaicRect = (rect) => {
                if (!ctx || !rect) return false;
                const source = getMosaicSourceImageData();
                if (!source) return false;
                const size = sizeInput ? Number(sizeInput.value || 16) : 16;
                const block = Math.max(4, Math.floor(size / 2));
                const x1 = Math.max(0, Math.floor(rect.x));
                const y1 = Math.max(0, Math.floor(rect.y));
                const x2 = Math.min(canvas.width, Math.ceil(rect.x + rect.w));
                const y2 = Math.min(canvas.height, Math.ceil(rect.y + rect.h));
                if (x2 <= x1 || y2 <= y1) return false;
                for (let y = y1; y < y2; y += block) {
                    for (let x = x1; x < x2; x += block) {
                        const sw = Math.min(block, x2 - x);
                        const sh = Math.min(block, y2 - y);
                        const cx = Math.min(canvas.width - 1, Math.max(0, x + Math.floor(sw / 2)));
                        const cy = Math.min(canvas.height - 1, Math.max(0, y + Math.floor(sh / 2)));
                        const idx = (cy * canvas.width + cx) * 4;
                        const r = source.data[idx];
                        const g = source.data[idx + 1];
                        const b = source.data[idx + 2];
                        ctx.fillStyle = `rgb(${r},${g},${b})`;
                        ctx.fillRect(x, y, sw, sh);
                    }
                }
                return true;
            };
            const start = (e) => {
                if (!ctx) return;
                activePointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
                if (activePointers.size >= 2) {
                    const pts = Array.from(activePointers.values());
                    const p1 = pts[0];
                    const p2 = pts[1];
                    pinching = true;
                    drawing = false;
                    strokePoints = [];
                    mosaicSelecting = false;
                    mosaicStartPoint = null;
                    markerState.mosaicPreviewRect = null;
                    pinchStartDist = getDist(p1, p2) || 1;
                    pinchStartScale = markerView.scale;
                    pinchStartOffset = { x: markerView.offsetX, y: markerView.offsetY };
                    pinchStartMid = getMid(p1, p2);
                    renderCropOverlay();
                    if (canvas.setPointerCapture) canvas.setPointerCapture(e.pointerId);
                    e.preventDefault();
                    return;
                }
                if (pinching) return;
                if (markerState.mode === 'crop') return;
                drawing = true;
                const p = getPoint(e);
                if (markerState.mode === 'mosaic') {
                    mosaicSelecting = true;
                    mosaicStartPoint = p;
                    markerState.mosaicPreviewRect = buildMosaicRectFromPoint(p);
                    renderCropOverlay();
                } else {
                    ensureDrawBuffers();
                    if (!drawBaseCtx || !drawLayerCtx) return;
                    drawBaseCtx.clearRect(0, 0, drawBaseCanvas.width, drawBaseCanvas.height);
                    drawBaseCtx.drawImage(canvas, 0, 0);
                    drawLayerCtx.clearRect(0, 0, drawLayerCanvas.width, drawLayerCanvas.height);
                    strokeSize = sizeInput ? Number(sizeInput.value || 16) : 16;
                    strokeStyle = normalizeMarkerHexColor(markerState.colorHex);
                    strokePoints = [];
                    appendStrokePoint(p);
                    renderStrokeLayer();
                    markerState.hasStroke = true;
                    renderDrawPreview();
                }
                if (canvas.setPointerCapture) canvas.setPointerCapture(e.pointerId);
                e.preventDefault();
            };
            const move = (e) => {
                if (activePointers.has(e.pointerId)) {
                    activePointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
                }
                if (pinching && activePointers.size >= 2) {
                    const pts = Array.from(activePointers.values());
                    const p1 = pts[0];
                    const p2 = pts[1];
                    const mid = getMid(p1, p2);
                    const dist = getDist(p1, p2) || 1;
                    const nextScale = pinchStartScale * (dist / pinchStartDist);
                    markerView.scale = Math.min(markerView.maxScale, Math.max(markerView.minScale, nextScale));
                    markerView.offsetX = pinchStartOffset.x + (mid.x - pinchStartMid.x);
                    markerView.offsetY = pinchStartOffset.y + (mid.y - pinchStartMid.y);
                    applyMarkerTransform();
                    e.preventDefault();
                    return;
                }
                if (!drawing || !ctx) return;
                const p = getPoint(e);
                if (markerState.mode === 'mosaic') {
                    if (!mosaicSelecting || !mosaicStartPoint) return;
                    markerState.mosaicPreviewRect = normalizeMosaicRect(mosaicStartPoint, p) || buildMosaicRectFromPoint(p);
                    renderCropOverlay();
                } else {
                    const changed = appendStrokePoint(p);
                    if (changed) {
                        renderStrokeLayer();
                        renderDrawPreview();
                    }
                }
                e.preventDefault();
            };
            const end = (e) => {
                const wasDrawing = drawing;
                activePointers.delete(e.pointerId);
                if (activePointers.size < 2) {
                    pinching = false;
                }
                if (activePointers.size === 0) {
                    drawing = false;
                    if (
                        wasDrawing &&
                        ctx &&
                        markerState.mode === 'draw' &&
                        strokePoints.length > 0
                    ) {
                        renderStrokeLayer();
                        renderDrawPreview();
                    }
                    if (wasDrawing && markerState.mode === 'mosaic' && mosaicStartPoint) {
                        const p = getPoint(e);
                        let rect = normalizeMosaicRect(mosaicStartPoint, p);
                        if (!rect || rect.w < 2 || rect.h < 2) {
                            rect = buildMosaicRectFromPoint(mosaicStartPoint);
                        }
                        if (applyMosaicRect(rect)) {
                            markerState.hasStroke = true;
                            markerState.mosaicRects.push(rect);
                        }
                    }
                    strokePoints = [];
                    mosaicSelecting = false;
                    mosaicStartPoint = null;
                    markerState.mosaicPreviewRect = null;
                    renderCropOverlay();
                    if (wasDrawing) {
                        saveMarkerHistory();
                    }
                }
                if (canvas.releasePointerCapture) canvas.releasePointerCapture(e.pointerId);
                e.preventDefault();
            };
            canvas.addEventListener('pointerdown', start);
            canvas.addEventListener('pointermove', move);
            canvas.addEventListener('pointerup', end);
            canvas.addEventListener('pointercancel', end);
        }
        function initCropCanvas() {
            const cropCanvas = get('marker-crop-canvas');
            if (!cropCanvas) return;
            const ctx = cropCanvas.getContext('2d');
            const activePointers = new Map();
            let dragging = false;
            let dragStart = null;
            let dragMode = null;
            let startRect = null;
            let pinching = false;
            let pinchStartDist = 0;
            let pinchStartScale = markerView.scale;
            let pinchStartOffset = { x: 0, y: 0 };
            let pinchStartMid = { x: 0, y: 0 };
            const MIN_CROP_SIZE = 8;
            const HANDLE_RADIUS = 14;
            const clamp = (val, min, max) => Math.min(max, Math.max(min, val));
            const getPoint = (e) => {
                const rect = cropCanvas.getBoundingClientRect();
                const x = (e.clientX - rect.left) * (cropCanvas.width / rect.width);
                const y = (e.clientY - rect.top) * (cropCanvas.height / rect.height);
                return { x, y };
            };
            const getMid = (a, b) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });
            const getDist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
            const ensureCropRect = () => {
                if (!markerState.cropRect) {
                    resetCropRectToFull();
                }
                return markerState.cropRect;
            };
            const hitTest = (p, rect) => {
                if (!rect) return 'move';
                const x1 = rect.x;
                const y1 = rect.y;
                const x2 = rect.x + rect.w;
                const y2 = rect.y + rect.h;
                const nearLeft = Math.abs(p.x - x1) <= HANDLE_RADIUS;
                const nearRight = Math.abs(p.x - x2) <= HANDLE_RADIUS;
                const nearTop = Math.abs(p.y - y1) <= HANDLE_RADIUS;
                const nearBottom = Math.abs(p.y - y2) <= HANDLE_RADIUS;
                if (nearLeft && nearTop) return 'nw';
                if (nearRight && nearTop) return 'ne';
                if (nearLeft && nearBottom) return 'sw';
                if (nearRight && nearBottom) return 'se';
                if (nearTop) return 'n';
                if (nearBottom) return 's';
                if (nearLeft) return 'w';
                if (nearRight) return 'e';
                const inside = (p.x > x1 + HANDLE_RADIUS && p.x < x2 - HANDLE_RADIUS && p.y > y1 + HANDLE_RADIUS && p.y < y2 - HANDLE_RADIUS);
                if (inside) return 'move';
                const outsideX = p.x < x1 ? 'left' : (p.x > x2 ? 'right' : null);
                const outsideY = p.y < y1 ? 'top' : (p.y > y2 ? 'bottom' : null);
                if (outsideX && outsideY) {
                    if (outsideX === 'left' && outsideY === 'top') return 'nw';
                    if (outsideX === 'right' && outsideY === 'top') return 'ne';
                    if (outsideX === 'left' && outsideY === 'bottom') return 'sw';
                    if (outsideX === 'right' && outsideY === 'bottom') return 'se';
                }
                if (outsideX) return outsideX === 'left' ? 'w' : 'e';
                if (outsideY) return outsideY === 'top' ? 'n' : 's';
                return 'move';
            };
            const start = (e) => {
                if (markerState.mode !== 'crop') return;
                activePointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
                if (activePointers.size >= 2) {
                    const pts = Array.from(activePointers.values());
                    const p1 = pts[0];
                    const p2 = pts[1];
                    pinching = true;
                    dragging = false;
                    pinchStartDist = getDist(p1, p2) || 1;
                    pinchStartScale = markerView.scale;
                    pinchStartOffset = { x: markerView.offsetX, y: markerView.offsetY };
                    pinchStartMid = getMid(p1, p2);
                    if (cropCanvas.setPointerCapture) cropCanvas.setPointerCapture(e.pointerId);
                    e.preventDefault();
                    return;
                }
                if (pinching) return;
                dragging = true;
                const p = getPoint(e);
                const rect = ensureCropRect();
                dragMode = hitTest(p, rect);
                dragStart = p;
                startRect = rect ? { x: rect.x, y: rect.y, w: rect.w, h: rect.h } : null;
                renderCropOverlay();
                if (cropCanvas.setPointerCapture) cropCanvas.setPointerCapture(e.pointerId);
                e.preventDefault();
            };
            const move = (e) => {
                if (markerState.mode !== 'crop') return;
                if (activePointers.has(e.pointerId)) {
                    activePointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
                }
                if (pinching && activePointers.size >= 2) {
                    const pts = Array.from(activePointers.values());
                    const p1 = pts[0];
                    const p2 = pts[1];
                    const mid = getMid(p1, p2);
                    const dist = getDist(p1, p2) || 1;
                    const nextScale = pinchStartScale * (dist / pinchStartDist);
                    markerView.scale = Math.min(markerView.maxScale, Math.max(markerView.minScale, nextScale));
                    markerView.offsetX = pinchStartOffset.x + (mid.x - pinchStartMid.x);
                    markerView.offsetY = pinchStartOffset.y + (mid.y - pinchStartMid.y);
                    applyMarkerTransform();
                    renderCropOverlay();
                    e.preventDefault();
                    return;
                }
                if (!dragging || !dragStart || !startRect) return;
                const p = getPoint(e);
                const maxW = cropCanvas.width;
                const maxH = cropCanvas.height;
                const rect = { x: startRect.x, y: startRect.y, w: startRect.w, h: startRect.h };
                const right = startRect.x + startRect.w;
                const bottom = startRect.y + startRect.h;
                const applyW = () => {
                    const nx = clamp(p.x, 0, right - MIN_CROP_SIZE);
                    rect.x = nx;
                    rect.w = right - nx;
                };
                const applyE = () => {
                    rect.w = clamp(p.x - startRect.x, MIN_CROP_SIZE, maxW - startRect.x);
                };
                const applyN = () => {
                    const ny = clamp(p.y, 0, bottom - MIN_CROP_SIZE);
                    rect.y = ny;
                    rect.h = bottom - ny;
                };
                const applyS = () => {
                    rect.h = clamp(p.y - startRect.y, MIN_CROP_SIZE, maxH - startRect.y);
                };
                switch (dragMode) {
                    case 'move': {
                        const dx = p.x - dragStart.x;
                        const dy = p.y - dragStart.y;
                        rect.x = clamp(startRect.x + dx, 0, maxW - startRect.w);
                        rect.y = clamp(startRect.y + dy, 0, maxH - startRect.h);
                        break;
                    }
                    case 'w':
                        applyW();
                        break;
                    case 'e':
                        applyE();
                        break;
                    case 'n':
                        applyN();
                        break;
                    case 's':
                        applyS();
                        break;
                    case 'nw':
                        applyN();
                        applyW();
                        break;
                    case 'ne':
                        applyN();
                        applyE();
                        break;
                    case 'sw':
                        applyS();
                        applyW();
                        break;
                    case 'se':
                        applyS();
                        applyE();
                        break;
                    default:
                        break;
                }
                rect.x = clamp(rect.x, 0, maxW - rect.w);
                rect.y = clamp(rect.y, 0, maxH - rect.h);
                markerState.cropRect = rect;
                renderCropOverlay();
                e.preventDefault();
            };
            const end = (e) => {
                activePointers.delete(e.pointerId);
                if (activePointers.size < 2) {
                    pinching = false;
                }
                if (activePointers.size === 0) {
                    renderCropOverlay();
                    dragging = false;
                    dragStart = null;
                    dragMode = null;
                    startRect = null;
                }
                if (cropCanvas.releasePointerCapture) cropCanvas.releasePointerCapture(e.pointerId);
                e.preventDefault();
            };
            cropCanvas.addEventListener('pointerdown', start);
            cropCanvas.addEventListener('pointermove', move);
            cropCanvas.addEventListener('pointerup', end);
            cropCanvas.addEventListener('pointercancel', end);
            cropCanvas.addEventListener('pointerleave', end);
        }
        async function saveMarkerToRow() {
            const row = markerState.row;
            const img = get('marker-image');
            const canvas = get('marker-canvas');
            if (!row || !img || !canvas) return;
            const attachOriginal = get('marker-attach-original');
            if (attachOriginal) {
                row.dataset.attachOriginal = attachOriginal.checked ? '1' : '';
            }
            let out = document.createElement('canvas');
            const w = markerState.naturalWidth || img.naturalWidth || canvas.width;
            const h = markerState.naturalHeight || img.naturalHeight || canvas.height;
            out.width = w;
            out.height = h;
            const octx = out.getContext('2d');
            if (!octx) return;
            octx.drawImage(img, 0, 0, w, h);
            octx.drawImage(canvas, 0, 0, w, h);
            if (markerState.cropRect) {
                const scaleX = w / canvas.width;
                const scaleY = h / canvas.height;
                const cx = Math.max(0, Math.floor(markerState.cropRect.x * scaleX));
                const cy = Math.max(0, Math.floor(markerState.cropRect.y * scaleY));
                const cw = Math.min(w, Math.max(1, Math.floor(markerState.cropRect.w * scaleX)));
                const ch = Math.min(h, Math.max(1, Math.floor(markerState.cropRect.h * scaleY)));
                const cropped = document.createElement('canvas');
                cropped.width = cw;
                cropped.height = ch;
                const cctx = cropped.getContext('2d');
                if (cctx) {
                    cctx.drawImage(out, cx, cy, cw, ch, 0, 0, cw, ch);
                    out = cropped;
                }
            }
            const blob = await new Promise((resolve) => out.toBlob(resolve, 'image/png', 0.92));
            if (!blob) {
                showToast("編集画像の生成に失敗しました", "error", true);
                return;
            }
            const originalName = markerState.filename || 'marked.png';
            const base = originalName.replace(/\.[^/.]+$/, '');
            const file = new File([blob], `${base}_marked.png`, { type: 'image/png' });
            const rowObj = {
                row,
                uploadId: row.dataset.uploadId,
                status: row.querySelector('.upload-status'),
                bar: row.querySelector('.upload-progress > div')
            };
            if (rowObj.status) rowObj.status.textContent = '編集反映中...';
            updateUploadRowFile(rowObj, file);
            const prevFilename = row.getAttribute('data-filename');
            const prevSource = getRowAttachmentSource(row);
            if (prevFilename && !row.dataset.originalFilename) {
                row.dataset.originalFilename = prevFilename;
                row.dataset.originalSource = prevSource;
                setAttachmentSourceForPath(prevFilename, prevSource);
            }
            const success = await uploadFileWithProgress(file, rowObj);
            if (success) {
                if (prevFilename) currentImageUrls = currentImageUrls.filter(x => x !== prevFilename);
                setRowAttachmentSource(row, 'upload');
                setRowMarkerState(row, true);
            } else {
                showToast("編集画像のアップロードに失敗しました", "error", true);
            }
            updateFilePreview();
            window.closeMarkerModal();
            markerState.row = null;
        }
        async function extractAudioFromVideo(file, row) {
            if (!isVideoFile(file)) return null;
            if (!HTMLMediaElement.prototype.captureStream) {
                return null;
            }
            if (row && row.status) row.status.textContent = '音声抽出中...';
            return new Promise((resolve) => {
                const video = document.createElement('video');
                video.preload = 'auto';
                video.muted = true;
                video.playsInline = true;
                video.src = URL.createObjectURL(file);
                let stream = null;
                let audioCtx = null;
                let processor = null;
                let source = null;
                let buffers = [];
                let timeoutId = null;

                const cleanup = () => {
                    if (timeoutId) clearTimeout(timeoutId);
                    try { URL.revokeObjectURL(video.src); } catch (e) {}
                    try { video.remove(); } catch (e) {}
                    if (stream) {
                        stream.getTracks().forEach(t => t.stop());
                    }
                    if (processor) {
                        try { processor.disconnect(); } catch (e) {}
                    }
                    if (source) {
                        try { source.disconnect(); } catch (e) {}
                    }
                    if (audioCtx) {
                        try { audioCtx.close(); } catch (e) {}
                    }
                };
                const fail = () => {
                    cleanup();
                    resolve(null);
                };

                video.onloadedmetadata = async () => {
                    try {
                        stream = video.captureStream();
                        const audioTracks = stream.getAudioTracks();
                        if (!audioTracks || !audioTracks.length) return fail();
                        audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
                        source = audioCtx.createMediaStreamSource(new MediaStream(audioTracks));
                        processor = audioCtx.createScriptProcessor(4096, 1, 1);
                        processor.onaudioprocess = (e) => {
                            const input = e.inputBuffer.getChannelData(0);
                            buffers.push(new Float32Array(input));
                        };
                        source.connect(processor);
                        processor.connect(audioCtx.destination);
                        const durationMs = isFinite(video.duration) ? Math.max(1, Math.ceil(video.duration * 1000)) : 0;
                        if (durationMs > 0) {
                            timeoutId = setTimeout(() => {
                                const baseName = (file.name || 'video').replace(/\.[^/.]+$/, '');
                                const wavBlob = encodeWav(buffers, audioCtx.sampleRate);
                                const audioFile = new File([wavBlob], `${baseName}.audio.wav`, { type: 'audio/wav' });
                                cleanup();
                                resolve(audioFile);
                            }, durationMs + 250);
                        }
                        await video.play();
                        video.onended = () => {
                            const baseName = (file.name || 'video').replace(/\.[^/.]+$/, '');
                            const wavBlob = encodeWav(buffers, audioCtx.sampleRate);
                            const audioFile = new File([wavBlob], `${baseName}.audio.wav`, { type: 'audio/wav' });
                            cleanup();
                            resolve(audioFile);
                        };
                    } catch (e) {
                        fail();
                    }
                };
                video.onerror = () => fail();
            });
        }
        async function handleFiles(fs, opts = {}) {
            if(!fs || !fs.length) return;
            const incoming = Array.from(fs).filter(Boolean);
            if (!incoming.length) return;
            const existingAttachments = collectImageUrlsForSend().length + browserFastLocalFiles.size + Math.max(0, Number(uploadProgressState.active) || 0);
            let allowedIncoming = incoming;
            if (existingAttachments + incoming.length > ATTACHMENT_MAX_FILES) {
                const remain = Math.max(0, ATTACHMENT_MAX_FILES - existingAttachments);
                if (remain <= 0) {
                    showToast(`添付は最大${ATTACHMENT_MAX_FILES}件です`, "error", true);
                    return;
                }
                allowedIncoming = incoming.slice(0, remain);
                showToast(`添付は最大${ATTACHMENT_MAX_FILES}件です。先頭${remain}件のみ追加します。`, "warning", true);
            }
            if (opts.openModal !== false) {
                openUploadModal();
            } else {
                syncUploadRowsFromCurrent();
            }

            uploadProgressState.total += allowedIncoming.length;
            uploadProgressState.active += allowedIncoming.length;
            updateFilePreview();

            const audioOnly = !!(get('upload-audio-only') && get('upload-audio-only').checked);
            const support = getModelMediaSupport(get('model-select').value);
            const processOne = async (f) => {
                let currentUploadId = null;
                try {
                    if (isAudioFile(f) && !support.audio) {
                        showToast("このモデルは音声入力に対応していません", "error", true);
                        if (uploadProgressState.total > 0) uploadProgressState.total--;
                        if (uploadProgressState.active > 0) uploadProgressState.active--;
                        return false;
                    }
                    if (isVideoFile(f) && !support.video) {
                        showToast("このモデルは動画入力に対応していません", "error", true);
                        if (uploadProgressState.total > 0) uploadProgressState.total--;
                        if (uploadProgressState.active > 0) uploadProgressState.active--;
                        return false;
                    }
                    if (browserFastModeEnabled && (!f.type || !f.type.startsWith('image/'))) {
                        showToast('高速モードでは画像ファイルだけを添付できます', 'error', true);
                        if (uploadProgressState.total > 0) uploadProgressState.total--;
                        if (uploadProgressState.active > 0) uploadProgressState.active--;
                        return false;
                    }
                    const rowObj = addUploadRow(f);
                    updateFilePreview(); // Immediate feedback
                    currentUploadId = rowObj.uploadId;
                    uploadProgressState.perFilePct[currentUploadId] = 0;

                    let t = f;
                    if (audioOnly && isVideoFile(f)) {
                        const audioFile = await extractAudioFromVideo(f, rowObj);
                        if (audioFile) {
                            t = audioFile;
                            updateUploadRowFile(rowObj, audioFile);
                            if (rowObj && rowObj.status) rowObj.status.textContent = '音声のみ';
                        } else {
                            if (rowObj && rowObj.status) rowObj.status.textContent = '抽出失敗: 動画送信';
                            showToast("音声抽出に失敗しました。動画のまま送信します。", "error", true);
                        }
                    }
                    if (get('enable-compression').checked && f.type.startsWith('image/')) {
                        try {
                            const outputType = getCompressionOutputType();
                            const formatOnly = getCompressionFormatOnly();
                            if (formatOnly) {
                                t = await convertImageFormatOnly(f, outputType);
                            } else {
                                const o = {
                                    maxSizeMB: getCompressionMaxSizeMB(),
                                    maxWidthOrHeight: getCompressionMaxDim(),
                                    useWebWorker: true
                                };
                                if (outputType && outputType !== 'original') o.fileType = outputType;
                                await ensureImageCompression();
                                const c = await window.imageCompression(f, o);
                                const compressedFile = new File(
                                    [c],
                                    imageFilenameForMime(f.name, c.type || (outputType !== 'original' ? outputType : f.type)),
                                    { type: c.type || f.type, lastModified: f.lastModified || Date.now() }
                                );
                                if (compressedFile.size > f.size) {
                                    showToast(`圧縮後にサイズが増加しました: ${formatBytes(f.size)} -> ${formatBytes(compressedFile.size)}（元ファイルを使用）`, "warning", true);
                                    t = f;
                                } else {
                                    t = compressedFile;
                                }
                            }
                            if (t !== f) updateUploadRowFile(rowObj, t);
                        } catch(e){}
                    }
                    if (browserFastModeEnabled) {
                        const existingBytes = Array.from(browserFastLocalFiles.values())
                            .reduce((sum, entry) => sum + Number(entry.file && entry.file.size || 0), 0);
                        if (browserFastLocalFiles.size >= BROWSER_FAST_MAX_IMAGES || existingBytes + t.size > BROWSER_FAST_MAX_BYTES) {
                            if (rowObj && rowObj.status) rowObj.status.textContent = '上限超過';
                            if (rowObj && rowObj.row) rowObj.row.remove();
                            showToast('高速モードの画像は4枚・合計12MBまでです', 'error', true);
                            return false;
                        }
                        browserFastLocalFiles.set(rowObj.uploadId, { file: t, rowObj });
                        if (rowObj.status) rowObj.status.textContent = 'ローカル保持（未保存）';
                        if (rowObj.bar) rowObj.bar.style.width = '100%';
                        if (rowObj.row) rowObj.row.dataset.browserFastLocal = '1';
                        return true;
                    }
                    return await uploadFileWithProgress(t, rowObj);
                } finally {
                    if (currentUploadId) {
                        if (uploadProgressState.perFilePct.hasOwnProperty(currentUploadId)) {
                            delete uploadProgressState.perFilePct[currentUploadId];
                            uploadProgressState.completed++;
                            uploadProgressState.active--;
                        }
                    }
                    if (uploadProgressState.active <= 0) {
                        uploadProgressState.total = 0;
                        uploadProgressState.completed = 0;
                        uploadProgressState.active = 0;
                        uploadProgressState.perFilePct = {};
                    }
                    updateFilePreview();
                }
            };
            let cursor = 0;
            const workerCount = Math.min(UPLOAD_CONCURRENCY, allowedIncoming.length);
            const workers = Array.from({ length: workerCount }).map(async () => {
                while (true) {
                    const idx = cursor++;
                    if (idx >= allowedIncoming.length) break;
                    await processOne(allowedIncoming[idx]);
                }
            });
            await Promise.all(workers);
        }
        get('clear-file-btn').onclick = () => { resetUploadState(); };
        if (get('clear-mask-btn')) {
            get('clear-mask-btn').onclick = () => {
                currentMaskImage = null;
                updateMaskPreview();
            };
        }
        if (get('mask-btn') && get('mask-input')) {
            get('mask-btn').onclick = () => {
                get('mask-input').click();
            };
            get('mask-input').addEventListener('change', async (e) => {
                const f = e.target.files && e.target.files[0];
                if (!f) return;
                await uploadMaskFile(f);
                e.target.value = '';
            });
        }

        const messageMeta = {};
        let markdownLibraryFallbackReported = false;
        function sanitizeMarkdownHtml(text, opts = {}) {
            const source = String(text || '');
            if (!window.marked || typeof window.marked.parse !== 'function'
                || !window.DOMPurify || typeof window.DOMPurify.sanitize !== 'function') {
                if (!markdownLibraryFallbackReported) {
                    markdownLibraryFallbackReported = true;
                    console.error('Markdown sanitizer is unavailable; rendering escaped plain text.');
                }
                return escapeHtml(source).replace(/\n/g, '<br>');
            }
            // marked が \( \[ のバックスラッシュを落とすため、数式を退避してから parse する
            const protectedMath = protectMathSegments(source);
            const parsed = window.marked.parse(protectedMath.text);
            const restored = restoreMathSegments(parsed, protectedMath.blocks, opts);
            return window.DOMPurify.sanitize(restored);
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
