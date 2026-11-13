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
            let autoFillPoint = null;
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
                const alpha = markerState.maskMode ? 1 : Math.max(MARKER_OPACITY_MIN_ALPHA, Math.min(1, Number(markerState.opacity) || 0.6));
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
                    autoFillPoint = null;
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
                if (markerState.mode === 'auto' && markerState.maskMode) {
                    autoFillPoint = p;
                } else if (markerState.mode === 'mosaic') {
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
                    strokeStyle = markerState.maskMode ? '#ffffff' : normalizeMarkerHexColor(markerState.colorHex);
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
                if (markerState.mode === 'auto' && markerState.maskMode) {
                    e.preventDefault();
                    return;
                }
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
                    const autoFillApplied = wasDrawing && e.type !== 'pointercancel' && markerState.mode === 'auto'
                        && markerState.maskMode && autoFillPoint ? fillMaskSimilarRegion(autoFillPoint) : false;
                    autoFillPoint = null;
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
                    if (wasDrawing && (markerState.mode !== 'auto' || autoFillApplied)) {
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
        function fillMaskSimilarRegion(point) {
            const canvas = get('marker-canvas');
            const source = markerState.baseImageData;
            if (!canvas || !source) {
                showToast("画像の色を読み取れませんでした", "error", true);
                return false;
            }
            const width = source.width;
            const height = source.height;
            const seedX = Math.floor(point.x);
            const seedY = Math.floor(point.y);
            if (seedX < 0 || seedY < 0 || seedX >= width || seedY >= height) return false;
            const pixels = source.data;
            const seedIndex = seedY * width + seedX;
            const colorOffset = seedIndex * 4;
            if (pixels[colorOffset + 3] < 16) return false;
            const red = pixels[colorOffset];
            const green = pixels[colorOffset + 1];
            const blue = pixels[colorOffset + 2];
            const tolerance = Number(get('marker-auto-tolerance')?.value || 32);
            const visited = new Uint8Array(width * height);
            const pending = [seedIndex];
            const spans = [];
            const MAX_SPANS = 250000;
            const matches = (index) => {
                const offset = index * 4;
                return pixels[offset + 3] >= 16
                    && Math.max(Math.abs(pixels[offset] - red), Math.abs(pixels[offset + 1] - green), Math.abs(pixels[offset + 2] - blue)) <= tolerance;
            };
            visited[seedIndex] = 1;
            while (pending.length) {
                const index = pending.pop();
                if (visited[index] === 2 || !matches(index)) continue;
                const y = Math.floor(index / width);
                let left = index % width;
                let right = left;
                while (left > 0 && visited[y * width + left - 1] !== 2 && matches(y * width + left - 1)) left--;
                while (right + 1 < width && visited[y * width + right + 1] !== 2 && matches(y * width + right + 1)) right++;
                for (let x = left; x <= right; x++) visited[y * width + x] = 2;
                spans.push(y, left, right + 1);
                if (spans.length / 3 > MAX_SPANS) {
                    showToast("範囲が複雑です。許容差を下げてください", "error", true);
                    return false;
                }
                for (const nextY of [y - 1, y + 1]) {
                    if (nextY < 0 || nextY >= height) continue;
                    let x = left;
                    while (x <= right) {
                        const nextIndex = nextY * width + x;
                        if (visited[nextIndex] === 0 && matches(nextIndex)) {
                            visited[nextIndex] = 1;
                            pending.push(nextIndex);
                            while (x + 1 <= right && visited[nextY * width + x + 1] === 0 && matches(nextY * width + x + 1)) {
                                x++;
                                visited[nextY * width + x] = 1;
                            }
                        }
                        x++;
                    }
                }
            }
            if (!spans.length) return false;
            const ctx = canvas.getContext('2d');
            if (!ctx) return false;
            const imageData = ctx.getImageData(0, 0, width, height);
            const data = imageData.data;
            for (let i = 0; i < spans.length; i += 3) {
                const offset = spans[i] * width + spans[i + 1];
                const end = spans[i] * width + spans[i + 2];
                for (let pixel = offset; pixel < end; pixel++) {
                    const p = pixel * 4;
                    data[p] = data[p + 1] = data[p + 2] = data[p + 3] = 255;
                }
            }
            ctx.putImageData(imageData, 0, 0);
            markerState.hasStroke = true;
            return true;
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
            if (markerState.maskMode) {
                if (!markerState.hasStroke) {
                    showToast("削除する範囲を白く塗ってください", "error", true);
                    return;
                }
                const mask = document.createElement('canvas');
                mask.width = w;
                mask.height = h;
                const maskCtx = mask.getContext('2d');
                if (!maskCtx) return;
                maskCtx.fillStyle = '#000000';
                maskCtx.fillRect(0, 0, w, h);
                maskCtx.drawImage(canvas, 0, 0, w, h);
                const blob = await new Promise((resolve) => mask.toBlob(resolve, 'image/png'));
                if (!blob) {
                    showToast("マスク画像の生成に失敗しました", "error", true);
                    return;
                }
                const base = (markerState.filename || 'image').replace(/\.[^/.]+$/, '');
                const uploaded = await uploadMaskFile(new File([blob], `${base}_mask.png`, { type: 'image/png' }));
                if (uploaded) {
                    window.closeMarkerModal();
                    markerState.row = null;
                    markerState.maskMode = false;
                }
                return;
            }
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
            markerState.maskMode = false;
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
            const allowedIncoming = incoming;
            const generation = uploadGeneration;
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
                if (generation !== uploadGeneration) return false;
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
                    rowObj.generation = generation;
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
                    if (!opts.skipCompression && f.type.startsWith('image/')) {
                        try {
                            t = await applyImageCompression(f, readComposerCompression());
                            if (t !== f) updateUploadRowFile(rowObj, t);
                        } catch(e){}
                    }
                    if (isUploadCancelled(rowObj)) return false;
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
                    const success = await uploadFileWithProgress(t, rowObj, opts);
                    if (!success && opts.cameraCapture && !isUploadCancelled(rowObj)) {
                        // Keep the captured bytes reachable until the user retries or removes the row.
                        const retry = document.createElement('button');
                        retry.className = 'text-xs border border-gray-700 rounded px-2 py-1';
                        retry.textContent = '再試行';
                        retry.onclick = () => {
                            if (isUploadCancelled(rowObj)) return;
                            retry.disabled = true;
                            const localUrl = rowObj.row.getAttribute('data-local-url');
                            if (localUrl) URL.revokeObjectURL(localUrl);
                            rowObj.row.remove();
                            handleFiles([t], { ...opts, openModal: true, skipCompression: true });
                        };
                        rowObj.row.appendChild(retry);
                    }
                    return success;
                } finally {
                    if (generation === uploadGeneration) {
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
                }
            };
            const results = new Array(allowedIncoming.length).fill(false);
            let cursor = 0;
            const workerCount = Math.min(UPLOAD_CONCURRENCY, allowedIncoming.length);
            const workers = Array.from({ length: workerCount }).map(async () => {
                while (true) {
                    const idx = cursor++;
                    if (idx >= allowedIncoming.length) break;
                    results[idx] = await processOne(allowedIncoming[idx]);
                }
            });
            await Promise.all(workers);
            return results;
        }
        get('clear-file-btn').onclick = (e) => { e.stopPropagation(); resetUploadState(); };
        if (get('clear-mask-btn')) {
            get('clear-mask-btn').onclick = () => {
                currentMaskImage = null;
                updateMaskPreview();
            };
        }
        if (get('mask-btn') && get('mask-input')) {
            get('mask-btn').onclick = () => {
                if (isIdeogramObjectRemoval()) {
                    const row = Array.from(document.querySelectorAll('#upload-list .upload-row'))
                        .find((candidate) => getRowImageSource(candidate));
                    if (!row) {
                        showToast("先にマスク対象の画像を添付してください", "error", true);
                        return;
                    }
                    openMarkerModalForRow(row, true);
                    return;
                }
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
            const inline = opts.inline === true && typeof window.marked.parseInline === 'function';
            if (inline) {
                // 1行のプレビューでも箇条書きや見出しの記号をそのまま出さず、強調・
                // インラインコード・数式だけを描けるよう、ブロック要素を作らない。
                return window.DOMPurify.sanitize(restoreMathSegments(
                    window.marked.parseInline(protectedMath.text), protectedMath.blocks, opts));
            }
            // ストリーミング中の閉じていない SVG コードブロックは描画しない（コードの renderer が参照する）
            markdownOpenFenceBody = /<\/svg/i.test(source) ? findOpenMarkdownFenceBody(source) : null;
            let parsed = '';
            try {
                parsed = window.marked.parse(protectedMath.text);
            } finally {
                markdownOpenFenceBody = null;
            }
            const restored = restoreMathSegments(parsed, protectedMath.blocks, opts);
            return window.DOMPurify.sanitize(restored);
        }
        // Body of the fenced block still open at the end of the text (an answer that is streaming), or null.
        let markdownOpenFenceBody = null;
        function findOpenMarkdownFenceBody(text) {
            const lines = String(text || '').split(/\r?\n/);
            let fence = null;
            let body = [];
            for (const line of lines) {
                if (!fence) {
                    const match = line.match(/^\s*(`{3,}|~{3,})(.*)$/);
                    if (match && !(match[1][0] === '`' && match[2].includes('`'))) {
                        fence = match[1];
                        body = [];
                    }
                    continue;
                }
                const trimmed = line.trim();
                if (trimmed.length >= fence.length && trimmed[0] === fence[0] && /^(`+|~+)$/.test(trimmed)) {
                    fence = null;
                    continue;
                }
                body.push(line);
            }
            return fence ? body.join('\n') : null;
        }
        const SVG_CODE_RENDER_MAX_CHARS = 300000;
        // A closed ```svg (or ```xml / ```mathsvg holding one <svg> document) fence: the markup to draw, or ''.
        function getRenderableSvgCode(lang, raw) {
            const token = String(lang || '').trim().toLowerCase();
            if (token !== 'svg' && token !== 'xml' && token !== 'image/svg+xml' && token !== 'mathsvg') return '';
            const code = String(raw || '').trim();
            if (!code || code.length > SVG_CODE_RENDER_MAX_CHARS) return '';
            const body = code.replace(/^(?:<\?xml[\s\S]*?\?>\s*|<!--[\s\S]*?-->\s*|<!DOCTYPE[^>]*>\s*)*/i, '');
            if (!/^<svg[\s>]/i.test(body) || !/<\/svg\s*>$/i.test(body)) return '';
            if (markdownOpenFenceBody !== null
                && markdownOpenFenceBody.replace(/\s+/g, '') === code.replace(/\s+/g, '')) return '';
            return code;
        }
        // Drawn through <img>, so scripts, external references, styles and ids stay inside the drawing.
