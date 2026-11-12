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
                if (get('camera-split-enabled').checked) {
                    await openImageSplit(filesToUpload);
                } else {
                    const results = await handleFiles(filesToUpload, { openModal: false, cameraCapture: true });
                    const attached = results.filter(Boolean).length;
                    if (attached > 0) showToast(`${attached}枚の画像を添付しました`, 'success');
                    if (attached < filesToUpload.length) {
                        showToast('添付できなかった画像はアップロード画面から再試行できます。', 'warning', true);
                        openUploadModal();
                    }
                }
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
            if (typeof window.hideDropOverlay === 'function') {
                window.hideDropOverlay();
            }
            syncUploadRowsFromCurrent();
            showModal('upload-modal');
            if (location.pathname !== '/upload') {
                history.pushState({ modal: 'upload' }, '', '/upload');
            }
            const vmi = get('vision-model-info');
            if (vmi) {
                const model = get('model-select') ? get('model-select').value : '';
                const uploadModelLower = model.toLowerCase();
                // V4.1 Flash accepts images natively, so the vision-model notice is not needed.
                const deepseekNativeVision = uploadModelLower === 'deepseek-v4.1-flash' || uploadModelLower === 'deepseek-v4-flash-vision-exp';
                const glmNativeVision = ['glm-5.3-flash', 'glm-5.3-flashx', 'glm-4.6v', 'glm-4.6v-flashx', 'glm-4.6v-flash', 'glm-4.5v'].includes(uploadModelLower);
                const needsVisionNotice = (uploadModelLower.includes('deepseek') && !deepseekNativeVision)
                    || (uploadModelLower.startsWith('glm-') && !glmNativeVision);
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
            if (typeof window.hideDropOverlay === 'function') {
                window.hideDropOverlay();
            }
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
            if (isUploadCancelled(row)) return false;
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
                    if (isUploadCancelled(row)) {
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
        function uploadFileWithProgress(file, row, opts = {}) {
            return new Promise((resolve) => {
                if (file && file.size > CHUNK_THRESHOLD_BYTES) {
                    uploadFileChunked(file, row).then(resolve);
                    return;
                }
                if (isUploadCancelled(row)) { resolve(false); return; }
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
                        if (isUploadCancelled(row)) {
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
                        // Camera JPEGs can be rejected by the proxy before /upload reaches Flask.
                        // Try the existing chunk transport once; application errors stay final.
                        const proxyFailure = [502, 503, 504].includes(xhr.status) || (xhr.status >= 520 && xhr.status <= 527);
                        if (opts.cameraCapture && proxyFailure && !d.error && !isUploadCancelled(row)) {
                            if (row && row.status) row.status.textContent = '分割して再送中...';
                            finishUploadOp();
                            uploadFileChunked(file, row).then(resolve);
                            return;
                        }
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
