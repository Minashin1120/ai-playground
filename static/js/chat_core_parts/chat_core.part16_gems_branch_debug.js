        function openLibraryImage(f) {
            if (!lib.files) return;
            const ordered = sortLibraryFiles(lib.files);
            const q = getLibSearchQuery();
            const filtered = q ? ordered.filter((x) => fileNameForSearch(x).includes(q)) : ordered;
            const images = filtered.filter((x) => x.type === 'image');
            if (!images.length) return;
            const items = images.map((x) => ({
                url: x.url,
                filename: x.filename || x.original_filename || x.url.split('/').pop(),
                element: null
            }));
            let idx = items.findIndex((x) => x.url === f.url);
            if (idx === -1) idx = 0;
            openViewerWithItems(items, idx);
        }
        function libraryFileIcon(ext) {
            const safe = {
                pdf: 'fa-file-pdf',
                image: 'fa-image',
                file: 'fa-file'
            };
            const e = String(ext || '').toLowerCase();
            if (e === 'pdf') return safe.pdf;
            if (['png','jpg','jpeg','gif','webp','bmp','svg','heic'].includes(e)) return safe.image;
            return safe.file;
        }
        function renderLibraryItem(f, i = 0) {
            const el = document.createElement('div');
            el.className = 'library-thumb-card';
            if (i !== null && i !== undefined) el.style.animationDelay = `${Math.min(i * 0.035, 0.45)}s`;
            const thumbSrc = f.thumbnail_url || f.thumb_url || f.url;
            const extName = String(f.ext || (f.filename || '').split('.').pop() || '').toLowerCase();
            const media = f.type === 'image'
                ? `<img src="${escapeHtml(thumbSrc)}" alt="${escapeHtml(f.filename)}" loading="lazy" decoding="async" class="library-thumb-media">`
                : `<div class="library-thumb-file"><div class="lib-file-icon"><i class="fas ${libraryFileIcon(extName)}"></i></div><span class="lib-file-badge">${escapeHtml(extName ? extName.toUpperCase() : 'FILE')}</span></div>`;
            const overlay = `<div class="lib-overlay"><a href="${escapeHtml(f.url)}" download="${escapeHtml(f.filename)}" class="lib-overlay-btn" onclick="event.stopPropagation()" title="ダウンロード"><i class="fas fa-download"></i></a></div>`;
            const actions = `<div class="lib-thumb-actions"><button class="lib-open-btn lib-action-circle" title="開く"><i class="fas fa-eye"></i></button><button class="lib-del-btn lib-action-circle lib-del" title="削除"><i class="fas fa-trash"></i></button></div>`;
            const bar = `<div class="lib-thumb-bar"><span class="lib-thumb-name" title="${escapeHtml(f.filename)}">${escapeHtml(f.filename)}</span></div>`;
            el.innerHTML = `<div class="lib-thumb-media-wrap">${media}</div>${overlay}${actions}${bar}`;
            el.onclick = () => {
                if (lib.selected.has(f.filepath)) {
                    lib.selected.delete(f.filepath);
                    el.classList.remove('is-selected');
                } else {
                    lib.selected.add(f.filepath);
                    el.classList.add('is-selected');
                }
                window.updateLibSelectionUi();
            };
            if (lib.selected && lib.selected.has(f.filepath)) {
                el.classList.add('is-selected');
            }
            const openBtns = el.querySelectorAll('.lib-open-btn');
            openBtns.forEach((btn) => {
                btn.onclick = (e) => {
                    e.stopPropagation();
                    if (f.type === 'image') {
                        openLibraryImage(f);
                    } else {
                        openFileViewer(f.url, f.filename);
                    }
                };
            });
            const delBtn = el.querySelector('.lib-del-btn');
            if (delBtn) {
                delBtn.onclick = async (e) => {
                    e.stopPropagation();
                    await deleteSingleLibraryFile(f.filepath, el);
                };
            }
            return el;
        }
        function renderLibrarySkeleton(grid) {
            if (!grid) return;
            grid.innerHTML = '';
            for (let i = 0; i < 12; i++) {
                const card = document.createElement('div');
                card.className = 'lib-skeleton-card';
                card.style.animationDelay = `${Math.min(i * 0.04, 0.5)}s`;
                card.innerHTML = '<div class="lib-skeleton-thumb"></div><div class="lib-skeleton-bar"><span class="lib-skeleton-line" style="width:78%"></span><span class="lib-skeleton-line" style="width:45%"></span></div>';
                grid.appendChild(card);
            }
        }
        function addLibraryFileFromPath(filepath) {
            if (!filepath) return;
            if (!lib.fileSet) lib.fileSet = new Set();
            if (lib.fileSet.has(filepath)) return;
            const filename = filepath.split('/').pop() || filepath;
            const ext = (filename.split('.').pop() || '').toLowerCase();
            const type = ['png','jpg','jpeg','webp','gif'].includes(ext) ? 'image' : 'file';
            const url = FILE_BASE_URL + filepath;
            const thumbnail_url = type === 'image' ? (FILE_THUMB_BASE_URL + filepath) : null;
            const f = { filename, original_filename: filename, filepath, url, thumbnail_url, type, ext, ts: Math.floor(Date.now() / 1000) };
            setAttachmentNameForPath(filepath, filename);
            lib.fileSet.add(filepath);
            if (!lib.files) lib.files = [];
            lib.files.unshift(f);
            const grid = get('lib-grid');
            if (grid && lib.modal && lib.modal.classList.contains('modal-open')) {
                renderLibraryGrid();
            }
        }
        async function renameSelectedLibraryFile() {
            if (!lib.selected || lib.selected.size !== 1) return;
            const filepath = Array.from(lib.selected)[0];
            const item = (lib.files || []).find((f) => f.filepath === filepath);
            const currentName = (item && item.filename) || (filepath.split('/').pop() || filepath);
            const nextNameRaw = prompt('新しいファイル名を入力してください', currentName);
            if (nextNameRaw === null) return;
            const nextName = (nextNameRaw || '').trim();
            if (!nextName) {
                showToast('ファイル名を入力してください', 'error', true);
                return;
            }
            try {
                const r = await apiFetch(CHAT_CONFIG.urls.renameLibraryFile, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ filepath, filename: nextName })
                });
                const d = await r.json().catch(() => ({}));
                if (!r.ok) {
                    showToast(d.error || '名前変更に失敗しました', 'error', true);
                    return;
                }
                if (item) {
                    item.filename = d.filename || nextName;
                    setAttachmentNameForPath(filepath, item.filename);
                }
                const uploadList = get('upload-list');
                if (uploadList) {
                    uploadList.querySelectorAll('[data-filename]').forEach((row) => {
                        if (row.getAttribute('data-filename') === filepath) {
                            setRowAttachmentName(row, item ? item.filename : (d.filename || nextName));
                        }
                    });
                }
                renderLibraryGrid();
                window.updateLibSelectionUi();
                showToast('ファイル名を変更しました', 'success');
            } catch (e) {
                showToast('名前変更に失敗しました', 'error', true);
            }
        }
        async function deleteSingleLibraryFile(filepath, el) {
            if (!filepath) return;
            if (!confirm('削除しますか？')) return;
            try {
                await apiFetch(CHAT_CONFIG.urls.deleteFilesBatch, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({filenames:[filepath]})});
                if (el && el.parentNode) el.remove();
                if (lib.files) lib.files = lib.files.filter(f => f.filepath !== filepath);
                if (lib.fileSet) lib.fileSet.delete(filepath);
                lib.selected.delete(filepath);
                renderLibraryGrid();
                window.updateLibSelectionUi();
            } catch(e) {
                showToast("削除に失敗しました", "error", true);
            }
        }
        async function loadLibraryFiles() {
            const grid = get('lib-grid');
            renderLibrarySkeleton(grid);
            let files = null;
            let lastErr = null;
            const baseUrl = CHAT_CONFIG.urls.getFilesLib;
            for (let i = 0; i < 2; i++) {
                try {
                    const url = i === 0 ? baseUrl : (baseUrl + (baseUrl.includes('?') ? '&' : '?') + 't=' + Date.now());
                    const r = await apiFetch(url, { cache: 'no-store', headers: { 'Accept': 'application/json' } });
                    if (!r.ok) throw new Error('HTTP ' + r.status);
                    const raw = await r.text();
                    let parsed = [];
                    try { parsed = JSON.parse(raw); } catch (e) { parsed = []; }
                    if (Array.isArray(parsed)) {
                        files = parsed;
                        lastErr = null;
                        break;
                    }
                } catch (e) {
                    lastErr = e;
                }
            }
            if (!Array.isArray(files)) files = [];
            try {
                const base = FILE_BASE_URL;
                const thumbBase = FILE_THUMB_BASE_URL;
                const seenPaths = new Set(files.map(f => f && f.filepath).filter(Boolean));
                const extra = Array.isArray(currentImageUrls) ? currentImageUrls : [];
                extra.forEach((fp) => {
                    if (!fp || seenPaths.has(fp)) return;
                    const filename = getAttachmentNameForPath(fp) || (fp.split('/').pop() || fp);
                    const ext = (filename.split('.').pop() || '').toLowerCase();
                    const type = ['png','jpg','jpeg','webp','gif'].includes(ext) ? 'image' : 'file';
                    const thumbUrl = type === 'image' ? (thumbBase + fp) : null;
                    files.unshift({ filename, original_filename: filename, filepath: fp, url: base + fp, thumbnail_url: thumbUrl, type, ext, ts: Math.floor(Date.now() / 1000) });
                    seenPaths.add(fp);
                });
            } catch (e) {}
            try {
                if (grid) grid.innerHTML = '';
                if (!lib.selected) lib.selected = new Set();
                lib.selected.clear();
                lib.files = files.filter(f => f && f.filepath && f.url);
                lib.files.forEach((f) => {
                    if (f && f.filepath) setAttachmentNameForPath(f.filepath, f.filename || f.original_filename || '');
                });
                lib.fileSet = new Set(lib.files.map(f => f.filepath));
                window.updateLibSelectionUi();
                renderLibraryGrid();
            } catch (e) {
                lastErr = lastErr || e;
            }
            if (lastErr && grid) {
                console.error('Library load failed:', lastErr);
                grid.innerHTML = '<div class="lib-empty-state"><div class="lib-empty-icon"><i class="fas fa-exclamation-triangle"></i></div><p class="lib-empty-title">ライブラリの読み込みに失敗しました</p><p class="lib-empty-sub">通信状況を確認して時間をおいて再度お試しください。</p></div>';
            }
        }
        async function deleteSelectedFiles() {
            if(!confirm('削除しますか？')) return;
            try{
                await apiFetch(CHAT_CONFIG.urls.deleteFilesBatch, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({filenames:Array.from(lib.selected)})});
                loadLibraryFiles();
            } catch(e){
                alert("削除エラー");
            }
        }
        function attachSelectedLibraryFiles() {
            if (!lib.selected.size) return;
            const support = getModelMediaSupport(get('model-select').value);
            let skippedAudio = 0;
            let skippedVideo = 0;
            const selected = Array.from(lib.selected);
            selected.forEach((fp) => {
                const isAudio = isAudioPath(fp);
                const isVideo = isVideoPath(fp);
                if ((isAudio && !support.audio) || (isVideo && !support.video)) {
                    if (isAudio) skippedAudio += 1;
                    if (isVideo) skippedVideo += 1;
                    return;
                }
                const norm = normalizeAttachmentPath(fp);
                if (!norm) return;
                const item = (lib.files || []).find((f) => f && f.filepath === fp);
                if (item && item.filename) {
                    setAttachmentNameForPath(norm, item.filename);
                }
                if (!currentImageUrls.includes(norm)) currentImageUrls.push(norm);
                setAttachmentSourceForPath(norm, 'library');
            });
            syncUploadRowsFromCurrent();
            updateFilePreview();
            lib.selected.clear();
            window.updateLibSelectionUi();
            window.closeLibModal();
            if (skippedAudio || skippedVideo) {
                const parts = [];
                if (skippedAudio) parts.push(`${skippedAudio}件の音声`);
                if (skippedVideo) parts.push(`${skippedVideo}件の動画`);
                showToast(`このモデルは${parts.join('・')}入力に非対応のため除外しました`, "error", true);
            } else {
                showToast("ライブラリから添付しました", "success");
            }
        }
        function downloadSelectedLibraryFiles() {
            if (!lib.selected || !lib.selected.size) return;
            const selected = Array.from(lib.selected);
            selected.forEach((fp) => {
                const item = (lib.files || []).find((f) => f && f.filepath === fp);
                if (item && item.url) {
                    const a = document.createElement('a');
                    a.href = item.url;
                    a.download = item.filename || item.original_filename || (fp.split('/').pop() || 'file');
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                }
            });
            showToast(`${selected.length}件のファイルをダウンロードしました`, "success");
        }
        window.showLegal = async (t) => {
            const title = t === 'terms' ? '利用規約' : 'プライバシーポリシー';
            get('legal-title').innerText = title;
            showModal('legal-modal');
            const res = await apiFetch("/static/legal/" + t + ".md?t=" + Date.now());
            if(!res.ok) return;
            const text = await res.text();
            get('legal-content').innerHTML = sanitizeMarkdownHtml(text);
        }
        window.showAlphaInfo = () => {
            if (typeof showModal === 'function') {
                showModal('alpha-info-modal');
                return;
            }
            const el = get('alpha-info-modal');
            if (el) {
                el.classList.remove('hidden');
                el.style.display = 'flex';
            }
        };
        window.copyCode = (btn, code) => {
            const text = decodeURIComponent(code);
            const restoreIcon = () => {
                const kind = btn.getAttribute('data-copy') || '';
                btn.innerHTML = kind === 'output'
                    ? '<i class="fas fa-align-left"></i>'
                    : '<i class="fas fa-copy"></i>';
            };
            copyToClipboard(text,
                () => { btn.innerHTML = '<i class="fas fa-check"></i>'; setTimeout(restoreIcon, 2000); },
                (err) => { console.error(err); btn.innerHTML = '<i class="fas fa-times"></i>'; setTimeout(restoreIcon, 2000); }
            );
        };
        window.copyMessage = (id, btn) => {
            const txt = messageStore[id] || "";
            copyToClipboard(txt,
                () => { btn.innerHTML = '<i class="fas fa-check"></i>'; setTimeout(() => btn.innerHTML = '<i class="fas fa-copy"></i>', 2000); },
                (err) => { console.error(err); btn.innerHTML = '<i class="fas fa-times"></i>'; setTimeout(() => btn.innerHTML = '<i class="fas fa-copy"></i>', 2000); }
            );
        };
        window.toggleThinking = (el) => { const c = el.nextElementSibling; if(c.classList.contains('collapsed')) { c.classList.remove('collapsed'); } else { c.classList.add('collapsed'); } };

        // --- Branch Management System ---
        let selectedBranchNodeId = null;
        let branchLabelNames = {};
        let threadFixedBranchId = null;

        function loadBranchData() {
            if (!currentThreadId) return;
            const names = localStorage.getItem(`branch_names_${currentThreadId}`);
            branchLabelNames = names ? JSON.parse(names) : {};
            threadFixedBranchId = localStorage.getItem(`fixed_branch_${currentThreadId}`);
        }

        function saveBranchData() {
            if (!currentThreadId) return;
            localStorage.setItem(`branch_names_${currentThreadId}`, JSON.stringify(branchLabelNames));
            if (threadFixedBranchId) {
                localStorage.setItem(`fixed_branch_${currentThreadId}`, threadFixedBranchId);
            } else {
                localStorage.removeItem(`fixed_branch_${currentThreadId}`);
            }
        }

        function getCumulativeTokensForNode(nodeId) {
            let total = 0;
            let curr = nodeId;
            const msgMap = {};
            (allMessages || []).forEach(m => msgMap[m.id] = m);
            while (curr && msgMap[curr]) {
                const m = msgMap[curr];
                total += (m.tokens || (Number(m.tokens_in || 0) + Number(m.tokens_out || 0)));
                curr = m.parent_id;
            }
            return total;
        }

        function getPerModelTokensForPath(nodeId) {
            const modelStats = {}; // { modelName: { total, in, out, thought } }
            let curr = nodeId;
            const msgMap = {};
            (allMessages || []).forEach(m => msgMap[m.id] = m);

            while (curr && msgMap[curr]) {
                const m = msgMap[curr];
                const model = m.model || 'Unknown';
                if (!modelStats[model]) {
                    modelStats[model] = { total: 0, in: 0, out: 0, thought: 0 };
                }
                const rowTotal = (m.tokens || (Number(m.tokens_in || 0) + Number(m.tokens_out || 0)));
                modelStats[model].total += rowTotal;
                modelStats[model].in += Number(m.tokens_in || 0);
                modelStats[model].out += Number(m.tokens_out || 0);
                modelStats[model].thought += Number(m.tokens_thought || 0);
                curr = m.parent_id;
            }
            return modelStats;
        }

        window.showBranchModal = () => {
            if (!currentThreadId) {
                showToast('チャットを選択してください', 'error');
                return;
            }
            loadBranchData();
            selectedBranchNodeId = null;
            renderBranchTreeVisualization();
            updateBranchDetailPane();
            showModal('branch-modal');
            if (location.pathname !== '/branch') {
                history.pushState({ modal: 'branch' }, '', '/branch');
            }
            const allTotals = buildTokenTotals(allMessages);
            get('branch-total-tokens').innerText = allTotals.tokens_total || 0;
        };
        window.closeBranchModal = (skipHistory = false) => {
            hideModal('branch-modal');
            if (!skipHistory && location.pathname === '/branch') {
                history.back();
            }
        };

        function renderBranchTreeVisualization() {
            const container = get('branch-tree-canvas');
            container.innerHTML = '';
            if (!allMessages || allMessages.length === 0) return;
            const nodes = {};
            const roots = [];
            allMessages.forEach(msg => nodes[msg.id] = { ...msg, children: [] });
            allMessages.forEach(msg => {
                if (msg.parent_id && nodes[msg.parent_id]) {
                    nodes[msg.parent_id].children.push(nodes[msg.id]);
                } else if (!msg.parent_id) {
                    roots.push(nodes[msg.id]);
                }
            });

            function renderNodeRecursive(node) {
                const nodeEl = document.createElement('div');
                nodeEl.className = 'flex flex-col items-center mt-4';
                const item = document.createElement('div');
                const isCurrent = (String(node.id) === String(currentLeafId));
                const isFixed = (node.id === threadFixedBranchId);
                const name = branchLabelNames[node.id] || (node.role === 'user' ? 'User' : 'AI');
                const pathTokens = getCumulativeTokensForNode(node.id);

                item.className = `ui-enter-scale px-3 py-2 rounded-lg border cursor-pointer transition-all text-[10px] min-w-[120px] max-w-[180px] text-center relative ${
                    selectedBranchNodeId === node.id ? 'ring-2 ring-purple-500 border-purple-400' : 'border-gray-700 hover:border-gray-500'
                } ${isCurrent ? 'bg-blue-900/40 border-blue-500/50' : 'bg-gray-800'}`;

                item.innerHTML = `
                    <div class="font-bold truncate">${escapeHtml(name)}</div>
                    <div class="text-[9px] text-gray-500 flex justify-between mt-1 gap-2">
                        <span class="truncate">${escapeHtml(node.model || '-')}</span>
                        <span class="text-blue-400 font-mono font-bold" title="Cumulative tokens for this path">${pathTokens}</span>
                    </div>
                    ${isFixed ? '<div class="absolute -top-1 -right-1 w-3 h-3 bg-amber-500 rounded-full border border-gray-900 shadow-sm" title="Fixed Branch"></div>' : ''}
                    ${isCurrent ? '<div class="absolute -top-1 -left-1 w-3 h-3 bg-blue-500 rounded-full border border-gray-900 shadow-sm" title="Current Branch"></div>' : ''}
                `;
                item.onclick = (e) => {
                    e.stopPropagation();
                    selectedBranchNodeId = node.id;
                    renderBranchTreeVisualization();
                    updateBranchDetailPane();
                };
                nodeEl.appendChild(item);
                if (node.children.length > 0) {
                    const connector = document.createElement('div');
                    connector.className = 'w-px h-4 bg-gray-700';
                    nodeEl.appendChild(connector);
                    const childrenContainer = document.createElement('div');
                    childrenContainer.className = 'flex gap-4 items-start';
                    node.children.forEach(child => childrenContainer.appendChild(renderNodeRecursive(child)));
                    nodeEl.appendChild(childrenContainer);
                }
                return nodeEl;
            }
            roots.forEach(root => container.appendChild(renderNodeRecursive(root)));
        }

        function updateBranchDetailPane() {
            const detailPanel = get('branch-detail-panel');
            const emptyPanel = get('branch-empty-panel');
            if (!selectedBranchNodeId || !allMessages) {
                detailPanel.classList.add('hidden');
                emptyPanel.classList.remove('hidden');
                return;
            }
            const node = allMessages.find(m => m.id === selectedBranchNodeId);
            if (!node) return;
            detailPanel.classList.remove('hidden');
            emptyPanel.classList.add('hidden');
            get('br-id').innerText = node.id;
            get('br-date').innerText = node.created_at || '-';
            get('br-model').innerText = node.model || '-';
            const nodeTokens = (node.tokens || (Number(node.tokens_in || 0) + Number(node.tokens_out || 0)));
            const pathTokens = getCumulativeTokensForNode(node.id);
            get('br-tokens').innerHTML = `<span title="Current message tokens">${nodeTokens}</span> <span class="text-gray-500">/</span> <span class="text-purple-400 font-bold" title="Path total tokens">${pathTokens} total</span>`;

            // Render Per-Model Breakdown
            const breakdownContainer = get('branch-model-breakdown');
            const modelStats = getPerModelTokensForPath(node.id);
            breakdownContainer.innerHTML = '';

            Object.entries(modelStats).sort((a, b) => b[1].total - a[1].total).forEach(([model, stats]) => {
                const div = document.createElement('div');
                div.className = 'bg-gray-800/50 p-2 rounded border border-gray-700/50';
                div.innerHTML = `
                    <div class="flex justify-between font-bold text-gray-300 mb-1">
                        <span class="truncate pr-2">${model}</span>
                        <span class="text-blue-400 shrink-0">${stats.total}</span>
                    </div>
                    <div class="grid grid-cols-3 gap-1 text-[9px] text-gray-500 font-mono">
                        <div title="Input tokens">In: ${stats.in}</div>
                        <div title="Output tokens">Out: ${stats.out}</div>
                        <div title="Thought/Reasoning tokens">${stats.thought > 0 ? `Th: ${stats.thought}` : ''}</div>
                    </div>
                `;
                breakdownContainer.appendChild(div);
            });

            get('br-name-input').value = branchLabelNames[node.id] || '';
            const fixBtn = get('br-fix-btn');
            if (selectedBranchNodeId === threadFixedBranchId) {
                fixBtn.innerText = '固定を解除';
                fixBtn.classList.replace('bg-amber-600', 'bg-gray-600');
            } else {
                fixBtn.innerText = 'メインルートに固定';
                fixBtn.classList.replace('bg-gray-600', 'bg-amber-600');
            }
        }

        // Branch UI Handlers
        if (get('branch-manage-btn')) get('branch-manage-btn').onclick = showBranchModal;
        get('br-save-name-btn').onclick = () => {
            if (!selectedBranchNodeId) return;
            const name = get('br-name-input').value.trim();
            if (name) branchLabelNames[selectedBranchNodeId] = name; else delete branchLabelNames[selectedBranchNodeId];
            saveBranchData(); renderBranchTreeVisualization(); showToast('名前を保存しました');
        };
        get('br-switch-btn').onclick = () => {
            if (!selectedBranchNodeId) return;
            switchVersion(selectedBranchNodeId); window.closeBranchModal(); showToast('ブランチを切り替えました');
        };
        get('br-fix-btn').onclick = () => {
            if (!selectedBranchNodeId) return;
            if (threadFixedBranchId === selectedBranchNodeId) { threadFixedBranchId = null; showToast('固定を解除しました'); }
            else { threadFixedBranchId = selectedBranchNodeId; showToast('メインルートに固定しました'); }
            saveBranchData(); renderBranchTreeVisualization(); updateBranchDetailPane();
        };
        get('br-delete-btn').onclick = () => {
            if (!selectedBranchNodeId) return;
            if (!confirm('このブランチを削除してもよろしいですか？（その後の全てのメッセージも削除されます）')) return;
            deleteMessage(selectedBranchNodeId);
            selectedBranchNodeId = null;
            setTimeout(() => { renderBranchTreeVisualization(); updateBranchDetailPane(); }, 500);
        };

        const showApiKeyRequiredModalAsync = (modelId) => new Promise((resolve) => {
            const modelName = getModelNameById(modelId);
            const info = getModelProviderInfo(modelId);
            get('api-key-modal-model-name').textContent = `${modelName}（${modelId}）`;
            get('api-key-modal-desc').textContent = `このモデルを使用するには${info ? info.label : 'APIキー'}の設定が必要です。`;
            get('api-key-modal-key-label').textContent = info ? info.label : 'API Key';
            const existingInput = info ? get(info.inputId) : null;
            get('api-key-modal-input').value = existingInput ? existingInput.value : '';
            get('api-key-modal-input').placeholder = 'APIキーを入力';
            const saveBtn = get('api-key-modal-save-btn');
            const fallbackBtn = get('api-key-modal-fallback-btn');
            const cancelBtn = get('api-key-modal-cancel-btn');
            const cleanup = () => {
                saveBtn.onclick = null;
                fallbackBtn.onclick = null;
                cancelBtn.onclick = null;
            };
            const onKeydown = (e) => {
                if (e.key === 'Enter') { e.preventDefault(); saveBtn.click(); }
            };
            get('api-key-modal-input').addEventListener('keydown', onKeydown);
            saveBtn.onclick = async () => {
                const key = get('api-key-modal-input').value.trim();
                if (!key) {
                    showToast('APIキーを入力してください', 'error');
                    return;
                }
                if (info) {
                    const input = get(info.inputId);
                    if (input) input.value = key;
                    try {
                        const res = await apiFetch(CHAT_CONFIG.urls.handleSettings, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ [info.keyField]: key })
                        });
                        if (!res.ok) {
                            showToast('APIキーの保存に失敗しました', 'error', true);
                            return;
                        }
                        if (userSettingsSnapshot) {
                            userSettingsSnapshot[info.keyField] = key;
                        }
                    } catch (e) {
                        showToast('APIキーの保存に失敗しました', 'error', true);
                        return;
                    }
                }
                hideModal('api-key-required-modal');
                get('api-key-modal-input').removeEventListener('keydown', onKeydown);
                cleanup();
                resolve('set');
            };
            fallbackBtn.onclick = () => {
                hideModal('api-key-required-modal');
                get('api-key-modal-input').removeEventListener('keydown', onKeydown);
                cleanup();
                resolve('switch');
            };
            cancelBtn.onclick = () => {
                hideModal('api-key-required-modal');
                get('api-key-modal-input').removeEventListener('keydown', onKeydown);
                cleanup();
                resolve('cancel');
            };
            showModal('api-key-required-modal');
            setTimeout(() => {
                const input = get('api-key-modal-input');
                if (input) input.focus();
            }, 350);
        });
        // --- Extended Client-Side Debug Logging System ---
        (function() {
            const originalLog = console.log;
            const originalError = console.error;
            const originalWarn = console.warn;
            const originalInfo = console.info;
            let isSending = false;

            async function sendToServer(level, args) {
                if (isSending) return;
                if (!isClientDebugLogEnabled()) return;
                if (args && args[0] === ADMIN_SIDEBAR_DEBUG_PREFIX) return;

                isSending = true;
                const message = args.map(arg => {
                    try {
                        if (arg instanceof Error) return arg.stack || arg.message;
                        return typeof arg === 'object' ? JSON.stringify(arg) : String(arg);
                    } catch (e) {
                        return "[Unserializable Object]";
                    }
                }).join(' ');

                try {
                    sendClientDebugLog(level, message);
                } catch (e) {
                    // fallthrough
                } finally {
                    isSending = false;
                }
            }

            console.log = function(...args) {
                originalLog.apply(console, args);
                sendToServer('log', args);
            };
            console.error = function(...args) {
                originalError.apply(console, args);
                sendToServer('error', args);
            };
            console.warn = function(...args) {
                originalWarn.apply(console, args);
                sendToServer('warn', args);
            };
            console.info = function(...args) {
                originalInfo.apply(console, args);
                sendToServer('info', args);
            };

            window.addEventListener('error', function(event) {
                sendToServer('exception', [event.message, event.filename, event.lineno, event.colno, event.error]);
            });
            window.addEventListener('unhandledrejection', function(event) {
                sendToServer('promise-rejection', [event.reason]);
            });

            // Trigger initial log to confirm system is active
            setTimeout(() => {
                console.log("Extended debug logging system active. Version: v4.8.506");
            }, 3000);
        })();
