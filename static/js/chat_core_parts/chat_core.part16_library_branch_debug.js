        function openLibraryImage(f) {
            if (!lib.files) return;
            const ordered = sortLibraryFiles(lib.files);
            const q = getLibSearchQuery();
            const filtered = q ? ordered.filter((x) => fileNameForSearch(x).includes(q)) : ordered;
            const images = filtered.filter((x) => x.type === 'image');
            const visibleImages = lib.favoritesOnly ? images.filter((x) => x.is_favorite) : images;
            if (!visibleImages.length) return;
            const items = visibleImages.map((x) => ({
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
            const favoriteClass = f.is_favorite ? ' is-favorite' : '';
            const favoriteIcon = f.is_favorite ? 'fas fa-star' : 'far fa-star';
            const favoriteLabel = f.is_favorite ? 'お気に入りから外す' : 'お気に入りに追加';
            const actions = `<div class="lib-thumb-actions"><button class="lib-favorite-btn lib-action-circle${favoriteClass}" title="${favoriteLabel}" aria-label="${favoriteLabel}" aria-pressed="${f.is_favorite ? 'true' : 'false'}"><i class="${favoriteIcon}"></i></button><button class="lib-open-btn lib-action-circle" title="開く"><i class="fas fa-eye"></i></button><button class="lib-del-btn lib-action-circle lib-del" title="削除"><i class="fas fa-trash"></i></button></div>`;
            const bar = `<div class="lib-thumb-bar"><span class="lib-thumb-name" title="${escapeHtml(f.filename)}">${escapeHtml(f.filename)}</span></div>`;
            el.innerHTML = `<div class="lib-thumb-media-wrap">${media}</div>${overlay}${actions}${bar}`;
            el.dataset.filepath = f.filepath;
            el.addEventListener('mousedown', (e) => {
                if (e.shiftKey) e.preventDefault();
            });
            el.onclick = (e) => {
                if (e && e.shiftKey && lib.anchorPath && lib.anchorPath !== f.filepath) {
                    const cards = Array.from(el.parentNode ? el.parentNode.querySelectorAll('.library-thumb-card') : []);
                    const anchorIdx = cards.findIndex((c) => c.dataset.filepath === lib.anchorPath);
                    const targetIdx = cards.indexOf(el);
                    if (anchorIdx !== -1 && targetIdx !== -1) {
                        const from = Math.min(anchorIdx, targetIdx);
                        const to = Math.max(anchorIdx, targetIdx);
                        for (let k = from; k <= to; k++) {
                            const path = cards[k].dataset.filepath;
                            if (!path) continue;
                            lib.selected.add(path);
                            cards[k].classList.add('is-selected');
                        }
                        try {
                            const sel = window.getSelection && window.getSelection();
                            if (sel) sel.removeAllRanges();
                        } catch (_) {}
                        window.updateLibSelectionUi();
                        return;
                    }
                }
                lib.anchorPath = f.filepath;
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
            const favoriteBtn = el.querySelector('.lib-favorite-btn');
            if (favoriteBtn) {
                favoriteBtn.onclick = async (e) => {
                    e.stopPropagation();
                    favoriteBtn.disabled = true;
                    try {
                        const r = await apiFetch(CHAT_CONFIG.urls.toggleFileFavorite, {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({filepath: f.filepath})
                        });
                        const d = await r.json().catch(() => ({}));
                        if (!r.ok || typeof d.is_favorite !== 'boolean') throw new Error(d.error || 'favorite update failed');
                        f.is_favorite = d.is_favorite;
                        renderLibraryGrid();
                        showToast(d.is_favorite ? 'お気に入りに追加しました' : 'お気に入りから外しました', 'success');
                    } catch (err) {
                        showToast('お気に入りの更新に失敗しました', 'error', true);
                        favoriteBtn.disabled = false;
                    }
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
        function closeFileUsageModal() {
            hideModal('lib-usage-modal');
        }
        window.closeFileUsageModal = closeFileUsageModal;
        async function showSelectedFileUsage() {
            if (!lib.selected || lib.selected.size !== 1) return;
            const filepath = Array.from(lib.selected)[0];
            const item = (lib.files || []).find((f) => f && f.filepath === filepath);
            const title = get('lib-usage-title');
            const list = get('lib-usage-list');
            if (!list) return;
            if (title) title.textContent = (item && (item.filename || item.original_filename)) || filepath.split('/').pop() || filepath;
            list.innerHTML = '<div class="text-sm text-gray-400 text-center py-8"><i class="fas fa-spinner fa-spin mr-2"></i>読み込み中…</div>';
            showModal('lib-usage-modal');
            try {
                const url = new URL(CHAT_CONFIG.urls.getFileUsageChats, window.location.origin);
                url.searchParams.set('filepath', filepath);
                const r = await apiFetch(url.toString(), { cache: 'no-store', headers: { 'Accept': 'application/json' } });
                const data = await r.json().catch(() => ({}));
                if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`);
                const chats = Array.isArray(data.chats) ? data.chats : [];
                if (!chats.length) {
                    list.innerHTML = '<div class="text-sm text-gray-400 text-center py-8"><i class="fas fa-comment-dots text-xl mb-2 block"></i>このファイルを使用しているチャットはありません。</div>';
                    return;
                }
                list.innerHTML = '';
                chats.forEach((chat) => {
                    const row = document.createElement('div');
                    row.className = 'flex items-center gap-3 rounded-lg border border-gray-700 bg-gray-800/70 p-3';
                    const updated = chat.updated_at ? new Date(chat.updated_at).toLocaleString() : '';
                    row.innerHTML = `<div class="min-w-0 flex-1"><div class="text-sm text-gray-200 truncate" title="${escapeHtml(chat.title || '')}">${escapeHtml(chat.title || '新しいチャット')}</div><div class="text-[11px] text-gray-500 mt-1">${escapeHtml(updated)}</div></div><button type="button" class="lib-action-btn lib-btn-accent shrink-0"><i class="fas fa-folder"></i><span>開く</span></button>`;
                    const openBtn = row.querySelector('button');
                    if (openBtn) {
                        openBtn.onclick = async () => {
                            closeFileUsageModal();
                            if (window.closeLibModal) window.closeLibModal(true);
                            await loadMessages(String(chat.id));
                        };
                    }
                    list.appendChild(row);
                });
                if (data.has_more) {
                    const note = document.createElement('p');
                    note.className = 'text-[11px] text-gray-500 text-center pt-2';
                    note.textContent = '表示できるチャットは最大100件です。';
                    list.appendChild(note);
                }
            } catch (e) {
                list.innerHTML = '<div class="text-sm text-red-300 text-center py-8"><i class="fas fa-exclamation-triangle mr-2"></i>使用チャットの取得に失敗しました。</div>';
            }
        }
        async function loadLibraryFiles(loadMore = false) {
            const grid = get('lib-grid');
            const loadMoreBtn = get('lib-load-more-btn');
            if (lib.loading) return;
            if (loadMore && !lib.hasMore) return;
            lib.loading = true;
            if (!loadMore) {
                lib.nextOffset = 0;
                lib.totalCount = 0;
                lib.hasMore = false;
            }
            if (!loadMore) renderLibrarySkeleton(grid);
            let lastErr = null;
            const baseUrl = CHAT_CONFIG.urls.getFilesLib;
            let payload = null;
            let requestSucceeded = false;
            try {
                const sort = getLibSortOrder();
                const query = getLibSearchQuery();
                const offset = loadMore ? lib.nextOffset : 0;
                const params = new URLSearchParams({
                    limit: String(LIBRARY_PAGE_SIZE),
                    offset: String(offset),
                    sort,
                    q: query,
                    favorites_only: lib.favoritesOnly ? '1' : '0'
                });
                const r = await apiFetch(baseUrl + '?' + params.toString(), { cache: 'no-store', headers: { 'Accept': 'application/json' } });
                if (!r.ok) throw new Error('HTTP ' + r.status);
                payload = await r.json();
                requestSucceeded = true;
            } catch (e) {
                lastErr = e;
            }
            if (!requestSucceeded) {
                console.error('Library load failed:', lastErr);
                if (!loadMore && grid) {
                    grid.innerHTML = '<div class="lib-empty-state"><div class="lib-empty-icon"><i class="fas fa-exclamation-triangle"></i></div><p class="lib-empty-title">ライブラリの読み込みに失敗しました</p><p class="lib-empty-sub">通信状況を確認して時間をおいて再度お試しください。</p></div>';
                } else if (loadMore) {
                    showToast('追加読み込みに失敗しました。もう一度お試しください。', 'error', true);
                }
                lib.loading = false;
                if (loadMoreBtn) {
                    loadMoreBtn.disabled = false;
                    loadMoreBtn.hidden = !lib.hasMore;
                }
                return;
            }
            let files = Array.isArray(payload) ? payload : (payload && Array.isArray(payload.files) ? payload.files : []);
            if (payload && !Array.isArray(payload)) {
                lib.totalCount = Number(payload.total) || 0;
                lib.hasMore = !!payload.has_more;
                lib.nextOffset = (Number(payload.offset) || 0) + (Number(payload.limit) || files.length);
            }
            try {
                const base = FILE_BASE_URL;
                const thumbBase = FILE_THUMB_BASE_URL;
                const seenPaths = new Set(files.map(f => f && f.filepath).filter(Boolean));
                const extra = !loadMore && Array.isArray(currentImageUrls) ? currentImageUrls : [];
                extra.forEach((fp) => {
                    if (files.length >= LIBRARY_PAGE_SIZE) return;
                    if (!fp || seenPaths.has(fp)) return;
                    const filename = getAttachmentNameForPath(fp) || (fp.split('/').pop() || fp);
                    const ext = (filename.split('.').pop() || '').toLowerCase();
                    const type = ['png','jpg','jpeg','webp','gif'].includes(ext) ? 'image' : 'file';
                    const thumbUrl = type === 'image' ? (thumbBase + fp) : null;
                    files.unshift({ filename, original_filename: filename, filepath: fp, url: base + fp, thumbnail_url: thumbUrl, type, ext, is_favorite: false, ts: Math.floor(Date.now() / 1000) });
                    seenPaths.add(fp);
                });
            } catch (e) {}
            try {
                if (!lib.selected) lib.selected = new Set();
                if (!loadMore) lib.selected.clear();
                const pageFiles = files.filter(f => f && f.filepath && f.url);
                let appendedFiles = [];
                if (loadMore) {
                    const existing = new Set(lib.files.map(f => f.filepath));
                    appendedFiles = pageFiles.filter(f => !existing.has(f.filepath));
                    lib.files.push(...appendedFiles);
                } else {
                    lib.files = pageFiles;
                }
                lib.files.forEach((f) => {
                    if (f && f.filepath) setAttachmentNameForPath(f.filepath, f.filename || f.original_filename || '');
                });
                lib.fileSet = new Set(lib.files.map(f => f.filepath));
                if (!lib.totalCount) lib.totalCount = lib.files.length;
                window.updateLibSelectionUi();
                renderLibraryGrid(loadMore ? appendedFiles : null);
            } catch (e) {
                lastErr = lastErr || e;
            }
            if (lastErr && grid) {
                console.error('Library load failed:', lastErr);
                if (loadMore) {
                    showToast('追加読み込みに失敗しました。もう一度お試しください。', 'error', true);
                } else {
                    grid.innerHTML = '<div class="lib-empty-state"><div class="lib-empty-icon"><i class="fas fa-exclamation-triangle"></i></div><p class="lib-empty-title">ライブラリの読み込みに失敗しました</p><p class="lib-empty-sub">通信状況を確認して時間をおいて再度お試しください。</p></div>';
                }
            }
            lib.loading = false;
            if (loadMoreBtn) {
                loadMoreBtn.disabled = false;
                loadMoreBtn.hidden = !lib.hasMore;
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

        function formatBranchCreatedAt(value) {
            if (!value) return '-';
            const date = new Date(value);
            if (isNaN(date.getTime())) return String(value);
            return date.toLocaleString('ja-JP', { year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' });
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
            get('br-date').innerText = formatBranchCreatedAt(node.created_at);
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
            deleteMessage(selectedBranchNodeId, true);
            selectedBranchNodeId = null;
            setTimeout(() => { renderBranchTreeVisualization(); updateBranchDetailPane(); }, 500);
        };

        // ---- Batch processing management -------------------------------
        // The history list is never trimmed automatically; finished rows stay
        // until the user deletes them from this modal.
        let batchJobsCache = [];
        let batchFilterMode = 'all';
        let batchListTimer = null;

        function batchProviderLabel(provider) {
            return { gemini: 'Gemini', openai: 'OpenAI', xai: 'xAI', zai: 'Z.AI' }[String(provider || '').toLowerCase()]
                || (provider || 'Batch');
        }

        function batchStateLabelShort(state) {
            return {
                JOB_STATE_QUEUED: '送信待ち',
                JOB_STATE_VALIDATING: '検証中',
                JOB_STATE_PENDING: '待機中',
                JOB_STATE_RUNNING: '実行中',
                JOB_STATE_FINALIZING: '結果取得中',
                JOB_STATE_SUCCEEDED: '完了',
                JOB_STATE_FAILED: '失敗',
                JOB_STATE_CANCELLING: '停止中',
                JOB_STATE_CANCELLED: '停止',
                JOB_STATE_EXPIRED: '期限切れ'
            }[String(state || '').toUpperCase()] || '確認中';
        }

        function batchStateTone(state) {
            const s = String(state || '').toUpperCase();
            if (s === 'JOB_STATE_SUCCEEDED') return 'border-emerald-500/40 bg-emerald-900/20 text-emerald-200';
            if (s === 'JOB_STATE_FAILED') return 'border-red-500/40 bg-red-900/20 text-red-200';
            if (s === 'JOB_STATE_CANCELLED' || s === 'JOB_STATE_EXPIRED') return 'border-gray-500/40 bg-gray-700/30 text-gray-300';
            if (s === 'JOB_STATE_CANCELLING') return 'border-amber-500/40 bg-amber-900/20 text-amber-200';
            return 'border-violet-500/40 bg-violet-900/20 text-violet-200';
        }

        function batchFormatTime(value) {
            if (!value) return '';
            let raw = String(value);
            if (!/[zZ]$/.test(raw) && !/[+-]\d\d:?\d\d$/.test(raw)) raw += 'Z';
            const date = new Date(raw);
            if (isNaN(date.getTime())) return String(value);
            return date.toLocaleString('ja-JP', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' });
        }

        function playBatchListAnimation() {
            const list = get('batch-list');
            if (!list) return;
            list.classList.remove('batch-list-enter');
            // Restart the CSS animation on every filter switch.
            void list.offsetWidth;
            list.classList.add('batch-list-enter');
        }

        function renderBatchJobs(options = {}) {
            const list = get('batch-list');
            if (!list) return;
            const jobs = batchJobsCache.filter((job) => {
                if (batchFilterMode === 'active') return !!job.is_active;
                if (batchFilterMode === 'done') return !job.is_active;
                return true;
            });
            const count = get('batch-count');
            if (count) count.textContent = `${jobs.length}件`;
            list.innerHTML = '';
            if (!jobs.length) {
                list.innerHTML = '<div class="batch-empty"><i class="fas fa-layer-group"></i><span>Batch処理の履歴はありません</span></div>';
                if (options.animate) playBatchListAnimation();
                return;
            }
            jobs.forEach((job) => {
                const row = document.createElement('div');
                row.className = 'batch-job-card';
                const title = escapeHtml(job.thread_title || '無題のチャット');
                const provider = escapeHtml(batchProviderLabel(job.provider));
                const model = escapeHtml(job.model || '');
                const created = escapeHtml(batchFormatTime(job.created_at));
                const status = escapeHtml(job.status_text || '');
                const tone = batchStateTone(job.state);
                const openButton = job.thread_exists
                    ? '<button type="button" data-batch-open class="batch-action-btn batch-action-open"><i class="fas fa-comment-dots"></i>開く</button>'
                    : '';
                const cancelButton = job.can_cancel
                    ? '<button type="button" data-batch-cancel class="batch-action-btn batch-action-cancel"><i class="fas fa-stop"></i>停止</button>'
                    : '';
                const deleteButton = !job.is_active
                    ? '<button type="button" data-batch-delete class="batch-action-btn batch-action-danger"><i class="fas fa-trash"></i>履歴から削除</button>'
                    : '';
                row.innerHTML = `
                    <div class="flex items-start justify-between gap-3">
                        <div class="min-w-0">
                            <div class="batch-job-title text-sm font-bold truncate" title="${title}">${title}</div>
                            <div class="batch-job-meta mt-1 flex flex-wrap items-center gap-2 text-[10px]">
                                <span class="inline-flex items-center gap-1"><i class="fas fa-layer-group"></i>${provider}</span>
                                <span class="truncate max-w-[16rem]">${model}</span>
                                <span><i class="fas fa-history mr-1"></i>${created}</span>
                            </div>
                        </div>
                        <span class="batch-state-badge shrink-0 ${tone}">${escapeHtml(batchStateLabelShort(job.state))}</span>
                    </div>
                    <div class="batch-job-status mt-2 text-[11px] break-words">${status}</div>
                    ${job.error ? `<div class="batch-job-error mt-1 text-[10px] break-words">${escapeHtml(job.error)}</div>` : ''}
                    <div class="mt-3 flex flex-wrap gap-2">
                        ${openButton}${cancelButton}${deleteButton}
                    </div>`;
                const openBtn = row.querySelector('[data-batch-open]');
                if (openBtn) openBtn.onclick = () => {
                    window.closeBatchModal();
                    loadMessages(job.thread_id);
                };
                const cancelBtn = row.querySelector('[data-batch-cancel]');
                if (cancelBtn) cancelBtn.onclick = () => cancelBatchJob(job);
                const delBtn = row.querySelector('[data-batch-delete]');
                if (delBtn) delBtn.onclick = () => deleteBatchJob(job);
                list.appendChild(row);
            });
            if (options.animate) playBatchListAnimation();
        }

        async function loadBatchJobs(opts = {}) {
            try {
                const response = await apiFetch('/api/batch/jobs');
                if (!response.ok) {
                    if (!opts.silent) showToast('Batch処理の履歴を取得できませんでした', 'error');
                    return;
                }
                const data = await response.json().catch(() => ({}));
                batchJobsCache = Array.isArray(data.jobs) ? data.jobs : [];
                renderBatchJobs();
            } catch (error) {
                if (!opts.silent) showToast('Batch処理の履歴を取得できませんでした', 'error');
            }
        }

        async function cancelBatchJob(job) {
            if (!confirm('このBatch処理を停止しますか？')) return;
            const response = await apiFetch(`/api/batch/jobs/${encodeURIComponent(job.job_id)}/cancel`, { method: 'POST' });
            const data = await response.json().catch(() => ({}));
            if (!response.ok) {
                showToast(data.error || 'Batch処理を停止できませんでした', 'error', true);
                return;
            }
            showToast('Batch処理を停止しました', 'success');
            await loadBatchJobs({ silent: true });
            if (String(job.thread_id) === String(currentThreadId)) {
                await loadMessages(currentThreadId, { preserveDraft: true, silent: true });
            }
        }

        async function deleteBatchJob(job) {
            if (!confirm('このBatch処理の履歴を削除しますか？')) return;
            const response = await apiFetch(`/api/batch/jobs/${encodeURIComponent(job.job_id)}`, { method: 'DELETE' });
            const data = await response.json().catch(() => ({}));
            if (!response.ok) {
                showToast(data.error || 'Batch履歴を削除できませんでした', 'error', true);
                return;
            }
            showToast('Batch履歴を削除しました', 'success');
            await loadBatchJobs({ silent: true });
        }

        window.showBatchModal = () => {
            showModal('batch-modal');
            if (location.pathname !== '/batch') {
                history.pushState({ modal: 'batch' }, '', '/batch');
            }
            loadBatchJobs();
            if (batchListTimer) clearInterval(batchListTimer);
            batchListTimer = setInterval(() => {
                const modal = get('batch-modal');
                if (!modal || modal.classList.contains('hidden')) return;
                loadBatchJobs({ silent: true });
            }, 5000);
        };
        window.closeBatchModal = (skipHistory = false) => {
            hideModal('batch-modal');
            if (batchListTimer) { clearInterval(batchListTimer); batchListTimer = null; }
            if (!skipHistory && location.pathname === '/batch') {
                history.back();
            }
        };

        if (get('batch-manage-btn')) get('batch-manage-btn').onclick = () => window.showBatchModal();
        if (get('batch-refresh-btn')) get('batch-refresh-btn').onclick = () => loadBatchJobs();
        document.querySelectorAll('.batch-filter-tab').forEach((tab) => {
            tab.onclick = () => {
                batchFilterMode = tab.dataset.batchFilter || 'all';
                document.querySelectorAll('.batch-filter-tab').forEach((other) => {
                    other.classList.toggle('is-active', other === tab);
                });
                renderBatchJobs({ animate: true });
            };
        });

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

        // 管理者のBot検出ログ画面（#bot-log-modal）: 記録のあるアカウントの一覧と、アカウントごとの状態・記録。
        // 記録はアカウントを削除しても残るため、削除済みアカウントも user_id で開ける。
        // 開閉と URL（/admin-bot-logs）は part08 の openBotLogModal / closeBotLogModal。
        const BotAdminLog = (() => {
            const PAGE_SIZE = 50;
            const EVENT_LABELS = {
                telemetry: { label: '疑わしい操作', cls: 'bg-yellow-600' },
                verify_ok: { label: 'Turnstile 確認成功', cls: 'bg-green-600' },
                verify_fail: { label: 'Turnstile 確認失敗', cls: 'bg-orange-600' },
                turnstile_blocked: { label: '未確認のため拒否', cls: 'bg-orange-600' },
                lock: { label: '一時ロック', cls: 'bg-yellow-600' },
                lock_blocked: { label: 'ロック中のため拒否', cls: 'bg-yellow-600' },
                ban: { label: 'BAN', cls: 'bg-red-600' },
                related_ban: { label: '関連アカウントからのBAN', cls: 'bg-red-600' },
                unban: { label: 'BAN解除', cls: 'bg-green-600' },
                admin_action: { label: '管理者の操作', cls: 'bg-blue-600' },
                account_deleted: { label: 'アカウント削除', cls: 'bg-gray-600' }
            };
            const CLIENT_LABELS = { web: 'Web', android: 'アプリ', admin: '管理者', server: 'サーバー' };
            const DETAIL_LABELS = {
                endpoint: '通信先', method: '方式', path: 'パス', client: '端末',
                lock_source: 'ロックの対象', lock_reason: 'ロックの理由', remaining_seconds: 'ロックの残り',
                origin_username: 'ロックをかけたアカウント', origin_user_id: 'ロックをかけたアカウントID', origin_client: 'ロックをかけた端末',
                lock_seconds: 'ロック時間', applied_to: 'ロックの範囲', lock_count: 'ロック回数（1時間）', ban_at_count: 'BANになる回数',
                source_username: 'BANの起点', source_user_id: '起点のアカウントID',
                action: '操作', admin: '管理者', enabled: '検出', by: '実行した人', count: '件数',
                previous_reason: '解除前のBANの理由', linked_from: '連鎖解除の起点'
            };
            const SOURCE_LABELS = { account: 'アカウント', ip: 'IPアドレス', cookie: '端末（Cookie）' };
            const ACTION_LABELS = {
                toggle_detection: '検出の切り替え', ban: 'BAN', unban: 'BAN解除（単独）',
                unban_linked: 'BAN解除（連鎖）', unlock: 'ロック解除', unblock_identifiers: 'IP・端末のBAN解除'
            };
            const state = { userId: 0, username: '', exists: false, type: '', items: [], hasMore: false, selected: new Set(), counts: {}, total: 0, data: null };
            const listState = { accounts: [], selected: new Set() };

            const view = () => get('bot-log-detail');
            const listView = () => get('bot-log-list-view');
            const formatTime = (iso) => {
                if (!iso) return '-';
                const d = new Date(iso);
                return isNaN(d.getTime()) ? String(iso) : d.toLocaleString('ja-JP');
            };
            const formatDuration = (sec) => {
                const s = Math.max(0, Math.floor(Number(sec) || 0));
                if (s >= 60) return `${Math.floor(s / 60)}分${s % 60 ? (s % 60) + '秒' : ''}`;
                return `${s}秒`;
            };
            const formatDetailValue = (key, value) => {
                if (value === null || value === undefined || value === '') return '-';
                if (key === 'lock_source') return SOURCE_LABELS[value] || String(value);
                if (key === 'client' || key === 'origin_client') return CLIENT_LABELS[value] || String(value);
                if (key === 'action') return ACTION_LABELS[value] || String(value);
                if (key === 'by') return value === 'self' ? '本人' : (value === 'admin' ? '管理者' : String(value));
                if (key === 'enabled') return value ? 'ON' : 'OFF';
                if (key === 'remaining_seconds' || key === 'lock_seconds') return formatDuration(value);
                if (key === 'applied_to' && Array.isArray(value)) return value.map(v => SOURCE_LABELS[v] || v).join('・');
                if (typeof value === 'object') return JSON.stringify(value, null, 2);
                return String(value);
            };
            const renderDetails = (raw) => {
                const text = String(raw || '').trim();
                if (!text) return '';
                let parsed = null;
                if (text.startsWith('{')) {
                    try { parsed = JSON.parse(text); } catch (e) { parsed = null; }
                }
                if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
                    const rows = Object.keys(parsed).map((key) => {
                        const val = formatDetailValue(key, parsed[key]);
                        const label = DETAIL_LABELS[key] || key;
                        const isBlock = val.length > 120 || val.includes('\n');
                        return isBlock
                            ? `<div class="mt-1"><div class="text-gray-500">${escapeHtml(label)}</div><pre class="bot-log-pre">${escapeHtml(val)}</pre></div>`
                            : `<div><span class="text-gray-500">${escapeHtml(label)}:</span> <span class="text-gray-300">${escapeHtml(val)}</span></div>`;
                    }).join('');
                    return text.length > 600
                        ? `<details class="mt-1"><summary class="cursor-pointer text-gray-400">詳細を表示</summary>${rows}</details>`
                        : `<div class="mt-1 space-y-1">${rows}</div>`;
                }
                return text.length > 300
                    ? `<details class="mt-1"><summary class="cursor-pointer text-gray-400">詳細を表示</summary><pre class="bot-log-pre">${escapeHtml(text)}</pre></details>`
                    : `<div class="mt-1 text-gray-400 bot-log-wrap">${escapeHtml(text)}</div>`;
            };
            const postJson = async (url, body) => {
                const res = await apiFetch(url, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(body)
                });
                const data = await res.json().catch(() => ({}));
                if (!res.ok) throw new Error(data.error || String(res.status));
                return data;
            };
            const badge = (text, cls) => `<span class="${cls} text-white px-2 py-0.5 rounded">${escapeHtml(text)}</span>`;

            // ---- アカウント一覧 ----
            const renderAccounts = () => {
                const list = get('bot-log-account-list');
                if (!list) return;
                const accounts = listState.accounts;
                const allChecked = accounts.length && accounts.every(a => listState.selected.has(a.user_id));
                const toolbar = get('bot-log-list-toolbar');
                if (toolbar) {
                    toolbar.innerHTML = `
                        <label class="flex items-center gap-1 text-gray-300"><input type="checkbox" class="bot-log-account-select-all" aria-label="表示中のアカウントをすべて選択" ${allChecked ? 'checked' : ''}>表示中をすべて選択</label>
                        <button class="bot-log-delete-accounts bg-red-600 hover:bg-red-500 text-white px-2 py-1 rounded" ${listState.selected.size ? '' : 'disabled'}>選択したアカウントの記録を一括削除${listState.selected.size ? `（${listState.selected.size}件）` : ''}</button>`;
                }
                if (!accounts.length) {
                    list.innerHTML = '<div class="text-xs text-gray-400">記録のあるアカウントはありません。</div>';
                    return;
                }
                list.innerHTML = accounts.map((a) => {
                    const badges = [];
                    if (!a.exists) badges.push(badge('削除済み', 'bg-gray-600'));
                    if (a.is_bot_banned) badges.push(badge('BAN中', 'bg-red-600'));
                    if (a.lock_remaining_seconds > 0) badges.push(badge(`ロック中（残り${Math.ceil(a.lock_remaining_seconds / 60)}分）`, 'bg-yellow-600'));
                    const summary = a.evidence_count
                        ? `記録 ${a.evidence_count}件・最終 ${escapeHtml(formatTime(a.last_event_at))}`
                        : '記録なし';
                    return `
                        <div class="flex items-center gap-2 bg-gray-900 border border-gray-700 rounded p-2 text-xs">
                            <input type="checkbox" class="bot-log-account-select" data-user-id="${a.user_id}" ${listState.selected.has(a.user_id) ? 'checked' : ''} aria-label="${escapeHtml(a.username || ('ID ' + a.user_id))} を選択">
                            <div class="flex-1 min-w-0">
                                <div class="flex flex-wrap items-center gap-1"><span class="${a.exists ? 'text-gray-200' : 'text-gray-400'} font-bold bot-log-wrap">${escapeHtml(a.username || ('ID ' + a.user_id))}</span>${badges.join('')}</div>
                                <div class="text-[10px] text-gray-500">${summary}</div>
                            </div>
                            <button class="bot-log-open-account shrink-0 whitespace-nowrap bg-gray-700 hover:bg-gray-600 text-white px-2 py-1 rounded" data-user-id="${a.user_id}" data-username="${escapeHtml(a.username || '')}">記録を見る</button>
                        </div>`;
                }).join('');
            };
            const loadAccounts = async () => {
                const list = get('bot-log-account-list');
                if (!list) return;
                list.innerHTML = '<div class="text-xs text-gray-400 py-2"><i class="fas fa-spinner fa-spin mr-1"></i>読み込み中...</div>';
                const search = get('bot-log-search');
                const q = search ? search.value.trim() : '';
                try {
                    const res = await apiFetch(`/api/bot/evidence/accounts?q=${encodeURIComponent(q)}`, { cache: 'no-store' });
                    const data = await res.json().catch(() => ({}));
                    if (!res.ok) throw new Error(data.error || String(res.status));
                    listState.accounts = data.accounts || [];
                    const visible = new Set(listState.accounts.map(a => a.user_id));
                    Array.from(listState.selected).forEach((id) => { if (!visible.has(id)) listState.selected.delete(id); });
                    renderAccounts();
                } catch (err) {
                    list.innerHTML = '<div class="text-xs text-red-400">アカウント一覧を取得できませんでした。</div>';
                    showToast('Bot検出ログの一覧を取得できませんでした', 'error', true);
                }
            };

            // ---- アカウントごとの状態と記録 ----
            const renderState = (st) => {
                if (!st) return '';
                const locks = st.locks || [];
                const idents = st.banned_identifiers || [];
                const badges = [];
                if (!st.exists) badges.push(badge('削除済み', 'bg-gray-600'));
                if (st.exists) {
                    badges.push(badge(st.detection_enabled ? '検出ON' : '検出OFF', st.detection_enabled ? 'bg-gray-600' : 'bg-gray-700'));
                    badges.push(badge(st.is_bot_banned ? 'BAN中' : 'BANなし', st.is_bot_banned ? 'bg-red-600' : 'bg-gray-600'));
                }
                badges.push(badge(locks.length ? 'ロック中' : 'ロックなし', locks.length ? 'bg-yellow-600' : 'bg-gray-600'));
                if (idents.length) badges.push(badge(`IP・端末のBAN ${idents.length}件`, 'bg-red-600'));
                if (st.is_admin) badges.push(badge('管理者（監視の対象外）', 'bg-blue-600'));
                const hasBanRecord = !!(state.counts.ban || state.counts.related_ban);
                const banLine = st.is_bot_banned
                    ? `<div class="text-gray-300">BANの理由: ${escapeHtml(st.bot_ban_reason || '-')}（${escapeHtml(formatTime(st.bot_banned_at))}）</div>${hasBanRecord ? '' : '<div class="text-gray-400">このBANの記録はありません（記録を始める前のBANか、記録が削除されています）。</div>'}`
                    : '';
                const lockLines = locks.map((lock) => {
                    const target = SOURCE_LABELS[lock.source] || lock.source;
                    const ident = lock.identifier ? ` ${escapeHtml(lock.identifier)}${lock.source === 'cookie' ? '…' : ''}` : '';
                    const origin = lock.origin
                        ? `・かけたアカウント: ${escapeHtml(lock.origin.username || String(lock.origin.user_id || '-'))}（${escapeHtml(CLIENT_LABELS[lock.origin.client] || lock.origin.client || '-')}）`
                        : '';
                    return `<div class="text-gray-300">ロック: ${escapeHtml(target)}${ident}・残り${escapeHtml(formatDuration(lock.remaining_seconds))}・${escapeHtml(lock.reason || '')}${origin}</div>`;
                }).join('');
                const identLines = idents.map((row) => `<div class="text-gray-300">IP・端末のBAN: ${escapeHtml(SOURCE_LABELS[row.kind] || row.kind)} ${escapeHtml(row.identifier)}${row.kind === 'cookie' ? '…' : ''}・${escapeHtml(row.reason || '-')}（${escapeHtml(formatTime(row.created_at))}）</div>`).join('');
                const identNote = idents.length ? '<div class="text-gray-400">IP・端末のBANがあると、同じIPアドレスや端末からは新しいアカウントを作れません。</div>' : '';
                const verified = st.turnstile_verified_seconds > 0
                    ? `確認済み（残り${formatDuration(st.turnstile_verified_seconds)}）`
                    : '未確認';
                const appeals = st.appeal_count ? `・異議申し立て ${escapeHtml(String(st.appeal_count))}件` : '';
                const actions = [];
                if (locks.length) actions.push('<button class="bot-log-unlock bg-yellow-600 hover:bg-yellow-500 text-white px-2 py-1 rounded">ロックを解除</button>');
                if (idents.length) actions.push('<button class="bot-log-unblock bg-red-600 hover:bg-red-500 text-white px-2 py-1 rounded">IP・端末のBANを解除</button>');
                return `
                    <div class="bg-gray-900 border border-gray-700 rounded p-2 text-xs space-y-1">
                        <div class="flex flex-wrap items-center gap-1">${badges.join('')}</div>
                        ${banLine}
                        ${lockLines}
                        ${identLines}
                        ${identNote}
                        <div class="text-gray-400">ロック回数（1時間）: ${escapeHtml(String(st.lock_count))} / ${escapeHtml(String(st.lock_count_limit))}・Turnstile: ${escapeHtml(verified)}・失敗 ${escapeHtml(String(st.turnstile_fail_count))} / ${escapeHtml(String(st.turnstile_fail_limit))}・判定スコア ${escapeHtml(String(Math.round((st.score || 0) * 10) / 10))}（操作 ${escapeHtml(String(Math.round((st.behavior_score || 0) * 10) / 10))}）${appeals}</div>
                        ${actions.length ? `<div class="flex flex-wrap gap-2 pt-1">${actions.join('')}</div>` : ''}
                    </div>`;
            };

            const renderFilters = () => {
                const chip = (type, label, count) => {
                    const active = state.type === type;
                    return `<button class="bot-log-filter ${active ? 'bg-blue-600 hover:bg-blue-500' : 'bg-gray-700 hover:bg-gray-600'} text-white px-2 py-1 rounded" data-type="${escapeHtml(type)}">${escapeHtml(label)} (${count})</button>`;
                };
                const chips = [chip('', 'すべて', state.total)];
                Object.keys(state.counts).sort((a, b) => state.counts[b] - state.counts[a]).forEach((type) => {
                    chips.push(chip(type, (EVENT_LABELS[type] || { label: type }).label, state.counts[type]));
                });
                return chips.join('');
            };

            const renderItem = (item) => {
                const ev = EVENT_LABELS[item.event_type] || { label: item.event_type, cls: 'bg-gray-600' };
                const client = item.client ? `<span class="bg-gray-700 text-gray-200 px-1.5 py-0.5 rounded">${escapeHtml(CLIENT_LABELS[item.client] || item.client)}</span>` : '';
                const checked = state.selected.has(item.id) ? 'checked' : '';
                const score = (item.score !== null && item.score !== undefined)
                    ? `<span class="text-gray-400">スコア ${escapeHtml(String(item.score))}${item.behavior_score !== null && item.behavior_score !== undefined ? `（操作 ${escapeHtml(String(item.behavior_score))}）` : ''}</span>`
                    : '';
                const reasons = item.reasons ? `<div class="mt-1 text-gray-300 bot-log-wrap">理由: ${escapeHtml(item.reasons)}</div>` : '';
                const source = (item.ip_address || item.user_agent)
                    ? `<div class="mt-1 text-[10px] text-gray-500 bot-log-wrap">${item.ip_address ? 'IP ' + escapeHtml(item.ip_address) : ''}${item.ip_address && item.user_agent ? '・' : ''}${escapeHtml(item.user_agent || '')}</div>`
                    : '';
                return `
                    <div class="bg-gray-900 border border-gray-700 rounded p-2 text-xs" data-log-id="${item.id}">
                        <div class="flex flex-wrap items-center gap-1">
                            <input type="checkbox" class="bot-log-select" data-log-id="${item.id}" ${checked} aria-label="この記録を選択">
                            <span class="text-gray-400">${escapeHtml(formatTime(item.created_at))}</span>
                            <span class="${ev.cls} text-white px-1.5 py-0.5 rounded">${escapeHtml(ev.label)}</span>
                            ${client}
                            ${score}
                            <button class="bot-log-delete-one ml-auto text-gray-400 hover:text-white px-1" data-log-id="${item.id}" title="この記録を削除" aria-label="この記録を削除"><i class="fas fa-trash"></i></button>
                        </div>
                        ${reasons}
                        ${renderDetails(item.details)}
                        ${source}
                    </div>`;
            };

            const render = () => {
                const el = view();
                if (!el) return;
                const items = state.items.length
                    ? state.items.map(renderItem).join('')
                    : '<div class="text-xs text-gray-400 py-2">記録はありません。</div>';
                const allChecked = state.items.length && state.items.every(item => state.selected.has(item.id));
                el.innerHTML = `
                    <div class="flex flex-wrap items-center gap-2 mb-2">
                        <button class="bot-log-back bg-gray-700 hover:bg-gray-600 text-white px-2 py-1 rounded text-xs"><i class="fas fa-arrow-left mr-1"></i>一覧に戻る</button>
                        <div class="text-sm font-bold text-white bot-log-wrap">${escapeHtml(state.username || ('ID ' + state.userId))}</div>
                        <button class="bot-log-reload ml-auto bg-gray-700 hover:bg-gray-600 text-white px-2 py-1 rounded text-xs">更新</button>
                    </div>
                    <div class="flex-1 overflow-y-auto space-y-2">
                        ${renderState(state.data && state.data.state)}
                        <div class="flex flex-wrap gap-1 text-xs">${renderFilters()}</div>
                        <div class="flex flex-wrap items-center gap-2 text-xs">
                            <label class="flex items-center gap-1 text-gray-300"><input type="checkbox" class="bot-log-select-all" aria-label="表示中の記録をすべて選択" ${allChecked ? 'checked' : ''}>表示中をすべて選択</label>
                            <button class="bot-log-delete-selected bg-red-600 hover:bg-red-500 text-white px-2 py-1 rounded" ${state.selected.size ? '' : 'disabled'}>選択した記録を削除${state.selected.size ? `（${state.selected.size}件）` : ''}</button>
                            <button class="bot-log-delete-all bg-red-800 hover:bg-red-700 text-white px-2 py-1 rounded" ${state.total ? '' : 'disabled'}>すべての記録を削除</button>
                        </div>
                        <div class="space-y-2">${items}</div>
                        ${state.hasMore ? '<button class="bot-log-more w-full bg-gray-700 hover:bg-gray-600 text-white px-2 py-1.5 rounded text-xs">さらに読み込む</button>' : ''}
                    </div>`;
            };

            const load = async (append = false) => {
                const el = view();
                if (!el || !state.userId) return;
                if (!append) {
                    el.innerHTML = '<div class="text-xs text-gray-400 py-2"><i class="fas fa-spinner fa-spin mr-1"></i>読み込み中...</div>';
                }
                const params = new URLSearchParams({ user_id: String(state.userId), limit: String(PAGE_SIZE) });
                if (state.type) params.set('type', state.type);
                if (append && state.items.length) params.set('before_id', String(state.items[state.items.length - 1].id));
                try {
                    const res = await apiFetch(`/api/bot/evidence?${params.toString()}`, { cache: 'no-store' });
                    const data = await res.json().catch(() => ({}));
                    if (!res.ok) throw new Error(data.error || String(res.status));
                    state.counts = data.counts || {};
                    state.total = data.total || 0;
                    state.hasMore = !!data.has_more;
                    if (data.user) {
                        state.username = data.user.username || state.username;
                        state.exists = !!data.user.exists;
                    }
                    if (append) {
                        state.items = state.items.concat(data.items || []);
                    } else {
                        state.items = data.items || [];
                        state.data = data;
                    }
                    render();
                } catch (err) {
                    if (!append) el.innerHTML = '<div class="text-xs text-red-400">記録の取得に失敗しました。</div>';
                    showToast('ボット検出の記録を取得できませんでした', 'error', true);
                }
            };

            const deleteLogs = async (payload, confirmText) => {
                if (!confirm(confirmText)) return;
                try {
                    const data = await postJson('/api/bot/evidence/delete', Object.assign({ user_id: state.userId }, payload));
                    showToast(`${data.deleted || 0}件の記録を削除しました`, 'success');
                    state.selected.clear();
                    await load(false);
                } catch (err) {
                    showToast('記録を削除できませんでした', 'error', true);
                }
            };

            const runAccountAction = async (url, confirmText, doneText) => {
                if (!confirm(confirmText)) return;
                try {
                    await postJson(url, { user_id: state.userId });
                    showToast(doneText, 'success');
                    await load(false);
                } catch (err) {
                    showToast('操作に失敗しました', 'error', true);
                }
            };

            const showList = () => {
                const el = view();
                if (el) { el.classList.add('hidden'); el.classList.remove('flex'); el.innerHTML = ''; }
                const lv = listView();
                if (lv) { lv.classList.remove('hidden'); lv.classList.add('flex'); }
                state.userId = 0;
            };

            const openList = async () => {
                showList();
                await loadAccounts();
            };

            const open = async (userId, username = '') => {
                const el = view();
                if (!el) return;
                Object.assign(state, { userId: Number(userId) || 0, username, exists: false, type: '', items: [], hasMore: false, counts: {}, total: 0, data: null });
                state.selected.clear();
                const lv = listView();
                if (lv) { lv.classList.add('hidden'); lv.classList.remove('flex'); }
                el.classList.remove('hidden');
                el.classList.add('flex');
                await load(false);
            };

            const bind = () => {
                const modal = get('bot-log-modal');
                if (!modal || modal.dataset.bound === '1') return;
                modal.dataset.bound = '1';
                const search = get('bot-log-search');
                if (search) search.addEventListener('keydown', (e) => { if (e.key === 'Enter') loadAccounts(); });
                modal.addEventListener('change', (e) => {
                    const target = e.target;
                    if (!target || !target.classList) return;
                    if (target.classList.contains('bot-log-select')) {
                        const id = Number(target.getAttribute('data-log-id'));
                        if (target.checked) state.selected.add(id); else state.selected.delete(id);
                        render();
                    } else if (target.classList.contains('bot-log-select-all')) {
                        state.items.forEach((item) => { if (target.checked) state.selected.add(item.id); else state.selected.delete(item.id); });
                        render();
                    } else if (target.classList.contains('bot-log-account-select')) {
                        const id = Number(target.getAttribute('data-user-id'));
                        if (target.checked) listState.selected.add(id); else listState.selected.delete(id);
                        renderAccounts();
                    } else if (target.classList.contains('bot-log-account-select-all')) {
                        listState.accounts.forEach((a) => { if (target.checked) listState.selected.add(a.user_id); else listState.selected.delete(a.user_id); });
                        renderAccounts();
                    }
                });
                modal.addEventListener('click', async (e) => {
                    const btn = e.target.closest('button');
                    if (!btn || btn.disabled) return;
                    if (btn.id === 'bot-log-search-btn' || btn.id === 'bot-log-refresh-btn') {
                        if (btn.id === 'bot-log-refresh-btn' && search) search.value = '';
                        await loadAccounts();
                    } else if (btn.classList.contains('bot-log-open-account')) {
                        const id = Number(btn.getAttribute('data-user-id'));
                        if (id) await open(id, btn.getAttribute('data-username') || '');
                    } else if (btn.classList.contains('bot-log-delete-accounts')) {
                        const ids = Array.from(listState.selected);
                        if (!ids.length || !confirm(`選択した${ids.length}件のアカウントの記録をすべて削除しますか？この操作は取り消せません。`)) return;
                        try {
                            const data = await postJson('/api/bot/evidence/delete', { user_ids: ids });
                            showToast(`${data.deleted || 0}件の記録を削除しました`, 'success');
                            listState.selected.clear();
                            await loadAccounts();
                        } catch (err) {
                            showToast('記録を削除できませんでした', 'error', true);
                        }
                    } else if (btn.classList.contains('bot-log-back')) {
                        await openList();
                    } else if (btn.classList.contains('bot-log-reload')) {
                        await load(false);
                    } else if (btn.classList.contains('bot-log-more')) {
                        await load(true);
                    } else if (btn.classList.contains('bot-log-filter')) {
                        state.type = btn.getAttribute('data-type') || '';
                        state.selected.clear();
                        await load(false);
                    } else if (btn.classList.contains('bot-log-delete-one')) {
                        const id = Number(btn.getAttribute('data-log-id'));
                        if (id) await deleteLogs({ ids: [id] }, 'この記録を削除しますか？この操作は取り消せません。');
                    } else if (btn.classList.contains('bot-log-delete-selected')) {
                        if (state.selected.size) await deleteLogs({ ids: Array.from(state.selected) }, `選択した${state.selected.size}件の記録を削除しますか？この操作は取り消せません。`);
                    } else if (btn.classList.contains('bot-log-delete-all')) {
                        await deleteLogs({ all: true }, `${state.username || 'このアカウント'} のボット検出の記録をすべて（${state.total}件）削除しますか？この操作は取り消せません。`);
                    } else if (btn.classList.contains('bot-log-unlock')) {
                        await runAccountAction('/api/bot/account/clear-lock',
                            `${state.username || 'このアカウント'} のロックを解除しますか？\nこのアカウントのIPアドレスと端末にかかっているロックも解除します。`,
                            'ロックを解除しました');
                    } else if (btn.classList.contains('bot-log-unblock')) {
                        await runAccountAction('/api/bot/account/clear-identifiers',
                            `${state.username || 'このアカウント'} が起点のIP・端末のBANを解除しますか？\n解除すると、そのIPアドレスや端末から新しいアカウントを作れるようになります。`,
                            'IP・端末のBANを解除しました');
                    }
                });
            };

            return { open, openList, showList, bind };
        })();
        window.BotAdminLog = BotAdminLog;
