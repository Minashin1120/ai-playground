
        // region: Projects (shared instructions + on-demand shared files)
        (function initProjectsModule() {
            const state = {
                projects: [],
                activeProjectUuid: null,
                activeProjectName: '',
                viewingProjectUuid: null,
                viewingProjectDetail: null,
                editingProjectUuid: null,
                movingThreadId: null,
                projectThreadsPage: 1,
                projectThreadsHasNext: false,
                projectThreadsQuery: '',
                searchTimer: null,
            };

            const byId = (id) => document.getElementById(id);

            function projectUrl(template, uuid) {
                return String(template || '').replace('/0', '/' + encodeURIComponent(String(uuid || '')));
            }

            function projectFileItemUrl(uuid, fileId) {
                const tpl = (window.CHAT_CONFIG && window.CHAT_CONFIG.urls && window.CHAT_CONFIG.urls.handleProjectFileItem) || '/api/projects/0/files/0';
                return tpl
                    .replace('/projects/0/', '/projects/' + encodeURIComponent(String(uuid || '')) + '/')
                    .replace(/\/files\/0$/, '/files/' + encodeURIComponent(String(fileId || '')));
            }

            function formatBytes(bytes) {
                const n = Number(bytes || 0);
                if (n < 1024) return `${n} B`;
                if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
                return `${(n / (1024 * 1024)).toFixed(1)} MB`;
            }

            function esc(str) {
                if (typeof escapeHtml === 'function') return escapeHtml(str == null ? '' : String(str));
                return String(str == null ? '' : str)
                    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
                    .replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
            }

            function listVisionCapableModels() {
                const results = [];
                try {
                    if (typeof MODELS !== 'undefined' && Array.isArray(MODELS)) {
                        MODELS.forEach((group) => {
                            (group.items || []).forEach((item) => {
                                const id = String(item.id || '');
                                const low = id.toLowerCase();
                                if (!id) return;
                                if (/(image|video|tts|audio|live|embedding|ocr|transcribe|lyria|veo|sora|ideogram|dall-e|imagen|search)/.test(low)) return;
                                if (low.startsWith('gemini-') || low.startsWith('gpt-') || low.startsWith('o1-') || low.startsWith('o3-') || low.startsWith('claude-') || low.startsWith('grok-') || low.startsWith('glm-')) {
                                    results.push({ id, name: item.name || id, group: group.group || '' });
                                }
                            });
                        });
                    }
                } catch (_) {}
                return results;
            }

            function populateVisionModelSelect(selectEl, selectedValue, defaultModelLabel) {
                if (!selectEl) return;
                const val = String(selectedValue || '').trim();
                const defLabel = defaultModelLabel
                    ? `ユーザー設定の既定値を使う (${defaultModelLabel})`
                    : 'ユーザー設定の既定値を使う';
                const models = listVisionCapableModels();
                let html = `<option value="">${esc(defLabel)}</option>`;
                let found = !val;
                models.forEach((m) => {
                    if (m.id === val) found = true;
                    html += `<option value="${esc(m.id)}">${esc(m.name)}</option>`;
                });
                if (val && !found) {
                    html += `<option value="${esc(val)}">${esc(val)}</option>`;
                }
                selectEl.innerHTML = html;
                selectEl.value = val;
            }

            function getDefaultVisionModelFromSettings() {
                try {
                    const sel = byId('set-default-vision-model');
                    if (sel && sel.value) return sel.value;
                    if (typeof currentVisionModel === 'string' && currentVisionModel) return currentVisionModel;
                } catch (_) {}
                return 'gemini-3-flash-preview';
            }

            function updateProjectChatBanner() {
                const banner = byId('project-chat-banner');
                const nameEl = byId('project-chat-banner-name');
                const metaEl = byId('project-chat-banner-meta');
                if (!banner) return;
                const viewEl = byId('project-view');
                const viewVisible = !!(viewEl && !viewEl.classList.contains('hidden'));
                if (!state.activeProjectUuid || viewVisible) {
                    banner.classList.add('hidden');
                    return;
                }
                const proj = state.projects.find((p) => p.uuid === state.activeProjectUuid);
                const label = state.activeProjectName || (proj && proj.name) || 'Project';
                if (nameEl) nameEl.textContent = label;
                if (metaEl) {
                    const fc = proj ? Number(proj.file_count || 0) : 0;
                    metaEl.textContent = fc > 0 ? `共有ファイル ${fc}件` : '';
                }
                banner.classList.remove('hidden');
            }

            function renderSidebarProjects() {
                const listEl = byId('project-list');
                if (!listEl) return;
                if (!state.projects.length) {
                    listEl.innerHTML = '<div class="px-2 py-1 text-[11px] text-gray-500">プロジェクトなし</div>';
                    return;
                }
                listEl.innerHTML = '';
                state.projects.forEach((proj) => {
                    const isSelected = state.viewingProjectUuid === proj.uuid || state.activeProjectUuid === proj.uuid;
                    const row = document.createElement('div');
                    row.className = `px-2 py-1.5 rounded hover:bg-gray-700 cursor-pointer text-xs text-gray-300 flex items-center justify-between gap-1 group ${isSelected ? 'bg-gray-700/60 border-l-2 border-amber-400' : ''}`;
                    row.dataset.projectUuid = proj.uuid;
                    const countText = Number(proj.thread_count || 0);
                    row.innerHTML = `
                        <div class="flex items-center gap-1.5 min-w-0 flex-1">
                            <i class="fas fa-folder text-amber-300/90 text-[11px] shrink-0"></i>
                            <span class="truncate">${esc(proj.name)}</span>
                        </div>
                        <div class="flex items-center gap-1 shrink-0">
                            <span class="text-[10px] text-gray-500">${countText}</span>
                            <button type="button" class="text-gray-500 hover:text-white px-1 opacity-100 md:opacity-0 md:group-hover:opacity-100 transition" data-project-edit="${esc(proj.uuid)}" title="プロジェクトを編集">
                                <i class="fas fa-pen text-[10px]"></i>
                            </button>
                        </div>
                    `;
                    row.onclick = (e) => {
                        const editBtn = e.target && e.target.closest ? e.target.closest('[data-project-edit]') : null;
                        if (editBtn) {
                            e.stopPropagation();
                            window.openProjectModal(proj.uuid);
                            return;
                        }
                        window.openProjectView(proj.uuid);
                    };
                    listEl.appendChild(row);
                });
            }

            async function loadProjects() {
                try {
                    const url = (window.CHAT_CONFIG && window.CHAT_CONFIG.urls && window.CHAT_CONFIG.urls.handleProjects) || '/api/projects';
                    const res = await apiFetch(url);
                    if (!res.ok) return state.projects;
                    const list = await res.json();
                    if (Array.isArray(list)) {
                        state.projects = list;
                        renderSidebarProjects();
                        updateProjectChatBanner();
                    }
                } catch (err) {
                    console.error('Failed to load projects:', err);
                }
                return state.projects;
            }

            function renderProjectFiles(project) {
                const listEl = byId('project-view-file-list');
                const countEl = byId('project-view-file-count');
                if (!listEl) return;
                const files = (project && Array.isArray(project.files)) ? project.files : [];
                if (countEl) countEl.textContent = `(${files.length})`;
                if (!files.length) {
                    listEl.innerHTML = '<div class="text-xs text-gray-500 py-3 text-center">共有ファイルはまだありません。アップロードまたはライブラリから追加できます。</div>';
                    return;
                }
                listEl.innerHTML = '';
                files.forEach((f) => {
                    const row = document.createElement('div');
                    row.className = 'flex items-center justify-between gap-2 rounded-lg border border-gray-700/70 bg-gray-900/70 px-2.5 py-2 text-xs';
                    const iconClass = f.is_image ? 'fa-image text-purple-300' : 'fa-file-lines text-blue-300';
                    const noteHtml = f.note ? `<div class="text-[10px] text-gray-400 truncate">${esc(f.note)}</div>` : '';
                    row.innerHTML = `
                        <div class="flex items-center gap-2 min-w-0 flex-1">
                            <i class="fas ${iconClass} shrink-0"></i>
                            <div class="min-w-0 flex-1">
                                <div class="text-gray-200 truncate" title="${esc(f.filename)}">${esc(f.filename)}</div>
                                ${noteHtml}
                                <div class="text-[10px] text-gray-500">${esc(formatBytes(f.size))}${f.is_image ? ' · 画像' : ''}</div>
                            </div>
                        </div>
                        <div class="flex items-center gap-1 shrink-0">
                            <button type="button" class="text-gray-400 hover:text-blue-300 p-1" data-pf-note="${f.id}" title="メモを編集"><i class="fas fa-note-sticky text-[11px]"></i></button>
                            <button type="button" class="text-gray-400 hover:text-red-400 p-1" data-pf-remove="${f.id}" title="プロジェクトから外す"><i class="fas fa-times text-[11px]"></i></button>
                        </div>
                    `;
                    row.onclick = async (e) => {
                        const noteBtn = e.target && e.target.closest ? e.target.closest('[data-pf-note]') : null;
                        if (noteBtn) {
                            e.stopPropagation();
                            const nextNote = prompt('このファイルの短い説明・メモ（モデルが参照判断に使います）:', f.note || '');
                            if (nextNote === null) return;
                            try {
                                const r = await apiFetch(projectFileItemUrl(project.uuid, f.id), {
                                    method: 'PUT',
                                    headers: { 'Content-Type': 'application/json' },
                                    body: JSON.stringify({ note: nextNote }),
                                });
                                if (r.ok) {
                                    await refreshViewingProject();
                                    showToast('ファイルのメモを更新しました', 'success');
                                } else {
                                    showToast('メモの更新に失敗しました', 'error', true);
                                }
                            } catch (_) {
                                showToast('メモの更新に失敗しました', 'error', true);
                            }
                            return;
                        }
                        const remBtn = e.target && e.target.closest ? e.target.closest('[data-pf-remove]') : null;
                        if (remBtn) {
                            e.stopPropagation();
                            if (!confirm(`「${f.filename}」をこのプロジェクトから外しますか？\n（ファイルライブラリの実体は残ります）`)) return;
                            try {
                                const r = await apiFetch(projectFileItemUrl(project.uuid, f.id), { method: 'DELETE' });
                                if (r.ok) {
                                    await refreshViewingProject();
                                    await loadProjects();
                                    showToast('プロジェクトからファイルを外しました', 'success');
                                } else {
                                    showToast('ファイルの解除に失敗しました', 'error', true);
                                }
                            } catch (_) {
                                showToast('ファイルの解除に失敗しました', 'error', true);
                            }
                        }
                    };
                    listEl.appendChild(row);
                });
            }

            function renderProjectVisionBar(project) {
                const toggle = byId('project-view-vision-toggle');
                const wrap = byId('project-view-vision-model-wrap');
                const sel = byId('project-view-vision-model-select');
                const hint = byId('project-view-vision-hint');
                if (!toggle) return;
                const enabled = !!(project && project.use_vision_for_images);
                toggle.checked = enabled;
                if (wrap) wrap.classList.toggle('hidden', !enabled);
                if (sel) populateVisionModelSelect(sel, project ? project.vision_model : '', getDefaultVisionModelFromSettings());
                if (hint) {
                    hint.textContent = enabled
                        ? `ON: 共有画像が参照されたとき ${project.effective_vision_model || getDefaultVisionModelFromSettings()} でテキスト解析します。`
                        : 'OFFのときは共有画像の中身を読まず、メタ情報と案内のみ返します。';
                }
            }

            async function loadProjectThreads(uuid, append = false) {
                const listEl = byId('project-view-thread-list');
                const moreBtn = byId('project-view-thread-more-btn');
                if (!listEl || !uuid) return;
                if (!append) {
                    state.projectThreadsPage = 1;
                    state.projectThreadsHasNext = false;
                }
                const base = projectUrl(window.CHAT_CONFIG?.urls?.handleProjectThreads || '/api/projects/0/threads', uuid);
                const url = `${base}?q=${encodeURIComponent(state.projectThreadsQuery || '')}&page=${state.projectThreadsPage}`;
                try {
                    const res = await apiFetch(url);
                    if (!res.ok) return;
                    const data = await res.json();
                    const items = Array.isArray(data.threads) ? data.threads : [];
                    if (!append) listEl.innerHTML = '';
                    if (!append && !items.length) {
                        listEl.innerHTML = '<div class="text-xs text-gray-500 py-4 text-center">このプロジェクトにはまだチャットがありません。「新規チャット」から始められます。</div>';
                    } else {
                        items.forEach((t) => {
                            const tid = String(t.id);
                            const row = document.createElement('div');
                            const star = t.is_bookmarked ? 'text-yellow-400' : 'text-gray-500';
                            const tempBadge = t.is_temporary ? '<span class="text-[9px] text-amber-300 border border-amber-500/50 rounded px-1">一時</span>' : '';
                            row.className = 'p-2 rounded-lg hover:bg-gray-700/70 cursor-pointer text-xs text-gray-200 flex items-center justify-between gap-2 group border border-transparent hover:border-gray-700';
                            row.innerHTML = `
                                <div class="flex items-center gap-1.5 min-w-0 flex-1">
                                    <button type="button" class="${star} hover:text-yellow-400 px-1" data-pt-star="${esc(tid)}"><i class="fas fa-star text-[10px]"></i></button>
                                    <span class="truncate">${esc(t.title || 'New Chat')}</span>
                                    ${tempBadge}
                                </div>
                                <div class="flex items-center gap-1 shrink-0 opacity-100 md:opacity-0 md:group-hover:opacity-100 transition">
                                    <button type="button" class="text-gray-400 hover:text-amber-300 px-1" data-pt-move="${esc(tid)}" title="プロジェクト移動・解除"><i class="fas fa-folder-open text-[11px]"></i></button>
                                    <button type="button" class="text-gray-400 hover:text-white px-1" data-pt-rename="${esc(tid)}" title="名前変更"><i class="fas fa-pen text-[11px]"></i></button>
                                    <button type="button" class="text-gray-400 hover:text-red-400 px-1" data-pt-delete="${esc(tid)}" title="削除"><i class="fas fa-trash text-[11px]"></i></button>
                                </div>
                            `;
                            row.onclick = async (e) => {
                                const starBtn = e.target?.closest?.('[data-pt-star]');
                                if (starBtn) {
                                    e.stopPropagation();
                                    if (typeof toggleBookmark === 'function') await toggleBookmark(e, tid);
                                    await loadProjectThreads(uuid, false);
                                    return;
                                }
                                const moveBtn = e.target?.closest?.('[data-pt-move]');
                                if (moveBtn) {
                                    e.stopPropagation();
                                    window.openMoveThreadProjectModal(e, tid, t.title || 'New Chat', uuid);
                                    return;
                                }
                                const renBtn = e.target?.closest?.('[data-pt-rename]');
                                if (renBtn) {
                                    e.stopPropagation();
                                    if (typeof renameThread === 'function') await renameThread(e, tid);
                                    setTimeout(() => loadProjectThreads(uuid, false), 300);
                                    return;
                                }
                                const delBtn = e.target?.closest?.('[data-pt-delete]');
                                if (delBtn) {
                                    e.stopPropagation();
                                    if (typeof deleteThread === 'function') await deleteThread(e, tid);
                                    setTimeout(() => { loadProjectThreads(uuid, false); loadProjects(); }, 300);
                                    return;
                                }
                                window.hideProjectViewForChat();
                                if (typeof loadMessages === 'function') loadMessages(tid);
                            };
                            listEl.appendChild(row);
                        });
                    }
                    state.projectThreadsHasNext = !!data.has_next;
                    if (state.projectThreadsHasNext) state.projectThreadsPage++;
                    if (moreBtn) moreBtn.classList.toggle('hidden', !state.projectThreadsHasNext);
                } catch (err) {
                    console.error('Failed to load project threads:', err);
                }
            }

            async function refreshViewingProject() {
                if (!state.viewingProjectUuid) return null;
                const url = projectUrl(window.CHAT_CONFIG?.urls?.handleProjectItem || '/api/projects/0', state.viewingProjectUuid);
                const res = await apiFetch(url);
                if (!res.ok) return null;
                const detail = await res.json();
                state.viewingProjectDetail = detail;
                state.activeProjectUuid = detail.uuid;
                state.activeProjectName = detail.name;
                if (byId('project-view-name')) byId('project-view-name').textContent = detail.name || '';
                const descEl = byId('project-view-desc');
                if (descEl) {
                    descEl.textContent = detail.description || '';
                    descEl.classList.toggle('hidden', !detail.description);
                }
                const instEl = byId('project-view-instruction-preview');
                if (instEl) {
                    instEl.textContent = (detail.instruction && String(detail.instruction).trim())
                        ? detail.instruction
                        : '（未設定）';
                }
                renderProjectVisionBar(detail);
                renderProjectFiles(detail);
                return detail;
            }

            window.openProjectView = async (uuid, opts = {}) => {
                if (!uuid) return;
                state.viewingProjectUuid = String(uuid);
                state.activeProjectUuid = String(uuid);
                const proj = state.projects.find((p) => p.uuid === state.viewingProjectUuid);
                if (proj) state.activeProjectName = proj.name;
                const viewEl = byId('project-view');
                if (viewEl) viewEl.classList.remove('hidden');
                if (byId('welcome-screen')) byId('welcome-screen').classList.add('hidden');
                updateProjectChatBanner();
                renderSidebarProjects();
                if (!opts.skipHistory) {
                    const targetPath = '/p/' + encodeURIComponent(state.viewingProjectUuid);
                    if (location.pathname !== targetPath) {
                        history.pushState({ projectUuid: state.viewingProjectUuid }, '', targetPath);
                    }
                }
                if (window.innerWidth < 768 && byId('overlay')) {
                    try { byId('overlay').click(); } catch (_) {}
                }
                state.projectThreadsQuery = '';
                if (byId('project-view-thread-search')) byId('project-view-thread-search').value = '';
                await refreshViewingProject();
                await loadProjectThreads(state.viewingProjectUuid, false);
            };

            window.hideProjectViewForChat = () => {
                const viewEl = byId('project-view');
                if (viewEl) viewEl.classList.add('hidden');
                state.viewingProjectUuid = null;
                updateProjectChatBanner();
                renderSidebarProjects();
            };

            window.closeProjectView = (opts = {}) => {
                const viewEl = byId('project-view');
                if (viewEl) viewEl.classList.add('hidden');
                state.viewingProjectUuid = null;
                state.viewingProjectDetail = null;
                if (!opts.keepContext) {
                    state.activeProjectUuid = null;
                    state.activeProjectName = '';
                }
                updateProjectChatBanner();
                renderSidebarProjects();
                if (!opts.skipHistory && location.pathname.startsWith('/p/')) {
                    history.pushState({}, '', '/');
                }
                if (!opts.keepWelcome && !currentThreadId && byId('welcome-screen')) {
                    byId('welcome-screen').classList.remove('hidden');
                }
            };

            window.setActiveProjectContext = (uuid, name) => {
                state.activeProjectUuid = uuid ? String(uuid) : null;
                state.activeProjectName = name ? String(name) : '';
                updateProjectChatBanner();
                renderSidebarProjects();
            };

            window.getActiveProjectId = () => state.activeProjectUuid || null;

            window.openProjectModal = async (editUuid = null) => {
                state.editingProjectUuid = editUuid ? String(editUuid) : null;
                const titleEl = byId('project-modal-title');
                const nameEl = byId('project-name-input');
                const descEl = byId('project-desc-input');
                const instEl = byId('project-inst-input');
                const visToggle = byId('project-use-vision-toggle');
                const visWrap = byId('project-vision-model-wrap');
                const visSel = byId('project-vision-model-select');
                let target = null;
                if (state.editingProjectUuid) {
                    target = (state.viewingProjectDetail && state.viewingProjectDetail.uuid === state.editingProjectUuid)
                        ? state.viewingProjectDetail
                        : state.projects.find((p) => p.uuid === state.editingProjectUuid);
                    if (!target) {
                        try {
                            const r = await apiFetch(projectUrl(window.CHAT_CONFIG?.urls?.handleProjectItem || '/api/projects/0', state.editingProjectUuid));
                            if (r.ok) target = await r.json();
                        } catch (_) {}
                    }
                }
                if (titleEl) titleEl.textContent = target ? 'Edit Project' : 'Create New Project';
                if (nameEl) nameEl.value = target ? (target.name || '') : '';
                if (descEl) descEl.value = target ? (target.description || '') : '';
                if (instEl) instEl.value = target ? (target.instruction || '') : '';
                const useVision = !!(target && target.use_vision_for_images);
                if (visToggle) visToggle.checked = useVision;
                if (visWrap) visWrap.classList.toggle('hidden', !useVision);
                if (visSel) populateVisionModelSelect(visSel, target ? target.vision_model : '', getDefaultVisionModelFromSettings());
                showModal('project-modal');
                if (nameEl) setTimeout(() => nameEl.focus(), 50);
            };

            window.closeProjectModal = (skipHistory = false) => {
                hideModal('project-modal');
                state.editingProjectUuid = null;
                if (!skipHistory && location.pathname === '/project') {
                    history.back();
                }
            };

            window.openMoveThreadProjectModal = async (e, threadId, threadTitle, currentProjUuid = null) => {
                if (e && typeof e.stopPropagation === 'function') e.stopPropagation();
                state.movingThreadId = String(threadId || '');
                if (!state.projects.length) await loadProjects();
                const nameEl = byId('move-thread-project-chat-name');
                if (nameEl) nameEl.textContent = threadTitle || 'チャット';
                const sel = byId('move-thread-project-select');
                if (sel) {
                    let html = '<option value="">（未所属 / 通常の履歴へ戻す）</option>';
                    state.projects.forEach((p) => {
                        html += `<option value="${esc(p.uuid)}">${esc(p.name)}</option>`;
                    });
                    sel.innerHTML = html;
                    sel.value = currentProjUuid ? String(currentProjUuid) : '';
                }
                showModal('move-thread-project-modal');
            };

            window.closeMoveThreadProjectModal = () => {
                hideModal('move-thread-project-modal');
                state.movingThreadId = null;
            };

            window.syncThreadProjectSettingsModal = (threadSettings) => {
                const sel = byId('thread-project-select');
                const controls = byId('thread-project-instruction-controls');
                const toggle = byId('thread-include-project-instruction');
                const preview = byId('thread-project-instruction-preview');
                const openBtn = byId('thread-open-project-btn');
                if (!sel) return;
                let html = '<option value="">（未所属）</option>';
                state.projects.forEach((p) => {
                    html += `<option value="${esc(p.uuid)}">${esc(p.name)}</option>`;
                });
                const pid = (threadSettings && threadSettings.project_id) ? String(threadSettings.project_id) : '';
                if (pid && !state.projects.some((p) => p.uuid === pid)) {
                    html += `<option value="${esc(pid)}">${esc(threadSettings.project_name || pid)}</option>`;
                }
                sel.innerHTML = html;
                sel.value = pid;
                if (toggle) toggle.checked = threadSettings ? threadSettings.include_project_instruction !== false : true;
                const updateControls = () => {
                    const cur = sel.value || '';
                    if (controls) controls.classList.toggle('hidden', !cur);
                    if (preview) {
                        if (cur && threadSettings && cur === String(threadSettings.project_id || '')) {
                            preview.value = threadSettings.project_instruction || '';
                        } else if (cur) {
                            const p = state.projects.find((item) => item.uuid === cur);
                            preview.value = p ? (p.instruction || '') : '';
                        } else {
                            preview.value = '';
                        }
                    }
                    if (openBtn) {
                        openBtn.onclick = () => {
                            if (!sel.value) return;
                            if (typeof window.closeThreadModal === 'function') window.closeThreadModal(true);
                            window.openProjectView(sel.value);
                        };
                    }
                };
                sel.onchange = updateControls;
                updateControls();
            };

            async function saveProjectVisionInline() {
                if (!state.viewingProjectUuid) return;
                const toggle = byId('project-view-vision-toggle');
                const sel = byId('project-view-vision-model-select');
                const useVision = !!(toggle && toggle.checked);
                const visionModel = sel ? (sel.value || '') : '';
                try {
                    const url = projectUrl(window.CHAT_CONFIG?.urls?.handleProjectItem || '/api/projects/0', state.viewingProjectUuid);
                    const res = await apiFetch(url, {
                        method: 'PUT',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ use_vision_for_images: useVision, vision_model: visionModel }),
                    });
                    if (res.ok) {
                        const updated = await res.json();
                        state.viewingProjectDetail = updated;
                        renderProjectVisionBar(updated);
                        await loadProjects();
                        showToast('画像のVisionモデル設定を保存しました', 'success');
                    } else {
                        showToast('Visionモデル設定の保存に失敗しました', 'error', true);
                    }
                } catch (_) {
                    showToast('Visionモデル設定の保存に失敗しました', 'error', true);
                }
            }

            async function attachFilepathsToViewingProject(filepaths) {
                if (!state.viewingProjectUuid || !Array.isArray(filepaths) || !filepaths.length) return;
                const url = projectUrl(window.CHAT_CONFIG?.urls?.handleProjectFiles || '/api/projects/0/files', state.viewingProjectUuid);
                try {
                    const res = await apiFetch(url, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ filepaths }),
                    });
                    const data = await res.json().catch(() => ({}));
                    if (!res.ok) {
                        showToast(data.error === 'project_file_limit_exceeded' ? '共有ファイルの上限（50件）に達しています' : 'ファイルの追加に失敗しました', 'error', true);
                        return;
                    }
                    const addedCount = Array.isArray(data.added) ? data.added.length : 0;
                    await refreshViewingProject();
                    await loadProjects();
                    if (addedCount > 0) showToast(`共有ファイルを ${addedCount} 件追加しました`, 'success');
                    else showToast('選択したファイルは既に追加済みです', 'info');
                } catch (_) {
                    showToast('ファイルの追加に失敗しました', 'error', true);
                }
            }

            async function uploadFilesForViewingProject(fileList) {
                if (!state.viewingProjectUuid || !fileList || !fileList.length) return;
                const uploadUrl = (window.CHAT_CONFIG && window.CHAT_CONFIG.urls && window.CHAT_CONFIG.urls.upload) || '/upload';
                const uploadedPaths = [];
                for (const file of Array.from(fileList)) {
                    const form = new FormData();
                    form.append('file', file);
                    try {
                        const res = await apiFetch(uploadUrl, { method: 'POST', body: form });
                        const data = await res.json().catch(() => ({}));
                        const ref = data.filepath || data.path || data.url;
                        if (res.ok && ref) {
                            uploadedPaths.push(ref);
                        } else {
                            showToast(`アップロード失敗: ${file.name}`, 'error', true);
                        }
                    } catch (_) {
                        showToast(`アップロード失敗: ${file.name}`, 'error', true);
                    }
                }
                if (uploadedPaths.length) {
                    await attachFilepathsToViewingProject(uploadedPaths);
                }
            }

            function bindProjectsUi() {
                if (byId('add-project-btn')) {
                    byId('add-project-btn').onclick = () => window.openProjectModal(null);
                }
                if (byId('project-use-vision-toggle')) {
                    byId('project-use-vision-toggle').onchange = () => {
                        const wrap = byId('project-vision-model-wrap');
                        if (wrap) wrap.classList.toggle('hidden', !byId('project-use-vision-toggle').checked);
                    };
                }
                if (byId('save-project-btn')) {
                    byId('save-project-btn').onclick = async () => {
                        const name = (byId('project-name-input')?.value || '').trim();
                        const description = (byId('project-desc-input')?.value || '').trim();
                        const instruction = byId('project-inst-input')?.value || '';
                        const use_vision_for_images = !!(byId('project-use-vision-toggle')?.checked);
                        const vision_model = byId('project-vision-model-select')?.value || '';
                        if (!name) {
                            showToast('プロジェクト名を入力してください', 'error', true);
                            return;
                        }
                        const isEdit = !!state.editingProjectUuid;
                        const url = isEdit
                            ? projectUrl(window.CHAT_CONFIG?.urls?.handleProjectItem || '/api/projects/0', state.editingProjectUuid)
                            : (window.CHAT_CONFIG?.urls?.handleProjects || '/api/projects');
                        try {
                            const res = await apiFetch(url, {
                                method: isEdit ? 'PUT' : 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({ name, description, instruction, use_vision_for_images, vision_model }),
                            });
                            const data = await res.json().catch(() => ({}));
                            if (!res.ok) {
                                showToast(data.error === 'name_taken' ? '同じ名前のプロジェクトが既にあります' : 'プロジェクトの保存に失敗しました', 'error', true);
                                return;
                            }
                            window.closeProjectModal(true);
                            await loadProjects();
                            if (isEdit && state.viewingProjectUuid === data.uuid) {
                                await refreshViewingProject();
                            } else if (!isEdit && data.uuid) {
                                await window.openProjectView(data.uuid);
                            }
                            showToast(isEdit ? 'プロジェクトを更新しました' : 'プロジェクトを作成しました', 'success');
                        } catch (_) {
                            showToast('プロジェクトの保存に失敗しました', 'error', true);
                        }
                    };
                }
                if (byId('save-move-thread-project-btn')) {
                    byId('save-move-thread-project-btn').onclick = async () => {
                        if (!state.movingThreadId) return;
                        const targetProjUuid = byId('move-thread-project-select')?.value || null;
                        try {
                            const res = await apiFetch(`/api/threads/${encodeURIComponent(state.movingThreadId)}/settings`, {
                                method: 'PUT',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({ project_id: targetProjUuid }),
                            });
                            if (!res.ok) {
                                showToast('チャットの移動に失敗しました', 'error', true);
                                return;
                            }
                            const data = await res.json().catch(() => ({}));
                            if (String(currentThreadId || '') === state.movingThreadId) {
                                window.setActiveProjectContext(data.project_id || null, data.project_name || '');
                            }
                            window.closeMoveThreadProjectModal();
                            await loadProjects();
                            if (typeof loadThreads === 'function') loadThreads();
                            if (state.viewingProjectUuid) await loadProjectThreads(state.viewingProjectUuid, false);
                            showToast('チャットの所属プロジェクトを更新しました', 'success');
                        } catch (_) {
                            showToast('チャットの移動に失敗しました', 'error', true);
                        }
                    };
                }
                if (byId('project-chat-banner-open')) {
                    byId('project-chat-banner-open').onclick = () => {
                        if (state.activeProjectUuid) window.openProjectView(state.activeProjectUuid);
                    };
                }
                if (byId('project-view-back-btn')) {
                    byId('project-view-back-btn').onclick = () => {
                        window.closeProjectView();
                        if (typeof startNewChat === 'function') startNewChat();
                    };
                }
                if (byId('project-view-new-chat-btn')) {
                    byId('project-view-new-chat-btn').onclick = () => {
                        const pid = state.viewingProjectUuid;
                        const pname = (state.viewingProjectDetail && state.viewingProjectDetail.name) || state.activeProjectName;
                        window.hideProjectViewForChat();
                        if (typeof startNewChat === 'function') startNewChat({ preserveProject: true, projectId: pid, projectName: pname });
                        window.setActiveProjectContext(pid, pname);
                        const input = byId('prompt-input');
                        if (input) input.focus();
                    };
                }
                if (byId('project-view-edit-btn')) {
                    byId('project-view-edit-btn').onclick = () => {
                        if (state.viewingProjectUuid) window.openProjectModal(state.viewingProjectUuid);
                    };
                }
                if (byId('project-view-edit-inst-btn')) {
                    byId('project-view-edit-inst-btn').onclick = () => {
                        if (state.viewingProjectUuid) window.openProjectModal(state.viewingProjectUuid);
                    };
                }
                if (byId('project-view-delete-btn')) {
                    byId('project-view-delete-btn').onclick = async () => {
                        const proj = state.viewingProjectDetail;
                        if (!proj) return;
                        if (!confirm(`プロジェクト「${proj.name}」を削除しますか？\n所属していたチャットは削除されず、通常の履歴（未所属）へ戻ります。`)) return;
                        try {
                            const url = projectUrl(window.CHAT_CONFIG?.urls?.handleProjectItem || '/api/projects/0', proj.uuid);
                            const res = await apiFetch(url, { method: 'DELETE' });
                            if (!res.ok) {
                                showToast('プロジェクトの削除に失敗しました', 'error', true);
                                return;
                            }
                            window.closeProjectView();
                            await loadProjects();
                            if (typeof loadThreads === 'function') loadThreads();
                            showToast('プロジェクトを削除しました（所属チャットは未所属へ移動しました）', 'success');
                        } catch (_) {
                            showToast('プロジェクトの削除に失敗しました', 'error', true);
                        }
                    };
                }
                if (byId('project-view-vision-toggle')) {
                    byId('project-view-vision-toggle').onchange = () => {
                        const wrap = byId('project-view-vision-model-wrap');
                        if (wrap) wrap.classList.toggle('hidden', !byId('project-view-vision-toggle').checked);
                        saveProjectVisionInline();
                    };
                }
                if (byId('project-view-vision-model-select')) {
                    byId('project-view-vision-model-select').onchange = () => saveProjectVisionInline();
                }
                if (byId('project-view-upload-file-btn') && byId('project-view-file-input')) {
                    byId('project-view-upload-file-btn').onclick = () => {
                        const inp = byId('project-view-file-input');
                        if (inp) { inp.value = ''; inp.click(); }
                    };
                    byId('project-view-file-input').onchange = (e) => {
                        const files = e.target && e.target.files;
                        if (files && files.length) uploadFilesForViewingProject(files);
                    };
                }
                if (byId('project-view-lib-file-btn')) {
                    byId('project-view-lib-file-btn').onclick = () => {
                        if (!state.viewingProjectUuid) return;
                        window.__libProjectAttachCallback = async (selectedPaths) => {
                            window.__libProjectAttachCallback = null;
                            if (typeof window.closeLibModal === 'function') window.closeLibModal(true);
                            await attachFilepathsToViewingProject(selectedPaths);
                        };
                        if (typeof openLibModal === 'function') openLibModal(true);
                    };
                }
                if (byId('project-view-thread-search')) {
                    byId('project-view-thread-search').oninput = (e) => {
                        state.projectThreadsQuery = (e.target && e.target.value || '').trim();
                        if (state.searchTimer) clearTimeout(state.searchTimer);
                        state.searchTimer = setTimeout(() => {
                            if (state.viewingProjectUuid) loadProjectThreads(state.viewingProjectUuid, false);
                        }, 250);
                    };
                }
                if (byId('project-view-thread-more-btn')) {
                    byId('project-view-thread-more-btn').onclick = () => {
                        if (state.viewingProjectUuid && state.projectThreadsHasNext) {
                            loadProjectThreads(state.viewingProjectUuid, true);
                        }
                    };
                }
            }

            window.loadProjects = loadProjects;
            window.refreshViewingProjectIfOpen = () => {
                if (state.viewingProjectUuid) {
                    refreshViewingProject();
                    loadProjectThreads(state.viewingProjectUuid, false);
                }
            };

            document.addEventListener('DOMContentLoaded', () => {
                try {
                    bindProjectsUi();
                    loadProjects().then(() => {
                        const initialPid = window.CHAT_CONFIG && window.CHAT_CONFIG.initialProjectId;
                        if (initialPid) {
                            window.openProjectView(initialPid, { skipHistory: true });
                        }
                    });
                } catch (err) {
                    console.error('projects init failed', err);
                }
            }, { once: true });
        })();
