            let importFileSelection = null;
            const updateImportFileSelectionUi = () => {
                if (!importFileSelection) return;
                const files = importFileSelection.files;
                const selection = importFileSelection.selection;
                let total = 0;
                files.forEach((f) => { if (selection.has(f.archive_path)) total += (Number(f.size_bytes) || 0); });
                const available = Number(importFileSelection.available_bytes) || 0;
                const over = total > available;
                if (importFileSummary) {
                    importFileSummary.textContent = `選択中: ${importFormatBytes(total)} / 利用可能: ${importFormatBytes(available)}${over ? ' （容量超過）' : ''}`;
                    importFileSummary.classList.toggle('text-red-300', over);
                }
                if (importFileInfo) importFileInfo.textContent = `${files.length} files`;
            };
            const renderImportFileItems = () => {
                if (!importFileGrid || !importFileSelection) return;
                importFileGrid.innerHTML = '';
                const files = importFileSelection.files;
                if (!files.length) {
                    importFileGrid.innerHTML = '<div class="text-xs text-gray-500">インポート可能なファイルがありません。</div>';
                    updateImportFileSelectionUi();
                    return;
                }
                files.forEach((f) => {
                    const label = document.createElement('label');
                    const checked = importFileSelection.selection.has(f.archive_path);
                    label.className = `relative bg-gray-800 border rounded flex items-center gap-2 p-2 cursor-pointer transition hover:border-blue-500 ${checked ? 'border-blue-500' : 'border-gray-600'}`;
                    label.innerHTML =
                        `<input type="checkbox" class="import-file-check accent-blue-500 w-4 h-4 shrink-0"${checked ? ' checked' : ''}>` +
                        `<div class="min-w-0 flex-1">` +
                            `<div class="text-xs text-gray-200 truncate" title="${escapeHtml(f.display_name)}">${escapeHtml(f.display_name)}</div>` +
                            `<div class="text-[10px] text-gray-500">${importFormatBytes(f.size_bytes)}</div>` +
                        `</div>`;
                    const checkbox = label.querySelector('.import-file-check');
                    checkbox.addEventListener('change', () => {
                        if (checkbox.checked) importFileSelection.selection.add(f.archive_path);
                        else importFileSelection.selection.delete(f.archive_path);
                        label.classList.toggle('border-blue-500', checkbox.checked);
                        label.classList.toggle('border-gray-600', !checkbox.checked);
                        updateImportFileSelectionUi();
                    });
                    importFileGrid.appendChild(label);
                });
                updateImportFileSelectionUi();
            };
            const showImportFileSelection = (payload) => new Promise((resolve) => {
                importFileSelection = {
                    files: payload.files || [],
                    selection: new Set((payload.files || []).map(f => f.archive_path)),
                    available_bytes: payload.available_bytes,
                    resolve,
                };
                renderImportFileItems();
                if (!get('import-files-modal')) {
                    resolve(null);
                    return;
                }
                showModal('import-files-modal');
            });
            const closeImportFileSelection = (result) => {
                hideModal('import-files-modal');
                if (importFileSelection) {
                    const resolver = importFileSelection.resolve;
                    importFileSelection = null;
                    resolver(result);
                }
            };
            const importFileCloseBtn = get('import-files-close');
            if (importFileCloseBtn) importFileCloseBtn.onclick = () => closeImportFileSelection(null);
            const importFileCancelBtn = get('import-files-cancel');
            if (importFileCancelBtn) importFileCancelBtn.onclick = () => closeImportFileSelection(null);
            const importFileConfirmBtn = get('import-files-confirm');
            if (importFileConfirmBtn) {
                importFileConfirmBtn.onclick = () => {
                    if (!importFileSelection) return;
                    const selected = Array.from(importFileSelection.selection);
                    closeImportFileSelection(selected.length ? selected.join(',') : '__none__');
                };
            }
            const importFileSelectAllBtn = get('import-files-select-all');
            if (importFileSelectAllBtn) {
                importFileSelectAllBtn.onclick = () => {
                    if (!importFileSelection) return;
                    importFileSelection.files.forEach((f) => importFileSelection.selection.add(f.archive_path));
                    renderImportFileItems();
                };
            }
            const importFileNoneBtn = get('import-files-none');
            if (importFileNoneBtn) {
                importFileNoneBtn.onclick = () => {
                    if (!importFileSelection) return;
                    importFileSelection.selection.clear();
                    renderImportFileItems();
                };
            }

            // --- Settings import confirmation --------------------------------
            const ACCOUNT_SETTING_LABELS = {
                system_prompt: 'システムプロンプト', system_prompt_enabled: 'システムプロンプトを使用',
                apply_global_system_prompt: '全体システムプロンプトを適用', apply_auto_system_prompt_notices: '自動注入システムプロンプトを適用',
                auto_system_prompt_notices_config: '自動注入システムプロンプトの種類別設定',
                gemini_backend: 'Gemini バックエンド', gemini_vertex_location: 'Vertex AI ロケーション',
                mic_transcribe_mode: 'マイク文字起こし方式', stt_model: 'STTモデル', llm_transcribe_prompt: 'LLM文字起こしプロンプト',
                enter_to_send: 'Enterキーで送信', use_sw_cache: 'Service Workerキャッシュ',
                clear_cache_on_version_update: 'バージョン更新時キャッシュ削除', theme_color: 'テーマカラー',
                liquid_glass_enabled: 'Liquid Glass', light_mode_enabled: 'ライトモード', auto_search_on_links: 'リンクで自動検索',
                compact_prompt_mode: 'プロンプトバー表示（コンパクト）', minimal_prompt_mode: 'プロンプトバー表示（ミニマル）',
                use_last_chat_settings: '直前のチャット設定を使用', voice_studio_ui: '音声スタジオUI',
                math_steps_enabled: '数学ステップ解説', math_steps_config: '数学ステップの設定',
                default_math_steps: '既定: 数学ステップ',
                temp_chat_timeout_seconds: '一時チャットの有効時間（秒）', default_model: '既定のモデル',
                default_enable_search: '既定: Search', default_gemini_free_tier: '既定: Gemini無料枠', default_enable_url_context: '既定: URLコンテキスト',
                default_enable_maps: '既定: Maps', default_enable_python: '既定: Python',
                default_enable_file_creation: '既定: File',
                default_enable_thinking: '既定: Thinking', default_thinking_level: '既定: Thinkingレベル',
                default_thinking_budget: '既定: Thinking budget', default_reasoning_effort: '既定: Reasoning effort',
                default_enable_system_prompt: '既定: システムプロンプト', default_enable_mcp: '既定: MCP',
                default_safety_setting: '既定: 安全設定',
                default_vision_model: 'Vision Model', rich_paste_prompt_default: 'リッチ貼り付けプロンプト',
                rich_paste_prompt_use_custom_default: 'リッチ貼り付けカスタムプロンプト既定',
                last_model: '直前のモデル', last_enable_search: '直前: Search', last_gemini_free_tier: '直前: Gemini無料枠', last_enable_url_context: '直前: URLコンテキスト',
                last_enable_maps: '直前: Maps', last_enable_python: '直前: Python', last_enable_file_creation: '直前: File', last_enable_thinking: '直前: Thinking',
                last_thinking_level: '直前: Thinkingレベル', last_thinking_budget: '直前: Thinking budget',
                last_reasoning_effort: '直前: Reasoning effort', last_enable_system_prompt: '直前: システムプロンプト', last_enable_mcp: '直前: MCP',
                last_safety_setting: '直前: 安全設定', last_math_steps: '直前: 数学ステップ', enable_latency_metrics: 'レスポンス速度の計測',
                enable_client_debug_log: 'デバッグログの拡張送信',
            };
            const formatAccountSettingValue = (value) => {
                if (value === true) return 'ON';
                if (value === false) return 'OFF';
                if (value === null || value === undefined || value === '') return '未設定';
                const text = String(value);
                return text.length > 60 ? text.slice(0, 60) + '…' : text;
            };
            let settingsImportConfirmationResolver = null;
            const resolveSettingsImportConfirmation = (result) => {
                if (settingsImportConfirmationResolver) {
                    const resolver = settingsImportConfirmationResolver;
                    settingsImportConfirmationResolver = null;
                    hideModal('settings-confirmation-modal');
                    resolver(result);
                }
            };
            const showSettingsImportConfirmation = (payload) => new Promise((resolve) => {
                const modal = get('settings-confirmation-modal');
                if (!modal) {
                    // No modal markup (unexpected).  Proceed so the import is not blocked.
                    resolve(true);
                    return;
                }
                settingsImportConfirmationResolver = resolve;
                const changes = Array.isArray(payload && payload.settings_changes) ? payload.settings_changes : [];
                const listEl = get('settings-confirmation-list');
                if (listEl) {
                    if (!changes.length) {
                        listEl.innerHTML = '<div class="text-xs text-gray-400">変更される設定はありませんでした。</div>';
                    } else {
                        listEl.innerHTML = changes.map((c) => {
                            const label = ACCOUNT_SETTING_LABELS[c.field] || c.field;
                            const current = formatAccountSettingValue(c.current);
                            const incoming = formatAccountSettingValue(c.incoming);
                            return `<div class="rounded border border-gray-700 bg-gray-800/60 p-2">
                                <div class="text-xs font-bold text-gray-100">${escapeHtml(label)}</div>
                                <div class="text-[11px] text-gray-400 mt-1">現在: ${escapeHtml(current)}</div>
                                <div class="text-[11px] text-emerald-300">→ ${escapeHtml(incoming)}</div>
                            </div>`;
                        }).join('');
                    }
                }
                const countEl = get('settings-confirmation-count');
                if (countEl) countEl.textContent = `${changes.length}件の設定が変更されます`;
                showModal('settings-confirmation-modal');
            });
            const settingsConfirmationModal = get('settings-confirmation-modal');
            if (settingsConfirmationModal) {
                settingsConfirmationModal.addEventListener('click', (e) => {
                    if (e.target === settingsConfirmationModal) resolveSettingsImportConfirmation(false);
                });
            }
            const settingsConfirmationCloseBtn = get('settings-confirmation-close');
            if (settingsConfirmationCloseBtn) settingsConfirmationCloseBtn.onclick = () => resolveSettingsImportConfirmation(false);
            const settingsConfirmationCancelBtn = get('settings-confirmation-cancel');
            if (settingsConfirmationCancelBtn) settingsConfirmationCancelBtn.onclick = () => resolveSettingsImportConfirmation(false);
            const settingsConfirmationConfirmBtn = get('settings-confirmation-confirm');
            if (settingsConfirmationConfirmBtn) settingsConfirmationConfirmBtn.onclick = () => resolveSettingsImportConfirmation(true);

            const accountImportBtn = get('account-import-btn');
            const inplaceToggle = get('account-import-inplace');
            const inplaceWarn = get('account-import-inplace-warning');
            if (inplaceToggle && inplaceWarn) {
                const syncInplaceWarn = () => inplaceWarn.classList.toggle('hidden', !inplaceToggle.checked);
                inplaceToggle.addEventListener('change', syncInplaceWarn);
                syncInplaceWarn();
            }
            if (accountImportBtn) {
                accountImportBtn.onclick = async () => {
                    const input = get('account-import-file');
                    const file = input && input.files ? input.files[0] : null;
                    const categoryBox = get('account-import-categories');
                    const categories = categoryBox
                        ? Array.from(categoryBox.querySelectorAll('input[type="checkbox"]:checked')).map(el => el.value)
                        : [];
                    const inplaceBox = get('account-import-inplace');
                    const restoreInplace = !!(inplaceBox && inplaceBox.checked);
                    const settingsBypassBox = get('account-import-settings-bypass');
                    const settingsBypass = !!(settingsBypassBox && settingsBypassBox.checked);
                    let settingsConfirmed = false;
                    if (!file) {
                        showToast('インポートするZIPファイルを選択してください', 'error', true);
                        return;
                    }
                    if (!categories.length) {
                        showToast('インポートするデータを1つ以上選択してください', 'error', true);
                        return;
                    }
                    const selectedLabels = categoryBox
                        ? Array.from(categoryBox.querySelectorAll('input[type="checkbox"]:checked')).map(el => (el.closest('label') && el.closest('label').textContent || el.value).trim())
                        : categories;
                    if (!confirm(`次のデータをインポートします。既存データは削除されません。すでに同じ内容のデータがある場合はスキップされます。\n\n${selectedLabels.join('、')}${restoreInplace ? '\n※「元の場所へ復元」: このアカウントの同名ファイルを上書きします' : ''}\n\n続行しますか？`)) return;
                    const transfer = {
                        id: createAccountTransferId(), type: 'import', stopped: false,
                        controller: new AbortController()
                    };
                    activeAccountTransfer = transfer;
                    setAccountTransferControls(true);
                    renderAccountTransferProgress({progress: 0, phase: 'uploading', message: 'アップロードを準備しています'});
                    const resultBox = get('account-import-result');
                    let pollPromise = Promise.resolve(null);
                    try {
                        const chunkSize = 10 * 1024 * 1024;
                        const totalChunks = Math.max(1, Math.ceil(file.size / chunkSize));
                        const startRes = await apiFetch('/api/account/import/upload/start', manualSpinnerRequestOptions({
                            method: 'POST', headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({size: file.size}), signal: transfer.controller.signal,
                        }));
                        const startData = await startRes.json().catch(() => ({}));
                        if (!startRes.ok) throw new Error(startData.error || 'アップロードを開始できません');
                        transfer.uploadId = startData.upload_id;
                        const actualChunkSize = startData.chunk_size || chunkSize;
                        let uploadedChunks = 0;
                        let nextChunk = 0;
                        const uploadWorker = async () => {
                            while (true) {
                                const index = nextChunk++;
                                if (index >= totalChunks) return;
                                const chunk = file.slice(index * actualChunkSize, Math.min(file.size, (index + 1) * actualChunkSize));
                                const chunkForm = new FormData();
                                chunkForm.append('chunk', chunk, file.name);
                                chunkForm.append('index', String(index));
                                const chunkRes = await apiFetch(`/api/account/import/upload/${encodeURIComponent(transfer.uploadId)}/chunk`, manualSpinnerRequestOptions({
                                    method: 'POST', body: chunkForm, signal: transfer.controller.signal,
                                }));
                                const chunkData = await chunkRes.json().catch(() => ({}));
                                if (!chunkRes.ok) throw new Error(chunkData.error || 'アップロードに失敗しました');
                                uploadedChunks++;
                                renderAccountTransferProgress({progress: Math.min(35, Math.round((uploadedChunks / totalChunks) * 35)), phase: 'uploading', message: `ZIPを並列アップロードしています（${uploadedChunks}/${totalChunks}）`});
                                if (window.ConnectionMonitor) window.ConnectionMonitor.reportActivity();
                            }
                        };
                        let importUploadOpStarted = false;
                        if (window.ConnectionMonitor) {
                            window.ConnectionMonitor.operationStarted();
                            importUploadOpStarted = true;
                        }
                        try {
                            await Promise.all([uploadWorker(), uploadWorker(), uploadWorker()]);
                            const completeRes = await apiFetch(`/api/account/import/upload/${encodeURIComponent(transfer.uploadId)}/complete`, manualSpinnerRequestOptions({method: 'POST', signal: transfer.controller.signal}));
                            const completeData = await completeRes.json().catch(() => ({}));
                            if (!completeRes.ok) throw new Error(completeData.error || 'アップロードを完了できません');
                            renderAccountTransferProgress({progress: 35, phase: 'validating', message: 'ZIPを検証しています'});
                        } finally {
                            if (importUploadOpStarted && window.ConnectionMonitor) window.ConnectionMonitor.operationEnded();
                        }
                        let selectedFiles = '';
                        let importDone = false;
                        let parseFailures = 0;
                        // After a settings / API-credentials import, re-fetch the saved
                        // settings and repopulate the (possibly still-open) settings modal
                        // so imported values are reflected instead of stale pre-import
                        // form values that a later save would overwrite.  The import has
                        // already persisted the values server-side, so the page is then
                        // reloaded: that rebuilds every client-side setting (the modal
                        // form, cached settings, theme, prompt-bar mode, model list, API
                        // keys) from the freshly imported state and is guaranteed to
                        // reflect the import.
                        const refreshSettingsFormAfterImport = async () => {
                            let reloadScheduled = false;
                            const scheduleReload = () => {
                                if (reloadScheduled) return;
                                reloadScheduled = true;
                                setTimeout(() => { location.reload(); }, 1100);
                            };
                            try {
                                const res = await apiFetch(CHAT_CONFIG.urls.handleSettingsQuery, {cache: 'no-store'});
                                const data = await res.json().catch(() => null);
                                if (!res.ok || !data) {
                                    scheduleReload();
                                    return;
                                }
                                cacheUserSettings(data);
                                const settingsModalEl = get('settings-modal');
                                if (settingsModalEl && settingsModalEl.classList.contains('modal-open')) {
                                    try { populateSettingsFormFromData(data); } catch (_) {}
                                }
                                // Apply UI-affecting settings immediately (theme / prompt-bar mode).
                                if (data.theme_color) applyThemeColor(data.theme_color, true);
                                if (Object.prototype.hasOwnProperty.call(data, 'minimal_prompt_mode') && data.minimal_prompt_mode) {
                                    setMinimalPromptMode(true);
                                } else if (Object.prototype.hasOwnProperty.call(data, 'compact_prompt_mode')) {
                                    setCompactPromptMode(!!data.compact_prompt_mode);
                                }
                            } catch (_) {}
                            scheduleReload();
                        };
                        const finishImportSuccess = (terminal) => {
                            const message = (terminal && terminal.message) || 'インポートが完了しました';
                            if (resultBox) {
                                resultBox.textContent = `完了: ${message}`;
                                resultBox.classList.remove('hidden', 'text-red-300');
                                resultBox.classList.add('text-emerald-300');
                            }
                            renderAccountTransferProgress({progress: 100, phase: 'completed', message});
                            showToast('選択したアカウントデータをインポートしました', 'success');
                            if (categories.includes('chats')) loadThreads();
                            if (categories.includes('skills')) loadSkills();
                            if (categories.includes('files')) loadStorageUsage();
                            if (categories.includes('settings') || categories.includes('api_credentials')) refreshSettingsFormAfterImport();
                        };
                        const fetchImportStatus = async () => {
                            try {
                                const res = await apiFetch(`/api/account/transfer/${transfer.id}`, manualSpinnerRequestOptions({cache: 'no-store'}));
                                const data = await res.json().catch(() => null);
                                return data && data.state ? data : null;
                            } catch (_) {
                                return null;
                            }
                        };
                        const settleUnreadableImport = async () => {
                            // The import response could not be read. The authoritative
                            // transfer status tells us what actually happened server-side:
                            //   completed -> import really finished (response was lost)
                            //   failed/cancelled/expired -> real server outcome
                            //   needs_selection -> the picker data survived in the status
                            //   running -> server still importing; wait for the terminal state
                            const outcome = await fetchImportStatus();
                            if (!outcome) return {status: 'unknown'};
                            if (outcome.state === 'completed') {
                                finishImportSuccess(outcome);
                                return {status: 'done'};
                            }
                            if (['failed', 'cancelled', 'expired'].includes(outcome.state)) {
                                throw new Error(outcome.message || 'インポートに失敗しました');
                            }
                            if (outcome.state === 'needs_selection' && Array.isArray(outcome.files)) {
                                const chosen = await showImportFileSelection({
                                    files: outcome.files,
                                    available_bytes: outcome.available_bytes,
                                });
                                if (chosen === null) {
                                    renderAccountTransferProgress({progress: 0, phase: 'cancelled', message: 'ファイル選択をキャンセルしました'});
                                    if (transfer.uploadId) {
                                        apiFetch(`/api/account/import/upload/${encodeURIComponent(transfer.uploadId)}`, manualSpinnerRequestOptions({method: 'DELETE'})).catch(() => null);
                                    }
                                    return {status: 'cancelled'};
                                }
                                selectedFiles = chosen;
                                return {status: 'reselect'};
                            }
                            if (outcome.state === 'needs_settings_confirmation' && Array.isArray(outcome.settings_changes)) {
                                const ok = await showSettingsImportConfirmation({ settings_changes: outcome.settings_changes });
                                if (!ok) {
                                    renderAccountTransferProgress({progress: 0, phase: 'cancelled', message: '設定のインポートをキャンセルしました'});
                                    if (transfer.uploadId) {
                                        apiFetch(`/api/account/import/upload/${encodeURIComponent(transfer.uploadId)}`, manualSpinnerRequestOptions({method: 'DELETE'})).catch(() => null);
                                    }
                                    return {status: 'cancelled'};
                                }
                                settingsConfirmed = true;
                                return {status: 'reselect'};
                            }
                            if (outcome.state === 'running') {
                                // The connection dropped but the server is still importing.
                                // The ongoing poller resolves once a terminal state is reached.
                                const terminal = await Promise.race([
                                    pollPromise.catch(() => null),
                                    new Promise(resolve => setTimeout(() => resolve(null), 60000)),
                                ]);
                                if (terminal && terminal.state === 'completed') {
                                    finishImportSuccess(terminal);
                                    return {status: 'done'};
                                }
                                if (terminal && ['failed', 'cancelled', 'expired'].includes(terminal.state)) {
                                    throw new Error(terminal.message || 'インポートに失敗しました');
                                }
                                throw new Error('インポート処理がサーバー側で継続中です。しばらくしてからページを再読み込みして確認してください');
                            }
                            // pending / unknown: the request never reached the server.
                            return {status: 'unknown'};
                        };
                        while (!importDone) {
                            transfer.stopped = true;
                            await pollPromise.catch(() => null);
                            transfer.stopped = false;
                            pollPromise = pollAccountTransfer(transfer);
                            let res;
                            try {
                                res = await apiFetch('/api/account/import', manualSpinnerRequestOptions({
                                    method: 'POST', headers: {'Content-Type': 'application/json'},
                                    body: JSON.stringify({upload_id: transfer.uploadId, categories: categories.join(','), job_id: transfer.id, selected_files: selectedFiles, restore_inplace: restoreInplace, confirm_settings: (settingsConfirmed || settingsBypass)}),
                                    signal: transfer.controller.signal,
                                }));
                            } catch (error) {
                                if (transfer.cancelRequested || (error && error.name === 'AbortError')) throw error;
                                const settled = await settleUnreadableImport();
                                if (settled.status === 'done') { importDone = true; break; }
                                if (settled.status === 'cancelled') return;
                                if (settled.status === 'reselect') continue;
                                if (parseFailures < 2) { parseFailures++; continue; }
                                throw new Error('インポート応答を取得できませんでした。通信環境をご確認のうえ、もう一度お試しください');
                            }
                            let data = null;
                            try {
                                data = await res.json();
                            } catch (_) {
                                data = null;
                            }
                            if (data === null) {
                                const settled = await settleUnreadableImport();
                                if (settled.status === 'done') { importDone = true; break; }
                                if (settled.status === 'cancelled') return;
                                if (settled.status === 'reselect') continue;
                                if (res.ok) throw new Error('インポート結果を確認できませんでした。ページを再読み込みして確認してください');
                                if (parseFailures < 2) { parseFailures++; continue; }
                                throw new Error('インポート応答を取得できませんでした。通信環境をご確認のうえ、もう一度お試しください');
                            }
                            if (!res.ok && data.error === 'storage_limit_files' && data.files) {
                                const chosen = await showImportFileSelection(data);
                                if (chosen === null) {
                                    renderAccountTransferProgress({progress: 0, phase: 'cancelled', message: 'ファイル選択をキャンセルしました'});
                                    if (transfer.uploadId) {
                                        apiFetch(`/api/account/import/upload/${encodeURIComponent(transfer.uploadId)}`, manualSpinnerRequestOptions({method: 'DELETE'})).catch(() => null);
                                    }
                                    return;
                                }
                                selectedFiles = chosen;
                                continue;
                            }
                            if (data && data.status === 'settings_confirmation' && Array.isArray(data.settings_changes)) {
                                const ok = await showSettingsImportConfirmation(data);
                                if (!ok) {
                                    renderAccountTransferProgress({progress: 0, phase: 'cancelled', message: '設定のインポートをキャンセルしました'});
                                    if (transfer.uploadId) {
                                        apiFetch(`/api/account/import/upload/${encodeURIComponent(transfer.uploadId)}`, manualSpinnerRequestOptions({method: 'DELETE'})).catch(() => null);
                                    }
                                    return;
                                }
                                settingsConfirmed = true;
                                continue;
                            }
                            if (!res.ok) throw new Error(data.error || 'インポートに失敗しました');
                            const imported = data.imported || {};
                            const detail = [
                                `設定 ${imported.settings || 0}件`, `API認証 ${imported.api_credentials || 0}件`,
                                `チャット ${imported.chats || 0}件`, `Skill ${imported.skills || 0}件`,
                                `ファイル ${imported.files || 0}件`, `フィードバック ${imported.feedback || 0}件`,
                                `診断データ ${imported.diagnostics || 0}件`,
                            ].join(' / ');
                            const duplicated = data.duplicates || {};
                            const dupLabels = {chats: 'チャット', skills: 'Skill', files: 'ファイル', feedback: 'フィードバック', diagnostics: '診断データ'};
                            const dupParts = [];
                            for (const key of Object.keys(dupLabels)) {
                                const count = Number(duplicated[key]) || 0;
                                if (count > 0) dupParts.push(`${dupLabels[key]} ${count}件`);
                            }
                            const dupNote = dupParts.length ? `（重複をスキップ: ${dupParts.join('、')}）` : '';
                            if (resultBox) {
                                resultBox.textContent = `完了: ${detail}${dupNote}`;
                                resultBox.classList.remove('hidden', 'text-red-300');
                                resultBox.classList.add('text-emerald-300');
                            }
                            renderAccountTransferProgress({progress: 100, phase: 'completed', message: 'インポートが完了しました'});
                            showToast('選択したアカウントデータをインポートしました', 'success');
                            if (categories.includes('chats')) loadThreads();
                            if (categories.includes('skills')) loadSkills();
                            if (categories.includes('files')) loadStorageUsage();
                            if (categories.includes('settings') || categories.includes('api_credentials')) refreshSettingsFormAfterImport();
                            importDone = true;
                        }
                    } catch (error) {
                        if (transfer.uploadId) {
                            apiFetch(`/api/account/import/upload/${encodeURIComponent(transfer.uploadId)}`, manualSpinnerRequestOptions({method: 'DELETE'})).catch(() => null);
                        }
                        if (transfer.cancelRequested || (error && error.name === 'AbortError')) return;
                        const rawMessage = error && error.message ? error.message : '';
                        const friendlyMessage = rawMessage === 'storage_limit_exceeded'
                            ? 'ストレージ上限を超えるためインポートできません'
                            : (rawMessage || 'インポートに失敗しました');
                        renderAccountTransferProgress({progress: 0, phase: 'failed', message: friendlyMessage});
                        if (resultBox) {
                            resultBox.textContent = friendlyMessage;
                            resultBox.classList.remove('hidden', 'text-emerald-300');
                            resultBox.classList.add('text-red-300');
                        }
                        showToast(friendlyMessage, 'error', true);
                    } finally {
                        transfer.stopped = true;
                        await pollPromise.catch(() => null);
                        finishAccountTransfer(transfer);
                    }
                };
            }
            const accountDedupeBtn = get('account-dedupe-btn');
            const accountDedupeResult = get('account-dedupe-result');
            const showDedupeResult = (message, isError = false) => {
                if (!accountDedupeResult) return;
                accountDedupeResult.textContent = message;
                accountDedupeResult.classList.remove('hidden');
                accountDedupeResult.classList.toggle('text-red-300', !!isError);
                accountDedupeResult.classList.toggle('text-emerald-300', !isError);
            };
            if (accountDedupeBtn) {
                accountDedupeBtn.onclick = async () => {
                    const run = async () => {
                        const previewRes = await apiFetch('/api/account/dedupe/preview', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({}),
                        });
                        const preview = await previewRes.json().catch(() => null);
                        if (!previewRes.ok || !preview) throw new Error((preview && preview.error) || '重複データを確認できませんでした');
                        if (!preview.has_duplicates) {
                            showDedupeResult('重複データは見つかりませんでした');
                            return;
                        }
                        const parts = [];
                        const labels = {chats: 'チャット', skills: 'Skill', files: 'ファイル', feedback: 'フィードバック', diagnostics: '診断データ'};
                        for (const key of ['chats', 'skills', 'files', 'feedback', 'diagnostics']) {
                            const count = Number(preview.duplicates && preview.duplicates[key]) || 0;
                            if (count > 0) parts.push(`${labels[key]} ${count}件`);
                        }
                        const keptNote = Number(preview.kept_referenced_files) > 0
                            ? `\n※チャットから参照されているため、ファイル ${preview.kept_referenced_files}件は削除せず残します。`
                            : '';
                        if (!confirm(`重複データが ${preview.total}件 見つかりました。\n\n${parts.join('、')}${keptNote}\n\n同じ内容のデータは最も古い1件を残して削除します。続行しますか？`)) {
                            return;
                        }
                        const execRes = await apiFetch('/api/account/dedupe/execute', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({}),
                        });
                        const executed = await execRes.json().catch(() => null);
                        if (!execRes.ok || !executed) throw new Error((executed && executed.error) || '重複データの修復に失敗しました');
                        const removedParts = [];
                        for (const key of ['chats', 'skills', 'files', 'feedback', 'diagnostics']) {
                            const count = Number(executed.removed && executed.removed[key]) || 0;
                            if (count > 0) removedParts.push(`${labels[key]} ${count}件`);
                        }
                        const keptAfter = Number(executed.kept_referenced_files) > 0
                            ? `（参照のため残したファイル ${executed.kept_referenced_files}件）`
                            : '';
                        showDedupeResult(`重複データを修復しました: ${removedParts.join('、') || '0件'}${keptAfter}`);
                        loadThreads();
                        loadSkills();
                        loadStorageUsage();
                    };
                    if (accountDedupeBtn.disabled) return;
                    accountDedupeBtn.disabled = true;
                    showDedupeResult('重複データを確認しています...');
                    try {
                        await run();
                    } catch (error) {
                        showDedupeResult((error && error.message) || '重複データの修復に失敗しました', true);
                    } finally {
                        accountDedupeBtn.disabled = false;
                    }
                };
            }
            const siteCacheRefreshBtn = get('site-cache-usage-refresh');
            if (siteCacheRefreshBtn) siteCacheRefreshBtn.onclick = () => loadSiteCacheUsage();
            const clearSiteCacheBtn = get('clear-site-cache-btn');
            if (clearSiteCacheBtn) {
                clearSiteCacheBtn.onclick = async () => {
                    if (!confirm('サイトキャッシュを削除しますか？\nCookie は削除されません。')) return;
                    await clearSiteCacheAndReload(clearSiteCacheBtn);
                };
            }
            const encScanResult = get('enc-scan-result');
            const runEncScan = async (threadId = null) => {
                if (encScanResult) encScanResult.textContent = 'スキャン中...';
                let url = '/api/encryption_scan';
                if (threadId) url += `?thread_id=${encodeURIComponent(threadId)}`;
                try {
                    const res = await apiFetch(url, { cache: 'no-store' });
                    const data = await res.json();
                    if (!res.ok) {
                        if (encScanResult) encScanResult.textContent = data.error || '失敗しました';
                        return;
                    }
                    const total = data.total || 0;
                    const enc = data.encrypted || 0;
                    const unenc = data.unencrypted || 0;
                    let html = `Total: ${total} / Encrypted: ${enc} / Plain: ${unenc}`;
                    if (data.samples && data.samples.length) {
                        const items = data.samples.slice(0, 8).map(s => {
                            const t = s.timestamp ? new Date(s.timestamp).toLocaleString() : '';
                            return `#${s.id} (${s.role || ''}) ${t}`;
                        }).join(' / ');
                        html += `<div class="text-[10px] text-gray-400 mt-1">例: ${items}</div>`;
                    }
                    if (encScanResult) encScanResult.innerHTML = html;
                } catch (e) {
                    if (encScanResult) encScanResult.textContent = '失敗しました';
                }
            };
            const encScanAllBtn = get('enc-scan-all');
            if (encScanAllBtn) encScanAllBtn.onclick = () => runEncScan(null);
            const encScanThreadBtn = get('enc-scan-thread');
            if (encScanThreadBtn) encScanThreadBtn.onclick = () => currentThreadId ? runEncScan(currentThreadId) : showToast('スレッドがありません', 'error', true);
            const adminEncList = get('admin-enc-list');
            let currentThreadEncrypted = null;
            let adminThreadEncBusy = false;

            const computeThreadEncryptedFromMessages = (messages) => {
                if (!messages || !messages.length) return null;
                return messages.some(m => !!m.is_encrypted);
            };

            const refreshCurrentThreadEncStateFromMessages = () => {
                currentThreadEncrypted = computeThreadEncryptedFromMessages(allMessages);
            };

            const setAdminThreadEncryption = async (threadId, enable, { confirmPrompt = true, reloadCurrent = true } = {}) => {
                if (!threadId) {
                    showToast('チャットがありません', 'error', true);
                    return false;
                }
                const action = enable ? '再暗号化' : '復号化';
                if (confirmPrompt && !confirm(`このチャットを${action}しますか？`)) return false;
                adminThreadEncBusy = true;
                try {
                    const res = await apiFetch(`/api/admin/threads/${encodeURIComponent(threadId)}/encryption`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ enable })
                    });
                    const data = await res.json().catch(() => ({}));
                    if (!res.ok) {
                        showToast(data.error || `${action}に失敗しました`, 'error', true);
                        return false;
                    }
                    showToast(`${action}しました（${data.changed || 0}件を変換）`, 'success');
                    currentThreadEncrypted = !!enable;
                    if (reloadCurrent && currentThreadId && String(currentThreadId) === String(threadId)) {
                        await loadMessages(currentThreadId, { preserveDraft: true, silent: true, skipHistory: true });
                    }
                    if (adminEncList) await loadAdminEncThreads();
                    return true;
                } catch (err) {
                    showToast(`${action}に失敗しました`, 'error', true);
                    return false;
                } finally {
                    adminThreadEncBusy = false;
                }
            };

            const renderAdminEncThreads = (data) => {
                if (!adminEncList) return;
                const threads = data.threads || [];
                if (!threads.length) {
                    adminEncList.innerHTML = '<div class="text-[11px] text-gray-400">チャットがありません。</div>';
                    return;
                }
                adminEncList.innerHTML = threads.map(t => {
                    const encState = t.encrypted_count > 0 ? 'enc' : 'plain';
                    const btnLabel = encState === 'enc' ? '復号化' : '再暗号化';
                    const btnColor = encState === 'enc' ? 'bg-amber-600 hover:bg-amber-500' : 'bg-cyan-700 hover:bg-cyan-600';
                    const updated = t.updated_at ? new Date(t.updated_at).toLocaleString() : '';
                    const tid = escapeHtml(String(t.thread_id));
                    const isCurrent = currentThreadId && String(currentThreadId) === String(t.thread_id);
                    return `<div class="flex items-center gap-2 bg-gray-800/60 border border-gray-700 rounded p-2">
                        <div class="flex-1 min-w-0">
                            <div class="font-bold text-gray-200 truncate" title="${escapeHtml(t.title || '')}">${escapeHtml(t.title || '(無題)')}${isCurrent ? ' <span class="text-[10px] text-cyan-300 font-normal">（表示中）</span>' : ''}</div>
                            <div class="text-[10px] text-gray-500">${updated} / メッセージ: ${t.message_count} / 暗号化: ${t.encrypted_count}</div>
                        </div>
                        <button type="button" class="admin-enc-open bg-gray-700 hover:bg-gray-600 text-white px-2 py-1 rounded shrink-0" data-id="${tid}" title="このチャットを開く"><i class="fas fa-external-link-alt mr-1"></i>開く</button>
                        <button type="button" class="admin-enc-toggle ${btnColor} text-white px-2 py-1 rounded shrink-0" data-id="${tid}" data-enable="${encState === 'enc' ? '0' : '1'}" data-progress-expected-slow="true">${btnLabel}</button>
                    </div>`;
                }).join('');
            };
            const loadAdminEncThreads = async () => {
                if (!adminEncList) return;
                adminEncList.innerHTML = '<div class="text-[11px] text-gray-400"><i class="fas fa-spinner fa-spin mr-1"></i>読み込み中...</div>';
                try {
                    const res = await apiFetch('/api/admin/threads', { cache: 'no-store' });
                    const data = await res.json().catch(() => ({}));
                    if (!res.ok) {
                        adminEncList.innerHTML = `<div class="text-[11px] text-red-400">${escapeHtml(data.error || '読み込みに失敗しました')}</div>`;
                        return;
                    }
                    renderAdminEncThreads(data);
                    if (currentThreadId && Array.isArray(data.threads)) {
                        const cur = data.threads.find(t => String(t.thread_id) === String(currentThreadId));
                        if (cur) {
                            currentThreadEncrypted = !!cur.encrypted;
                        }
                    }
                } catch (e) {
                    adminEncList.innerHTML = '<div class="text-[11px] text-red-400">読み込みに失敗しました</div>';
                }
            };
            if (get('admin-enc-load')) {
                get('admin-enc-load').onclick = () => loadAdminEncThreads();
            }
            // Expose for loadMessages / openSettingsModal / encryption modal (outside this block).
            window.__loadAdminEncThreads = loadAdminEncThreads;
            window.__refreshAdminThreadEncState = refreshCurrentThreadEncStateFromMessages;
            window.__setAdminThreadEncryption = setAdminThreadEncryption;
            const encModalAdminToggle = get('encryption-status-admin-toggle');
            if (encModalAdminToggle) {
                encModalAdminToggle.addEventListener('click', (e) => {
                    e.preventDefault();
                    if (typeof toggleThreadEncryptionFromModal === 'function') {
                        toggleThreadEncryptionFromModal();
                    }
                });
            }
            if (adminEncList) {
                adminEncList.onclick = async (e) => {
                    const openBtn = e.target.closest('.admin-enc-open');
                    if (openBtn) {
                        e.preventDefault();
                        const threadId = openBtn.getAttribute('data-id');
                        if (!threadId) return;
                        if (typeof closeSettingsModal === 'function') {
                            closeSettingsModal();
                        } else if (typeof hideModal === 'function') {
                            hideModal('settings-modal');
                        }
                        try {
                            await loadMessages(threadId);
                        } catch (err) {
                            showToast('チャットを開けませんでした', 'error', true);
                        }
                        return;
                    }
                    const btn = e.target.closest('.admin-enc-toggle');
                    if (!btn || adminThreadEncBusy) return;
                    const threadId = btn.getAttribute('data-id');
                    const enable = btn.getAttribute('data-enable') === '1';
                    const action = enable ? '再暗号化' : '復号化';
                    if (!confirm(`このチャットを${action}しますか？`)) return;
                    btn.disabled = true;
                    const original = btn.textContent;
                    btn.textContent = '処理中...';
                    try {
                        await setAdminThreadEncryption(threadId, enable, { confirmPrompt: false, reloadCurrent: true });
                    } finally {
                        btn.disabled = false;
                        btn.textContent = original;
                        await loadAdminEncThreads();
                    }
                };
            }
            get('file-input').onchange = (e) => {
                const files = Array.from(e.target.files || []);
                // Clear after copying to avoid losing selections during async uploads.
                e.target.value = '';
                if (files.length) handleFiles(files);
            };
            if (get('photo-input')) {
                get('photo-input').onchange = (e) => {
                    const files = Array.from(e.target.files || []);
                    e.target.value = '';
                    if (files.length) handleFiles(files);
                };
            }
            const renderBanAppeals = (items) => {
                const box = get('ban-appeal-list');
                if (!box) return;
                if (!items || !items.length) {
                    box.innerHTML = '<div class="text-[11px] text-gray-500">現在、申し立てはありません。</div>';
                    return;
                }
                box.innerHTML = items.map(a => {
                    const statusLabel = a.status || 'new';
                    const readTag = a.admin_read_at ? '<span class="text-[10px] text-gray-500 ml-2">既読</span>' : '<span class="text-[10px] text-yellow-300 ml-2">未読</span>';
                    const time = a.created_at ? new Date(a.created_at).toLocaleString() : '';
                    const repliedAt = a.replied_at ? new Date(a.replied_at).toLocaleString() : '';
                    const replyValue = a.admin_reply || '';
                    return `
                        <div class="border border-gray-700/70 rounded p-2 bg-gray-900/60" data-appeal-id="${a.id}">
                            <div class="flex items-center justify-between">
                                <div class="text-xs text-blue-200 font-bold">${escapeHtml(a.username || '')}${readTag}</div>
                                <div class="text-[10px] text-gray-500">${escapeHtml(time)}</div>
                            </div>
                            <div class="text-[11px] text-gray-400 mt-1">Status: ${escapeHtml(statusLabel)}</div>
                            <div class="text-xs text-gray-200 mt-2 whitespace-pre-wrap">${escapeHtml(a.message || '')}</div>
                            <div class="text-[10px] text-gray-500 mt-2">BAN理由: ${escapeHtml(a.ban_reason || 'N/A')}</div>
                            ${a.evidence ? `<details class="mt-2"><summary class="text-[10px] text-cyan-300 cursor-pointer">不審な履歴（記録）を表示</summary><pre class="mt-1 text-[10px] text-gray-300 whitespace-pre-wrap bg-gray-950/70 border border-gray-700 rounded p-2 max-h-60 overflow-auto">${escapeHtml(a.evidence)}</pre></details>` : ''}
                            <div class="mt-3">
                                <label class="text-[10px] text-gray-400">管理者返信</label>
                                <textarea class="ban-appeal-reply w-full mt-1 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-[11px] text-gray-100" rows="3" placeholder="返信内容">${escapeHtml(replyValue)}</textarea>
                                ${replyValue ? `<div class=\"text-[10px] text-gray-500 mt-1\">返信日時: ${escapeHtml(repliedAt)}</div>` : ''}
                            </div>
                            <div class="mt-2 flex flex-wrap gap-2">
                                <button class="ban-appeal-mark text-[10px] px-2 py-1 bg-gray-700 hover:bg-gray-600 rounded" data-id="${a.id}">既読</button>
                                <button class="ban-appeal-status text-[10px] px-2 py-1 bg-blue-700 hover:bg-blue-600 rounded" data-id="${a.id}" data-status="in_review">対応中</button>
                                <button class="ban-appeal-status text-[10px] px-2 py-1 bg-green-700 hover:bg-green-600 rounded" data-id="${a.id}" data-status="resolved">完了</button>
                                <button class="ban-appeal-status text-[10px] px-2 py-1 bg-red-700 hover:bg-red-600 rounded" data-id="${a.id}" data-status="rejected">却下</button>
                                <button class="ban-appeal-reply-send text-[10px] px-2 py-1 bg-sky-700 hover:bg-sky-600 rounded" data-id="${a.id}">返信送信</button>
                                <button class="ban-appeal-block text-[10px] px-2 py-1 bg-rose-700 hover:bg-rose-600 rounded" data-id="${a.id}">申し立てブロック</button>
                            </div>
                        </div>
                    `;
                }).join('');
            };
            const refreshBanAppealSummary = async (notify = false) => {
                if (!isAdminUser) return;
                const countBox = get('ban-appeal-count');
                if (!countBox) return;
                try {
                    const res = await apiFetch('/api/ban/appeals/summary', { cache: 'no-store' });
                    if (!res.ok) return;
                    const data = await res.json();
                    const count = data.unread_count || 0;
                    countBox.textContent = String(count);
                    if (notify && count > 0) {
                        showToast(`BAN異議申し立てが${count}件あります。`, 'success');
                    }
                } catch (e) {}
            };
            const loadBanAppeals = async () => {
                if (!isAdminUser) return;
                const listBox = get('ban-appeal-list');
                if (!listBox) return;
                listBox.innerHTML = '<div class="text-[11px] text-gray-500">読み込み中...</div>';
                try {
                    const res = await apiFetch('/api/ban/appeals?limit=80', { cache: 'no-store' });
                    if (!res.ok) return;
                    const data = await res.json();
                    renderBanAppeals(data.items || []);
                    await refreshBanAppealSummary(false);
                } catch (e) {}
            };
            const markBanAppealsRead = async (ids = null) => {
                if (!isAdminUser) return;
                const payload = ids ? { ids } : { all: true };
                try {
                    const res = await apiFetch('/api/ban/appeals/mark_read', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload) });
                    if (res.ok) {
                        await loadBanAppeals();
                    }
                } catch (e) {}
            };
            const updateBanAppealStatus = async (payload) => {
                if (!isAdminUser) return;
                try {
                    const res = await apiFetch('/api/ban/appeals/update', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload) });
                    if (res.ok) {
                        await loadBanAppeals();
                    }
                } catch (e) {}
            };
            const ensureTemporaryChatSettingsCard = () => {
                const tab = get('tab-general');
                if (!tab || get('temp-chat-settings-card')) return;
                const card = document.createElement('div');
                card.id = 'temp-chat-settings-card';
                card.className = 'settings-card';
                card.innerHTML = `
                    <h3 class="settings-card-title">一時チャット</h3>
                    <div class="space-y-3 text-xs text-gray-300">
                        <label class="text-xs text-gray-500 block">切断タイムアウト（秒）</label>
                        <input id="set-temp-chat-timeout-seconds" type="number" min="${TEMP_CHAT_TIMEOUT_MIN_SECONDS}" max="${TEMP_CHAT_TIMEOUT_MAX_SECONDS}" step="1" class="w-28 bg-gray-800 border border-gray-600 rounded px-2 py-1 text-xs text-white">
                        <div class="text-[10px] text-gray-500">一時チャットでページの表示/接続が途切れた状態がこの秒数を超えると、自動削除されます。</div>
                    </div>
                `;
                tab.appendChild(card);
            };
            const ensureLlmTranscribePromptSettingsUi = () => {
                const sttModelEl = get('set-stt-model');
                if (!sttModelEl || get('set-llm-transcribe-prompt')) return;
                const host = sttModelEl.closest('.space-y-2');
                if (!host) return;
                const wrap = document.createElement('div');
                wrap.className = 'pt-2 border-t border-gray-700/60';
                wrap.innerHTML = `
                    <label class="text-xs text-gray-500 block">LLM文字起こしプロンプト（LLM方式）</label>
                    <textarea id="set-llm-transcribe-prompt" class="w-full h-24 bg-gray-800 border border-gray-600 rounded px-2 py-2 text-xs text-white mt-1" placeholder=""></textarea>
                    <div class="flex items-center gap-2 mt-2">
                        <button type="button" id="reset-llm-transcribe-prompt" class="bg-gray-700 hover:bg-gray-600 text-white px-2 py-1 rounded text-[10px] font-bold btn-hover">既定に戻す</button>
                        <div class="text-[10px] text-gray-500">LLM方式のマイク文字起こし時のみ使用。空欄で保存すると既定文面を使います（無音時の安全ガードは別途自動付与）。</div>
                    </div>
                `;
                host.appendChild(wrap);
                const resetBtn = get('reset-llm-transcribe-prompt');
                if (resetBtn) {
                    resetBtn.onclick = () => {
                        const ta = get('set-llm-transcribe-prompt');
                        if (ta) ta.value = '';
                        showToast('LLM文字起こしプロンプトを既定値に戻しました（保存してください）', 'success');
                    };
                }
            };
            const AUTO_SYS_PROMPT_ITEMS = [
                { key: 'python', label: 'Python 実行案内' },
                { key: 'deepseek_python', label: 'DeepSeek Python（ツール実行）' },
                { key: 'gemini_code_execution', label: 'Gemini コード実行（サンドボックスの実行時間制限）' },
                { key: 'python_file_creation', label: 'Python + ファイル作成（Gemini コード実行時）' },
                { key: 'gemini_local_python', label: 'Gemini 音声/動画/PDF/DOCX + Python（ローカル実行）' },
                { key: 'grok_search', label: 'Search補助（Grok）' },
                { key: 'openai_search', label: 'Search補助（OpenAI/xAI Responses）' },
                { key: 'marker', label: 'Marker編集時' },
                { key: 'attachment_names', label: '添付ファイル名（LLM入力時）', hint: '利用可能変数: {{attachment_names}} / {{attachment_count}}' },
                { key: 'quote_source', label: '引用元（引用して送信時）', hint: '利用可能変数: {{quote_source}}（引用元の発言者と会話内の番号が入ります）' },
                { key: 'mathjax', label: 'MathJax（LaTeX数式）' },
                { key: 'image_analysis', label: '画像解析（Vision Model指示文）' },
                { key: 'mcp', label: 'MCP（外部ツール接続）', hint: '利用可能変数: {{mcp_tools}}（接続中のMCPツール一覧が入ります）', barLocked: true, barLabel: 'MCP' },
                { key: 'skills', label: 'Skills（保存した指示の利用）', hint: '利用可能変数: {{skills}} / {{skill_count}}', barLocked: true, barLabel: 'Skills' }
            ];
            window.buildAutoSystemPromptRows = (prefix, compact = false) => {
                const textClass = compact
                    ? 'w-full h-14 bg-gray-950 border border-gray-700 rounded p-2 text-[11px] text-gray-200'
                    : 'w-full h-20 bg-gray-950 border border-gray-700 rounded p-2 text-xs text-gray-200';
                return AUTO_SYS_PROMPT_ITEMS.map((item) => {
                    const isBarLocked = item.barLocked === true;
                    const barLabel = item.barLabel || 'MCP';
                    const lockNote = isBarLocked
                        ? `<div class="text-[10px] text-cyan-300/70 mt-1">この項目のオン・オフはプロンプトバーの${barLabel}スイッチに連動します（オフ時は案内文の注入とツール付与自体が無効）。文面は編集できます。</div>`
                        : '';
                    const enabledInput = isBarLocked
                        ? `<input type="checkbox" id="${prefix}-auto-sys-${item.key}-enabled" class="accent-yellow-500 w-3 h-3" disabled>`
                        : `<input type="checkbox" id="${prefix}-auto-sys-${item.key}-enabled" class="accent-yellow-500 w-3 h-3">`;
                    return `
                    <div class="rounded border border-gray-700 p-2 bg-gray-950/40">
                        <div class="flex items-center justify-between mb-1">
                            <div class="text-[11px] text-gray-300">${item.label}</div>
                            <label class="flex items-center gap-1 text-[10px] text-gray-500" ${isBarLocked ? `title="プロンプトバーの${barLabel}スイッチに連動します"` : ''}>
                                ${enabledInput}
                                <span>適用</span>
                            </label>
                        </div>
                        <textarea id="${prefix}-auto-sys-${item.key}-text" class="${textClass}" placeholder="自動注入文言"></textarea>
                        ${item.hint ? `<div class="text-[10px] text-gray-500 mt-1">${item.hint}</div>` : ''}
                        ${lockNote}
                    </div>
                `;
                }).join('');
            };
            window.applyAutoSystemPromptConfigToForm = (prefix, cfg = {}) => {
                AUTO_SYS_PROMPT_ITEMS.forEach((item) => {
                    const row = cfg && typeof cfg === 'object' ? (cfg[item.key] || {}) : {};
                    const enabledEl = get(`${prefix}-auto-sys-${item.key}-enabled`);
                    const textEl = get(`${prefix}-auto-sys-${item.key}-text`);
                    if (enabledEl) {
                        // MCP / Skills行は読取専用。オン・オフはプロンプトバーに連動する。
                        if (item.barLocked === true) enabledEl.disabled = true;
                        else enabledEl.checked = row.enabled !== false;
                    }
                    if (textEl) {
                        textEl.value = row.text || '';
                        textEl.placeholder = row.default_text || '自動注入文言';
                    }
                });
                if (typeof syncMcpAutoSysRows === 'function') syncMcpAutoSysRows();
            };
            const resetAutoSystemPromptConfigToCodeDefaults = (prefix, applyToggleId = null) => {
                if (applyToggleId) {
                    const applyToggleEl = get(applyToggleId);
                    if (applyToggleEl) applyToggleEl.checked = true;
                }
                AUTO_SYS_PROMPT_ITEMS.forEach((item) => {
                    const enabledEl = get(`${prefix}-auto-sys-${item.key}-enabled`);
                    const textEl = get(`${prefix}-auto-sys-${item.key}-text`);
                    if (enabledEl) {
                        // MCP / Skills行は既定に戻しても適用トグルは変更しない（プロンプトバー連動）。
                        if (item.barLocked !== true) enabledEl.checked = true;
                        else enabledEl.disabled = true;
                    }
                    if (textEl) {
                        const defaultText = textEl.placeholder || '';
                        textEl.value = defaultText;
                    }
                });
                if (typeof syncMcpAutoSysRows === 'function') syncMcpAutoSysRows();
            };
            const collectAutoSystemPromptConfigFromForm = (prefix) => {
                const cfg = {};
                AUTO_SYS_PROMPT_ITEMS.forEach((item) => {
                    const enabledEl = get(`${prefix}-auto-sys-${item.key}-enabled`);
                    const textEl = get(`${prefix}-auto-sys-${item.key}-text`);
                    // MCP / Skillsは設定側でオフにできない（常に有効）。保存時も true で固定する。
                    cfg[item.key] = {
                        enabled: item.barLocked === true ? true : (enabledEl ? enabledEl.checked : true),
                        text: textEl ? textEl.value : ''
                    };
                });
                return cfg;
            };
            // チャット設定モーダル（part16、DOMContentLoaded の外）からも使うため公開する。
