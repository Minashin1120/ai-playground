/* ================= MCP（外部モデル連携）設定UI ================= */
        // 状態
        let mcpServers = [];
        let mcpLoaded = false;
        let mcpLoadPromise = null;
        let mcpOauthPopups = [];
        const MCP_URLS = {
            servers: () => '/api/mcp/servers',
            server: (id) => `/api/mcp/servers/${encodeURIComponent(id)}`,
            test: (id) => `/api/mcp/servers/${encodeURIComponent(id)}/test`,
            authStart: (id) => `/api/mcp/servers/${encodeURIComponent(id)}/auth/start`,
            authDisconnect: (id) => `/api/mcp/servers/${encodeURIComponent(id)}/auth/disconnect`,
            tools: (id) => `/api/mcp/servers/${encodeURIComponent(id)}/tools`,
            oauthClient: () => '/api/mcp/oauth-client',
            permission: (id, tool) => `/api/mcp/servers/${encodeURIComponent(id)}/tools/${encodeURIComponent(tool)}/permission`
        };
        const mcpGoogleProviderKey = 'google_workspace';

        const mcpEsc = (s) => {
            return String(s == null ? '' : s).replace(/[&<>"']/g, (c) => ({
                '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
            }[c]));
        };
        const mcpStatusMsg = (elId, msg, isError) => {
            const el = get(elId);
            if (!el) return;
            el.textContent = msg || '';
            el.style.color = isError ? '#f87171' : '#9ca3af';
        };

        function mcpAuthStatusLabel(srv) {
            if (srv.auth_type === 'none') return '認証不要';
            if (srv.auth_status === 'connected') return '接続済み';
            if (srv.auth_status === 'expired') return '期限切れ（再認証）';
            if (srv.auth_status === 'needs_auth') return '認証が必要';
            return '未認証';
        }
        function mcpConnectionStateLabel(srv) {
            if (srv.connection_state === 'error') return 'エラー';
            if (srv.connection_state === 'connected') return '接続OK';
            if (srv.connection_state === 'needs_auth') return '認証待ち';
            return '未接続';
        }
        function mcpBadgeClass(kind) {
            if (kind === 'ok' || kind === 'connected') return 'bg-emerald-700/60 text-emerald-100';
            if (kind === 'error' || kind === 'expired') return 'bg-red-700/60 text-red-100';
            if (kind === 'auth') return 'bg-amber-600/50 text-amber-100';
            return 'bg-gray-700 text-gray-300';
        }
        function mcpStateBadge(srv) {
            const label = mcpAuthStatusLabel(srv);
            const cls = (srv.auth_status === 'connected') ? 'ok'
                : (srv.auth_status === 'expired') ? 'expired'
                : (srv.auth_status === 'needs_auth') ? 'auth' : 'neutral';
            return `<span class="text-[9px] font-bold px-2 py-0.5 rounded-full ${mcpBadgeClass(cls)}">${mcpEsc(label)}</span>`;
        }

        function mcpOauthProviderLabel(pk) {
            if (pk === 'google_workspace') return 'Google Workspace';
            return pk || 'OAuth';
        }

        async function loadMcpServers(force) {
            const listEl = get('mcp-server-list');
            if (!listEl) return;
            // Settings can be opened while the initial preload is still running.
            // Share that request so opening the modal never starts a second slow
            // request (and so the prompt-bar state has one source of truth).
            if (mcpLoadPromise) {
                await mcpLoadPromise;
                if (!force) return;
            }
            if (!force && mcpLoaded) { renderMcpServers(); return; }
            mcpStatusMsg('mcp-status-msg', '読み込み中...', false);
            let request;
            request = (async () => {
                try {
                    const res = await apiFetch(MCP_URLS.servers());
                    if (!res.ok) {
                        const d = await res.json().catch(() => ({}));
                        mcpStatusMsg('mcp-status-msg', d.error || 'MCPサーバー一覧の取得に失敗しました', true);
                        return;
                    }
                    const data = await res.json();
                    mcpServers = (data && Array.isArray(data.servers)) ? data.servers : [];
                    mcpLoaded = true;
                    renderMcpServers();
                    applyMcpPromptChipUi();
                } catch (e) {
                    mcpStatusMsg('mcp-status-msg', 'MCPサーバー一覧の取得に失敗しました: ' + (e && e.message ? e.message : e), true);
                } finally {
                    if (mcpLoadPromise === request) mcpLoadPromise = null;
                }
            })();
            mcpLoadPromise = request;
            await request;
        }

        // 有効（enabled）なMCPサーバーが1つ以上あるか（プロンプトバーMCPチップの表示条件）。
        function mcpHasEnabledServer() {
            return (mcpServers || []).some((srv) => !!srv.enabled);
        }

        // プロンプトバーのMCPスイッチが実効ONか（チップ表示中かつチェックON）。
        function isMcpEnabledForSend() {
            const cont = get('mcp-container');
            if (!cont || cont.classList.contains('hidden')) return false;
            const chk = get('enable-mcp');
            return !!chk && chk.checked;
        }

        // このモデルでMCPツールが利用可能か（バックエンドのテキストLLM付与対象と揃える）。
        function mcpModelSupported() {
            try {
                const m = String((get('model-select') && get('model-select').value) || '').toLowerCase();
                if (!m) return false;
                // Claude / Kimi は isLlmModel に含まれないが、バックエンドではMCPを付与する。
                if (m.includes('claude') || m.startsWith('kimi')) return true;
                if (typeof isLlmModel === 'function' && isLlmModel()) return true;
                return false;
            } catch (e) { return false; }
        }

        // プロンプトバーのMCPチップ表示/非表示を、現在のモデルとMCP利用可否で更新する。
        function applyMcpPromptChipUi() {
            const cont = get('mcp-container');
            if (!cont) return;
            const show = mcpModelSupported() && mcpHasEnabledServer();
            cont.classList.toggle('hidden', !show);
            syncMcpAutoSysRows();
            if (typeof refreshMinimalOptionsIfOpen === 'function') {
                try { refreshMinimalOptionsIfOpen(); } catch (e) {}
            }
        }

        // 「自動注入システムプロンプト（ユーザー単位）」のMCP行を読取専用（プロンプトバー連動）に保つ。
        // 設定モーダル(set)とスレッド設定(thread)の両方に反映する。
        function syncMcpAutoSysRows() {
            ['set', 'thread'].forEach((prefix) => {
                const el = get(`${prefix}-auto-sys-mcp-enabled`);
                if (!el) return;
                el.disabled = true;
                el.checked = isMcpEnabledForSend();
            });
        }

        function renderMcpServers() {
            const listEl = get('mcp-server-list');
            const countEl = get('mcp-server-count');
            if (!listEl) return;
            const count = mcpServers.length;
            if (countEl) countEl.textContent = `${count}件`;
            if (!count) {
                listEl.innerHTML = '<div class="text-[11px] text-gray-600 py-2">まだサーバーがありません。上のカスタム追加フォームから登録するか、Google Workspace の認証をしてください。</div>';
                mcpStatusMsg('mcp-status-msg', '');
                return;
            }
            const html = mcpServers.map((srv, idx) => mcpServerCard(srv, idx)).join('');
            listEl.innerHTML = html;
            mcpStatusMsg('mcp-status-msg', '');
        }

        function mcpServerCard(srv, idx) {
            const isPreset = !!srv.is_preset;
            const isOauth = srv.auth_type === 'oauth';
            const isBearer = srv.auth_type === 'bearer';
            const showAuthAction = isOauth || isBearer;
            const needOauthClient = isOauth && !srv.oauth_client_registered;
            const toolCount = Number(srv.tool_count || 0);
            const toolHint = toolCount > 0 ? `${toolCount}ツール` : 'ツール未取得';
            const badge = mcpStateBadge(srv);
            const tag = isPreset
                ? `<span class="text-[9px] font-bold px-1.5 py-0.5 rounded bg-blue-700/50 text-blue-100">プリセット</span>`
                : `<span class="text-[9px] font-bold px-1.5 py-0.5 rounded bg-purple-700/50 text-purple-100">カスタム</span>`;
            const authBlock = mcpAuthBlock(srv);
            const oauthClientBlock = isOauth ? mcpOauthClientBlock(srv) : '';
            return `
<div class="rounded border border-gray-700 bg-gray-950/50 p-3" data-mcp-server="${mcpEsc(srv.slug)}">
    <div class="flex items-center justify-between gap-2 flex-wrap">
        <div class="flex items-center gap-2 min-w-0">
            <i class="fas fa-plug ${srv.enabled ? 'text-cyan-300' : 'text-gray-600'}"></i>
            <div class="min-w-0">
                <span class="text-xs font-bold text-white">${mcpEsc(srv.name)}</span>
                ${tag} ${badge}
            </div>
        </div>
        <div class="flex items-center gap-1 shrink-0">
            ${showAuthAction ? mcpAuthActionButton(srv) : ''}
            ${isBearer && srv.auth_status !== 'connected' ? '' : ''}
            ${!isPreset ? `<button type="button" data-progress-no-spinner="true" class="mcp-mini-btn mcp-danger-btn" data-act="delete" data-id="${srv.id}">削除</button>` : ''}
            <label class="relative inline-flex items-center cursor-pointer ml-1" title="${srv.enabled ? '無効化' : '有効化'}">
                <input type="checkbox" class="sr-only peer mcp-enable-toggle" data-id="${srv.id}" ${srv.enabled ? 'checked' : ''}>
                <div class="w-9 h-5 bg-gray-700 peer-focus:outline-none rounded-full peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-[var(--theme-600)]"></div>
            </label>
        </div>
    </div>
    <div class="text-[10px] text-gray-500 mt-1 break-all">${mcpEsc(srv.url)}</div>
    ${srv.description ? `<div class="text-[10px] text-gray-500 mt-0.5">${mcpEsc(srv.description)}</div>` : ''}
    ${srv.last_error ? `<div class="text-[10px] text-red-400 mt-1">${mcpEsc(srv.last_error)}</div>` : ''}
    <div class="flex items-center justify-between gap-2 mt-2 flex-wrap">
        <div class="text-[10px] text-gray-500 flex items-center gap-2">
            <span class="${toolCount > 0 ? 'text-emerald-300' : 'text-gray-500'}">${toolHint}</span>
            <button type="button" data-progress-no-spinner="true" class="mcp-mini-btn" data-act="tools" data-id="${srv.id}">ツール一覧</button>
        </div>
        <div class="flex items-center gap-1 flex-wrap">
            <button type="button" data-progress-no-spinner="true" class="mcp-mini-btn" data-act="test" data-id="${srv.id}"><i class="fas fa-plug"></i> 接続テスト</button>
            <span class="text-[9px] text-gray-600">${mcpEsc(mcpConnectionStateLabel(srv))}</span>
        </div>
    </div>
    ${toolCount > 0 ? `<div class="hidden mt-2" data-mcp-toolbox="${srv.id}"></div>` : `<div class="hidden mt-2" data-mcp-toolbox="${srv.id}"><div class="text-[10px] text-gray-600">接続テスト後にツール一覧が表示されます。</div></div>`}
    ${oauthClientBlock}
    ${authBlock}
</div>`;
        }

        function mcpAuthActionButton(srv) {
            if (srv.auth_type === 'bearer') {
                return '';
            }
            if (srv.auth_status === 'connected' || srv.auth_status === 'expired') {
                return `<button type="button" data-progress-no-spinner="true" class="mcp-mini-btn mcp-auth-btn" data-act="reconnect" data-id="${srv.id}"><i class="fas fa-sync"></i> 再認証</button>
                        <button type="button" data-progress-no-spinner="true" class="mcp-mini-btn mcp-danger-btn" data-act="disconnect" data-id="${srv.id}"><i class="fas fa-unlink"></i> 接続解除</button>`;
            }
            return `<button type="button" data-progress-no-spinner="true" class="mcp-mini-btn mcp-auth-btn" data-act="auth" data-id="${srv.id}"><i class="fas fa-key"></i> 認証する</button>`;
        }

        // OAuthクライアント情報の編集ブロック（サーバーごと）
        function mcpOauthClientBlock(srv) {
            const pk = srv.oauth_provider_key || srv.slug || '';
            const providerLabel = mcpOauthProviderLabel(pk);
            if (srv.oauth_client_registered) {
                return `
<div class="mt-2 rounded border border-gray-800 bg-black/20 p-2">
    <div class="text-[10px] text-gray-400 flex items-center justify-between">
        <span>OAuthクライアント（${mcpEsc(providerLabel)}）: ${mcpEsc(srv.oauth_client_id_masked || '登録済み')}</span>
        <button type="button" data-progress-no-spinner="true" class="mcp-mini-btn" data-act="edit-oauth" data-id="${srv.id}">変更</button>
    </div>
</div>`;
            }
            return `
<div class="mt-2 rounded border border-amber-700/50 bg-amber-950/20 p-2">
    <div class="text-[10px] text-amber-300 mb-1">${mcpEsc(providerLabel)} の OAuth クライアント情報（Client ID / Secret）が必要です。</div>
    <div class="grid grid-cols-1 md:grid-cols-2 gap-1">
        <input type="text" data-oauth-pk="${mcpEsc(pk)}" data-oauth-role="cid" placeholder="Client ID" autocomplete="off" data-1p-ignore="true" class="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs text-white">
        <input type="password" data-oauth-pk="${mcpEsc(pk)}" data-oauth-role="secret" placeholder="Client Secret" autocomplete="off" data-1p-ignore="true" class="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs text-white">
    </div>
    <div class="flex justify-end mt-1">
        <button type="button" data-progress-no-spinner="true" class="mcp-mini-btn" data-act="save-oauth" data-id="${srv.id}" data-pk="${mcpEsc(pk)}">保存</button>
    </div>
</div>`;
        }

        // 認証ブロック（bearer入力など）
        function mcpAuthBlock(srv) {
            if (srv.auth_type === 'bearer') {
                const hasToken = !!srv.auth_has_token;
                return `
<div class="mt-2 rounded border border-gray-800 bg-black/20 p-2">
    <div class="text-[10px] text-gray-400 mb-1">Bearer トークン ${hasToken ? '<span class="text-emerald-300">（保存済み・********）</span>' : '<span class="text-amber-300">（未設定）</span>'}</div>
    <div class="flex gap-1">
        <input type="password" data-bearer-id="${srv.id}" placeholder="Bearer トークン" autocomplete="off" data-1p-ignore="true" class="flex-1 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs text-white">
        <button type="button" data-progress-no-spinner="true" class="mcp-mini-btn mcp-auth-btn" data-act="save-bearer" data-id="${srv.id}">保存</button>
    </div>
</div>`;
            }
            if (srv.auth_type === 'oauth') {
                const pk = srv.oauth_provider_key || srv.slug || '';
                const needClient = !srv.oauth_client_registered;
                return `
<div class="text-[10px] text-gray-600 mt-1">${needClient ? 'OAuthクライアント情報を保存すると「認証する」が使えます。' : ''}</div>`;
            }
            return '';
        }

        async function mcpToggleEnabled(serverId, enabled) {
            mcpStatusMsg('mcp-status-msg', enabled ? '有効化しています...' : '無効化しています...', false);
            try {
                const res = await apiFetch(MCP_URLS.server(serverId), {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ enabled: enabled })
                });
                if (!res.ok) {
                    const d = await res.json().catch(() => ({}));
                    mcpStatusMsg('mcp-status-msg', d.error || '更新に失敗しました', true);
                    return;
                }
                const data = await res.json();
                mcpStatusMsg('mcp-status-msg', enabled ? '有効にしました。チャットのモデルへツールが公開されます。' : '無効にしました。', false);
                loadMcpServers(true);
            } catch (e) {
                mcpStatusMsg('mcp-status-msg', '更新に失敗しました: ' + (e && e.message ? e.message : e), true);
            }
        }

        async function mcpOpenAuth(serverId) {
            mcpStatusMsg('mcp-status-msg', '認可URLを準備しています...', false);
            try {
                const res = await apiFetch(MCP_URLS.authStart(serverId), {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({})
                });
                if (!res.ok) {
                    const d = await res.json().catch(() => ({}));
                    if (d.requires_oauth_client) {
                        mcpStatusMsg('mcp-status-msg', d.error || 'OAuthクライアント情報を先に登録してください。', true);
                    } else {
                        mcpStatusMsg('mcp-status-msg', d.error || '認可URLの取得に失敗しました', true);
                    }
                    return;
                }
                const data = await res.json();
                if (!data.url) {
                    mcpStatusMsg('mcp-status-msg', '認可URLが返りませんでした', true);
                    return;
                }
                const pop = window.open(data.url, '_blank', 'width=520,height=680');
                if (pop) {
                    mcpOauthPopups.push(pop);
                    mcpStatusMsg('mcp-status-msg', 'Googleの画面で許可してください。完了後このタブに反映されます。', false);
                    const poll = window.setInterval(() => {
                        if (!pop || pop.closed) {
                            window.clearInterval(poll);
                            loadMcpServers(true);
                        }
                    }, 1200);
                } else {
                    mcpStatusMsg('mcp-status-msg', 'ポップアップがブロックされました。', true);
                }
            } catch (e) {
                mcpStatusMsg('mcp-status-msg', '認可URLの取得に失敗しました: ' + (e && e.message ? e.message : e), true);
            }
        }

        async function mcpDisconnect(serverId) {
            if (!window.confirm('このサーバーの認証情報（トークン）を削除して接続を解除しますか？')) return;
            try {
                const res = await apiFetch(MCP_URLS.authDisconnect(serverId), {
                    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}'
                });
                if (!res.ok) {
                    const d = await res.json().catch(() => ({}));
                    mcpStatusMsg('mcp-status-msg', d.error || '接続解除に失敗しました', true);
                    return;
                }
                mcpStatusMsg('mcp-status-msg', '接続を解除しました。', false);
                loadMcpServers(true);
            } catch (e) {
                mcpStatusMsg('mcp-status-msg', '接続解除に失敗しました: ' + (e && e.message ? e.message : e), true);
            }
        }

        async function mcpDeleteServer(serverId) {
            if (!window.confirm('このカスタムMCPサーバーを削除しますか？')) return;
            try {
                const res = await apiFetch(MCP_URLS.server(serverId), { method: 'DELETE' });
                if (!res.ok) {
                    const d = await res.json().catch(() => ({}));
                    mcpStatusMsg('mcp-status-msg', d.error || '削除に失敗しました', true);
                    return;
                }
                mcpStatusMsg('mcp-status-msg', '削除しました。', false);
                loadMcpServers(true);
            } catch (e) {
                mcpStatusMsg('mcp-status-msg', '削除に失敗しました: ' + (e && e.message ? e.message : e), true);
            }
        }

        async function mcpTestServer(serverId, showTools) {
            mcpStatusMsg('mcp-status-msg', '接続テスト中...', false);
            try {
                const res = await apiFetch(MCP_URLS.test(serverId), {
                    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}'
                });
                const d = await res.json().catch(() => ({}));
                if (!res.ok) {
                    mcpStatusMsg('mcp-status-msg', d.error || '接続テストに失敗しました', true);
                    return;
                }
                if (d.probe && d.probe.message) mcpStatusMsg('mcp-status-msg', d.probe.message, !d.probe.ok);
                loadMcpServers(true);
            } catch (e) {
                mcpStatusMsg('mcp-status-msg', '接続テストに失敗しました: ' + (e && e.message ? e.message : e), true);
            }
        }

        async function mcpLoadTools(serverId) {
            const box = document.querySelector(`[data-mcp-toolbox="${serverId}"]`);
            if (!box) return;
            box.classList.remove('hidden');
            box.innerHTML = '<div class="text-[10px] text-gray-500">読み込み中...</div>';
            try {
                const res = await apiFetch(MCP_URLS.tools(serverId));
                const d = await res.json().catch(() => ({}));
                if (!res.ok) {
                    box.innerHTML = `<div class="text-[10px] text-red-400">${mcpEsc(d.error || '取得に失敗しました')}</div>`;
                    return;
                }
                const tools = (d && Array.isArray(d.tools)) ? d.tools : [];
                if (!tools.length) {
                    box.innerHTML = '<div class="text-[10px] text-gray-600">ツール一覧がありません。「接続テスト」で取得してください。</div>';
                    return;
                }
                const rows = tools.map((t, i) => `
<div class="flex items-start justify-between gap-2 py-1 border-b border-gray-800 last:border-0">
    <div class="min-w-0">
        <div class="text-[11px] text-cyan-200 font-mono">${mcpEsc(t.name)}</div>
        <div class="text-[10px] text-gray-500 line-clamp-2">${mcpEsc(t.description || '')}</div>
    </div>
    <span class="text-[9px] shrink-0 px-1.5 py-0.5 rounded ${t.read_only ? 'bg-emerald-800/40 text-emerald-200' : 'bg-amber-800/40 text-amber-200'}">${t.read_only ? '読み取り' : '変更'}</span>
</div>`).join('');
                box.innerHTML = `<div class="rounded border border-gray-800 bg-black/20 p-2">${rows}</div>`;
            } catch (e) {
                box.innerHTML = '<div class="text-[10px] text-red-400">取得に失敗しました</div>';
            }
        }

        async function mcpSaveOauthClient(providerKey, clientId, clientSecret, fromCardId) {
            mcpStatusMsg('mcp-status-msg', '保存しています...', false);
            const body = { provider_key: providerKey, client_id: clientId, client_secret: clientSecret };
            try {
                const res = await apiFetch(MCP_URLS.oauthClient(), {
                    method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body)
                });
                const d = await res.json().catch(() => ({}));
                if (!res.ok) {
                    mcpStatusMsg('mcp-status-msg', d.error || '保存に失敗しました', true);
                    return;
                }
                mcpStatusMsg('mcp-status-msg', 'OAuthクライアント情報を保存しました。', false);
                loadMcpServers(true);
            } catch (e) {
                mcpStatusMsg('mcp-status-msg', '保存に失敗しました: ' + (e && e.message ? e.message : e), true);
            }
        }

        async function mcpAddCustomServer() {
            const nameEl = get('mcp-custom-name');
            const urlEl = get('mcp-custom-url');
            const authEl = get('mcp-custom-auth');
            const descEl = get('mcp-custom-desc');
            const bearerEl = get('mcp-custom-bearer');
            const statusEl = get('mcp-custom-status');
            const btnEl = get('mcp-add-server-btn');
            if (!nameEl || !urlEl || !authEl) return;
            const name = (nameEl.value || '').trim();
            const url = (urlEl.value || '').trim();
            const auth = authEl.value || 'none';
            const desc = descEl ? (descEl.value || '').trim() : '';
            const bearer = (bearerEl && bearerEl.value || '').trim();
            if (!name || !url) {
                if (statusEl) { statusEl.textContent = '表示名とURLは必須です'; statusEl.style.color = '#f87171'; }
                return;
            }
            if (btnEl) btnEl.disabled = true;
            if (statusEl) { statusEl.textContent = '接続テスト中...'; statusEl.style.color = '#9ca3af'; }
            const body = { name, url, auth_type: auth, description: desc };
            if (auth === 'bearer' && bearer) body.bearer_token = bearer;
            try {
                const res = await apiFetch(MCP_URLS.servers(), {
                    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body)
                });
                const d = await res.json().catch(() => ({}));
                if (!res.ok) {
                    if (statusEl) { statusEl.textContent = d.error || '追加に失敗しました'; statusEl.style.color = '#f87171'; }
                    return;
                }
                if (statusEl) { statusEl.textContent = (d.probe && d.probe.message) || '追加しました'; statusEl.style.color = (d.probe && d.probe.ok) ? '#34d399' : '#fbbf24'; }
                nameEl.value = '';
                urlEl.value = '';
                if (descEl) descEl.value = '';
                if (bearerEl) bearerEl.value = '';
                mcpLoaded = false;
                loadMcpServers(true);
            } catch (e) {
                if (statusEl) { statusEl.textContent = '追加に失敗しました: ' + (e && e.message ? e.message : e); statusEl.style.color = '#f87171'; }
            } finally {
                if (btnEl) btnEl.disabled = false;
            }
        }

        function bindMcpSettingsUi() {
            const listEl = get('mcp-server-list');
            if (!listEl) return;
            // 追加ボタン
            const addBtn = get('mcp-add-server-btn');
            if (addBtn) addBtn.addEventListener('click', mcpAddCustomServer);
            const authSel = get('mcp-custom-auth');
            const bearerWrap = get('mcp-custom-bearer-wrap');
            if (authSel && bearerWrap) {
                const syncBearer = () => {
                    bearerWrap.classList.toggle('hidden', authSel.value !== 'bearer');
                };
                authSel.addEventListener('change', syncBearer);
                syncBearer();
            }
            // Google連携クライアント情報（上部カード）
            const saveGoogleBtn = get('mcp-save-google-client-btn');
            if (saveGoogleBtn) {
                saveGoogleBtn.addEventListener('click', async () => {
                    const cidEl = get('mcp-google-client-id');
                    const secEl = get('mcp-google-client-secret');
                    const stateEl = get('mcp-google-client-state');
                    const cid = cidEl ? cidEl.value : '';
                    const sec = secEl ? secEl.value : '';
                    if (!cid && !sec) {
                        if (stateEl) { stateEl.textContent = 'Client ID を入力してください'; stateEl.style.color = '#f87171'; }
                        return;
                    }
                    await mcpSaveOauthClient(mcpGoogleProviderKey, cid || '********', sec || '********', null);
                });
            }
            // 一覧内の操作（イベント委譲）
            listEl.addEventListener('click', async (ev) => {
                const btn = ev.target.closest('[data-act]');
                if (!btn) return;
                const act = btn.getAttribute('data-act');
                const id = btn.getAttribute('data-id');
                if (act === 'test') { ev.preventDefault(); mcpTestServer(id); return; }
                if (act === 'tools') { ev.preventDefault(); mcpLoadTools(id); return; }
                if (act === 'auth' || act === 'reconnect') { ev.preventDefault(); mcpOpenAuth(id); return; }
                if (act === 'disconnect') { ev.preventDefault(); mcpDisconnect(id); return; }
                if (act === 'delete') { ev.preventDefault(); mcpDeleteServer(id); return; }
                if (act === 'edit-oauth') {
                    ev.preventDefault();
                    const card = btn.closest('[data-mcp-server]');
                    if (card) {
                        const pk = btn.getAttribute('data-oauth-pk') || '';
                        const srv = mcpServers.find(s => String(s.id) === String(id));
                        const wrap = document.createElement('div');
                        wrap.className = 'mt-2 rounded border border-amber-700/50 bg-amber-950/20 p-2';
                        wrap.innerHTML = `
    <div class="grid grid-cols-1 md:grid-cols-2 gap-1">
        <input type="text" placeholder="Client ID" autocomplete="off" data-1p-ignore="true" class="mcp-oauth-edit-cid w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs text-white" value="">
        <input type="password" placeholder="Client Secret" autocomplete="off" data-1p-ignore="true" class="mcp-oauth-edit-sec w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs text-white">
    </div>
    <div class="flex justify-end mt-1 gap-1">
        <button type="button" data-progress-no-spinner="true" class="mcp-mini-btn" data-act="save-oauth" data-id="${id}" data-pk="${mcpEsc((srv && (srv.oauth_provider_key || srv.slug)) || '')}">保存</button>
    </div>`;
                        const prev = btn.closest('div');
                        prev.parentNode.insertBefore(wrap, prev.nextSibling);
                        btn.remove();
                    }
                    return;
                }
                if (act === 'save-oauth') {
                    ev.preventDefault();
                    const pk = btn.getAttribute('data-pk') || '';
                    // 上部カード内の入力 or カード内入力 or インライン編集
                    const scope = btn.closest('[data-mcp-server]') || document;
                    const cidInputs = scope.querySelectorAll('[data-oauth-role="cid"], .mcp-oauth-edit-cid');
                    const secInputs = scope.querySelectorAll('[data-oauth-role="secret"], .mcp-oauth-edit-sec');
                    const cid = cidInputs.length ? cidInputs[cidInputs.length - 1].value : '';
                    const sec = secInputs.length ? secInputs[secInputs.length - 1].value : '';
                    if (!cid && !sec) { mcpStatusMsg('mcp-status-msg', 'Client ID を入力してください', true); return; }
                    mcpSaveOauthClient(pk, cid || '********', sec || '********', id);
                    return;
                }
                if (act === 'save-bearer') {
                    ev.preventDefault();
                    const input = document.querySelector(`[data-bearer-id="${id}"]`);
                    const token = input ? input.value : '';
                    if (!token || token.trim() === '') { mcpStatusMsg('mcp-status-msg', 'Bearerトークンを入力してください', true); return; }
                    mcpStatusMsg('mcp-status-msg', '保存しています...', false);
                    try {
                        const res = await apiFetch(MCP_URLS.server(id), {
                            method: 'PUT', headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ bearer_token: token })
                        });
                        const d = await res.json().catch(() => ({}));
                        if (!res.ok) { mcpStatusMsg('mcp-status-msg', d.error || '保存に失敗しました', true); return; }
                        mcpStatusMsg('mcp-status-msg', 'Bearerトークンを保存しました。', false);
                        loadMcpServers(true);
                    } catch (e) {
                        mcpStatusMsg('mcp-status-msg', '保存に失敗しました', true);
                    }
                    return;
                }
            });
            // 有効トグル
            listEl.addEventListener('change', (ev) => {
                const t = ev.target.closest('.mcp-enable-toggle');
                if (!t) return;
                mcpToggleEnabled(t.getAttribute('data-id'), t.checked);
            });
        }

        const initMcpUi = () => {
            try { bindMcpSettingsUi(); } catch (e) {}
            // Load MCP metadata as part of chat initialization.  The settings
            // modal is hidden by default, so waiting for it to open made the
            // prompt-bar MCP switch appear late or not at all.
            try { loadMcpServers(); } catch (e) {}
        };
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', initMcpUi, { once: true });
        } else {
            initMcpUi();
        }

        function bindMcpPromptToggle() {
            const chk = get('enable-mcp');
            if (!chk) return;
            chk.addEventListener('change', () => {
                syncMcpAutoSysRows();
                if (typeof refreshMinimalOptionsIfOpen === 'function') {
                    try { refreshMinimalOptionsIfOpen(); } catch (e) {}
                }
            });
        }
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => { try { bindMcpPromptToggle(); } catch (e) {} });
        } else {
            try { bindMcpPromptToggle(); } catch (e) {}
        }

        function getModelTags(m, group) {
            const tags = [];
            const id = (m.id || '').toLowerCase();
            const name = (m.name || '').toLowerCase();
            const desc = (m.desc || '').toLowerCase();
            const cat = (group.category || '').toLowerCase();
            if (
                cat.includes('gemini') ||
                id.includes('gemini') ||
                name.includes('gemini') ||
                desc.includes('gemini') ||
                cat.includes('banana') ||
                name.includes('banana')
            ) tags.push('gemini');
            if (
                cat.includes('deepseek') ||
                id.includes('deepseek') ||
                name.includes('deepseek') ||
                desc.includes('deepseek')
            ) tags.push('deepseek');
            if (
                cat.includes('mistral') ||
                id.includes('mistral') ||
                name.includes('mistral') ||
                desc.includes('mistral') ||
                id.includes('ocr') ||
                cat.includes('ocr')
            ) tags.push('mistral');
            if (
                cat.includes('gpt') ||
                cat.includes('openai') ||
                id.includes('gpt') ||
                name.includes('gpt') ||
                desc.includes('openai')
            ) tags.push('openai');
            if (
                cat.includes('xai') ||
                cat.includes('grok') ||
                id.includes('grok') ||
                name.includes('grok') ||
                desc.includes('xai')
            ) tags.push('xai');
            if (cat.includes('image') || id.includes('image') || name.includes('image') || desc.includes('image')) tags.push('image');
            if (
                cat.includes('audio') ||
                cat.includes('speech') ||
                id.includes('tts') ||
                name.includes('tts') ||
                desc.includes('tts') ||
                id.includes('realtime') ||
                id.includes('live') ||
                id.includes('voice-agent') ||
                id.includes('native-audio') ||
                name.includes('audio') ||
                desc.includes('audio')
            ) tags.push('audio');
            if (id.includes('reasoning') || name.includes('reasoning') || desc.includes('reasoning')) tags.push('reasoning');
            if ((cat.includes('deepseek') || id.includes('deepseek') || name.includes('deepseek')) && !tags.includes('reasoning')) tags.push('reasoning');
            if (id.includes('fast') || name.includes('fast') || desc.includes('fast') || cat.includes('fast')) tags.push('fast');
            if ((id.includes('deepseek-v4-flash') || (cat.includes('deepseek') && name.includes('flash'))) && !tags.includes('fast')) tags.push('fast');
            if (m.agenticView) tags.push('agentic view');
            return tags;
        }

        function updateModelTagUi() {
            const bar = get('model-tag-bar');
            if (!bar) return;
            const btns = bar.querySelectorAll('.model-tag-btn');
            btns.forEach(b => {
                const t = b.innerText.trim().toLowerCase();
                const active = (t === 'all' ? 'all' : t) === activeModelTag;
                b.className = `model-tag-btn px-2 py-1 text-[10px] rounded border transition ${active ? 'bg-blue-600/20 border-blue-500 text-blue-300' : 'bg-gray-800 border-gray-700 text-gray-300 hover:border-gray-500'}`;
            });
        }

        const modelListGroups = [];
        let modelListBanner = null;
        let modelListEmpty = null;
        let modelListBuilt = false;
        let modelListAnimated = false;
        let modelListRenderFrame = 0;
        function buildModelList() {
            const container = get('model-list-container');
            if (!container || modelListBuilt) return;
            container.innerHTML = '';
            modelListBanner = document.createElement('div');
            modelListBanner.className = 'hidden mb-4 px-3 py-2 rounded-lg border border-teal-500/40 bg-teal-900/20 text-[11px] text-teal-200';
            container.appendChild(modelListBanner);

            MODELS.forEach(group => {
                const availableItems = group.items.filter(m => !m.deprecated);
                if (!availableItems.length) return;
                const groupEl = document.createElement('section');
                groupEl.className = 'model-list-group';
                groupEl.innerHTML = `
                    <div class="flex items-center gap-2 mb-3 px-2">
                        <i class="${group.icon}"></i>
                        <div>
                            <h3 class="font-bold text-gray-200 text-sm">${group.category}</h3>
                            <p class="text-[10px] text-gray-500">${group.description}</p>
                        </div>
                    </div>
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-2 mb-6"></div>
                `;
                const grid = groupEl.querySelector('.grid');
                const entries = availableItems.map(m => {
                    const item = document.createElement('button');
                    const apiModelName = String(m.apiId || m.id || '').trim();
                    const agenticViewHtml = m.agenticView ? `<span class="inline-flex items-center gap-1 rounded-full border border-teal-500/40 bg-teal-900/20 px-2 py-0.5 text-[9px] font-semibold text-teal-200 whitespace-nowrap" title="Agentic View対応：画像をクロップして再観察しながら推論を継続できます"><i class="fas fa-eye" aria-hidden="true"></i>Agentic View</span>` : '';
                    const apiModelHtml = apiModelName ? `<div class="text-[10px] text-cyan-300/90 mt-1.5 font-mono break-all"><span class="font-sans text-gray-500 mr-1">API model:</span>${escapeHtml(apiModelName)}</div>` : '';
                    const priceHtml = m.price ? `<div class="text-[10px] text-amber-400/90 mt-1.5 font-mono flex items-start gap-1"><i class="fas fa-tag text-[9px] mt-0.5 opacity-70 shrink-0"></i><span>${m.price}</span></div>` : '';
                    item.type = 'button';
                    item.className = 'flex flex-col text-left p-3 rounded-lg border transition bg-gray-800 border-gray-700 hover:border-gray-500 hover:bg-gray-750';
                    item.dataset.selected = '0';
                    item.onclick = () => selectModel(m.id, m.name);
                    item.innerHTML = `
                        <div class="flex justify-between items-start gap-2 w-full mb-1">
                            <div class="flex flex-wrap items-center gap-2 min-w-0">
                                <span class="font-bold text-sm text-gray-200">${m.name}</span>
                                ${agenticViewHtml}
                            </div>
                            <i class="model-selected-icon fas fa-check-circle text-blue-400 hidden shrink-0 mt-0.5"></i>
                        </div>
                        <span class="text-[10px] text-gray-400">${m.desc}</span>
                        ${apiModelHtml}
                        ${priceHtml}
                    `;
                    grid.appendChild(item);
                    return {
                        model: m,
                        button: item,
                        searchText: `${m.name} ${m.id} ${apiModelName} ${m.agenticView ? 'agentic view' : ''}`.toLowerCase(),
                        provider: getModelApiProvider(m.id),
                        tags: new Set(getModelTags(m, group)),
                    };
                });
                modelListGroups.push({ element: groupEl, entries });
                container.appendChild(groupEl);
            });

            modelListEmpty = document.createElement('div');
            modelListEmpty.className = 'hidden text-center text-gray-500 py-8';
            container.appendChild(modelListEmpty);
            modelListBuilt = true;
        }

        function updateModelButtonSelection(entry, selectedModel) {
            const isSelected = selectedModel === entry.model.id;
            if (entry.button.dataset.selected === (isSelected ? '1' : '0')) return;
            entry.button.dataset.selected = isSelected ? '1' : '0';
            entry.button.classList.toggle('bg-blue-600/20', isSelected);
            entry.button.classList.toggle('border-blue-500', isSelected);
            entry.button.classList.toggle('ring-1', isSelected);
            entry.button.classList.toggle('ring-blue-500', isSelected);
            entry.button.classList.toggle('bg-gray-800', !isSelected);
            entry.button.classList.toggle('border-gray-700', !isSelected);
            entry.button.classList.toggle('hover:border-gray-500', !isSelected);
            entry.button.classList.toggle('hover:bg-gray-750', !isSelected);
            const icon = entry.button.querySelector('.model-selected-icon');
            if (icon) icon.classList.toggle('hidden', !isSelected);
        }

        function renderModelList(filter = "", options = {}) {
            const container = get('model-list-container');
            if (!container) return;
            buildModelList();
            const f = filter.toLowerCase();
            const lockedProvider = window._visionPickerActive ? null : getPromptCacheLockedProvider();
            const lockedLabel = lockedProvider ? (PROVIDER_LABELS[lockedProvider] || lockedProvider) : '';
            const selectedModel = get('model-select') ? get('model-select').value : '';
            let visibleCount = 0;

            modelListBanner.classList.toggle('hidden', !lockedProvider);
            if (lockedProvider) {
                modelListBanner.innerHTML = `<i class="fas fa-database mr-1.5"></i>PromptCache 有効中: <strong>${lockedLabel}</strong> のモデルのみ選択できます（他APIへの切替は不可）`;
            }

            modelListGroups.forEach(group => {
                let groupVisibleCount = 0;
                group.entries.forEach(entry => {
                    const visible = entry.searchText.includes(f)
                        && (!lockedProvider || entry.provider === lockedProvider)
                        && (activeModelTag === 'all' || entry.tags.has(activeModelTag));
                    entry.button.classList.toggle('hidden', !visible);
                    updateModelButtonSelection(entry, selectedModel);
                    if (visible) groupVisibleCount += 1;
                });
                group.element.classList.toggle('hidden', groupVisibleCount === 0);
                visibleCount += groupVisibleCount;
            });

            modelListEmpty.classList.toggle('hidden', visibleCount !== 0);
            if (visibleCount === 0) {
                modelListEmpty.textContent = lockedProvider
                    ? `No ${lockedLabel} models found.`
                    : 'No models found.';
            }
            if (options.animate && !modelListAnimated) {
                modelListAnimated = true;
                container.classList.add('model-list-animate');
            }
        }

        function scheduleModelListRender(filter) {
            if (modelListRenderFrame) cancelAnimationFrame(modelListRenderFrame);
            modelListRenderFrame = requestAnimationFrame(() => {
                modelListRenderFrame = 0;
                renderModelList(filter);
            });
        }

        function openModelModal() {
            if (location.pathname !== '/model') {
                history.pushState({ modal: 'model' }, '', '/model');
            }
            const search = get('model-search');
            if (search) search.value = '';
            updateModelTagUi();
            // Build/update while hidden so opening animation never competes with DOM construction.
            renderModelList('', { animate: true });
            showModal('model-modal');
            // Prevent auto-focus on mobile to avoid keyboard popup
            if (search && window.innerWidth > 768) {
                requestAnimationFrame(() => search.focus({ preventScroll: true }));
            }
        }
        window.closeModelModal = (skipHistory = false) => {
            hideModal('model-modal');
            if (!skipHistory && location.pathname === '/model') {
                history.back();
            }
        };

        function selectModel(id, name) {
            if (window._visionPickerActive) {
                currentVisionModel = id;
                window._visionPickerActive = false;
                window.closeModelModal();
                _syncVisionModelDisplay();
                return;
            }
            if (isPromptCacheEnabled()) {
                const currentProvider = getModelApiProvider(get('model-select') ? get('model-select').value : '');
                const nextProvider = getModelApiProvider(id);
                if (currentProvider && nextProvider && currentProvider !== nextProvider) {
                    const curLabel = PROVIDER_LABELS[currentProvider] || currentProvider;
                    const nextLabel = PROVIDER_LABELS[nextProvider] || nextProvider;
                    showToast(`PromptCache 有効中は他API（${nextLabel}）のモデルに変更できません。現在: ${curLabel}`, 'warning', true);
                    return;
                }
            }
            const el = get('model-select');
            el.value = id;
            get('model-selector-text').innerText = name;
            window.closeModelModal();
            const event = new Event('change');
            el.dispatchEvent(event);
        }
        function selectModelById(id) {
            let name = id;
            for (const g of MODELS) {
                const found = g.items.find(i => i.id === id);
                if (found) { name = found.name; break; }
            }
            selectModel(id, name);
        }

        function populateAiSafeFormFields(d) {
            // Mirror a subset of the population in openSettingsModal for live update after AI apply
            if (!d) return;
            try {
                if (get('set-default-model')) get('set-default-model').value = d.default_model || get('set-default-model').value;
                if (get('set-default-vision-model')) get('set-default-vision-model').value = d.default_vision_model || 'gemini-3-flash-preview';
                if (get('set-default-search')) get('set-default-search').checked = !!d.default_enable_search;
                if (get('set-default-url-context')) get('set-default-url-context').checked = !!d.default_enable_url_context;
                if (get('set-default-maps')) get('set-default-maps').checked = !!d.default_enable_maps;
                if (get('set-default-python')) get('set-default-python').checked = !!d.default_enable_python;
                if (get('set-default-file-creation')) get('set-default-file-creation').checked = !!d.default_enable_file_creation;
                if (get('set-default-thinking')) get('set-default-thinking').checked = !!d.default_enable_thinking;
                if (get('set-default-sys-prompt')) get('set-default-sys-prompt').checked = !!d.default_enable_system_prompt;
                if (get('set-default-mcp')) get('set-default-mcp').checked = d.default_enable_mcp !== false;
                if (get('set-default-thinking-level')) get('set-default-thinking-level').value = d.default_thinking_level || 'high';
                if (get('set-default-thinking-budget')) get('set-default-thinking-budget').value = d.default_thinking_budget || 4096;
                if (get('set-default-reasoning-effort')) get('set-default-reasoning-effort').value = d.default_reasoning_effort || 'medium';
                if (get('set-default-safety')) get('set-default-safety').value = d.default_safety_setting || 'default';
                if (get('sys-prompt-text')) get('sys-prompt-text').value = d.system_prompt || '';
                if (get('set-global-sys-prompt-enabled')) get('set-global-sys-prompt-enabled').checked = d.system_prompt_enabled !== false;
                if (get('set-apply-global-sys-prompt')) get('set-apply-global-sys-prompt').checked = d.apply_global_system_prompt !== false;
                if (get('set-apply-auto-sys-prompt-notices')) get('set-apply-auto-sys-prompt-notices').checked = d.apply_auto_system_prompt_notices !== false;
                if (get('set-mic-transcribe-mode')) get('set-mic-transcribe-mode').value = d.mic_transcribe_mode || 'stt_api';
                if (get('set-stt-model')) get('set-stt-model').value = d.stt_model || 'gpt-4o-mini-transcribe';
                if (get('set-llm-transcribe-prompt')) get('set-llm-transcribe-prompt').value = d.llm_transcribe_prompt || '';
                if (get('set-enter-to-send')) get('set-enter-to-send').checked = !!d.enter_to_send;
                if (get('set-compact-prompt-mode') || get('set-minimal-prompt-mode') || get('set-prompt-bar-mode-normal')) {
                    writePromptBarModeToForm(!!d.compact_prompt_mode, !!d.minimal_prompt_mode);
                }
                if (d.minimal_prompt_mode) setMinimalPromptMode(true);
                else if (Object.prototype.hasOwnProperty.call(d, 'compact_prompt_mode') || Object.prototype.hasOwnProperty.call(d, 'minimal_prompt_mode')) {
                    setCompactPromptMode(!!d.compact_prompt_mode);
                }
                if (get('set-use-sw-cache')) get('set-use-sw-cache').checked = !!d.use_sw_cache;
                if (get('set-liquid-glass')) get('set-liquid-glass').checked = !!d.liquid_glass_enabled;
                applyLiquidGlassMode(!!d.liquid_glass_enabled);
                if (get('set-auto-search-links')) get('set-auto-search-links').checked = d.auto_search_on_links !== false;
                if (get('set-use-last-settings')) get('set-use-last-settings').checked = !!d.use_last_chat_settings;
                if (get('set-voice-studio-ui')) get('set-voice-studio-ui').checked = d.voice_studio_ui !== false;
                if (get('set-latency-metrics')) get('set-latency-metrics').checked = !!d.enable_latency_metrics;
                if (get('set-client-debug-log')) syncClientDebugLogToggle(!!d.enable_client_debug_log, 'ai-settings');
                if (get('set-bot-detect')) get('set-bot-detect').checked = d.bot_detection_enabled !== false;
                if (get('set-skip-2fa-google')) get('set-skip-2fa-google').checked = !!d.skip_2fa_on_google_login;
                if (get('set-default-2fa-method')) get('set-default-2fa-method').value = d.default_2fa_method || 'totp';
                // theme etc handled elsewhere if needed
            } catch (e) { /* element missing ok */ }
        }

        if (get('model-search')) {
            get('model-search').addEventListener('input', (e) => scheduleModelListRender(e.target.value));
        }
        if (get('model-tag-bar')) {
            get('model-tag-bar').addEventListener('click', (e) => {
                const btn = e.target.closest('.model-tag-btn');
                if (!btn) return;
                const t = btn.innerText.trim().toLowerCase();
                activeModelTag = MODEL_TAGS.includes(t) ? t : 'all';
                updateModelTagUi();
                renderModelList(get('model-search').value);
            });
            updateModelTagUi();
        }

        window.quickStart = (m) => {
            selectModelById(m);
            get('welcome-screen').classList.add('hidden');
        };

        const BROWSER_FAST_DISABLED_OPTIONS = [
            ['enable-search', 'search-container'],
            ['enable-url-context', 'url-context-container'],
            ['enable-maps', 'maps-grounding-container'],
            ['enable-sys-prompt', 'sys-prompt-option'],
            ['enable-prompt-cache', 'prompt-cache-container'],
            ['enable-mcp', 'mcp-container'],
            ['enable-file-creation', 'file-creation-container'],
        ];

        function applyBrowserFastModeRestrictions() {
            if (!browserFastModeEnabled) return;
            if (!browserFastPreviousOptions) {
                browserFastPreviousOptions = {
                    checks: Object.fromEntries(BROWSER_FAST_DISABLED_OPTIONS.map(([id]) => [id, !!(get(id) && get(id).checked)])),
                    coding: !!codingModeEnabled,
                };
            }
            BROWSER_FAST_DISABLED_OPTIONS.forEach(([id, containerId]) => {
                const checkbox = get(id);
                const container = get(containerId);
                if (checkbox) {
                    checkbox.checked = false;
                    checkbox.disabled = true;
                }
                if (container) container.classList.add('opacity-50', 'pointer-events-none');
            });
            if (codingModeEnabled) syncCodingModeUi(false, { persist: false });
            const codingCheckbox = get('enable-coding-mode');
            const codingContainer = get('coding-mode-container');
            if (codingCheckbox) codingCheckbox.disabled = true;
            if (codingContainer) codingContainer.classList.add('opacity-50', 'pointer-events-none');
            if (typeof syncMcpAutoSysRows === 'function') syncMcpAutoSysRows();
            refreshMinimalOptionsIfOpen();
        }

        function restoreBrowserFastModeOptions() {
            const previous = browserFastPreviousOptions;
            if (!previous) return;
            BROWSER_FAST_DISABLED_OPTIONS.forEach(([id, containerId]) => {
                const checkbox = get(id);
                const container = get(containerId);
                if (checkbox) {
                    checkbox.disabled = false;
                    if (previous && previous.checks && Object.prototype.hasOwnProperty.call(previous.checks, id)) {
                        checkbox.checked = !!previous.checks[id];
                    }
                }
                if (container) container.classList.remove('opacity-50', 'pointer-events-none');
            });
            const codingCheckbox = get('enable-coding-mode');
            const codingContainer = get('coding-mode-container');
            if (codingCheckbox) codingCheckbox.disabled = false;
            if (codingContainer) codingContainer.classList.remove('opacity-50', 'pointer-events-none');
            if (previous && previous.coding) syncCodingModeUi(true, { persist: false });
            browserFastPreviousOptions = null;
            if (typeof updatePromptCacheUi === 'function') updatePromptCacheUi();
            if (typeof syncMcpAutoSysRows === 'function') syncMcpAutoSysRows();
            refreshMinimalOptionsIfOpen();
        }

        function setBrowserFastModeEnabled(enabled, opts = {}) {
            browserFastModeEnabled = !!enabled;
            const toggle = get('enable-browser-fast-mode');
            if (toggle) toggle.checked = browserFastModeEnabled;
            const container = get('browser-fast-mode-container');
            if (container) {
                container.classList.toggle('ring-1', browserFastModeEnabled);
                container.classList.toggle('ring-amber-300', browserFastModeEnabled);
            }
            if (!browserFastModeEnabled && opts.clearKey !== false) {
                browserFastApiKey = '';
                browserFastApiKeyModel = '';
                browserFastBootstrap = null;
            }
            if (browserFastModeEnabled) applyBrowserFastModeRestrictions();
            else if (opts.restoreOptions !== false) restoreBrowserFastModeOptions();
        }

        function openBrowserFastModeModal(showWarning = true) {
            const warning = get('browser-fast-mode-warning');
            const ignoreRow = get('browser-fast-mode-ignore-row');
            if (warning) warning.classList.toggle('hidden', !showWarning);
            if (ignoreRow) ignoreRow.classList.toggle('hidden', !showWarning);
            const description = get('browser-fast-mode-key-description');
            const model = String(get('model-select') ? get('model-select').value : 'Gemini');
            if (description) description.textContent = `${model} のモデル別キー → 共通Geminiキーの順に、サーバーから自動取得します。`;
            showModal('browser-fast-mode-modal');
        }

        function browserFastBootstrapMatches(data, model, threadId, parentId) {
            if (!data || data.model !== model) return false;
            if (String(data.thread_id || '') !== String(threadId || '')) return false;
            return String(data.parent_id || '') === String(parentId || '');
        }

        async function fetchBrowserFastBootstrap(force = false) {
            const model = String(get('model-select') ? get('model-select').value : '').trim();
            const threadId = currentThreadId || null;
            const parentId = threadId ? (currentParentId || null) : null;
            if (!force && browserFastBootstrapMatches(browserFastBootstrap, model, threadId, parentId) && browserFastApiKey) {
                return browserFastBootstrap;
            }
            const response = await apiFetch('/api/browser_fast_mode/bootstrap', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model, thread_id: threadId, parent_id: parentId }),
            });
            const data = await response.json().catch(() => ({}));
            if (!response.ok || !data.api_key) {
                throw new Error(data.error || 'サーバー保存済みのGemini APIキーを取得できませんでした');
            }
            browserFastApiKey = String(data.api_key);
            browserFastApiKeyModel = model;
            browserFastBootstrap = data;
            return data;
        }

        async function requestBrowserFastModeEnable() {
            const model = String(get('model-select') ? get('model-select').value : '').toLowerCase();
            if (!model.startsWith('gemini-') || /(image|native-audio|tts|live)/.test(model)) {
                showToast('高速モードはGeminiテキストモデル専用です', 'warning', true);
                setBrowserFastModeEnabled(false);
                return;
            }
            if (currentImageUrls.length || uploadProgressState.active > 0 || browserFastLocalFiles.size) {
                showToast('高速モードへ切り替える前に添付ファイルをクリアしてください', 'warning', true);
                setBrowserFastModeEnabled(false);
                return;
            }
            const warningIgnored = (() => {
                try { return localStorage.getItem(BROWSER_FAST_IGNORE_WARNING_STORAGE) === '1'; } catch (e) { return false; }
            })();
            if (warningIgnored) {
                try {
                    await fetchBrowserFastBootstrap(true);
                    setBrowserFastModeEnabled(true, { clearKey: false });
                    showToast('高速モードを有効にしました', 'warning', false);
                } catch (error) {
                    setBrowserFastModeEnabled(false);
                    showToast(error.message || '高速モードを有効化できませんでした', 'error', true);
                }
                return;
            }
            openBrowserFastModeModal(!warningIgnored);
        }

        // Critical UI initializations - independent listener to survive errors in main init
        document.addEventListener('DOMContentLoaded', () => {
            if (get('menu-btn')) {
                get('menu-btn').onclick = () => { get('sidebar').classList.toggle('open'); get('overlay').classList.toggle('active'); };
            }
            if (get('overlay')) {
                get('overlay').onclick = () => { get('sidebar').classList.remove('open'); get('overlay').classList.remove('active'); };
            }
        });
