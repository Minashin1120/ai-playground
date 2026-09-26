@file:OptIn(androidx.compose.foundation.layout.ExperimentalLayoutApi::class)

package com.minashin1120.aiplayground.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.minashin1120.aiplayground.AppUpdatePhase
import com.minashin1120.aiplayground.AppUpdateUiState
import com.minashin1120.aiplayground.BuildConfig
import com.minashin1120.aiplayground.ChatState
import com.minashin1120.aiplayground.ChatViewModel
import com.minashin1120.aiplayground.R
import com.minashin1120.aiplayground.data.CacheCategory
import com.minashin1120.aiplayground.data.HistoryCacheMode
import com.minashin1120.aiplayground.data.formatByteSize

/** Android-only cards and the tabs whose native implementation follows in a later release. */
internal class SettingsExtras(
    val appUpdate: AppUpdateUiState,
    val onCheckForUpdate: () -> Unit,
    val onBubble: () -> Unit,
    val onWeb: (String) -> Unit,
    val onConfirmCacheClear: (CacheCategory) -> Unit,
)

/** ANDROID_ONLY.md: the "Android" card at the end of the General tab (account, app update, bubble). */
internal fun androidCard(state: ChatState, extras: SettingsExtras): SettingsCardSpec =
    SettingsCardSpec(SettingsTab.General, "android", "Android", "Android アカウント アプリ更新 更新を確認 バブル") {
        val update = extras.appUpdate
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            SettingsFieldLabel("アカウント")
            Text(state.account?.name.orEmpty(), fontSize = 12.sp, color = settingsLabelColor())
            SettingsFieldLabel("アプリ更新", Modifier.padding(top = 4.dp))
            SettingsDesc("現在のバージョン: ${BuildConfig.VERSION_NAME}")
            when (update.phase) {
                AppUpdatePhase.Checking -> SettingsDesc("更新を確認中…")
                AppUpdatePhase.UpToDate -> SettingsDesc("最新のAndroid版を使用しています。")
                AppUpdatePhase.Available -> update.update?.let { SettingsDesc("Android版 ${it.versionName} が利用できます。") }
                AppUpdatePhase.Error -> SettingsDesc(update.errorMessage ?: "更新を確認できませんでした。", color = Tw.red300)
                else -> Unit
            }
            val busy = update.phase in setOf(AppUpdatePhase.Checking, AppUpdatePhase.Downloading, AppUpdatePhase.Ready,
                AppUpdatePhase.AwaitingInstallPermission, AppUpdatePhase.Installing)
            SettingsSmallButton(if (update.phase == AppUpdatePhase.Checking) "確認中…" else "更新を確認", extras.onCheckForUpdate,
                enabled = !busy, fill = true)
            SettingsFieldLabel("バブル", Modifier.padding(top = 4.dp))
            SettingsSmallButton(
                if (android.os.Build.VERSION.SDK_INT >= com.minashin1120.aiplayground.ANDROID_17_APP_BUBBLE_API) "バブルに追加する方法" else "バブルで開く",
                extras.onBubble, fill = true,
            )
        }
    }

internal fun dataCards(state: ChatState, model: ChatViewModel, form: SettingsForm, extras: SettingsExtras): List<SettingsCardSpec> = listOf(
    // ANDROID_ONLY.md: the device cache card replaces the Web Service Worker cache card.
    SettingsCardSpec(SettingsTab.Data, "device-cache", "端末キャッシュ", "端末キャッシュ 履歴の保存範囲 表示済み部分のみ 全件同期 モバイルデータ通信 削除") {
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            SettingsDesc("表示済みのチャット履歴とファイルは、Androidのシステムキャッシュとは別の暗号化領域に保存されます。")
            SettingsFieldLabel("履歴の保存範囲")
            SettingsSelect(
                if (state.historyCacheMode == HistoryCacheMode.FULL) "full" else "viewed",
                webOptions("viewed" to "表示済み部分のみ", "full" to "全件同期"),
                { model.saveCacheSettings(if (it == "full") HistoryCacheMode.FULL else HistoryCacheMode.VIEWED, state.cacheMobileDataAllowed) },
            )
            SettingsCheck("モバイルデータ通信でもキャッシュを同期", state.cacheMobileDataAllowed,
                { model.saveCacheSettings(state.historyCacheMode, it) })
            val stats = state.offlineCacheStats
            SettingsDesc("チャット履歴: ${formatByteSize(stats.historyBytes)} / ファイル: ${formatByteSize(stats.fileBytes)}")
            if (state.cacheSyncing) {
                val fraction = if (state.cacheSyncTotal > 0) (state.cacheSyncProgress.toFloat() / state.cacheSyncTotal).coerceIn(0f, 1f) else 0f
                UsageBar(fraction)
                SettingsSmallButton("同期をキャンセル", model::cancelCacheSync, fill = true)
            } else SettingsSmallButton("今すぐ全件同期", model::syncOfflineCache, fill = true)
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                SettingsSmallButton("履歴を削除", { extras.onConfirmCacheClear(CacheCategory.CHAT_HISTORY) }, Modifier.weight(1f), tone = SettingsButtonTone.Orange)
                SettingsSmallButton("ファイルを削除", { extras.onConfirmCacheClear(CacheCategory.FILES) }, Modifier.weight(1f), tone = SettingsButtonTone.Orange)
            }
        }
    },
    SettingsCardSpec(SettingsTab.Data, "storage", "ストレージ", "ストレージ 更新 アップロード済みファイルの使用量を表示します。") {
        val web = LocalWebPalette.current
        val storage = state.storage
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text(
                when {
                    storage == null -> "読み込み中..."
                    storage.unlimited -> "${storage.usedMb} MB / 無制限"
                    else -> "${storage.usedMb} MB / ${storage.limitMb} MB"
                },
                fontSize = 12.sp, color = if (web.isLight) web.text else Tw.gray300,
            )
            UsageBar(if (storage == null || storage.unlimited || storage.limitBytes <= 0) 0f else (storage.usedBytes.toFloat() / storage.limitBytes).coerceIn(0f, 1f))
            Row(verticalAlignment = Alignment.CenterVertically) {
                SettingsDesc("アップロード済みファイルの使用量を表示します。", Modifier.weight(1f))
                SettingsSmallButton("更新", { model.loadStorageUsage() }, fontSize = 10.sp)
            }
        }
    },
    SettingsCardSpec(SettingsTab.Data, "debug", "デバッグ設定",
        "デバッグ設定 レスポンス速度の計測 プロンプト送信から初回トークンまでの待機時間を計測し、サーバーへ記録します。 デバッグログの拡張送信") {
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            SettingsSwitchRow("レスポンス速度の計測", "プロンプト送信から初回トークンまでの待機時間を計測し、サーバーへ記録します。",
                form.latencyMetrics, { form.latencyMetrics = it })
            SettingsSwitchRow("デバッグログの拡張送信", "ブラウザのコンソールログをサーバーへ送信します（トラブルシューティング用）。",
                form.clientDebugLog, { form.clientDebugLog = it }, accent = Tw.amber600)
        }
    },
)

@Composable
private fun UsageBar(fraction: Float) {
    val web = LocalWebPalette.current
    val shape = RoundedCornerShape(4.dp)
    Box(Modifier.fillMaxWidth().height(8.dp).clip(shape).background(web.twBg(Tw.gray800)).border(1.dp, web.twBorder(Tw.gray700), shape)) {
        Box(Modifier.fillMaxHeight().fillMaxWidth(fraction).background(Tw.blue500.copy(alpha = 0.8f)))
    }
}

internal fun feedbackCards(state: ChatState, model: ChatViewModel, notify: (String) -> Unit): List<SettingsCardSpec> = listOf(
    SettingsCardSpec(SettingsTab.Feedback, "send", "フィードバック送信", "フィードバック送信 タイトル (任意) バグ報告・要望などを入力してください 送信",
        titleIcon = R.drawable.fa_solid_bug) {
        var title by remember { mutableStateOf("") }
        var message by remember { mutableStateOf("") }
        Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
            SettingsTextField(title, { title = it.take(200) }, Modifier.fillMaxWidth(), placeholder = "タイトル (任意)")
            SettingsTextField(message, { message = it.take(100_000) }, Modifier.fillMaxWidth(), placeholder = "バグ報告・要望などを入力してください",
                minHeight = 112.dp, singleLine = false)
            SettingsSmallButton("送信", {
                if (message.isBlank()) notify("フィードバック内容を入力してください")
                else { model.submitFeedback(title.trim(), message.trim()); title = ""; message = "" }
            }, tone = SettingsButtonTone.Blue, enabled = !state.feedbackBusy, fill = true)
        }
    },
    SettingsCardSpec(SettingsTab.Feedback, "mine", "あなたのフィードバック", "あなたのフィードバック") {
        val web = LocalWebPalette.current
        Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
            state.feedbackItems.forEach { item ->
                val shape = RoundedCornerShape(4.dp)
                Column(
                    Modifier.fillMaxWidth().clip(shape).background(web.twBg(Tw.gray800).copy(alpha = 0.5f))
                        .border(1.dp, web.twBorder(Tw.gray700), shape).padding(8.dp),
                ) {
                    Text(item.createdAt, fontSize = 11.sp, color = Tw.gray400)
                    Text(item.title.ifBlank { "No Title" }, fontSize = 14.sp, fontWeight = FontWeight.Bold, color = web.text)
                    Text(item.message, fontSize = 14.sp, color = web.text)
                    Text("Status: ${item.status}", fontSize = 11.sp, color = Tw.gray400, modifier = Modifier.padding(top = 4.dp))
                    if (item.adminReply.isNotBlank()) Text("Reply: ${item.adminReply}", fontSize = 11.sp, color = Tw.green300,
                        modifier = Modifier.padding(top = 4.dp))
                }
            }
        }
    },
)

internal fun mcpCards(state: ChatState, model: ChatViewModel, extras: SettingsExtras, ops: AccountOps): List<SettingsCardSpec> {
    return listOf(
        SettingsCardSpec(SettingsTab.Mcp, "servers", "外部MCPサーバー連携",
            "外部MCPサーバー連携 MCP（Model Context Protocol）で、チャット中に Gmail や Google Drive などの外部ツールをモデルから利用できるようになります。 Google Workspace 連携（OAuthクライアント情報） 登録済みサーバー 接続テスト ツール一覧") {
            val web = LocalWebPalette.current
            val status = remember { McpStatus() }
            SettingsDesc("MCP（Model Context Protocol）で、チャット中に Gmail や Google Drive などの外部ツールをモデルから利用できるようになります。サーバーごとに「有効」にすると、そのツールがチャットで使えるようになります（変更を伴う操作は実行前に確認されます）。",
                Modifier.padding(bottom = 12.dp))
            // ANDROID_ONLY.md: OAuth client credentials are registered on Web.
            SettingsSubBox(Modifier.padding(bottom = 12.dp)) {
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    FaIcon(R.drawable.fa_brands_google, null, size = 12.dp, tint = Tw.cyan300)
                    Text("Google Workspace 連携（OAuthクライアント情報）", fontSize = 12.sp, fontWeight = FontWeight.Bold, color = Tw.cyan300)
                }
                SettingsDesc("OAuthクライアント情報の登録と、認証が必要なサーバーの認証はWeb版で行います。")
                SettingsSmallButton("WebでMCP秘密設定を開く", { extras.onWeb("/settings") }, fill = true)
            }
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                FaIcon(R.drawable.fa_solid_server, null, size = 12.dp, tint = Tw.gray400)
                Text("登録済みサーバー", fontSize = 12.sp, fontWeight = FontWeight.Bold, color = if (web.isLight) web.text else Color.White,
                    modifier = Modifier.weight(1f))
                Text("${state.mcpServers.size}件", fontSize = 10.sp, color = Tw.gray500)
            }
            Column(Modifier.padding(top = 8.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                if (state.mcpServers.isEmpty() && !state.mcpBusy) Text("まだサーバーがありません。上のカスタム追加フォームから登録するか、Google Workspace の認証をしてください。",
                    fontSize = 11.sp, color = Tw.gray600, modifier = Modifier.padding(vertical = 8.dp))
                state.mcpServers.forEach { server -> McpServerRow(server, model, ops, status) }
            }
            val loadingText = if (state.mcpBusy && state.mcpServers.isEmpty()) "読み込み中..." else null
            (loadingText ?: status.message)?.let { Text(it, fontSize = 11.sp, color = if (status.error) Color(0xFFF87171) else Tw.gray400, modifier = Modifier.padding(top = 8.dp)) }
        },
        SettingsCardSpec(SettingsTab.Mcp, "custom", "カスタムMCPサーバーを追加",
            // `fa-plus-circle` is not in the Web icon subset, so the title has no glyph there either.
            "カスタムMCPサーバーを追加 表示名 MCP URL（Streamable HTTP） 認証方式 認証なし 説明（任意） 接続テストして追加") {
            McpCustomForm(model, ops)
        },
    )
}

/** `#mcp-status-msg` for the whole list. */
@Stable
private class McpStatus {
    var message by mutableStateOf<String?>(null)
    var error by mutableStateOf(false)
    fun set(text: String?, isError: Boolean) { message = text; error = isError }
}

@Composable
private fun McpServerRow(server: com.minashin1120.aiplayground.data.McpServerInfo, model: ChatViewModel, ops: AccountOps, status: McpStatus) {
    val web = LocalWebPalette.current
    var toolsOpen by remember(server.id) { mutableStateOf(false) }
    var tools by remember(server.id) { mutableStateOf<List<com.minashin1120.aiplayground.data.McpTool>?>(null) }
    var toolsError by remember(server.id) { mutableStateOf<String?>(null) }
    var confirmDelete by remember { mutableStateOf(false) }
    SettingsSubBox {
        FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text(server.name, fontSize = 12.sp, fontWeight = FontWeight.Bold, color = if (web.isLight) web.text else Color.White,
                modifier = Modifier.align(Alignment.CenterVertically))
            McpBadge(if (server.isPreset) "プリセット" else "カスタム", if (server.isPreset) Tw.blue700.copy(alpha = 0.5f) else Tw.purple700.copy(alpha = 0.5f),
                if (server.isPreset) Tw.blue100 else Tw.purple100, Modifier.align(Alignment.CenterVertically), round = false)
            val (label, kind) = when {
                server.authType == "none" -> "認証不要" to "neutral"
                server.authStatus == "connected" -> "接続済み" to "ok"
                server.authStatus == "expired" -> "期限切れ（再認証）" to "expired"
                server.authStatus == "needs_auth" -> "認証が必要" to "auth"
                else -> "未認証" to "neutral"
            }
            val (bg, fg) = when (kind) {
                "ok" -> Tw.emerald700.copy(alpha = 0.6f) to Tw.emerald100
                "expired" -> Tw.red700.copy(alpha = 0.6f) to Tw.red100
                "auth" -> Tw.amber600.copy(alpha = 0.5f) to Tw.amber100
                else -> web.twBg(Tw.gray700) to Tw.gray300
            }
            McpBadge(label, bg, fg, Modifier.align(Alignment.CenterVertically))
        }
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            Spacer(Modifier.weight(1f))
            if (!server.isPreset) SettingsSmallButton("削除", { confirmDelete = true }, tone = SettingsButtonTone.Red, fontSize = 10.sp)
            WebToggle(server.enabled, { enabled ->
                status.set(if (enabled) "有効化しています..." else "無効化しています...", false)
                model.setMcpServerEnabled(server, enabled)
                status.set(if (enabled) "有効にしました。チャットのモデルへツールが公開されます。" else "無効にしました。", false)
            }, contentDescription = if (server.enabled) "無効化" else "有効化")
        }
        if (server.url.isNotBlank()) Text(server.url, fontSize = 10.sp, color = Tw.gray500)
        if (server.description.isNotBlank()) Text(server.description, fontSize = 10.sp, color = Tw.gray500)
        if (server.lastError.isNotBlank()) Text(server.lastError, fontSize = 10.sp, color = Tw.red400)
        if (server.authType == "bearer") Text("Bearer トークン ${if (server.authHasToken) "（保存済み・********）" else "（未設定）"}",
            fontSize = 10.sp, color = if (server.authHasToken) Tw.emerald300 else Tw.amber300)
        if (server.authType == "oauth" && !server.oauthClientRegistered) SettingsDesc("OAuthクライアント情報を保存すると「認証する」が使えます。")
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Text(if (server.toolCount > 0) "${server.toolCount}ツール" else "ツール未取得", fontSize = 10.sp,
                color = if (server.toolCount > 0) Tw.emerald300 else Tw.gray500)
            SettingsSmallButton("ツール一覧", {
                toolsOpen = true
                tools = null
                toolsError = null
                ops.run("取得に失敗しました") {
                    try { tools = mcpTools(server.id) } catch (e: Exception) {
                        if (e is kotlinx.coroutines.CancellationException) throw e
                        toolsError = serverErrorText(e, "取得に失敗しました")
                    }
                }
            }, fontSize = 10.sp)
            Spacer(Modifier.weight(1f))
            SettingsSmallButton("接続テスト", {
                status.set("接続テスト中...", false)
                ops.run("接続テストに失敗しました") {
                    try {
                        val reply = mcpTest(server.id)
                        val probe = reply.optJSONObject("probe")
                        probe?.optString("message")?.takeIf { it.isNotBlank() }?.let { status.set(it, !probe.optBoolean("ok")) }
                        model.loadMcpServers()
                    } catch (e: Exception) {
                        if (e is kotlinx.coroutines.CancellationException) throw e
                        status.set(serverErrorText(e, "接続テストに失敗しました"), true)
                    }
                }
            }, fontSize = 10.sp, icon = null)
            Text(when (server.connectionState) {
                "error" -> "エラー"
                "connected" -> "接続OK"
                "needs_auth" -> "認証待ち"
                else -> "未接続"
            }, fontSize = 9.sp, color = Tw.gray600)
        }
        if (toolsOpen) {
            val list = tools
            when {
                toolsError != null -> Text(toolsError.orEmpty(), fontSize = 10.sp, color = Tw.red400)
                list == null -> Text("読み込み中...", fontSize = 10.sp, color = Tw.gray500)
                list.isEmpty() -> Text("ツール一覧がありません。「接続テスト」で取得してください。", fontSize = 10.sp, color = Tw.gray600)
                else -> Column(Modifier.fillMaxWidth().clip(RoundedCornerShape(4.dp)).background(Color.Black.copy(alpha = 0.2f))
                    .border(1.dp, web.twBorder(Tw.gray800), RoundedCornerShape(4.dp)).padding(8.dp)) {
                    list.forEach { tool ->
                        Row(Modifier.fillMaxWidth().padding(vertical = 4.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            Column(Modifier.weight(1f)) {
                                Text(tool.name, fontSize = 11.sp, fontFamily = FontFamily.Monospace, color = Tw.cyan200)
                                if (tool.description.isNotBlank()) Text(tool.description, fontSize = 10.sp, color = Tw.gray500, maxLines = 2,
                                    overflow = androidx.compose.ui.text.style.TextOverflow.Ellipsis)
                            }
                            McpBadge(if (tool.readOnly) "読み取り" else "変更",
                                if (tool.readOnly) Tw.emerald800.copy(alpha = 0.4f) else Tw.amber800.copy(alpha = 0.4f),
                                if (tool.readOnly) Tw.emerald200 else Tw.amber200, round = false)
                        }
                    }
                }
            }
        }
    }
    if (confirmDelete) BrowserConfirmDialog("このカスタムMCPサーバーを削除しますか？") { ok ->
        confirmDelete = false
        if (ok) ops.run("削除に失敗しました") {
            try { mcpDelete(server.id); status.set("削除しました。", false); model.loadMcpServers() } catch (e: Exception) {
                if (e is kotlinx.coroutines.CancellationException) throw e
                status.set(serverErrorText(e, "削除に失敗しました"), true)
            }
        }
    }
}

@Composable
private fun McpBadge(text: String, background: Color, color: Color, modifier: Modifier = Modifier, round: Boolean = true) {
    Text(text, fontSize = 9.sp, fontWeight = FontWeight.Bold, color = color,
        modifier = modifier.clip(if (round) CircleShape else RoundedCornerShape(4.dp)).background(background)
            .padding(horizontal = if (round) 8.dp else 6.dp, vertical = 2.dp))
}

/** `カスタムMCPサーバーを追加` (unauthenticated servers only, ANDROID_ONLY.md). */
@Composable
private fun McpCustomForm(model: ChatViewModel, ops: AccountOps) {
    var name by remember { mutableStateOf("") }
    var url by remember { mutableStateOf("") }
    var description by remember { mutableStateOf("") }
    var status by remember { mutableStateOf("") }
    var statusColor by remember { mutableStateOf(Tw.gray400) }
    var busy by remember { mutableStateOf(false) }
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Column { SettingsFieldLabel("表示名"); SettingsTextField(name, { name = it.take(120) }, Modifier.fillMaxWidth(), placeholder = "例: 社内ナレッジMCP") }
        Column { SettingsFieldLabel("MCP URL（Streamable HTTP）"); SettingsTextField(url, { url = it.take(2048) }, Modifier.fillMaxWidth(), placeholder = "https://example.com/mcp") }
        Column {
            SettingsFieldLabel("認証方式")
            SettingsSelect("none", webOptions("none" to "認証なし"), {})
            SettingsDesc("Bearer トークンと OAuth 2.0 のサーバーはWeb版で追加します。", Modifier.padding(top = 4.dp))
        }
        Column { SettingsFieldLabel("説明（任意）"); SettingsTextField(description, { description = it.take(500) }, Modifier.fillMaxWidth(), placeholder = "このサーバーの説明") }
        Row(Modifier.fillMaxWidth().padding(top = 4.dp), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp, Alignment.End)) {
            Text(status, fontSize = 10.sp, color = statusColor, modifier = Modifier.weight(1f))
            WebButton(onClick = {
                if (name.isBlank() || url.isBlank()) { status = "表示名とURLは必須です"; statusColor = Color(0xFFF87171) }
                else {
                    busy = true
                    status = "接続テスト中..."; statusColor = Tw.gray400
                    ops.run("追加に失敗しました") {
                        try {
                            val reply = mcpAdd(name.trim(), url.trim(), description.trim())
                            val probe = reply.optJSONObject("probe")
                            status = probe?.optString("message")?.ifBlank { null } ?: "追加しました"
                            statusColor = if (probe?.optBoolean("ok") == true) Color(0xFF34D399) else Color(0xFFFBBF24)
                            name = ""; url = ""; description = ""
                            model.loadMcpServers()
                        } catch (e: Exception) {
                            if (e is kotlinx.coroutines.CancellationException) throw e
                            status = serverErrorText(e, "追加に失敗しました"); statusColor = Color(0xFFF87171)
                        } finally { busy = false }
                    }
                }
            }, variant = WebButtonVariant.Primary, enabled = !busy, radius = 11.dp,
                contentPadding = PaddingValues(horizontal = 14.dp, vertical = 6.dp), fontSize = 12.sp) {
                Text("接続テストして追加")
            }
        }
        SettingsDesc("URLはサーバー側で安全性（SSRF対策）を検査します。内部ネットワーク・ループバック等への接続はできません。", color = Tw.gray400)
    }
}
