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
    SettingsCardSpec(SettingsTab.Data, "account-data", "アカウントデータ",
        "アカウントデータ 設定、チャット、Gem、ファイルなどをZIPで保存・復元できます。 エクスポートZIPを作成 インポート 重複データの修復") {
        SettingsDesc("設定、チャット、Gem、ファイルなどをZIPで保存・復元できます。ユーザー名、パスワード、2FA、パスキー、ログイン連携・セッション、権限、BAN情報は対象外です。")
        SettingsSmallButton("Webでデータ移行を開く", { extras.onWeb("/settings") }, Modifier.padding(top = 8.dp), fill = true)
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

internal fun accountCards(state: ChatState, extras: SettingsExtras): List<SettingsCardSpec> {
    val prefs = state.preferences
    return listOf(
        SettingsCardSpec(SettingsTab.Account, "account", "アカウント設定", "アカウント設定 ユーザー名変更 パスワード変更") {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                SettingsFieldLabel("ユーザー名")
                Text(prefs?.username.orEmpty(), fontSize = 13.sp, color = settingsLabelColor())
                SettingsSmallButton("Webでアカウント設定を開く", { extras.onWeb("/settings") }, fill = true)
            }
        },
        linkCard("google", "Google 連携", R.drawable.fa_brands_google, prefs?.googleEmail.orEmpty(), "Google アカウントでログインできるようになります。"),
        linkCard("minashin", "Minashin 連携", null, prefs?.minashinEmail.orEmpty(), "Minashin アカウントでログインできるようになります。"),
    )
}

private fun linkCard(key: String, title: String, icon: Int?, email: String, hint: String) =
    SettingsCardSpec(SettingsTab.Account, key, title, "$title 未連携 $hint", titleIcon = icon) {
        val web = LocalWebPalette.current
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            Box(Modifier.size(40.dp).clip(CircleShape).background(web.twBg(Tw.gray800)), contentAlignment = Alignment.Center) {
                if (icon != null) FaIcon(icon, null, size = 20.dp, tint = if (web.isLight) web.text else Tw.gray200)
            }
            Column {
                Text(if (email.isBlank()) "未連携" else "連携済み", fontSize = 14.sp, fontWeight = FontWeight.Bold,
                    color = if (web.isLight) web.text else Tw.gray200)
                SettingsDesc(email.ifBlank { hint })
            }
        }
    }

internal fun securityCards(state: ChatState, extras: SettingsExtras): List<SettingsCardSpec> = listOf(
    SettingsCardSpec(SettingsTab.Security, "e2ee", null, "E2EE (End-to-End Encryption) チャット履歴とファイルを暗号化します。") {
        val web = LocalWebPalette.current
        Row(verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) {
                Text("E2EE (End-to-End Encryption)", fontSize = 14.sp, fontWeight = FontWeight.Bold, color = if (web.isLight) web.text else Color.White)
                Text("チャット履歴とファイルを暗号化します。", fontSize = 12.sp, color = if (web.isLight) Color(92, 103, 121) else Tw.gray500)
            }
            WebToggle(state.preferences?.e2eeEnabled == true, { extras.onWeb("/settings") }, activeColor = Tw.green600)
        }
    },
    SettingsCardSpec(SettingsTab.Security, "sessions", "ログインセッション",
        "ログインセッション 現在ログイン中の端末を確認し、不要なセッションをログアウトできます。") {
        SettingsDesc("現在ログイン中の端末を確認し、不要なセッションをログアウトできます。")
        SettingsSmallButton("Webでセキュリティ設定を開く", { extras.onWeb("/settings") }, Modifier.padding(top = 8.dp), fill = true)
    },
)

internal fun twoFactorCards(state: ChatState, model: ChatViewModel, form: SettingsForm): List<SettingsCardSpec> {
    val security = state.security
    return listOf(
        SettingsCardSpec(SettingsTab.TwoFactor, "status", "Status", "Status Two-Factor Authentication ENABLED DISABLED") {
            val web = LocalWebPalette.current
            val enabled = security?.is2faEnabled == true
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text("Two-Factor Authentication", fontSize = 14.sp, color = if (web.isLight) web.text else Tw.gray300, modifier = Modifier.weight(1f))
                Text(if (enabled) "ENABLED" else "DISABLED", fontSize = 12.sp, fontWeight = FontWeight.Bold,
                    color = if (enabled) Color.White else Tw.gray400,
                    modifier = Modifier.clip(RoundedCornerShape(4.dp)).background(if (enabled) Tw.green600 else web.twBg(Tw.gray700))
                        .padding(horizontal = 8.dp, vertical = 2.dp))
            }
            state.securityError?.let { SettingsDesc(it, Modifier.padding(top = 8.dp), color = Tw.red300) }
        },
        SettingsCardSpec(SettingsTab.TwoFactor, "totp", "Authenticator App (TOTP)", "Authenticator App (TOTP) Google Authenticator, Authy, etc. Setup TOTP Verify",
            titleIcon = R.drawable.fa_solid_mobile_alt) {
            var code by remember { mutableStateOf("") }
            Text("Google Authenticator, Authy, etc.", fontSize = 12.sp, color = Tw.gray400, modifier = Modifier.padding(bottom = 12.dp))
            if (security?.hasTotp == true) {
                SettingsTextField(code, { code = it.filter(Char::isDigit).take(8) }, Modifier.fillMaxWidth(), placeholder = "000000", number = true)
                SettingsSmallButton("TOTPを無効化", { model.disableTotp(code); code = "" }, Modifier.padding(top = 8.dp),
                    tone = SettingsButtonTone.Red, enabled = code.isNotBlank() && !state.securityBusy, fill = true)
            } else {
                SettingsSmallButton("Setup TOTP", model::startTotpSetup, tone = SettingsButtonTone.Purple, enabled = !state.securityBusy, fill = true)
                state.securityTotpSecret?.let { secret ->
                    Column(Modifier.padding(top = 16.dp).fillMaxWidth().clip(RoundedCornerShape(4.dp)).background(LocalWebPalette.current.twBg(Tw.gray800)).padding(16.dp),
                        horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        SelectionContainer { Text(secret, fontSize = 12.sp, fontFamily = FontFamily.Monospace, color = Tw.gray400) }
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
                            SettingsTextField(code, { code = it.filter(Char::isDigit).take(8) }, Modifier.weight(1f), placeholder = "000000", number = true)
                            SettingsSmallButton("Verify", { model.enableTotp(code); code = "" }, tone = SettingsButtonTone.Blue,
                                enabled = code.isNotBlank() && !state.securityBusy)
                        }
                    }
                }
            }
        },
        SettingsCardSpec(SettingsTab.TwoFactor, "passkey", "Security Key / Passkey",
            "Security Key / Passkey YubiKey, Touch ID, Windows Hello. 複数のパスキーを登録できます。 Register Key 登録済みパスキー",
            titleIcon = R.drawable.fa_solid_key) {
            val web = LocalWebPalette.current
            Text("YubiKey, Touch ID, Windows Hello. 複数のパスキーを登録できます。", fontSize = 12.sp, color = Tw.gray400,
                modifier = Modifier.padding(bottom = 12.dp))
            SettingsSmallButton("Register Key", model::beginPasskeyRegistration, tone = SettingsButtonTone.Green,
                enabled = !state.securityBusy, fill = true)
            val keys = security?.passkeys.orEmpty()
            Text("登録済みパスキー: ${keys.size}", fontSize = 11.sp, color = Tw.gray400, modifier = Modifier.padding(top = 12.dp, bottom = 8.dp))
            keys.forEach { key ->
                val shape = RoundedCornerShape(6.dp)
                Row(
                    Modifier.fillMaxWidth().padding(bottom = 8.dp).clip(shape).background(web.twBg(Tw.gray900).copy(alpha = 0.7f))
                        .border(1.dp, web.twBorder(Tw.gray700), shape).padding(horizontal = 12.dp, vertical = 8.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Text(key.name, fontSize = 11.sp, color = if (web.isLight) web.text else Tw.gray200, modifier = Modifier.weight(1f))
                    SettingsSmallButton("削除", { model.removePasskey(key.id) }, tone = SettingsButtonTone.Red, fontSize = 10.sp)
                }
            }
        },
        SettingsCardSpec(SettingsTab.TwoFactor, "auth-prefs", "Authentication Preferences",
            "Authentication Preferences Google ログイン時に2FAをスキップ 既定の2要素認証方式 パスキーのみログインを有効化",
            titleIcon = R.drawable.fa_solid_user_shield) {
            val web = LocalWebPalette.current
            val labelColor = if (web.isLight) web.text else Tw.gray300
            Column(verticalArrangement = Arrangement.spacedBy(16.dp)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text("Google ログイン時に2FAをスキップ", fontSize = 12.sp, color = labelColor, modifier = Modifier.weight(1f))
                    WebToggle(form.skip2faGoogle, { form.skip2faGoogle = it }, activeColor = Tw.emerald500)
                }
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text("既定の2要素認証方式", fontSize = 12.sp, color = labelColor, modifier = Modifier.weight(1f))
                    SettingsSelect(form.default2fa, webOptions("totp" to "Authenticator App (TOTP)", "webauthn" to "Security Key / Passkey"),
                        { form.default2fa = it }, fillWidth = false, fontSize = 12.sp)
                }
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text("パスキーのみログインを有効化", fontSize = 12.sp, color = labelColor, modifier = Modifier.weight(1f))
                    WebToggle(form.passkeyOnly, { form.passkeyOnly = it }, activeColor = Tw.emerald500)
                }
            }
            if (security?.passkeys.isNullOrEmpty()) SettingsDesc("パスキー未登録のため有効化できません。", Modifier.padding(top = 8.dp), color = Tw.red300)
        },
    )
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

internal fun mcpCards(state: ChatState, model: ChatViewModel, extras: SettingsExtras): List<SettingsCardSpec> = listOf(
    SettingsCardSpec(SettingsTab.Mcp, "servers", "外部MCPサーバー連携",
        "外部MCPサーバー連携 MCP（Model Context Protocol）で、チャット中に Gmail や Google Drive などの外部ツールをモデルから利用できるようになります。 登録済みサーバー 有効") {
        val web = LocalWebPalette.current
        SettingsDesc("MCP（Model Context Protocol）で、チャット中に Gmail や Google Drive などの外部ツールをモデルから利用できるようになります。サーバーごとに「有効」にすると、そのツールがチャットで使えるようになります（変更を伴う操作は実行前に確認されます）。",
            Modifier.padding(bottom = 12.dp))
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            FaIcon(R.drawable.fa_solid_server, null, size = 12.dp, tint = Tw.gray400)
            Text("登録済みサーバー", fontSize = 12.sp, fontWeight = FontWeight.Bold, color = if (web.isLight) web.text else Color.White,
                modifier = Modifier.weight(1f))
            Text("${state.mcpServers.size}件", fontSize = 10.sp, color = Tw.gray500)
        }
        Column(Modifier.padding(top = 8.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            state.mcpServers.forEach { server ->
                SettingsSubBox {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Column(Modifier.weight(1f)) {
                            Text(server.name, fontSize = 12.sp, fontWeight = FontWeight.Bold, color = if (web.isLight) web.text else Color.White)
                            SettingsDesc("${server.toolCount} tools · ${server.connectionState}")
                        }
                        WebToggle(server.enabled, { model.setMcpServerEnabled(server, it) })
                    }
                }
            }
        }
        // ANDROID_ONLY.md: OAuth clients and Bearer/OAuth custom servers are managed on Web.
        SettingsSmallButton("WebでMCP秘密設定を開く", { extras.onWeb("/settings") }, Modifier.padding(top = 12.dp), fill = true)
    },
)
