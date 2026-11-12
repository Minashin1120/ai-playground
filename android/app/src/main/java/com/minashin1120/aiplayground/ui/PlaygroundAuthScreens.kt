package com.minashin1120.aiplayground.ui

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.autofill.ContentType
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentType
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.minashin1120.aiplayground.ChatState
import com.minashin1120.aiplayground.ChatViewModel
import com.minashin1120.aiplayground.ImportSettingChange
import com.minashin1120.aiplayground.data.BanAppealState
import com.minashin1120.aiplayground.data.ConnectionStatus
import com.minashin1120.aiplayground.data.Diagnostics
import com.minashin1120.aiplayground.data.ServerOrigin
import com.minashin1120.aiplayground.data.defaultMessage
import com.minashin1120.aiplayground.data.originLabel
import kotlinx.coroutines.launch

// region: Banned screen
@Composable
internal fun BannedScreen(
    reason: String,
    bannedAt: String,
    appeal: BanAppealState,
    onSubmitAppeal: suspend (String) -> Boolean,
    onLogout: () -> Unit,
) {
    val colors = MaterialTheme.colorScheme
    val scope = rememberCoroutineScope()
    var message by rememberSaveable { mutableStateOf("") }
    val success = Color(34, 197, 94)
    Surface(color = colors.background, contentColor = colors.onBackground, modifier = Modifier.fillMaxSize()) {
        Box(
            Modifier.fillMaxSize().imePadding().padding(24.dp),
            contentAlignment = Alignment.Center,
        ) {
            Card(
                modifier = Modifier.fillMaxWidth().widthIn(max = 520.dp),
                colors = CardDefaults.cardColors(containerColor = colors.surfaceContainer),
                border = androidx.compose.foundation.BorderStroke(1.dp, colors.error.copy(alpha = 0.35f)),
            ) {
                Column(
                    Modifier.padding(24.dp).verticalScroll(rememberScrollState()),
                    verticalArrangement = Arrangement.spacedBy(10.dp),
                ) {
                    Text("アカウントはBANされています", style = MaterialTheme.typography.headlineSmall,
                        color = colors.error, fontWeight = FontWeight.Bold)
                    Text("ボット検出により、このアカウントは停止中です。", color = colors.onSurfaceVariant)
                    Text("解除できるのは管理者のみです。", color = colors.onSurfaceVariant)
                    if (reason.isNotBlank()) Text("理由: $reason", style = MaterialTheme.typography.bodySmall)
                    if (bannedAt.isNotBlank()) Text("日時: $bannedAt", style = MaterialTheme.typography.bodySmall)
                    HorizontalDivider(Modifier.padding(top = 8.dp), color = colors.outlineVariant)
                    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                        Text("異議申し立て", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.Bold)
                        Text("管理者に通知されます", fontSize = 11.sp, color = colors.primary,
                            modifier = Modifier.clip(CircleShape).background(colors.primary.copy(alpha = 0.14f))
                                .border(1.dp, colors.primary.copy(alpha = 0.35f), CircleShape).padding(horizontal = 6.dp, vertical = 2.dp))
                    }
                    Text("誤判定の可能性がある場合は、状況を詳しく記入してください。", color = colors.onSurfaceVariant)
                    if (appeal.latestAt.isNotBlank() || appeal.latestStatus.isNotBlank()) {
                        Text("最新の申し立て: ${appeal.latestAt} / 状態: ${appeal.latestStatus}", style = MaterialTheme.typography.bodySmall)
                        if (appeal.adminReply.isNotBlank()) BanAlert("管理者からの返信: ${appeal.adminReply}", success)
                    }
                    if (appeal.submitted) BanAlert("申し立てを送信しました。管理者が確認します。", success)
                    if (appeal.error.isNotBlank()) BanAlert(appeal.error, colors.error)
                    if (appeal.blocked) {
                        BanAlert("異議申し立てはブロックされています。" +
                            (if (appeal.blockReason.isNotBlank()) "理由: ${appeal.blockReason}" else ""), colors.error)
                    } else {
                        OutlinedTextField(
                            value = message,
                            onValueChange = { message = it },
                            modifier = Modifier.fillMaxWidth().heightIn(min = 120.dp),
                            placeholder = { Text("例: 利用環境や操作内容、誤判定の可能性について記載してください。") },
                            enabled = !appeal.sending,
                            textStyle = MaterialTheme.typography.bodyMedium,
                        )
                        Text("10〜3000文字で入力してください。", style = MaterialTheme.typography.bodySmall, color = colors.onSurfaceVariant)
                        Button(
                            onClick = { scope.launch { if (onSubmitAppeal(message)) message = "" } },
                            enabled = !appeal.sending,
                            modifier = Modifier.fillMaxWidth(),
                        ) {
                            Text("異議申し立てを送信")
                        }
                    }
                    OutlinedButton(onClick = onLogout, modifier = Modifier.fillMaxWidth()) {
                        Text("ログアウト")
                    }
                }
            }
        }
    }
}

/** Web `banned.html` `.alert.success` / `.alert.error`. */
@Composable
private fun BanAlert(text: String, tint: Color) {
    Text(text, style = MaterialTheme.typography.bodySmall, color = tint,
        modifier = Modifier.fillMaxWidth().clip(RoundedCornerShape(8.dp)).background(tint.copy(alpha = 0.15f))
            .border(1.dp, tint.copy(alpha = 0.35f), RoundedCornerShape(8.dp)).padding(horizontal = 10.dp, vertical = 8.dp))
}
// endregion

// region: Connection banner
/**
 * Web `#offline-banner` (`connection_monitor.js`): a pill with the state's colours, icon and wording.
 * The "再試行" button is Android's own (ANDROID_ONLY.md).
 */
@Composable
internal fun ConnectionBanner(status: ConnectionStatus, message: String, onRetry: () -> Unit) {
    val (fg, border, background) = when (status) {
        ConnectionStatus.MAINTENANCE -> Triple(Color(0xFFE9D5FF), Color(192, 132, 252).copy(alpha = 0.62f), Color(32, 12, 52).copy(alpha = 0.94f))
        ConnectionStatus.UNSTABLE -> Triple(Color(0xFFFEF3C7), Color(245, 158, 11).copy(alpha = 0.55f), Color(40, 24, 0).copy(alpha = 0.92f))
        ConnectionStatus.ONLINE -> Triple(Color(0xFFD1FAE5), Color(52, 211, 153).copy(alpha = 0.58f), Color(3, 34, 26).copy(alpha = 0.92f))
        ConnectionStatus.SERVER_DOWN -> Triple(Color(0xFFFECACA), Color(239, 68, 68).copy(alpha = 0.75f), Color(48, 5, 12).copy(alpha = 0.95f))
        ConnectionStatus.OFFLINE, ConnectionStatus.UNKNOWN -> Triple(Color(0xFFFECACA), Color(248, 113, 113).copy(alpha = 0.6f), Color(28, 8, 14).copy(alpha = 0.92f))
    }
    val icon = when (status) {
        ConnectionStatus.MAINTENANCE -> com.minashin1120.aiplayground.R.drawable.fa_solid_screwdriver_wrench
        ConnectionStatus.UNSTABLE -> com.minashin1120.aiplayground.R.drawable.fa_solid_exclamation_triangle
        ConnectionStatus.ONLINE -> com.minashin1120.aiplayground.R.drawable.fa_solid_check_circle
        ConnectionStatus.SERVER_DOWN -> com.minashin1120.aiplayground.R.drawable.fa_solid_server
        ConnectionStatus.OFFLINE, ConnectionStatus.UNKNOWN -> com.minashin1120.aiplayground.R.drawable.fa_solid_unlink
    }
    val reduce = LocalReduceMotion.current
    // Web `bannerPulse`: the icon pulses once whenever the connection gets worse.
    val pulse = remember { Animatable(1f) }
    LaunchedEffect(status) {
        if (reduce || status == ConnectionStatus.ONLINE) return@LaunchedEffect
        pulse.animateTo(1.25f, tween(PlaygroundMotion.SHORT, easing = PlaygroundMotion.Emphasized))
        pulse.animateTo(1f, tween(PlaygroundMotion.MEDIUM, easing = PlaygroundMotion.Standard))
    }
    val screen = LocalConfiguration.current.screenWidthDp.dp
    Row(
        Modifier.widthIn(max = minOf(screen - 32.dp, 420.dp)).shadow(20.dp, CircleShape, ambientColor = Color.Black.copy(alpha = 0.35f))
            .clip(CircleShape).background(background).border(1.dp, border, CircleShape).padding(horizontal = 12.dp, vertical = 9.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        FaIcon(icon, null, size = 12.dp, tint = fg, modifier = Modifier.graphicsLayer { scaleX = pulse.value; scaleY = pulse.value })
        Text(message.ifBlank { status.defaultMessage() }, fontSize = 12.sp, lineHeight = 16.sp, color = fg, modifier = Modifier.weight(1f, fill = false))
        if (status != ConnectionStatus.ONLINE) Text("再試行", fontSize = 12.sp, fontWeight = FontWeight.Bold, color = fg,
            modifier = Modifier.clip(CircleShape).clickable(role = Role.Button, onClick = onRetry).padding(horizontal = 6.dp, vertical = 2.dp))
    }
}
// endregion

// region: Authentication screen
@Composable
internal fun AuthScreen(state: ChatState, model: ChatViewModel, onWeb: (String) -> Unit) {
    var signup by rememberSaveable { mutableStateOf(false) }
    var username by rememberSaveable { mutableStateOf("") }
    var password by rememberSaveable { mutableStateOf("") }
    var confirmation by rememberSaveable { mutableStateOf("") }
    var totp by rememberSaveable { mutableStateOf("") }
    val clipboard = LocalClipboardManager.current
    val colors = MaterialTheme.colorScheme
    LaunchedEffect(state.authTurnstileUrl) {
        state.authTurnstileUrl?.let { path -> onWeb(path) }
    }
    LazyColumn(
        Modifier.fillMaxSize(),
        contentPadding = PaddingValues(24.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        item {
            Surface(shape = RoundedCornerShape(28.dp), color = colors.primaryContainer, contentColor = colors.onPrimaryContainer) {
                Column(Modifier.fillMaxWidth().padding(28.dp)) {
                    Text("✦", fontSize = 48.sp, color = colors.primary)
                    Text("ひとつの場所で、\nいろいろなAIと。", style = MaterialTheme.typography.headlineLarge, fontWeight = FontWeight.Bold)
                    Spacer(Modifier.height(12.dp))
                    Text("アプリ内でアカウントを作成・ログインできます。", color = colors.onPrimaryContainer)
                }
            }
        }
        item { ServerOriginField(state, model) }
        item { LocalProfileEntry(state, model) }
        item {
            Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                if (state.authTwoFactorTransaction != null) {
                    Text("2段階認証", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.SemiBold)
                    Text("認証アプリに表示されたコードを入力してください。", color = colors.onSurfaceVariant)
                    OutlinedTextField(
                        value = totp, onValueChange = { totp = it }, label = { Text("認証コード") },
                        singleLine = true, modifier = Modifier.fillMaxWidth(), enabled = !state.authBusy,
                    )
                    state.authError?.let { Text(it, color = colors.error) }
                    Button(
                        onClick = { model.verifyTotp(totp) },
                        enabled = !state.authBusy && totp.trim().isNotEmpty(),
                        modifier = Modifier.fillMaxWidth().heightIn(min = 50.dp),
                    ) { Text(if (state.authBusy) "確認しています…" else "ログイン") }
                    if (state.auth2faMethod == "webauthn") {
                        OutlinedButton(
                            onClick = { model.beginWebauthnTwoFactor() },
                            enabled = !state.authBusy,
                            modifier = Modifier.fillMaxWidth().heightIn(min = 50.dp),
                        ) { Text("パスキーで認証") }
                    }
                } else {
                    Text(if (signup) "アカウントを作成" else "ログイン", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.SemiBold)
                    OutlinedTextField(
                        value = username, onValueChange = { username = it }, label = { Text("ユーザー名") },
                        singleLine = true, enabled = !state.authBusy,
                        modifier = Modifier.fillMaxWidth().semantics { contentType = if (signup) ContentType.NewUsername else ContentType.Username },
                    )
                    OutlinedTextField(
                        value = password, onValueChange = { password = it }, label = { Text("パスワード") },
                        singleLine = true, visualTransformation = PasswordVisualTransformation(),
                        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Password),
                        modifier = Modifier.fillMaxWidth().secretField(if (signup) ContentType.NewPassword else ContentType.Password),
                        enabled = !state.authBusy,
                    )
                    if (signup) {
                        OutlinedTextField(
                            value = confirmation, onValueChange = { confirmation = it }, label = { Text("パスワード（確認）") },
                            singleLine = true, visualTransformation = PasswordVisualTransformation(),
                            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Password),
                            modifier = Modifier.fillMaxWidth().secretField(ContentType.NewPassword),
                            enabled = !state.authBusy,
                        )
                    }
                    state.authError?.let { Text(it, color = colors.error) }
                    state.googleAuthDiagnostics?.let { diagnostics ->
                        Surface(
                            color = colors.surfaceVariant,
                            shape = RoundedCornerShape(12.dp),
                            modifier = Modifier.fillMaxWidth(),
                        ) {
                            Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                                Text("Googleログイン診断ログ", style = MaterialTheme.typography.titleSmall)
                                Text(
                                    "IDトークンやnonceは含まれていません。OAuth設定の確認に利用できます。",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = colors.onSurfaceVariant,
                                )
                                SelectionContainer {
                                    Text(diagnostics, style = MaterialTheme.typography.bodySmall)
                                }
                                OutlinedButton(
                                    onClick = {
                                        clipboard.setText(AnnotatedString(diagnostics))
                                        model.notify("診断ログをコピーしました。")
                                    },
                                    modifier = Modifier.fillMaxWidth(),
                                ) { Text("診断ログをコピー") }
                            }
                        }
                    }
                    Button(
                        onClick = {
                            if (signup && password != confirmation) model.notify("パスワードが一致しません。")
                            else if (signup) model.signup(username, password) else model.login(username, password)
                        },
                        enabled = !state.authBusy && username.isNotBlank() && password.isNotBlank(),
                        modifier = Modifier.fillMaxWidth().heightIn(min = 50.dp),
                    ) { Text(if (state.authBusy) "処理しています…" else if (signup) "アカウントを作成" else "ログイン") }
                    TextButton(onClick = { signup = !signup; confirmation = "" }) {
                        Text(if (signup) "すでにアカウントをお持ちですか？ログイン" else "アカウントを新規作成")
                    }
                    if (!signup) {
                        val server = state.serverInfo
                        if (server?.passkey != false) OutlinedButton(
                            onClick = { model.beginPasskeyLogin(username) },
                            enabled = !state.authBusy && username.isNotBlank(),
                            modifier = Modifier.fillMaxWidth().heightIn(min = 50.dp),
                        ) { Text("パスキーでログイン") }
                        if (server?.googleNative ?: ServerOrigin.isOfficial) Button(
                            onClick = model::beginGoogleLogin,
                            enabled = !state.authBusy,
                            modifier = Modifier.fillMaxWidth().heightIn(min = 50.dp),
                        ) { Text(if (state.authBusy) "Googleログインを処理しています…" else "Googleでログイン") }
                    }
                }
            }
        }
        item {
            HorizontalDivider()
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("ブラウザーでログイン", style = MaterialTheme.typography.titleMedium)
                if (state.serverInfo?.googleBrowser ?: ServerOrigin.isOfficial) {
                    OutlinedButton(onClick = { onWeb(model.browserLoginPath("google")) }, modifier = Modifier.fillMaxWidth()) { Text("Google（ブラウザー・旧方式）") }
                }
                if (state.serverInfo?.minashin ?: ServerOrigin.isOfficial) {
                    OutlinedButton(onClick = { onWeb(model.browserLoginPath("minashin")) }, modifier = Modifier.fillMaxWidth()) { Text("Minashinでログイン") }
                }
                TextButton(onClick = { onWeb("/") }, modifier = Modifier.fillMaxWidth()) { Text("Web版を開く") }
            }
        }
    }
}

// region: Local profile entry
/** Login screen "サーバーを使わずに始める": the no-account profile (ANDROID_ONLY.md). */
@Composable
private fun LocalProfileEntry(state: ChatState, model: ChatViewModel) {
    val colors = MaterialTheme.colorScheme
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        OutlinedButton(onClick = model::startLocalProfile, enabled = !state.authBusy,
            modifier = Modifier.fillMaxWidth().heightIn(min = 50.dp)) { Text("サーバーを使わずに始める") }
        Text("アカウントなしで、この端末だけで使います。APIキーとチャットは端末内に暗号化して保存し、AIの各社APIへ直接送信します。あとからサーバーにログインしてチャットを同期できます。",
            style = MaterialTheme.typography.bodySmall, color = colors.onSurfaceVariant)
    }
}
// endregion

// region: Server origin field
/** Login screen "接続先": the official server by default, or a self-hosted AI Playground server. */
@Composable
private fun ServerOriginField(state: ChatState, model: ChatViewModel) {
    val colors = MaterialTheme.colorScheme
    var editing by rememberSaveable { mutableStateOf(false) }
    var input by rememberSaveable(state.serverLabel) { mutableStateOf(state.serverLabel) }
    LaunchedEffect(state.serverLabel, state.serverChecking) { if (!state.serverChecking && state.serverError == null) editing = false }
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) {
                Text("接続先", style = MaterialTheme.typography.labelMedium, color = colors.onSurfaceVariant)
                Text(state.serverInfo?.name?.takeIf { it != state.serverLabel }?.let { "$it（${state.serverLabel}）" } ?: state.serverLabel,
                    style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.SemiBold)
            }
            TextButton(onClick = { editing = !editing }, enabled = !state.authBusy && !state.serverChecking) {
                Text(if (editing) "閉じる" else "変更")
            }
        }
        if (editing) {
            OutlinedTextField(
                value = input, onValueChange = { input = it }, label = { Text("サーバーのドメイン") },
                placeholder = { Text("ai.example.com") }, singleLine = true, enabled = !state.serverChecking,
                modifier = Modifier.fillMaxWidth(),
            )
            val recent = (listOf(originLabel(ServerOrigin.DEFAULT)) + state.savedServers).distinct()
            Row(Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                recent.forEach { label -> AssistChip(onClick = { input = label; model.selectServer(label) }, label = { Text(label) }) }
            }
            state.serverError?.let { Text(it, color = colors.error, style = MaterialTheme.typography.bodySmall) }
            Button(onClick = { model.selectServer(input) }, enabled = !state.serverChecking && input.isNotBlank(),
                modifier = Modifier.fillMaxWidth()) { Text(if (state.serverChecking) "確認しています…" else "この接続先を使う") }
            Text("本アプリに対応した AI Playground のサーバーだけに接続できます（https のみ）。", style = MaterialTheme.typography.bodySmall,
                color = colors.onSurfaceVariant)
        }
    }
}
// endregion
// endregion

// region: Initial setup screen
@Composable
internal fun SetupScreen(state: ChatState, model: ChatViewModel, onWeb: (String) -> Unit) {
    var selectedModel by rememberSaveable(state.setupDefaultModel) { mutableStateOf(state.setupDefaultModel) }
    var modelMenu by remember { mutableStateOf(false) }
    var openai by rememberSaveable { mutableStateOf("") }
    var gemini by rememberSaveable { mutableStateOf("") }
    var anthropic by rememberSaveable { mutableStateOf("") }
    var deepseek by rememberSaveable { mutableStateOf("") }
    var kimi by rememberSaveable { mutableStateOf("") }
    var mistral by rememberSaveable { mutableStateOf("") }
    var ideogram by rememberSaveable { mutableStateOf("") }
    var zai by rememberSaveable { mutableStateOf("") }
    var xai by rememberSaveable { mutableStateOf("") }
    var google by rememberSaveable { mutableStateOf("") }
    var googleProject by rememberSaveable { mutableStateOf("") }
    var vertexProject by rememberSaveable { mutableStateOf("") }
    var vertexLocation by rememberSaveable { mutableStateOf("global") }
    var vertexJson by rememberSaveable { mutableStateOf("") }
    var e2ee by rememberSaveable { mutableStateOf(false) }
    val zipPicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) model.importAccountZip(uri)
    }
    val colors = MaterialTheme.colorScheme
    if (state.setupImportSettingsChanges.isNotEmpty()) {
        NoticeAlertDialog(
            onDismissRequest = { if (!state.setupImportBusy) model.cancelSetupImportConfirmation() },
            // Cancelling here drops the uploaded archive, so only the buttons (or back) may do it.
            properties = androidx.compose.ui.window.DialogProperties(dismissOnClickOutside = false),
            title = { Text("設定の変更を確認") },
            text = {
                LazyColumn(verticalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.heightIn(max = 360.dp)) {
                    item {
                        Text("ZIPに含まれる設定で、現在の設定を上書きします。内容を確認してください。")
                    }
                    items(state.setupImportSettingsChanges) { change ->
                        ImportSettingChangeRow(change)
                    }
                }
            },
            confirmButton = {
                TextButton(
                    onClick = model::confirmSetupImportSettings,
                    enabled = !state.setupImportBusy,
                ) { Text(if (state.setupImportBusy) "適用中…" else "設定を上書きして続行") }
            },
            dismissButton = {
                TextButton(
                    onClick = model::cancelSetupImportConfirmation,
                    enabled = !state.setupImportBusy,
                ) { Text("取り消す") }
            },
        )
    }
    LazyColumn(
        Modifier.fillMaxSize(), contentPadding = PaddingValues(24.dp), verticalArrangement = Arrangement.spacedBy(14.dp),
    ) {
        item {
            Text("はじめに設定しましょう", style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.Bold)
            Spacer(Modifier.height(6.dp))
            Text("${state.account?.name ?: "アカウント"} の初回セットアップです。後からWebの設定でも変更できます。", color = colors.onSurfaceVariant)
        }
        item {
            Text("既定のモデル", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
            Box {
                OutlinedButton(onClick = { modelMenu = true }, modifier = Modifier.fillMaxWidth()) { Text(selectedModel) }
                DropdownMenu(expanded = modelMenu, onDismissRequest = { modelMenu = false }) {
                    state.setupModels.filter { it.selectable }.take(40).forEach { option ->
                        DropdownMenuItem(text = { Text(option.name) }, onClick = { selectedModel = option.id; modelMenu = false })
                    }
                }
            }
        }
        item { Text("APIキー（必要なプロバイダーだけ入力）", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold) }
        item { SetupSecretField("OpenAI", openai, { openai = it }) }
        item { SetupSecretField("Gemini", gemini, { gemini = it }) }
        item { SetupSecretField("Anthropic", anthropic, { anthropic = it }) }
        item { SetupSecretField("DeepSeek", deepseek, { deepseek = it }) }
        item { SetupSecretField("Kimi", kimi, { kimi = it }) }
        item { SetupSecretField("Mistral", mistral, { mistral = it }) }
        item { SetupSecretField("Ideogram", ideogram, { ideogram = it }) }
        item { SetupSecretField("Z.AI", zai, { zai = it }) }
        item { SetupSecretField("xAI", xai, { xai = it }) }
        item { SetupSecretField("Google API", google, { google = it }) }
        item { SetupSecretField("Google Cloudプロジェクト", googleProject, { googleProject = it }) }
        item { SetupSecretField("Vertex AIプロジェクト", vertexProject, { vertexProject = it }) }
        item { SetupSecretField("Vertex AIリージョン", vertexLocation, { vertexLocation = it }) }
        item {
            OutlinedTextField(
                vertexJson, { vertexJson = it }, label = { Text("Vertex AIサービスアカウントJSON（任意）") },
                minLines = 3, maxLines = 8, modifier = Modifier.fillMaxWidth(),
            )
        }
        item {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Checkbox(checked = e2ee, onCheckedChange = { e2ee = it })
                Column(Modifier.weight(1f)) {
                    Text("保存データを暗号化する")
                    Text("既存データがある場合はバックグラウンドで移行します。", style = MaterialTheme.typography.bodySmall, color = colors.onSurfaceVariant)
                }
            }
        }
        item {
            state.authError?.let { Text(it, color = colors.error) }
            Button(
                onClick = {
                    model.finishSetup(selectedModel, openai, gemini, anthropic, deepseek, kimi, mistral, ideogram, zai, xai,
                        google, googleProject, vertexProject, vertexLocation, vertexJson, e2ee)
                }, enabled = !state.authBusy, modifier = Modifier.fillMaxWidth().heightIn(min = 52.dp),
            ) { Text(if (state.authBusy) "保存しています…" else "セットアップを完了") }
        }
        item {
            Text("アカウントZIPのインポート（任意）", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
            Text("Web版で書き出したZIPから、履歴・設定・APIキーなどを取り込めます。", color = colors.onSurfaceVariant, style = MaterialTheme.typography.bodySmall)
            if (state.setupImportBusy) {
                Text(
                    "${state.setupImportName} をアップロード中… (${state.setupImportProgress}/${state.setupImportTotalChunks})",
                    style = MaterialTheme.typography.bodySmall,
                )
                val progress = if (state.setupImportTotalChunks > 0) {
                    state.setupImportProgress.toFloat() / state.setupImportTotalChunks
                } else 0f
                LinearProgressIndicator(progress = { progress.coerceIn(0f, 1f) }, modifier = Modifier.fillMaxWidth())
                TextButton(onClick = model::cancelSetupImport) { Text("インポートをキャンセル") }
            } else if (state.setupImportDone) {
                Text("インポートが完了しました。", color = colors.primary, style = MaterialTheme.typography.bodySmall)
            } else {
                Button(onClick = {
                    zipPicker.launch(arrayOf("application/zip", "application/x-zip-compressed", "application/octet-stream"))
                }, modifier = Modifier.fillMaxWidth()) { Text("ZIPを選択してインポート") }
            }
            state.setupImportError?.let { Text(it, color = colors.error, style = MaterialTheme.typography.bodySmall) }
        }
        item {
            TextButton(onClick = { onWeb("/setup") }, modifier = Modifier.fillMaxWidth()) {
                Text("Webのセットアップ画面を開く")
            }
        }
    }
}
// endregion

// region: Setup helpers
@Composable
private fun ImportSettingChangeRow(change: ImportSettingChange) {
    val label = mapOf(
        "default_model" to "既定のモデル",
        "theme_color" to "テーマ色",
        "system_prompt" to "システムプロンプト",
        "system_prompt_enabled" to "システムプロンプトの有効化",
        "light_mode_enabled" to "ライトモード",
        "liquid_glass_enabled" to "Liquid Glass",
        "gemini_backend" to "Geminiバックエンド",
        "gemini_vertex_location" to "Vertex AIリージョン",
    )[change.field] ?: change.field
    Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
        Text(label, fontWeight = FontWeight.SemiBold)
        Text("現在: ${shortImportValue(change.current)}", style = MaterialTheme.typography.bodySmall)
        Text("取り込み後: ${shortImportValue(change.incoming)}", style = MaterialTheme.typography.bodySmall)
    }
}

private fun shortImportValue(value: String): String {
    val normalized = value.replace("\n", " ").trim()
    return if (normalized.length > 180) normalized.take(180) + "…" else normalized
}

@Composable
private fun SetupSecretField(label: String, value: String, onValueChange: (String) -> Unit) {
    OutlinedTextField(
        value = value, onValueChange = onValueChange, label = { Text(label) },
        singleLine = true, visualTransformation = PasswordVisualTransformation(),
        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Password), modifier = Modifier.fillMaxWidth().secretField(),
    )
}
// endregion

