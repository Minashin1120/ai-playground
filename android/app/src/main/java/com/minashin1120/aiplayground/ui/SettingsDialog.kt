package com.minashin1120.aiplayground.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.minashin1120.aiplayground.AppUpdatePhase
import com.minashin1120.aiplayground.AppUpdateUiState
import com.minashin1120.aiplayground.BuildConfig
import com.minashin1120.aiplayground.ChatState
import com.minashin1120.aiplayground.ChatViewModel
import com.minashin1120.aiplayground.data.CompressionSettings
import org.json.JSONObject

private val SETTINGS_TABS = listOf(
    "一般", "APIキー", "プロンプト", "表示", "データ", "アカウント",
    "セキュリティ", "2要素認証", "フィードバック", "MCP", "画像圧縮", "セッション",
)
private val THEME_PRESETS = listOf("#0DD4BF", "#38BDF8", "#A855F7", "#F97316", "#22C55E", "#FF00BB")
private val THINKING_LEVELS = listOf("minimal", "low", "medium", "high")
private val EFFORTS = listOf("none", "low", "medium", "high", "xhigh", "max")
private val STT_MODELS = listOf(
    "gpt-transcribe", "gpt-4o-mini-transcribe", "gpt-4o-transcribe", "gpt-4o-transcribe-diarize", "whisper-1",
)

@OptIn(ExperimentalLayoutApi::class)
@Composable
fun SettingsDialog(
    state: ChatState,
    model: ChatViewModel,
    onDismiss: () -> Unit,
    onLogout: () -> Unit,
    onWeb: (String) -> Unit,
    appUpdate: AppUpdateUiState = AppUpdateUiState(),
    onCheckForUpdate: () -> Unit = {},
) {
    val prefs = state.preferences
    var search by remember { mutableStateOf("") }
    var tab by remember { mutableStateOf("一般") }
    var defaultModel by remember(prefs?.defaultModel) { mutableStateOf(prefs?.defaultModel.orEmpty()) }
    var visionModel by remember(prefs?.defaultVisionModel) { mutableStateOf(prefs?.defaultVisionModel.orEmpty()) }
    var thinking by remember(prefs?.defaultEnableThinking) { mutableStateOf(prefs?.defaultEnableThinking ?: false) }
    var searchDefault by remember(prefs?.defaultEnableSearch) { mutableStateOf(prefs?.defaultEnableSearch ?: false) }
    var urlContext by remember(prefs?.defaultEnableUrlContext) { mutableStateOf(prefs?.defaultEnableUrlContext ?: false) }
    var maps by remember(prefs?.defaultEnableMaps) { mutableStateOf(prefs?.defaultEnableMaps ?: false) }
    var python by remember(prefs?.defaultEnablePython) { mutableStateOf(prefs?.defaultEnablePython ?: false) }
    var fileCreation by remember(prefs?.defaultEnableFileCreation) { mutableStateOf(prefs?.defaultEnableFileCreation ?: true) }
    var systemPromptDefault by remember(prefs?.defaultEnableSystemPrompt) { mutableStateOf(prefs?.defaultEnableSystemPrompt ?: false) }
    var mcp by remember(prefs?.defaultEnableMcp) { mutableStateOf(prefs?.defaultEnableMcp ?: true) }
    var thinkingLevel by remember(prefs?.defaultThinkingLevel) { mutableStateOf(prefs?.defaultThinkingLevel ?: "high") }
    var thinkingBudget by remember(prefs?.defaultThinkingBudget) { mutableStateOf((prefs?.defaultThinkingBudget ?: 4096).toString()) }
    var reasoningEffort by remember(prefs?.defaultReasoningEffort) { mutableStateOf(prefs?.defaultReasoningEffort ?: "medium") }
    var safetySetting by remember(prefs?.defaultSafetySetting) { mutableStateOf(prefs?.defaultSafetySetting ?: "default") }
    var enterToSend by remember(prefs?.enterToSend) { mutableStateOf(prefs?.enterToSend ?: false) }
    var lightMode by remember(prefs?.lightModeEnabled) { mutableStateOf(prefs?.lightModeEnabled ?: false) }
    var liquidGlass by remember(prefs?.liquidGlassEnabled) { mutableStateOf(prefs?.liquidGlassEnabled ?: false) }
    var themeColor by remember(prefs?.themeColor) { mutableStateOf(prefs?.themeColor.orEmpty()) }
    var autoSearch by remember(prefs?.autoSearchOnLinks) { mutableStateOf(prefs?.autoSearchOnLinks ?: true) }
    var timeout by remember(prefs?.tempChatTimeoutSeconds) { mutableStateOf((prefs?.tempChatTimeoutSeconds ?: 90).toString()) }
    var useLast by remember(prefs?.useLastChatSettings) { mutableStateOf(prefs?.useLastChatSettings ?: false) }
    var voiceStudio by remember(prefs?.voiceStudioUi) { mutableStateOf(prefs?.voiceStudioUi ?: true) }
    var promptBar by remember(prefs?.effectivePromptBarMode) { mutableStateOf(prefs?.effectivePromptBarMode ?: "normal") }
    var micMode by remember(prefs?.micTranscribeMode) { mutableStateOf(prefs?.micTranscribeMode ?: "stt_api") }
    var sttModel by remember(prefs?.sttModel) { mutableStateOf(prefs?.sttModel ?: "gpt-4o-mini-transcribe") }
    var userPrompt by remember(prefs?.systemPrompt) { mutableStateOf(prefs?.systemPrompt.orEmpty()) }
    var userPromptEnabled by remember(prefs?.systemPromptEnabled) { mutableStateOf(prefs?.systemPromptEnabled ?: true) }
    var applyGlobal by remember(prefs?.applyGlobalSystemPrompt) { mutableStateOf(prefs?.applyGlobalSystemPrompt ?: true) }
    var applyNotices by remember(prefs?.applyAutoSystemPromptNotices) { mutableStateOf(prefs?.applyAutoSystemPromptNotices ?: true) }
    var richPasteCustom by remember(prefs?.richPastePromptUseCustomDefault) { mutableStateOf(prefs?.richPastePromptUseCustomDefault ?: false) }
    var richPastePrompt by remember(prefs?.richPastePromptDefault) { mutableStateOf(prefs?.richPastePromptDefault.orEmpty()) }
    var skip2faGoogle by remember(prefs?.skip2faOnGoogleLogin) { mutableStateOf(prefs?.skip2faOnGoogleLogin ?: false) }
    var default2fa by remember(prefs?.default2faMethod) { mutableStateOf(prefs?.default2faMethod ?: "totp") }
    var compressionEnabled by remember(state.compression) { mutableStateOf(state.compression.enabled) }
    var maxSizeMB by remember(state.compression) { mutableStateOf(state.compression.maxSizeMB.toString()) }
    var maxDim by remember(state.compression) { mutableStateOf(state.compression.maxDimension.toString()) }
    var formatOnly by remember(state.compression) { mutableStateOf(state.compression.formatOnly) }
    var outputType by remember(state.compression) { mutableStateOf(state.compression.outputType) }
    var modelPicker by remember { mutableStateOf(false) }
    var visionPicker by remember { mutableStateOf(false) }
    var confirmLogout by remember { mutableStateOf(false) }
    var feedbackTitle by remember { mutableStateOf("") }
    var feedbackMessage by remember { mutableStateOf("") }
    LaunchedEffect(Unit) {
        model.loadPreferences()
        model.loadStorageUsage()
        model.loadFeedback()
        model.loadMcpServers()
    }
    val query = search.trim()
    fun matches(vararg haystacks: String): Boolean =
        query.isBlank() || haystacks.any { it.contains(query, ignoreCase = true) }
    val visibleTabs = SETTINGS_TABS.filter { tabName ->
        query.isBlank() || tabName.contains(query, ignoreCase = true) || when (tabName) {
            "一般" -> matches("Enter", "モデル", "Thinking", "Search", "URLs", "Maps", "Python", "File", "SysPrompt", "MCP", "音声", "STT", "一時チャット", "プロンプトバー", "アプリ更新", "更新")
            "APIキー" -> matches("API", "キー", "OpenAI", "Gemini", "xAI")
            "プロンプト" -> matches("システムプロンプト", "全体", "ユーザー")
            "表示" -> matches("テーマ", "ライト", "Liquid", "色")
            "データ" -> matches("ストレージ", "キャッシュ", "エクスポート", "リッチ")
            "アカウント" -> matches("ユーザー名", "パスワード", "Google", "Minashin")
            "セキュリティ" -> matches("E2EE", "暗号化", "セッション", "削除")
            "2要素認証" -> matches("2FA", "TOTP", "パスキー")
            "フィードバック" -> matches("バグ", "要望")
            "MCP" -> matches("MCP", "サーバー", "Gmail")
            "画像圧縮" -> matches("圧縮", "JPEG", "WebP")
            "セッション" -> matches("端末", "ログアウト")
            else -> false
        }
    }
    PlaygroundDialog(
        onDismissRequest = onDismiss,
        title = { Text("設定") },
        text = {
            Column(Modifier.heightIn(max = 520.dp).verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                if (state.prefsBusy) LinearProgressIndicator(Modifier.fillMaxWidth())
                OutlinedTextField(search, { search = it }, singleLine = true, label = { Text("設定を検索") },
                    modifier = Modifier.fillMaxWidth())
                Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    visibleTabs.forEach { name -> FilterChip(tab == name, { tab = name }, { Text(name) }) }
                }
                if (visibleTabs.isEmpty()) Text("該当する設定はありません。", style = MaterialTheme.typography.bodySmall)
                else if (tab !in visibleTabs) tab = visibleTabs.first()
                if (tab == "一般" && tab in visibleTabs) {
                    Text(state.account?.name.orEmpty(), style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    Text("送信設定", fontWeight = FontWeight.SemiBold)
                    CheckboxRow("Enterで送信（改行はShift+Enter）", enterToSend) { enterToSend = it }
                    Text("プロンプトバー表示", fontWeight = FontWeight.SemiBold)
                    listOf("normal" to "通常表示", "compact" to "コンパクト表示", "minimal" to "ミニマル表示").forEach { (value, label) ->
                        FilterChip(promptBar == value, { promptBar = value }, { Text(label) })
                    }
                    CheckboxRow("音声系モデルで音声スタジオを使う", voiceStudio) { voiceStudio = it }
                    Text("チャット既定値", fontWeight = FontWeight.SemiBold)
                    TextButton(onClick = { modelPicker = true }, modifier = Modifier.fillMaxWidth()) {
                        Text("既定モデル: ${state.account?.models?.firstOrNull { it.id == defaultModel }?.name ?: defaultModel.ifBlank { "未設定" }} ▾")
                    }
                    TextButton(onClick = { visionPicker = true }, modifier = Modifier.fillMaxWidth()) {
                        Text("Vision Model: ${state.account?.models?.firstOrNull { it.id == visionModel }?.name ?: visionModel.ifBlank { "未設定" }} ▾")
                    }
                    CheckboxRow("前回の設定を保存して継続する", useLast) { useLast = it }
                    CheckboxRow("既定でThinkingを使う", thinking) { thinking = it }
                    ChoiceRow("Thinking level", thinkingLevel, THINKING_LEVELS) { thinkingLevel = it }
                    OutlinedTextField(thinkingBudget, { thinkingBudget = it.filter(Char::isDigit).take(5) }, singleLine = true,
                        label = { Text("Thinking Budget") }, modifier = Modifier.fillMaxWidth())
                    ChoiceRow("Reasoning effort", reasoningEffort, EFFORTS) { reasoningEffort = it }
                    ChoiceRow("Safety", safetySetting, listOf("default", "none")) { safetySetting = it }
                    CheckboxRow("既定でWeb検索を使う", searchDefault) { searchDefault = it }
                    CheckboxRow("既定でURLsを使う", urlContext) { urlContext = it }
                    CheckboxRow("既定でMapsを使う", maps) { maps = it }
                    CheckboxRow("既定でPythonを使う", python) { python = it }
                    CheckboxRow("既定でFileを使う", fileCreation) { fileCreation = it }
                    CheckboxRow("既定でSysPromptを使う", systemPromptDefault) { systemPromptDefault = it }
                    CheckboxRow("既定でMCPを使う", mcp) { mcp = it }
                    CheckboxRow("リンクのX投稿を自動検索", autoSearch) { autoSearch = it }
                    OutlinedTextField(timeout, { timeout = it.filter { c -> c.isDigit() }.take(6) }, singleLine = true,
                        label = { Text("一時チャットの自動削除（秒）") }, modifier = Modifier.fillMaxWidth())
                    Text("音声設定", fontWeight = FontWeight.SemiBold)
                    ChoiceRow("マイク文字起こし", micMode, listOf("stt_api", "llm")) { micMode = it }
                    ChoiceRow("STTモデル", sttModel, STT_MODELS) { sttModel = it }
                    Text("STT APIはWebの録音経路で使います。この端末のマイクボタンはOSの音声認識です。",
                        style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    Text("アプリ更新", fontWeight = FontWeight.SemiBold)
                    Text("現在のバージョン: ${BuildConfig.VERSION_NAME}", style = MaterialTheme.typography.bodySmall)
                    when (appUpdate.phase) {
                        AppUpdatePhase.Checking -> Text("更新を確認中…", style = MaterialTheme.typography.bodySmall)
                        AppUpdatePhase.UpToDate -> Text("最新のAndroid版を使用しています。", style = MaterialTheme.typography.bodySmall)
                        AppUpdatePhase.Available -> appUpdate.update?.let { update ->
                            Text("Android版 ${update.versionName} が利用できます。", style = MaterialTheme.typography.bodySmall)
                        }
                        AppUpdatePhase.Error -> Text(appUpdate.errorMessage ?: "更新を確認できませんでした。", style = MaterialTheme.typography.bodySmall)
                        else -> Unit
                    }
                    TextButton(
                        onClick = onCheckForUpdate,
                        enabled = appUpdate.phase != AppUpdatePhase.Checking &&
                            appUpdate.phase != AppUpdatePhase.Downloading &&
                            appUpdate.phase != AppUpdatePhase.Ready &&
                            appUpdate.phase != AppUpdatePhase.AwaitingInstallPermission &&
                            appUpdate.phase != AppUpdatePhase.Installing,
                        modifier = Modifier.fillMaxWidth(),
                    ) { Text(if (appUpdate.phase == AppUpdatePhase.Checking) "確認中…" else "更新を確認") }
                }
                if (tab == "APIキー" && tab in visibleTabs) {
                    Text("APIキーは端末へ渡しません。認証済みブラウザーの設定画面で登録・変更します。",
                        style = MaterialTheme.typography.bodySmall)
                    TextButton(onClick = { onWeb("/settings") }, modifier = Modifier.fillMaxWidth()) { Text("WebでAPIキーを開く") }
                }
                if (tab == "プロンプト" && tab in visibleTabs) {
                    Text("全体システムプロンプト（参照のみ）", fontWeight = FontWeight.SemiBold)
                    OutlinedTextField(prefs?.globalSystemPrompt.orEmpty(), {}, readOnly = true, minLines = 3, maxLines = 8,
                        modifier = Modifier.fillMaxWidth())
                    CheckboxRow("全体システムプロンプトを適用", applyGlobal) { applyGlobal = it }
                    CheckboxRow("自動システム通知を適用", applyNotices) { applyNotices = it }
                    CheckboxRow("ユーザーシステムプロンプトを有効", userPromptEnabled) { userPromptEnabled = it }
                    OutlinedTextField(userPrompt, { userPrompt = it.take(100_000) }, minLines = 4, maxLines = 10,
                        label = { Text("ユーザーシステムプロンプト") }, modifier = Modifier.fillMaxWidth())
                    TextButton(onClick = { userPrompt = ""; userPromptEnabled = false }) { Text("リセット（空にして無効化）") }
                }
                if (tab == "表示" && tab in visibleTabs) {
                    Text("テーマ", fontWeight = FontWeight.SemiBold)
                    OutlinedTextField(themeColor, { themeColor = it.take(7) }, singleLine = true,
                        label = { Text("テーマカラー（HEX）") }, placeholder = { Text("#0DD4BF") }, modifier = Modifier.fillMaxWidth())
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
                        THEME_PRESETS.forEach { color ->
                            val parsed = runCatching { Color(android.graphics.Color.parseColor(color)) }.getOrNull()
                            if (parsed != null) Box(Modifier.size(28.dp).clip(CircleShape).background(parsed).clickable { themeColor = color })
                        }
                    }
                    TextButton(onClick = { themeColor = "" }) { Text("テーマをリセット") }
                    CheckboxRow("手動ライトモード", lightMode) { lightMode = it }
                    CheckboxRow("Liquid Glassモード", liquidGlass) { liquidGlass = it }
                }
                if (tab == "データ" && tab in visibleTabs) {
                    Text("ストレージ", fontWeight = FontWeight.SemiBold)
                    val storage = state.storage
                    if (storage == null) Text("読み込み中…", style = MaterialTheme.typography.bodySmall)
                    else {
                        Text(if (storage.unlimited) "使用量 ${storage.usedMb} MB（上限なし）"
                            else "使用量 ${storage.usedMb} / ${storage.limitMb} MB",
                            style = MaterialTheme.typography.bodySmall)
                        if (!storage.unlimited && storage.limitBytes > 0) {
                            LinearProgressIndicator(
                                progress = { (storage.usedBytes.toFloat() / storage.limitBytes).coerceIn(0f, 1f) },
                                modifier = Modifier.fillMaxWidth(),
                            )
                        }
                    }
                    TextButton(onClick = { model.loadStorageUsage() }) { Text("使用量を更新") }
                    Text("サイトキャッシュとアカウントZIPの移行はブラウザー専用です。", style = MaterialTheme.typography.bodySmall)
                    TextButton(onClick = { onWeb("/settings") }) { Text("Webでデータ移行を開く") }
                    CheckboxRow("リッチ貼り付けのカスタム既定プロンプトを使う", richPasteCustom) { richPasteCustom = it }
                    if (richPasteCustom) OutlinedTextField(richPastePrompt, { richPastePrompt = it.take(20_000) }, minLines = 3, maxLines = 8,
                        label = { Text("リッチ貼り付けプロンプト") }, modifier = Modifier.fillMaxWidth())
                }
                if (tab == "アカウント" && tab in visibleTabs) {
                    Text("ユーザー名: ${prefs?.username.orEmpty()}", style = MaterialTheme.typography.bodyMedium)
                    Text("Google: ${prefs?.googleEmail?.ifBlank { "未連携" } ?: "未連携"}", style = MaterialTheme.typography.bodySmall)
                    Text("Minashin: ${prefs?.minashinEmail?.ifBlank { "未連携" } ?: "未連携"}", style = MaterialTheme.typography.bodySmall)
                    Text("ユーザー名・パスワード・SSO連携の変更はWeb設定で行います。", style = MaterialTheme.typography.bodySmall)
                    TextButton(onClick = { onWeb("/settings") }) { Text("Webでアカウント設定を開く") }
                }
                if (tab == "セキュリティ" && tab in visibleTabs) {
                    Text("暗号化: ${if (prefs?.e2eeEnabled == true) "有効（サーバー管理鍵）" else "無効"}", style = MaterialTheme.typography.bodySmall)
                    Text("E2EEの切替、他端末のセッション失効、アカウント削除はWeb設定で行います。", style = MaterialTheme.typography.bodySmall)
                    TextButton(onClick = { onWeb("/settings") }) { Text("Webでセキュリティ設定を開く") }
                }
                if (tab == "2要素認証" && tab in visibleTabs) {
                    Text("状態: ${if (prefs?.twoFactorEnabled == true) "有効" else "無効"} / TOTP: ${if (prefs?.hasTotp == true) "登録済" else "未登録"} / パスキー: ${if (prefs?.hasWebauthn == true) "登録済" else "未登録"}",
                        style = MaterialTheme.typography.bodySmall)
                    CheckboxRow("Googleログイン時に2FAをスキップ", skip2faGoogle) { skip2faGoogle = it }
                    ChoiceRow("既定の2要素認証方式", default2fa, listOf("totp", "webauthn")) { default2fa = it }
                    Text("TOTP・パスキーの登録と無効化はWeb設定で行います。", style = MaterialTheme.typography.bodySmall)
                    TextButton(onClick = { onWeb("/settings") }) { Text("Webで2FA設定を開く") }
                }
                if (tab == "フィードバック" && tab in visibleTabs) {
                    OutlinedTextField(feedbackTitle, { feedbackTitle = it.take(200) }, singleLine = true,
                        label = { Text("タイトル（任意）") }, modifier = Modifier.fillMaxWidth())
                    OutlinedTextField(feedbackMessage, { feedbackMessage = it.take(100_000) }, minLines = 4, maxLines = 8,
                        label = { Text("バグ報告・要望") }, modifier = Modifier.fillMaxWidth())
                    TextButton(onClick = { model.submitFeedback(feedbackTitle, feedbackMessage); feedbackTitle = ""; feedbackMessage = "" },
                        enabled = feedbackMessage.isNotBlank() && !state.feedbackBusy) { Text("送信") }
                    Text("あなたのフィードバック", fontWeight = FontWeight.SemiBold)
                    if (state.feedbackItems.isEmpty()) Text("まだありません。", style = MaterialTheme.typography.bodySmall)
                    state.feedbackItems.take(20).forEach { item ->
                        Surface(shape = MaterialTheme.shapes.small, color = MaterialTheme.colorScheme.surfaceContainerHigh) {
                            Column(Modifier.fillMaxWidth().padding(10.dp), verticalArrangement = Arrangement.spacedBy(2.dp)) {
                                Text(item.title.ifBlank { "（無題）" }, fontWeight = FontWeight.SemiBold)
                                Text(item.message.take(400), style = MaterialTheme.typography.bodySmall)
                                Text("${item.status} · ${item.createdAt}", style = MaterialTheme.typography.labelSmall)
                                if (item.adminReply.isNotBlank()) Text("返信: ${item.adminReply.take(400)}", style = MaterialTheme.typography.bodySmall)
                            }
                        }
                    }
                }
                if (tab == "MCP" && tab in visibleTabs) {
                    Text("接続中のサーバーをこの端末で有効／無効にできます。OAuthやBearerの秘密設定はWebで行います。",
                        style = MaterialTheme.typography.bodySmall)
                    if (state.mcpBusy) LinearProgressIndicator(Modifier.fillMaxWidth())
                    if (state.mcpServers.isEmpty() && !state.mcpBusy) Text("登録済みサーバーはありません。", style = MaterialTheme.typography.bodySmall)
                    state.mcpServers.forEach { server ->
                        Surface(shape = MaterialTheme.shapes.small, color = MaterialTheme.colorScheme.surfaceContainerHigh) {
                            Column(Modifier.fillMaxWidth().padding(10.dp)) {
                                CheckboxRow("${server.name}（${server.toolCount}ツール）", server.enabled) { model.setMcpServerEnabled(server, it) }
                                Text("${server.connectionState} / ${server.authStatus}", style = MaterialTheme.typography.labelSmall)
                            }
                        }
                    }
                    TextButton(onClick = { onWeb("/settings") }) { Text("WebでMCP秘密設定を開く") }
                }
                if (tab == "画像圧縮" && tab in visibleTabs) {
                    Text("画像の圧縮", fontWeight = FontWeight.SemiBold)
                    CheckboxRow("画像を圧縮して送信", compressionEnabled) { compressionEnabled = it }
                    if (compressionEnabled) {
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            OutlinedTextField(maxSizeMB, { maxSizeMB = it.filter { c -> c.isDigit() || c == '.' }.take(5) },
                                singleLine = true, label = { Text("最大サイズ (MB)") }, modifier = Modifier.weight(1f))
                            OutlinedTextField(maxDim, { maxDim = it.filter { c -> c.isDigit() }.take(4) },
                                singleLine = true, label = { Text("最大辺 (px)") }, modifier = Modifier.weight(1f))
                        }
                        CheckboxRow("形式のみ変換（サイズ・寸法は変更しない）", formatOnly) { formatOnly = it }
                        FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                            listOf("original" to "元の形式", "image/jpeg" to "JPEG", "image/png" to "PNG", "image/webp" to "WebP").forEach { (value, label) ->
                                FilterChip(outputType == value, { outputType = value }, { Text(label) })
                            }
                        }
                    }
                }
                if (tab == "セッション" && tab in visibleTabs) {
                    Text("この端末のセッション", fontWeight = FontWeight.SemiBold)
                    Text("端末: ${prefs?.deviceName?.ifBlank { "このAndroid端末" } ?: "このAndroid端末"}", style = MaterialTheme.typography.labelSmall)
                    if (prefs != null) {
                        Text("連携日時: ${prefs.sessionCreatedAt}", style = MaterialTheme.typography.labelSmall)
                        Text("有効期限: ${prefs.sessionExpiresAt}", style = MaterialTheme.typography.labelSmall)
                    }
                    TextButton(onClick = { confirmLogout = true }) { Text("この端末の連携を取り消す") }
                }
            }
        },
        confirmButton = {
            TextButton(
                onClick = {
                    model.saveCompressionSettings(CompressionSettings(
                        enabled = compressionEnabled,
                        maxSizeMB = maxSizeMB.toFloatOrNull()?.coerceIn(0.05f, 50f) ?: 1.0f,
                        maxDimension = maxDim.toIntOrNull()?.coerceIn(256, 8192) ?: 1920,
                        outputType = outputType,
                        formatOnly = formatOnly,
                    ))
                    model.savePreferences(JSONObject()
                        .put("default_model", defaultModel)
                        .put("default_vision_model", visionModel)
                        .put("default_enable_thinking", thinking)
                        .put("default_enable_search", searchDefault)
                        .put("default_enable_url_context", urlContext)
                        .put("default_enable_maps", maps)
                        .put("default_enable_python", python)
                        .put("default_enable_file_creation", fileCreation)
                        .put("default_enable_system_prompt", systemPromptDefault)
                        .put("default_enable_mcp", mcp)
                        .put("default_thinking_level", thinkingLevel)
                        .put("default_thinking_budget", thinkingBudget.toIntOrNull()?.coerceIn(0, 32768) ?: 4096)
                        .put("default_reasoning_effort", reasoningEffort)
                        .put("default_safety_setting", safetySetting)
                        .put("enter_to_send", enterToSend)
                        .put("light_mode_enabled", lightMode)
                        .put("liquid_glass_enabled", liquidGlass)
                        .put("auto_search_on_links", autoSearch)
                        .put("temp_chat_timeout_seconds", timeout.toIntOrNull() ?: 90)
                        .put("theme_color", themeColor.trim())
                        .put("use_last_chat_settings", useLast)
                        .put("voice_studio_ui", voiceStudio)
                        .put("prompt_bar_mode", promptBar)
                        .put("mic_transcribe_mode", micMode)
                        .put("stt_model", sttModel)
                        .put("system_prompt", userPrompt)
                        .put("system_prompt_enabled", userPromptEnabled)
                        .put("apply_global_system_prompt", applyGlobal)
                        .put("apply_auto_system_prompt_notices", applyNotices)
                        .put("rich_paste_prompt_use_custom_default", richPasteCustom)
                        .put("rich_paste_prompt_default", richPastePrompt)
                        .put("skip_2fa_on_google_login", skip2faGoogle)
                        .put("default_2fa_method", default2fa))
                },
                enabled = !state.prefsBusy,
            ) { Text("保存") }
        },
        dismissButton = { TextButton(onClick = onDismiss) { Text("閉じる") } },
    )
    if (modelPicker) ModelPicker(state, onDismiss = { modelPicker = false }, onSelect = { defaultModel = it; modelPicker = false }, selectedId = defaultModel)
    if (visionPicker) ModelPicker(state, onDismiss = { visionPicker = false }, onSelect = { visionModel = it; visionPicker = false }, selectedId = visionModel)
    if (confirmLogout) AlertDialog(onDismissRequest = { confirmLogout = false }, title = { Text("この端末からログアウト") },
        text = { Text("このAndroid端末の連携を取り消します。Webや他の端末のログインは継続します。") },
        confirmButton = { TextButton(onClick = { confirmLogout = false; onLogout() }) { Text("ログアウト") } },
        dismissButton = { TextButton(onClick = { confirmLogout = false }) { Text("キャンセル") } })
}

@Composable
private fun CheckboxRow(label: String, checked: Boolean, onChange: (Boolean) -> Unit) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Checkbox(checked, onChange)
        Text(label, modifier = Modifier.weight(1f))
    }
}

@Composable
private fun ChoiceRow(label: String, value: String, choices: List<String>, onChange: (String) -> Unit) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Text("$label: $value", modifier = Modifier.weight(1f), style = MaterialTheme.typography.bodySmall)
        TextButton(onClick = { onChange(choices[(choices.indexOf(value).let { if (it < 0) 0 else it } + 1) % choices.size]) }) { Text("変更") }
    }
}
