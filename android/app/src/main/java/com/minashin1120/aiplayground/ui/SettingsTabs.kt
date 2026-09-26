package com.minashin1120.aiplayground.ui

import androidx.annotation.DrawableRes
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.minashin1120.aiplayground.ChatState
import com.minashin1120.aiplayground.R
import com.minashin1120.aiplayground.data.ModelInfo

/** The ten Web settings tabs (`#settings-tabs`). A null icon means the Web subset has no glyph for it. */
internal enum class SettingsTab(val id: String, val label: String, @DrawableRes val icon: Int?) {
    General("general", "一般", R.drawable.fa_solid_sliders_h),
    Api("api", "APIキー", R.drawable.fa_solid_key),
    Prompt("prompt", "プロンプト", R.drawable.fa_solid_comment_dots),
    Display("display", "表示", R.drawable.fa_solid_palette),
    Data("data", "データ", R.drawable.fa_solid_database),
    Account("account", "アカウント", R.drawable.fa_solid_user_shield),
    Security("security", "セキュリティ", R.drawable.fa_solid_shield_alt),
    TwoFactor("2fa", "2要素認証", R.drawable.fa_solid_lock),
    Feedback("feedback", "フィードバック", R.drawable.fa_solid_bug),
    // `fa-plug` is not in the Web icon subset, so the Web tab shows no glyph either.
    Mcp("mcp", "MCP", null),
}

/**
 * One `.settings-card`. [search] is the card's text content, which the Web search matches
 * (`child.textContent.includes(q)`); [title] is its heading in the result list.
 */
internal class SettingsCardSpec(
    val tab: SettingsTab,
    val key: String,
    val title: String?,
    val search: String,
    val danger: Boolean = false,
    @DrawableRes val titleIcon: Int? = null,
    val content: @Composable ColumnScope.() -> Unit,
)

private val STT_OPTIONS = webOptions(
    "gpt-transcribe" to "gpt-transcribe（推奨・高精度）",
    "gpt-4o-mini-transcribe" to "gpt-4o-mini-transcribe",
    "gpt-4o-transcribe" to "gpt-4o-transcribe",
    "gpt-4o-transcribe-diarize" to "gpt-4o-transcribe-diarize",
    "whisper-1" to "whisper-1",
    "grok-voice-transcribe-2.0" to "grok-voice-transcribe-2.0（xAI）",
    "grok-voice-transcribe-1.0" to "grok-voice-transcribe-1.0（xAI）",
)
private val THINKING_OPTIONS = webOptions("minimal" to "Min", "low" to "Low", "medium" to "Mid", "high" to "High")
private val EFFORT_OPTIONS = webOptions("none" to "None", "low" to "Low", "medium" to "Med", "high" to "High", "xhigh" to "XHigh", "max" to "Max")
private val SAFETY_OPTIONS = webOptions("default" to "Default", "none" to "None")
internal val THEME_PRESETS = listOf("#0dd4bf", "#38bdf8", "#a855f7", "#f97316", "#22c55e", "#ff00bb")

/** `populateDefaultModelOptions`: every model grouped by its catalog category. */
internal fun defaultModelOptions(models: List<ModelInfo>): List<WebOption> =
    models.filter { it.selectable && !it.deprecated }.map { WebOption(it.id, it.name, it.category.ifBlank { null }) }

/** `populateDefaultVisionModelOptions`: Gemini / GPT-4o / Claude / Grok 3 models, marked with ★. */
internal fun visionModelOptions(models: List<ModelInfo>): List<WebOption> =
    models.filter { it.selectable && !it.deprecated }.filter { m ->
        val id = m.id.lowercase()
        id.startsWith("gemini-") || id.startsWith("gpt-4o") || id.startsWith("claude-") || id.startsWith("grok-3")
    }.map { WebOption(it.id, "${it.name} ★", it.category.ifBlank { null }) }

internal fun generalCards(
    state: ChatState,
    form: SettingsForm,
    localPythonDialog: Boolean,
    onLocalPythonDialog: (Boolean) -> Unit,
    notify: (String) -> Unit,
): List<SettingsCardSpec> {
    val models = state.account?.models.orEmpty()
    return listOf(
        SettingsCardSpec(SettingsTab.General, "send", "送信設定", "送信設定 Enterで送信（改行はShift+Enter）") {
            SettingsCheck("Enterで送信（改行はShift+Enter）", form.enterToSend, { form.enterToSend = it })
        },
        SettingsCardSpec(SettingsTab.General, "prompt-bar", "プロンプトバー表示",
            "プロンプトバー表示 通常表示 コンパクト表示（モデル選択のみ表示） 詳細設定を折りたたみ、下部バーをモデル選択中心にします。 ミニマル表示（送信・プラスのみ） 下部バーを送信ボタンとプラスボタンだけにし、モデル選択を画面上部へ移します。") {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                SettingsRadio("通常表示", form.promptBarMode == "normal", { form.promptBarMode = "normal" })
                SettingsRadio("コンパクト表示（モデル選択のみ表示）", form.promptBarMode == "compact", { form.promptBarMode = "compact" })
                SettingsDesc("詳細設定を折りたたみ、下部バーをモデル選択中心にします。", Modifier.padding(start = 21.dp))
                SettingsRadio("ミニマル表示（送信・プラスのみ）", form.promptBarMode == "minimal", { form.promptBarMode = "minimal" })
                SettingsDesc("下部バーを送信ボタンとプラスボタンだけにし、モデル選択を画面上部へ移します。", Modifier.padding(start = 21.dp))
            }
        },
        SettingsCardSpec(SettingsTab.General, "voice-studio", "音声スタジオUI",
            "音声スタジオUI 音声系モデル（WebSocket）で音声ドックを使う 入力欄の位置にマイク・文字起こし・折りたたみ式の音声設定をまとめ、必要なときだけ拡大表示できます。オフにすると、すべての音声設定を常に表示します。") {
            SettingsCheck("音声系モデル（WebSocket）で音声ドックを使う", form.voiceStudio, { form.voiceStudio = it })
            SettingsDesc("入力欄の位置にマイク・文字起こし・折りたたみ式の音声設定をまとめ、必要なときだけ拡大表示できます。オフにすると、すべての音声設定を常に表示します。",
                Modifier.padding(top = 4.dp))
        },
        SettingsCardSpec(SettingsTab.General, "confirm-dialogs", "確認ダイアログ",
            "確認ダイアログ Gemini音声/動画 + Python のローカル実行切替を送信前に確認 OFF にすると確認を表示しません（このブラウザに保存）。") {
            SettingsCheck("Gemini音声/動画 + Python のローカル実行切替を送信前に確認", localPythonDialog, onLocalPythonDialog)
            SettingsDesc("OFF にすると確認を表示しません（このブラウザに保存）。", Modifier.padding(top = 4.dp))
        },
        SettingsCardSpec(SettingsTab.General, "chat-defaults", "チャット既定値",
            "チャット既定値 既定のモデル 新規チャット開始時に適用されるモデルです。 Vision Model（画像解析用） DeepSeek V4 Pro などの画像非対応モデルで画像が添付された場合、このモデルで解析します。V4.1 Flash は画像入力に対応しています。 前回の設定を保存して継続する ONの場合は下の「デフォルト設定」は無視され、直近の送信時設定を引き継ぎます。 Search URLs Maps Python File Thinking SysPrompt MCP Thinking Level Budget Effort Safety") {
            Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                Column {
                    SettingsFieldLabel("既定のモデル", Modifier.padding(bottom = 4.dp))
                    SettingsSelect(form.defaultModel, defaultModelOptions(models), { form.defaultModel = it })
                    SettingsDesc("新規チャット開始時に適用されるモデルです。", Modifier.padding(top = 4.dp))
                }
                Column {
                    SettingsFieldLabel("Vision Model（画像解析用）", Modifier.padding(bottom = 4.dp))
                    SettingsSelect(form.visionModel, visionModelOptions(models), { form.visionModel = it })
                    SettingsDesc("DeepSeek V4 Pro などの画像非対応モデルで画像が添付された場合、このモデルで解析します。V4.1 Flash は画像入力に対応しています。",
                        Modifier.padding(top = 4.dp))
                }
                SettingsCheck("前回の設定を保存して継続する", form.useLast, { form.useLast = it })
                SettingsDesc("ONの場合は下の「デフォルト設定」は無視され、直近の送信時設定を引き継ぎます。")
                DefaultOptionChecks(form)
                DefaultOptionSelects(form)
            }
        },
        SettingsCardSpec(SettingsTab.General, "search", "検索設定",
            "検索設定 Xリンク検出時に検索ON＋Grok 4 Fast Reasoningへ自動切替 OFFにすると、リンク検出時は毎回確認します。") {
            SettingsCheck("Xリンク検出時に検索ON＋Grok 4 Fast Reasoningへ自動切替", form.autoSearch, { form.autoSearch = it })
            SettingsDesc("OFFにすると、リンク検出時は毎回確認します。", Modifier.padding(top = 4.dp))
        },
        SettingsCardSpec(SettingsTab.General, "voice", "音声設定",
            "音声設定 マイク文字起こし方式 STT API（既定） 現在のLLM（既存モデル） 添付ボタン横のSTTボタンに適用されます（現在選択中モデルを使用）。 STTモデル 「STT API」選択時に使用するモデルです。grok-voice-transcribe は xAI APIキーを使用します。ライブ文字起こしはモデル一覧の gpt-live-transcribe を選択してください。 LLM文字起こしプロンプト（LLM方式） 既定に戻す") {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                SettingsFieldLabel("マイク文字起こし方式")
                SettingsSelect(form.micMode, webOptions("stt_api" to "STT API（既定）", "llm" to "現在のLLM（既存モデル）"), { form.micMode = it })
                SettingsDesc("添付ボタン横のSTTボタンに適用されます（現在選択中モデルを使用）。")
                SettingsFieldLabel("STTモデル", Modifier.padding(top = 4.dp))
                SettingsSelect(form.sttModel, STT_OPTIONS, { form.sttModel = it })
                SettingsDesc("「STT API」選択時に使用するモデルです。grok-voice-transcribe は xAI APIキーを使用します。ライブ文字起こしはモデル一覧の gpt-live-transcribe を選択してください。")
                val rule = LocalWebPalette.current.line.copy(alpha = 0.6f)
                Column(Modifier.fillMaxWidth().settingsTopRule(rule, 8.dp)) {
                    SettingsFieldLabel("LLM文字起こしプロンプト（LLM方式）")
                    SettingsTextField(form.llmPrompt, { form.llmPrompt = it.take(100_000) }, Modifier.fillMaxWidth().padding(top = 4.dp),
                        placeholder = form.llmPromptDefault, minHeight = 96.dp, singleLine = false, fontSize = 12.sp)
                    Row(Modifier.padding(top = 8.dp), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        SettingsSmallButton("既定に戻す", {
                            form.llmPrompt = ""
                            notify("LLM文字起こしプロンプトを既定値に戻しました（保存してください）")
                        }, fontSize = 10.sp)
                        SettingsDesc("LLM方式のマイク文字起こし時のみ使用。空欄で保存すると既定文面を使います（無音時の安全ガードは別途自動付与）。", Modifier.weight(1f))
                    }
                }
            }
        },
        SettingsCardSpec(SettingsTab.General, "temp-chat", "一時チャット",
            "一時チャット 切断タイムアウト（秒） 一時チャットでページの表示/接続が途切れた状態がこの秒数を超えると、自動削除されます。") {
            Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                SettingsFieldLabel("切断タイムアウト（秒）")
                SettingsTextField(form.tempTimeout, { form.tempTimeout = it.filter(Char::isDigit).take(4) }, Modifier.width(112.dp), number = true)
                SettingsDesc("一時チャットでページの表示/接続が途切れた状態がこの秒数を超えると、自動削除されます。")
            }
        },
    )
}

/** `set-default-*` checkboxes with the composer chip label colors. */
@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun DefaultOptionChecks(form: SettingsForm) {
    val web = LocalWebPalette.current
    fun c(dark: Color, light: Color) = if (web.isLight) light else dark
    FlowRow(horizontalArrangement = Arrangement.spacedBy(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        SettingsCheck("Search", form.search, { form.search = it })
        SettingsCheck("URLs", form.urlContext, { form.urlContext = it })
        SettingsCheck("Maps", form.maps, { form.maps = it })
        SettingsCheck("Python", form.python, { form.python = it }, labelColor = c(Color(254, 240, 138), Color(133, 77, 14)))
        SettingsCheck("File", form.fileCreation, { form.fileCreation = it }, labelColor = c(Color(253, 186, 116), Color(154, 52, 18)))
        SettingsCheck("Thinking", form.thinking, { form.thinking = it }, labelColor = c(Color(216, 180, 254), Color(126, 34, 206)))
        SettingsCheck("SysPrompt", form.sysPrompt, { form.sysPrompt = it }, labelColor = c(Color(134, 239, 172), Color(21, 128, 61)))
        SettingsCheck("MCP", form.mcp, { form.mcp = it }, labelColor = c(Color(165, 243, 252), Color(14, 116, 144)))
    }
}

/** Thinking Level / Budget / Effort / Safety defaults. */
@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun DefaultOptionSelects(form: SettingsForm) {
    val web = LocalWebPalette.current
    val labelColor = if (web.isLight) Color(92, 103, 121) else Color(156, 163, 175)
    FlowRow(horizontalArrangement = Arrangement.spacedBy(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        val center = Modifier.align(Alignment.CenterVertically)
        Text("Thinking Level", fontSize = 11.sp, color = labelColor, modifier = center)
        SettingsSelect(form.thinkingLevel, THINKING_OPTIONS, { form.thinkingLevel = it }, center, fillWidth = false, fontSize = 12.sp)
        Text("Budget", fontSize = 11.sp, color = labelColor, modifier = center)
        SettingsTextField(form.thinkingBudget, { form.thinkingBudget = it.filter(Char::isDigit).take(5) }, center.width(80.dp),
            number = true, fontSize = 11.sp)
        Text("Effort", fontSize = 11.sp, color = labelColor, modifier = center)
        SettingsSelect(form.effort, EFFORT_OPTIONS, { form.effort = it }, center, fillWidth = false, fontSize = 12.sp)
        Text("Safety", fontSize = 11.sp, color = labelColor, modifier = center)
        SettingsSelect(form.safety, SAFETY_OPTIONS, { form.safety = it }, center, fillWidth = false, fontSize = 12.sp)
    }
}

internal fun apiCards(state: ChatState, form: SettingsForm, notify: (String) -> Unit): List<SettingsCardSpec> = listOf(
    SettingsCardSpec(SettingsTab.Api, "api-keys", "APIキー",
        "APIキー OpenAI API Key Gemini API Key DeepSeek API Key Kimi (Moonshot) API Key Mistral API Key Anthropic API Key Gemini 接続方式 Gemini API API Key で認証 Vertex AI Project/Location + ADC or JSON Vertex AI Project ID Vertex AI Location Vertex Service Account JSON (任意) xAI API Key Google API Key (TTS) Google Cloud Project ID (TTS) モデル別APIキー（特例） 通常のプロバイダーAPIキーより優先して、指定モデルにだけ適用します。 モデル別のAPIキーを設定する") {
        Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
            KeyField("OpenAI API Key", "openai_key", form)
            KeyField("Gemini API Key", "gemini_key", form)
            KeyField("DeepSeek API Key", "deepseek_key", form)
            KeyField("Kimi (Moonshot) API Key", "kimi_key", form)
            KeyField("Mistral API Key", "mistral_key", form)
            KeyField("Anthropic API Key", "anthropic_key", form)
            GeminiBackendBox(form)
            if (form.geminiBackend == "vertex_ai") Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                LabeledField("Vertex AI Project ID") {
                    SettingsTextField(form.vertexProject, { form.vertexProject = it.take(4096) }, Modifier.fillMaxWidth(), placeholder = "my-gcp-project")
                }
                LabeledField("Vertex AI Location") {
                    SettingsTextField(form.vertexLocation, { form.vertexLocation = it.take(128) }, Modifier.fillMaxWidth(), placeholder = "global")
                }
                SettingsDesc("Vertex AI モードでは、このアプリの Gemini Files API 経路は使えません（大容量ファイル時は Gemini API モード推奨）。")
                LabeledField("Vertex Service Account JSON (任意)") {
                    SettingsTextField(form.vertexJson, { form.vertexJson = it.take(100_000) }, Modifier.fillMaxWidth(),
                        placeholder = "{\"type\":\"service_account\", ...}", minHeight = 112.dp, singleLine = false, mono = true, fontSize = 11.sp)
                }
                SettingsDesc("未入力時はサーバー側ADCを使用します。入力するとこのユーザーの設定だけでVertex認証できます。")
            }
            KeyField("xAI API Key", "xai_key", form)
            KeyField("Google API Key (TTS)", "google_key", form)
            LabeledField("Google Cloud Project ID (TTS)") {
                SettingsTextField(form.googleProject, { form.googleProject = it.take(4096) }, Modifier.fillMaxWidth())
            }
            ModelKeysBox(state, form, notify)
        }
    },
)

@Composable
private fun LabeledField(label: String, field: @Composable () -> Unit) {
    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        SettingsFieldLabel(label)
        field()
    }
}

/** `<input type="password">` holding the server mask until the user types a new key. */
@Composable
private fun KeyField(label: String, field: String, form: SettingsForm) {
    LabeledField(label) {
        SettingsTextField(form.providerKeys[field].orEmpty(), { form.providerKeys[field] = it.take(4096) }, Modifier.fillMaxWidth(), password = true)
    }
}

/** `#gemini-backend-toggle` with its status and note. */
@Composable
private fun GeminiBackendBox(form: SettingsForm) {
    SettingsSubBox {
        Text("Gemini 接続方式", fontSize = 12.sp, fontWeight = FontWeight.SemiBold,
            color = if (LocalWebPalette.current.isLight) Color(92, 103, 121) else Color(156, 163, 175))
        ModeToggle(
            listOf(Triple("gemini_api", "Gemini API", "API Key で認証"), Triple("vertex_ai", "Vertex AI", "Project/Location + ADC or JSON")),
            form.geminiBackend,
        ) { form.geminiBackend = it }
        val vertex = form.geminiBackend == "vertex_ai"
        Text(if (vertex) "現在: Vertex AI（Project ID / Location / 認証情報が必要）" else "現在: Gemini API（Gemini API Key を使用）",
            fontSize = 11.sp, fontWeight = FontWeight.Bold, color = Tw.cyan300, modifier = Modifier.padding(top = 2.dp))
        SettingsDesc(if (vertex) "Vertex AI を利用します。Project ID / Location を設定し、ADC または Vertex Service Account JSON を用意してください。"
            else "Gemini API を利用します。API Key を設定してください。")
    }
}

/** Two option buttons (`syncToggleButtons`: active `border-cyan-400 bg-cyan-900/30`). */
@Composable
private fun ModeToggle(options: List<Triple<String, String, String>>, value: String, onSelect: (String) -> Unit) {
    val web = LocalWebPalette.current
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        options.forEach { (key, title, sub) ->
            val active = key == value
            val shape = RoundedCornerShape(6.dp)
            Column(
                Modifier.fillMaxWidth().clip(shape)
                    .background(if (active) Tw.cyan900.copy(alpha = 0.3f) else web.twBg(Tw.gray800).copy(alpha = 0.7f))
                    .border(1.dp, if (active) Tw.cyan400 else web.twBorder(Tw.gray600), shape)
                    .clickable(role = Role.RadioButton) { onSelect(key) }
                    .padding(horizontal = 12.dp, vertical = 8.dp),
            ) {
                Text(title, fontSize = 12.sp, fontWeight = FontWeight.Bold, color = if (web.isLight) web.text else Color.White)
                Text(sub, fontSize = 10.sp, color = if (web.isLight) Color(92, 103, 121) else Tw.gray400, modifier = Modifier.padding(top = 4.dp))
            }
        }
    }
}

/** `#model-api-keys-panel`: per-model keys that override the provider key. */
@Composable
private fun ModelKeysBox(state: ChatState, form: SettingsForm, notify: (String) -> Unit) {
    val web = LocalWebPalette.current
    var open by remember { mutableStateOf(false) }
    var model by remember { mutableStateOf("") }
    var key by remember { mutableStateOf("") }
    val models = state.account?.models.orEmpty().filter { !it.deprecated }
    SettingsSubBox {
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Column(Modifier.weight(1f)) {
                Text("モデル別APIキー（特例）", fontSize = 12.sp, fontWeight = FontWeight.Bold, color = Tw.cyan300)
                SettingsDesc("通常のプロバイダーAPIキーより優先して、指定モデルにだけ適用します。", Modifier.padding(top = 4.dp))
            }
            SettingsSmallButton(if (open) "モデル別APIキー設定を閉じる" else "モデル別のAPIキーを設定する", { open = !open },
                tone = SettingsButtonTone.Cyan)
        }
        if (open) Column(Modifier.padding(top = 6.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            LabeledField("モデル") {
                SettingsSelect(model, listOf(WebOption("", "モデルを選択")) +
                    models.map { WebOption(it.id, "${it.name} (${it.id})", it.category.ifBlank { null }) }, { model = it }, fontSize = 12.sp)
            }
            LabeledField("APIキー") {
                SettingsTextField(key, { key = it.take(1024) }, Modifier.fillMaxWidth(), placeholder = "モデル専用APIキー", password = true, fontSize = 12.sp)
            }
            SettingsSmallButton("追加/更新", {
                when {
                    model.isBlank() -> notify("モデルを選択してください")
                    key.isBlank() -> notify("APIキーを入力してください")
                    else -> {
                        form.modelKeys[model] = key.trim()
                        notify("モデル別APIキーを設定: $model")
                        key = ""
                    }
                }
            }, tone = SettingsButtonTone.Cyan)
            if (form.modelKeys.isEmpty()) Text("モデル別キーは未設定です。", fontSize = 11.sp, color = if (web.isLight) Color(92, 103, 121) else Tw.gray500)
            form.modelKeys.entries.sortedBy { it.key }.forEach { (id, value) ->
                val shape = RoundedCornerShape(6.dp)
                Row(
                    Modifier.fillMaxWidth().clip(shape).background(web.twBg(Tw.gray900).copy(alpha = 0.7f))
                        .border(1.dp, web.twBorder(Tw.gray700), shape).padding(horizontal = 12.dp, vertical = 8.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(12.dp),
                ) {
                    Column(Modifier.weight(1f)) {
                        val name = models.firstOrNull { it.id == id }?.name ?: id
                        Text("$name ($id)", fontSize = 11.sp, color = if (web.isLight) web.text else Tw.gray200, maxLines = 1)
                        Text(maskApiKeyPreview(value), fontSize = 10.sp, fontFamily = FontFamily.Monospace, color = Tw.cyan300)
                    }
                    SettingsSmallButton("削除", {
                        form.modelKeys.remove(id)
                        notify("モデル別APIキーを削除: $id")
                    }, tone = SettingsButtonTone.Red, fontSize = 10.sp)
                }
            }
            SettingsDesc("未入力で追加はできません。削除すると通常のAPIキー設定に戻ります。")
        }
    }
}

internal fun promptCards(state: ChatState, form: SettingsForm): List<SettingsCardSpec> {
    val prefs = state.preferences
    val autoText = form.autoPrompts.joinToString(" ") { "${it.label} ${it.hint}" }
    return listOf(
        SettingsCardSpec(SettingsTab.Prompt, "system-prompt", "システムプロンプト",
            "システムプロンプト 全体システムプロンプト（全ユーザーに適用 / 参照のみ） 全体システムプロンプトを適用 使用 SysPromptのON/OFFに関わらず適用されます。 ユーザーシステムプロンプト 有効 リセット この欄を空にして無効化します。 自動注入システムプロンプト（ユーザー単位） 既定に戻す 全体適用 $autoText") {
            val web = LocalWebPalette.current
            Column(verticalArrangement = Arrangement.spacedBy(16.dp)) {
                Column {
                    Text("全体システムプロンプト（全ユーザーに適用 / 参照のみ）", fontSize = 12.sp,
                        color = if (web.isLight) Color(92, 103, 121) else Tw.gray400, modifier = Modifier.padding(bottom = 8.dp))
                    SettingsTextField(prefs?.globalSystemPromptEffective.orEmpty(), {}, Modifier.fillMaxWidth(),
                        placeholder = "全体システムプロンプト", minHeight = 128.dp, singleLine = false, readOnly = true, deep = true)
                    SettingsDesc(when {
                        prefs?.globalSystemPromptEnabled == false -> "現在は無効化されています。"
                        prefs?.globalSystemPromptUsesTimeFallback == true -> "管理者設定が空欄のため、時刻の既定プロンプトが適用されています。"
                        else -> "管理者が設定した全体システムプロンプトが適用されています。"
                    }, Modifier.padding(top = 8.dp))
                }
                Column {
                    LabelWithCheck("全体システムプロンプトを適用", "使用", form.applyGlobal) { form.applyGlobal = it }
                    SettingsDesc("SysPromptのON/OFFに関わらず適用されます。OFFにすると全体システムプロンプト（空欄時の時刻プロンプトを含む）を自分のチャットに適用しません。")
                }
                Column {
                    LabelWithCheck("ユーザーシステムプロンプト", "有効", form.userPromptEnabled) { form.userPromptEnabled = it }
                    SettingsTextField(form.userPrompt, { form.userPrompt = it.take(500_000) }, Modifier.fillMaxWidth(),
                        placeholder = "自分だけに適用するシステムプロンプト", minHeight = 128.dp, singleLine = false)
                    Row(Modifier.padding(top = 8.dp), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        SettingsSmallButton("リセット", { form.userPrompt = ""; form.userPromptEnabled = false })
                        SettingsDesc("この欄を空にして無効化します。")
                    }
                }
                AutoSystemPromptSection(form)
            }
        },
    )
}

@Composable
private fun LabelWithCheck(label: String, check: String, checked: Boolean, onChange: (Boolean) -> Unit) {
    Row(Modifier.fillMaxWidth().padding(bottom = 4.dp), verticalAlignment = Alignment.CenterVertically) {
        SettingsFieldLabel(label, Modifier.weight(1f))
        SettingsCheck(check, checked, onChange, fontSize = 10.sp, boxSize = 14.dp,
            labelColor = if (LocalWebPalette.current.isLight) Color(92, 103, 121) else Tw.gray400)
    }
}

/** `#auto-sys-prompt-settings` built by `ensureAutoSystemPromptSettingsCard`. */
@Composable
private fun AutoSystemPromptSection(form: SettingsForm) {
    val web = LocalWebPalette.current
    Column(Modifier.fillMaxWidth().settingsTopRule(web.twBorder(Tw.gray700)), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            SettingsFieldLabel("自動注入システムプロンプト（ユーザー単位）", Modifier.weight(1f))
            SettingsSmallButton("既定に戻す", form::resetAutoPrompts, fontSize = 10.sp)
            SettingsCheck("全体適用", form.applyAutoNotices, { form.applyAutoNotices = it }, fontSize = 10.sp, boxSize = 14.dp,
                labelColor = if (web.isLight) Color(92, 103, 121) else Tw.gray400)
        }
        form.autoPrompts.forEach { row ->
            val shape = RoundedCornerShape(6.dp)
            Column(
                Modifier.fillMaxWidth().clip(shape)
                    .background(if (web.isLight) Color(15, 23, 42).copy(alpha = 0.03f) else Color(3, 7, 18).copy(alpha = 0.4f))
                    .border(1.dp, web.twBorder(Tw.gray700), shape).padding(8.dp),
                verticalArrangement = Arrangement.spacedBy(4.dp),
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(row.label, fontSize = 11.sp, color = if (web.isLight) web.text else Tw.gray300, modifier = Modifier.weight(1f))
                    SettingsCheck("適用", row.enabled,
                        { value -> form.updateAutoPrompt(row.key) { it.copy(enabled = value) } },
                        enabled = !row.mcpLocked, fontSize = 10.sp, boxSize = 14.dp,
                        labelColor = if (web.isLight) Color(92, 103, 121) else Tw.gray500)
                }
                SettingsTextField(row.text, { value -> form.updateAutoPrompt(row.key) { it.copy(text = value.take(100_000)) } },
                    Modifier.fillMaxWidth(), placeholder = row.defaultText.ifBlank { "自動注入文言" }, minHeight = 80.dp,
                    singleLine = false, fontSize = 12.sp, deep = true)
                if (row.hint.isNotBlank()) SettingsDesc(row.hint)
                if (row.mcpLocked) SettingsDesc("この項目のオン・オフはプロンプトバーのMCPスイッチに連動します（オフ時は案内文の注入とツール付与自体が無効）。文面は編集できます。",
                    color = Tw.cyan300.copy(alpha = 0.7f))
            }
        }
        SettingsDesc("各文面はユーザー単位で編集されます。空欄で保存すると既定文面に戻ります。")
    }
}

internal fun displayCards(form: SettingsForm, onPickColor: () -> Unit): List<SettingsCardSpec> = listOf(
    SettingsCardSpec(SettingsTab.Display, "theme", "テーマ", "テーマ リセット 主要アクセントカラーを変更します。") {
        Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                val current = normalizeWebHex(form.themeColor) ?: THEME_DEFAULT
                val shape = RoundedCornerShape(12.dp)
                Box(
                    Modifier.size(42.dp).clip(shape).border(1.dp, LocalWebPalette.current.twBorder(Tw.gray600), shape)
                        .clickable(role = Role.Button, onClick = onPickColor).padding(4.dp)
                        .clip(RoundedCornerShape(8.dp)).background(parseHexColor(current)),
                )
                SettingsTextField(form.themeColor, { form.themeColor = it.take(9) }, Modifier.weight(1f), placeholder = THEME_DEFAULT, fontSize = 12.sp)
                SettingsSmallButton("リセット", { form.themeColor = THEME_DEFAULT })
            }
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                val current = normalizeWebHex(form.themeColor)
                THEME_PRESETS.forEach { hex ->
                    val active = current == hex
                    val theme = LocalWebPalette.current.theme
                    Box(
                        Modifier.size(22.dp)
                            .drawBehind { if (active) drawCircle(theme.rgb(0.6f), radius = size.minDimension / 2 + 2.dp.toPx()) }
                            .clip(CircleShape).background(parseHexColor(hex))
                            .border(2.dp, if (active) Color.White.copy(alpha = 0.6f) else Color(148, 163, 184).copy(alpha = 0.4f), CircleShape)
                            .clickable(role = Role.Button) { form.themeColor = hex },
                    )
                }
            }
            SettingsDesc("主要アクセントカラーを変更します。")
        }
    },
    SettingsCardSpec(SettingsTab.Display, "light", "ライトモード", "ライトモード 手動ライトモード OSの配色設定に関係なく、明るい配色を使用します。") {
        // `fa-sun` is not in the Web icon subset; the Web shows the text without a glyph.
        SettingsSwitchRow("手動ライトモード", "OSの配色設定に関係なく、明るい配色を使用します。", form.lightMode, { form.lightMode = it })
    },
    SettingsCardSpec(SettingsTab.Display, "style", "表示スタイル",
        "表示スタイル Liquid Glassモード 操作・ナビゲーション層を、光の屈折や反射を感じるApple風の素材表現に切り替えます。") {
        SettingsSwitchRow("Liquid Glassモード", "操作・ナビゲーション層を、光の屈折や反射を感じるApple風の素材表現に切り替えます。",
            form.liquidGlass, { form.liquidGlass = it }, icon = R.drawable.fa_solid_droplet, iconTint = Tw.cyan300)
    },
)

internal fun parseHexColor(hex: String): Color =
    runCatching { Color(android.graphics.Color.parseColor(hex)) }.getOrDefault(Color(0xFF0DD4BF))
