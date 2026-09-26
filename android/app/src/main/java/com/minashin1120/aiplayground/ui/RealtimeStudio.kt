package com.minashin1120.aiplayground.ui

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.Mic
import androidx.compose.material.icons.rounded.OpenInFull
import androidx.compose.material.icons.rounded.Stop
import androidx.compose.material.icons.rounded.Tune
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.minashin1120.aiplayground.ChatState
import com.minashin1120.aiplayground.ChatViewModel
import com.minashin1120.aiplayground.data.ModelInfo
import com.minashin1120.aiplayground.data.RealtimeState

private const val REALTIME_AUDIO_MODE = "realtime_audio"
private const val REALTIME_TRANSCRIPTION_ONLY_MODEL = "gpt-realtime-whisper"

internal fun isRealtimeAudioModel(model: ModelInfo?): Boolean =
    model != null && model.mode == REALTIME_AUDIO_MODE && model.selectable &&
        model.id != REALTIME_TRANSCRIPTION_ONLY_MODEL

internal fun realtimeModels(state: ChatState): List<Pair<String, String>> =
    state.account?.models.orEmpty()
        .filter(::isRealtimeAudioModel)
        .map { it.id to it.name }

private fun isGeminiLive(model: String): Boolean = model.startsWith("gemini-3.1-flash-live") ||
    model == "gemini-3.8-live" || model == "gemini-3.8-live-extended-thinking" ||
    model == "gemini-3.5-live-translate-preview" || model == "gemini-3.5-transcribe-live"

private fun isGeminiExtendedThinking(model: String): Boolean = model == "gemini-3.8-live-extended-thinking"

private fun isGrokLiveTranscribe(model: String): Boolean = model == "grok-voice-transcribe-2.0"

// Same voice lists as the Web STS panel (chat_core part05: *_STS_VOICES).
internal val OPENAI_REALTIME_VOICES = listOf("alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse", "marin", "cedar")
internal val GROK_REALTIME_VOICES = listOf("Ara", "Rex", "Sal", "Eve", "Leo")
internal val GEMINI_REALTIME_VOICES = listOf(
    "Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus", "Aoede", "Callirrhoe", "Autonoe",
    "Enceladus", "Iapetus", "Umbriel", "Algieba", "Despina", "Erinome", "Algenib", "Rasalgethi", "Laomedeia", "Achernar",
    "Alnilam", "Schedar", "Gacrux", "Pulcherrima", "Achird", "Zubenelgenubi", "Vindemiatrix", "Sadachbia", "Sadaltager", "Sulafat",
)
internal val REALTIME_TARGET_LANGUAGES = listOf(
    "ja" to "日本語", "en" to "English", "ko" to "한국어", "zh-CN" to "中文（簡体）", "zh-TW" to "中文（繁体）",
    "fr" to "Français", "de" to "Deutsch", "es" to "Español", "it" to "Italiano", "pt" to "Português",
    "ru" to "Русский", "hi" to "हिन्दी", "th" to "ไทย", "vi" to "Tiếng Việt", "id" to "Bahasa Indonesia", "ar" to "العربية",
)

/** Settings shared by the composer voice dock and the enlarged studio dialog. */
data class RealtimeOptions(
    val voice: String = "",
    val targetLanguage: String = "ja",
    val thinkingLevel: String = "",
    val transcriptionMode: String = "VERBATIM",
    val customVocabulary: String = "",
    /** Web `#sts-include-thoughts` (Thoughts). */
    val includeThoughts: Boolean = false,
)

internal fun isRealtimeTranscription(model: String): Boolean =
    model == "gemini-3.5-transcribe-live" || isGrokLiveTranscribe(model) || model.contains("transcribe")

internal fun realtimeVoices(model: String): List<String> = when {
    isRealtimeTranscription(model) || model == "gemini-3.5-live-translate-preview" -> emptyList()
    model.contains("gpt-realtime") -> OPENAI_REALTIME_VOICES
    model.contains("grok-voice") -> GROK_REALTIME_VOICES
    isGeminiLive(model) || model.startsWith("gemini-") -> GEMINI_REALTIME_VOICES
    else -> emptyList()
}

/** Thinking levels the Web panel offers for this model; empty when it has no Thinking control. */
internal fun realtimeThinkingLevels(model: String): List<String> = when {
    isGeminiExtendedThinking(model) -> listOf("low", "medium", "high")
    model == "gemini-3.8-live" || model == "gemini-3.5-live-translate-preview" || isRealtimeTranscription(model) -> emptyList()
    model.startsWith("gemini-") -> listOf("minimal", "low", "medium", "high")
    else -> emptyList()
}

internal fun resolvedRealtimeVoice(model: String, options: RealtimeOptions): String {
    val voices = realtimeVoices(model)
    return voices.firstOrNull { it == options.voice } ?: voices.firstOrNull() ?: "Kore"
}

internal fun resolvedRealtimeThinking(model: String, options: RealtimeOptions): String {
    val levels = realtimeThinkingLevels(model)
    return when {
        options.thinkingLevel in levels -> options.thinkingLevel
        isGeminiExtendedThinking(model) -> "medium"
        else -> levels.firstOrNull() ?: "minimal"
    }
}

internal fun startRealtimeWith(model: ChatViewModel, modelId: String, options: RealtimeOptions) {
    model.startRealtime(modelId, resolvedRealtimeVoice(modelId, options), options.targetLanguage,
        resolvedRealtimeThinking(modelId, options), options.transcriptionMode, options.customVocabulary,
        includeThoughts = realtimeThinkingLevels(modelId).isNotEmpty() && options.includeThoughts)
}

private fun realtimeModeLabel(model: String): String = when {
    isRealtimeTranscription(model) -> "リアルタイム文字起こし"
    model == "gemini-3.5-live-translate-preview" -> "リアルタイム音声翻訳"
    else -> "音声会話"
}

private fun realtimeModeNote(model: String): String = when {
    isRealtimeTranscription(model) -> "音声は返さず、文字起こしを履歴へ保存します。"
    model == "gemini-3.5-live-translate-preview" -> "話した音声を指定言語へリアルタイム翻訳します。"
    else -> "停止するとチャットに保存されます。"
}

@Composable
private fun OptionChips(label: String, values: List<Pair<String, String>>, selected: String, onSelect: (String) -> Unit) {
    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        Text(label, style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            values.forEach { (value, text) -> FilterChip(selected == value, { onSelect(value) }, { Text(text) }) }
        }
    }
}

/** Model-specific realtime settings rendered as chips, mirroring the Web STS options. */
@Composable
fun RealtimeOptionsEditor(modelId: String, options: RealtimeOptions, enabled: Boolean, onChange: (RealtimeOptions) -> Unit) {
    Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
        val voices = realtimeVoices(modelId)
        if (voices.isNotEmpty()) OptionChips("Voice", voices.map { it to it }, resolvedRealtimeVoice(modelId, options)) {
            if (enabled) onChange(options.copy(voice = it))
        }
        if (modelId == "gemini-3.5-live-translate-preview") {
            OptionChips("翻訳先", REALTIME_TARGET_LANGUAGES, options.targetLanguage) { if (enabled) onChange(options.copy(targetLanguage = it)) }
        }
        val levels = realtimeThinkingLevels(modelId)
        if (levels.isNotEmpty()) OptionChips("Thinking", levels.map { it to it.replaceFirstChar(Char::uppercase) },
            resolvedRealtimeThinking(modelId, options)) { if (enabled) onChange(options.copy(thinkingLevel = it)) }
        if (levels.isNotEmpty()) SettingsCheck("Thoughts", options.includeThoughts, { if (enabled) onChange(options.copy(includeThoughts = it)) },
            fontSize = 10.sp, boxSize = 12.dp, enabled = enabled)
        if (modelId == "gemini-3.5-transcribe-live") {
            OptionChips("モード", listOf("VERBATIM" to "Verbatim", "SMART" to "Smart"), options.transcriptionMode) {
                if (enabled) onChange(options.copy(transcriptionMode = it))
            }
        }
        if (modelId == "gemini-3.5-transcribe-live" || isGrokLiveTranscribe(modelId)) {
            OutlinedTextField(options.customVocabulary, { onChange(options.copy(customVocabulary = it)) }, enabled = enabled,
                minLines = 1, maxLines = 4,
                label = { Text(if (isGrokLiveTranscribe(modelId)) "キーワード" else "カスタム語彙") },
                supportingText = { Text(if (isGrokLiveTranscribe(modelId)) "カンマまたは改行区切り（任意・最大100件）" else "カンマまたは改行区切り（任意）") },
                modifier = Modifier.fillMaxWidth())
        }
    }
}

@Composable
private fun RealtimeTranscript(realtime: RealtimeState, compact: Boolean) {
    val lines = listOfNotNull(
        realtime.userText.takeIf { it.isNotBlank() }?.let { "あなた" to it },
        realtime.assistantText.takeIf { it.isNotBlank() }?.let { "AI" to it },
        realtime.thoughtText.takeIf { it.isNotBlank() && !compact }?.let { "思考" to it },
    )
    if (lines.isEmpty()) return
    Surface(shape = RoundedCornerShape(12.dp), color = MaterialTheme.colorScheme.surfaceContainerLow,
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant), modifier = Modifier.fillMaxWidth()) {
        Column(
            Modifier.heightIn(max = if (compact) 120.dp else 320.dp).verticalScroll(rememberScrollState(), reverseScrolling = true)
                .padding(horizontal = 10.dp, vertical = 8.dp),
            verticalArrangement = Arrangement.spacedBy(4.dp),
        ) {
            lines.forEach { (label, text) ->
                Text("$label: $text", style = if (label == "思考") MaterialTheme.typography.bodySmall else MaterialTheme.typography.bodyMedium,
                    color = if (label == "あなた") MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurface)
            }
        }
    }
}

/**
 * Composer replacement for realtime audio models (Web: inline voice dock).
 * The conversation stays in the chat; the dialog is only an enlarged view.
 */
@Composable
fun RealtimeVoiceDock(
    state: ChatState,
    model: ChatViewModel,
    options: RealtimeOptions,
    onOptions: (RealtimeOptions) -> Unit,
    onStart: () -> Unit,
    onExpand: () -> Unit,
) {
    val colors = MaterialTheme.colorScheme
    val realtime = state.realtime
    val modelId = if (realtime.active) realtime.model else state.model
    var settingsOpen by remember { mutableStateOf(false) }
    val canStart = !state.offline && !state.streaming && !state.busy && state.model.isNotBlank()
    Surface(shape = RoundedCornerShape(16.dp), color = colors.surfaceContainerLow,
        border = BorderStroke(1.dp, if (realtime.active) colors.primary.copy(alpha = 0.55f) else colors.outline),
        modifier = Modifier.fillMaxWidth()) {
        Column(Modifier.padding(10.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                FilledIconButton(
                    onClick = { if (realtime.active) model.stopRealtime(save = true) else onStart() },
                    enabled = realtime.active || canStart,
                    shape = CircleShape,
                    colors = IconButtonDefaults.filledIconButtonColors(
                        containerColor = if (realtime.active) colors.error else colors.primary,
                        contentColor = if (realtime.active) colors.onError else colors.onPrimary),
                    modifier = Modifier.size(52.dp),
                ) {
                    Icon(if (realtime.active) Icons.Rounded.Stop else Icons.Rounded.Mic,
                        contentDescription = if (realtime.active) "停止してチャットに保存" else "音声セッションを開始")
                }
                Column(Modifier.weight(1f)) {
                    Text(realtimeModeLabel(modelId), style = MaterialTheme.typography.labelSmall, color = colors.primary)
                    Text(if (realtime.active) realtime.status.ifBlank { "接続中…" } else "タップして話し始める",
                        style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.SemiBold,
                        maxLines = 1, overflow = TextOverflow.Ellipsis)
                    Text(realtimeModeNote(modelId), style = MaterialTheme.typography.labelSmall, color = colors.onSurfaceVariant,
                        maxLines = 1, overflow = TextOverflow.Ellipsis)
                }
                if (realtime.active) {
                    TextButton(onClick = { model.stopRealtime(save = false) }) { Text("キャンセル") }
                } else {
                    IconButton(onClick = { settingsOpen = !settingsOpen }) {
                        Icon(Icons.Rounded.Tune, contentDescription = if (settingsOpen) "音声設定を閉じる" else "音声設定",
                            tint = if (settingsOpen) colors.primary else colors.onSurfaceVariant)
                    }
                }
                IconButton(onClick = onExpand) { Icon(Icons.Rounded.OpenInFull, contentDescription = "音声スタジオで拡大表示") }
            }
            if (realtime.active) {
                RealtimeTranscript(realtime, compact = true)
                realtime.error?.let { Text(it, color = colors.error, style = MaterialTheme.typography.bodySmall) }
                if (!isRealtimeTranscription(modelId)) {
                    OutlinedButton(onClick = model::commitRealtime, modifier = Modifier.fillMaxWidth()) { Text("発話を確定") }
                }
            }
            AnimatedVisibility(settingsOpen && !realtime.active) {
                RealtimeOptionsEditor(modelId, options, enabled = true, onChange = onOptions)
            }
        }
    }
}

@Composable
fun RealtimeStudioDialog(
    state: ChatState,
    model: ChatViewModel,
    options: RealtimeOptions,
    onOptions: (RealtimeOptions) -> Unit,
    onDismiss: () -> Unit,
) {
    val models = realtimeModels(state)
    var selectedModel by remember(models, state.model) {
        mutableStateOf(models.firstOrNull { it.first == state.model }?.first ?: models.firstOrNull()?.first.orEmpty())
    }
    val realtime = state.realtime
    val colors = MaterialTheme.colorScheme
    PlaygroundDialog(
        // The session keeps running when the enlarged view is closed; the composer dock still controls it.
        onDismissRequest = onDismiss,
        title = { Text(if (realtime.active) realtimeModeLabel(realtime.model) else "音声スタジオ") },
        text = {
            Column(Modifier.verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                if (!realtime.active) {
                    if (models.isEmpty()) {
                        Text("使用可能なRealtimeモデルがありません。モデル一覧を更新してください。",
                            style = MaterialTheme.typography.bodySmall, color = colors.onSurfaceVariant)
                    } else {
                        OptionChips("モデル", models, selectedModel) { selectedModel = it }
                    }
                    RealtimeOptionsEditor(selectedModel, options, enabled = true, onChange = onOptions)
                    Text(realtimeModeNote(selectedModel) + " APIキーは端末へ渡さず、認証済みサーバーが接続します。",
                        style = MaterialTheme.typography.bodySmall, color = colors.onSurfaceVariant)
                    Button(onClick = { startRealtimeWith(model, selectedModel, options) },
                        enabled = selectedModel.isNotBlank() && !state.offline, modifier = Modifier.fillMaxWidth().height(52.dp)) {
                        Icon(Icons.Rounded.Mic, contentDescription = null)
                        Spacer(Modifier.width(8.dp))
                        Text("話し始める")
                    }
                } else {
                    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                        FilledIconButton(onClick = { model.stopRealtime(save = true) }, shape = CircleShape,
                            colors = IconButtonDefaults.filledIconButtonColors(containerColor = colors.error, contentColor = colors.onError),
                            modifier = Modifier.size(64.dp)) {
                            Icon(Icons.Rounded.Stop, contentDescription = "停止してチャットに保存")
                        }
                        Column(Modifier.weight(1f)) {
                            Text(realtime.status.ifBlank { "接続中…" }, style = MaterialTheme.typography.titleSmall)
                            Text("${realtime.model} · 再生済み ${realtime.audioBytes / 1024} KB",
                                style = MaterialTheme.typography.labelSmall, color = colors.onSurfaceVariant)
                        }
                    }
                    RealtimeTranscript(realtime, compact = false)
                    if (realtime.userText.isBlank() && realtime.assistantText.isBlank()) {
                        Text("会話の文字起こしがここに表示されます。", style = MaterialTheme.typography.bodySmall, color = colors.onSurfaceVariant)
                    }
                    realtime.error?.let { Text(it, color = colors.error, style = MaterialTheme.typography.bodySmall) }
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        if (!isRealtimeTranscription(realtime.model)) OutlinedButton(onClick = model::commitRealtime) { Text("発話を確定") }
                        TextButton(onClick = { model.stopRealtime(save = false) }) { Text("保存せず終了") }
                    }
                }
            }
        },
        confirmButton = { TextButton(onClick = onDismiss) { Text(if (realtime.active) "チャットに戻る" else "閉じる") } },
    )
}

@Composable
fun LyriaStudioDialog(state: ChatState, model: ChatViewModel, onDismiss: () -> Unit) {
    val lyria = state.lyria
    var prompt by remember { mutableStateOf(lyria.prompt) }
    PlaygroundDialog(
        onDismissRequest = { if (!lyria.active) onDismiss() },
        title = { Text("Lyria RealTime") },
        text = {
            Column(Modifier.verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Text("プロンプトから音楽をリアルタイム生成します。再生中も一時停止・再開できます。",
                    style = MaterialTheme.typography.bodySmall)
                if (!lyria.active) {
                    OutlinedTextField(prompt, { prompt = it }, minLines = 3, maxLines = 8,
                        label = { Text("音楽プロンプト") }, modifier = Modifier.fillMaxWidth())
                    Button(onClick = { model.startLyria(prompt) }, enabled = prompt.isNotBlank(), modifier = Modifier.fillMaxWidth()) { Text("生成を開始") }
                } else {
                    Text(lyria.status.ifBlank { "生成中…" }, style = MaterialTheme.typography.labelMedium)
                    Text("再生済み音声: ${lyria.audioBytes / 1024} KB", style = MaterialTheme.typography.labelSmall)
                    lyria.error?.let { Text(it, color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall) }
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        TextButton(onClick = { model.lyriaControl("PAUSE") }) { Text("一時停止") }
                        TextButton(onClick = { model.lyriaControl("PLAY") }) { Text("再開") }
                        TextButton(onClick = { model.stopLyria(true); onDismiss() }) { Text("保存して終了") }
                        TextButton(onClick = { model.stopLyria(false); onDismiss() }) { Text("終了") }
                    }
                }
            }
        },
        confirmButton = { if (!lyria.active) TextButton(onClick = onDismiss) { Text("閉じる") } },
        dismissButton = if (lyria.active) ({ TextButton(onClick = { model.stopLyria(false); onDismiss() }) { Text("キャンセル") } }) else null,
    )
}
