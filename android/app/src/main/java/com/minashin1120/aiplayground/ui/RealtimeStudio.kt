package com.minashin1120.aiplayground.ui

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.minashin1120.aiplayground.ChatState
import com.minashin1120.aiplayground.ChatViewModel
import com.minashin1120.aiplayground.data.ModelInfo

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

@Composable
fun RealtimeStudioDialog(state: ChatState, model: ChatViewModel, onDismiss: () -> Unit) {
    val models = realtimeModels(state)
    var selectedModel by remember(models, state.model) {
        mutableStateOf(models.firstOrNull { it.first == state.model }?.first ?: models.firstOrNull()?.first.orEmpty())
    }
    var voice by remember { mutableStateOf("alloy") }
    var targetLanguage by remember { mutableStateOf("ja") }
    var thinkingLevel by remember { mutableStateOf("medium") }
    var transcriptionMode by remember { mutableStateOf("VERBATIM") }
    var customVocabulary by remember { mutableStateOf("") }
    val realtime = state.realtime
    PlaygroundDialog(
        onDismissRequest = { if (!realtime.active) onDismiss() },
        title = { Text("Realtime音声") },
        text = {
            Column(Modifier.verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Text("マイクの音声をサーバーのRealtimeセッションへ送り、返答を低遅延で再生します。APIキーは端末へ渡しません。",
                    style = MaterialTheme.typography.bodySmall)
                if (!realtime.active) {
                    Text("モデル", style = MaterialTheme.typography.labelMedium)
                    if (models.isEmpty()) {
                        Text("使用可能なRealtimeモデルがありません。モデル一覧を更新してください。",
                            style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    } else {
                        Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                            models.forEach { (id, label) -> FilterChip(selectedModel == id, { selectedModel = id }, { Text(label) }) }
                        }
                    }
                    if (selectedModel == "gemini-3.5-live-translate-preview") {
                        OutlinedTextField(targetLanguage, { targetLanguage = it }, singleLine = true,
                            label = { Text("翻訳先言語コード") }, supportingText = { Text("例: ja / en / ko") },
                            modifier = Modifier.fillMaxWidth())
                    }
                    if (selectedModel == "gemini-3.5-transcribe-live") {
                        OutlinedTextField(transcriptionMode, { transcriptionMode = it }, singleLine = true,
                            label = { Text("文字起こしモード") }, supportingText = { Text("SMART / VERBATIM") },
                            modifier = Modifier.fillMaxWidth())
                        OutlinedTextField(customVocabulary, { customVocabulary = it }, minLines = 2, maxLines = 4,
                            label = { Text("カスタム語彙") }, supportingText = { Text("カンマまたは改行区切り（任意）") },
                            modifier = Modifier.fillMaxWidth())
                    }
                    if (isGeminiExtendedThinking(selectedModel)) {
                        OutlinedTextField(thinkingLevel, { thinkingLevel = it }, singleLine = true,
                            label = { Text("Thinking level") }, supportingText = { Text("low / medium / high") },
                            modifier = Modifier.fillMaxWidth())
                    }
                    if (selectedModel != "gemini-3.5-live-translate-preview" && selectedModel != "gemini-3.5-transcribe-live") {
                        OutlinedTextField(voice, { voice = it }, singleLine = true, label = { Text("Voice") },
                            supportingText = { Text("OpenAI: alloy等 / Grok: Ara等 / Gemini: Kore等") }, modifier = Modifier.fillMaxWidth())
                    }
                    Text(if (selectedModel == "gemini-3.5-transcribe-live")
                        "リアルタイム文字起こし。音声は返さず、入力テキストを履歴へ保存します。"
                    else if (selectedModel == "gemini-3.5-live-translate-preview")
                        "音声を指定言語へリアルタイム翻訳します。"
                    else "APIキーは端末へ渡さず、認証済みサーバーがGeminiへ接続します。",
                        style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    Button(onClick = {
                        model.startRealtime(selectedModel, voice.trim().ifBlank { "Kore" }, targetLanguage, thinkingLevel,
                            transcriptionMode, customVocabulary)
                    }, enabled = selectedModel.isNotBlank(), modifier = Modifier.fillMaxWidth()) { Text("セッションを開始") }
                } else {
                    Text("${realtime.model} · ${realtime.status.ifBlank { "接続中…" }}", style = MaterialTheme.typography.labelMedium)
                    if (realtime.userText.isNotBlank()) Text("あなた: ${realtime.userText}", style = MaterialTheme.typography.bodyMedium)
                    if (realtime.assistantText.isNotBlank()) Text("AI: ${realtime.assistantText}", style = MaterialTheme.typography.bodyMedium)
                    if (realtime.thoughtText.isNotBlank()) Text("思考: ${realtime.thoughtText}", style = MaterialTheme.typography.bodySmall)
                    Text("再生済み音声: ${realtime.audioBytes / 1024} KB", style = MaterialTheme.typography.labelSmall)
                    realtime.error?.let { Text(it, color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall) }
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        Button(onClick = model::commitRealtime) { Text("発話を確定") }
                        TextButton(onClick = { model.stopRealtime(save = true); onDismiss() }) { Text("保存して終了") }
                        TextButton(onClick = { model.stopRealtime(save = false); onDismiss() }) { Text("終了") }
                    }
                }
            }
        },
        confirmButton = { if (!realtime.active) TextButton(onClick = onDismiss) { Text("閉じる") } },
        dismissButton = if (realtime.active) ({ TextButton(onClick = { model.stopRealtime(false); onDismiss() }) { Text("キャンセル") } }) else null,
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
