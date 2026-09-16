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

private val realtimeModels = listOf(
    "gpt-realtime" to "OpenAI Realtime",
    "grok-voice-latest" to "Grok Voice",
    "grok-voice-think-fast-2.0" to "Grok Voice Think",
)

@Composable
fun RealtimeStudioDialog(state: ChatState, model: ChatViewModel, onDismiss: () -> Unit) {
    var selectedModel by remember { mutableStateOf(realtimeModels.first().first) }
    var voice by remember { mutableStateOf("alloy") }
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
                    Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                        realtimeModels.forEach { (id, label) -> FilterChip(selectedModel == id, { selectedModel = id }, { Text(label) }) }
                    }
                    OutlinedTextField(voice, { voice = it }, singleLine = true, label = { Text("Voice") },
                        supportingText = { Text("OpenAI: alloy等 / Grok: Ara等") }, modifier = Modifier.fillMaxWidth())
                    Button(onClick = { model.startRealtime(selectedModel, voice.trim().ifBlank { "alloy" }) }, modifier = Modifier.fillMaxWidth()) { Text("セッションを開始") }
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
