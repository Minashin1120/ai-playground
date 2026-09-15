package com.minashin1120.aiplayground.ui

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.minashin1120.aiplayground.ChatState
import com.minashin1120.aiplayground.ChatViewModel
import com.minashin1120.aiplayground.data.ChatMessage

@OptIn(ExperimentalLayoutApi::class)
@Composable
fun Composer(state: ChatState, model: ChatViewModel, pickModel: () -> Unit, pickFiles: () -> Unit) {
    Surface(tonalElevation = 3.dp) {
        Column(Modifier.fillMaxWidth().navigationBarsPadding().padding(horizontal = 16.dp, vertical = 8.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                TextButton(onClick = pickModel, enabled = !state.streaming, modifier = Modifier.weight(1f)) {
                    Text("${state.model.ifBlank { "モデルを選択" }} ▾", maxLines = 1)
                }
                TextButton(onClick = pickFiles, enabled = !state.uploading && !state.streaming) { Text("＋ 添付") }
            }
            if (state.attachments.isNotEmpty()) FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                state.attachments.forEach { attachment ->
                    InputChip(selected = true, onClick = { model.removeAttachment(attachment.reference) }, label = { Text("${attachment.name.take(24)} ×") })
                }
            }
            if (state.uploading) { LinearProgressIndicator(Modifier.fillMaxWidth()); Text("添付をアップロード中…", style = MaterialTheme.typography.labelSmall) }
            Row(verticalAlignment = Alignment.Bottom, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedTextField(state.draft, model::draft, placeholder = { Text("メッセージを入力…") }, minLines = 1, maxLines = 6,
                    shape = RoundedCornerShape(20.dp), modifier = Modifier.weight(1f), enabled = !state.streaming)
                if (state.streaming) FilledTonalButton(onClick = model::stop, enabled = state.jobId != null) { Text("停止") }
                else Button(onClick = model::send, enabled = !state.busy && !state.uploading && (state.draft.isNotBlank() || state.attachments.isNotEmpty())) { Text("送信") }
            }
            Text("AIの回答は必ずしも正確とは限りません。", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }
}

@Composable
fun MessageCard(message: ChatMessage, onFile: (String) -> Unit) {
    val user = message.role == "user"
    var thoughtExpanded by remember(message.id) { mutableStateOf(false) }
    Surface(color = if (user) MaterialTheme.colorScheme.secondaryContainer else MaterialTheme.colorScheme.surfaceContainerLow,
        shape = RoundedCornerShape(topStart = 20.dp, topEnd = 20.dp, bottomStart = if (user) 20.dp else 4.dp, bottomEnd = if (user) 4.dp else 20.dp),
        modifier = Modifier.fillMaxWidth().padding(start = if (user) 24.dp else 0.dp, end = if (user) 0.dp else 12.dp)) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Text(if (user) "あなた" else "✦ AI", style = MaterialTheme.typography.labelMedium, fontWeight = FontWeight.Bold)
            if (message.thought.isNotBlank()) {
                TextButton(onClick = { thoughtExpanded = !thoughtExpanded }) { Text(if (thoughtExpanded) "思考を閉じる" else "思考を表示") }
                if (thoughtExpanded) SelectionContainer { Text(message.thought, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant) }
            }
            if (message.content.isNotEmpty()) FormattedText(message.content)
            message.files.forEach { file -> OutlinedButton(onClick = { onFile(file) }) { Text("添付: ${file.substringAfterLast('/').take(48)}") } }
        }
    }
}

@Composable
private fun FormattedText(text: String) {
    SelectionContainer {
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            text.split("```").forEachIndexed { index, part ->
                if (index % 2 == 0) Text(part)
                else Surface(color = MaterialTheme.colorScheme.surfaceContainerHighest, shape = RoundedCornerShape(12.dp)) {
                    Text(part.trim(), fontFamily = FontFamily.Monospace, style = MaterialTheme.typography.bodySmall, modifier = Modifier.padding(12.dp))
                }
            }
        }
    }
}

@Composable
fun LiveMessage(state: ChatState, onFile: (String) -> Unit) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            if (state.streaming) CircularProgressIndicator(Modifier.size(16.dp), strokeWidth = 2.dp)
            Text(state.status.ifBlank { "受信した回答" }, style = MaterialTheme.typography.labelMedium)
        }
        if (state.liveContent.isNotEmpty() || state.liveThought.isNotEmpty()) MessageCard(ChatMessage("live", "assistant", state.liveContent, state.liveThought), onFile)
    }
}
