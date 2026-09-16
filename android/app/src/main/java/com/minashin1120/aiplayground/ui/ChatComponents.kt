package com.minashin1120.aiplayground.ui

import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.minashin1120.aiplayground.ChatState
import com.minashin1120.aiplayground.ChatViewModel
import com.minashin1120.aiplayground.data.ChatMessage
import com.minashin1120.aiplayground.data.StatusCard
import com.minashin1120.aiplayground.data.attachmentKind
import com.minashin1120.aiplayground.data.attachmentKindIcon
import com.minashin1120.aiplayground.data.formatByteSize
import com.minashin1120.aiplayground.data.gemMentionQuery
import com.minashin1120.aiplayground.data.isImageReference
import com.minashin1120.aiplayground.data.numericId

@OptIn(ExperimentalLayoutApi::class)
@Composable
fun Composer(state: ChatState, model: ChatViewModel, pickModel: () -> Unit, pickFiles: () -> Unit) {
    val selectedModel = state.account?.models?.firstOrNull { it.id == state.model }
    Surface(tonalElevation = 3.dp) {
        Column(Modifier.fillMaxWidth().navigationBarsPadding().padding(horizontal = 16.dp, vertical = 8.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            if (state.editingMessageId != null) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text("メッセージを編集中（送信で分岐を作成）", style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.weight(1f))
                    TextButton(onClick = model::cancelEdit) { Text("キャンセル", style = MaterialTheme.typography.labelMedium) }
                }
            }
            Row(verticalAlignment = Alignment.CenterVertically) {
                TextButton(onClick = pickModel, enabled = !state.streaming, modifier = Modifier.weight(1f)) {
                    Text("${selectedModel?.name ?: state.model.ifBlank { "モデルを選択" }} ▾", maxLines = 1)
                }
                TextButton(onClick = pickFiles, enabled = !state.uploading && !state.streaming) { Text("＋ 添付") }
            }
            selectedModel?.let { info ->
                FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    if (info.supports("thinking")) FilterChip(state.enableThinking, model::toggleThinking, { Text("Thinking") })
                    if (info.supports("search")) FilterChip(state.enableSearch, model::toggleSearch, { Text("Web検索") })
                    if (info.supports("prompt_cache")) FilterChip(state.enablePromptCache, model::togglePromptCache, { Text("Prompt Cache") })
                    if (info.supports("batch")) FilterChip(state.batchMode, model::toggleBatchMode, { Text("Batch") })
                    if (info.supports("python")) FilterChip(state.enablePython, model::togglePython, { Text("Python") })
                    if (info.supports("mcp")) FilterChip(state.enableMcp, model::toggleMcp, { Text("MCP") })
                }
            }
            state.selectedGem?.let { gem ->
                InputChip(selected = true, onClick = { model.chooseGem(null) }, label = { Text("Gem: ${gem.name.take(20)} ×") })
            }
            val gemMention = gemMentionQuery(state.draft)
            if (gemMention != null && state.gems.isNotEmpty()) {
                val candidates = state.gems.filter { it.name.contains(gemMention, ignoreCase = true) }.take(6)
                if (candidates.isNotEmpty()) {
                    FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                        candidates.forEach { gem ->
                            InputChip(selected = false, onClick = { model.applyGemMention(gem, gemMention) },
                                label = { Text("@${gem.name.take(20)}") })
                        }
                    }
                }
            }
            if (state.attachments.isNotEmpty()) FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                state.attachments.forEach { attachment ->
                    InputChip(selected = true, onClick = { model.removeAttachment(attachment.reference) },
                        label = { Text("${attachmentKindIcon(attachmentKind(attachment.name, attachment.mime))} ${attachment.name.take(20)} ×") })
                }
            }
            if (state.uploading) {
                val fraction = if (state.uploadTotal > 0) (state.uploadSent.toFloat() / state.uploadTotal).coerceIn(0f, 1f) else null
                Column(Modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text("アップロード中: ${state.uploadName.ifBlank { "添付" }.take(24)}",
                            style = MaterialTheme.typography.labelSmall, modifier = Modifier.weight(1f))
                        TextButton(onClick = model::cancelUpload) { Text("キャンセル", style = MaterialTheme.typography.labelMedium) }
                    }
                    if (fraction != null) {
                        LinearProgressIndicator(progress = { fraction }, modifier = Modifier.fillMaxWidth())
                        Text("${formatByteSize(state.uploadSent)} / ${formatByteSize(state.uploadTotal)}",
                            style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    } else {
                        LinearProgressIndicator(Modifier.fillMaxWidth())
                    }
                }
            }
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
fun MessageCard(
    message: ChatMessage,
    onFile: (String) -> Unit,
    onQuote: (String) -> Unit = {},
    loader: FileBytesLoader? = null,
    onEdit: (ChatMessage) -> Unit = {},
    onRegenerate: (ChatMessage) -> Unit = {},
    branchIndex: Int = 0,
    branchCount: Int = 0,
    onSwitchBranch: (Int) -> Unit = {},
) {
    val user = message.role == "user"
    val persisted = numericId(message) != null
    var thoughtExpanded by remember(message.id) { mutableStateOf(false) }
    val clipboard = LocalClipboardManager.current
    Surface(color = if (user) MaterialTheme.colorScheme.secondaryContainer else MaterialTheme.colorScheme.surfaceContainerLow,
        shape = RoundedCornerShape(topStart = 20.dp, topEnd = 20.dp, bottomStart = if (user) 20.dp else 4.dp, bottomEnd = if (user) 4.dp else 20.dp),
        modifier = Modifier.fillMaxWidth().padding(start = if (user) 24.dp else 0.dp, end = if (user) 0.dp else 12.dp)) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Text(if (user) "あなた" else "✦ AI", style = MaterialTheme.typography.labelMedium, fontWeight = FontWeight.Bold)
            if (message.thought.isNotBlank()) {
                TextButton(onClick = { thoughtExpanded = !thoughtExpanded }) { Text(if (thoughtExpanded) "思考を閉じる" else "思考を表示") }
                if (thoughtExpanded) SelectionContainer { Text(message.thought, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant) }
            }
            if (message.content.isNotEmpty()) MarkdownText(message.content, loader, onFile)
            message.files.forEach { file ->
                // Assistant images are already rendered inline from the message body.
                val inlineImage = !user && message.content.contains(file)
                if (!inlineImage) {
                    if (isImageReference(file)) {
                        ProtectedImage(file, loader, onFile, modifier = Modifier.fillMaxWidth(), thumbnail = true,
                            contentDescription = file.substringAfterLast('/'))
                    } else {
                        val kind = attachmentKind(file)
                        OutlinedButton(onClick = { onFile(file) }) {
                            Text("${attachmentKindIcon(kind)} ${file.substringAfterLast('/').take(40)}")
                        }
                    }
                }
            }
            if (message.content.isNotBlank() || branchCount > 1 || persisted) {
                Row(horizontalArrangement = Arrangement.spacedBy(4.dp), verticalAlignment = Alignment.CenterVertically) {
                    if (branchCount > 1) {
                        TextButton(onClick = { onSwitchBranch(branchIndex - 1) }, enabled = branchIndex > 0) { Text("◀", style = MaterialTheme.typography.labelMedium) }
                        Text("${branchIndex + 1}/$branchCount", style = MaterialTheme.typography.labelSmall)
                        TextButton(onClick = { onSwitchBranch(branchIndex + 1) }, enabled = branchIndex < branchCount - 1) { Text("▶", style = MaterialTheme.typography.labelMedium) }
                    }
                    if (message.content.isNotBlank()) {
                        TextButton(onClick = { clipboard.setText(AnnotatedString(message.content)) }) { Text("コピー", style = MaterialTheme.typography.labelMedium) }
                        TextButton(onClick = { onQuote(message.content) }) { Text("引用", style = MaterialTheme.typography.labelMedium) }
                    }
                    if (persisted && user) TextButton(onClick = { onEdit(message) }) { Text("編集", style = MaterialTheme.typography.labelMedium) }
                    if (persisted && !user) TextButton(onClick = { onRegenerate(message) }, enabled = message.parentId != null) { Text("再生成", style = MaterialTheme.typography.labelMedium) }
                }
            }
        }
    }
}

@Composable
fun StatusCardView(card: StatusCard) {
    Surface(color = MaterialTheme.colorScheme.surfaceContainerHigh, shape = RoundedCornerShape(12.dp), modifier = Modifier.fillMaxWidth()) {
        Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                if (card.done) Text("✓", color = MaterialTheme.colorScheme.primary, fontWeight = FontWeight.Bold)
                else CircularProgressIndicator(Modifier.size(14.dp), strokeWidth = 2.dp)
                Text(card.label, style = MaterialTheme.typography.labelMedium, fontWeight = FontWeight.SemiBold,
                    modifier = Modifier.padding(start = 8.dp))
            }
            if (card.code.isNotBlank()) {
                Surface(color = MaterialTheme.colorScheme.surfaceVariant, shape = RoundedCornerShape(8.dp)) {
                    Text(card.code, fontFamily = FontFamily.Monospace, style = MaterialTheme.typography.bodySmall,
                        modifier = Modifier.horizontalScroll(rememberScrollState()).padding(10.dp))
                }
            }
            if (card.detail.isNotBlank()) Text(card.detail, style = MaterialTheme.typography.bodySmall)
            if (card.output.isNotBlank()) {
                Text("出力", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                SelectionContainer {
                    Text(card.output, fontFamily = FontFamily.Monospace, style = MaterialTheme.typography.bodySmall,
                        maxLines = 12, overflow = TextOverflow.Ellipsis)
                }
            }
        }
    }
}

@Composable
fun LiveMessage(state: ChatState, onFile: (String) -> Unit, onQuote: (String) -> Unit,
                loader: FileBytesLoader? = null, onMcpDecision: (Boolean) -> Unit = {}) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        state.cards.forEach { card -> key(card.id + card.kind) { StatusCardView(card) } }
        state.mcpDecision?.let { decision ->
            Surface(color = MaterialTheme.colorScheme.tertiaryContainer, shape = RoundedCornerShape(12.dp)) {
                Column(Modifier.fillMaxWidth().padding(12.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    Text("MCPツールの実行確認", fontWeight = FontWeight.SemiBold)
                    Text("${decision.serverName} / ${decision.toolName}", style = MaterialTheme.typography.bodySmall)
                    if (decision.argsPreview.isNotBlank()) SelectionContainer {
                        Text(decision.argsPreview.take(2000), fontFamily = FontFamily.Monospace,
                            style = MaterialTheme.typography.bodySmall, maxLines = 10, overflow = TextOverflow.Ellipsis)
                    }
                    Row { TextButton(onClick = { onMcpDecision(false) }) { Text("拒否") }
                        Button(onClick = { onMcpDecision(true) }) { Text("今回だけ許可") } }
                }
            }
        }
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            if (state.streaming) CircularProgressIndicator(Modifier.size(16.dp), strokeWidth = 2.dp)
            Text(state.status.ifBlank { "受信した回答" }, style = MaterialTheme.typography.labelMedium)
        }
        if (state.liveContent.isNotEmpty() || state.liveThought.isNotEmpty()) {
            MessageCard(ChatMessage("live", "assistant", state.liveContent, state.liveThought), onFile, onQuote, loader)
        }
    }
}
