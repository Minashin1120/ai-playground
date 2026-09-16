package com.minashin1120.aiplayground.ui

import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.input.key.*
import androidx.compose.ui.text.input.ImeAction
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
fun Composer(state: ChatState, model: ChatViewModel, pickModel: () -> Unit, pickFiles: () -> Unit, onVoice: () -> Unit) {
    var details by remember { mutableStateOf(false) }
    val selectedModel = state.account?.models?.firstOrNull { it.id == state.model }
    val colors = MaterialTheme.colorScheme
    Surface(color = colors.surface.copy(alpha = 0.96f), shadowElevation = 8.dp) {
        Box(Modifier.fillMaxWidth(), contentAlignment = Alignment.Center) {
        Column(
            Modifier.widthIn(max = PlaygroundDimens.contentMax).fillMaxWidth().navigationBarsPadding().padding(horizontal = 12.dp, vertical = 9.dp),
            verticalArrangement = Arrangement.spacedBy(7.dp),
        ) {
            if (state.editingMessageId != null) {
                Surface(shape = RoundedCornerShape(12.dp), color = colors.secondary.copy(alpha = 0.10f), border = androidx.compose.foundation.BorderStroke(1.dp, colors.secondary.copy(alpha = 0.25f))) {
                    Row(Modifier.fillMaxWidth().padding(start = 10.dp), verticalAlignment = Alignment.CenterVertically) {
                    Icon(Icons.Rounded.Edit, contentDescription = null, tint = colors.secondary, modifier = Modifier.size(16.dp))
                    Text("メッセージを編集中（送信で分岐を作成）", style = MaterialTheme.typography.labelMedium,
                        color = colors.onSurfaceVariant, modifier = Modifier.padding(start = 7.dp).weight(1f))
                    TextButton(onClick = model::cancelEdit) { Text("キャンセル", style = MaterialTheme.typography.labelMedium) }
                    }
                }
            }
            Row(verticalAlignment = Alignment.CenterVertically) {
                Surface(onClick = pickModel, enabled = !state.streaming, color = colors.surfaceContainerHigh, shape = RoundedCornerShape(50), border = androidx.compose.foundation.BorderStroke(1.dp, colors.outlineVariant), modifier = Modifier.weight(1f)) {
                    Row(Modifier.padding(horizontal = 11.dp, vertical = 7.dp), verticalAlignment = Alignment.CenterVertically) {
                        Text(selectedModel?.name ?: state.model.ifBlank { "モデルを選択" }, style = MaterialTheme.typography.labelLarge, fontWeight = FontWeight.SemiBold, maxLines = 1, overflow = TextOverflow.Ellipsis, modifier = Modifier.weight(1f))
                        Icon(Icons.Rounded.KeyboardArrowDown, contentDescription = null, modifier = Modifier.size(18.dp))
                    }
                }
                TextButton(onClick = { details = !details }) { Text(if (details) "詳細を閉じる" else "詳細") }
            }
            if (details) selectedModel?.let { info ->
                Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    if (info.supports("thinking")) FilterChip(state.enableThinking, model::toggleThinking, { Text("Thinking") })
                    if (info.supports("search")) FilterChip(state.enableSearch, model::toggleSearch, { Text("Web検索") })
                    if (info.supports("prompt_cache")) FilterChip(state.enablePromptCache, model::togglePromptCache, { Text("Prompt Cache") })
                    if (info.supports("batch")) FilterChip(state.batchMode, model::toggleBatchMode, { Text("Batch") })
                    if (info.supports("python")) FilterChip(state.enablePython, model::togglePython, { Text("Python") })
                    if (info.supports("mcp")) FilterChip(state.enableMcp, model::toggleMcp, { Text("MCP") })
                }
                GenerationOptionsPanel(info, state.generationValues[info.id].orEmpty(), !state.streaming, model::generationOption)
            }
            state.selectedGem?.let { gem ->
                InputChip(selected = true, onClick = { model.chooseGem(null) }, label = { Text("Gem: ${gem.name.take(20)} ×") })
                if (gem.fixedPrompts.isNotEmpty()) Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    gem.fixedPrompts.forEach { prompt ->
                        SuggestionChip(onClick = { model.draft(prompt.content); model.send() },
                            enabled = !state.streaming && !state.busy && !state.uploading,
                            label = { Text(prompt.name) })
                    }
                }
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
                Surface(shape = RoundedCornerShape(16.dp), color = colors.surfaceContainerLow, border = androidx.compose.foundation.BorderStroke(1.dp, colors.outline), shadowElevation = 2.dp, modifier = Modifier.weight(1f)) {
                    Row(Modifier.padding(start = 2.dp, end = 5.dp), verticalAlignment = Alignment.Bottom) {
                        IconButton(onClick = pickFiles, enabled = !state.uploading && !state.streaming, modifier = Modifier.padding(bottom = 4.dp)) {
                            Icon(Icons.Rounded.AttachFile, contentDescription = "添付を追加")
                        }
                        IconButton(onClick = onVoice, enabled = !state.streaming, modifier = Modifier.padding(bottom = 4.dp)) {
                            Icon(Icons.Rounded.Mic, contentDescription = "音声入力")
                        }
                        OutlinedTextField(
                            state.draft, model::draft, placeholder = { Text("メッセージを入力…") }, minLines = 1, maxLines = 6,
                            modifier = Modifier.weight(1f).onPreviewKeyEvent { event ->
                                val sendKey = event.key == Key.Enter && !event.isShiftPressed &&
                                    (event.isCtrlPressed || state.preferences?.enterToSend == true)
                                if (sendKey) {
                                    if (event.type == KeyEventType.KeyDown) model.send()
                                    true
                                } else false
                            }, enabled = !state.streaming,
                            keyboardOptions = KeyboardOptions(imeAction = if (state.preferences?.enterToSend == true) ImeAction.Send else ImeAction.Default),
                            keyboardActions = KeyboardActions(onSend = { model.send() }),
                            colors = OutlinedTextFieldDefaults.colors(focusedBorderColor = Color.Transparent, unfocusedBorderColor = Color.Transparent, disabledBorderColor = Color.Transparent, focusedContainerColor = Color.Transparent, unfocusedContainerColor = Color.Transparent, disabledContainerColor = Color.Transparent),
                        )
                        if (state.streaming) FilledIconButton(onClick = model::stop, enabled = state.jobId != null, colors = IconButtonDefaults.filledIconButtonColors(containerColor = colors.error, contentColor = colors.onError), shape = RoundedCornerShape(12.dp), modifier = Modifier.padding(bottom = 5.dp)) {
                            Icon(Icons.Rounded.Stop, contentDescription = "生成を停止")
                        } else FilledIconButton(onClick = model::send, enabled = !state.busy && !state.uploading && (state.draft.isNotBlank() || state.attachments.isNotEmpty()), shape = RoundedCornerShape(12.dp), modifier = Modifier.padding(bottom = 5.dp)) {
                            Icon(Icons.Rounded.Send, contentDescription = "送信")
                        }
                    }
                }
            }
        }
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
    val colors = MaterialTheme.colorScheme
    val shape = RoundedCornerShape(topStart = 20.dp, topEnd = 20.dp, bottomStart = if (user) 20.dp else 5.dp, bottomEnd = if (user) 5.dp else 20.dp)
    val fill = if (user) Modifier.background(Brush.linearGradient(listOf(colors.primary, Color(0xFF08B2A5))))
        else Modifier.background(colors.surface.copy(alpha = 0.90f))
    Box(
        Modifier.fillMaxWidth().padding(start = if (user) 40.dp else 0.dp, end = if (user) 0.dp else 24.dp)
            .shadow(8.dp, shape).clip(shape).then(fill)
            .border(1.dp, if (user) colors.primary.copy(alpha = 0.38f) else colors.outlineVariant.copy(alpha = 0.72f), shape)
    ) {
        CompositionLocalProvider(LocalContentColor provides if (user) colors.onPrimary else colors.onSurface) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                Icon(if (user) Icons.Rounded.Person else Icons.Rounded.AutoAwesome, contentDescription = null, modifier = Modifier.size(16.dp), tint = if (user) colors.onPrimary else colors.primary)
                Text(if (user) "あなた" else "AI", style = MaterialTheme.typography.labelMedium, fontWeight = FontWeight.Bold)
            }
            if (message.thought.isNotBlank()) {
                TextButton(onClick = { thoughtExpanded = !thoughtExpanded }, colors = ButtonDefaults.textButtonColors(contentColor = if (user) colors.onPrimary else colors.primary)) { Text(if (thoughtExpanded) "思考を閉じる" else "思考を表示") }
                if (thoughtExpanded) SelectionContainer { Text(message.thought, style = MaterialTheme.typography.bodySmall, color = if (user) colors.onPrimary.copy(alpha = 0.78f) else colors.onSurfaceVariant) }
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
                        IconButton(onClick = { onSwitchBranch(branchIndex - 1) }, enabled = branchIndex > 0, modifier = Modifier.size(34.dp)) { Icon(Icons.Rounded.ChevronLeft, contentDescription = "前の分岐", modifier = Modifier.size(18.dp)) }
                        Text("${branchIndex + 1}/$branchCount", style = MaterialTheme.typography.labelSmall)
                        IconButton(onClick = { onSwitchBranch(branchIndex + 1) }, enabled = branchIndex < branchCount - 1, modifier = Modifier.size(34.dp)) { Icon(Icons.Rounded.ChevronRight, contentDescription = "次の分岐", modifier = Modifier.size(18.dp)) }
                    }
                    if (message.content.isNotBlank()) {
                        IconButton(onClick = { clipboard.setText(AnnotatedString(message.content)) }, modifier = Modifier.size(36.dp)) { Icon(Icons.Rounded.ContentCopy, contentDescription = "コピー", modifier = Modifier.size(17.dp)) }
                        IconButton(onClick = { onQuote(message.content) }, modifier = Modifier.size(36.dp)) { Icon(Icons.Rounded.FormatQuote, contentDescription = "引用", modifier = Modifier.size(18.dp)) }
                    }
                    if (persisted && user) IconButton(onClick = { onEdit(message) }, modifier = Modifier.size(36.dp)) { Icon(Icons.Rounded.Edit, contentDescription = "編集", modifier = Modifier.size(18.dp)) }
                    if (persisted && !user) IconButton(onClick = { onRegenerate(message) }, enabled = message.parentId != null, modifier = Modifier.size(36.dp)) { Icon(Icons.Rounded.Replay, contentDescription = "再生成", modifier = Modifier.size(18.dp)) }
                }
            }
        }
        }
    }
}

@Composable
fun StatusCardView(card: StatusCard) {
    Surface(color = MaterialTheme.colorScheme.surfaceContainerHigh, shape = RoundedCornerShape(12.dp), border = androidx.compose.foundation.BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant), modifier = Modifier.fillMaxWidth()) {
        Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                if (card.done) Icon(Icons.Rounded.CheckCircle, contentDescription = null, tint = MaterialTheme.colorScheme.primary, modifier = Modifier.size(16.dp))
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
