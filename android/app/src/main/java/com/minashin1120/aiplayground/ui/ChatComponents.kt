package com.minashin1120.aiplayground.ui

import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.Crossfade
import androidx.compose.animation.EnterTransition
import androidx.compose.animation.ExitTransition
import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.slideOutVertically
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
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
import androidx.compose.ui.graphics.graphicsLayer
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
import com.minashin1120.aiplayground.data.SlashCommand
import com.minashin1120.aiplayground.data.matchingSlashCommands
import com.minashin1120.aiplayground.data.numericId
import com.minashin1120.aiplayground.data.parseSlashAction

@OptIn(ExperimentalLayoutApi::class)
@Composable
fun Composer(state: ChatState, model: ChatViewModel, pickModel: () -> Unit, pickFiles: () -> Unit, onVoice: () -> Unit,
             onRichPaste: () -> Unit = {}, onMask: () -> Unit = {}, onSettings: () -> Unit = {},
             onRealtime: () -> Unit = {}, onLyria: () -> Unit = {}) {
    val promptMode = state.preferences?.effectivePromptBarMode ?: "normal"
    var details by remember(promptMode) { mutableStateOf(promptMode == "normal") }
    val selectedModel = state.account?.models?.firstOrNull { it.id == state.model }
    val colors = MaterialTheme.colorScheme
    val reduce = LocalReduceMotion.current
    val slashMatches = matchingSlashCommands(state.draft)
    val runSlash: (SlashCommand) -> Unit = { command ->
        when (command.id) {
            "settings" -> { model.draft(""); onSettings() }
            "options" -> { model.draft(""); details = true }
            "attach" -> { model.draft(""); pickFiles() }
            "voice" -> { model.draft(""); onVoice() }
            "paste" -> { model.draft(""); onRichPaste() }
            "realtime" -> { model.draft(""); onRealtime() }
            "lyria" -> { model.draft(""); onLyria() }
            else -> parseSlashAction(if (state.draft.startsWith(command.label)) state.draft else command.label)?.let(model::applySlash)
                ?: model.draft(command.label + " ")
        }
    }
    val sendOrSlash: () -> Unit = {
        val action = parseSlashAction(state.draft)
        val command = slashMatches.firstOrNull { it.id == action?.id } ?: slashMatches.singleOrNull()
        if (command != null && (action != null || command.id in listOf("settings", "options", "attach", "voice", "paste", "realtime", "lyria"))) {
            if (action != null && command.id !in listOf("settings", "options", "attach", "voice", "paste", "realtime", "lyria")) model.applySlash(action)
            else runSlash(command)
        } else model.send()
    }
    Surface(color = colors.surface.copy(alpha = 0.96f), shadowElevation = 8.dp) {
        Box(Modifier.fillMaxWidth(), contentAlignment = Alignment.Center) {
        Column(
            Modifier.widthIn(max = PlaygroundDimens.contentMax).fillMaxWidth().navigationBarsPadding().padding(horizontal = 12.dp, vertical = 9.dp),
            verticalArrangement = Arrangement.spacedBy(7.dp),
        ) {
            AnimatedVisibility(state.editingMessageId != null, enter = expandFadeIn(reduce), exit = shrinkFadeOut(reduce)) {
                Surface(shape = RoundedCornerShape(12.dp), color = colors.secondary.copy(alpha = 0.10f), border = androidx.compose.foundation.BorderStroke(1.dp, colors.secondary.copy(alpha = 0.25f))) {
                    Row(Modifier.fillMaxWidth().padding(start = 10.dp), verticalAlignment = Alignment.CenterVertically) {
                    Icon(Icons.Rounded.Edit, contentDescription = null, tint = colors.secondary, modifier = Modifier.size(16.dp))
                    Text("メッセージを編集中（送信で分岐を作成）", style = MaterialTheme.typography.labelMedium,
                        color = colors.onSurfaceVariant, modifier = Modifier.padding(start = 7.dp).weight(1f))
                    TextButton(onClick = model::cancelEdit) { Text("キャンセル", style = MaterialTheme.typography.labelMedium) }
                    }
                }
            }
            if (promptMode != "minimal") Row(verticalAlignment = Alignment.CenterVertically) {
                Surface(onClick = pickModel, enabled = !state.streaming, color = colors.surfaceContainerHigh, shape = RoundedCornerShape(50), border = androidx.compose.foundation.BorderStroke(1.dp, colors.outlineVariant), modifier = Modifier.weight(1f)) {
                    Row(Modifier.padding(horizontal = 11.dp, vertical = 7.dp), verticalAlignment = Alignment.CenterVertically) {
                        Icon(Icons.Rounded.SmartToy, contentDescription = null, tint = colors.primary, modifier = Modifier.size(16.dp))
                        Text(selectedModel?.name ?: state.model.ifBlank { "モデルを選択" }, style = MaterialTheme.typography.labelLarge, fontWeight = FontWeight.SemiBold, maxLines = 1, overflow = TextOverflow.Ellipsis, modifier = Modifier.weight(1f))
                        Icon(Icons.Rounded.KeyboardArrowDown, contentDescription = null, modifier = Modifier.size(18.dp))
                    }
                }
                TextButton(onClick = { details = !details }) { Text(if (details) "詳細を閉じる" else "詳細") }
            }
            if (promptMode == "minimal") {
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    FilterChip(details, { details = !details }, { Text("＋") })
                    Text(selectedModel?.name ?: state.model.ifBlank { "モデル" }, style = MaterialTheme.typography.labelMedium,
                        maxLines = 1, overflow = TextOverflow.Ellipsis, modifier = Modifier.weight(1f))
                    TextButton(onClick = pickModel, enabled = !state.streaming) { Text("変更") }
                }
            }
            AnimatedVisibility(
                visible = details,
                enter = expandFadeIn(LocalReduceMotion.current),
                exit = shrinkFadeOut(LocalReduceMotion.current),
            ) {
                selectedModel?.let { info ->
                    Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                        if (info.supports("thinking")) FilterChip(state.enableThinking, model::toggleThinking, { Text("Thinking") })
                        if (info.supports("search")) FilterChip(state.enableSearch, model::toggleSearch, { Text("Web検索") })
                        if (info.supports("url_context") || info.id.startsWith("gemini-")) FilterChip(state.enableUrlContext, model::toggleUrlContext, { Text("URLs") })
                        if (info.supports("maps") || info.id.startsWith("gemini-3")) FilterChip(state.enableMaps, model::toggleMaps, { Text("Maps") })
                        if (info.mode == "chat" || info.mode == "agent") FilterChip(state.enableFileCreation, model::toggleFileCreation, { Text("File") })
                        if (info.mode == "chat" || info.mode == "agent") FilterChip(state.enableSystemPrompt, model::toggleSystemPrompt, { Text("SysPrompt") })
                        if (info.supports("prompt_cache")) FilterChip(state.enablePromptCache, model::togglePromptCache, { Text("Prompt Cache") })
                        if (info.supports("batch")) FilterChip(state.batchMode, model::toggleBatchMode, { Text("Batch") })
                        if (info.supports("python")) FilterChip(state.enablePython, model::togglePython, { Text("Python") })
                        if (info.supports("mcp")) FilterChip(state.enableMcp, model::toggleMcp, { Text("MCP") })
                        if (info.mode == "chat" || info.mode == "agent") FilterChip(state.canvasMode, model::toggleCanvas, { Text("Canvas") })
                        if (info.mode == "chat" || info.mode == "agent") FilterChip(state.codingMode, model::toggleCoding, { Text("Coding") })
                        FilterChip(state.selected?.isTemporary == true || state.newThreadTemporary, model::toggleTemporaryChat, { Text("一時チャット") })
                        FilterChip(state.compression.enabled, { model.saveCompressionSettings(state.compression.copy(enabled = !state.compression.enabled)) }, { Text("Compress") })
                    }
                    GenerationOptionsPanel(info, state.generationValues[info.id].orEmpty(), !state.streaming, model::generationOption)
                    if (state.codingMode) CodingTargetPanel(state.allMessages, state.codingTarget, model)
                }
            }
            val shownGem = rememberRetained(state.selectedGem)
            AnimatedVisibility(state.selectedGem != null, enter = expandFadeIn(reduce), exit = shrinkFadeOut(reduce)) {
                shownGem?.let { gem ->
                Column(verticalArrangement = Arrangement.spacedBy(7.dp)) {
                InputChip(selected = true, onClick = { model.chooseGem(null) }, label = { Text("Gem: ${gem.name.take(20)} ×") })
                if (gem.fixedPrompts.isNotEmpty()) Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    gem.fixedPrompts.forEach { prompt ->
                        SuggestionChip(onClick = { model.draft(prompt.content); model.send() },
                            enabled = !state.offline && !state.streaming && !state.busy && !state.uploading,
                            label = { Text(prompt.name) })
                    }
                }
                }
                }
            }
            AnimatedVisibility(state.imageMask != null, enter = expandFadeIn(reduce), exit = shrinkFadeOut(reduce)) {
                InputChip(selected = true, onClick = { model.setImageMask(null) },
                    label = { Text("🎭 マスク適用中 ×") })
            }
            val shownSlash = rememberRetained(slashMatches.takeIf { it.isNotEmpty() })
            AnimatedVisibility(slashMatches.isNotEmpty(), enter = expandFadeIn(reduce), exit = shrinkFadeOut(reduce)) {
                FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    shownSlash.orEmpty().take(8).forEach { command ->
                        SuggestionChip(onClick = { runSlash(command) }, label = { Text("${command.label}  ${command.description}") })
                    }
                }
            }
            val gemMention = gemMentionQuery(state.draft)
            val mentionCandidates = if (gemMention != null && state.gems.isNotEmpty()) {
                state.gems.filter { it.name.contains(gemMention, ignoreCase = true) }.take(6)
            } else emptyList()
            val shownMention = rememberRetained(gemMention?.takeIf { mentionCandidates.isNotEmpty() }?.let { it to mentionCandidates })
            AnimatedVisibility(mentionCandidates.isNotEmpty(), enter = expandFadeIn(reduce), exit = shrinkFadeOut(reduce)) {
                shownMention?.let { (query, candidates) ->
                    FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                        candidates.forEach { gem ->
                            InputChip(selected = false, onClick = { model.applyGemMention(gem, query) },
                                label = { Text("@${gem.name.take(20)}") })
                        }
                    }
                }
            }
            AnimatedVisibility(state.attachments.isNotEmpty(), enter = expandFadeIn(reduce), exit = shrinkFadeOut(reduce)) {
                FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    state.attachments.forEach { attachment ->
                        key(attachment.reference) {
                            AppearOnce(enter = ::popIn) {
                                InputChip(selected = true, onClick = { model.removeAttachment(attachment.reference) },
                                    label = { Text("${attachmentKindIcon(attachmentKind(attachment.name, attachment.mime))} ${attachment.name.take(20)} ×") })
                            }
                        }
                    }
                }
            }
            AnimatedVisibility(state.uploading, enter = expandFadeIn(reduce), exit = shrinkFadeOut(reduce)) {
                val fraction = if (state.uploadTotal > 0) (state.uploadSent.toFloat() / state.uploadTotal).coerceIn(0f, 1f) else null
                val shownFraction by animateFloatAsState(fraction ?: 0f, motionTween(reduce, PlaygroundMotion.MEDIUM), label = "upload progress")
                Column(Modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text("アップロード中: ${state.uploadName.ifBlank { "添付" }.take(24)}",
                            style = MaterialTheme.typography.labelSmall, modifier = Modifier.weight(1f))
                        TextButton(onClick = model::cancelUpload) { Text("キャンセル", style = MaterialTheme.typography.labelMedium) }
                    }
                    if (fraction != null) {
                        LinearProgressIndicator(progress = { shownFraction }, modifier = Modifier.fillMaxWidth())
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
                        IconButton(onClick = pickFiles, enabled = !state.offline && !state.uploading && !state.streaming, modifier = Modifier.padding(bottom = 4.dp)) {
                            Icon(Icons.Rounded.AttachFile, contentDescription = "添付を追加")
                        }
                        IconButton(onClick = onRichPaste, enabled = !state.streaming, modifier = Modifier.padding(bottom = 4.dp)) {
                            Icon(Icons.Rounded.ContentPaste, contentDescription = "リッチ貼り付け")
                        }
                        if (selectedModel?.id?.startsWith("gpt-image") == true) {
                            IconButton(onClick = onMask, enabled = !state.streaming, modifier = Modifier.padding(bottom = 4.dp)) {
                                Icon(Icons.Rounded.Brush, contentDescription = "画像マスク")
                            }
                        }
                        IconButton(onClick = onVoice, enabled = !state.streaming, modifier = Modifier.padding(bottom = 4.dp)) {
                            Icon(Icons.Rounded.Mic, contentDescription = "音声入力")
                        }
                        OutlinedTextField(
                            state.draft, model::draft,
                            placeholder = { Text(if (state.preferences?.enterToSend == true) "Enterで送信…" else "Ctrl + Enterで送信…") },
                            minLines = 1, maxLines = 6,
                            modifier = Modifier.weight(1f).onPreviewKeyEvent { event ->
                                val sendKey = event.key == Key.Enter && !event.isShiftPressed &&
                                    (event.isCtrlPressed || state.preferences?.enterToSend == true)
                                if (sendKey) {
                                    if (event.type == KeyEventType.KeyDown) sendOrSlash()
                                    true
                                } else false
                            }, enabled = !state.streaming,
                            keyboardOptions = KeyboardOptions(imeAction = if (state.preferences?.enterToSend == true) ImeAction.Send else ImeAction.Default),
                            keyboardActions = KeyboardActions(onSend = { sendOrSlash() }),
                            colors = OutlinedTextFieldDefaults.colors(focusedBorderColor = Color.Transparent, unfocusedBorderColor = Color.Transparent, disabledBorderColor = Color.Transparent, focusedContainerColor = Color.Transparent, unfocusedContainerColor = Color.Transparent, disabledContainerColor = Color.Transparent),
                        )
                        val canSend = !state.offline && !state.busy && !state.uploading && (state.draft.isNotBlank() || state.attachments.isNotEmpty())
                        val sendContainer by animateColorAsState(if (canSend) colors.primary else colors.onSurface.copy(alpha = 0.12f),
                            motionTween(reduce, PlaygroundMotion.MEDIUM), label = "send container")
                        val sendContent by animateColorAsState(if (canSend) colors.onPrimary else colors.onSurface.copy(alpha = 0.38f),
                            motionTween(reduce, PlaygroundMotion.MEDIUM), label = "send content")
                        // Web `btn-swap-pop`: the send and stop buttons trade places with a small scale.
                        AnimatedContent(
                            targetState = state.streaming,
                            transitionSpec = { popIn(reduce) togetherWith popOut(reduce) },
                            modifier = Modifier.padding(bottom = 5.dp),
                            label = "send stop swap",
                        ) { streaming ->
                            if (streaming) FilledIconButton(onClick = model::stop, enabled = state.jobId != null, colors = IconButtonDefaults.filledIconButtonColors(containerColor = colors.error, contentColor = colors.onError), shape = RoundedCornerShape(12.dp)) {
                                Icon(Icons.Rounded.Stop, contentDescription = "生成を停止")
                            } else FilledIconButton(onClick = sendOrSlash, enabled = canSend, shape = RoundedCornerShape(12.dp),
                                colors = IconButtonDefaults.filledIconButtonColors(containerColor = sendContainer, contentColor = sendContent,
                                    disabledContainerColor = sendContainer, disabledContentColor = sendContent)) {
                                Icon(Icons.Rounded.Send, contentDescription = "送信")
                            }
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
                AnimatedVisibility(
                    visible = thoughtExpanded,
                    enter = expandFadeIn(LocalReduceMotion.current),
                    exit = shrinkFadeOut(LocalReduceMotion.current),
                ) {
                    SelectionContainer { Text(message.thought, style = MaterialTheme.typography.bodySmall, color = if (user) colors.onPrimary.copy(alpha = 0.78f) else colors.onSurfaceVariant) }
                }
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
                Crossfade(card.done, modifier = Modifier.size(16.dp), animationSpec = motionTween(LocalReduceMotion.current), label = "status card icon") { done ->
                    Box(contentAlignment = Alignment.Center) {
                        if (done) Icon(Icons.Rounded.CheckCircle, contentDescription = null, tint = MaterialTheme.colorScheme.primary, modifier = Modifier.size(16.dp))
                        else CircularProgressIndicator(Modifier.size(14.dp), strokeWidth = 2.dp)
                    }
                }
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
    val reduce = LocalReduceMotion.current
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        state.cards.forEach { card -> key(card.id + card.kind) { AppearOnce { StatusCardView(card) } } }
        state.mcpDecision?.let { decision -> key(decision.id) { AppearOnce {
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
        } } }
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            if (state.streaming) CircularProgressIndicator(Modifier.size(16.dp), strokeWidth = 2.dp)
            AnimatedContent(
                targetState = state.status.ifBlank { "受信した回答" },
                transitionSpec = {
                    if (reduce) EnterTransition.None togetherWith ExitTransition.None
                    else (slideInVertically(tween(PlaygroundMotion.MEDIUM, easing = PlaygroundMotion.Emphasized)) { it / 2 } +
                        fadeIn(tween(PlaygroundMotion.MEDIUM))) togetherWith
                        (slideOutVertically(tween(PlaygroundMotion.SHORT, easing = PlaygroundMotion.Exit)) { -it / 2 } +
                            fadeOut(tween(PlaygroundMotion.SHORT)))
                },
                label = "live status",
            ) { status -> Text(status, style = MaterialTheme.typography.labelMedium) }
        }
        val hasText = state.liveContent.isNotEmpty() || state.liveThought.isNotEmpty()
        AnimatedVisibility(state.streaming && !hasText, enter = fadeIn(motionTween(reduce)), exit = fadeOut(motionTween(reduce, PlaygroundMotion.SHORT))) {
            Surface(shape = RoundedCornerShape(topStart = 20.dp, topEnd = 20.dp, bottomStart = 5.dp, bottomEnd = 20.dp),
                color = MaterialTheme.colorScheme.surface.copy(alpha = 0.90f),
                border = androidx.compose.foundation.BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.72f))) {
                TypingDots(Modifier.padding(horizontal = 18.dp, vertical = 16.dp))
            }
        }
        AnimatedVisibility(hasText, enter = expandFadeIn(reduce), exit = ExitTransition.None) {
            MessageCard(ChatMessage("live", "assistant", state.liveContent, state.liveThought), onFile, onQuote, loader)
        }
    }
}

/**
 * Keeps conversation list keys stable when the stream's local rows are replaced by stored messages,
 * so the sent message and the streamed reply stay in place instead of fading out and back in.
 * Also remembers which keys are genuinely new so only those play the entry motion.
 */
internal class ConversationKeyTracker {
    private var lastMessages: List<ChatMessage>? = null
    private var lastLiveShown = false
    private var lastStreaming = false
    private val aliases = HashMap<String, String>()
    private val fresh = HashSet<String>()
    private var liveSerial = 0

    /** True once the streamed reply has been handed to a stored message during the current stream. */
    var liveConsumed = false
        private set

    val liveKey: String get() = "live-$liveSerial"

    private var resolved: Map<String, String> = emptyMap()
    private var resolvedList: List<String> = emptyList()

    /** Key the row was shown with in the list last passed to [update]. */
    fun keyOf(message: ChatMessage): String = resolved[message.id] ?: aliasOf(message)

    /** LazyColumn key for the row at [index] of the list last passed to [update]; unique within it. */
    fun keyAt(index: Int, message: ChatMessage): String = resolvedList.getOrNull(index) ?: aliasOf(message)

    private fun aliasOf(message: ChatMessage): String = aliases[message.id] ?: message.id

    /** Returns true only the first time a newly added key asks, so re-composition never replays the entry. */
    fun consumeFresh(key: String): Boolean = fresh.remove(key)

    fun update(messages: List<ChatMessage>, streaming: Boolean, liveVisible: Boolean) {
        if (streaming && !lastStreaming) liveConsumed = false
        lastStreaming = streaming
        val previous = lastMessages
        if (previous !== messages) {
            lastMessages = messages
            if (previous != null) carryOver(previous, messages)
        }
        val shown = liveVisible && !liveConsumed
        if (shown && !lastLiveShown && previous != null) fresh += liveKey
        lastLiveShown = shown
        resolvedList = uniqueKeys(messages)
        resolved = HashMap<String, String>(messages.size).also { map ->
            messages.forEachIndexed { index, message -> map.putIfAbsent(message.id, resolvedList[index]) }
        }
    }

    /**
     * A carried-over key may meet its original row again (a retried local message or a stale outgoing
     * snapshot), and a duplicate LazyColumn key crashes; colliding rows fall back to their own ids.
     */
    private fun uniqueKeys(messages: List<ChatMessage>): List<String> {
        // The list's own rows ("older", "welcome") and the streamed row share the same key space.
        val used = hashSetOf(liveKey, "older", "welcome")
        return messages.mapIndexed { index, message ->
            var key = aliasOf(message)
            if (key in used) key = message.id
            var attempt = 0
            while (key in used) key = "${message.id}#$index-${attempt++}"
            used += key
            key
        }
    }

    private fun carryOver(previous: List<ChatMessage>, next: List<ChatMessage>) {
        val previousIds = previous.mapTo(HashSet()) { it.id }
        val added = next.filter { it.id !in previousIds }
        if (added.isEmpty()) return
        val nextIds = next.mapTo(HashSet()) { it.id }
        val claimed = HashSet<String>()
        val removedLocal = previous.filter { it.role == "user" && it.id.startsWith("local-") && it.id !in nextIds }
        val addedUsers = added.filter { it.role == "user" }
        removedLocal.forEach { local ->
            val match = addedUsers.firstOrNull { it.id !in claimed && it.content == local.content }
                ?: addedUsers.singleOrNull()?.takeIf { removedLocal.size == 1 && it.id !in claimed }
                ?: return@forEach
            claimed += match.id
            aliases[match.id] = keyOf(local)
        }
        if (lastLiveShown) {
            added.lastOrNull { it.role == "assistant" }?.let { reply ->
                claimed += reply.id
                aliases[reply.id] = liveKey
                liveSerial += 1
                liveConsumed = true
                lastLiveShown = false
            }
        }
        added.forEach { if (it.id !in claimed) fresh += keyOf(it) }
    }
}

/** Three softly pulsing dots shown while the reply has not produced text yet. */
@Composable
internal fun TypingDots(modifier: Modifier = Modifier) {
    val reduce = LocalReduceMotion.current
    val color = MaterialTheme.colorScheme.primary
    val transition = rememberInfiniteTransition(label = "typing dots")
    val phase by transition.animateFloat(
        initialValue = 0f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(tween(1_050, easing = LinearEasing)),
        label = "typing phase",
    )
    Row(modifier, horizontalArrangement = Arrangement.spacedBy(5.dp), verticalAlignment = Alignment.CenterVertically) {
        repeat(3) { index ->
            val alpha = if (reduce) 0.7f else typingDotAlpha(phase, index)
            Box(Modifier.size(7.dp).graphicsLayer { this.alpha = alpha }.clip(CircleShape).background(color))
        }
    }
}

/** Each dot peaks a third of a cycle after the previous one. */
internal fun typingDotAlpha(phase: Float, index: Int): Float {
    val local = ((phase - index / 3f) % 1f + 1f) % 1f
    val wave = if (local < 0.5f) local * 2f else (1f - local) * 2f
    return 0.3f + 0.7f * wave
}
