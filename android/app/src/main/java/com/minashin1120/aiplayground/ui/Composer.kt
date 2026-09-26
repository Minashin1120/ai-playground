package com.minashin1120.aiplayground.ui

import androidx.annotation.DrawableRes
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.interaction.collectIsFocusedAsState
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.selection.toggleable
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Shape
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.input.key.*
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.minashin1120.aiplayground.ChatState
import com.minashin1120.aiplayground.ChatViewModel
import com.minashin1120.aiplayground.R
import com.minashin1120.aiplayground.data.ComposerRules
import com.minashin1120.aiplayground.data.OptionRule
import com.minashin1120.aiplayground.data.SlashCommand
import com.minashin1120.aiplayground.data.SLASH_COMMANDS
import com.minashin1120.aiplayground.data.slashPaletteFilter
import com.minashin1120.aiplayground.data.stripSlashCommand
import com.minashin1120.aiplayground.data.visibleSlashCommands
import com.minashin1120.aiplayground.data.THINKING_LEVELS
import com.minashin1120.aiplayground.data.codingBarText
import com.minashin1120.aiplayground.data.isAudioPath
import com.minashin1120.aiplayground.data.isVideoPath
import com.minashin1120.aiplayground.data.composerRules
import com.minashin1120.aiplayground.data.gemMentionQuery
import com.minashin1120.aiplayground.data.historyCodingTargets
import com.minashin1120.aiplayground.data.isImageReference
import com.minashin1120.aiplayground.data.tokenEstimateLine
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.delay
import org.json.JSONObject

/*
 * The Web prompt bar (`templates/chat/composer*.html` with the V4.8.925 "prompt bar visual refresh"
 * styles of chat.custom.css). Phones use the `max-width: 640px` sizes of the stylesheet.
 */


@Composable
fun Composer(
    state: ChatState,
    model: ChatViewModel,
    pickModel: () -> Unit,
    pickFiles: () -> Unit,
    onVoice: () -> Unit,
    onRichPaste: () -> Unit = {},
    onMask: () -> Unit = {},
    onSettings: () -> Unit = {},
    onRealtime: () -> Unit = {},
    onLyria: () -> Unit = {},
    realtimeOptions: RealtimeOptions = RealtimeOptions(),
    onRealtimeOptions: (RealtimeOptions) -> Unit = {},
    onRealtimeStart: () -> Unit = {},
    onChatInstructions: () -> Unit = {},
    onCompressionSettings: () -> Unit = {},
    onTemporarySettings: () -> Unit = {},
    loader: FileBytesLoader? = null,
    onOpenFile: (String) -> Unit = {},
) {
    val web = LocalWebPalette.current
    val reduce = LocalReduceMotion.current
    val phone = LocalConfiguration.current.screenWidthDp <= 640
    val promptMode = state.preferences?.effectivePromptBarMode ?: "normal"
    val minimal = promptMode == "minimal"
    val compact = promptMode == "compact"
    var expanded by rememberSaveable(promptMode) { mutableStateOf(false) }
    val showDetails = if (minimal) expanded else !compact || expanded
    val selectedModel = state.account?.models?.firstOrNull { it.id == state.model }
    val mcpServerOn = state.mcpServers.any { it.enabled }
    val rules = remember(state.model, mcpServerOn) { composerRules(state.model, mcpServerOn) }
    // Web slash palette: shown while the input starts with `/` and no command is pending.
    val pendingSlash = state.pendingSlashCommand
    val slashFilter = if (pendingSlash == null) slashPaletteFilter(state.draft) else null
    val slashMatches = slashFilter?.let { visibleSlashCommands(it, minimal) }.orEmpty()
    val runLocal: (String) -> Unit = { id ->
        when (id) {
            "options" -> expanded = true
            "attach" -> pickFiles()
            "voice" -> onVoice()
            "paste" -> onRichPaste()
        }
    }
    /** Web `selectSlashCommand`. */
    val selectSlash: (SlashCommand) -> Unit = { command ->
        val argument = stripSlashCommand(state.draft).trim()
        when {
            command.autocompleteArgument && argument.isEmpty() -> model.draft("${command.label} ")
            command.minimal && (!command.requiresArgument || argument.isNotEmpty()) -> {
                model.draft("")
                model.runSlashCommand(command, argument, runLocal)
            }
            else -> {
                model.draft(stripSlashCommand(state.draft))
                model.setPendingSlashCommand(command.id)
            }
        }
    }
    // Web `confirmGeminiLocalPythonSwitch`: asked before sending audio or video to Gemini with Python on.
    val context = LocalContext.current
    var localPythonConfirm by remember { mutableStateOf(false) }
    val guardedSend: () -> Unit = {
        val hasAudio = state.attachments.any { isAudioPath(it.reference) || isAudioPath(it.name) }
        val hasVideo = state.attachments.any { isVideoPath(it.reference) || isVideoPath(it.name) }
        val ask = context.getSharedPreferences("settings_local", 0).getBoolean(GEMINI_LOCAL_PY_DIALOG_PREF, true)
        if (state.model == "lyria-realtime-exp") {
            // Web: Lyria RealTime has no text generation; the text opens the studio instead.
            val text = state.draft
            model.draft("")
            model.prepareLyriaPrompt(text)
            onLyria()
        } else if (ask && isGeminiLocalPythonMode(state.model, hasAudio, hasVideo, state.enablePython)) localPythonConfirm = true
        else model.send()
    }
    if (localPythonConfirm) GeminiLocalPythonDialog { proceed, dontShow ->
        localPythonConfirm = false
        if (dontShow) context.getSharedPreferences("settings_local", 0).edit().putBoolean(GEMINI_LOCAL_PY_DIALOG_PREF, false).apply()
        if (proceed) model.send()
    }
    val sendOrSlash: () -> Unit = {
        val instruction = state.draft.trim()
        val settingsInline = Regex("^/settings(?:\\s|$)", RegexOption.IGNORE_CASE)
        when {
            // Web: Enter with the palette open picks the highlighted (first) command.
            slashMatches.isNotEmpty() && !settingsInline.containsMatchIn(instruction) -> selectSlash(slashMatches.first())
            pendingSlash == "settings" -> {
                if (instruction.isEmpty()) model.notify("設定変更の指示を入力してください（例: デフォルトモデルをgemini-2.5-flashに）")
                else model.runAiSettings(instruction)
            }
            pendingSlash != null -> {
                val command = SLASH_COMMANDS.firstOrNull { it.id == pendingSlash }
                if (command != null && model.runSlashCommand(command, instruction, runLocal)) {
                    model.draft("")
                    model.setPendingSlashCommand(null)
                }
            }
            settingsInline.containsMatchIn(instruction) -> {
                val text = instruction.replace(Regex("^/settings\\s*", RegexOption.IGNORE_CASE), "").trim()
                if (text.isEmpty()) {
                    model.notify("使い方: /settings デフォルトモデルを gemini-2.5-flash に変更して thinking をオンに")
                    model.draft("/settings ")
                } else model.runAiSettings(text)
            }
            else -> guardedSend()
        }
    }
    val history = remember(state.messages) { historyCodingTargets(state.messages) }

    Box(Modifier.fillMaxWidth().composerDock(web, minimal, phone)) {
        Column(
            Modifier
                .align(Alignment.TopCenter)
                .widthIn(max = 832.dp)
                .fillMaxWidth()
                .navigationBarsPadding()
                .padding(start = 8.8.dp, end = 8.8.dp, top = if (minimal) 7.2.dp else 8.dp, bottom = 12.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            val shownQuote = rememberRetained(state.quote.takeIf { it.isNotBlank() })
            AnimatedVisibility(state.quote.isNotBlank(), enter = expandFadeIn(reduce), exit = shrinkFadeOut(reduce)) {
                QuoteBar(shownQuote.orEmpty(), onClear = model::clearQuote)
            }
            AnimatedVisibility(state.codingMode, enter = expandFadeIn(reduce), exit = shrinkFadeOut(reduce)) {
                CodingTargetBar(
                    text = codingBarText(state.draft, history, state.codingTarget),
                    showClear = state.codingTarget != null,
                    onClear = { model.selectCodingTarget(null) },
                )
            }
            AnimatedVisibility(state.editingMessageId != null, enter = expandFadeIn(reduce), exit = shrinkFadeOut(reduce)) {
                EditBar(onCancel = model::cancelEdit)
            }
            // Minimal mode keeps only the model button (Web `#top-model-bar`); "+" opens the options.
            if (minimal) Box(Modifier.fillMaxWidth(), contentAlignment = Alignment.Center) { ModelButton(state, pickModel) }
            if (!minimal || expanded) ControlsRow(
                state = state, model = model, rules = rules, compact = compact, minimal = minimal,
                showDetails = showDetails, onToggleDetails = { expanded = !expanded },
                pickModel = pickModel, onChatInstructions = onChatInstructions,
                onCompressionSettings = onCompressionSettings, onTemporarySettings = onTemporarySettings,
            )
            AttachmentPreview(state, model, loader, onOpenFile, onEdit = pickFiles)
            // Web `#file-preview` while a recording is transcribed.
            if (state.micMode == "transcribing") TranscribingRow()
            AnimatedVisibility(state.imageMask != null, enter = expandFadeIn(reduce), exit = shrinkFadeOut(reduce)) {
                MaskPreview(state.imageMask.orEmpty().substringAfterLast('/'), onClear = { model.setImageMask(null) })
            }
            GenerationOptionsPanel(state.model, state.generationValues, !state.streaming, model::generationOption,
                onOpenLyriaStudio = onLyria, minimal = minimal)
            // Web voice dock: realtime audio models replace the text row with inline voice controls.
            val voiceDock = state.realtime.active ||
                (state.preferences?.voiceStudioUi != false && isRealtimeAudioModel(selectedModel))
            if (voiceDock) RealtimeVoiceDock(state, model, realtimeOptions, onRealtimeOptions,
                onStart = onRealtimeStart, onExpand = onRealtime)
            else Column {
                val gem = state.selectedGem
                val shownGem = rememberRetained(gem)
                AnimatedVisibility(gem != null && gem.fixedPrompts.isNotEmpty(), enter = expandFadeIn(reduce), exit = shrinkFadeOut(reduce)) {
                    FixedPromptsBar(shownGem?.fixedPrompts.orEmpty().map { it.name to it.content }) { content ->
                        model.draft(content); guardedSend()
                    }
                }
                AnimatedVisibility(gem != null, enter = expandFadeIn(reduce), exit = shrinkFadeOut(reduce)) {
                    GemIndicator(shownGem?.name.orEmpty(), onClear = { model.chooseGem(null) })
                }
                val shownSlash = rememberRetained(slashMatches.takeIf { it.isNotEmpty() })
                AnimatedVisibility(slashMatches.isNotEmpty(), enter = expandFadeIn(reduce), exit = shrinkFadeOut(reduce)) {
                    SuggestionPalette(
                        title = "コマンド",
                        items = shownSlash.orEmpty().map { PaletteItem(it.label, it.description, slashIcon(it.iconName), mono = true) },
                        onPick = { index -> shownSlash?.getOrNull(index)?.let(selectSlash) },
                    )
                }
                AnimatedVisibility(pendingSlash != null, enter = expandFadeIn(reduce), exit = shrinkFadeOut(reduce)) {
                    val shownPending = rememberRetained(pendingSlash)
                    SlashCommandIndicator(SLASH_COMMANDS.firstOrNull { it.id == shownPending }?.label ?: "/${shownPending.orEmpty()}") {
                        model.setPendingSlashCommand(null)
                    }
                }
                AnimatedVisibility(state.xLinkPrompt, enter = expandFadeIn(reduce), exit = shrinkFadeOut(reduce)) {
                    XLinkBanner(onResolve = model::resolveXLinkPrompt)
                }
                val mention = gemMentionQuery(state.draft)
                val mentionCandidates = if (mention != null) state.gems.filter {
                    it.name.contains(mention, ignoreCase = true) || it.description.contains(mention, ignoreCase = true)
                } else emptyList()
                val shownMention = rememberRetained(mention?.takeIf { mentionCandidates.isNotEmpty() }?.let { it to mentionCandidates })
                AnimatedVisibility(mentionCandidates.isNotEmpty(), enter = expandFadeIn(reduce), exit = shrinkFadeOut(reduce)) {
                    shownMention?.let { (query, candidates) ->
                        SuggestionPalette(
                            title = "Gem",
                            items = candidates.map { PaletteItem(it.name, it.description, R.drawable.fa_solid_gem) },
                            onPick = { index -> candidates.getOrNull(index)?.let { model.applyGemMention(it, query) } },
                        )
                    }
                }
                InputRow(state, model, rules, phone, minimal, pickFiles, onRichPaste, onMask, onVoice,
                    onPlus = { expanded = !expanded }, onSend = sendOrSlash)
                if (state.micMode == "preparing" || state.micMode == "recording") MicRecordingIndicator(state.micMode, state.micLevels, phone)
                TokenEstimate(state)
            }
        }
    }
}

/** `.composer-dock` surface with the theme hairline (`::before`) along its top edge. */
private fun Modifier.composerDock(web: WebPalette, minimal: Boolean, phone: Boolean): Modifier {
    if (minimal) return this
    val bg = if (web.isLight) Color.White.copy(alpha = 0.84f) else Color(6, 9, 18).copy(alpha = 0.58f)
    val border = if (web.isLight) Color(15, 23, 42).copy(alpha = 0.10f) else Color(226, 232, 240).copy(alpha = 0.08f)
    val glow = web.theme.rgb(0.62f * 0.8f)
    return background(bg).drawBehind {
        drawLine(border, Offset(0f, 0f), Offset(size.width, 0f), strokeWidth = 1.dp.toPx())
        val inset = (if (phone) 16.dp else 20.dp).toPx().coerceAtLeast(size.width / 2f - 416.dp.toPx())
        drawRect(
            Brush.horizontalGradient(listOf(Color.Transparent, glow, Color.Transparent), startX = inset, endX = size.width - inset),
            topLeft = Offset(inset, 0f),
            size = androidx.compose.ui.geometry.Size((size.width - inset * 2).coerceAtLeast(0f), 1.dp.toPx()),
        )
    }
}

/** `#quote-bar`: the quoted text with its left rule, and × to drop it. */
@Composable
private fun QuoteBar(text: String, onClear: () -> Unit) {
    val web = LocalWebPalette.current
    ComposerBar(
        background = if (web.isLight) Color.White.copy(alpha = 0.9f) else Color(8, 14, 28).copy(alpha = 0.9f),
        border = web.lineSoft,
    ) {
        FaIcon(R.drawable.fa_solid_quote_left, null, size = 11.dp, tint = web.theme300)
        Text(
            text.replace('\n', ' '),
            fontSize = 12.sp, fontStyle = FontStyle.Italic, maxLines = 1, overflow = TextOverflow.Ellipsis,
            color = if (web.isLight) web.text else Color(0xFFCBD5F5),
            modifier = Modifier.weight(1f)
                .drawBehind { drawRect(web.theme.t500, size = androidx.compose.ui.geometry.Size(3.dp.toPx(), size.height)) }
                .padding(start = 8.dp),
        )
        BarClose("引用を解除", onClear)
    }
}

/** `#coding-target-bar`. */
@Composable
private fun CodingTargetBar(text: String, showClear: Boolean, onClear: () -> Unit) {
    val web = LocalWebPalette.current
    ComposerBar(
        background = if (web.isLight) Color.White.copy(alpha = 0.9f) else Color(2, 44, 34).copy(alpha = 0.88f),
        border = web.lineSoft,
    ) {
        FaIcon(R.drawable.fa_solid_code, null, size = 11.dp, tint = if (web.isLight) Color(4, 120, 87) else Color(110, 231, 183))
        Text(text, fontSize = 12.sp, maxLines = 1, overflow = TextOverflow.Ellipsis,
            color = if (web.isLight) web.text else Color(209, 250, 229), modifier = Modifier.weight(1f))
        if (showClear) BarClose("明示的な選択を解除して最新コードを使う", onClear)
    }
}

/** `#edit-bar`. */
@Composable
private fun EditBar(onCancel: () -> Unit) {
    val web = LocalWebPalette.current
    ComposerBar(
        background = if (web.isLight) Color.White.copy(alpha = 0.9f) else Color(10, 16, 30).copy(alpha = 0.85f),
        border = if (web.isLight) Color(15, 23, 42).copy(alpha = 0.10f) else web.line,
    ) {
        FaIcon(R.drawable.fa_solid_pen, null, size = 11.dp, tint = if (web.isLight) Color(29, 78, 216) else web.theme300)
        Text("編集中: このメッセージを編集しています", fontSize = 11.sp, lineHeight = 16.sp, maxLines = 1, overflow = TextOverflow.Ellipsis,
            color = if (web.isLight) web.text else Color(209, 213, 219), modifier = Modifier.weight(1f))
        Text("キャンセル", fontSize = 11.sp, lineHeight = 16.sp,
            color = if (web.isLight) Color(91, 102, 117) else Color(156, 163, 175),
            modifier = Modifier.clickable(role = Role.Button, onClick = onCancel))
    }
}

@Composable
private fun ComposerBar(background: Color, border: Color, content: @Composable RowScope.() -> Unit) {
    val shape = RoundedCornerShape(12.dp)
    Row(
        Modifier.fillMaxWidth().clip(shape).background(background).border(1.dp, border, shape)
            .padding(horizontal = 11.2.dp, vertical = 6.4.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp),
        content = content,
    )
}

@Composable
private fun BarClose(label: String, onClick: () -> Unit) {
    val web = LocalWebPalette.current
    Box(Modifier.clickable(role = Role.Button, onClick = onClick).semantics { contentDescription = label }.padding(start = 8.dp)) {
        FaIcon(R.drawable.fa_solid_times, null, size = 11.dp, tint = if (web.isLight) Color(91, 102, 117) else Color(156, 163, 175))
    }
}

/** `#prompt-controls-row`: model button and mode chips, then the detail chips. */
@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun ControlsRow(
    state: ChatState,
    model: ChatViewModel,
    rules: ComposerRules,
    compact: Boolean,
    minimal: Boolean,
    showDetails: Boolean,
    onToggleDetails: () -> Unit,
    pickModel: () -> Unit,
    onChatInstructions: () -> Unit,
    onCompressionSettings: () -> Unit,
    onTemporarySettings: () -> Unit,
) {
    val web = LocalWebPalette.current
    val reduce = LocalReduceMotion.current
    val collapsed = compact && !showDetails
    val railShape = RoundedCornerShape(13.dp)
    Column(
        if (collapsed) Modifier.fillMaxWidth().clip(railShape)
            .background(
                if (web.isLight) SolidColor(Color(15, 23, 42).copy(alpha = 0.03f))
                else Brush.horizontalGradient(listOf(Color(14, 22, 41).copy(alpha = 0.72f), Color(9, 15, 29).copy(alpha = 0.52f))),
            )
            .border(1.dp, Color(148, 163, 184).copy(alpha = 0.13f), railShape)
            .padding(3.52.dp)
        else Modifier.fillMaxWidth(),
    ) {
        if (!minimal) FlowRow(horizontalArrangement = Arrangement.spacedBy(6.4.dp), verticalArrangement = Arrangement.spacedBy(6.4.dp)) {
            ModelButton(state, pickModel, Modifier.align(Alignment.CenterVertically))
            ModeChip("Canvas", state.canvasMode, ChipTone.Canvas, rules.canvas, model::toggleCanvas, Modifier.align(Alignment.CenterVertically))
            ModeChip("Coding", state.codingMode, ChipTone.Coding, rules.coding, model::toggleCoding, Modifier.align(Alignment.CenterVertically))
            if (rules.batch.visible) ModeChip("Batch", state.batchMode, ChipTone.Batch, rules.batch, model::toggleBatchMode,
                Modifier.align(Alignment.CenterVertically))
            if (compact) DetailsToggle(showDetails, onToggleDetails, Modifier.align(Alignment.CenterVertically))
        }
        AnimatedVisibility(showDetails, enter = expandFadeIn(reduce), exit = shrinkFadeOut(reduce)) {
            Column(
                Modifier.fillMaxWidth().padding(top = if (minimal) 0.dp else 8.48.dp)
                    .drawBehind {
                        if (!minimal) drawLine(Color(148, 163, 184).copy(alpha = 0.11f), Offset(0f, 0f), Offset(size.width, 0f), 1.dp.toPx())
                    }
                    .padding(top = if (minimal) 0.dp else 6.72.dp),
            ) {
                DetailChips(state, model, rules, onChatInstructions, onCompressionSettings, onTemporarySettings)
            }
        }
    }
}

/** `#model-selector-btn`. */
@Composable
private fun ModelButton(state: ChatState, onClick: () -> Unit, modifier: Modifier = Modifier) {
    val web = LocalWebPalette.current
    val name = state.account?.models?.firstOrNull { it.id == state.model }?.name ?: state.model
    Row(
        modifier
            .heightIn(min = 32.dp)
            .widthIn(max = 150.dp)
            .clip(CircleShape)
            .then(
                if (web.isLight) Modifier.background(Color.White)
                else Modifier.background(Brush.linearGradient(listOf(web.theme.rgb(0.14f), Color(13, 20, 38).copy(alpha = 0.78f))))
            )
            .border(1.dp, if (web.isLight) Color(15, 23, 42).copy(alpha = 0.12f) else web.theme.rgb(0.24f), CircleShape)
            .clickable(role = Role.Button, onClick = onClick)
            .padding(horizontal = 11.52.dp, vertical = 6.08.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        FaIcon(R.drawable.fa_solid_robot, null, size = 12.dp, tint = web.theme300)
        Text(name, fontSize = 12.sp, lineHeight = 16.sp, fontWeight = FontWeight.SemiBold, letterSpacing = 0.12.sp,
            color = if (web.isLight) web.text else Color.White, maxLines = 1, overflow = TextOverflow.Ellipsis,
            modifier = Modifier.weight(1f, fill = false))
        FaIcon(R.drawable.fa_solid_chevron_down, null, size = 10.dp, tint = if (web.isLight) Color(91, 102, 117) else Color(156, 163, 175))
    }
}

@Composable
private fun ModeChip(label: String, checked: Boolean, tone: ChipTone, rule: OptionRule, onToggle: () -> Unit, modifier: Modifier = Modifier) {
    WebCheckChip(label, checked, { onToggle() }, tone.accent, composerChipStyle(checked, tone), modifier,
        enabled = !rule.dimmed && !rule.disabled)
}

/** `#prompt-controls-toggle-btn` (compact mode only). */
@Composable
private fun DetailsToggle(expanded: Boolean, onClick: () -> Unit, modifier: Modifier = Modifier) {
    val web = LocalWebPalette.current
    val color = if (web.isLight) web.theme.t700 else web.theme200
    Row(
        modifier.heightIn(min = 30.4.dp).clip(CircleShape).background(web.theme.rgb(0.10f))
            .border(1.dp, web.theme.rgb(0.27f), CircleShape)
            .clickable(role = Role.Button, onClick = onClick)
            .padding(horizontal = 10.88.dp, vertical = 5.12.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(4.dp),
    ) {
        Text(if (expanded) "折りたたむ" else "詳細", fontSize = 10.sp, fontWeight = FontWeight.SemiBold, letterSpacing = 0.2.sp, color = color)
        FaIcon(if (expanded) R.drawable.fa_solid_chevron_up else R.drawable.fa_solid_chevron_down, null, size = 10.dp, tint = color)
    }
}

/** `#standard-chat-controls` and `#temporary-chat-controls`. */
@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun DetailChips(
    state: ChatState,
    model: ChatViewModel,
    rules: ComposerRules,
    onChatInstructions: () -> Unit,
    onCompressionSettings: () -> Unit,
    onTemporarySettings: () -> Unit,
) {
    val web = LocalWebPalette.current
    val light = web.isLight
    fun label(dark: Color, lightColor: Color) = if (light) lightColor else dark
    val gray = label(Color(156, 163, 175), Color(91, 102, 117))
    FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
        val center = Modifier.align(Alignment.CenterVertically)
        OptChip("Search", state.enableSearch, rules.search, model::toggleSearch, Tw.blue500, center)
        if (rules.urls.visible) OptChip("URLs", state.enableUrlContext, rules.urls, model::toggleUrlContext, Tw.indigo500, center)
        if (rules.maps.visible) OptChip("Maps", state.enableMaps, rules.maps, model::toggleMaps, Tw.emerald500, center)
        if (rules.python.visible) OptChip("Python", state.enablePython, rules.python, model::togglePython, Tw.yellow500, center,
            labelColor = label(Color(254, 240, 138), Color(133, 77, 14)))
        OptChip("File", state.enableFileCreation, rules.file, model::toggleFileCreation, Tw.orange500, center,
            labelColor = if (light) Color(154, 52, 18) else null)
        if (rules.mcp.visible) OptChip("MCP", state.enableMcp, rules.mcp, model::toggleMcp, Tw.cyan500, center,
            labelColor = label(Color(165, 243, 252), Color(14, 116, 144)))
        OptGroup(state.enableSystemPrompt, center, dimmed = rules.sysPrompt.dimmed) {
            GroupCheck("SysPrompt", state.enableSystemPrompt, rules.sysPrompt, model::toggleSystemPrompt,
                label(Color(134, 239, 172), Color(21, 128, 61)))
            GearButton("Chat Settings", onChatInstructions)
        }
        if (rules.thinking.visible) OptGroup(state.enableThinking, center, dimmed = rules.thinking.dimmed) {
            GroupCheck("Thinking", state.enableThinking, rules.thinking, model::toggleThinking, label(Color(216, 180, 254), Color(126, 34, 206)))
            val levels = THINKING_LEVELS.filter { it in rules.thinkingLevels }
            ChipSelect(
                value = state.chipValues["thinking_level"].orEmpty(),
                options = THINKING_LEVELS.filter { it in levels }.map { WebOption(it, THINKING_LABELS.getValue(it)) },
                enabled = levels.isNotEmpty(),
                onSelect = { model.generationOption("thinking_level", it) },
                display = THINKING_LABELS,
            )
            Text("Budget", fontSize = 10.sp, color = label(Color(216, 180, 254).copy(alpha = 0.8f), Color(126, 34, 206)))
            BudgetField(state.chipValues["thinking_budget"].orEmpty(), rules.budgetEnabled) { model.generationOption("thinking_budget", it) }
        }
        if (rules.effort.visible) OptGroup(false, center) {
            Text("Effort:", fontSize = 10.sp, color = gray)
            ChipSelect(
                value = state.chipValues["reasoning_effort"].orEmpty(),
                options = rules.effortOptions.map { WebOption(it, EFFORT_LABELS.getValue(it)) },
                onSelect = { model.generationOption("reasoning_effort", it) },
                display = EFFORT_LABELS,
            )
        }
        OptGroup(false, center) {
            Text("Safety:", fontSize = 10.sp, color = gray)
            ChipSelect(
                value = state.chipValues["safety_setting"].orEmpty(),
                options = listOf(WebOption("default", "Default"), WebOption("none", "None")),
                onSelect = { model.generationOption("safety_setting", it) },
            )
        }
        OptGroup(state.enablePromptCache, center, dimmed = rules.promptCache.dimmed) {
            GroupCheck("PromptCache", state.enablePromptCache, rules.promptCache, model::togglePromptCache,
                label(Color(94, 234, 212), Color(15, 118, 110)))
        }
        OptGroup(state.compression.enabled, center) {
            GroupCheck("Compress", state.compression.enabled, OptionRule(),
                { model.saveCompressionSettings(state.compression.copy(enabled = !state.compression.enabled)) }, gray)
            GearButton("Compression Settings", onCompressionSettings)
        }
        Row(center, verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(4.dp)) {
            val temporary = state.selected?.isTemporary == true || state.newThreadTemporary
            OptChip("一時チャット", temporary, OptionRule(), model::toggleTemporaryChat, Tw.amber500,
                labelColor = label(Color(252, 211, 77), Color(146, 64, 14)))
            TemporarySettingsButton(onTemporarySettings)
        }
    }
}

private val THINKING_LABELS = mapOf("minimal" to "Min", "low" to "Low", "medium" to "Mid", "high" to "High")
private val EFFORT_LABELS = mapOf("none" to "None", "low" to "Low", "medium" to "Med", "high" to "High", "xhigh" to "XHigh", "max" to "Max")

/** `<label class="composer-opt">` checkbox chip. */
@Composable
private fun OptChip(
    label: String,
    checked: Boolean,
    rule: OptionRule,
    onToggle: () -> Unit,
    accent: Color,
    modifier: Modifier = Modifier,
    labelColor: Color? = null,
) {
    WebCheckChip(label, checked, { onToggle() }, accent, composerOptStyle(checked), modifier,
        labelColor = labelColor, enabled = !rule.dimmed && !rule.disabled)
}

/** `<div class="composer-opt composer-opt-group">`: a round pill that holds a checkbox label and extra controls. */
@Composable
private fun OptGroup(checked: Boolean, modifier: Modifier = Modifier, dimmed: Boolean = false, content: @Composable RowScope.() -> Unit) {
    WebPill(composerOptStyle(checked, group = true), modifier, enabled = !dimmed, content = content)
}

@Composable
private fun GroupCheck(label: String, checked: Boolean, rule: OptionRule, onToggle: () -> Unit, labelColor: Color) {
    Row(
        Modifier.toggleable(checked, enabled = !rule.dimmed && !rule.disabled, role = Role.Checkbox) { onToggle() },
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(4.dp),
    ) {
        WebCheckbox(checked, onCheckedChange = null)
        Text(label, fontSize = 10.sp, color = labelColor, maxLines = 1)
    }
}

/** The small `fa-cog` button inside SysPrompt / Compress. */
@Composable
private fun GearButton(label: String, onClick: () -> Unit) {
    val web = LocalWebPalette.current
    Box(
        Modifier.clip(RoundedCornerShape(12.dp)).clickable(role = Role.Button, onClick = onClick)
            .semantics { contentDescription = label }.padding(4.dp),
    ) {
        FaIcon(R.drawable.fa_solid_cog, null, size = 10.dp, tint = if (web.isLight) Color(105, 117, 136) else Color(107, 114, 128))
    }
}

/** The amber "設定" button next to 一時チャット. */
@Composable
private fun TemporarySettingsButton(onClick: () -> Unit) {
    val web = LocalWebPalette.current
    val shape = RoundedCornerShape(12.dp)
    Box(
        Modifier.heightIn(min = 22.dp).clip(shape)
            .background(Brush.verticalGradient(listOf(Color.White.copy(alpha = 0.04f), Color.Black.copy(alpha = 0.06f))))
            .border(1.dp, Color.White.copy(alpha = 0.06f), shape)
            .clickable(role = Role.Button, onClick = onClick)
            .padding(horizontal = 6.dp, vertical = 2.dp),
        contentAlignment = Alignment.Center,
    ) {
        Text("設定", fontSize = 10.sp, lineHeight = 16.sp, color = if (web.isLight) Color(146, 64, 14) else Color(252, 211, 77))
    }
}

/** A `<select>` inside a detail chip (`#thinking-level`, `#reasoning-effort`, `#safety-setting`). */
@Composable
private fun ChipSelect(
    value: String,
    options: List<WebOption>,
    onSelect: (String) -> Unit,
    enabled: Boolean = true,
    display: Map<String, String> = emptyMap(),
) {
    val web = LocalWebPalette.current
    val shown = if (options.any { it.value == value }) options else options + WebOption(value, display[value] ?: value)
    WebSelect(
        value = value, options = shown, onSelect = onSelect, enabled = enabled, fontSize = 10.sp,
        background = if (web.isLight) Color.White else Color(8, 14, 28).copy(alpha = 0.7f),
        borderColor = if (web.isLight) Color(15, 23, 42).copy(alpha = 0.14f) else Color(148, 163, 184).copy(alpha = 0.16f),
        textColor = if (web.isLight) web.text else Color(226, 232, 240),
        shape = RoundedCornerShape(6.dp),
        contentPadding = PaddingValues(horizontal = 4.dp, vertical = 4.dp),
    )
}

/** `#thinking-budget` (`type=number`, 0–32768). */
@Composable
private fun BudgetField(value: String, enabled: Boolean, onChange: (String) -> Unit) {
    val web = LocalWebPalette.current
    val shape = RoundedCornerShape(6.dp)
    val color = if (web.isLight) web.text else Color(226, 232, 240)
    BasicTextField(
        value = value,
        onValueChange = { next -> if (next.length <= 5 && next.all(Char::isDigit)) onChange(next) },
        enabled = enabled,
        singleLine = true,
        textStyle = TextStyle(fontSize = 10.sp, color = color, fontFamily = WebFonts.sans),
        cursorBrush = SolidColor(color),
        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
        modifier = Modifier.width(80.dp).alpha(if (enabled) 1f else 0.5f).clip(shape)
            .background(if (web.isLight) Color.White else Color(8, 14, 28).copy(alpha = 0.7f))
            .border(1.dp, if (web.isLight) Color(15, 23, 42).copy(alpha = 0.14f) else Color(148, 163, 184).copy(alpha = 0.16f), shape)
            .padding(horizontal = 4.dp, vertical = 4.dp)
            .semantics { contentDescription = "Thinking Budget" },
    )
}

/** `#file-preview`: thumbnails, then "N files ready" / "Preparing... (x/y)" with the clear button. */
@Composable
private fun AttachmentPreview(state: ChatState, model: ChatViewModel, loader: FileBytesLoader?, onOpen: (String) -> Unit, onEdit: () -> Unit) {
    val web = LocalWebPalette.current
    val reduce = LocalReduceMotion.current
    val visible = state.attachments.isNotEmpty() || state.uploading
    AnimatedVisibility(visible, enter = expandFadeIn(reduce), exit = shrinkFadeOut(reduce)) {
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            if (state.attachments.isNotEmpty()) Row(
                Modifier.horizontalScroll(rememberScrollState()).padding(start = 4.dp),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                state.attachments.forEach { attachment ->
                    key(attachment.reference) {
                        AppearOnce(enter = ::popIn) {
                            val shape = RoundedCornerShape(8.dp)
                            val thumb = Modifier.size(48.dp).clip(shape).background(Tw.gray800)
                                .border(1.dp, Color.White.copy(alpha = 0.16f), shape)
                            if (isImageReference(attachment.name) || attachment.mime.startsWith("image/")) {
                                ProtectedImage(attachment.reference, loader, onOpen = onOpen, modifier = thumb,
                                    shape = shape, contentScale = ContentScale.Crop, contentDescription = attachment.name)
                            } else Box(thumb, contentAlignment = Alignment.Center) {
                                Text("FILE", fontSize = 9.sp, fontWeight = FontWeight.Bold, color = Tw.gray500)
                            }
                        }
                    }
                }
            }
            val shape = RoundedCornerShape(4.dp)
            Row(
                Modifier.fillMaxWidth().clip(shape)
                    .background(if (web.isLight) Color(15, 23, 42).copy(alpha = 0.04f) else Tw.gray700.copy(alpha = 0.5f))
                    .clickable(role = Role.Button, onClick = onEdit)
                    .padding(horizontal = 12.dp, vertical = 8.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                val text = if (state.uploading) "Preparing... (${state.uploadCompleted}/${state.uploadCount})"
                    else "${state.attachments.size} files ready"
                Text(text, fontSize = 12.sp, color = web.theme300, maxLines = 1, overflow = TextOverflow.Ellipsis, modifier = Modifier.weight(1f))
                Text("✕", fontSize = 12.sp, color = Tw.gray400,
                    modifier = Modifier.padding(start = 8.dp).clickable(role = Role.Button) {
                        if (state.uploading) model.cancelUpload() else model.clearAttachments()
                    })
            }
            if (state.uploading) {
                val fraction = if (state.uploadCount > 0) {
                    val current = if (state.uploadTotal > 0) (state.uploadSent.toFloat() / state.uploadTotal).coerceIn(0f, 1f) else 0f
                    ((state.uploadCompleted + current) / state.uploadCount).coerceIn(0f, 1f)
                } else 0f
                val shown by animateFloatAsState(fraction, motionTween(reduce, 300), label = "upload total")
                Box(Modifier.fillMaxWidth().height(4.dp).clip(CircleShape).background(Tw.gray800)) {
                    Box(Modifier.fillMaxHeight().fillMaxWidth(shown).background(Tw.blue500))
                }
            }
        }
    }
}

/** `#mask-preview`. */
@Composable
private fun MaskPreview(name: String, onClear: () -> Unit) {
    val shape = RoundedCornerShape(4.dp)
    Row(
        Modifier.fillMaxWidth().clip(shape).background(Tw.gray700.copy(alpha = 0.4f)).padding(horizontal = 12.dp, vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text("Mask: $name", fontSize = 10.sp, color = Tw.amber200, maxLines = 1, modifier = Modifier.weight(1f))
        Text("✕", fontSize = 10.sp, color = Tw.gray400, modifier = Modifier.padding(start = 8.dp).clickable(role = Role.Button, onClick = onClear))
    }
}

/** `#fixed-prompts-bar`: tapping a prompt fills the input and sends it. */
@Composable
private fun FixedPromptsBar(prompts: List<Pair<String, String>>, onSend: (String) -> Unit) {
    val web = LocalWebPalette.current
    Row(
        Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()).padding(bottom = 12.dp),
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        prompts.forEach { (name, content) ->
            Box(
                Modifier.clip(CircleShape)
                    .background(if (web.isLight) Color(0xFFDBE3EE) else Tw.gray700)
                    .border(1.dp, if (web.isLight) Color(0xFFCDD7E4) else Tw.gray600.copy(alpha = 0.5f), CircleShape)
                    .clickable(role = Role.Button) { onSend(content) }
                    .padding(horizontal = 11.2.dp, vertical = 4.48.dp),
            ) {
                Text(name, fontSize = 11.sp, fontWeight = FontWeight.Bold, maxLines = 1,
                    color = if (web.isLight) Color(0xFF1B2436) else Color(0xFFF3F4F6))
            }
        }
    }
}

/** `#gem-active-indicator`. */
@Composable
private fun GemIndicator(name: String, onClear: () -> Unit) {
    val web = LocalWebPalette.current
    val shape = RoundedCornerShape(8.dp)
    val color = if (web.isLight) Color(29, 78, 216) else web.theme200
    Row(
        Modifier.fillMaxWidth().padding(bottom = 8.dp).clip(shape)
            .background(if (web.isLight) Color(59, 130, 246).copy(alpha = 0.12f) else Color(30, 58, 138).copy(alpha = 0.4f))
            .border(1.dp, if (web.isLight) Color(59, 130, 246).copy(alpha = 0.25f) else Color(29, 78, 216).copy(alpha = 0.5f), shape)
            .padding(8.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        FaIcon(R.drawable.fa_solid_gem, null, size = 12.dp, tint = color)
        Text("Using Gem:", fontSize = 12.sp, color = color)
        Text(name, fontSize = 12.sp, fontWeight = FontWeight.Bold, color = color, maxLines = 1, overflow = TextOverflow.Ellipsis,
            modifier = Modifier.weight(1f))
        Text("×", fontSize = 12.sp, color = color,
            modifier = Modifier.clickable(role = Role.Button, onClick = onClear).padding(horizontal = 8.dp))
    }
}

private data class PaletteItem(val title: String, val description: String, @DrawableRes val icon: Int?, val mono: Boolean = false)

/** `#slash-command-suggestions` / `#gem-suggestions`: header row, then icon + title + one-line description. */
@Composable
private fun SuggestionPalette(title: String, items: List<PaletteItem>, onPick: (Int) -> Unit) {
    val web = LocalWebPalette.current
    val shape = RoundedCornerShape(8.dp)
    Column(
        Modifier.fillMaxWidth().padding(bottom = 4.dp).clip(shape)
            .background(if (web.isLight) Color.White.copy(alpha = 0.9f) else Tw.gray800)
            .border(1.dp, if (web.isLight) Color(15, 23, 42).copy(alpha = 0.10f) else Tw.gray700, shape),
    ) {
        Row(
            Modifier.fillMaxWidth()
                .drawBehind { drawLine(if (web.isLight) web.line else Tw.gray700, Offset(0f, size.height), Offset(size.width, size.height), 1.dp.toPx()) }
                .padding(horizontal = 12.dp, vertical = 6.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Text(title, fontSize = 10.sp, color = if (web.isLight) Color(105, 117, 136) else Tw.gray500)
            Spacer(Modifier.weight(1f))
            Text("↑↓ 選択 / Enter 確定 / Esc 閉じる", fontSize = 10.sp, color = if (web.isLight) Color(105, 117, 136) else Tw.gray400)
        }
        Column(Modifier.heightIn(max = 260.dp).verticalScroll(rememberScrollState()).padding(vertical = 4.dp)) {
            items.forEachIndexed { index, item ->
                Row(
                    Modifier.fillMaxWidth()
                        .background(if (index == 0) (if (web.isLight) Color(231, 237, 246) else Tw.gray700) else Color.Transparent)
                        .clickable { onPick(index) }
                        .padding(horizontal = 12.dp, vertical = 8.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(12.dp),
                ) {
                    Box(Modifier.width(16.dp), contentAlignment = Alignment.Center) {
                        item.icon?.let { FaIcon(it, null, size = 14.dp, tint = web.theme300) }
                    }
                    Column(Modifier.weight(1f)) {
                        Text(item.title, fontSize = 14.sp, lineHeight = 20.sp, maxLines = 1,
                            fontFamily = if (item.mono) FontFamily.Monospace else WebFonts.sans,
                            color = if (web.isLight) Color(29, 78, 216) else web.theme300)
                        if (item.description.isNotBlank()) Text(item.description, fontSize = 11.sp, lineHeight = 20.sp, maxLines = 1,
                            overflow = TextOverflow.Ellipsis, color = if (web.isLight) Color(91, 102, 117) else Tw.gray400)
                    }
                }
            }
        }
    }
}

/** `#input-row`: tool buttons, the textarea and the send / stop button inside `.composer-input-shell`. */
@Composable
private fun InputRow(
    state: ChatState,
    model: ChatViewModel,
    rules: ComposerRules,
    phone: Boolean,
    minimal: Boolean,
    pickFiles: () -> Unit,
    onRichPaste: () -> Unit,
    onMask: () -> Unit,
    onVoice: () -> Unit,
    onPlus: () -> Unit,
    onSend: () -> Unit,
) {
    val web = LocalWebPalette.current
    val reduce = LocalReduceMotion.current
    val interaction = remember { MutableInteractionSource() }
    val focused by interaction.collectIsFocusedAsState()
    val shellShape = RoundedCornerShape(if (phone) 17.dp else 20.dp)
    val shellBorder by animateColorAsState(
        when {
            focused -> web.theme.rgb(0.62f)
            web.isLight -> Color(15, 23, 42).copy(alpha = 0.10f)
            else -> Color(148, 163, 184).copy(alpha = 0.18f)
        },
        motionTween(reduce, 200), label = "composer shell border",
    )
    val toolSize = if (phone) 34.88.dp else 38.08.dp
    val toolShape = RoundedCornerShape(if (phone) 11.dp else 13.dp)
    val uploadActive = state.uploading
    Row(
        Modifier.fillMaxWidth().padding(top = 1.28.dp).heightIn(min = 46.4.dp).clip(shellShape)
            .then(
                if (web.isLight) Modifier.background(Color.White)
                else Modifier.background(Brush.linearGradient(listOf(Color(21, 31, 55).copy(alpha = 0.82f), Color(7, 12, 25).copy(alpha = 0.93f))))
            )
            .border(1.dp, shellBorder, shellShape)
            .padding(4.48.dp),
        verticalAlignment = Alignment.Bottom,
        horizontalArrangement = Arrangement.spacedBy(4.dp),
    ) {
        ToolButton(
            if (minimal) R.drawable.fa_solid_plus else R.drawable.fa_solid_paperclip,
            if (minimal) "オプション" else "Upload", toolSize, toolShape,
            enabled = !state.offline,
            onClick = if (minimal) onPlus else pickFiles,
        )
        if (!minimal) {
            ToolButton(R.drawable.fa_solid_paste, "リッチ貼り付け → PDF化", toolSize, toolShape, onClick = onRichPaste,
                tint = Color(0xFFFBBF24), plain = true)
            if (rules.mask) ToolButton(R.drawable.fa_solid_mask, "Mask (GPT-Image)", toolSize, toolShape, onClick = onMask,
                enabled = !uploadActive, tint = Color(0xFFA9B5C7))
            ToolButton(R.drawable.fa_solid_microphone, "Voice Input", toolSize, toolShape, onClick = onVoice, enabled = !uploadActive,
                active = state.micMode == "recording")
        }
        PromptField(state, model, focused, interaction, onSend, Modifier.weight(1f))
        SendStopButton(state, phone, onSend, onStop = model::stop, enabled = !uploadActive)
    }
}

@Composable
private fun ToolButton(
    @DrawableRes icon: Int,
    label: String,
    size: Dp,
    shape: Shape,
    onClick: () -> Unit,
    enabled: Boolean = true,
    tint: Color? = null,
    plain: Boolean = false,
    /** Recording: `bg-red-600 animate-pulse` like the Web mic button. */
    active: Boolean = false,
) {
    val web = LocalWebPalette.current
    val reduce = LocalReduceMotion.current
    val pulse = if (active && !reduce) rememberInfiniteTransition(label = "mic pulse").animateFloat(
        1f, 0.5f, infiniteRepeatable(tween(1000), RepeatMode.Reverse), label = "mic alpha",
    ).value else 1f
    val background = when {
        active -> Tw.red600
        plain -> Color.Transparent
        web.isLight -> Color.White.copy(alpha = 0.92f)
        else -> Color(13, 21, 40).copy(alpha = 0.6f)
    }
    val border = when {
        plain -> Color.Transparent
        web.isLight -> Color(0xFFD9E1EC)
        else -> web.line
    }
    Box(
        Modifier.size(size).alpha(if (enabled) pulse else 0.5f).clip(shape).background(background).border(1.dp, if (active) Tw.red600 else border, shape)
            .clickable(enabled = enabled, role = Role.Button, onClick = onClick)
            .semantics { contentDescription = label },
        contentAlignment = Alignment.Center,
    ) {
        FaIcon(icon, null, size = 14.dp, tint = if (active) Color.White else tint ?: if (web.isLight) web.text else Color(226, 232, 240))
    }
}

/** `#prompt-input`: grows to 150px, then scrolls. */
@Composable
private fun PromptField(
    state: ChatState,
    model: ChatViewModel,
    focused: Boolean,
    interaction: MutableInteractionSource,
    onSend: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val web = LocalWebPalette.current
    val enterToSend = state.preferences?.enterToSend == true
    val pendingCommand = state.pendingSlashCommand?.let { id -> SLASH_COMMANDS.firstOrNull { it.id == id } }
    val placeholder = when {
        pendingCommand != null -> pendingCommand.argumentHint.ifBlank { "設定変更の指示を入力（例: デフォルトモデルをgemini-2.5-flashに変更）..." }
        state.editingMessageId != null -> "編集中... (Enter送信は設定に従います)"
        enterToSend -> "Enter で送信 (Shift+Enter で改行)"
        else -> "Ctrl + Enter で送信..."
    }
    val textColor = if (web.isLight) web.text else Color(0xFFF4F7FB)
    val style = TextStyle(fontSize = 14.sp, lineHeight = 20.3.sp, letterSpacing = 0.07.sp, color = textColor, fontFamily = WebFonts.sans)
    BasicTextField(
        value = state.draft,
        onValueChange = model::draft,
        textStyle = style,
        cursorBrush = SolidColor(textColor),
        interactionSource = interaction,
        keyboardOptions = KeyboardOptions(imeAction = if (enterToSend) ImeAction.Send else ImeAction.Default),
        keyboardActions = KeyboardActions(onSend = { onSend() }),
        modifier = modifier
            .heightIn(min = 37.6.dp, max = 150.dp)
            .onPreviewKeyEvent { event ->
                val sendKey = event.key == Key.Enter && !event.isShiftPressed && (event.isCtrlPressed || enterToSend)
                if (sendKey) {
                    if (event.type == KeyEventType.KeyDown) onSend()
                    true
                } else false
            }
            .semantics { contentDescription = "メッセージ入力" },
        decorationBox = { inner ->
            Box(Modifier.padding(horizontal = 6.72.dp, vertical = 7.2.dp), contentAlignment = Alignment.CenterStart) {
                if (state.draft.isEmpty()) Text(placeholder, style = style.copy(color = if (web.isLight) web.muted else Color(0xFF7F8DA4)),
                    maxLines = 1, overflow = TextOverflow.Ellipsis)
                inner()
            }
        },
    )
}

/** `#send-btn` and its stop mode (`■`), swapped with the Web `btn-swap-pop`. */
@Composable
private fun SendStopButton(state: ChatState, phone: Boolean, onSend: () -> Unit, onStop: () -> Unit, enabled: Boolean) {
    val web = LocalWebPalette.current
    val reduce = LocalReduceMotion.current
    val buttonSize = if (phone) 36.48.dp else 40.8.dp
    val shape = RoundedCornerShape(if (phone) 13.dp else 15.dp)
    AnimatedContent(
        targetState = state.streaming,
        transitionSpec = { popIn(reduce) togetherWith popOut(reduce) },
        label = "send stop swap",
    ) { streaming ->
        if (streaming) Box(
            Modifier.size(buttonSize).clip(shape)
                .background(Brush.linearGradient(listOf(Color(0xFFEF4444), Color(0xFFDC2626))))
                .border(1.dp, Color(239, 68, 68).copy(alpha = 0.45f), shape)
                .clickable(role = Role.Button, onClick = onStop)
                .semantics { contentDescription = "生成を停止" },
            contentAlignment = Alignment.Center,
        ) {
            Text("■", fontSize = 20.sp, lineHeight = 20.sp, color = Color.White)
        } else Box(
            Modifier.size(buttonSize).alpha(if (enabled) 1f else 0.5f).clip(shape)
                .background(Brush.linearGradient(listOf(web.theme.t500, web.theme.t700)))
                .drawBehind {
                    // `radial-gradient(circle at 32% 18%, rgba(255,255,255,.26), transparent 30%)`
                    val center = Offset(this.size.width * 0.32f, this.size.height * 0.18f)
                    val farthest = kotlin.math.hypot(this.size.width * 0.68f, this.size.height * 0.82f)
                    drawRect(Brush.radialGradient(listOf(Color.White.copy(alpha = 0.26f), Color.Transparent),
                        center = center, radius = farthest * 0.3f))
                }
                .border(1.dp, web.theme.rgb(0.48f), shape)
                .clickable(enabled = enabled, role = Role.Button, onClick = onSend)
                .semantics { contentDescription = "送信" },
            contentAlignment = Alignment.Center,
        ) {
            FaIcon(R.drawable.fa_solid_paper_plane, null, size = 14.dp, tint = Color.White)
        }
    }
}

/** `#prompt-token-estimate`, refreshed 300 ms after the input settles (Web `schedulePromptTokenEstimate`). */
@Composable
private fun TokenEstimate(state: ChatState) {
    val web = LocalWebPalette.current
    val model = state.model
    val message = state.draft
    val quote = state.quote
    val files = state.attachments.map { it.reference }
    val hasInput = message.isNotBlank() || quote.isNotBlank() || files.isNotEmpty()
    var pending by remember { mutableStateOf(false) }
    var result by remember { mutableStateOf<JSONObject?>(null) }
    val viewModel = LocalComposerEstimator.current
    LaunchedEffect(model, message, quote, files) {
        if (!hasInput || viewModel == null) { pending = false; result = null; return@LaunchedEffect }
        delay(300)
        pending = true
        result = try { viewModel(model, message, quote, files) }
            catch (e: CancellationException) { throw e }
            catch (e: Exception) { null }
        pending = false
    }
    val line = tokenEstimateLine(hasInput && viewModel != null, pending, result) ?: return
    val color = when (line.tone) {
        "error" -> Tw.red300
        "count" -> if (web.isLight) Color(14, 116, 144) else Tw.cyan300
        else -> if (web.isLight) Color(105, 117, 136) else Tw.gray500
    }
    Text(line.text, fontSize = 10.sp, lineHeight = 20.sp, color = color, maxLines = 2,
        modifier = Modifier.padding(top = 4.dp, start = 5.6.dp, end = 4.dp).alpha(0.85f))
}

/** Supplies the token-estimate request so previews and tests can render the composer without a network. */
internal val LocalComposerEstimator = staticCompositionLocalOf<(suspend (String, String, String, List<String>) -> JSONObject)?> { null }

/** Web slash-command icons (`fa-*`); `fa-file-lines` and `fa-plug` are not in the Web icon subset, so no glyph. */
@DrawableRes
private fun slashIcon(name: String): Int? = when (name) {
    "cog" -> R.drawable.fa_solid_cog
    "plus" -> R.drawable.fa_solid_plus
    "paperclip" -> R.drawable.fa_solid_paperclip
    "microphone" -> R.drawable.fa_solid_microphone
    "paste" -> R.drawable.fa_solid_paste
    "window-restore" -> R.drawable.fa_solid_window_restore
    "code-branch" -> R.drawable.fa_solid_code_branch
    "search" -> R.drawable.fa_solid_search
    "link" -> R.drawable.fa_solid_link
    "map-location-dot" -> R.drawable.fa_solid_map_location_dot
    "code" -> R.drawable.fa_solid_code
    "terminal" -> R.drawable.fa_solid_terminal
    "brain" -> R.drawable.fa_solid_brain
    "sliders-h" -> R.drawable.fa_solid_sliders_h
    "shield-halved" -> R.drawable.fa_solid_shield_halved
    "database" -> R.drawable.fa_solid_database
    "compress-alt" -> R.drawable.fa_solid_compress_alt
    "hourglass-half" -> R.drawable.fa_solid_hourglass_half
    else -> null
}

/** `#slash-command-indicator`: コマンドモード: /x with the × cancel. */
@Composable
private fun SlashCommandIndicator(label: String, onCancel: () -> Unit) {
    val web = LocalWebPalette.current
    val shape = RoundedCornerShape(8.dp)
    Row(
        Modifier.fillMaxWidth().padding(bottom = 8.dp).clip(shape).background(web.twBg(Tw.purple900, 0.4f))
            .border(1.dp, web.twBorder(Tw.purple700, 0.5f), shape).padding(6.dp),
        verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        val color = web.twText(Tw.purple200)
        FaIcon(R.drawable.fa_solid_terminal, null, size = 12.dp, tint = color)
        Text("コマンドモード:", fontSize = 12.sp, lineHeight = 16.sp, color = color)
        Text(label, fontSize = 12.sp, lineHeight = 16.sp, fontWeight = FontWeight.Bold, fontFamily = FontFamily.Monospace, color = color,
            modifier = Modifier.weight(1f))
        Text("×", fontSize = 12.sp, color = web.twText(Tw.purple300),
            modifier = Modifier.clip(RoundedCornerShape(4.dp)).clickable(onClickLabel = "キャンセル", role = Role.Button, onClick = onCancel)
                .padding(horizontal = 8.dp))
    }
}

/** `#auto-search-banner`: X link detected; continue with search or answer without it. */
@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun XLinkBanner(onResolve: (Boolean, Boolean) -> Unit) {
    val web = LocalWebPalette.current
    var remember by remember { mutableStateOf(false) }
    val shape = RoundedCornerShape(4.dp)
    val text = web.twText(Tw.yellow200)
    Column(
        Modifier.fillMaxWidth().padding(bottom = 8.dp).clip(shape).background(web.twBg(Tw.yellow900, 0.3f))
            .border(1.dp, web.twBorder(Tw.yellow600, 0.6f), shape).padding(horizontal = 12.dp, vertical = 8.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            FaIcon(R.drawable.fa_solid_bolt, null, size = 12.dp, tint = text)
            Text("Xリンクを検出しました。検索ON＋Grok 4 Fast Reasoningに切替します。", fontSize = 12.sp, lineHeight = 16.sp, color = text)
        }
        FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalArrangement = Arrangement.spacedBy(6.dp),
            itemVerticalAlignment = Alignment.CenterVertically) {
            Row(
                Modifier.toggleable(remember, role = Role.Checkbox) { remember = it },
                verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(4.dp),
            ) {
                WebCheckbox(remember, null, size = 12.dp)
                Text("次回から尋ねない", fontSize = 10.sp, color = web.twText(Tw.yellow100).copy(alpha = 0.9f))
            }
            Text("検索ONで続行", fontSize = 11.sp, fontWeight = FontWeight.Bold, color = Color.Black,
                modifier = Modifier.clip(shape).background(Tw.yellow600).clickable(role = Role.Button) { onResolve(true, remember) }
                    .padding(horizontal = 8.dp, vertical = 4.dp))
            Text("検索OFFで回答", fontSize = 11.sp, color = Color.White,
                modifier = Modifier.clip(shape).background(web.twBg(Tw.gray700)).clickable(role = Role.Button) { onResolve(false, false) }
                    .padding(horizontal = 8.dp, vertical = 4.dp))
        }
    }
}

/** `#mic-recording-indicator`: 録音準備中… / 録音中… with the 24-bar `#mic-waveform`. */
@Composable
private fun MicRecordingIndicator(mode: String, levels: List<Float>, phone: Boolean) {
    val color = if (mode == "recording") Color(252, 165, 165) else Color(253, 224, 71)
    Row(Modifier.padding(start = 4.dp, end = 4.dp, top = 4.dp), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        Text(if (mode == "recording") "録音中…" else "録音準備中…", fontSize = 11.sp, lineHeight = 16.sp, color = color)
        val shape = RoundedCornerShape(4.dp)
        Row(
            Modifier.height(16.dp).width(if (phone) 112.dp else 144.dp).clip(shape).background(Color(69, 10, 10).copy(alpha = 0.25f))
                .border(1.dp, Color(185, 28, 28).copy(alpha = 0.3f), shape).padding(horizontal = 4.dp, vertical = 2.dp),
            verticalAlignment = Alignment.Bottom, horizontalArrangement = Arrangement.spacedBy(2.dp),
        ) {
            val padded = List(24 - levels.size.coerceAtMost(24)) { 0f } + levels.takeLast(24)
            padded.forEach { level ->
                Box(Modifier.width(2.dp).height((2 + level * 10).dp).alpha(0.35f + level * 0.65f).clip(CircleShape)
                    .background(Color(252, 165, 165).copy(alpha = 0.92f)))
            }
        }
    }
}

/** `#file-preview` row showing "Transcribing...". */
@Composable
private fun TranscribingRow() {
    val web = LocalWebPalette.current
    Text("Transcribing...", fontSize = 12.sp, lineHeight = 16.sp, color = web.twText(Tw.blue300), maxLines = 1,
        modifier = Modifier.fillMaxWidth().clip(RoundedCornerShape(4.dp)).background(web.twBg(Tw.gray700, 0.5f)).padding(horizontal = 12.dp, vertical = 8.dp))
}
