package com.minashin1120.aiplayground.ui

import androidx.annotation.DrawableRes
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.MutableTransitionState
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.scaleIn
import androidx.compose.animation.scaleOut
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.slideOutVertically
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectVerticalDragGestures
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.TransformOrigin
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.IntRect
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.LayoutDirection
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.em
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import androidx.compose.ui.window.Popup
import androidx.compose.ui.window.PopupPositionProvider
import com.minashin1120.aiplayground.ChatState
import com.minashin1120.aiplayground.ChatViewModel
import com.minashin1120.aiplayground.R
import com.minashin1120.aiplayground.data.ComposerRules
import com.minashin1120.aiplayground.data.OptionRule
import com.minashin1120.aiplayground.data.THINKING_LEVELS
import kotlinx.coroutines.delay
import kotlin.math.abs

/** `#top-model-bar`: in minimal mode the model button sits under the header. */
@Composable
internal fun TopModelBar(state: ChatState, onPickModel: () -> Unit) {
    val web = LocalWebPalette.current
    Box(
        Modifier.fillMaxWidth()
            .background(if (web.isLight) Color.White.copy(alpha = 0.72f) else Color(7, 10, 20).copy(alpha = 0.72f))
            .drawBehind { drawRect(web.lineSoft, Offset(0f, size.height - 1.dp.toPx()), androidx.compose.ui.geometry.Size(size.width, 1.dp.toPx())) }
            .padding(horizontal = 12.dp, vertical = 5.6.dp),
        contentAlignment = Alignment.Center,
    ) { ModelButton(state, onPickModel, maxWidth = 280.dp) }
}

/** What a Web `MINIMAL_POPUP_ITEMS` row does. */
private sealed interface MinimalAction {
    data class Toggle(val rule: OptionRule?, val checked: Boolean, val toggle: () -> Unit) : MinimalAction
    data class Run(val run: () -> Unit, val upload: Boolean = false) : MinimalAction
    data class Select(val value: String, val options: List<WebOption>, val onSelect: (String) -> Unit) : MinimalAction
    data class Thinking(val rule: OptionRule, val checked: Boolean) : MinimalAction
}

private data class MinimalItem(
    val key: String,
    @DrawableRes val icon: Int?,
    val label: String,
    val action: MinimalAction,
    val gear: (() -> Unit)? = null,
)

/**
 * Web `#minimal-options-popup`: the ＋ button's option rows (two columns, "ファイルを添付" across) and the
 * "モデル設定" section with the generation panels. "高速" is left out (ANDROID_ONLY.md).
 */
@Composable
internal fun MinimalOptionsPopup(
    state: ChatState,
    model: ChatViewModel,
    rules: ComposerRules,
    bottomInset: Dp,
    onDismiss: () -> Unit,
    onAttach: () -> Unit,
    onVoice: () -> Unit,
    onRichPaste: () -> Unit,
    onShowThinkingSlider: () -> Unit,
    onChatInstructions: () -> Unit,
    onCompressionSettings: () -> Unit,
    onTemporarySettings: () -> Unit,
    onLyria: () -> Unit,
) {
    val web = LocalWebPalette.current
    val reduce = LocalReduceMotion.current
    val visible = remember { MutableTransitionState(false).apply { targetState = true } }
    var pending by remember { mutableStateOf<(() -> Unit)?>(null) }
    val close: (() -> Unit) -> Unit = { then -> pending = then; visible.targetState = false }
    LaunchedEffect(visible.currentState, visible.targetState) {
        if (!visible.targetState && !visible.currentState) {
            onDismiss()
            pending?.invoke()
        }
    }
    fun on(value: Boolean, rule: OptionRule) = rule.forced ?: value
    val temporary = state.selected?.isTemporary ?: state.newThreadTemporary
    val items = buildList {
        add(MinimalItem("attach", R.drawable.fa_solid_paperclip, "ファイルを添付", MinimalAction.Run(onAttach, upload = true)))
        add(MinimalItem("voice-input", R.drawable.fa_solid_microphone, "Voice Input", MinimalAction.Run(onVoice)))
        add(MinimalItem("rich-paste", R.drawable.fa_solid_paste, "リッチ貼り付け", MinimalAction.Run(onRichPaste)))
        if (rules.canvas.visible) add(MinimalItem("canvas", R.drawable.fa_solid_window_restore, "Canvas",
            MinimalAction.Toggle(rules.canvas, on(state.canvasMode, rules.canvas), model::toggleCanvas)))
        if (rules.coding.visible) add(MinimalItem("coding", R.drawable.fa_solid_code_branch, "Coding",
            MinimalAction.Toggle(rules.coding, on(state.codingMode, rules.coding), model::toggleCoding)))
        if (rules.batch.visible) add(MinimalItem("batch", R.drawable.fa_solid_layer_group, "Batch",
            MinimalAction.Toggle(rules.batch, on(state.batchMode, rules.batch), model::toggleBatchMode)))
        if (rules.search.visible) add(MinimalItem("search", R.drawable.fa_solid_search, "Search",
            MinimalAction.Toggle(rules.search, on(state.enableSearch, rules.search), model::toggleSearch)))
        if (rules.urls.visible) add(MinimalItem("urls", R.drawable.fa_solid_link, "URLs",
            MinimalAction.Toggle(rules.urls, on(state.enableUrlContext, rules.urls), model::toggleUrlContext)))
        if (rules.maps.visible) add(MinimalItem("maps", R.drawable.fa_solid_map_location_dot, "Maps",
            MinimalAction.Toggle(rules.maps, on(state.enableMaps, rules.maps), model::toggleMaps)))
        if (rules.python.visible) add(MinimalItem("python", R.drawable.fa_solid_code, "Python",
            MinimalAction.Toggle(rules.python, on(state.enablePython, rules.python), model::togglePython)))
        if (rules.file.visible) add(MinimalItem("file", slashIcon("file-lines"), "File",
            MinimalAction.Toggle(rules.file, on(state.enableFileCreation, rules.file), model::toggleFileCreation)))
        if (rules.mcp.visible) add(MinimalItem("mcp", slashIcon("plug"), "MCP",
            MinimalAction.Toggle(rules.mcp, on(state.enableMcp, rules.mcp), model::toggleMcp)))
        if (rules.sysPrompt.visible) add(MinimalItem("sysprompt", R.drawable.fa_solid_terminal, "SysPrompt",
            MinimalAction.Toggle(rules.sysPrompt, on(state.enableSystemPrompt, rules.sysPrompt), model::toggleSystemPrompt),
            gear = onChatInstructions))
        if (rules.thinking.visible) add(MinimalItem("thinking", R.drawable.fa_solid_brain, "Thinking",
            MinimalAction.Thinking(rules.thinking, on(state.enableThinking, rules.thinking))))
        if (rules.effort.visible) add(MinimalItem("effort", R.drawable.fa_solid_sliders_h, "Effort", MinimalAction.Select(
            state.chipValues["reasoning_effort"].orEmpty(), rules.effortOptions.map { WebOption(it, EFFORT_LABELS.getValue(it)) },
        ) { model.generationOption("reasoning_effort", it) }))
        add(MinimalItem("safety", R.drawable.fa_solid_shield_halved, "Safety", MinimalAction.Select(
            state.chipValues["safety_setting"].orEmpty(), listOf(WebOption("default", "Default"), WebOption("none", "None")),
        ) { model.generationOption("safety_setting", it) }))
        if (rules.promptCache.visible) add(MinimalItem("promptcache", R.drawable.fa_solid_database, "PromptCache",
            MinimalAction.Toggle(rules.promptCache, on(state.enablePromptCache, rules.promptCache), model::togglePromptCache)))
        add(MinimalItem("compress", R.drawable.fa_solid_compress_alt, "Compress", MinimalAction.Toggle(null, state.compression.enabled) {
            model.saveCompressionSettings(state.compression.copy(enabled = !state.compression.enabled))
        }, gear = onCompressionSettings))
        add(MinimalItem("tempchat", R.drawable.fa_solid_hourglass_half, "一時チャット",
            MinimalAction.Toggle(null, temporary, model::toggleTemporaryChat), gear = onTemporarySettings))
    }
    Dialog(onDismissRequest = { close {} }, properties = DialogProperties(usePlatformDefaultWidth = false, decorFitsSystemWindows = false)) {
        WebModalWindow(2.dp)
        Box(Modifier.fillMaxSize()) {
            AnimatedVisibility(visible.targetState, enter = fadeIn(tween(if (reduce) 0 else 180)), exit = fadeOut(tween(if (reduce) 0 else 180))) {
                Box(Modifier.fillMaxSize().background(Color(2, 5, 12).copy(alpha = 0.45f))
                    .clickable(interactionSource = remember { MutableInteractionSource() }, indication = null) { close {} })
            }
            val maxPanel = (androidx.compose.ui.platform.LocalConfiguration.current.screenHeightDp.dp * 0.66f).coerceAtMost(544.dp)
            AnimatedVisibility(
                visible,
                modifier = Modifier.align(Alignment.BottomCenter).padding(start = 12.dp, end = 12.dp, bottom = bottomInset + 11.2.dp),
                enter = fadeIn(tween(if (reduce) 0 else 180)) + scaleIn(tween(if (reduce) 0 else 300), initialScale = 0.82f,
                    transformOrigin = TransformOrigin(0.06f, 1.1f)) + slideInVertically(tween(if (reduce) 0 else 300)) { it / 20 },
                exit = fadeOut(tween(if (reduce) 0 else 180)) + scaleOut(tween(if (reduce) 0 else 300), targetScale = 0.82f,
                    transformOrigin = TransformOrigin(0.06f, 1.1f)) + slideOutVertically(tween(if (reduce) 0 else 300)) { it / 20 },
            ) {
                var dragY by remember { mutableFloatStateOf(0f) }
                val shape = RoundedCornerShape(18.dp)
                Column(
                    Modifier.widthIn(max = 416.dp).fillMaxWidth().heightIn(max = maxPanel)
                        .graphicsLayer { translationY = dragY }
                        .clip(shape)
                        .background(if (web.isLight) Color.White.copy(alpha = 0.96f) else Color(10, 16, 32).copy(alpha = 0.92f))
                        .border(1.dp, if (web.isLight) Color(15, 23, 42).copy(alpha = 0.12f) else Color(226, 232, 240).copy(alpha = 0.14f), shape)
                        // Web: a downward swipe from the top closes the panel like a bottom sheet.
                        .pointerInput(Unit) {
                            detectVerticalDragGestures(
                                onDragEnd = { if (dragY > 70.dp.toPx() * 0.6f) close {} else dragY = 0f },
                                onDragCancel = { dragY = 0f },
                            ) { change, amount ->
                                val next = (dragY + amount * 0.6f).coerceIn(0f, 140.dp.toPx())
                                if (next != dragY) { dragY = next; change.consume() }
                            }
                        },
                ) {
                    Row(
                        Modifier.fillMaxWidth()
                            .drawBehind { drawRect(Color(226, 232, 240).copy(alpha = 0.1f), Offset(0f, size.height - 1.dp.toPx()),
                                androidx.compose.ui.geometry.Size(size.width, 1.dp.toPx())) }
                            .padding(horizontal = 14.4.dp, vertical = 9.6.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        FaIcon(R.drawable.fa_solid_sliders_h, null, size = 11.5.dp, tint = web.theme300)
                        Text("オプション", fontSize = 11.5.sp, fontWeight = FontWeight.Bold, letterSpacing = 0.04.em, color = web.theme300,
                            modifier = Modifier.padding(start = 4.dp).weight(1f))
                        Box(Modifier.size(28.dp).clip(CircleShape).clickable(role = Role.Button) { close {} }, contentAlignment = Alignment.Center) {
                            FaIcon(R.drawable.fa_solid_times, "閉じる", size = 12.dp, tint = Color(0xFF94A3B8))
                        }
                    }
                    Column(Modifier.weight(1f, fill = false).verticalScroll(rememberScrollState())) {
                        Column(Modifier.padding(horizontal = 7.2.dp, vertical = 6.4.dp), verticalArrangement = Arrangement.spacedBy(4.8.dp)) {
                            val upload = items.first()
                            MinimalOptionRow(upload, state, Modifier.fillMaxWidth()) { close(onAttach) }
                            items.drop(1).chunked(2).forEach { pair ->
                                Row(horizontalArrangement = Arrangement.spacedBy(4.8.dp)) {
                                    pair.forEach { item ->
                                        MinimalOptionRow(item, state, Modifier.weight(1f), onGear = item.gear?.let { gear -> { close(gear) } }) {
                                            when (val action = item.action) {
                                                is MinimalAction.Run -> close(action.run)
                                                is MinimalAction.Toggle -> if (action.rule?.interactive != false) action.toggle()
                                                is MinimalAction.Thinking -> {
                                                    // Web: a forced checkbox still opens the slider so the level can change.
                                                    if (!action.rule.disabled && !action.rule.dimmed) {
                                                        model.toggleThinking()
                                                        if (!action.checked) close(onShowThinkingSlider)
                                                    } else close(onShowThinkingSlider)
                                                }
                                                is MinimalAction.Select -> Unit
                                            }
                                        }
                                    }
                                    if (pair.size == 1) Spacer(Modifier.weight(1f))
                                }
                            }
                        }
                        val hasModelPanels = com.minashin1120.aiplayground.data.generationPanels(state.model, state.generationValues)
                            .any { it.id in MINIMAL_MODEL_PANEL_IDS }
                        if (hasModelPanels) {
                            Column(Modifier.fillMaxWidth().drawBehind {
                                drawRect(Color(226, 232, 240).copy(alpha = 0.1f), Offset.Zero, androidx.compose.ui.geometry.Size(size.width, 1.dp.toPx()))
                            }) {
                                Text("モデル設定", fontSize = 10.9.sp, fontWeight = FontWeight.Bold, letterSpacing = 0.04.em, color = web.theme300,
                                    modifier = Modifier.padding(start = 14.4.dp, end = 14.4.dp, top = 7.2.dp, bottom = 4.8.dp))
                                Box(Modifier.padding(start = 9.6.dp, end = 9.6.dp, bottom = 9.6.dp)) {
                                    GenerationOptionsPanel(state.model, state.generationValues, !state.streaming, model::generationOption,
                                        onOpenLyriaStudio = { close(onLyria) }, popup = true)
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/** `.minimal-option-item` (`.on` / `.off` / `.disabled`, and `.action-upload` for the first row). */
@Composable
private fun MinimalOptionRow(item: MinimalItem, state: ChatState, modifier: Modifier, onGear: (() -> Unit)? = null, onClick: () -> Unit) {
    val web = LocalWebPalette.current
    val action = item.action
    val upload = action is MinimalAction.Run && action.upload
    val on = when (action) {
        is MinimalAction.Toggle -> action.checked
        is MinimalAction.Thinking -> action.checked
        else -> false
    }
    val disabled = when (action) {
        is MinimalAction.Toggle -> action.rule?.let { it.disabled || it.dimmed } ?: false
        is MinimalAction.Thinking -> action.rule.dimmed
        else -> false
    }
    val shape = RoundedCornerShape(12.dp)
    val blue = Color(0xFFBFDBFE)
    val fg = when {
        upload -> blue
        on -> web.theme300
        web.isLight -> web.text
        else -> Color(0xFFE2E8F0)
    }
    Row(
        modifier.alpha(if (disabled) 0.45f else 1f).clip(shape)
            .background(if (upload) Color(37, 99, 235).copy(alpha = 0.16f) else if (web.isLight) Color(15, 23, 42).copy(alpha = 0.03f) else Color.White.copy(alpha = 0.025f))
            .then(
                if (upload) Modifier.drawBehind {
                    drawRoundRect(Color(96, 165, 250).copy(alpha = 0.55f), cornerRadius = CornerRadius(12.dp.toPx()),
                        style = Stroke(1.dp.toPx(), pathEffect = PathEffect.dashPathEffect(floatArrayOf(4.dp.toPx(), 3.dp.toPx()))))
                } else Modifier.border(1.dp, if (web.isLight) Color(15, 23, 42).copy(alpha = 0.08f) else Color(226, 232, 240).copy(alpha = 0.09f), shape)
            )
            .clickable(enabled = !disabled, role = Role.Button, onClick = onClick)
            .padding(horizontal = if (upload) 12.dp else 10.4.dp, vertical = if (upload) 9.92.dp else 8.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = if (upload) Arrangement.spacedBy(8.8.dp, Alignment.CenterHorizontally) else Arrangement.spacedBy(7.2.dp),
    ) {
        Box(Modifier.width(16.dp), contentAlignment = Alignment.Center) {
            item.icon?.let { FaIcon(it, null, size = 13.dp, tint = if (upload) blue else if (on) web.theme.t400 else if (web.isLight) web.muted else Color(0xFF94A3B8)) }
        }
        Text(item.label, fontSize = 11.84.sp, lineHeight = 14.2.sp, fontWeight = FontWeight.SemiBold, color = fg,
            maxLines = 1, overflow = TextOverflow.Ellipsis, modifier = if (upload) Modifier else Modifier.weight(1f))
        if (action is MinimalAction.Thinking) {
            Text(THINKING_LABELS[state.chipValues["thinking_level"].orEmpty()] ?: "High", fontSize = 11.5.sp, fontWeight = FontWeight.Bold,
                color = Color(0xFFC4B5FD))
        }
        if (action is MinimalAction.Select) {
            WebSelect(
                action.value, action.options, action.onSelect, modifier = Modifier.widthIn(max = 88.dp),
                fontSize = 10.9.sp, background = if (web.isLight) Color.White else Color(8, 14, 28).copy(alpha = 0.8f),
                borderColor = Color(148, 163, 184).copy(alpha = 0.18f), textColor = if (web.isLight) web.text else Color(0xFFE2E8F0),
                shape = RoundedCornerShape(8.dp), contentPadding = PaddingValues(horizontal = 4.8.dp, vertical = 2.4.dp),
                contentDescription = item.label,
            )
        }
        if (onGear != null) {
            Box(Modifier.size(22.4.dp).clip(RoundedCornerShape(8.dp)).clickable(role = Role.Button, onClickLabel = "${item.label}設定", onClick = onGear),
                contentAlignment = Alignment.Center) { FaIcon(R.drawable.fa_solid_cog, "${item.label}設定", size = 11.dp, tint = Color(0xFF94A3B8)) }
        }
    }
}

/** Places a popup just above its anchor (the composer dock), centred horizontally. */
private class AboveAnchor(private val gap: Int) : PopupPositionProvider {
    override fun calculatePosition(anchorBounds: IntRect, windowSize: IntSize, layoutDirection: LayoutDirection, popupContentSize: IntSize) =
        IntOffset((windowSize.width - popupContentSize.width) / 2, anchorBounds.top - popupContentSize.height - gap)
}

/**
 * Web `#thinking-slide-bar`: a temporary purple bar above the composer that picks the Thinking level
 * (only levels the model allows), closes after 2.5s without input, with × or by swiping down.
 */
@Composable
internal fun ThinkingSlideBar(state: ChatState, model: ChatViewModel, rules: ComposerRules, onDismiss: () -> Unit) {
    val reduce = LocalReduceMotion.current
    val gap = with(LocalDensity.current) { 11.2.dp.roundToPx() }
    val visible = remember { MutableTransitionState(false).apply { targetState = true } }
    var touched by remember { mutableIntStateOf(0) }
    LaunchedEffect(touched) { delay(2500); visible.targetState = false }
    LaunchedEffect(visible.currentState, visible.targetState) { if (!visible.targetState && !visible.currentState) onDismiss() }
    val value = state.chipValues["thinking_level"].orEmpty()
    val index = THINKING_LEVELS.indexOf(value).takeIf { it >= 0 } ?: 3
    Popup(popupPositionProvider = remember(gap) { AboveAnchor(gap) }, onDismissRequest = { visible.targetState = false }) {
        AnimatedVisibility(
            visible,
            enter = fadeIn(tween(if (reduce) 0 else 300)) + slideInVertically(tween(if (reduce) 0 else 340)) { it / 3 },
            exit = fadeOut(tween(if (reduce) 0 else 300)) + slideOutVertically(tween(if (reduce) 0 else 340)) { it / 3 },
        ) {
            var dragY by remember { mutableFloatStateOf(0f) }
            val shape = RoundedCornerShape(16.dp)
            val screen = androidx.compose.ui.platform.LocalConfiguration.current.screenWidthDp.dp
            Column(
                Modifier.width(minOf(352.dp, screen - 32.dp))
                    .graphicsLayer { translationY = dragY }
                    .clip(shape).background(Color(24, 16, 44).copy(alpha = 0.94f))
                    .border(1.dp, Color(167, 139, 250).copy(alpha = 0.32f), shape)
                    .pointerInput(Unit) {
                        detectVerticalDragGestures(
                            onDragEnd = { if (dragY > 50.dp.toPx()) visible.targetState = false else { dragY = 0f; touched++ } },
                            onDragCancel = { dragY = 0f; touched++ },
                        ) { change, amount ->
                            val next = (dragY + amount * 0.5f).coerceIn(0f, 120.dp.toPx())
                            if (abs(next - dragY) > 0f) { dragY = next; change.consume() }
                        }
                    }
                    .padding(start = 13.6.dp, end = 13.6.dp, top = 11.2.dp, bottom = 9.6.dp),
            ) {
                Row(Modifier.padding(bottom = 5.6.dp), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(7.2.dp)) {
                    FaIcon(R.drawable.fa_solid_brain, null, size = 12.dp, tint = Color(0xFFD8B4FE))
                    Text("Thinking", fontSize = 11.84.sp, fontWeight = FontWeight.Bold, color = Color(0xFFE9D5FF))
                    Spacer(Modifier.weight(1f))
                    Text(THINKING_LABELS.getValue(THINKING_LEVELS[index]), fontSize = 11.5.sp, fontWeight = FontWeight.Bold, color = Color(0xFFC4B5FD))
                    Box(Modifier.size(24.dp).clip(CircleShape).clickable(role = Role.Button) { visible.targetState = false },
                        contentAlignment = Alignment.Center) { FaIcon(R.drawable.fa_solid_times, "閉じる", size = 11.dp, tint = Color(0xFFA78BFA)) }
                }
                val purple = Color(0xFFA78BFA)
                Slider(
                    index.toFloat(),
                    { raw ->
                        val target = raw.toInt().coerceIn(0, 3)
                        val allowed = THINKING_LEVELS.withIndex().filter { it.value in rules.thinkingLevels }.map { it.index }
                        if (allowed.isNotEmpty()) {
                            val pick = if (target in allowed) target else allowed.minBy { abs(it - target) }
                            if (THINKING_LEVELS[pick] != value) model.generationOption("thinking_level", THINKING_LEVELS[pick])
                        }
                        touched++
                    },
                    valueRange = 0f..3f, steps = 2,
                    colors = SliderDefaults.colors(thumbColor = purple, activeTrackColor = purple, inactiveTrackColor = Color(0xFF4C3A70),
                        activeTickColor = Color.Transparent, inactiveTickColor = Color.Transparent),
                    modifier = Modifier.fillMaxWidth().height(24.dp),
                )
                Row(Modifier.fillMaxWidth().padding(top = 2.4.dp), horizontalArrangement = Arrangement.SpaceBetween) {
                    listOf("Min", "Low", "Mid", "High").forEach { Text(it, fontSize = 9.6.sp, fontWeight = FontWeight.SemiBold, color = Color(0xFF8B7AA8)) }
                }
            }
        }
    }
}
