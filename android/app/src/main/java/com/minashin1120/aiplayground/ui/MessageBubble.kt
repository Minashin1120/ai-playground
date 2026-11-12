package com.minashin1120.aiplayground.ui

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.CubicBezierEasing
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.draw.drawWithContent
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.lerp
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.RectangleShape
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.graphics.drawscope.clipRect
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.em
import androidx.compose.ui.unit.sp
import androidx.compose.ui.zIndex
import com.minashin1120.aiplayground.R
import com.minashin1120.aiplayground.data.ChatMessage
import com.minashin1120.aiplayground.data.ProcessingTime
import com.minashin1120.aiplayground.data.TokenTotals
import com.minashin1120.aiplayground.data.isAudioPath
import com.minashin1120.aiplayground.data.isImageReference
import com.minashin1120.aiplayground.data.isVideoPath
import com.minashin1120.aiplayground.data.messageTokenLabel
import com.minashin1120.aiplayground.data.numericId
import kotlinx.coroutines.delay
import org.json.JSONObject

/** What the token-detail modal (`#token-detail-modal`) shows for a message or a conversation total. */
internal data class TokenDetail(
    val title: String,
    val total: Int?,
    val tokensIn: Int?,
    val tokensOut: Int?,
    val tokensContent: Int?,
    val tokensThought: Int?,
    val encrypted: Boolean?,
    val cost: com.minashin1120.aiplayground.data.CostEstimate? = null,
    val pricing: com.minashin1120.aiplayground.data.ModelCostEstimator? = null,
    val cacheRead: Int? = null,
    val cacheWrite: Int? = null,
    val cacheWriteTtls: Map<String, Int> = emptyMap(),
)

internal fun ChatMessage.tokenDetail(): TokenDetail {
    val total = tokens ?: if (tokensIn != null || tokensOut != null) (tokensIn ?: 0) + (tokensOut ?: 0) else null
    val cache = com.minashin1120.aiplayground.data.cacheUsage(thought)
    return TokenDetail(if (model.isNotBlank()) "$model ($role)" else role, total, tokensIn, tokensOut, tokensContent, tokensThought, encrypted,
        cacheRead = cache?.read, cacheWrite = cache?.write, cacheWriteTtls = cache?.ttls ?: emptyMap())
}

internal fun TokenTotals.detail(allBranches: Boolean, cost: com.minashin1120.aiplayground.data.CostEstimate? = null,
                               pricing: com.minashin1120.aiplayground.data.ModelCostEstimator? = null): TokenDetail = TokenDetail(
    if (allBranches) "Conversation (All branches) (total)" else "Conversation (total)",
    total, tokensIn, tokensOut, tokensContent, tokensThought, null, cost, pricing,
    cacheRead, cacheWrite, cacheWriteTtls,
)

/** Callbacks of the controls above a bubble (`.msg-controls`) and its footer. */
internal class MessageActions(
    val onEdit: (ChatMessage) -> Unit = {},
    val onRegenerate: (ChatMessage) -> Unit = {},
    val onDelete: (ChatMessage) -> Unit = {},
    val onBranch: (ChatMessage) -> Unit = {},
    val canBranch: Boolean = true,
    val onSwitchBranch: (Int) -> Unit = {},
    val onTokenDetail: (TokenDetail) -> Unit = {},
    val onProcessingTime: (ProcessingTime) -> Unit = {},
    val onEncryption: (Boolean) -> Unit = {},
    /** Web `openPythonExecDetail`: the footer's "Python" button. */
    val onPython: (List<com.minashin1120.aiplayground.data.PythonExecution>) -> Unit = {},
)

/** Web thought text: stored as JSON `{"text": …}` or as plain text. */
internal fun thoughtText(raw: String): String =
    com.minashin1120.aiplayground.data.reasoningText(raw)

/**
 * `.message-group` + `.message-bubble` (chat_core part14 `renderMessage`): right-aligned theme
 * gradient for the user, dark glass for the assistant, max 90% wide (80% from 768px). Tapping the
 * bubble shows the hover controls like a tap does in the mobile browser.
 */
@OptIn(ExperimentalLayoutApi::class)
@Composable
internal fun MessageBubble(
    message: ChatMessage,
    onFile: (String) -> Unit,
    loader: FileBytesLoader?,
    actions: MessageActions,
    controlsVisible: Boolean,
    onToggleControls: () -> Unit,
    branchIndex: Int = 0,
    branchCount: Int = 0,
    streaming: Boolean = false,
    /** Streaming answer parts the Web inserts into the bubble: boxes above the thought, the skeleton, and rows below. */
    liveTop: (@Composable () -> Unit)? = null,
    liveSkeleton: (@Composable () -> Unit)? = null,
    liveBottom: (@Composable () -> Unit)? = null,
    /** Tapping the quote strip at the top of the bubble returns to the quoted message. */
    onQuoteJump: (() -> Unit)? = null,
    /** A rising token starts the post-jump flash of [flashQuote] inside this bubble. */
    flashToken: Long = 0L,
    flashQuote: String? = null,
    /** Window Y of the flashed quote range once laid out, so the jump can align it under the top bar. */
    onQuoteFlashLocated: ((Float) -> Unit)? = null,
    /** The reasoning placeholder is shown collapsed (`.thought-content.collapsed`). */
    thoughtCollapsed: Boolean = false,
    preserveLiveLayout: Boolean = false,
    sendMotion: Modifier = Modifier,
) {
    val web = LocalWebPalette.current
    val user = message.role == "user"
    val persisted = numericId(message) != null
    // Web `renderMessage`: Python runs leave the answer (footer button) and MCP notices move after the prose.
    val (display, pythonRuns) = remember(message.content, user) {
        if (user) message.content to emptyList()
        else com.minashin1120.aiplayground.data.extractPythonExecutions(message.content)
            .let { (text, runs) -> com.minashin1120.aiplayground.data.moveMcpNotesToEnd(text) to runs }
    }
    val flight = LocalPromptSendMotion.current?.takeIf { it.id == message.id }
    val morph = flight?.let { PlaygroundMotion.WebEaseOut.transform(it.progress.value) } ?: 1f
    val corner = (12f + 8f * morph).dp
    val smallCorner = (12f - 4f * morph).dp
    val shape = if (user) RoundedCornerShape(corner, corner, smallCorner, corner) else RoundedCornerShape(20.dp, 20.dp, 20.dp, 8.dp)
    val background = when {
        user -> SolidColor(if (flight == null) web.theme.t600 else lerp(if (web.isLight) Color.White else Color(0xFF151F37), web.theme.t600, morph))
        web.isLight -> Brush.verticalGradient(listOf(Color.White, Color(0xFFF4F7FB)))
        else -> Brush.verticalGradient(listOf(Color(16, 22, 40).copy(alpha = 0.82f), Color(10, 14, 26).copy(alpha = 0.86f)))
    }
    val border = when {
        user -> Color.Transparent
        web.isLight -> Color(15, 23, 42).copy(alpha = 0.08f)
        else -> Color.White.copy(alpha = 0.06f)
    }
    val textColor = if (user) web.textInverse else if (web.isLight) web.text else Color.White
    // After a quote jump the quoted text flashes twice (once with reduced motion) so the reader sees
    // where the selection landed. The bubble tints faintly as a fallback for quotes inside code or
    // spanning several Markdown blocks, where the text range cannot be highlighted.
    val flash = remember { Animatable(0f) }
    val reduceQuoteFlash = LocalReduceMotion.current
    LaunchedEffect(flashToken) {
        if (flashToken == 0L || flashQuote.isNullOrBlank()) return@LaunchedEffect
        flash.snapTo(0f)
        val pulses = if (reduceQuoteFlash) 1 else 2
        repeat(pulses) { index ->
            flash.animateTo(1f, tween(if (reduceQuoteFlash) 260 else 120, easing = LinearEasing))
            flash.animateTo(0f, tween(if (reduceQuoteFlash) 360 else 200, easing = LinearEasing))
            if (!reduceQuoteFlash && index == 0) delay(100)
        }
    }
    BoxWithConstraints(Modifier.fillMaxWidth(), contentAlignment = if (user) Alignment.TopEnd else Alignment.TopStart) {
        val maxBubble = maxWidth * (if (maxWidth >= PlaygroundDimens.breakpoint) 0.8f else 0.9f)
        Box(Modifier.widthIn(max = maxBubble)) {
            Column(
                sendMotion
                    .then(if (user) Modifier else Modifier.shadow(14.dp, shape,
                        ambientColor = Color.Black.copy(alpha = 0.18f),
                        spotColor = Color.Black.copy(alpha = 0.18f)))
                    .clip(shape)
                    .background(background)
                    .background(QuoteFlashColor.copy(alpha = 0.10f * flash.value))
                    .border(1.dp, border, shape)
                    .clickable(interactionSource = remember { MutableInteractionSource() }, indication = null, onClick = onToggleControls)
                    .padding(16.dp),
            ) {
              // One selection for the whole bubble, so a long-press selection runs across paragraphs like Web.
              MessageSelection { Column {
                if (message.quote.isNotBlank()) QuoteStrip(message.quote, onQuoteJump)
                if (message.skillName.isNotBlank()) SkillNotice(message.skillName)
                liveTop?.invoke()
                if (!user && liveTop == null) {
                    val searched = remember(message.thought) { com.minashin1120.aiplayground.data.reasoningSearchUsed(message.thought) }
                    if (searched) LiveSearchBox("done")
                }
                if (!user && message.thought.isNotBlank()) {
                    val thought = remember(message.thought) { thoughtText(message.thought) }
                    if (thought.isNotBlank()) ThoughtContainer(thought,
                        openInitially = (streaming || preserveLiveLayout) && !thoughtCollapsed,
                        preparing = streaming && thoughtCollapsed)
                }
                if (user) {
                    // User messages are shown as raw text (`whitespace-pre-wrap font-sans text-sm`).
                    MessageSelection {
                        val body = remember(message.content, flashQuote, flash.value) {
                            if (flashQuote == null || flash.value <= 0f) AnnotatedString(message.content)
                            else quoteFlashText(message.content, flashQuote, flash.value)
                        }
                        Text(body, color = textColor, fontSize = 14.sp, lineHeight = 20.sp, fontFamily = FontFamily.Default)
                    }
                } else if (liveSkeleton != null) {
                    liveSkeleton()
                } else {
                    message.batch?.let { BatchStatusCard(it) }
                    if (display.isNotEmpty()) MarkdownText(com.minashin1120.aiplayground.data.missingFileMarkup(display, message.missingFiles), loader, onFile,
                        startCollapsed = true, canvasMode = LocalCanvasMode.current,
                        flashQuote = flashQuote, flashAlpha = flash.value,
                        onQuoteFlashLocated = onQuoteFlashLocated)
                }
                liveBottom?.invoke()
                val files = message.files.filterNot { !user && message.content.contains(it) }
                val maskImage = if (user) message.promptOptions?.optString("image_mask").orEmpty() else ""
                // An answer's audio file plays in place (ANDROID_ONLY §2); `.webm` may be video, so it stays a card.
                val (audio, others) = if (user) emptyList<String>() to files
                    else files.partition { isAudioPath(it) && !isVideoPath(it) && it !in message.missingFiles }
                audio.forEach { InlineAudioPlayer(it, onFile) }
                val (videos, cards) = if (user) emptyList<String>() to others
                    else others.partition { isVideoPath(it) && it !in message.missingFiles }
                videos.forEach { InlineVideoPlayer(it, onFile) }
                if (cards.isNotEmpty()) AttachmentGrid(cards, loader, onFile, message.missingFiles)
                if (maskImage.isNotBlank()) MaskAttachmentPreview(maskImage, loader, onFile)
                if (branchCount > 1) VersionSwitcher(branchIndex, branchCount, actions.onSwitchBranch)
                MessageFooter(message, user, actions, pythonRuns)
                if (!user) message.processing?.let { processing ->
                    ProcessingTimeFooter(processing) { actions.onProcessingTime(processing) }
                }
              } }
            }
            // `.msg-controls`: absolute, 12px above the bubble's top-right corner.
            val controlsAlpha by animateFloatAsState(if (controlsVisible) 1f else 0f, motionTween(LocalReduceMotion.current, 220), label = "msg controls")
            if (controlsAlpha > 0f) {
                Row(
                    Modifier
                        .align(Alignment.TopEnd)
                        .offset(y = (-12).dp + (2.dp * (1f - controlsAlpha)))
                        .zIndex(1f)
                        .graphicsLayer { alpha = controlsAlpha },
                    horizontalArrangement = Arrangement.spacedBy(4.dp),
                ) {
                    var copied by remember { mutableStateOf<Boolean?>(null) }
                    LaunchedEffect(copied) { if (copied != null) { delay(2000); copied = null } }
                    val clipboard = LocalClipboardManager.current
                    ControlButton(when (copied) { true -> R.drawable.fa_solid_check; false -> R.drawable.fa_solid_times; null -> R.drawable.fa_solid_copy }, "コピー") {
                        copied = runCatching { clipboard.setText(AnnotatedString(message.content)) }.isSuccess
                    }
                    if (user && persisted) ControlButton(R.drawable.fa_solid_pen, "編集") { actions.onEdit(message) }
                    if (!user) ControlButton(R.drawable.fa_solid_rotate_right, "再生成") { actions.onRegenerate(message) }
                    if (persisted && !streaming && actions.canBranch) ControlButton(R.drawable.fa_solid_code_branch, "新しいチャットに分岐") { actions.onBranch(message) }
                    if (persisted) ControlButton(R.drawable.fa_solid_trash, "削除") { actions.onDelete(message) }
                }
            }
        }
    }
}

@Composable
private fun MaskAttachmentPreview(file: String, loader: FileBytesLoader?, onFile: (String) -> Unit) {
    val web = LocalWebPalette.current
    Column(Modifier.padding(top = 8.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(5.dp)) {
            FaIcon(R.drawable.fa_solid_mask, null, size = 11.dp, tint = web.twText(Tw.amber200))
            Text("マスク画像（削除範囲）", fontSize = 10.sp, fontWeight = FontWeight.SemiBold, color = web.twText(Tw.amber200))
        }
        AttachmentGrid(listOf(file), loader, onFile)
    }
}

/** `.ctrl-btn`: 26px (24 + border), 10px corners, dark glass, 10px glyph. */
@Composable
private fun ControlButton(@androidx.annotation.DrawableRes icon: Int, label: String, onClick: () -> Unit) {
    val web = LocalWebPalette.current
    val shape = RoundedCornerShape(10.dp)
    Box(
        Modifier
            .size(26.dp)
            .clip(shape)
            .background(if (web.isLight) Color.White.copy(alpha = 0.94f) else Color(8, 12, 22).copy(alpha = 0.88f))
            .border(1.dp, Color(148, 163, 184).copy(alpha = 0.14f), shape)
            .clickable(onClickLabel = label, role = Role.Button, onClick = onClick),
        contentAlignment = Alignment.Center,
    ) { FaIcon(icon, label, size = 10.dp, tint = if (web.isLight) web.muted else Color(0xFF9AA3B2)) }
}

/** Quote strip at the top of a bubble (`bg-black/20 border-l-4 … italic truncate`). */
@Composable
private fun QuoteStrip(quote: String, onJump: (() -> Unit)?) {
    val web = LocalWebPalette.current
    val color = web.twText(Tw.gray300)
    Row(
        Modifier
            .fillMaxWidth()
            .padding(bottom = 8.dp)
            .clip(RoundedCornerShape(4.dp))
            .background(Color.Black.copy(alpha = 0.2f))
            .then(if (onJump != null) Modifier.clickable(role = Role.Button, onClickLabel = "引用元へ移動", onClick = onJump) else Modifier)
            .drawBehind { drawRect(web.theme.t500, size = androidx.compose.ui.geometry.Size(4.dp.toPx(), size.height)) }
            .padding(start = 12.dp, end = 8.dp, top = 8.dp, bottom = 8.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        FaIcon(R.drawable.fa_solid_quote_left, null, size = 12.dp, tint = color.copy(alpha = color.alpha * 0.5f),
            modifier = Modifier.padding(end = 4.dp))
        Text(quote, color = color, fontSize = 12.sp, lineHeight = 16.sp, fontStyle = FontStyle.Italic,
            maxLines = 1, overflow = TextOverflow.Ellipsis, modifier = Modifier.weight(1f))
    }
}

/** `.thought-container`: "Thinking Process" header that toggles the monospace reasoning text. */
@Composable
private fun ThoughtContainer(thought: String, openInitially: Boolean, preparing: Boolean) {
    val web = LocalWebPalette.current
    var open by remember { mutableStateOf(openInitially) }
    LaunchedEffect(openInitially) { if (!openInitially) open = false }
    val shape = RoundedCornerShape(16.dp)
    Column(
        Modifier
            .fillMaxWidth()
            .padding(bottom = 16.dp)
            .clip(shape)
            .background(if (web.isLight) Color(15, 23, 42).copy(alpha = 0.035f) else Color(10, 16, 40).copy(alpha = 0.6f))
            .border(1.dp, if (web.isLight) Color(15, 23, 42).copy(alpha = 0.08f) else Color(148, 163, 184).copy(alpha = 0.14f), shape),
    ) {
        Row(
            Modifier
                .fillMaxWidth()
                .background(web.theme.rgb(0.10f))
                .then(thinkingShimmer(preparing))
                .clickable(onClickLabel = "Thinking Process", role = Role.Button) { open = !open }
                .padding(horizontal = 13.6.dp, vertical = 8.8.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            FaIcon(R.drawable.fa_solid_brain, null, size = 12.dp, tint = Tw.purple400)
            Text("Thinking Process", color = if (web.isLight) web.theme.t600 else web.theme200, fontSize = 12.sp, lineHeight = 20.sp,
                fontWeight = FontWeight.SemiBold)
        }
        AnimatedVisibility(open, enter = expandFadeIn(LocalReduceMotion.current), exit = shrinkFadeOut(LocalReduceMotion.current)) {
            Box(
                Modifier
                    .fillMaxWidth()
                    .heightIn(max = 400.dp)
                    .background(if (web.isLight) Color(15, 23, 42).copy(alpha = 0.04f) else Color(6, 10, 24).copy(alpha = 0.8f))
                    .drawBehind { drawRect(Color(148, 163, 184).copy(alpha = 0.16f), size = androidx.compose.ui.geometry.Size(size.width, 1.dp.toPx())) }
                    .verticalScroll(rememberScrollState())
                    .padding(13.6.dp),
            ) {
                // Its full text extends beyond the clipped 400dp viewport. Registering it with
                // the bubble makes that hidden text overlap the answer in selection hit testing.
                MessageSelection(independent = true) {
                    Text(thought, color = if (web.isLight) Color(0xFF334155) else Color(0xFFCBD5F5), fontSize = 13.6.sp, lineHeight = 20.sp,
                        fontFamily = FontFamily.Monospace)
                }
            }
        }
    }
}

/** Web `.thinking-shimmer`: only the reasoning placeholder sweeps; received thoughts are static. */
@Composable
private fun thinkingShimmer(preparing: Boolean): Modifier {
    if (!preparing || LocalReduceMotion.current) return Modifier
    val highlight = LocalWebPalette.current.theme.rgb(0.15f)
    val sweep = rememberInfiniteTransition(label = "thinking shimmer").animateFloat(
        initialValue = -1.2f,
        targetValue = 1.2f,
        animationSpec = infiniteRepeatable(tween(1900, easing = CubicBezierEasing(0.45f, 0f, 0.15f, 1f))),
        label = "thinking shimmer sweep",
    )
    return Modifier.drawWithContent {
        drawContent()
        val left = size.width * sweep.value
        val width = size.width * 0.6f
        if (width > 0f) clipRect {
            drawRect(
                brush = Brush.horizontalGradient(
                    0f to Color.Transparent,
                    0.3f to Color.White.copy(alpha = 0.01f),
                    0.5f to highlight,
                    0.7f to Color.White.copy(alpha = 0.01f),
                    1f to Color.Transparent,
                    startX = left, endX = left + width,
                ),
                topLeft = Offset(left, 0f),
                size = androidx.compose.ui.geometry.Size(width, size.height),
            )
        }
    }
}

/** `.image-grid` (grid-1 … grid-4, grid-multi) with file thumbnails for non-images. */
@Composable
internal fun AttachmentGrid(files: List<String>, loader: FileBytesLoader?, onFile: (String) -> Unit, missing: Set<String> = emptySet()) {
    val count = files.size
    val shape = RoundedCornerShape(14.dp)
    Box(
        Modifier
            .padding(top = 8.dp)
            .widthIn(max = 400.dp)
            .fillMaxWidth()
            .clip(shape)
            .background(if (count in 2..4) Color.Black else Color.Transparent),
    ) {
        when {
            count == 1 -> AttachmentCell(files[0], loader, onFile, Modifier.fillMaxWidth(), single = true, missing = files[0] in missing)
            count == 2 -> Row(Modifier.fillMaxWidth().aspectRatio(2f), horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                files.forEach { AttachmentCell(it, loader, onFile, Modifier.weight(1f).fillMaxHeight(), missing = it in missing) }
            }
            count == 3 -> Row(Modifier.fillMaxWidth().aspectRatio(1f), horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                AttachmentCell(files[0], loader, onFile, Modifier.weight(1f).fillMaxHeight(), missing = files[0] in missing)
                Column(Modifier.weight(1f).fillMaxHeight(), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    AttachmentCell(files[1], loader, onFile, Modifier.weight(1f).fillMaxWidth(), missing = files[1] in missing)
                    AttachmentCell(files[2], loader, onFile, Modifier.weight(1f).fillMaxWidth(), missing = files[2] in missing)
                }
            }
            count == 4 -> Column(Modifier.fillMaxWidth().aspectRatio(1f), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                files.chunked(2).forEach { row ->
                    Row(Modifier.weight(1f).fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                        row.forEach { AttachmentCell(it, loader, onFile, Modifier.weight(1f).fillMaxHeight(), missing = it in missing) }
                    }
                }
            }
            else -> BoxWithConstraints(Modifier.fillMaxWidth()) {
                // `repeat(auto-fill, minmax(80px, 1fr))` with 4px gaps and 8px rounded squares.
                val columns = maxOf(1, ((maxWidth + 4.dp) / (80.dp + 4.dp)).toInt())
                Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    files.chunked(columns).forEach { row ->
                        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                            row.forEach { AttachmentCell(it, loader, onFile, Modifier.weight(1f).aspectRatio(1f), corner = 8, missing = it in missing) }
                            repeat(columns - row.size) { Spacer(Modifier.weight(1f)) }
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun AttachmentCell(
    file: String,
    loader: FileBytesLoader?,
    onFile: (String) -> Unit,
    modifier: Modifier,
    single: Boolean = false,
    corner: Int = 0,
    missing: Boolean = false,
) {
    if (missing) {
        val palette = LocalWebPalette.current
        Box(modifier.background(palette.twBg(Tw.gray800)).padding(12.dp), contentAlignment = Alignment.Center) { Text("ファイルがありません", fontSize = 12.sp, color = palette.twText(Tw.gray400)) }
        return
    }
    val web = LocalWebPalette.current
    val shape = if (corner > 0) RoundedCornerShape(corner.dp) else RectangleShape
    if (isImageReference(file)) {
        // A single image is narrower than the grid, so it carries the grid's 14px corners itself.
        val imageShape = if (single) RoundedCornerShape(14.dp) else shape
        if (single) {
            ProtectedImage(
                file, loader, onFile, modifier = modifier, thumbnail = true, compact = true,
                contentDescription = file.substringAfterLast('/'), shape = imageShape,
                contentScale = ContentScale.Fit,
                background = Color.Transparent,
                // `.grid-1`: up to the grid width and 400px tall, framed to the image so the margin does not open it.
                fitToImage = true, fitMaxHeight = 400.dp, fitAlignment = Alignment.TopCenter,
            )
        } else {
            // A multi-image cell is exactly the image, so the whole cell opens the viewer. Keeping the
            // click on the cell (not only on the decoded bitmap) also covers the loading placeholder.
            Box(
                modifier.clip(imageShape)
                    .clickable(role = Role.Button, onClickLabel = file.substringAfterLast('/')) { onFile(file) },
            ) {
                ProtectedImage(
                    file, loader, null, modifier = Modifier.fillMaxSize(), thumbnail = true, compact = false,
                    contentDescription = file.substringAfterLast('/'), shape = imageShape,
                    contentScale = ContentScale.Crop, background = Color(0xFF111111),
                )
            }
        }
    } else {
        // `.file-thumb`: file glyph and a truncated name on a gray tile.
        Column(
            modifier
                .clip(if (single) RoundedCornerShape(4.dp) else shape)
                .background(web.twBg(Tw.gray800))
                .border(1.dp, web.twBorder(Tw.gray600), if (single) RoundedCornerShape(4.dp) else shape)
                .clickable(onClickLabel = file.substringAfterLast('/'), role = Role.Button) { onFile(file) }
                .padding(vertical = 8.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center,
        ) {
            FaIcon(R.drawable.fa_solid_file, null, size = 24.dp, tint = web.twText(Tw.gray400), modifier = Modifier.padding(bottom = 4.dp))
            Text(file.substringAfterLast('/'), color = web.twText(Tw.white), fontSize = 9.sp, lineHeight = 12.sp, maxLines = 1,
                overflow = TextOverflow.Ellipsis, modifier = Modifier.width(80.dp), textAlign = androidx.compose.ui.text.style.TextAlign.Center)
        }
    }
}

/** `.skill-notice`: the Skill shown above the message content, like the search box. */
@Composable
private fun SkillNotice(name: String) {
    val web = LocalWebPalette.current
    val color = if (web.isLight) Color(126, 34, 206) else Tw.purple300
    val shape = RoundedCornerShape(11.2.dp)
    Row(
        Modifier.padding(bottom = 8.dp).fillMaxWidth().clip(shape)
            .background(if (web.isLight) Color.White else Color(8, 14, 28).copy(alpha = 0.8f))
            .border(1.dp, Tw.purple500.copy(alpha = 0.38f), shape)
            .padding(horizontal = 12.8.dp, vertical = 8.8.dp),
        verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        FaIcon(R.drawable.fa_solid_wand_magic_sparkles, null, size = 12.dp, tint = color, fixedWidth = true)
        Text("Skill: ", fontSize = 12.sp, lineHeight = 16.sp, color = color)
        Text(name, fontSize = 12.sp, lineHeight = 16.sp, fontWeight = FontWeight.Bold, color = color,
            maxLines = 1, overflow = TextOverflow.Ellipsis, modifier = Modifier.weight(1f, fill = false))
    }
}

/** Branch switcher under the content: `‹ 1 / 2 ›` (10px, gray-400, disabled at the ends). */
@Composable
private fun VersionSwitcher(index: Int, count: Int, onSwitch: (Int) -> Unit) {
    val web = LocalWebPalette.current
    val color = web.twText(Tw.gray400)
    Row(Modifier.padding(top = 4.dp), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        val hasPrevious = index > 0
        val hasNext = index < count - 1
        Box(Modifier.alpha(if (hasPrevious) 1f else 0.3f).clickable(enabled = hasPrevious, onClickLabel = "前の分岐", role = Role.Button) { onSwitch(index - 1) }) {
            FaIcon(R.drawable.fa_solid_chevron_left, "前の分岐", size = 10.dp, tint = color)
        }
        Text("${index + 1} / $count", color = color, fontSize = 10.sp, lineHeight = 14.sp)
        Box(Modifier.alpha(if (hasNext) 1f else 0.3f).clickable(enabled = hasNext, onClickLabel = "次の分岐", role = Role.Button) { onSwitch(index + 1) }) {
            FaIcon(R.drawable.fa_solid_chevron_right, "次の分岐", size = 10.dp, tint = color)
        }
    }
}

/** `.message-footer-meta`: model • token counts (dotted underline) • lock, 10px monospace, right-aligned. */
@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun MessageFooter(message: ChatMessage, user: Boolean, actions: MessageActions,
                          pythonRuns: List<com.minashin1120.aiplayground.data.PythonExecution> = emptyList()) {
    val web = LocalWebPalette.current
    val color = if (web.isLight) Color(0xFF3F4A5C) else Color(203, 213, 225).copy(alpha = 0.9f)
    val style = TextStyle(color = color, fontSize = 10.sp, lineHeight = 20.sp, fontFamily = FontFamily.Monospace)
    val parts = mutableListOf<@Composable () -> Unit>()
    if (!user && message.model.isNotBlank()) parts += { Text(message.model, style = style) }
    messageTokenLabel(message)?.let { label ->
        parts += {
            Text(
                label, style = style.copy(color = if (web.isLight) web.text else color),
                modifier = Modifier
                    .dottedUnderline(if (web.isLight) web.text else color)
                    .clickable(onClickLabel = "トークンの詳細", role = Role.Button) { actions.onTokenDetail(message.tokenDetail()) },
            )
        }
    }
    message.encrypted?.let { encrypted ->
        parts += {
            Box(
                Modifier.clickable(onClickLabel = if (encrypted) "Encrypted" else "Plain", role = Role.Button) { actions.onEncryption(encrypted) },
            ) {
                FaIcon(if (encrypted) R.drawable.fa_solid_lock else R.drawable.fa_solid_lock_open, if (encrypted) "Encrypted" else "Plain",
                    size = 10.dp, tint = if (web.isLight) (if (user) Color.White else web.text) else Color(203, 213, 225).copy(alpha = 0.8f))
            }
        }
    }
    if (!user && pythonRuns.isNotEmpty()) parts += {
        // `.python-exec-btn`
        Row(
            Modifier.padding(start = 2.4.dp).clip(CircleShape).background(Color(245, 158, 11).copy(alpha = 0.12f))
                .border(1.dp, Color(245, 158, 11).copy(alpha = 0.35f), CircleShape)
                .clickable(onClickLabel = "Python実行結果を表示", role = Role.Button) { actions.onPython(pythonRuns) }
                .padding(horizontal = 7.2.dp, vertical = 2.4.dp),
            verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(4.8.dp),
        ) {
            FaIcon(R.drawable.fa_solid_terminal, null, size = 8.5.dp, tint = Color(0xFFFDE68A))
            Text(if (pythonRuns.size > 1) "Python ×${pythonRuns.size}" else "Python",
                style = style.copy(color = Color(0xFFFDE68A), lineHeight = 12.sp))
        }
    }
    if (parts.isEmpty()) return
    FlowRow(
        Modifier.fillMaxWidth().padding(top = 8.dp),
        horizontalArrangement = Arrangement.End,
        verticalArrangement = Arrangement.Center,
    ) {
        parts.forEachIndexed { index, part ->
            Box(Modifier.heightIn(min = 20.dp), contentAlignment = Alignment.Center) { part() }
            if (index < parts.lastIndex) Text(" • ", style = style)
        }
    }
}

internal fun formatProcessingDuration(ms: Long): String {
    if (ms < 0) return "-"
    if (ms < 60_000) return String.format(java.util.Locale.US, "%.2f秒", ms / 1000.0)
    val seconds = ms / 1000.0
    val minutes = (seconds / 60).toInt()
    return String.format(java.util.Locale.US, "%d分%.1f秒", minutes, seconds - minutes * 60)
}

/** Web `.processing-time-row`: the answer's total processing time under the footer. */
@Composable
private fun ProcessingTimeFooter(processing: ProcessingTime, onClick: () -> Unit) {
    val web = LocalWebPalette.current
    val color = if (web.isLight) Color(0xFF3F4A5C).copy(alpha = 0.8f) else Color(148, 163, 184).copy(alpha = 0.8f)
    Row(Modifier.fillMaxWidth().padding(top = 4.dp), horizontalArrangement = Arrangement.End) {
        Text(
            "処理時間 ${formatProcessingDuration(processing.totalMs)}",
            color = color,
            fontSize = 10.sp,
            lineHeight = 14.sp,
            fontFamily = FontFamily.Monospace,
            modifier = Modifier.dottedUnderline(color)
                .clickable(onClickLabel = "回答処理時間の内訳", role = Role.Button, onClick = onClick),
        )
    }
}

private val PROCESSING_PHASE_LABELS = mapOf(
    "send" to "送信〜受付",
    "queue" to "待ち行列",
    "provider" to "初回応答まで",
    "generation" to "生成",
    "receive" to "受信・表示",
)

/** `#processing-time-modal`. */
@Composable
internal fun ProcessingTimeDialog(processing: ProcessingTime, onDismiss: () -> Unit) {
    PlaygroundDialog(
        onDismissRequest = onDismiss,
        title = { Text("回答処理時間") },
        text = {
            Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                InfoRow("合計", formatProcessingDuration(processing.totalMs))
                processing.firstResponseMs?.let { InfoRow("最初の応答", formatProcessingDuration(it)) }
                if (processing.phases.isNotEmpty()) {
                    Box(Modifier.fillMaxWidth().height(1.dp).background(LocalWebPalette.current.lineSoft))
                    processing.phases.forEach { phase ->
                        InfoRow(PROCESSING_PHASE_LABELS[phase.key] ?: phase.key, formatProcessingDuration(phase.ms))
                    }
                }
                Text(
                    when {
                        processing.source == "client" -> "回答完了時に端末で記録した時間です。詳細な内訳は履歴の再読み込み後に表示されます。"
                        processing.phases.isNotEmpty() -> "記録できた処理区間の内訳です。"
                        else -> "この回答では合計時間だけを記録しています。"
                    },
                    color = LocalWebPalette.current.twText(Tw.gray400), fontSize = 11.sp, lineHeight = 16.sp,
                )
            }
        },
        confirmButton = {},
        showFooter = false,
        icon = R.drawable.fa_solid_clock_rotate_left,
        subtitle = "回答にかかった時間の内訳",
        tone = WebModalTone.Info,
        panelMaxWidth = 440.dp,
    )
}

/** `underline decoration-dotted` for a single-line label. */
private fun Modifier.dottedUnderline(color: Color): Modifier = drawBehind {
    val y = size.height - 3.dp.toPx()
    drawLine(color, Offset(0f, y), Offset(size.width, y), strokeWidth = 1.dp.toPx(),
        pathEffect = PathEffect.dashPathEffect(floatArrayOf(1.dp.toPx(), 2.dp.toPx())))
}

/** `#total-token-bar`: confirmed totals and the current input estimate, each opening its own details. */
@OptIn(ExperimentalLayoutApi::class)
@Composable
internal fun TotalTokenBar(path: TokenTotals, showEstimate: Boolean = false, estimate: @Composable () -> Unit = {},
                          pathCost: com.minashin1120.aiplayground.data.CostEstimate? = null,
                          pricing: com.minashin1120.aiplayground.data.ModelCostEstimator? = null, onDetail: (TokenDetail) -> Unit) {
    val web = LocalWebPalette.current
    if (path.total <= 0 && !showEstimate) return
    val text = if (web.isLight) web.text else Tw.slate200
    Column(Modifier.fillMaxWidth().background(if (web.isLight) Color.White.copy(alpha = 0.9f) else Color(8, 14, 28).copy(alpha = 0.55f))) {
        Row(Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 6.dp), verticalAlignment = Alignment.CenterVertically) {
            FaIcon(R.drawable.fa_solid_coins, null, size = 12.dp, tint = web.twText(Tw.amber300))
            Spacer(Modifier.width(8.dp))
            FlowRow(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                val style = TextStyle(color = text, fontSize = 12.sp, lineHeight = 16.sp, fontFamily = WebFonts.sans)
                Text("Total: ${path.total} tokens", style = style,
                    modifier = Modifier.dottedUnderline(text).clickable(role = Role.Button) { onDetail(path.detail(false, pathCost, pricing)) })
                pathCost?.let { cost -> Text(cost.label(), style = style,
                    modifier = Modifier.dottedUnderline(text).clickable(role = Role.Button) { onDetail(path.detail(false, cost, pricing)) }) }
                if (showEstimate) estimate()
            }
        }
        Box(Modifier.fillMaxWidth().height(1.dp).background(if (web.isLight) Color(15, 23, 42).copy(alpha = 0.10f) else web.lineSoft))
    }
}

/** `#token-detail-modal`. */
@Composable
internal fun TokenDetailDialog(detail: TokenDetail, onDismiss: () -> Unit) {
    PlaygroundDialog(
        onDismissRequest = onDismiss,
        title = { Text(detail.title) },
        text = {
        Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            fun value(v: Int?) = v?.toString() ?: "-"
            listOf(
                "Total" to value(detail.total), "In" to value(detail.tokensIn), "Out" to value(detail.tokensOut),
                "Content" to value(detail.tokensContent), "Thought" to value(detail.tokensThought),
                "Encryption" to when (detail.encrypted) { null -> "-"; true -> "Encrypted"; false -> "Plain" },
            ).forEach { (label, v) -> InfoRow(label, v) }
            if (detail.cacheRead != null || detail.cacheWrite != null) {
                InfoRow("Cache read", value(detail.cacheRead))
                InfoRow("Cache write", value(detail.cacheWrite))
                val ttls = listOf("5m", "1h", "30m", "12h").mapNotNull { ttl ->
                    detail.cacheWriteTtls[ttl]?.takeIf { it > 0 }?.let { "$ttl: $it" }
                }
                if (ttls.isNotEmpty()) {
                    Text("TTL ${ttls.joinToString(" · ")}", color = LocalWebPalette.current.twText(Tw.slate200),
                        fontSize = 11.sp, fontFamily = WebFonts.sans)
                }
            }
            detail.cost?.let { cost ->
                Text(cost.label(), color = LocalWebPalette.current.twText(Tw.slate200), fontSize = 12.sp)
                Text("集計したAPI呼び出し: ${cost.requests}回", color = LocalWebPalette.current.twText(Tw.slate200), fontSize = 12.sp)
                Text(com.minashin1120.aiplayground.data.COST_ESTIMATE_NOTE, color = LocalWebPalette.current.twText(Tw.slate200), fontSize = 12.sp)
                Text("公式料金確認日: ${detail.pricing?.checkedAt ?: "取得できませんでした"}", color = LocalWebPalette.current.twText(Tw.slate200), fontSize = 12.sp)
                val uriHandler = androidx.compose.ui.platform.LocalUriHandler.current
                detail.pricing?.sources?.forEach { (provider, url) ->
                    Text("$provider 公式料金", color = Tw.blue400, fontSize = 12.sp,
                        modifier = Modifier.clickable(role = Role.Button) { uriHandler.openUri(url) })
                }
            }
        }
        },
        confirmButton = {},
        showFooter = false,
        icon = R.drawable.fa_solid_coins,
        subtitle = "使用量と費用の内訳",
        tone = WebModalTone.Info,
        panelMaxWidth = 440.dp,
    )
}

/**
 * `#encryption-status-modal`. For an admin with an open chat it adds Web's `#encryption-status-admin-actions`
 * button that decrypts (encrypted message) or re-encrypts (plain message) the whole chat; [busy] shows 処理中....
 */
@Composable
internal fun EncryptionStatusDialog(
    encrypted: Boolean,
    admin: Boolean,
    canToggle: Boolean,
    busy: Boolean,
    onToggle: () -> Unit,
    onSettings: () -> Unit,
    onDismiss: () -> Unit,
) {
    val web = LocalWebPalette.current
    PlaygroundDialog(
        onDismissRequest = onDismiss,
        title = { Text(if (encrypted) "暗号化されています" else "暗号化されていません") },
        text = {
        Column {
        Text(
            when {
                encrypted && admin -> "このメッセージはE2EEで暗号化されています。管理者は下のボタンでこのチャット全体を復号化できます。"
                encrypted -> "このメッセージはE2EEで暗号化されています。"
                admin -> "このメッセージは暗号化されていません。管理者は下のボタンでこのチャット全体を再暗号化できます。"
                else -> "このメッセージは暗号化されていません。"
            },
            color = web.twText(Tw.gray300), fontSize = 12.sp, lineHeight = 16.sp, modifier = Modifier.padding(top = 12.dp),
        )
        if (canToggle) Column(Modifier.padding(top = 16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("管理者: このチャット全体のメッセージを変換できます。", color = web.twText(Tw.cyan200).copy(alpha = 0.8f),
                fontSize = 10.sp, lineHeight = 14.sp)
            Box(
                Modifier.fillMaxWidth().clip(RoundedCornerShape(4.dp)).background(if (encrypted) Tw.amber600 else Tw.cyan700)
                    .clickable(enabled = !busy, role = Role.Button, onClick = onToggle).padding(horizontal = 12.dp, vertical = 8.dp),
                contentAlignment = Alignment.Center,
            ) {
                Text(if (busy) "処理中..." else if (encrypted) "このチャットを復号化" else "このチャットを再暗号化",
                    color = web.twText(Tw.white), fontSize = 12.sp, lineHeight = 16.sp, fontWeight = FontWeight.Bold)
            }
        }
        }
        },
        confirmButton = { WebButton(onClick = onSettings, variant = WebButtonVariant.Primary) { Text("設定へ") } },
        dismissButton = { WebButton(onClick = onDismiss, variant = WebButtonVariant.Ghost) { Text("閉じる") } },
        icon = R.drawable.fa_solid_lock,
        subtitle = "メッセージの暗号化状態",
        tone = WebModalTone.Info,
        panelMaxWidth = 440.dp,
    )
}

@Composable
private fun InfoRow(label: String, value: String) {
    val web = LocalWebPalette.current
    val color = web.twText(Tw.gray300)
    Row(Modifier.fillMaxWidth()) {
        Text(label, color = color, fontSize = 12.sp, lineHeight = 16.sp, modifier = Modifier.weight(1f))
        Text(value, color = color, fontSize = 12.sp, lineHeight = 16.sp, fontFamily = FontFamily.Monospace)
    }
}

@Composable
private fun SmallModalButton(label: String, background: Color, onClick: () -> Unit) {
    val web = LocalWebPalette.current
    Box(
        Modifier.clip(RoundedCornerShape(4.dp)).background(background).clickable(role = Role.Button, onClick = onClick)
            .padding(horizontal = 12.dp, vertical = 6.dp),
    ) { Text(label, color = web.twText(Tw.white), fontSize = 12.sp, lineHeight = 16.sp) }
}

/** `bg-gray-900 border border-gray-700 rounded-xl max-w-sm p-4` modal with the corner close button. */
@Composable
internal fun SmallInfoModal(onDismiss: () -> Unit, content: @Composable () -> Unit) {
    val web = LocalWebPalette.current
    WebOverlayModal(onDismiss, if (web.isLight) Color(0xFFF7F9FC) else Tw.gray900.copy(alpha = 0.9f), 4.dp) { _ ->
        val shape = RoundedCornerShape(12.dp)
        Box(
            Modifier
                .padding(16.dp)
                .widthIn(max = 384.dp)
                .fillMaxWidth()
                .modalPanelTaps()
                .clip(shape)
                .background(web.twBg(Tw.gray900))
                .border(1.dp, web.twBorder(Tw.gray700), shape)
                .padding(16.dp),
        ) {
            Column { content() }
            Box(Modifier.align(Alignment.TopEnd).clickable(onClickLabel = "閉じる", role = Role.Button, onClick = onDismiss)) {
                FaIcon(R.drawable.fa_solid_times, "閉じる", size = 14.dp, tint = web.twText(Tw.gray400))
            }
        }
    }
}

/**
 * `#welcome-screen`: skill heading with gradient text, subtitle and the quick-access model buttons
 * (one column, two from 640px), over the theme radial glow.
 */
@Composable
internal fun WelcomeScreen(models: List<Pair<String, String>>, onChoose: (String) -> Unit, modifier: Modifier = Modifier) {
    val web = LocalWebPalette.current
    BoxWithConstraints(
        modifier
            .fillMaxSize()
            .drawBehind {
                drawRect(Brush.radialGradient(listOf(web.theme.rgb(0.12f), Color.Transparent),
                    center = Offset(size.width / 2f, size.height * 0.18f), radius = 640.dp.toPx() * 0.66f / 2f * 1.4f))
                drawRect(Brush.radialGradient(listOf(Color(99, 102, 241).copy(alpha = 0.06f), Color.Transparent),
                    center = Offset(size.width / 2f, size.height), radius = 420.dp.toPx() * 0.62f / 2f * 1.4f))
            }
            .padding(horizontal = 8.dp, vertical = 24.dp),
        contentAlignment = Alignment.Center,
    ) {
        val twoColumns = maxWidth >= 640.dp
        Column(horizontalAlignment = Alignment.CenterHorizontally, modifier = Modifier.verticalScroll(rememberScrollState())) {
            StaggerIn(0) {
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    FaIcon(R.drawable.fa_solid_wand_magic_sparkles, null, size = 22.4.dp, tint = web.theme.t500)
                    val gradient = if (web.isLight) listOf(Color(0xFF0F766E), web.theme.t600, Color(0xFFB45309))
                        else listOf(Color(0xFFF8FAFC), web.theme.t200, Tw.amber200)
                    Text("AI Skills & Chat", fontSize = 22.4.sp, lineHeight = 32.sp, fontWeight = FontWeight.Bold, letterSpacing = (-0.035).em,
                        style = TextStyle(brush = Brush.linearGradient(gradient)))
                }
            }
            StaggerIn(1) {
                Text("使いたいモデルを選んで、すぐに会話を始められます",
                    color = if (web.isLight) web.muted else Color(203, 213, 225).copy(alpha = 0.78f),
                    fontSize = 14.sp, lineHeight = 20.sp, fontWeight = FontWeight.Medium, letterSpacing = 0.01.em,
                    textAlign = androidx.compose.ui.text.style.TextAlign.Center, modifier = Modifier.padding(top = 20.dp))
            }
            Column(
                Modifier.padding(top = 20.dp).widthIn(max = 576.dp).fillMaxWidth().padding(horizontal = 16.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                models.chunked(if (twoColumns) 2 else 1).forEachIndexed { rowIndex, row ->
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        row.forEachIndexed { columnIndex, (id, label) ->
                            Box(Modifier.weight(1f)) {
                                StaggerIn(2 + rowIndex * row.size + columnIndex) { WelcomeModelButton(label) { onChoose(id) } }
                            }
                        }
                        if (twoColumns && row.size == 1) Spacer(Modifier.weight(1f))
                    }
                }
            }
        }
    }
}

@Composable
private fun WelcomeModelButton(label: String, onClick: () -> Unit) {
    val web = LocalWebPalette.current
    val shape = RoundedCornerShape(16.dp)
    Box(
        Modifier
            .fillMaxWidth()
            .clip(shape)
            .background(
                if (web.isLight) Brush.verticalGradient(listOf(Color.White.copy(alpha = 0.86f), Color.White.copy(alpha = 0.86f)))
                else Brush.verticalGradient(listOf(Color(16, 22, 40).copy(alpha = 0.78f), Color(10, 14, 26).copy(alpha = 0.8f))),
            )
            .border(1.dp, Color.White.copy(alpha = 0.08f), shape)
            .clickable(role = Role.Button, onClick = onClick)
            .padding(horizontal = 14.4.dp, vertical = 11.2.dp),
    ) {
        Text(label, color = if (web.isLight) Color(0xFF697588) else Tw.gray500, fontSize = 14.sp, lineHeight = 20.sp,
            fontWeight = FontWeight.SemiBold, letterSpacing = 0.01.em, maxLines = 1, overflow = TextOverflow.Ellipsis)
    }
}

/** `.chat-scroll-to-bottom`: centered pill "一番下へ" with a down arrow. */
@Composable
internal fun ScrollToBottomPill(onClick: () -> Unit, modifier: Modifier = Modifier) {
    val web = LocalWebPalette.current
    Row(
        modifier
            .heightIn(min = 37.6.dp)
            .shadow(20.dp, RoundedCornerShape(999.dp), ambientColor = Color(2, 6, 23).copy(alpha = 0.42f), spotColor = Color(2, 6, 23).copy(alpha = 0.42f))
            .clip(RoundedCornerShape(999.dp))
            .background(Color(10, 16, 32).copy(alpha = 0.94f))
            .border(1.dp, web.theme.rgb(0.42f), RoundedCornerShape(999.dp))
            .clickable(onClickLabel = "一番下まで移動して自動スクロールを再開", role = Role.Button, onClick = onClick)
            .padding(horizontal = 14.4.dp, vertical = 8.8.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(7.2.dp),
    ) {
        FaIcon(R.drawable.fa_solid_arrow_down, null, size = 12.dp, tint = Color(0xFFF8FAFC))
        Text("一番下へ", color = Color(0xFFF8FAFC), fontSize = 12.sp, lineHeight = 12.sp, fontWeight = FontWeight.Bold)
    }
}

/** Web `.batch-status-card`: Batch state above the answer (violet while running, green done, red failed). */
@Composable
private fun BatchStatusCard(info: com.minashin1120.aiplayground.data.BatchInfo) {
    val web = LocalWebPalette.current
    val (border, background, text) = when (info.state) {
        "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED" ->
            Triple(Tw.red400.copy(alpha = 0.4f), Color(69, 10, 10).copy(alpha = 0.3f), web.twText(Tw.red100))
        "JOB_STATE_SUCCEEDED" -> Triple(Tw.emerald400.copy(alpha = 0.4f), Tw.emerald950.copy(alpha = 0.3f), web.twText(Tw.emerald100))
        else -> Triple(Tw.violet400.copy(alpha = 0.4f), Color(46, 16, 101).copy(alpha = 0.3f), web.twText(Color(0xFFEDE9FE)))
    }
    val shape = RoundedCornerShape(8.dp)
    Column(
        Modifier.padding(bottom = 12.dp).fillMaxWidth().clip(shape).background(background).border(1.dp, border, shape)
            .padding(horizontal = 12.dp, vertical = 8.dp),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            FaIcon(R.drawable.fa_solid_layer_group, null, size = 11.dp, tint = text, modifier = Modifier.padding(end = 4.dp))
            Text("Batch", fontSize = 12.sp, lineHeight = 16.sp, fontWeight = FontWeight.SemiBold, color = text)
        }
        Text(com.minashin1120.aiplayground.data.batchStatusText(info), fontSize = 12.sp, lineHeight = 16.sp,
            color = text.copy(alpha = 0.9f), modifier = Modifier.padding(top = 4.dp))
    }
}
