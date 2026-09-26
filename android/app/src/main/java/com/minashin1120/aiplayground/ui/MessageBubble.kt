package com.minashin1120.aiplayground.ui

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.animateFloatAsState
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
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.selection.SelectionContainer
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
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.RectangleShape
import androidx.compose.ui.graphics.graphicsLayer
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
import com.minashin1120.aiplayground.data.TokenTotals
import com.minashin1120.aiplayground.data.isImageReference
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
)

internal fun ChatMessage.tokenDetail(): TokenDetail {
    val total = tokens ?: if (tokensIn != null || tokensOut != null) (tokensIn ?: 0) + (tokensOut ?: 0) else null
    return TokenDetail(if (model.isNotBlank()) "$model ($role)" else role, total, tokensIn, tokensOut, tokensContent, tokensThought, encrypted)
}

internal fun TokenTotals.detail(allBranches: Boolean): TokenDetail = TokenDetail(
    if (allBranches) "Conversation (All branches) (total)" else "Conversation (total)",
    total, tokensIn, tokensOut, tokensContent, tokensThought, null,
)

/** Callbacks of the controls above a bubble (`.msg-controls`) and its footer. */
internal class MessageActions(
    val onEdit: (ChatMessage) -> Unit = {},
    val onRegenerate: (ChatMessage) -> Unit = {},
    val onDelete: (ChatMessage) -> Unit = {},
    val onSwitchBranch: (Int) -> Unit = {},
    val onTokenDetail: (TokenDetail) -> Unit = {},
    val onEncryption: (Boolean) -> Unit = {},
)

/** Web thought text: stored as JSON `{"text": …}` or as plain text. */
internal fun thoughtText(raw: String): String =
    runCatching { JSONObject(raw).optString("text") }.getOrNull()?.takeIf { raw.trimStart().startsWith("{") } ?: raw

/**
 * `.message-group` + `.message-bubble` (chat_core part13 `renderMessage`): right-aligned theme
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
    /** The reasoning placeholder is shown collapsed (`.thought-content.collapsed`). */
    thoughtCollapsed: Boolean = false,
) {
    val web = LocalWebPalette.current
    val user = message.role == "user"
    val persisted = numericId(message) != null
    val shape = if (user) RoundedCornerShape(20.dp, 20.dp, 8.dp, 20.dp) else RoundedCornerShape(20.dp, 20.dp, 20.dp, 8.dp)
    val background = when {
        user -> Brush.verticalGradient(listOf(web.theme.t500, web.theme.t600))
        web.isLight -> Brush.verticalGradient(listOf(Color.White, Color(0xFFF4F7FB)))
        else -> Brush.verticalGradient(listOf(Color(16, 22, 40).copy(alpha = 0.82f), Color(10, 14, 26).copy(alpha = 0.86f)))
    }
    val border = when {
        web.isLight -> Color(15, 23, 42).copy(alpha = 0.08f)
        user -> web.theme.rgb(0.28f)
        else -> Color.White.copy(alpha = 0.06f)
    }
    val textColor = if (user) web.textInverse else if (web.isLight) web.text else Color.White
    BoxWithConstraints(Modifier.fillMaxWidth(), contentAlignment = if (user) Alignment.TopEnd else Alignment.TopStart) {
        val maxBubble = maxWidth * (if (maxWidth >= PlaygroundDimens.breakpoint) 0.8f else 0.9f)
        Box(Modifier.widthIn(max = maxBubble)) {
            Column(
                Modifier
                    .shadow(if (user) 10.dp else 14.dp, shape,
                        ambientColor = if (user) web.theme.rgb(0.16f) else Color.Black.copy(alpha = 0.18f),
                        spotColor = if (user) web.theme.rgb(0.16f) else Color.Black.copy(alpha = 0.18f))
                    .clip(shape)
                    .background(background)
                    .border(1.dp, border, shape)
                    .clickable(interactionSource = remember { MutableInteractionSource() }, indication = null, onClick = onToggleControls)
                    .padding(16.dp),
            ) {
                if (message.quote.isNotBlank()) QuoteStrip(message.quote)
                liveTop?.invoke()
                if (!user && message.thought.isNotBlank()) {
                    val thought = remember(message.thought) { thoughtText(message.thought) }
                    if (thought.isNotBlank()) ThoughtContainer(thought, openInitially = streaming && !thoughtCollapsed)
                }
                if (user) {
                    // User messages are shown as raw text (`whitespace-pre-wrap font-sans text-sm`).
                    SelectionContainer {
                        Text(message.content, color = textColor, fontSize = 14.sp, lineHeight = 20.sp, fontFamily = FontFamily.Default)
                    }
                } else if (liveSkeleton != null) {
                    liveSkeleton()
                } else if (message.content.isNotEmpty()) {
                    MarkdownText(message.content, loader, onFile, startCollapsed = !streaming)
                }
                liveBottom?.invoke()
                val files = message.files.filterNot { !user && message.content.contains(it) }
                if (files.isNotEmpty()) AttachmentGrid(files, loader, onFile)
                if (branchCount > 1) VersionSwitcher(branchIndex, branchCount, actions.onSwitchBranch)
                MessageFooter(message, user, actions)
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
                    if (persisted) ControlButton(R.drawable.fa_solid_trash, "削除") { actions.onDelete(message) }
                }
            }
        }
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
private fun QuoteStrip(quote: String) {
    val web = LocalWebPalette.current
    val color = web.twText(Tw.gray300)
    Row(
        Modifier
            .fillMaxWidth()
            .padding(bottom = 8.dp)
            .clip(RoundedCornerShape(4.dp))
            .background(Color.Black.copy(alpha = 0.2f))
            .drawBehind { drawRect(web.theme.t500, size = androidx.compose.ui.geometry.Size(4.dp.toPx(), size.height)) }
            .padding(start = 12.dp, end = 8.dp, top = 8.dp, bottom = 8.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        FaIcon(R.drawable.fa_solid_quote_left, null, size = 12.dp, tint = color.copy(alpha = color.alpha * 0.5f),
            modifier = Modifier.padding(end = 4.dp))
        Text(quote, color = color, fontSize = 12.sp, lineHeight = 16.sp, fontStyle = FontStyle.Italic,
            maxLines = 1, overflow = TextOverflow.Ellipsis)
    }
}

/** `.thought-container`: "Thinking Process" header that toggles the monospace reasoning text. */
@Composable
private fun ThoughtContainer(thought: String, openInitially: Boolean) {
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
                SelectionContainer {
                    Text(thought, color = if (web.isLight) Color(0xFF334155) else Color(0xFFCBD5F5), fontSize = 13.6.sp, lineHeight = 20.sp,
                        fontFamily = FontFamily.Monospace)
                }
            }
        }
    }
}

/** `.image-grid` (grid-1 … grid-4, grid-multi) with file thumbnails for non-images. */
@Composable
private fun AttachmentGrid(files: List<String>, loader: FileBytesLoader?, onFile: (String) -> Unit) {
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
            count == 1 -> AttachmentCell(files[0], loader, onFile, Modifier.fillMaxWidth().heightIn(max = 400.dp), single = true)
            count == 2 -> Row(Modifier.fillMaxWidth().aspectRatio(2f), horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                files.forEach { AttachmentCell(it, loader, onFile, Modifier.weight(1f).fillMaxHeight()) }
            }
            count == 3 -> Row(Modifier.fillMaxWidth().aspectRatio(1f), horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                AttachmentCell(files[0], loader, onFile, Modifier.weight(1f).fillMaxHeight())
                Column(Modifier.weight(1f).fillMaxHeight(), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    AttachmentCell(files[1], loader, onFile, Modifier.weight(1f).fillMaxWidth())
                    AttachmentCell(files[2], loader, onFile, Modifier.weight(1f).fillMaxWidth())
                }
            }
            count == 4 -> Column(Modifier.fillMaxWidth().aspectRatio(1f), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                files.chunked(2).forEach { row ->
                    Row(Modifier.weight(1f).fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                        row.forEach { AttachmentCell(it, loader, onFile, Modifier.weight(1f).fillMaxHeight()) }
                    }
                }
            }
            else -> BoxWithConstraints(Modifier.fillMaxWidth()) {
                // `repeat(auto-fill, minmax(80px, 1fr))` with 4px gaps and 8px rounded squares.
                val columns = maxOf(1, ((maxWidth + 4.dp) / (80.dp + 4.dp)).toInt())
                Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    files.chunked(columns).forEach { row ->
                        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                            row.forEach { AttachmentCell(it, loader, onFile, Modifier.weight(1f).aspectRatio(1f), corner = 8) }
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
) {
    val web = LocalWebPalette.current
    val shape = if (corner > 0) RoundedCornerShape(corner.dp) else RectangleShape
    if (isImageReference(file)) {
        ProtectedImage(
            file, loader, onFile, modifier = modifier, thumbnail = true, compact = single,
            contentDescription = file.substringAfterLast('/'), shape = shape,
            contentScale = if (single) ContentScale.Fit else ContentScale.Crop,
            background = if (single) Color.Transparent else Color(0xFF111111),
        )
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

/** `.message-footer-meta`: model • Gem • token counts (dotted underline) • lock, 10px monospace, right-aligned. */
@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun MessageFooter(message: ChatMessage, user: Boolean, actions: MessageActions) {
    val web = LocalWebPalette.current
    val color = if (web.isLight) Color(0xFF3F4A5C) else Color(203, 213, 225).copy(alpha = 0.9f)
    val style = TextStyle(color = color, fontSize = 10.sp, lineHeight = 20.sp, fontFamily = FontFamily.Monospace)
    val parts = mutableListOf<@Composable () -> Unit>()
    if (!user && message.model.isNotBlank()) parts += { Text(message.model, style = style) }
    if (message.gemName.isNotBlank()) parts += {
        Row(verticalAlignment = Alignment.CenterVertically) {
            FaIcon(R.drawable.fa_solid_gem, null, size = 10.dp, tint = Tw.purple300.copy(alpha = 0.9f), modifier = Modifier.padding(end = 2.dp))
            Text(message.gemName, style = style.copy(color = Tw.purple300.copy(alpha = 0.9f)))
        }
    }
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

/** `underline decoration-dotted` for a single-line label. */
private fun Modifier.dottedUnderline(color: Color): Modifier = drawBehind {
    val y = size.height - 3.dp.toPx()
    drawLine(color, Offset(0f, y), Offset(size.width, y), strokeWidth = 1.dp.toPx(),
        pathEffect = PathEffect.dashPathEffect(floatArrayOf(1.dp.toPx(), 2.dp.toPx())))
}

/** `#total-token-bar`: coins glyph, path total and the all-branches total (both open the token details). */
@OptIn(ExperimentalLayoutApi::class)
@Composable
internal fun TotalTokenBar(path: TokenTotals, all: TokenTotals, onDetail: (TokenDetail) -> Unit) {
    val web = LocalWebPalette.current
    if (path.total <= 0 && all.total <= 0) return
    val text = if (web.isLight) web.text else Tw.slate200
    Column(Modifier.fillMaxWidth().background(if (web.isLight) Color.White.copy(alpha = 0.9f) else Color(8, 14, 28).copy(alpha = 0.55f))) {
        Row(Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 6.dp), verticalAlignment = Alignment.CenterVertically) {
            FaIcon(R.drawable.fa_solid_coins, null, size = 12.dp, tint = web.twText(Tw.amber300))
            Spacer(Modifier.width(8.dp))
            FlowRow(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                val style = TextStyle(color = text, fontSize = 12.sp, lineHeight = 16.sp, fontFamily = WebFonts.sans)
                Text("Total: ${path.total} tokens", style = style,
                    modifier = Modifier.dottedUnderline(text).clickable(role = Role.Button) { onDetail(path.detail(allBranches = false)) })
                if (all.total > 0) Text("All branches: ${all.total} tokens", style = style,
                    modifier = Modifier.dottedUnderline(text).clickable(role = Role.Button) { onDetail(all.detail(allBranches = true)) })
            }
        }
        Box(Modifier.fillMaxWidth().height(1.dp).background(if (web.isLight) Color(15, 23, 42).copy(alpha = 0.10f) else web.lineSoft))
    }
}

/** `#token-detail-modal`. */
@Composable
internal fun TokenDetailDialog(detail: TokenDetail, onDismiss: () -> Unit) {
    SmallInfoModal(onDismiss) {
        Text(detail.title, color = Tw.blue300, fontSize = 14.sp, lineHeight = 20.sp, fontWeight = FontWeight.Bold)
        Column(Modifier.padding(top = 12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            fun value(v: Int?) = v?.toString() ?: "-"
            listOf(
                "Total" to value(detail.total), "In" to value(detail.tokensIn), "Out" to value(detail.tokensOut),
                "Content" to value(detail.tokensContent), "Thought" to value(detail.tokensThought),
                "Encryption" to when (detail.encrypted) { null -> "-"; true -> "Encrypted"; false -> "Plain" },
            ).forEach { (label, v) -> InfoRow(label, v) }
        }
    }
}

/** `#encryption-status-modal` for a non-admin account. */
@Composable
internal fun EncryptionStatusDialog(encrypted: Boolean, onSettings: () -> Unit, onDismiss: () -> Unit) {
    val web = LocalWebPalette.current
    SmallInfoModal(onDismiss) {
        Text(if (encrypted) "暗号化されています" else "暗号化されていません", color = Tw.blue300, fontSize = 14.sp, lineHeight = 20.sp,
            fontWeight = FontWeight.Bold)
        Text(if (encrypted) "このメッセージはE2EEで暗号化されています。" else "このメッセージは暗号化されていません。",
            color = web.twText(Tw.gray300), fontSize = 12.sp, lineHeight = 16.sp, modifier = Modifier.padding(top = 12.dp))
        Row(Modifier.fillMaxWidth().padding(top = 16.dp), horizontalArrangement = Arrangement.spacedBy(8.dp, Alignment.End)) {
            SmallModalButton("閉じる", web.twBg(Tw.gray700), onDismiss)
            SmallModalButton("設定へ", Tw.blue600, onSettings)
        }
    }
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
private fun SmallInfoModal(onDismiss: () -> Unit, content: @Composable () -> Unit) {
    val web = LocalWebPalette.current
    WebOverlayModal(onDismiss, if (web.isLight) Color(0xFFF7F9FC) else Tw.gray900.copy(alpha = 0.9f), 4.dp) { _ ->
        val shape = RoundedCornerShape(12.dp)
        Box(
            Modifier
                .padding(16.dp)
                .widthIn(max = 384.dp)
                .fillMaxWidth()
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
 * `#welcome-screen`: gem heading with gradient text, subtitle and the quick-access model buttons
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
                    FaIcon(R.drawable.fa_solid_gem, null, size = 22.4.dp, tint = web.theme.t500)
                    val gradient = if (web.isLight) listOf(Color(0xFF0F766E), web.theme.t600, Color(0xFFB45309))
                        else listOf(Color(0xFFF8FAFC), web.theme.t200, Tw.amber200)
                    Text("AI Gems & Chat", fontSize = 22.4.sp, lineHeight = 32.sp, fontWeight = FontWeight.Bold, letterSpacing = (-0.035).em,
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
