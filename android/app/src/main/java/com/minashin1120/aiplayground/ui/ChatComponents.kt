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
import androidx.compose.foundation.clickable
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.liveRegion
import androidx.compose.ui.semantics.semantics
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
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
import androidx.compose.ui.unit.sp
import androidx.compose.ui.draw.drawBehind
import com.minashin1120.aiplayground.R
import com.minashin1120.aiplayground.data.CardKind
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
import com.minashin1120.aiplayground.data.numericId

/** Web `.mcp-box`: running (spinner, 実行中...), done (✓, 実行しました) or failed (✕, 失敗) with the dashed note. */
@Composable
private fun McpBox(card: StatusCard) {
    val (border, color) = when {
        card.failed -> Color(248, 113, 113).copy(alpha = 0.5f) to Color(0xFFFECACA)
        card.done -> Color(52, 211, 153).copy(alpha = 0.4f) to Color(0xFFA7F3D0)
        else -> Color(34, 211, 238).copy(alpha = 0.45f) to Color(0xFFA5F3FC)
    }
    val web = LocalWebPalette.current
    val shape = RoundedCornerShape(11.2.dp)
    Column(
        Modifier.fillMaxWidth().clip(shape).background(if (web.isLight) Color.White else Color(8, 14, 28).copy(alpha = 0.85f))
            .border(1.dp, border, shape).padding(horizontal = 12.dp, vertical = 8.dp),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            when {
                card.failed -> FaIcon(R.drawable.fa_solid_times_circle, null, size = 11.5.dp, tint = Color(0xFFF87171))
                card.done -> FaIcon(R.drawable.fa_solid_check_circle, null, size = 11.5.dp, tint = Color(0xFF34D399))
                else -> CircularProgressIndicator(Modifier.size(12.dp), strokeWidth = 2.dp, color = Color(0xFF22D3EE),
                    trackColor = Color(148, 163, 184).copy(alpha = 0.3f))
            }
            Text(card.label, fontSize = 11.5.sp, fontWeight = FontWeight.Bold, color = if (web.isLight) web.text else color)
            Text(if (card.failed) "失敗" else if (card.done) "実行しました" else "実行中...", fontSize = 10.9.sp, color = Color(0xFF94A3B8))
        }
        if (card.note.isNotBlank()) {
            val dash = Color(148, 163, 184).copy(alpha = 0.2f)
            Text(
                card.note, fontSize = 10.9.sp, color = if (card.failed) Color(0xFFFCA5A5) else Color(0xFF94A3B8),
                modifier = Modifier.padding(top = 4.dp).fillMaxWidth()
                    .drawBehind {
                        drawLine(dash, androidx.compose.ui.geometry.Offset.Zero, androidx.compose.ui.geometry.Offset(size.width, 0f), 1.dp.toPx(),
                            pathEffect = androidx.compose.ui.graphics.PathEffect.dashPathEffect(floatArrayOf(3.dp.toPx(), 3.dp.toPx())))
                    }
                    .padding(top = 4.8.dp),
            )
        }
    }
}

/**
 * Web `.coding-live-diff` ("Live Code Changes"): each Coding edit's diff as it arrives, with added, removed,
 * hunk and file lines coloured like `renderCodingDiffLines`.
 */
@Composable
fun CodingLiveDiff(edits: List<StatusCard>) {
    val shape = RoundedCornerShape(12.dp)
    Column(
        Modifier.padding(vertical = 12.dp).fillMaxWidth().clip(shape).background(Color(2, 18, 17).copy(alpha = 0.82f))
            .border(1.dp, Color(52, 211, 153).copy(alpha = 0.32f), shape),
    ) {
        Row(
            Modifier.fillMaxWidth().background(Color(6, 78, 59).copy(alpha = 0.28f)).padding(horizontal = 10.dp, vertical = 8.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            FaIcon(R.drawable.fa_solid_code_branch, null, size = 11.dp, tint = Color(0xFFA7F3D0), modifier = Modifier.padding(end = 4.dp))
            Text("Live Code Changes", fontSize = 11.sp, fontWeight = FontWeight.Bold, color = Color(0xFFA7F3D0), modifier = Modifier.weight(1f))
            Text("${edits.size} edit${if (edits.size == 1) "" else "s"}", fontSize = 10.sp, fontWeight = FontWeight.Medium, color = Color(0xFF6EE7B7))
        }
        Column(Modifier.heightIn(max = 420.dp).verticalScroll(rememberScrollState())) {
            edits.forEachIndexed { index, edit ->
                if (index > 0) Box(Modifier.fillMaxWidth().height(1.dp).background(Color(148, 163, 184).copy(alpha = 0.16f)))
                Text(edit.label, fontSize = 10.sp, lineHeight = 14.sp, fontFamily = WebFonts.mono, color = Color(0xFF94A3B8),
                    modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp))
                Column(Modifier.fillMaxWidth().background(Color(2, 6, 23).copy(alpha = 0.58f)).horizontalScroll(rememberScrollState())
                    .padding(top = 8.dp, bottom = 10.dp)) {
                    edit.output.split('\n').forEach { line ->
                        val (color, background) = when {
                            line.startsWith("+++") || line.startsWith("---") -> Color(0xFF7DD3FC) to Color.Transparent
                            line.startsWith("@@") -> Color(0xFFC4B5FD) to Color(124, 58, 237).copy(alpha = 0.1f)
                            line.startsWith("+") -> Color(0xFFBBF7D0) to Color(22, 163, 74).copy(alpha = 0.16f)
                            line.startsWith("-") -> Color(0xFFFECACA) to Color(220, 38, 38).copy(alpha = 0.14f)
                            else -> Color(0xFFCBD5E1) to Color.Transparent
                        }
                        Text(line.ifEmpty { " " }, fontSize = 11.sp, lineHeight = 17.sp, fontFamily = WebFonts.mono, color = color, softWrap = false,
                            fontWeight = if (line.startsWith("+++") || line.startsWith("---")) FontWeight.Bold else null,
                            modifier = Modifier.background(background).padding(horizontal = 10.dp))
                    }
                }
            }
        }
    }
}

/**
 * The Web pending / streaming answer bubble (`sendMessage`): the skeleton with its status until the first
 * answer event, then the search box, Python boxes, image analysis and "Thinking Process" above the
 * streamed text, MCP cards after it and a stream error at the end.
 */
@Composable
fun LiveMessage(state: ChatState, onFile: (String) -> Unit, onQuote: (String) -> Unit,
                loader: FileBytesLoader? = null, onMcpDecision: (Boolean) -> Unit = {}) {
    val live = state.live
    val thought = state.liveThought.ifBlank { live.thoughtPlaceholder.orEmpty() }
    val python = state.cards.filter { it.kind == CardKind.PYTHON }
    val mcp = state.cards.filter { it.kind == CardKind.MCP }
    val coding = state.cards.filter { it.kind == CardKind.CODING }
    val pending = live.pendingStatus
    val skeleton: (@Composable () -> Unit)? =
        if (pending != null) { { PendingSkeleton(live.model.ifBlank { state.model }, pending, live.pendingSub) } } else null
    MessageBubble(
        ChatMessage("live", "assistant", state.liveContent, thought), onFile, loader,
        MessageActions(), controlsVisible = false, onToggleControls = {}, streaming = true,
        thoughtCollapsed = state.liveThought.isBlank(),
        liveTop = {
            if (live.search.isNotEmpty()) LiveSearchBox(live.search)
            python.asReversed().forEach { card -> key(card.id) { PythonExecutionBox(card) } }
            live.imageAnalysis?.let { LiveImageAnalysis(it) }
        },
        liveSkeleton = skeleton,
        liveBottom = {
            if (coding.isNotEmpty()) CodingLiveDiff(coding)
            if (mcp.isNotEmpty()) Column(Modifier.padding(top = 12.dp)) {
                mcp.forEach { card -> key(card.id) { Box(Modifier.padding(bottom = 8.dp)) { McpBox(card) } } }
            }
            live.error?.let { Box(Modifier.padding(top = 8.dp)) { ChatErrorBlock(it) } }
        },
    )
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

/** Web `#batch-notification-banner`: a finished Batch job, with 開く (opens its chat) and ×. */
@Composable
internal fun BatchCompletionBanner(text: String, onOpen: () -> Unit, onClose: () -> Unit) {
    val web = LocalWebPalette.current
    val shape = RoundedCornerShape(12.dp)
    Box(Modifier.fillMaxWidth(), contentAlignment = Alignment.Center) {
        Row(
            Modifier.padding(12.dp).widthIn(max = 544.dp).fillMaxWidth().clip(shape)
                .background(if (web.isLight) Color(0xFFF5F3FF) else Color(46, 16, 101).copy(alpha = 0.95f))
                .border(1.dp, Tw.violet400.copy(alpha = 0.4f), shape)
                .padding(horizontal = 16.dp, vertical = 12.dp)
                .semantics { liveRegion = androidx.compose.ui.semantics.LiveRegionMode.Polite },
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            val fg = if (web.isLight) Color(0xFF4C1D95) else Color(0xFFEDE9FE)
            FaIcon(R.drawable.fa_solid_layer_group, null, size = 14.dp, tint = if (web.isLight) Color(0xFF6D28D9) else Tw.violet300)
            Text(text, fontSize = 14.sp, lineHeight = 20.sp, color = fg, modifier = Modifier.weight(1f))
            Text("開く", fontSize = 12.sp, color = fg,
                modifier = Modifier.clip(RoundedCornerShape(8.dp)).background(Color(139, 92, 246).copy(alpha = 0.3f))
                    .clickable(role = Role.Button, onClick = onOpen).padding(horizontal = 10.dp, vertical = 4.dp))
            Box(Modifier.size(24.dp).clip(CircleShape).clickable(role = Role.Button, onClick = onClose), contentAlignment = Alignment.Center) {
                FaIcon(R.drawable.fa_solid_times, "閉じる", size = 12.dp, tint = (if (web.isLight) Color(0xFF6D28D9) else Tw.violet200).copy(alpha = 0.7f))
            }
        }
    }
}
