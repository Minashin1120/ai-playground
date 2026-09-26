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
        // MCP confirmations use the Web modal (`McpDecisionDialog`).
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            if (state.streaming) CircularProgressIndicator(Modifier.size(16.dp), strokeWidth = 2.dp)
            AnimatedContent(
                targetState = state.status.ifBlank { "回答を生成中..." },
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
            MessageBubble(ChatMessage("live", "assistant", state.liveContent, state.liveThought), onFile, loader,
                MessageActions(), controlsVisible = false, onToggleControls = {}, streaming = true)
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
