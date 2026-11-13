package com.minashin1120.aiplayground.ui

import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.EnterTransition
import androidx.compose.animation.ExitTransition
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.slideOutVertically
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.foundation.gestures.scrollBy
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.nestedscroll.NestedScrollConnection
import androidx.compose.ui.input.nestedscroll.NestedScrollSource
import androidx.compose.ui.input.nestedscroll.nestedScroll
import androidx.compose.ui.input.pointer.PointerEventPass
import androidx.compose.ui.input.pointer.PointerInputChange
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.boundsInWindow
import androidx.compose.ui.layout.onGloballyPositioned
import androidx.compose.ui.layout.positionInWindow
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.minashin1120.aiplayground.ChatViewModel
import com.minashin1120.aiplayground.R
import com.minashin1120.aiplayground.data.ActivityLog
import com.minashin1120.aiplayground.data.ChatAutoScroll
import com.minashin1120.aiplayground.data.ChatMessage
import com.minashin1120.aiplayground.data.MathStepsConfig
import com.minashin1120.aiplayground.data.buildTokenTotals
import com.minashin1120.aiplayground.data.numericId
import com.minashin1120.aiplayground.data.recentWebModels
import com.minashin1120.aiplayground.data.siblingGroup
import com.minashin1120.aiplayground.ChatState
import com.minashin1120.aiplayground.ChatTransitionKind
import java.util.LinkedHashMap
import java.util.UUID
import kotlin.math.abs
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.launch

// region: Conversation screen
/** Top-level conversation area: fades between chats and renders the message list. */
@Composable
internal fun Conversation(
    state: ChatState,
    model: ChatViewModel,
    onFile: (String) -> Unit,
    loader: FileBytesLoader?,
    animationsEnabled: Boolean,
    bottomOverlay: androidx.compose.ui.unit.Dp = 0.dp,
    onSettingJump: (String) -> Unit = {},
) {
    // What each chat pane last showed, so a leaving pane keeps its own chat while the next one fades in.
    val shownByTransition = remember { LinkedHashMap<Long, ChatState>() }
    shownByTransition.remove(state.chatTransitionId)
    shownByTransition[state.chatTransitionId] = state
    while (shownByTransition.size > 4) shownByTransition.remove(shownByTransition.keys.first())
    val slideOffsetPx = with(LocalDensity.current) { 22.dp.roundToPx() }
    AnimatedContent(
        targetState = state.chatTransitionId,
        modifier = Modifier.fillMaxSize(),
        transitionSpec = {
            if (!animationsEnabled || targetState == initialState) {
                EnterTransition.None togetherWith ExitTransition.None
            } else {
                // Symmetric dissolve with a shared vertical drift so the swap reads as one motion.
                (fadeIn(animationSpec = tween(260, easing = FastOutSlowInEasing)) +
                    slideInVertically(animationSpec = tween(260, easing = FastOutSlowInEasing)) { slideOffsetPx }) togetherWith
                    (fadeOut(animationSpec = tween(260, easing = FastOutSlowInEasing)) +
                        slideOutVertically(animationSpec = tween(260, easing = FastOutSlowInEasing)) { -slideOffsetPx / 2 })
            }
        },
        label = "chat conversation transition",
    ) { transitionId ->
        val contentState = if (transitionId == state.chatTransitionId) state else shownByTransition[transitionId] ?: state
        ProvideQuoteSelection({ quoted -> model.quoteMessage(quoted) }) { ConversationContent(contentState, model, onFile, loader, onSettingJump, bottomOverlay) }
    }
}
// endregion

// region: Chat transition veil
/** Duration of the Web-parity chat transition veil. */
private const val CHAT_TRANSITION_VEIL_MS = 460

/**
 * Piecewise-linear opacity keyframes matching the Web CSS `chatTransitionVeil` / `chatTransitionNew`.
 * New-chat transitions peak earlier and higher than history transitions.
 */
internal fun chatTransitionVeilAlpha(progress: Float, newChat: Boolean): Float {
    val frames = if (newChat) {
        listOf(0f to 0f, 0.36f to 0.58f, 1f to 0f)
    } else {
        listOf(0f to 0f, 0.30f to 0.22f, 0.52f to 0.72f, 1f to 0f)
    }
    val p = progress.coerceIn(0f, 1f)
    for (index in 0 until frames.lastIndex) {
        val (start, startAlpha) = frames[index]
        val (end, endAlpha) = frames[index + 1]
        if (p <= end) return startAlpha + (endAlpha - startAlpha) * ((p - start) / (end - start))
    }
    return frames.last().second
}

/**
 * A quiet theme-colored sweep over the whole screen when a history item is opened or a new chat
 * starts, mirroring the Web `#chat-transition-veil` motion so the swap reads as one continuous move.
 */
@Composable
internal fun ChatTransitionVeil(
    navigationId: Long,
    kind: ChatTransitionKind,
    animationsEnabled: Boolean,
) {
    val accent = MaterialTheme.colorScheme.primary
    val progress = remember { Animatable(0f) }
    var active by remember { mutableStateOf(false) }
    LaunchedEffect(navigationId) {
        if (navigationId <= 0L || !animationsEnabled) return@LaunchedEffect
        active = true
        progress.snapTo(0f)
        progress.animateTo(
            targetValue = 1f,
            animationSpec = tween(durationMillis = CHAT_TRANSITION_VEIL_MS, easing = LinearEasing),
        )
        active = false
    }
    if (!active) return
    val alpha = chatTransitionVeilAlpha(progress.value, kind == ChatTransitionKind.NEW_CHAT)
    if (alpha <= 0.001f) return
    Canvas(Modifier.fillMaxSize()) {
        val width = size.width
        val center = width * (-0.7f + 1.9f * progress.value)
        val half = width * 0.8f
        drawRect(
            brush = Brush.horizontalGradient(
                colors = listOf(Color.Transparent, accent.copy(alpha = 0.30f), Color.Transparent),
                startX = center - half,
                endX = center + half,
            ),
            alpha = alpha,
        )
    }
}
// endregion

// region: Conversation content
@Composable
private fun ConversationContent(
    state: ChatState,
    model: ChatViewModel,
    onFile: (String) -> Unit,
    loader: FileBytesLoader?,
    onSettingJump: (String) -> Unit = {},
    bottomOverlay: androidx.compose.ui.unit.Dp = 0.dp,
) {
    val reduce = LocalReduceMotion.current
    val scroll = rememberLazyListState()
    val sendMotion = LocalPromptSendMotion.current
    val scope = rememberCoroutineScope()
    val keys = remember { ConversationKeyTracker() }
    val answerMotions = remember(state.chatNavigationId) { AnswerBubbleMotions() }
    val streamActive = state.streaming || state.liveContent.isNotEmpty() || state.liveThought.isNotEmpty() || state.cards.isNotEmpty()
    // Web `buildChatLoadingSkeletonHtml` in `#chat-container` while an opened chat's history loads.
    val loadingThread = state.threadLoadingId != 0L && state.threadLoadingId == state.chatTransitionId && state.messages.isEmpty()
    keys.update(state.messages, state.streaming, streamActive, loadingHistory = loadingThread)
    // Once the stored reply has taken over the streamed row, the placeholder must not reappear.
    val live = streamActive && !keys.liveConsumed
    // Web `part05` auto-scroll, restarted for every opened or new chat (Web `resumeChatAutoScroll` in `loadMessages`).
    val thresholdPx = with(LocalDensity.current) { 64.dp.toPx() }
    val follow = remember(state.chatNavigationId) { ChatAutoScroll(thresholdPx) }
    fun distanceToBottom(): Float {
        val info = scroll.layoutInfo
        val last = info.visibleItemsInfo.lastOrNull() ?: return 0f
        if (last.index < info.totalItemsCount - 1) return Float.MAX_VALUE
        return (last.offset + last.size + info.afterContentPadding - info.viewportEndOffset).toFloat().coerceAtLeast(0f)
    }
    // Web `performChatAutoScroll` (`scrollTop = scrollHeight`): the very end, even inside a long last message.
    suspend fun jumpToBottom() {
        val last = scroll.layoutInfo.totalItemsCount - 1
        if (last < 0) return
        try {
            scroll.scrollToItem(last)
            scroll.scrollBy(1_000_000f)
        } catch (e: CancellationException) {
            // A finger on the list wins over the programmatic scroll; only a cancelled effect stops here.
            currentCoroutineContext().ensureActive()
        }
    }
    // Web ResizeObserver / MutationObserver: new rows, streamed text, opened thoughts and loaded images.
    LaunchedEffect(follow) {
        snapshotFlow {
            val info = scroll.layoutInfo
            val last = info.visibleItemsInfo.lastOrNull()
            Triple(info.totalItemsCount, last?.index, last?.size)
        }.collect { if (follow.following) jumpToBottom() }
    }
    LaunchedEffect(follow) { snapshotFlow { distanceToBottom() }.collect { follow.onPosition(it) } }
    // Preserve a reader's paused position when sending; only existing followers jump.
    LaunchedEffect(state.streaming) { if (state.streaming && follow.following) jumpToBottom() }
    SideEffect { sendMotion?.following = follow.following }
    DisposableEffect(follow) { onDispose { sendMotion?.finish() } }
    val quoteFocusManager = LocalFocusManager.current
    // The row the last quote jump landed on and the selection to flash there; the token restarts it.
    var quoteFlashRow by remember(state.chatNavigationId) { mutableStateOf<String?>(null) }
    var quoteFlashText by remember(state.chatNavigationId) { mutableStateOf("") }
    var quoteFlashToken by remember(state.chatNavigationId) { mutableLongStateOf(0L) }
    // The token whose quoted range was already aligned under the top bar; later layout reports are ignored.
    var quoteAlignedToken by remember(state.chatNavigationId) { mutableLongStateOf(-1L) }
    var listTopWindow by remember(state.chatNavigationId) { mutableFloatStateOf(0f) }
    val quoteTopMarginPx = with(LocalDensity.current) { QUOTE_JUMP_TOP_MARGIN_DP.dp.toPx() }
    fun markQuoteFlash(row: Int, quote: String?) {
        if (quote.isNullOrBlank()) return
        quoteFlashRow = state.messages.getOrNull(row)?.let { keys.keyAt(row, it) } ?: keys.liveKey
        quoteFlashText = quote.trim()
        quoteFlashToken += 1
    }
    /**
     * Web has no quote jump. Android first lands on the message row, then moves its quoted range to just
     * under the token bar; landing on the row alone can leave a selection from a long answer below the
     * composer, where its flash cannot be seen.
     */
    fun alignQuoteUnderTopBar(windowY: Float) {
        val token = quoteFlashToken
        if (token == 0L || token == quoteAlignedToken) return
        quoteAlignedToken = token
        val delta = quoteScrollDelta(windowY, listTopWindow, quoteTopMarginPx)
        if (abs(delta) < 0.5f) return
        scope.launch {
            try { scroll.scrollBy(delta) }
            catch (e: CancellationException) { throw e }
            catch (e: Exception) {}
        }
    }
    /** Shared by the composer quote bar and a sent message's quote strip. */
    suspend fun scrollToQuoteRow(row: Int) {
        quoteFocusManager.clearFocus()
        follow.pause()
        val index = row + (if (loadingThread) 1 else 0) + (if (state.hasOlder) 1 else 0)
        if (reduce) scroll.scrollToItem(index) else scroll.animateScrollToItem(index)
    }
    fun requestQuoteRow(row: Int, quote: String? = null) {
        val op = UUID.randomUUID().toString()
        fun trace(phase: String) = ActivityLog.log("ui.interaction",
            "op" to op, "ui_element" to "message", "interaction_stage" to "scroll", "phase" to phase)
        trace("start")
        scope.launch {
            try {
                scrollToQuoteRow(row)
                trace("done")
                markQuoteFlash(row, quote)
            }
            catch (e: CancellationException) { trace("cancelled"); throw e }
            catch (e: Exception) { trace("error") }
        }
    }
    LaunchedEffect(state.quoteNavigationRequest) {
        if (state.quoteNavigationRequest == 0L) return@LaunchedEffect
        val target = resolveQuoteTarget(state.messages, state.quoteSourceId, state.quoteMessageId,
            keys.liveKey, live) { keys.keyOf(it) }
        // A draft edit restores the message's quote without its source id; place it by the quote text,
        // searching the messages before the edited one (or the whole chat for a selection quote).
        val resolved = if (target is QuoteTarget.Missing) {
            val before = state.editingMessageId
                ?.let { id -> state.messages.indexOfFirst { it.id == id } }
                ?.takeIf { it >= 0 }
                ?: state.messages.size
            findQuoteTextIndex(state.messages, before, state.quote)?.let { QuoteTarget.Message(it) }
                ?: QuoteTarget.Missing
        } else target
        val op = UUID.randomUUID().toString()
        fun trace(phase: String) = ActivityLog.log("ui.interaction",
            "op" to op, "ui_element" to "message", "interaction_stage" to "scroll", "phase" to phase)
        trace("start")
        val row = when (resolved) {
            is QuoteTarget.Message -> resolved.index
            // The streamed answer is not in `messages` yet; it is the row after them.
            QuoteTarget.Live -> state.messages.size
            QuoteTarget.Missing -> { trace("error"); return@LaunchedEffect }
        }
        try {
            scrollToQuoteRow(row)
            trace("done")
            markQuoteFlash(row, state.quote)
        } catch (e: CancellationException) { trace("cancelled"); throw e }
        catch (e: Exception) { trace("error") }
    }
    val showScrollToBottom by remember(follow) { derivedStateOf { follow.showsJumpButton(distanceToBottom()) } }
    val followConnection = remember(follow) {
        object : NestedScrollConnection {
            override fun onPreScroll(available: Offset, source: NestedScrollSource): Offset {
                if (source == NestedScrollSource.UserInput && available.y != 0f) {
                    sendMotion?.finish()
                    follow.onUserScroll(towardTop = available.y > 0f, canScrollBackward = scroll.canScrollBackward)
                }
                return Offset.Zero
            }
        }
    }
    val temporary = state.selected?.isTemporary == true || (state.selected == null && state.newThreadTemporary)
    val loadFailed = !loadingThread && state.threadLoadFailedId != null && state.threadLoadFailedId == state.selected?.id &&
        state.messages.isEmpty() && !live
    // Web shows the hover controls of the bubble the user last tapped.
    var activeMessageId by remember { mutableStateOf<String?>(null) }
    var deletingMessage by remember { mutableStateOf<ChatMessage?>(null) }
    var tokenDetail by remember { mutableStateOf<TokenDetail?>(null) }
    var processingDetail by remember { mutableStateOf<com.minashin1120.aiplayground.data.ProcessingTime?>(null) }
    var pythonRuns by remember { mutableStateOf<List<com.minashin1120.aiplayground.data.PythonExecution>?>(null) }
    var encryptionState by remember { mutableStateOf<Boolean?>(null) }
    // Admin decrypt / re-encrypt from the lock modal: the pending confirm, the request and a re-auth retry.
    var encryptionConfirm by remember { mutableStateOf<Boolean?>(null) }
    var encryptionBusy by remember { mutableStateOf(false) }
    var encryptionReauth by remember { mutableStateOf<(() -> Unit)?>(null) }
    // Choosing a quick-access model hides the welcome screen until the next chat, as on Web.
    var welcomeDismissed by remember(state.chatTransitionId) { mutableStateOf(false) }
    val pathTotals = remember(state.messages) { buildTokenTotals(state.messages) }
    val costContext = LocalContext.current
    val costPricing = remember(costContext) {
        com.minashin1120.aiplayground.data.ModelCostEstimator(costContext.assets.open("model-pricing.json").bufferedReader().use { it.readText() })
    }
    val pathCost = remember(state.messages, state.allMessages, costPricing) {
        if (state.messages.any { it.role == "assistant" && (it.content.isNotEmpty() || (it.tokensOut ?: 0) > 0) })
            costPricing.estimate(state.messages, state.allMessages.ifEmpty { state.messages }) else null
    }
    var branchingMessage by remember { mutableStateOf<ChatMessage?>(null) }
    val actions = MessageActions(
        onEdit = { model.beginEdit(it) },
        onRegenerate = { model.regenerate(it) },
        onDelete = { deletingMessage = it },
        onBranch = { if (state.selected?.isTemporary == true) branchingMessage = it else model.branchChat(it) },
        onTokenDetail = { tokenDetail = it },
        onProcessingTime = { processingDetail = it },
        onEncryption = { encryptionState = it },
        onPython = { runs -> if (runs.isEmpty()) model.notify("Python実行結果がありません") else pythonRuns = runs },
    )
    Column(Modifier.fillMaxSize()) {
        state.branchOrigin?.let { origin ->
            val palette = LocalWebPalette.current
            Column(Modifier.fillMaxWidth()) {
                Text(if (origin.optBoolean("available")) "分岐元のチャットを開く" else "分岐元のチャットは利用できません",
                    fontSize = 12.sp, lineHeight = 16.sp,
                    color = if (origin.optBoolean("available")) palette.twText(Tw.blue400) else palette.twText(Tw.gray400),
                    modifier = Modifier.fillMaxWidth().clickable(enabled = origin.optBoolean("available")) { model.openBranchOrigin() }
                        .padding(horizontal = 16.dp, vertical = 8.dp))
                Box(Modifier.fillMaxWidth().height(1.dp).background(if (palette.isLight) Color(15, 23, 42).copy(alpha = 0.10f) else Tw.gray700))
            }
        }
        CompositionLocalProvider(LocalComposerEstimator provides model::estimatePromptTokens,
            LocalComposerDetailsEstimator provides model::estimatePromptTokenDetails) {
            val hasDraft = state.editingAttachment == null && (state.draft.isNotBlank() || state.quote.isNotBlank() || state.attachments.isNotEmpty())
            TotalTokenBar(pathTotals, showEstimate = hasDraft,
                pathCost = pathCost, pricing = costPricing,
                estimate = { PromptTokenEstimate(state) }) { tokenDetail = it }
        }
        state.batchBanner?.let { (text, _) ->
            BatchCompletionBanner(text, onOpen = { model.dismissBatchBanner(open = true) }, onClose = { model.dismissBatchBanner(open = false) })
        }
        val screenWidth = LocalConfiguration.current.screenWidthDp.dp
        val sideCanvas = state.canvasMode && screenWidth >= CANVAS_SIDE_PANEL_MIN_WIDTH
        Row(Modifier.weight(1f).fillMaxWidth()) {
        val focusManager = LocalFocusManager.current
        Box(Modifier.weight(1f).fillMaxHeight().onBackgroundTap(onTap = { focusManager.clearFocus() }, onUnhandledTap = { activeMessageId = null },
            listPosition = { scroll.firstVisibleItemIndex to scroll.firstVisibleItemScrollOffset })) {
            LazyColumn(
                state = scroll,
                modifier = Modifier.fillMaxSize().widthIn(max = 832.dp).align(Alignment.TopCenter).nestedScroll(followConnection)
                    .onGloballyPositioned { coordinates ->
                        val next = coordinates.boundsInWindow()
                        val old = sendMotion?.viewport
                        if (old != null && old.width != next.width) sendMotion?.finish()
                        sendMotion?.viewport = next
                        // Quote jumps align the flashed range under this viewport's top edge.
                        listTopWindow = coordinates.positionInWindow().y
                    },
                // `#chat-container`: 12px padding on phones, 20px between message groups.
                contentPadding = PaddingValues(start = 12.dp, end = 12.dp, top = 12.dp, bottom = 12.dp + bottomOverlay),
                verticalArrangement = Arrangement.spacedBy(20.dp),
            ) {
            if (loadingThread) item(key = CHAT_LOAD_SKELETON_KEY) { ChatLoadingSkeleton() }
            if (state.hasOlder) item(key = "older") {
                Box(Modifier.fillMaxWidth().padding(bottom = 12.dp), contentAlignment = Alignment.Center) {
                    val web = LocalWebPalette.current
                    val enabled = !state.busy && !state.offline
                    Row(
                        Modifier
                            .clip(RoundedCornerShape(4.dp))
                            .border(1.dp, web.twBorder(Tw.gray600), RoundedCornerShape(4.dp))
                            .clickable(enabled = enabled, onClick = model::olderMessages)
                            .graphicsLayer { alpha = if (enabled) 1f else 0.5f }
                            .padding(horizontal = 12.dp, vertical = 6.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        FaIcon(R.drawable.fa_solid_clock_rotate_left, null, size = 12.dp,
                            tint = web.twText(Tw.gray200), modifier = Modifier.padding(end = 4.dp))
                        Text(if (state.busy) "読み込み中..." else "過去メッセージを読み込む", color = web.twText(Tw.gray200), fontSize = 12.sp, lineHeight = 16.sp)
                    }
                }
            }
            itemsIndexed(state.messages, key = { row, message -> keys.keyAt(row, message) }) { row, message ->
                val rowKey = keys.keyAt(row, message)
                val entering = remember { keys.consumeFresh(rowKey) }
                val siblings = siblingGroup(state.allMessages, message)
                val index = siblings.indexOfFirst { it.id == message.id }
                val parentMathSteps = message.parentId?.let { parentId ->
                    state.allMessages.firstOrNull { it.id.toIntOrNull() == parentId }
                        ?.promptOptions?.optBoolean("math_steps") == true
                } == true
                val hasMathStepHeadings = remember(message.content) {
                    message.role == "assistant" &&
                        com.minashin1120.aiplayground.data.parseMathSteps(message.content).isNotEmpty()
                }
                Box(Modifier.animateItem(fadeInSpec = null, placementSpec = listPlacement(reduce), fadeOutSpec = null)) {
                    ConversationMessageEntry(rowKey, answer = message.role == "assistant",
                        animate = entering && message.id != state.promptSendMotionId, answers = answerMotions) {
                        // Web sends `quote_message_id`: a selection inside this bubble remembers which message it came from.
                        ProvideQuoteSelection({ quoted ->
                            model.quoteMessage(quoted, numericId(message), rowKey)
                        }) {
                        MessageBubble(
                            message = message,
                            sendMotion = promptSendBubbleModifier(message.id),
                            onFile = onFile,
                            loader = loader,
                            onQuoteJump = if (message.quote.isBlank()) null else {
                                { findQuotedMessageIndex(state.messages, row, message.quote)?.let { requestQuoteRow(it, quote = message.quote) } }
                            },
                            flashToken = if (quoteFlashRow == rowKey) quoteFlashToken else 0L,
                            flashQuote = if (quoteFlashRow == rowKey) quoteFlashText else null,
                            onQuoteFlashLocated = if (quoteFlashRow == rowKey) { y -> alignQuoteUnderTopBar(y) } else null,
                            actions = MessageActions(
                                onEdit = actions.onEdit, onRegenerate = actions.onRegenerate, onDelete = actions.onDelete, onBranch = actions.onBranch,
                                canBranch = !state.busy && !state.streaming && !state.branchBusy,
                                onSwitchBranch = { target -> model.switchBranchByIndex(siblings, target) },
                                onTokenDetail = actions.onTokenDetail, onProcessingTime = actions.onProcessingTime,
                                onEncryption = actions.onEncryption,
                                onPython = actions.onPython,
                            ),
                            controlsVisible = activeMessageId == message.id,
                            preserveLiveLayout = keys.isLiveReply(message),
                            thoughtCollapsed = keys.isLiveReply(message),
                            // Web's tap-to-hover: tapping the bubble again keeps its controls; a tap elsewhere hides them.
                            onToggleControls = { activeMessageId = message.id },
                            branchIndex = if (index < 0) 0 else index,
                            branchCount = if (numericId(message) != null) siblings.size else 0,
                            mathSteps = message.role == "assistant" &&
                                (hasMathStepHeadings ||
                                    message.promptOptions?.optBoolean("math_steps") == true ||
                                    parentMathSteps ||
                                    message.mathStepsState != null ||
                                    (keys.isLiveReply(message) && state.mathStepsMode)),
                            mathStepsActions = MathStepsActions(
                                onProgress = { target, step, max -> model.saveMathStepProgress(target, step, max) },
                                onSimplify = { target, step, done -> model.simplifyMathStep(target, step, done) },
                                onRestoreOriginal = { target, step -> model.clearMathStepSimplified(target, step) },
                            ),
                            mathStepsConfig = state.preferences?.mathStepsConfig ?: MathStepsConfig(),
                        )
                        }
                    }
                }
            }
                if (live) item(key = keys.liveKey) {
                    val entering = remember { keys.consumeFresh(keys.liveKey) }
                    Box(Modifier.animateItem(fadeInSpec = null, placementSpec = listPlacement(reduce), fadeOutSpec = null)) {
                        ConversationMessageEntry(keys.liveKey, answer = true, animate = entering, answers = answerMotions) {
                            LiveMessage(state, onFile, { quoted -> model.quoteMessage(quoted, null, keys.liveKey) }, loader, model::resolveMcpDecision)
                        }
                    }
                }
                items(state.settingsBubbles, key = { it.id }) { bubble ->
                    Box(Modifier.animateItem(fadeInSpec = null, placementSpec = listPlacement(reduce), fadeOutSpec = null)) {
                        SettingsBubbleView(bubble, state.model, onFile, loader, onSettingJump)
                    }
                }
            }
            if (loadFailed) ChatLoadError(onRetry = model::retryOpenThread, modifier = Modifier.align(Alignment.TopCenter))
            val showWelcome = state.messages.isEmpty() && !state.busy && !loadingThread && !loadFailed && !live && !welcomeDismissed
            androidx.compose.animation.AnimatedVisibility(
                visible = showWelcome,
                modifier = Modifier.padding(bottom = bottomOverlay),
                enter = fadeIn(motionTween(reduce)),
                exit = fadeOut(motionTween(reduce, PlaygroundMotion.SHORT)),
                label = "welcome",
            ) {
                if (temporary) TemporaryChatWelcome(state.preferences?.tempChatTimeoutSeconds ?: 90,
                    onSettings = { model.requestSettings(SettingsTab.General.id, "temp-chat") })
                else WelcomeScreen(
                    recentWebModels(state.account?.models.orEmpty()).map { info -> info.id to "${info.emoji} ${info.name}".trim() },
                    onChoose = { id -> model.chooseModelByUser(id); welcomeDismissed = true },
                )
            }
            // Qualified so the outer Column's scoped overload is not picked inside this Box.
            androidx.compose.animation.AnimatedVisibility(
                visible = showScrollToBottom && !showWelcome,
                modifier = Modifier.align(Alignment.BottomCenter).padding(bottom = 16.dp + bottomOverlay),
                enter = popIn(reduce),
                exit = popOut(reduce),
                label = "scroll to bottom",
            ) {
                // Web `scrollToBottom(true)`: back to the very bottom, following again.
                ScrollToBottomPill(onClick = { follow.resume(); scope.launch { jumpToBottom() } })
            }
        }
        if (sideCanvas) CanvasPanel(canvasSourceText(state), fullScreen = false, notify = model::notify, onClose = model::toggleCanvas,
            modifier = Modifier.width(canvasSidePanelWidth(screenWidth)).fillMaxHeight())
        }
    }
    branchingMessage?.let { message ->
        NoticeAlertDialog(onDismissRequest = { branchingMessage = null },
            title = { Text("新しいチャットに分岐") },
            text = { Text(com.minashin1120.aiplayground.data.BRANCH_SAVE_WARNING) },
            confirmButton = { TextButton(onClick = { branchingMessage = null; model.branchChat(message, confirmed = true) }) { Text("保存して分岐") } },
            dismissButton = { TextButton(onClick = { branchingMessage = null }) { Text("キャンセル") } })
    }
    deletingMessage?.let { message ->
        BrowserConfirmDialog("Delete this message and subsequent history?") { ok ->
            if (ok) model.deleteMessage(message)
            deletingMessage = null
        }
    }
    ModalValueHost(tokenDetail) { detail -> TokenDetailDialog(detail, onDismiss = { tokenDetail = null }) }
    ModalValueHost(processingDetail) { detail -> ProcessingTimeDialog(detail, onDismiss = { processingDetail = null }) }
    ModalValueHost(pythonRuns) { runs -> PythonExecutionDialog(runs, onDismiss = { pythonRuns = null }, loader = loader, onFile = onFile) }
    ModalValueHost(encryptionState) { encrypted ->
        EncryptionStatusDialog(
            encrypted,
            admin = state.preferences?.isAdmin == true,
            canToggle = state.preferences?.isAdmin == true && state.selected != null && !state.offline,
            busy = encryptionBusy,
            onToggle = { encryptionConfirm = !encrypted },
            // Web `goToEncryptionSettings`: the security tab, scrolled to the E2EE card.
            onSettings = { encryptionState = null; model.requestSettings(SettingsTab.Security.id, "e2ee") },
            onDismiss = { encryptionState = null },
        )
    }
    encryptionConfirm?.let { enable ->
        BrowserConfirmDialog("このチャットを${if (enable) "再暗号化" else "復号化"}しますか？") { ok ->
            encryptionConfirm = null
            if (ok) {
                encryptionBusy = true
                model.setThreadEncryption(enable,
                    onReauth = { retry -> encryptionReauth = { encryptionBusy = true; retry() } },
                    onDone = { done -> encryptionBusy = false; if (done) encryptionState = null })
            }
        }
    }
    encryptionReauth?.let { retry ->
        ReauthDialog(model::accountApi, onDismiss = { encryptionReauth = null }) {
            encryptionReauth = null
            retry()
        }
    }
}
// endregion

// region: Temporary chat welcome
/** `#welcome-temporary-content`: amber card that replaces the welcome screen in a temporary chat. */
@Composable
private fun TemporaryChatWelcome(timeoutSeconds: Int, onSettings: () -> Unit) {
    val web = LocalWebPalette.current
    val seconds = timeoutSeconds.coerceIn(10, 3600)
    Box(Modifier.fillMaxSize().padding(horizontal = 8.dp, vertical = 24.dp), contentAlignment = Alignment.Center) {
        Column(
            Modifier
                .widthIn(max = 448.dp)
                .padding(horizontal = 16.dp)
                .clip(RoundedCornerShape(12.dp))
                .background(web.twBg(Tw.amber900, 0.2f))
                .border(1.dp, Tw.amber500.copy(alpha = 0.5f), RoundedCornerShape(12.dp))
                .padding(16.dp),
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                FaIcon(R.drawable.fa_solid_user_secret, null, size = 14.dp, tint = web.twText(Tw.amber300),
                    modifier = Modifier.padding(end = 8.dp))
                Text("一時チャットモード", color = web.twText(Tw.amber300), fontSize = 14.sp, lineHeight = 20.sp, fontWeight = FontWeight.Bold)
            }
            Text(
                "このページが非表示/切断の状態で $seconds 秒経過すると、この一時チャットとこのチャットでアップロードした添付を自動削除します（ライブラリ添付は除外）。",
                color = if (web.isLight) Color(0xFF92400E) else Color(0xFFFEF3C7).copy(alpha = 0.9f), fontSize = 12.sp, lineHeight = 19.5.sp,
                modifier = Modifier.padding(top = 8.dp),
            )
            // `openTemporaryChatSettings()` button: `border-amber-400/50 bg-amber-950/30 text-amber-200 text-[11px] font-bold`.
            Row(Modifier.fillMaxWidth().padding(top = 12.dp), horizontalArrangement = Arrangement.End) {
                val shape = RoundedCornerShape(4.dp)
                Row(
                    Modifier.clip(shape).background(Color(69, 26, 3).copy(alpha = 0.3f)).border(1.dp, Tw.amber400.copy(alpha = 0.5f), shape)
                        .clickable(role = Role.Button, onClick = onSettings).padding(horizontal = 12.dp, vertical = 6.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    FaIcon(R.drawable.fa_solid_cog, null, size = 11.dp, tint = web.twText(Tw.amber200),
                        modifier = Modifier.padding(end = 4.dp))
                    Text("一時チャット設定", color = web.twText(Tw.amber200), fontSize = 11.sp, lineHeight = 16.sp, fontWeight = FontWeight.Bold)
                }
            }
        }
    }
}
// endregion

// region: Background tap handling
/**
 * Web on a phone: a tap outside the textarea blurs it (closing the keyboard and the `/`・`@` palettes) and a
 * tap that no control handles drops the bubble's tap-hover. Scrolling is not a tap.
 *
 * With ログの収集を強化, a press held for a long press (or cancelled) is recorded as `ui.press`: how far the
 * finger moved, whether another handler consumed the pointer, whether the list scrolled under it, whether a
 * text selection started and how often a handle drag changed the selection since the last entry. When the press
 * touched the selection, `ui.selection_windows` follows with the windows shown shortly after release (the
 * selection handles and the toolbar are windows of their own). It records no text.
 */
private fun Modifier.onBackgroundTap(onTap: () -> Unit, onUnhandledTap: () -> Unit, listPosition: () -> Pair<Int, Int>): Modifier = pointerInput(Unit) {
    awaitEachGesture {
        val down = awaitFirstDown(requireUnconsumed = false, pass = PointerEventPass.Initial)
        val startList = listPosition()
        val startSelection = SelectionSignals.snapshot()
        var moved = false
        var maxDistance = 0f
        var downConsumed: Boolean? = null
        var firstConsumedMs: Long? = null
        var up: PointerInputChange? = null
        SelectionSignals.pressStarted()
        try {
            while (true) {
                val change = awaitPointerEvent(PointerEventPass.Final).changes.firstOrNull { it.id == down.id } ?: break
                val distance = (change.position - down.position).getDistance()
                maxDistance = maxOf(maxDistance, distance)
                if (distance > viewConfiguration.touchSlop) moved = true
                if (downConsumed == null) downConsumed = change.isConsumed
                else if (change.isConsumed && firstConsumedMs == null) firstConsumedMs = change.uptimeMillis - down.uptimeMillis
                if (!change.pressed) { up = change; break }
            }
        } finally {
            SelectionSignals.pressEnded()
        }
        val held = (up?.uptimeMillis ?: currentEvent.changes.firstOrNull()?.uptimeMillis ?: down.uptimeMillis) - down.uptimeMillis
        if (ActivityLog.enabled && (up == null || held >= viewConfiguration.longPressTimeoutMillis)) {
            val endList = listPosition()
            val selection = SelectionSignals.snapshot()
            ActivityLog.log(
                "ui.press",
                "x" to down.position.x.toDp().value.toInt(), "y" to down.position.y.toDp().value.toInt(),
                "ms" to held, "cancelled" to (up == null),
                "moved_dp" to maxDistance.toDp().value.toInt(), "slop_dp" to viewConfiguration.touchSlop.toDp().value.toInt(),
                "long_press_ms" to viewConfiguration.longPressTimeoutMillis,
                "down_consumed" to downConsumed, "consumed_after_ms" to firstConsumedMs,
                "scrolled_px" to (if (endList.first == startList.first) endList.second - startList.second else null),
                "item_changed" to (endList.first != startList.first),
                "selection_changes" to selection.first - startSelection.first,
                "selection_menus" to selection.second - startSelection.second,
                "handle_changes" to SelectionSignals.takeHandleChanges(),
            )
            if (selection != startSelection) {
                val density = 1.dp.toPx()
                android.os.Handler(android.os.Looper.getMainLooper()).postDelayed({
                    ActivityLog.log("ui.selection_windows", "after_ms" to SELECTION_WINDOWS_DELAY_MS,
                        "windows" to com.minashin1120.aiplayground.data.LogPolicy.windows(visibleWindows(density)))
                }, SELECTION_WINDOWS_DELAY_MS)
            }
        }
        val released = up ?: return@awaitEachGesture
        // A selection long press also ends with an unmoved pointer-up. Treating that as a
        // background tap clears focus and dismisses the selection as soon as the finger lifts.
        if (moved || released.uptimeMillis - down.uptimeMillis >= viewConfiguration.longPressTimeoutMillis) return@awaitEachGesture
        onTap()
        if (!released.isConsumed) onUnhandledTap()
    }
}

/** Long enough after release for Compose to show the selection toolbar, which follows the handles. */
private const val SELECTION_WINDOWS_DELAY_MS = 400L
// endregion
