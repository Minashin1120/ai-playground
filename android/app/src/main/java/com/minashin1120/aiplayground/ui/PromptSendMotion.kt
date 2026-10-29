package com.minashin1120.aiplayground.ui

import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.drawWithContent
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.graphics.layer.GraphicsLayer
import androidx.compose.ui.graphics.layer.drawLayer
import androidx.compose.ui.graphics.rememberGraphicsLayer
import androidx.compose.ui.graphics.drawscope.clipRect
import androidx.compose.ui.graphics.drawscope.withTransform
import androidx.compose.ui.layout.boundsInWindow
import androidx.compose.ui.layout.positionInWindow
import androidx.compose.ui.unit.toSize
import androidx.compose.ui.layout.onGloballyPositioned
import androidx.compose.ui.semantics.clearAndSetSemantics
import androidx.compose.ui.zIndex
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.minashin1120.aiplayground.ChatState
import com.minashin1120.aiplayground.data.ActivityLog
import kotlinx.coroutines.CancellationException

/** A UI-only request. Neither its coordinates nor its recorded drawing leave this screen. */
internal class PromptSendMotion {
    var source: Rect? = null
    var quoteSource: Rect? = null
    var filesSource: Rect? = null
    var following = true
    var viewport by mutableStateOf<Rect?>(null)
    var target by mutableStateOf<Rect?>(null)
    var id by mutableStateOf<String?>(null)
    var ready by mutableStateOf(false)
    var layer: GraphicsLayer? = null
    var origin: Rect? = null
    var progress = Animatable(0f)
        private set

    fun finish() { id = null; layer = null; target = null; ready = false }
    fun begin(request: String?, reduce: Boolean) {
        finish()
        // Reset synchronously: a new request must never draw the previous flight's
        // completed progress while its animation coroutine is waiting to start.
        progress = Animatable(0f)
        val regions = listOfNotNull(source, quoteSource, filesSource).filterNot { it.isEmpty }
        origin = if (regions.isEmpty()) null else Rect(regions.minOf { it.left }, regions.minOf { it.top },
            regions.maxOf { it.right }, regions.maxOf { it.bottom })
        if (request != null && following && !reduce && origin?.isEmpty == false) id = request
    }
}

internal val LocalPromptSendMotion = staticCompositionLocalOf<PromptSendMotion?> { null }

// Use one opaque drawing at a time. Crossfading two copies reduces opacity even
// when their combined alpha is one, which makes text and tall bubbles flash.
internal fun promptSendShowsBubble(progress: Float): Boolean = progress >= 1f

internal fun promptSendClip(origin: Rect, viewport: Rect, progress: Float): Rect {
    val start = Rect(minOf(origin.left, viewport.left), minOf(origin.top, viewport.top),
        maxOf(origin.right, viewport.right), maxOf(origin.bottom, viewport.bottom))
    val p = progress.coerceIn(0f, 1f)
    return Rect(start.left + (viewport.left - start.left) * p,
        start.top + (viewport.top - start.top) * p,
        start.right + (viewport.right - start.right) * p,
        start.bottom + (viewport.bottom - start.bottom) * p)
}

@Composable
internal fun PromptSendMotionHost(state: ChatState, content: @Composable () -> Unit) {
    val motion = remember { PromptSendMotion() }
    val reduce = LocalReduceMotion.current
    var root by remember { mutableStateOf(Offset.Zero) }
    var rootSize by remember { mutableStateOf(IntSize.Zero) }
    LaunchedEffect(state.draft) { if (state.draft.isNotEmpty()) motion.finish() }
    // Initial composition/restoration must not replay an old send request.
    var seen by remember { mutableStateOf(state.promptSendMotionId) }
    // Prepare before composing the row. LaunchedEffect runs after composition,
    // leaving the newly inserted bubble visible at its destination for a frame.
    remember(state.chatNavigationId, reduce) { motion.finish(); Unit }
    if (seen != state.promptSendMotionId) {
        seen = state.promptSendMotionId
        motion.begin(state.promptSendMotionId, reduce)
    }
    LaunchedEffect(motion.id) {
        val request = motion.id ?: return@LaunchedEffect
        try {
            val progress = motion.progress
            // No network wait: the optimistic row is inserted before thread creation.
            repeat(12) { if (!motion.ready) withFrameNanos { } }
            val target = motion.target
            val viewport = motion.viewport
            if (!motion.ready || target == null || viewport == null || !target.overlaps(viewport)) return@LaunchedEffect
            ActivityLog.log("ui.show")
            progress.animateTo(1f, tween(350, easing = LinearEasing))
        } catch (e: CancellationException) { throw e }
        catch (_: Exception) { ActivityLog.log("console.warn") }
        finally { if (motion.id == request) motion.finish(); ActivityLog.log("ui.hide") }
    }
    val owner = LocalLifecycleOwner.current
    DisposableEffect(owner) {
        val observer = LifecycleEventObserver { _, event -> if (event == Lifecycle.Event.ON_PAUSE) motion.finish() }
        owner.lifecycle.addObserver(observer)
        onDispose { owner.lifecycle.removeObserver(observer); motion.finish() }
    }
    CompositionLocalProvider(LocalPromptSendMotion provides motion) {
        Box(Modifier.fillMaxSize().onSizeChanged {
            // IME animation changes height; keep the flight and track the target.
            if (rootSize != IntSize.Zero && rootSize.width != it.width) motion.finish()
            rootSize = it
        }.onGloballyPositioned {
            val next = it.boundsInWindow().topLeft
            root = next
        }) {
            content()
            val target = motion.target
            val origin = motion.origin
            val layer = motion.layer
            val viewport = motion.viewport
            if (motion.id != null && motion.ready && target != null && origin != null && layer != null && viewport != null) {
                Canvas(Modifier.fillMaxSize().zIndex(35f).clearAndSetSemantics { }) {
                    if (promptSendShowsBubble(motion.progress.value)) return@Canvas
                    val p = PlaygroundMotion.WebEaseOut.transform(motion.progress.value)
                    val startScale = minOf(origin.width / target.width, origin.height.coerceAtLeast(42f) / target.height, 1f)
                    val cardScale = startScale + (1f - startScale) * p
                    val from = Offset(origin.left + (origin.width - target.width * startScale) / 2f,
                        origin.top + (origin.height - target.height * startScale) / 2f)
                    val position = from + (target.topLeft - from) * p - root
                    // Match LazyColumn clipping at arrival, including a bubble taller
                    // than the viewport. The initial clip also includes the composer.
                    val clip = promptSendClip(origin, viewport, p).translate(-root)
                    clipRect(clip.left, clip.top, clip.right, clip.bottom) {
                        withTransform({ translate(position.x, position.y); scale(cardScale, cardScale, Offset.Zero) }) { drawLayer(layer) }
                    }
                }
            }
        }
    }
}

/** Record the actual bubble, including all text and attachments, rather than composing a duplicate. */
@Composable
internal fun promptSendBubbleModifier(messageId: String): Modifier {
    val motion = LocalPromptSendMotion.current ?: return Modifier
    if (motion.id != messageId) return Modifier
    val layer = rememberGraphicsLayer()
    DisposableEffect(messageId, layer) { onDispose { if (motion.layer === layer) motion.finish() } }
    return Modifier.onGloballyPositioned { coordinates ->
        val old = motion.target
        val next = Rect(coordinates.positionInWindow(), coordinates.size.toSize())
        if (old != null && old.size != next.size && motion.ready) motion.finish()
        else motion.target = next
    }.drawWithContent {
        if (motion.id != messageId) { drawContent(); return@drawWithContent }
        layer.record { this@drawWithContent.drawContent() }
        motion.layer = layer
        motion.ready = true
        if (promptSendShowsBubble(motion.progress.value)) drawContent()
    }
}
