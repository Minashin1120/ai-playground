package com.minashin1120.aiplayground.ui

import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.ime
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.drawWithContent
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.graphics.Paint
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.graphics.layer.GraphicsLayer
import androidx.compose.ui.graphics.layer.drawLayer
import androidx.compose.ui.graphics.rememberGraphicsLayer
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
    var viewport: Rect? = null
    var target by mutableStateOf<Rect?>(null)
    var id by mutableStateOf<String?>(null)
    var ready by mutableStateOf(false)
    var layer: GraphicsLayer? = null
    var origin: Rect? = null
    val progress = Animatable(0f)

    fun finish() { id = null; layer = null; target = null; ready = false }
    fun begin(request: String?, reduce: Boolean) {
        finish()
        val regions = listOfNotNull(source, quoteSource, filesSource).filterNot { it.isEmpty }
        origin = if (regions.isEmpty()) null else Rect(regions.minOf { it.left }, regions.minOf { it.top },
            regions.maxOf { it.right }, regions.maxOf { it.bottom })
        if (request != null && following && !reduce && origin?.isEmpty == false) id = request
    }
}

internal val LocalPromptSendMotion = staticCompositionLocalOf<PromptSendMotion?> { null }

internal fun promptSendReveal(progress: Float): Float = ((progress - 0.8f) / 0.2f).coerceIn(0f, 1f)

@Composable
internal fun PromptSendMotionHost(state: ChatState, content: @Composable () -> Unit) {
    val motion = remember { PromptSendMotion() }
    val reduce = LocalReduceMotion.current
    var root by remember { mutableStateOf(Offset.Zero) }
    var rootSize by remember { mutableStateOf(IntSize.Zero) }
    val keyboardHeight = WindowInsets.ime.getBottom(LocalDensity.current)
    LaunchedEffect(keyboardHeight) { motion.finish() }
    LaunchedEffect(state.draft) { if (state.draft.isNotEmpty()) motion.finish() }
    // Initial composition/restoration must not replay an old send request.
    var seen by remember { mutableStateOf(state.promptSendMotionId) }
    LaunchedEffect(state.promptSendMotionId, state.chatNavigationId, reduce) {
        val request = state.promptSendMotionId
        if (seen == request) { motion.finish(); return@LaunchedEffect }
        seen = request
        motion.begin(request, reduce)
    }
    LaunchedEffect(state.chatNavigationId) { motion.finish() }
    LaunchedEffect(motion.id) {
        val request = motion.id ?: return@LaunchedEffect
        try {
            motion.progress.snapTo(0f)
            // No network wait: the optimistic row is inserted before thread creation.
            repeat(12) { if (!motion.ready) withFrameNanos { } }
            val target = motion.target
            val viewport = motion.viewport
            if (!motion.ready || target == null || viewport == null || !target.overlaps(viewport)) return@LaunchedEffect
            ActivityLog.log("ui.show")
            motion.progress.animateTo(1f, tween(350, easing = LinearEasing))
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
            if (rootSize != IntSize.Zero && rootSize != it) motion.finish()
            rootSize = it
        }.onGloballyPositioned {
            val next = it.boundsInWindow().topLeft
            if (next != root) motion.finish()
            root = next
        }) {
            content()
            val target = motion.target
            val origin = motion.origin
            val layer = motion.layer
            if (motion.id != null && motion.ready && target != null && origin != null && layer != null) {
                Canvas(Modifier.fillMaxSize().zIndex(35f).clearAndSetSemantics { }
                    .graphicsLayer { alpha = 1f - promptSendReveal(motion.progress.value) }) {
                    val p = PlaygroundMotion.WebEaseOut.transform(motion.progress.value)
                    val startScale = minOf(origin.width / target.width, origin.height.coerceAtLeast(42f) / target.height, 1f)
                    val cardScale = startScale + (1f - startScale) * p
                    val from = Offset(origin.left + (origin.width - target.width * startScale) / 2f,
                        origin.top + (origin.height - target.height * startScale) / 2f)
                    val position = from + (target.topLeft - from) * p - root
                    withTransform({ translate(position.x, position.y); scale(cardScale, cardScale, Offset.Zero) }) { drawLayer(layer) }
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
        val alpha = promptSendReveal(motion.progress.value)
        if (alpha > 0f) {
            drawContext.canvas.saveLayer(Rect(Offset.Zero, size), Paint().apply { this.alpha = alpha })
            drawLayer(layer)
            drawContext.canvas.restore()
        }
    }
}
