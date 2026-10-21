package com.minashin1120.aiplayground.ui

import android.os.Handler
import android.os.Looper
import android.view.View
import androidx.compose.foundation.ScrollState
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.runtime.snapshotFlow
import androidx.compose.ui.Modifier
import androidx.compose.ui.focus.onFocusChanged
import androidx.compose.ui.input.pointer.PointerEventPass
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.boundsInRoot
import androidx.compose.ui.layout.onGloballyPositioned
import androidx.compose.ui.layout.positionInRoot
import androidx.compose.ui.platform.LocalView
import com.minashin1120.aiplayground.data.LogPolicy
import com.minashin1120.aiplayground.data.ActivityLog
import java.util.UUID
import kotlin.math.abs
import kotlin.math.roundToInt
import kotlinx.coroutines.flow.collect

/** Random, composition-local identities only: never derive an identity from text or TeX. */
internal val LocalInteractionTrace = staticCompositionLocalOf<InteractionTrace?> { null }

internal class InteractionTrace(val element: String, private val parent: InteractionTrace? = null, private val view: View? = null) {
    val op: String = UUID.randomUUID().toString()
    var active = true
    var focused = false
    var attached = false
    var clipped = false
    var x = 0
    var y = 0
    var width = 0
    var height = 0
    var selectionChanges = 0
    var toolbarRequests = 0
    var toolbarCreates = 0
    var toolbarHides = 0
    var toolbarDestroys = 0
    var actionModeActive = false

    fun log(stage: String, phase: String = "done", vararg fields: Pair<String, Any?>) {
        if (!ActivityLog.enabled) return
        ActivityLog.log("ui.interaction", "op" to op, "parent_op" to parent?.op,
            "ui_element" to element, "interaction_stage" to stage, "phase" to phase,
            "focused" to focused, "attached" to attached,
            "window_focused" to view?.hasWindowFocus(), "clipped" to clipped,
            "dpr" to view?.resources?.displayMetrics?.density,
            "x" to x, "y" to y, "w" to width, "h" to height,
            "selection_changes" to selectionChanges, "toolbar_requests" to toolbarRequests,
            "toolbar_creates" to toolbarCreates, "toolbar_hides" to toolbarHides,
            "toolbar_destroys" to toolbarDestroys, "action_mode_active" to actionModeActive,
            *fields)
    }

    /** Selection UI may be scheduled after pointer-up; record the result as well as the request. */
    fun afterRelease(gesture: String, scroll: ScrollState?) {
        if (!ActivityLog.enabled) return
        Handler(Looper.getMainLooper()).postDelayed({
            if (active) log("settled", "done", "gesture" to gesture, "after_ms" to 400,
                "scroll_x" to scroll?.value, "scroll_max" to scroll?.maxValue,
                "windows" to view?.let { LogPolicy.windows(visibleWindows(it.resources.displayMetrics.density)) })
        }, 400)
    }
}

@Composable
internal fun rememberInteractionTrace(element: String): InteractionTrace {
    val parent = LocalInteractionTrace.current
    val view = LocalView.current
    val trace = remember(parent, element, view) { InteractionTrace(element, parent, view) }
    DisposableEffect(trace) {
        trace.active = true
        onDispose { trace.log("dispose"); trace.active = false }
    }
    return trace
}

/** Observes Initial/Final passes without consuming or changing any input event. */
internal fun Modifier.observeInteraction(trace: InteractionTrace, scroll: ScrollState? = null): Modifier =
    onGloballyPositioned { coordinates ->
        val bounds = coordinates.boundsInRoot()
        trace.attached = coordinates.isAttached
        val position = coordinates.positionInRoot()
        trace.x = position.x.roundToInt()
        trace.y = position.y.roundToInt()
        trace.clipped = bounds.width + 1f < coordinates.size.width || bounds.height + 1f < coordinates.size.height
        trace.width = coordinates.size.width
        trace.height = coordinates.size.height
    }.onFocusChanged {
        if (trace.focused != it.hasFocus) {
            trace.focused = it.hasFocus
            trace.log("focus")
        }
    }.pointerInput(trace, scroll) {
        awaitEachGesture {
            val down = awaitFirstDown(requireUnconsumed = false, pass = PointerEventPass.Initial)
            val gesture = if (ActivityLog.enabled) UUID.randomUUID().toString() else null
            val started = down.uptimeMillis
            val from = scroll?.value
            var movedX = 0f
            var movedY = 0f
            var consumed = false
            var released = false
            var duration = 0L
            if (gesture != null) trace.log("press", "start", "gesture" to gesture,
                "scroll_x" to from, "scroll_max" to scroll?.maxValue,
                "press_x" to down.position.x.roundToInt(), "press_y" to down.position.y.roundToInt())
            try {
                while (true) {
                    val event = awaitPointerEvent(PointerEventPass.Final)
                    val change = event.changes.firstOrNull { it.id == down.id } ?: break
                    movedX = maxOf(movedX, abs(change.position.x - down.position.x))
                    movedY = maxOf(movedY, abs(change.position.y - down.position.y))
                    consumed = change.isConsumed
                    duration = change.uptimeMillis - started
                    if (!change.pressed) { released = true; break }
                }
            } finally {
                if (gesture != null) {
                    trace.log("press", if (released) "done" else "cancelled", "gesture" to gesture,
                        "ms" to duration, "down_consumed" to down.isConsumed, "up_consumed" to consumed,
                        "moved_x" to movedX.roundToInt(), "moved_y" to movedY.roundToInt(),
                        "long_press_ms" to viewConfiguration.longPressTimeoutMillis,
                        "scroll_x" to scroll?.value, "scroll_max" to scroll?.maxValue,
                        "scrolled_px" to from?.let { scroll!!.value - it })
                    trace.afterRelease(gesture, scroll)
                }
            }
        }
    }

/** One render-state entry, plus one entry when scrolling settles (including the fling). */
@Composable
internal fun TraceMathRendering(trace: InteractionTrace, state: MathState, scroll: ScrollState?, naturalWidth: Int?, scale: Float) {
    val latestWidth by rememberUpdatedState(naturalWidth)
    val latestScale by rememberUpdatedState(scale)
    LaunchedEffect(trace, state, naturalWidth, scale) {
        trace.log("render", if (state is MathState.Failed) "error" else if (state is MathState.Loading) "start" else "done",
            "ready" to (state is MathState.Ready), "content_width" to naturalWidth, "scale" to scale,
            "scroll_x" to scroll?.value, "scroll_max" to scroll?.maxValue)
    }
    LaunchedEffect(trace, scroll) {
        if (scroll == null) return@LaunchedEffect
        var moving = false
        snapshotFlow { scroll.isScrollInProgress }.collect { running ->
            if (running || moving) trace.log("scroll", if (running) "start" else "done",
                "scroll_x" to scroll.value, "scroll_max" to scroll.maxValue,
                "content_width" to latestWidth, "scale" to latestScale)
            moving = running
        }
    }
}
