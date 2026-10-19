package com.minashin1120.aiplayground.ui

import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.snap
import androidx.compose.animation.core.tween
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.gestures.detectVerticalDragGestures
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.input.pointer.util.VelocityTracker
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.semantics.LiveRegionMode
import androidx.compose.ui.semantics.dismiss
import androidx.compose.ui.semantics.liveRegion
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.zIndex
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.launch
import kotlin.math.abs

/** Ignore small drags; either a 40dp drag or an intentional vertical flick dismisses one message. */
internal fun noticeSwipeDirection(distance: Float, velocity: Float, density: Float): Int? {
    val motion = when {
        abs(velocity) >= 800f * density -> velocity
        abs(distance) >= 40f * density -> distance
        else -> return null
    }
    return if (motion > 0f) 1 else -1
}

@Composable
internal fun InAppNoticeStack(state: InAppNoticeState, width: Dp, windowPadding: PaddingValues) {
    val reduce = LocalReduceMotion.current
    val list = rememberLazyListState()
    val firstId = state.notices.firstOrNull()?.id
    LaunchedEffect(firstId) {
        // LazyColumn normally preserves the old top item's key when a newer item is prepended.
        list.scrollToItem(0)
    }
    LazyColumn(
        modifier = Modifier.width(width).padding(windowPadding).testTag("in_app_notices"),
        state = list,
        userScrollEnabled = false,
    ) {
        items(state.notices, key = { it.id }) { notice ->
            InAppNoticeBanner(
                notice,
                state,
                Modifier.animateItem(
                    fadeInSpec = null,
                    fadeOutSpec = null,
                    placementSpec = if (reduce) snap() else tween(PlaygroundMotion.MEDIUM, easing = PlaygroundMotion.Standard),
                ),
            )
        }
    }
}

@Composable
private fun InAppNoticeBanner(notice: InAppNotice, state: InAppNoticeState, modifier: Modifier) {
    val reduce = LocalReduceMotion.current
    val density = LocalDensity.current.density
    val offset = remember(notice.id) { Animatable(if (notice.entered) 0f else -24f * density) }
    val alpha = remember(notice.id) { Animatable(if (notice.entered) 1f else 0f) }
    var height by remember(notice.id) { mutableIntStateOf(1) }
    var dragging by remember(notice.id) { mutableStateOf(false) }
    var dragOffset by remember(notice.id) { mutableFloatStateOf(0f) }
    val velocity = remember(notice.id) { VelocityTracker() }
    val closing by rememberUpdatedState(notice.exitDirection != null)
    val returnOffset by animateFloatAsState(
        dragOffset,
        animationSpec = if (dragging || reduce) snap() else tween(PlaygroundMotion.SHORT, easing = PlaygroundMotion.Standard),
        label = "notice drag return",
    )
    DisposableEffect(notice.id, state) {
        onDispose { state.setDragging(notice.id, false) }
    }
    LaunchedEffect(notice.id, notice.exitDirection, reduce) {
        val direction = notice.exitDirection
        val duration = if (reduce) 0 else if (direction == null) PlaygroundMotion.MEDIUM else PlaygroundMotion.SHORT
        coroutineScope {
            launch { alpha.animateTo(if (direction == null) 1f else 0f, tween(duration)) }
            launch { offset.animateTo(if (direction == null) 0f else direction * height.toFloat(), tween(duration)) }
        }
        if (direction == null) state.finishEnter(notice.id) else state.finishExit(notice.id)
    }
    val endDrag: () -> Unit = {
        dragging = false
        state.setDragging(notice.id, false)
        dragOffset = 0f
        state.expire()
    }
    Surface(
        modifier = modifier
            .fillMaxWidth()
            .zIndex(if (dragging || notice.exitDirection != null) 1f else 0f)
            .onSizeChanged { height = it.height }
            .testTag("in_app_notice_${notice.id}")
            .graphicsLayer {
                translationY = offset.value + if (dragging) dragOffset else returnOffset
                this.alpha = alpha.value * (1f - (abs(dragOffset) / height).coerceIn(0f, 0.6f))
            }
            .semantics(mergeDescendants = true) {
                liveRegion = LiveRegionMode.Polite
                dismiss { state.dismiss(notice.id); true }
            }
            .pointerInput(notice.id) {
                detectVerticalDragGestures(
                    onDragStart = {
                        if (!closing) {
                            dragOffset = returnOffset
                            dragging = true
                            state.setDragging(notice.id, true)
                            velocity.resetTracking()
                        }
                    },
                    onVerticalDrag = { change, amount ->
                        if (dragging && !closing) {
                            change.consume()
                            dragOffset += amount
                            velocity.addPosition(change.uptimeMillis, change.position)
                        }
                    },
                    onDragEnd = {
                        if (dragging) {
                            val direction = noticeSwipeDirection(dragOffset, velocity.calculateVelocity().y, density)
                            if (direction == null) endDrag()
                            else {
                                dragging = false
                                state.dismiss(notice.id, direction)
                            }
                        }
                    },
                    onDragCancel = endDrag,
                )
            },
        color = MaterialTheme.colorScheme.surfaceContainerHigh,
        contentColor = MaterialTheme.colorScheme.onSurface,
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant),
        shadowElevation = 4.dp,
    ) {
        Text(notice.message, Modifier.padding(horizontal = 16.dp, vertical = 14.dp), style = MaterialTheme.typography.bodyMedium)
    }
}
