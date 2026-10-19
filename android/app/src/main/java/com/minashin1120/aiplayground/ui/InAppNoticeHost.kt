package com.minashin1120.aiplayground.ui

import android.os.SystemClock
import android.view.ViewTreeObserver
import android.view.WindowManager
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.material3.AlertDialog
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.compositionLocalOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.LocalLayoutDirection
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.IntRect
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.LayoutDirection
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.DialogProperties
import androidx.compose.ui.window.Popup
import androidx.compose.ui.window.PopupPositionProvider
import androidx.compose.ui.window.PopupProperties
import kotlinx.coroutines.delay

internal val LocalInAppNotices = compositionLocalOf<InAppNoticeState?> { null }

private data class NoticeWindowLayout(val width: Dp, val padding: PaddingValues)
private val LocalNoticeWindowLayout = compositionLocalOf { NoticeWindowLayout(0.dp, PaddingValues()) }

/** Read insets in the Activity: an AlertDialog's own inset values describe its small central panel. */
@Composable
internal fun InAppNoticeProvider(content: @Composable () -> Unit) {
    val state = rememberSaveable(saver = InAppNoticeState.Saver) { InAppNoticeState() }
    val density = LocalDensity.current
    val direction = LocalLayoutDirection.current
    val insets = WindowInsets.safeDrawing
    val left = with(density) { insets.getLeft(this, direction).toDp() }
    val right = with(density) { insets.getRight(this, direction).toDp() }
    val top = with(density) { insets.getTop(this).toDp() }
    val layout = NoticeWindowLayout(
        LocalConfiguration.current.screenWidthDp.dp,
        PaddingValues(start = if (direction == LayoutDirection.Ltr) left else right,
            top = top, end = if (direction == LayoutDirection.Ltr) right else left),
    )
    // A window handover never restarts this timer. Elapsed realtime also ages messages while asleep.
    val deadline = state.nextDeadline
    LaunchedEffect(deadline) {
        if (deadline != null) {
            delay((deadline - SystemClock.elapsedRealtime()).coerceAtLeast(0L))
            state.expire()
        }
    }
    CompositionLocalProvider(LocalInAppNotices provides state, LocalNoticeWindowLayout provides layout) {
        NoticeWindowHost()
        content()
    }
}

private object NoticeTopPosition : PopupPositionProvider {
    override fun calculatePosition(
        anchorBounds: IntRect,
        windowSize: IntSize,
        layoutDirection: LayoutDirection,
        popupContentSize: IntSize,
    ) = IntOffset((windowSize.width - popupContentSize.width) / 2, 0)
}

/** Register once inside each window, outside the animated/blurred panel. Only the front window draws. */
@Composable
internal fun NoticeWindowHost() {
    val state = LocalInAppNotices.current ?: return
    val view = LocalView.current
    val window = remember(view) { Any() }
    val layout = LocalNoticeWindowLayout.current
    DisposableEffect(state, view, window) {
        state.registerWindow(window)
        val observer = view.viewTreeObserver
        val listener = ViewTreeObserver.OnWindowFocusChangeListener { focused ->
            if (focused) state.focusWindow(window)
        }
        observer.addOnWindowFocusChangeListener(listener)
        onDispose {
            if (observer.isAlive) observer.removeOnWindowFocusChangeListener(listener)
            state.unregisterWindow(window)
        }
    }
    if (state.activeWindow === window && state.notices.isNotEmpty()) {
        Popup(
            popupPositionProvider = NoticeTopPosition,
            properties = PopupProperties(
                // Use screen coordinates even when the parent is a small, centred AlertDialog.
                flags = WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN,
                dismissOnBackPress = false,
                dismissOnClickOutside = false,
                excludeFromSystemGesture = false,
                usePlatformDefaultWidth = false,
            ),
        ) {
            InAppNoticeStack(state, layout.width, layout.padding)
        }
    }
}

/** The button slot is composed inside Material's dialog window, so it can register that window. */
@Composable
internal fun NoticeAlertDialog(
    onDismissRequest: () -> Unit,
    confirmButton: @Composable () -> Unit,
    dismissButton: @Composable (() -> Unit)? = null,
    title: @Composable (() -> Unit)? = null,
    text: @Composable (() -> Unit)? = null,
    properties: DialogProperties = DialogProperties(),
) {
    AlertDialog(
        onDismissRequest = onDismissRequest,
        confirmButton = { NoticeWindowHost(); confirmButton() },
        dismissButton = dismissButton,
        title = title,
        text = text,
        properties = properties,
    )
}
