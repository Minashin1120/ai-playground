package com.minashin1120.aiplayground.ui

import android.os.Build
import android.view.ActionMode
import android.view.Menu
import android.view.MenuItem
import android.view.View
import android.view.inspector.WindowInspector
import androidx.annotation.RequiresApi
import androidx.compose.foundation.ComposeFoundationFlags
import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.hapticfeedback.HapticFeedback
import androidx.compose.ui.hapticfeedback.HapticFeedbackType
import androidx.compose.ui.platform.ClipEntry
import androidx.compose.ui.platform.Clipboard
import androidx.compose.ui.platform.ClipboardManager
import androidx.compose.ui.platform.LocalClipboard
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalHapticFeedback
import androidx.compose.ui.platform.LocalTextToolbar
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.platform.TextToolbar
import androidx.compose.ui.platform.TextToolbarStatus
import androidx.compose.ui.text.AnnotatedString

/*
 * Web `#quote-popover`: selecting text in the conversation offers "Quote", which puts the selection in
 * the composer quote bar. Compose does not expose the selected text, so the Quote item asks the
 * selection to copy itself and the clipboard wrappers below hand that one copy to [onQuote] instead of
 * the system clipboard.
 */

/**
 * Compose Foundation 1.9 turned on its new text context menu by default, and SelectionContainer then
 * builds its own toolbar instead of asking [LocalTextToolbar], so "Quote" never appeared. Keep the
 * [TextToolbar] path; call once before any content is composed.
 */
@OptIn(ExperimentalFoundationApi::class)
internal fun useQuoteSelectionToolbar() {
    ComposeFoundationFlags.isNewContextMenuEnabled = false
}

/**
 * Counts what the conversation's text selections did, for the `ui.press` entry of ログの収集を強化: Compose
 * plays [HapticFeedbackType.TextHandleMove] whenever a touch selection changes, and shows the toolbar after it.
 * A change while no finger is down on the conversation can only come from dragging a selection handle, whose
 * popup window takes that touch.
 */
internal object SelectionSignals {
    @Volatile private var changes = 0
    @Volatile private var menus = 0
    @Volatile private var pressing = false
    @Volatile private var handleChanges = 0

    fun snapshot(): Pair<Int, Int> = changes to menus
    fun changed() {
        changes++
        if (!pressing) handleChanges++
    }
    fun menuShown() { menus++ }
    fun pressStarted() { pressing = true }
    fun pressEnded() { pressing = false }
    /** Handle-drag changes since the last call. */
    fun takeHandleChanges(): Int = handleChanges.also { handleChanges = 0 }
}

/**
 * The app's shown windows in screen dp, as "class x,y wxh" (Compose popups such as the selection handles are
 * `PopupLayout`, the floating toolbar is `PopupDecorView`), so a log shows whether the handles appeared and where.
 */
internal fun visibleWindows(density: Float): List<String> =
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q && density > 0f) runCatching { shownWindows(density) }.getOrDefault(emptyList())
    else emptyList()

@RequiresApi(Build.VERSION_CODES.Q)
private fun shownWindows(density: Float): List<String> =
    WindowInspector.getGlobalWindowViews().filter { it.isShown }.map { view ->
        val at = IntArray(2)
        view.getLocationOnScreen(at)
        fun dp(px: Int) = (px / density).toInt()
        "${view.javaClass.simpleName} ${dp(at[0])},${dp(at[1])} ${dp(view.width)}x${dp(view.height)}"
    }

private class SelectionHaptics(private val base: HapticFeedback) : HapticFeedback {
    override fun performHapticFeedback(hapticFeedbackType: HapticFeedbackType) {
        if (hapticFeedbackType == HapticFeedbackType.TextHandleMove) SelectionSignals.changed()
        base.performHapticFeedback(hapticFeedbackType)
    }
}

private class QuoteCapture {
    var armed = false
}

private class QuoteClipboardManager(
    private val base: ClipboardManager,
    private val capture: QuoteCapture,
    private val onQuote: (String) -> Unit,
) : ClipboardManager by base {
    override fun setText(annotatedString: AnnotatedString) {
        if (capture.armed) {
            capture.armed = false
            onQuote(annotatedString.text)
        } else base.setText(annotatedString)
    }
}

private class QuoteClipboard(
    private val base: Clipboard,
    private val capture: QuoteCapture,
    private val onQuote: (String) -> Unit,
) : Clipboard by base {
    override suspend fun setClipEntry(clipEntry: ClipEntry?) {
        if (capture.armed) {
            capture.armed = false
            val text = clipEntry?.clipData?.takeIf { it.itemCount > 0 }?.getItemAt(0)?.text?.toString().orEmpty()
            onQuote(text)
        } else base.setClipEntry(clipEntry)
    }
}

/** Floating selection toolbar with the platform Copy / Select all plus "Quote". */
private class QuoteTextToolbar(private val view: View, private val capture: QuoteCapture) : TextToolbar {
    private var actionMode: ActionMode? = null
    private var contentRect = Rect.Zero
    private var onCopy: (() -> Unit)? = null
    private var onSelectAll: (() -> Unit)? = null

    override var status: TextToolbarStatus = TextToolbarStatus.Hidden
        private set

    private val callback = object : ActionMode.Callback2() {
        override fun onCreateActionMode(mode: ActionMode, menu: Menu): Boolean {
            populate(menu)
            return true
        }

        override fun onPrepareActionMode(mode: ActionMode, menu: Menu): Boolean {
            menu.clear()
            populate(menu)
            return true
        }

        override fun onActionItemClicked(mode: ActionMode, item: MenuItem): Boolean {
            when (item.itemId) {
                // The copy may reach the clipboard asynchronously; the wrappers disarm when they take it.
                ID_QUOTE -> { capture.armed = true; onCopy?.invoke() }
                ID_COPY -> onCopy?.invoke()
                ID_SELECT_ALL -> onSelectAll?.invoke()
                else -> return false
            }
            if (item.itemId != ID_SELECT_ALL) mode.finish()
            return true
        }

        override fun onDestroyActionMode(mode: ActionMode) {
            actionMode = null
            status = TextToolbarStatus.Hidden
        }

        override fun onGetContentRect(mode: ActionMode, view: View, outRect: android.graphics.Rect) {
            outRect.set(contentRect.left.toInt(), contentRect.top.toInt(), contentRect.right.toInt(), contentRect.bottom.toInt())
        }
    }

    private fun populate(menu: Menu) {
        if (onCopy != null) {
            menu.add(0, ID_QUOTE, 0, "Quote").setShowAsAction(MenuItem.SHOW_AS_ACTION_ALWAYS)
            menu.add(0, ID_COPY, 1, android.R.string.copy).setShowAsAction(MenuItem.SHOW_AS_ACTION_ALWAYS)
        }
        if (onSelectAll != null) menu.add(0, ID_SELECT_ALL, 2, android.R.string.selectAll).setShowAsAction(MenuItem.SHOW_AS_ACTION_IF_ROOM)
    }

    override fun showMenu(
        rect: Rect,
        onCopyRequested: (() -> Unit)?,
        onPasteRequested: (() -> Unit)?,
        onCutRequested: (() -> Unit)?,
        onSelectAllRequested: (() -> Unit)?,
    ) {
        contentRect = rect
        capture.armed = false
        SelectionSignals.menuShown()
        onCopy = onCopyRequested
        onSelectAll = onSelectAllRequested
        val mode = actionMode
        if (mode == null) {
            status = TextToolbarStatus.Shown
            actionMode = view.startActionMode(callback, ActionMode.TYPE_FLOATING)
        } else {
            mode.invalidate()
            mode.invalidateContentRect()
        }
    }

    override fun hide() {
        status = TextToolbarStatus.Hidden
        actionMode?.finish()
        actionMode = null
    }

    private companion object {
        const val ID_QUOTE = 1
        const val ID_COPY = 2
        const val ID_SELECT_ALL = 3
    }
}

/** Gives every text selection inside [content] the Web "Quote" action. */
@Suppress("DEPRECATION")
@Composable
internal fun ProvideQuoteSelection(onQuote: (String) -> Unit, content: @Composable () -> Unit) {
    val view = LocalView.current
    val latest = rememberUpdatedState(onQuote)
    val capture = remember { QuoteCapture() }
    val toolbar = remember(view) { QuoteTextToolbar(view, capture) }
    val baseManager = LocalClipboardManager.current
    val baseClipboard = LocalClipboard.current
    val baseHaptics = LocalHapticFeedback.current
    val haptics = remember(baseHaptics) { SelectionHaptics(baseHaptics) }
    val manager = remember(baseManager) { QuoteClipboardManager(baseManager, capture) { text -> latest.value(text.trim()) } }
    val clipboard = remember(baseClipboard) { QuoteClipboard(baseClipboard, capture) { text -> latest.value(text.trim()) } }
    CompositionLocalProvider(
        LocalTextToolbar provides toolbar,
        LocalClipboardManager provides manager,
        LocalClipboard provides clipboard,
        LocalHapticFeedback provides haptics,
        content = content,
    )
}
