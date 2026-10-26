package com.minashin1120.aiplayground.ui

import android.content.ClipboardManager
import android.content.Context
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.platform.LocalTextToolbar
import androidx.compose.ui.platform.TextToolbar
import androidx.compose.ui.platform.TextToolbarStatus
import androidx.compose.ui.semantics.SemanticsActions
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.longClick
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performSemanticsAction
import androidx.compose.ui.test.performTouchInput
import androidx.compose.ui.text.TextLayoutResult
import androidx.compose.ui.unit.dp
import com.minashin1120.aiplayground.data.ChatMessage
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment
import org.robolectric.annotation.Config
import org.robolectric.annotation.GraphicsMode

/** Hidden reasoning must not intercept selection in the answer below its scroll viewport. */
@RunWith(RobolectricTestRunner::class)
@GraphicsMode(GraphicsMode.Mode.NATIVE)
@Config(sdk = [35], qualifiers = "w411dp-h923dp-xxhdpi", shadows = [MathSelectionTest.NoMagnifier::class])
class MessageSelectionTest {
    @get:Rule val rule = createComposeRule()

    private val answer = "Answerword continues in the visible answer."
    private val thought = (1..80).joinToString("\n") { "Thoughtword reasoning line $it" }

    @Test fun latestAnswerWithExpandedThoughtCanBeSelectedAndCopied() =
        selectAndCopy(answer, expanded = true, target = answer, expected = "Answerword")

    @Test fun latestBlockquoteWithExpandedThoughtCanBeSelectedAndCopied() =
        selectAndCopy("> $answer", expanded = true, target = answer, expected = "Answerword")

    @Test fun olderAnswerWithCollapsedThoughtCanStillBeSelectedAndCopied() =
        selectAndCopy(answer, expanded = false, target = answer, expected = "Answerword")

    @Test fun expandedThoughtRetainsItsOwnSelectionAndCopy() =
        selectAndCopy(answer, expanded = true, target = thought, expected = "Thoughtword")

    private fun selectAndCopy(content: String, expanded: Boolean, target: String, expected: String) {
        var copy: (() -> Unit)? = null
        val toolbar = object : TextToolbar {
            override var status = TextToolbarStatus.Hidden
            override fun showMenu(rect: Rect, onCopyRequested: (() -> Unit)?, onPasteRequested: (() -> Unit)?,
                onCutRequested: (() -> Unit)?, onSelectAllRequested: (() -> Unit)?) {
                copy = onCopyRequested
                status = TextToolbarStatus.Shown
            }
            override fun hide() { status = TextToolbarStatus.Hidden }
        }
        rule.setContent {
            PlaygroundTheme(darkTheme = true) {
                CompositionLocalProvider(LocalTextToolbar provides toolbar) {
                    Box(Modifier.width(411.dp).height(760.dp)) {
                        LazyColumn {
                            item {
                                MessageBubble(ChatMessage("1", "assistant", content, thought = thought),
                                    onFile = {}, loader = null, actions = MessageActions(),
                                    controlsVisible = false, onToggleControls = {}, preserveLiveLayout = expanded)
                            }
                        }
                    }
                }
            }
        }
        rule.waitForIdle()
        // The bubble merges accessibility nodes. Gesture coordinates must come from the text
        // itself, otherwise a first-line press lands on the bubble padding / thought header.
        val node = rule.onNodeWithText(target, useUnmergedTree = true)
        val layouts = mutableListOf<TextLayoutResult>()
        node.performSemanticsAction(SemanticsActions.GetTextLayoutResult) { it(layouts) }
        // Use the first line, which is visible even when the thought is much taller than its viewport.
        val point = layouts.single().getBoundingBox(4).center
        node.performTouchInput { longClick(point) }
        rule.mainClock.advanceTimeBy(600)
        rule.waitForIdle()
        assertNotNull("Selection must offer Copy for the visible text", copy)
        rule.runOnIdle { copy!!.invoke() }
        rule.waitForIdle()
        val clipboard = RuntimeEnvironment.getApplication().getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
        assertEquals(expected, clipboard.primaryClip?.getItemAt(0)?.text?.toString())
    }
}
