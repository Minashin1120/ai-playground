package com.minashin1120.aiplayground.ui

import android.util.LruCache
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.text.InlineTextContent
import androidx.compose.foundation.text.appendInlineContent
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.hapticfeedback.HapticFeedback
import androidx.compose.ui.hapticfeedback.HapticFeedbackType
import androidx.compose.ui.layout.layout
import androidx.compose.ui.platform.LocalHapticFeedback
import androidx.compose.ui.platform.LocalTextToolbar
import androidx.compose.ui.platform.TextToolbar
import androidx.compose.ui.platform.TextToolbarStatus
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.longClick
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.performTouchInput
import androidx.compose.ui.text.Placeholder
import androidx.compose.ui.text.PlaceholderVerticalAlign
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.em
import com.minashin1120.aiplayground.data.ChatMessage
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import org.robolectric.annotation.GraphicsMode

/** A long-press on the words of a message starts a selection, also when the paragraph holds a formula. */
@RunWith(RobolectricTestRunner::class)
@GraphicsMode(GraphicsMode.Mode.NATIVE)
@Config(sdk = [34], qualifiers = "w360dp-h2400dp-xxhdpi")
class MathSelectionTest {
    @get:Rule val rule = createComposeRule()

    private val markup = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 -905.6 891 916.6"><g stroke="currentColor" fill="currentColor" stroke-width="0"><rect width="891" height="916"></rect></g></svg>"""

    private class Signals {
        var menus = 0
        var haptics = 0
    }

    @Suppress("UNCHECKED_CAST")
    private fun typeset(tex: String) {
        val field = MathJaxRenderer::class.java.getDeclaredField("cache").apply { isAccessible = true }
        (field.get(null) as LruCache<String, MathSvg>).put("I:$tex", parseMathSvg(markup)!!)
    }

    private fun longPressStartsSelection(content: @Composable () -> Unit): Signals {
        val signals = Signals()
        val toolbar = object : TextToolbar {
            override var status = TextToolbarStatus.Hidden
            override fun showMenu(rect: Rect, onCopyRequested: (() -> Unit)?, onPasteRequested: (() -> Unit)?,
                onCutRequested: (() -> Unit)?, onSelectAllRequested: (() -> Unit)?) {
                signals.menus++
                status = TextToolbarStatus.Shown
            }
            override fun hide() { status = TextToolbarStatus.Hidden }
        }
        val haptic = object : HapticFeedback {
            override fun performHapticFeedback(hapticFeedbackType: HapticFeedbackType) { signals.haptics++ }
        }
        rule.setContent {
            PlaygroundTheme(darkTheme = true) {
                CompositionLocalProvider(LocalTextToolbar provides toolbar, LocalHapticFeedback provides haptic) {
                    androidx.compose.foundation.layout.Box(Modifier.width(340.dp).testTag("root")) { content() }
                }
            }
        }
        rule.waitForIdle()
        rule.mainClock.advanceTimeBy(2_000)
        // The first word of the bubble's first line: plain text, left of any formula.
        rule.onNodeWithTag("root").performTouchInput { longClick(Offset(100f, 80f)) }
        rule.waitForIdle()
        return signals
    }

    private fun bubble(content: String) = @Composable {
        MessageBubble(ChatMessage("1", "assistant", content, encrypted = false), onFile = {}, loader = null,
            actions = MessageActions(), controlsVisible = false, onToggleControls = {})
    }

    private fun assertSelected(name: String, signals: Signals) =
        assertTrue("$name: menus=${signals.menus} haptics=${signals.haptics}", signals.menus > 0 || signals.haptics > 0)

    @Test fun plainParagraph() = assertSelected("plain", longPressStartsSelection(bubble(WORDS + " after the words")))

    @Test fun paragraphWithInlineCode() = assertSelected("code", longPressStartsSelection(bubble(WORDS + " `code` after")))

    @Test fun paragraphWithLoadingFormula() = assertSelected("loading", longPressStartsSelection(bubble(WORDS + " \\(y^3\\) after")))

    @Test fun paragraphWithTypesetFormula() {
        typeset("x^2")
        assertSelected("typeset", longPressStartsSelection(bubble(WORDS + " \\(x^2\\) after")))
    }

    @Test fun bareTextWithPlaceholder() = assertSelected("bare placeholder", longPressStartsSelection { BarePlaceholder(Modifier) })

    @Test fun bareTextWithPlaceholderAndLayout() = assertSelected("bare placeholder + layout", longPressStartsSelection {
        BarePlaceholder(Modifier.layout { measurable, constraints ->
            val placeable = measurable.measure(constraints)
            layout(placeable.width, placeable.height) { placeable.place(0, 0) }
        })
    })

    @Composable
    private fun BarePlaceholder(modifier: Modifier) {
        val text = buildAnnotatedString { append("$WORDS "); appendInlineContent("p", "x2"); append(" after") }
        val inline = mapOf("p" to InlineTextContent(Placeholder(1.em, 1.em, PlaceholderVerticalAlign.TextCenter)) {
            Canvas(Modifier.fillMaxSize()) { drawRect(Color.Red) }
        })
        SelectionContainer {
            androidx.compose.foundation.layout.Column(Modifier.testTag("bare")) {
                androidx.compose.foundation.layout.Spacer(Modifier.width(1.dp).then(Modifier.layout { m, c ->
                    val p = m.measure(c.copy(minHeight = 0)); layout(p.width, 40) { p.place(0, 0) }
                }))
                Text(text, inlineContent = inline, color = Color.White, modifier = modifier)
            }
        }
    }

    private companion object {
        const val WORDS = "Selectable words before the formula"
    }
}
