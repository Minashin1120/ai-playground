package com.minashin1120.aiplayground.ui

import android.content.ClipboardManager
import android.content.Context
import android.util.LruCache
import android.widget.Magnifier
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.getValue
import androidx.compose.runtime.key
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.hapticfeedback.HapticFeedback
import androidx.compose.ui.hapticfeedback.HapticFeedbackType
import androidx.compose.ui.platform.LocalHapticFeedback
import androidx.compose.ui.test.SemanticsNodeInteraction
import androidx.compose.ui.test.TouchInjectionScope
import androidx.compose.ui.test.click
import androidx.compose.ui.test.onNodeWithText
import org.junit.Assert.assertTrue
import androidx.compose.ui.platform.LocalTextToolbar
import androidx.compose.ui.platform.TextToolbar
import androidx.compose.ui.platform.TextToolbarStatus
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.isRoot
import androidx.compose.ui.test.onFirst
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.TextMeasurer
import androidx.compose.ui.text.rememberTextMeasurer
import com.github.takahirom.roborazzi.captureRoboImage
import androidx.compose.ui.test.longClick
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.performTouchInput
import androidx.compose.ui.unit.dp
import com.minashin1120.aiplayground.data.ChatMessage
import org.junit.Assert.assertEquals
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment
import org.robolectric.annotation.Config
import org.robolectric.annotation.GraphicsMode
import org.robolectric.annotation.Implementation
import org.robolectric.annotation.Implements

/**
 * A long press on a typeset formula selects the whole formula. Its approximation used to sit under the
 * placeholder as several characters, and a press selected a word inside it with zero width: the menu showed,
 * but there was no highlight and both handles stood on the formula's end.
 */
@RunWith(RobolectricTestRunner::class)
@GraphicsMode(GraphicsMode.Mode.NATIVE)
@Config(sdk = [35], qualifiers = "w411dp-h923dp-xxhdpi", shadows = [MathSelectionTest.NoMagnifier::class])
class MathSelectionTest {
    /** Robolectric has no surface for the selection magnifier; the gesture only needs it to do nothing. */
    @Implements(Magnifier::class)
    class NoMagnifier {
        @Implementation fun show(sourceCenterX: Float, sourceCenterY: Float) = Unit
        @Implementation fun show(sourceCenterX: Float, sourceCenterY: Float, magnifierCenterX: Float, magnifierCenterY: Float) = Unit
        @Implementation fun update() = Unit
        @Implementation fun dismiss() = Unit
    }

    @get:Rule val rule = createComposeRule()

    private fun <T> field(name: String): T {
        @Suppress("UNCHECKED_CAST")
        return MathJaxRenderer::class.java.getDeclaredField(name).apply { isAccessible = true }.get(null) as T
    }

    /** Typesets every formula the composition asked MathJax for. */
    private fun typesetPending() {
        val waiting = field<HashMap<String, *>>("waiting")
        val cache = field<LruCache<String, MathSvg>>("cache")
        waiting.keys.toList().forEach { id ->
            cache.put(id, parseMathSvg("""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 -800 6000 1050"><rect y="-800" width="6000" height="1050"></rect></svg>""")!!)
        }
        waiting.clear()
    }

    /** Diagnostic: the word Compose's long press selects around a candidate formula character between kana. */
    @Test fun wordBoundaryAroundCandidateCharacters() {
        var measurer: TextMeasurer? = null
        rule.setContent { measurer = rememberTextMeasurer() }
        rule.waitForIdle()
        val out = StringBuilder("==== word boundaries\n")
        for ((name, code) in listOf("tangut" to 0x17000, "yi" to 0xA000, "hangul" to 0xAC00, "pua" to 0xF0000, "bmp-pua" to 0xE000)) {
            val c = String(Character.toChars(code))
            val text = "あ${c}い ab${c}cd ${c}${c}"
            val layout = measurer!!.measure(AnnotatedString(text))
            val cells = (0..text.length).joinToString(" ") { offset ->
                val range = layout.getWordBoundary(offset)
                "$offset:${range.start}-${range.end}"
            }
            out.append("$name len=${c.length} letter=${Character.isLetterOrDigit(code)} | $cells\n")
        }
        println(out)
    }

    @Test fun longPressOnADisplayFormulaSelectsAndCopiesTheWholeFormula() {
        var menus = 0
        var haptics = 0
        var copy: (() -> Unit)? = null
        val toolbar = object : TextToolbar {
            override var status = TextToolbarStatus.Hidden
            override fun showMenu(rect: Rect, onCopyRequested: (() -> Unit)?, onPasteRequested: (() -> Unit)?,
                onCutRequested: (() -> Unit)?, onSelectAllRequested: (() -> Unit)?) {
                menus++
                copy = onCopyRequested
                status = TextToolbarStatus.Shown
            }
            override fun hide() { status = TextToolbarStatus.Hidden }
        }
        val haptic = object : HapticFeedback {
            override fun performHapticFeedback(hapticFeedbackType: HapticFeedbackType) { haptics++ }
        }
        var generation by mutableIntStateOf(0)
        rule.setContent {
            PlaygroundTheme(darkTheme = true) {
                CompositionLocalProvider(LocalTextToolbar provides toolbar, LocalHapticFeedback provides haptic) {
                    Box(Modifier.width(411.dp)) {
                        LazyColumn {
                            item {
                                key(generation) {
                                    MessageBubble(ChatMessage("1", "assistant", "前の項に足す数\n\n$$6(3)^2 = 54$$\n\nとなります。", encrypted = false),
                                        onFile = {}, loader = null, actions = MessageActions(), controlsVisible = false, onToggleControls = {})
                                }
                            }
                        }
                    }
                }
            }
        }
        rule.waitForIdle()
        typesetPending()
        generation++
        rule.mainClock.advanceTimeBy(1_000)
        rule.waitForIdle()
        val clipboard = RuntimeEnvironment.getApplication().getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager

        /** Long-presses [at] in [node], copies the selection and returns what was copied with the counts seen. */
        fun pressAndCopy(name: String, node: SemanticsNodeInteraction, at: TouchInjectionScope.() -> Offset): String {
            menus = 0; haptics = 0; copy = null
            clipboard.setPrimaryClip(android.content.ClipData.newPlainText("", ""))
            node.performTouchInput { longClick(at()) }
            rule.mainClock.advanceTimeBy(600)
            rule.waitForIdle()
            rule.onAllNodes(isRoot()).onFirst().captureRoboImage("build/outputs/roborazzi/math_selection_$name.png")
            copy?.invoke()
            rule.waitForIdle()
            val copied = clipboard.primaryClip?.getItemAt(0)?.text?.toString()
            println("==== $name: haptics=$haptics menus=$menus copy=${copy != null} copied=[$copied]")
            node.performTouchInput { click(at()) }
            rule.mainClock.advanceTimeBy(600)
            rule.waitForIdle()
            return copied.orEmpty()
        }

        val approximation = latexToDisplay("6(3)^2 = 54")
        // A control: plain text in the same bubble.
        val plain = pressAndCopy("text", rule.onNodeWithText("前の項に足す数", useUnmergedTree = true)) { centerLeft.copy(x = centerLeft.x + 20f) }
        // The right end of the formula, where a word of the approximation ("54") used to be selected on its own.
        val formula = pressAndCopy("formula", rule.onNodeWithContentDescription(approximation, useUnmergedTree = true)) { centerRight.copy(x = centerRight.x - 4f) }
        assertTrue("the control selects text: [$plain]", plain.isNotEmpty())
        assertEquals(approximation, formula)
    }
}
