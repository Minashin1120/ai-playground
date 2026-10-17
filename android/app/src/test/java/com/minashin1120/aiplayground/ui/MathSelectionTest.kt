package com.minashin1120.aiplayground.ui

import android.util.LruCache
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.hapticfeedback.HapticFeedback
import androidx.compose.ui.hapticfeedback.HapticFeedbackType
import androidx.compose.ui.platform.LocalHapticFeedback
import androidx.compose.ui.platform.LocalTextToolbar
import androidx.compose.ui.platform.TextToolbar
import androidx.compose.ui.platform.TextToolbarStatus
import androidx.compose.ui.semantics.SemanticsProperties
import androidx.compose.ui.semantics.getOrNull
import androidx.compose.ui.test.SemanticsMatcher
import androidx.compose.ui.test.click
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.longClick
import androidx.compose.ui.test.isRoot
import androidx.compose.ui.test.onFirst
import androidx.compose.ui.test.performTouchInput
import androidx.compose.ui.unit.dp
import com.minashin1120.aiplayground.data.ChatMessage
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import org.robolectric.annotation.GraphicsMode

/**
 * Diagnostic: long-presses a grid over an answer with formulas and prints where a selection starts
 * ("S") and where it does not ("."), next to the same answer without formulas.
 */
@RunWith(RobolectricTestRunner::class)
@GraphicsMode(GraphicsMode.Mode.NATIVE)
@Config(sdk = [27], qualifiers = "w360dp-h2400dp-xxhdpi")
class MathSelectionTest {
    @get:Rule val rule = createComposeRule()

    private class Signals {
        var menus = 0
        var haptics = 0
        var hides = 0
        val total get() = menus + haptics
    }

    private fun svg(width: Int, ascent: Int, depth: Int) =
        """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 -$ascent $width ${ascent + depth}"><g stroke="currentColor" fill="currentColor" stroke-width="0"><rect width="$width" height="${ascent + depth}"></rect></g></svg>"""

    @Suppress("UNCHECKED_CAST")
    private fun typeset() {
        val field = MathJaxRenderer::class.java.getDeclaredField("cache").apply { isAccessible = true }
        val cache = field.get(null) as LruCache<String, MathSvg>
        cache.put("I:ax^2+bx+c=0", parseMathSvg(svg(6200, 750, 250))!!)
        cache.put("I:x=\\frac{-b\\pm\\sqrt{b^2-4ac}}{2a}", parseMathSvg(svg(9000, 1500, 900))!!)
        cache.put("I:x", parseMathSvg(svg(600, 450, 10))!!)
        cache.put("D:\\int_0^1 x^2\\,dx = \\frac{1}{3}", parseMathSvg(svg(7000, 1600, 900))!!)
    }

    /** The window holding the conversation; message controls add popup roots. */
    private fun main() = rule.onAllNodes(isRoot()).onFirst()

    private fun scan(name: String, content: String) {
        val signals = Signals()
        val toolbar = object : TextToolbar {
            override var status = TextToolbarStatus.Hidden
            override fun showMenu(rect: Rect, onCopyRequested: (() -> Unit)?, onPasteRequested: (() -> Unit)?,
                onCutRequested: (() -> Unit)?, onSelectAllRequested: (() -> Unit)?) {
                signals.menus++
                status = TextToolbarStatus.Shown
            }
            override fun hide() { signals.hides++; status = TextToolbarStatus.Hidden }
        }
        val haptic = object : HapticFeedback {
            override fun performHapticFeedback(hapticFeedbackType: HapticFeedbackType) { signals.haptics++ }
        }
        rule.setContent {
            PlaygroundTheme(darkTheme = true) {
                CompositionLocalProvider(LocalTextToolbar provides toolbar, LocalHapticFeedback provides haptic) {
                    Box(Modifier.width(360.dp)) {
                        LazyColumn {
                            item { Answer(content) }
                        }
                    }
                }
            }
        }
        rule.waitForIdle()
        rule.mainClock.advanceTimeBy(2_000)
        val out = StringBuilder("==== $name\n")
        rule.onAllNodes(SemanticsMatcher.keyIsDefined(SemanticsProperties.Text)).fetchSemanticsNodes().forEach { node ->
            val text = node.config.getOrNull(SemanticsProperties.Text)?.joinToString("") { it.text }.orEmpty()
            out.append("text ${node.boundsInRoot} ${text.take(24).replace('\n', ' ')}\n")
        }
        val root = main().fetchSemanticsNode().size
        for (y in 30 until minOf(root.height, 1500) step 36) {
            out.append(String.format("%4d ", y))
            for (x in 60 until root.width - 30 step 60) {
                val before = signals.total
                main().performTouchInput { longClick(Offset(x.toFloat(), y.toFloat())) }
                rule.waitForIdle()
                val selected = signals.total > before
                out.append(if (selected) 'S' else '.')
                if (selected) {
                    main().performTouchInput { click(Offset(x.toFloat(), y.toFloat())) }
                    rule.mainClock.advanceTimeBy(400)
                    rule.waitForIdle()
                }
            }
            out.append('\n')
        }
        println(out)
    }

    @Composable
    private fun Answer(content: String) {
        MessageBubble(ChatMessage("1", "assistant", content, encrypted = false), onFile = {}, loader = null,
            actions = MessageActions(), controlsVisible = false, onToggleControls = {})
    }

    @Test fun withFormulas() {
        typeset()
        scan("with formulas", WITH_MATH)
    }

    @Test fun withoutFormulas() = scan("without formulas", WITHOUT_MATH)

    private companion object {
        val WITH_MATH = """最初の段落には数式がありません。普通の文字だけです。

二次方程式 \(ax^2+bx+c=0\) の解は \(x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}\) です。ここは普通の文字が続く文章です。

$$
\int_0^1 x^2\,dx = \frac{1}{3}
$$

変数 \(x\) を使った短い段落です。

最後の段落は数式を含みません。長押しで選択できるはずです。"""

        val WITHOUT_MATH = """最初の段落には数式がありません。普通の文字だけです。

二次方程式 AAAAAAAA の解は BBBBBBBBBBBB です。ここは普通の文字が続く文章です。

CCCCCCCCCC

変数 D を使った短い段落です。

最後の段落は数式を含みません。長押しで選択できるはずです。"""
    }
}
