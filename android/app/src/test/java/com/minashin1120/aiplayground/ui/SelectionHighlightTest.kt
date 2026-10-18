package com.minashin1120.aiplayground.ui

import android.util.LruCache
import android.widget.Magnifier
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.material3.DrawerValue
import androidx.compose.material3.ModalDrawerSheet
import androidx.compose.material3.ModalNavigationDrawer
import androidx.compose.material3.rememberDrawerState
import androidx.compose.ui.draw.blur
import androidx.compose.runtime.getValue
import androidx.compose.runtime.key
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.test.click
import androidx.compose.ui.test.isRoot
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.longClick
import androidx.compose.ui.test.onFirst
import androidx.compose.ui.test.performTouchInput
import androidx.compose.ui.unit.dp
import com.github.takahirom.roborazzi.captureRoboImage
import com.minashin1120.aiplayground.data.ChatMessage
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import org.robolectric.annotation.GraphicsMode
import org.robolectric.annotation.Implementation
import org.robolectric.annotation.Implements

/**
 * Diagnostic: long-presses words in an answer laid out as in the conversation (theme, bubble, Quote toolbar,
 * list) and captures the screen right after, to see whether the selection highlight is drawn. A phone showed
 * the handles and the Quote / Copy menu but no highlight.
 */
@RunWith(RobolectricTestRunner::class)
@GraphicsMode(GraphicsMode.Mode.NATIVE)
@Config(sdk = [35], qualifiers = "w411dp-h923dp-xxhdpi", shadows = [SelectionHighlightTest.NoMagnifier::class])
class SelectionHighlightTest {
    @Implements(Magnifier::class)
    class NoMagnifier {
        @Implementation fun show(sourceCenterX: Float, sourceCenterY: Float) = Unit
        @Implementation fun show(sourceCenterX: Float, sourceCenterY: Float, magnifierCenterX: Float, magnifierCenterY: Float) = Unit
        @Implementation fun update() = Unit
        @Implementation fun dismiss() = Unit
    }

    @get:Rule val rule = createComposeRule()

    private fun svg(width: Int, ascent: Int, depth: Int) =
        """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 -$ascent $width ${ascent + depth}"><g stroke="currentColor" fill="currentColor" stroke-width="0"><rect y="-$ascent" width="$width" height="${ascent + depth}"></rect></g></svg>"""

    private fun <T> field(name: String): T {
        @Suppress("UNCHECKED_CAST")
        return MathJaxRenderer::class.java.getDeclaredField(name).apply { isAccessible = true }.get(null) as T
    }

    private fun typesetPending() {
        val waiting = field<HashMap<String, *>>("waiting")
        val cache = field<LruCache<String, MathSvg>>("cache")
        waiting.keys.toList().forEach { id -> cache.put(id, parseMathSvg(svg(400 + 260 * id.length, 800, 250))!!) }
        waiting.clear()
    }

    private fun main() = rule.onAllNodes(isRoot()).onFirst()

    private fun capture(name: String, content: String, dark: Boolean, screen: Boolean = false) {
        var generation by mutableIntStateOf(0)
        rule.setContent {
            PlaygroundTheme(darkTheme = dark) {
                val list = @androidx.compose.runtime.Composable {
                    Box(Modifier.fillMaxSize()) {
                        LazyColumn(Modifier.fillMaxSize(), contentPadding = PaddingValues(12.dp)) {
                            item {
                                key(generation) {
                                    ProvideQuoteSelection({}) {
                                        MessageBubble(ChatMessage("1", "assistant", content, model = "test-model", encrypted = false),
                                            onFile = {}, loader = null, actions = MessageActions(), controlsVisible = false, onToggleControls = {})
                                    }
                                }
                            }
                        }
                    }
                }
                // The phone layout: the history drawer around the conversation, which is blurred (0dp when closed).
                if (screen) ModalNavigationDrawer(drawerState = rememberDrawerState(DrawerValue.Closed), gesturesEnabled = true,
                    drawerContent = { ModalDrawerSheet { } }) { Box(Modifier.blur(0.dp)) { list() } }
                else list()
            }
        }
        rule.waitForIdle()
        typesetPending()
        generation++
        rule.waitForIdle()
        rule.mainClock.advanceTimeBy(2_000)
        rule.waitForIdle()
        val size = main().fetchSemanticsNode().size
        main().captureRoboImage("build/outputs/roborazzi/selection_highlight_${name}_before.png")
        // Points over the first lines of the answer (12dp list padding + 16dp bubble padding).
        val density = size.width / 411f
        listOf(40 to 40, 120 to 40, 60 to 70, 200 to 100).forEachIndexed { index, (x, y) ->
            val at = Offset(x * density, y * density)
            main().performTouchInput { longClick(at) }
            rule.mainClock.advanceTimeBy(600)
            rule.waitForIdle()
            main().captureRoboImage("build/outputs/roborazzi/selection_highlight_${name}_${index}_${x}x$y.png")
            if (index == 0) {
                rule.mainClock.advanceTimeBy(3_000)
                rule.waitForIdle()
                main().captureRoboImage("build/outputs/roborazzi/selection_highlight_${name}_${index}_${x}x${y}_later.png")
            }
            main().performTouchInput { click(at) }
            rule.mainClock.advanceTimeBy(400)
            rule.waitForIdle()
        }
        println("==== $name window ${size.width}x${size.height}")
    }

    @Test fun plainDark() = capture("plain_dark", PLAIN, dark = true)
    @Test fun plainLight() = capture("plain_light", PLAIN, dark = false)
    @Test fun mathDark() = capture("math_dark", MATH, dark = true)
    @Test fun codeAndMathDark() = capture("code_math_dark", CODE + MATH, dark = true)
    @Test fun screenDark() = capture("screen_dark", PLAIN, dark = true, screen = true)
    @Config(sdk = [36]) @Test fun plainDarkSdk36() = capture("plain_dark_sdk36", PLAIN, dark = true)
    @Config(sdk = [36]) @Test fun screenDarkSdk36() = capture("screen_dark_sdk36", PLAIN, dark = true, screen = true)

    private companion object {
        val PLAIN = "結論から言うと、計算はすべて正しいです。\n\nこのまま次の行の計算へ進めて問題ありません。カッコの中をまとめると答えが出ます。"
        val MATH = "計算過程の確認：\n1. **シグマの公式の適用**:\n   \\(9\\sum k^2 - 6\\sum k + \\sum 1\\)\n" +
            "   \\(= \\frac{3}{2}n(2n+1)(n+1)\\) （**正解**）\n\nこのまま次の行の計算へ進めて問題ありません。"
        val CODE = "```python\nprint(1 + 1)\n```\n\n**Output:**\n```\n2\n```\n"
    }
}
