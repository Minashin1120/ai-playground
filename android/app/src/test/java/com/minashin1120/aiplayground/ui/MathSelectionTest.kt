package com.minashin1120.aiplayground.ui

import android.util.LruCache
import android.widget.Magnifier
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.getValue
import androidx.compose.runtime.key
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.hapticfeedback.HapticFeedback
import androidx.compose.ui.hapticfeedback.HapticFeedbackType
import androidx.compose.ui.platform.LocalHapticFeedback
import androidx.compose.ui.platform.LocalTextToolbar
import androidx.compose.ui.platform.TextToolbar
import androidx.compose.ui.platform.TextToolbarStatus
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
 * Diagnostic: long-presses a grid over an answer laid out like a graded math worksheet (headings, nested
 * lists, bold formulas) and prints where a selection starts ("S") and where it does not (".").
 */
@RunWith(RobolectricTestRunner::class)
@GraphicsMode(GraphicsMode.Mode.NATIVE)
@Config(sdk = [35], qualifiers = "w411dp-h4000dp-xxhdpi", shadows = [MathSelectionTest.NoMagnifier::class])
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

    private class Signals {
        var menus = 0
        var haptics = 0
        val total get() = menus + haptics
    }

    private fun svg(width: Int, ascent: Int, depth: Int) =
        """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 -$ascent $width ${ascent + depth}"><g stroke="currentColor" fill="currentColor" stroke-width="0"><rect y="-$ascent" width="$width" height="${ascent + depth}"></rect></g></svg>"""

    private fun <T> field(name: String): T {
        @Suppress("UNCHECKED_CAST")
        return MathJaxRenderer::class.java.getDeclaredField(name).apply { isAccessible = true }.get(null) as T
    }

    /** Typesets every formula the composition asked MathJax for, with a size from its source. */
    private fun typesetPending(): Int {
        val waiting = field<HashMap<String, *>>("waiting")
        val cache = field<LruCache<String, MathSvg>>("cache")
        val ids = waiting.keys.toList()
        ids.forEach { id ->
            val tall = id.contains("frac") || id.contains("left")
            cache.put(id, parseMathSvg(svg(400 + 260 * id.length, if (tall) 1450 else 800, if (tall) 900 else 250))!!)
        }
        waiting.clear()
        return ids.size
    }

    /** The window holding the conversation; message controls add popup roots. */
    private fun main() = rule.onAllNodes(isRoot()).onFirst()

    private fun scan(name: String, content: String, typeset: Boolean) {
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
        var generation by mutableIntStateOf(0)
        rule.setContent {
            PlaygroundTheme(darkTheme = true) {
                CompositionLocalProvider(LocalTextToolbar provides toolbar, LocalHapticFeedback provides haptic) {
                    Box(Modifier.width(411.dp)) {
                        LazyColumn {
                            item { key(generation) { Answer(content) } }
                        }
                    }
                }
            }
        }
        rule.waitForIdle()
        val out = StringBuilder("==== $name\n")
        if (typeset) {
            out.append("typeset ${typesetPending()} formulas\n")
            generation++
            rule.waitForIdle()
        }
        rule.mainClock.advanceTimeBy(2_000)
        rule.waitForIdle()
        main().captureRoboImage("build/outputs/roborazzi/math_selection_$name.png")
        val size = main().fetchSemanticsNode().size
        out.append("root ${size.width}x${size.height}\n")
        for (y in 20 until size.height step 30) {
            out.append(String.format("%5d ", y))
            for (x in 60 until size.width - 30 step 70) {
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

    @Test fun worksheetTypeset() = scan("typeset", WORKSHEET, typeset = true)

    @Test fun worksheetWithoutFormulas() = scan("plain", WORKSHEET.replace(Regex("""\$[^$]+\$|\\\(.+?\\\)"""), "AB"), typeset = false)

    private companion object {
        // "¤" stands for "$", which a raw string would read as a template.
        val WORKSHEET = """採点結果と、それぞれのアドバイスをまとめました。

---

### 📄 1枚目（複素数の練習）

#### 1. 次の数を極形式で表しなさい。
* **(1) ¤1 - i¤**
  * **判定：惜しい**
  * **解説：** 絶対値（¤r = \sqrt{1^2 + (-1)^2} = \sqrt{2}¤）が書かれていません。
    正しくは **\(\sqrt{2}\left(\cos\frac{7}{4}\pi + i\sin\frac{7}{4}\pi\right)\)** となります。
* **(2) ¤\sqrt{3} + i¤**
  * **判定：正解 🎉**
  * **解説：** よくできています。

#### 2. 2つの複素数 ¤\alpha, \beta¤ について、¤\alpha\beta, \frac{\alpha}{\beta}¤ を求めなさい。
* **判定：途中まで**
* **解説：** 「¤\alpha\beta¤」と「¤\frac{\alpha}{\beta}¤」の計算がまだ書かれていません。

#### 3. 点 ¤z¤ をどのように移動した点か。
* **(1) \(\frac{1+\sqrt{3}i}{2}z\)**
  * **判定：正解 🎉**
  * **解説：** 「原点のまわりに \(\frac{\pi}{3}\) 回転した点」—— 説明も含めて十分です。

---

### 📄 2枚目（ド・モアブルの定理）

* **(1) \(\left(\cos\frac{\pi}{3} + i\sin\frac{\pi}{3}\right)^6\)**
  * **判定：正解 🎉**
  * **解説：** 定理を使って正しく計算できています（答え: ¤1¤）。
* **(2) \((1-i)^{8}\)**
  * **判定：間違い ❌**
  * **解説：** 途中の偏角に誤りがあり、答えが ¤4¤ になっていますが、正しくは ¤16¤ です。

#### 3. 方程式 ¤z^3 = i¤ を解きなさい。
* **判定：未解答**
* **解説：** 解答が書かれていません。""".replace('¤', '$')
    }
}
