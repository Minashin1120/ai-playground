package com.minashin1120.aiplayground.data

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** Step parsing and progress rules shared with the server contract. */
class MathStepsTest {
    @Test fun splitsJapaneseHeadingsAndKeepsBodies() {
        val text = """
            前置きは無視します。

            ## ステップ 1 問題を整理する
            与えられた数を確認します。

            ```mathsvg
            <svg viewBox="0 0 10 10"><circle cx="5" cy="5" r="4"/></svg>
            ```

            ## ステップ2 公式を使う
            公式に代入します。
        """.trimIndent()
        val steps = parseMathSteps(text)
        assertEquals(2, steps.size)
        assertEquals("問題を整理する", steps[0].title)
        assertTrue(steps[0].body.contains("与えられた数"))
        assertTrue(steps[0].body.contains("mathsvg"))
        assertEquals("公式を使う", steps[1].title)
        assertTrue(steps[1].body.contains("公式に代入"))
    }

    @Test fun acceptsEnglishAndLevelThreeHeadings() {
        val text = "### ステップ1 準備\n本文A\n\n## Step 2 計算\n本文B"
        val steps = parseMathSteps(text)
        assertEquals(listOf("準備", "計算"), steps.map { it.title })
        assertEquals(listOf(1, 2), steps.map { it.number })
    }

    @Test fun ignoresHeadingsInsideFencedCode() {
        val text = """
            ## ステップ 1 最初
            計算します。

            ```text
            ## ステップ 2 これはコード
            ```

            ## ステップ 2 次
            完了です。
        """.trimIndent()
        val steps = parseMathSteps(text)
        assertEquals(2, steps.size)
        assertEquals("次", steps[1].title)
        assertTrue(steps[0].body.contains("## ステップ 2 これはコード"))
    }

    @Test fun noHeadingFallsBackToNormalMarkdown() {
        assertTrue(parseMathSteps("ただの回答です。\n\n**太字**").isEmpty())
    }

    @Test fun progressClampsAndNeverDecreasesMax() {
        val parsed = MathStepsState.parse(JSONObject("""{"current":3,"max_reached":1,"simplified":{"2":"やさしい"}}"""))!!
        assertEquals(3, parsed.current)
        assertEquals(3, parsed.maxReached)
        assertEquals("やさしい", parsed.simplified[2])
        val moved = parsed.withCurrent(1)
        assertEquals(1, moved.current)
        assertEquals(3, moved.maxReached)
        assertEquals(50, MathStepsState.parse(JSONObject("""{"current":999}"""))!!.current)
        assertNull(MathStepsState.parse(null))
    }

    @Test fun configUsesDefaultsAndRejectsUnknownValues() {
        val defaults = MathStepsConfig()
        assertEquals(false, defaults.basics)
        assertEquals("easy", defaults.level)
        assertEquals("normal", defaults.stepSize)
        assertEquals("auto", defaults.diagrams)
        assertEquals(true, defaults.animation)

        val parsed = MathStepsConfig.parse(JSONObject("""{"basics":true,"level":"huge","step_size":"tiny","diagrams":"sometimes","animation":false}"""))
        assertEquals(true, parsed.basics)
        assertEquals("easy", parsed.level)
        assertEquals("normal", parsed.stepSize)
        assertEquals("auto", parsed.diagrams)
        assertEquals(false, parsed.animation)
    }

    @Test fun copyTextKeepsHeadingsAndReplacesMathSvg() {
        val text = "## ステップ 1 整理\n本文\n\n```mathsvg\n<svg viewBox=\"0 0 1 1\"></svg>\n```"
        val copied = mathStepsCopyText(text)
        assertTrue(copied.contains("## ステップ 1 整理"))
        assertTrue(copied.contains("［図］"))
        assertTrue(!copied.contains("```mathsvg"))
    }

    @Test fun promptValuesFollowEachOption() {
        val values = MathStepsConfig(basics = true, level = "detailed", stepSize = "small", diagrams = "off", animation = false).promptValues()
        assertEquals(5, values.size)
        assertTrue(values.getValue("math_basics").contains("前提"))
        assertTrue(values.getValue("math_level").contains("根拠"))
        assertTrue(values.getValue("math_step_size").contains("小さめ"))
        assertTrue(values.getValue("math_diagrams").contains("使わない"))
        assertTrue(values.getValue("math_animation").contains("静止画"))
    }
}
