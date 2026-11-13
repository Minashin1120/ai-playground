package com.minashin1120.aiplayground.data.direct

import com.minashin1120.aiplayground.data.MathStepsConfig
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/** Serverless mode fills the same five placeholders as the server prompt builder. */
class MathStepsPromptTest {
    @Test fun fillsEveryPlaceholderFromThePreferences() {
        val template = "base\n{{math_basics}} / {{ math_level }} / {{math_step_size}} / {{math_diagrams}} / {{math_animation}} / {{math_max_steps}}"
        val filled = SystemPromptBuilder.fillMathStepsPrompt(
            template,
            MathStepsConfig(basics = false, level = "normal", stepSize = "large", diagrams = "auto", animation = true),
        )
        assertFalse(filled.contains("{{"))
        assertTrue(filled.contains("問題の整理から始める"))
        assertTrue(filled.contains("標準"))
        assertTrue(filled.contains("大きめ"))
        assertTrue(filled.contains("必要なときだけ使う"))
        assertTrue(filled.contains("使ってよい"))
        assertTrue(filled.contains("50"))
    }

    @Test fun leavesTextWithoutPlaceholdersAlone() {
        val text = "## ステップ 1\n本文"
        assertTrue(SystemPromptBuilder.fillMathStepsPrompt(text, MathStepsConfig()) == text)
    }
}
