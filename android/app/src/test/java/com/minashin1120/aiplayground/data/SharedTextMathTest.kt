package com.minashin1120.aiplayground.data

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class SharedTextMathTest {
    private fun wrap(text: String) = SharedTextMath.addMissingDelimiters(text)

    @Test fun wrapsAFormulaInsideJapaneseText() {
        assertEquals(
            "二次方程式の解は \$x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}\$ です。",
            wrap("二次方程式の解は x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a} です。"),
        )
    }

    @Test fun aFormulaAloneOnItsLineBecomesDisplayMath() {
        assertEquals("説明\n\$\$x = \\frac{1}{2}\$\$\n終わり", wrap("説明\nx = \\frac{1}{2}\n終わり"))
    }

    @Test fun englishWordsStayOutsideTheMath() {
        assertEquals("where \$\\alpha\$ is the learning rate", wrap("where \\alpha is the learning rate"))
    }

    @Test fun listMarkersStayOutsideTheMath() {
        assertEquals("- \$\\alpha + \\beta\$", wrap("- \\alpha + \\beta"))
    }

    @Test fun textWithDelimitersOrWithoutMathCommandsIsUnchanged() {
        assertEquals("\$x\$ と \\alpha", wrap("\$x\$ と \\alpha"))
        assertEquals("\\(x\\) と \\alpha", wrap("\\(x\\) と \\alpha"))
        assertEquals("C:\\Users\\name\\notes", wrap("C:\\Users\\name\\notes"))
    }

    @Test fun codeAndEnvironmentsAreUnchanged() {
        assertEquals("```tex\n\\alpha\n```\n\$\$\\beta\$\$", wrap("```tex\n\\alpha\n```\n\\beta"))
        assertEquals("`\\alpha` は \$\\beta\$", wrap("`\\alpha` は \\beta"))
        val aligned = "\\begin{aligned}\na &= \\frac{1}{2} \\\\\n\\end{aligned}"
        assertEquals(aligned, wrap(aligned))
    }

    // Android's ICU regex engine throws on a bare brace that the JVM accepts (the share preview crashed on
    // opening), so the JVM test run cannot catch it by compiling the patterns; check the source form instead.
    @Test fun patternsEscapeEveryBraceForIcu() {
        SharedTextMath.PATTERNS.forEach { regex ->
            val source = regex.pattern.replace("\\\\", "").replace(Regex("""\{\d+(?:,\d*)?\}"""), "")
            val bare = Regex("""(?<!\\)[{}]""").find(source)
            assertTrue("bare brace in ${regex.pattern}", bare == null)
        }
    }
}
