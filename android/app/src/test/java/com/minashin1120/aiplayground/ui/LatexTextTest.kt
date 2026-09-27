package com.minashin1120.aiplayground.ui

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class LatexTextTest {
    @Test fun fractionsRootsAndScriptsBecomeReadableText() {
        assertEquals("(1)/(2)", latexToDisplay("\\frac{1}{2}"))
        assertEquals("\u221A(2)", latexToDisplay("\\sqrt{2}"))
        assertEquals("x\u00B2", latexToDisplay("x^2"))
        assertEquals("H\u2082O", latexToDisplay("H_2O"))
        assertEquals("cm\u00B3", latexToDisplay("\\text{cm}^3"))
        assertEquals("x\u207F\u207A\u00B9", latexToDisplay("x^{n+1}"))
    }

    @Test fun greekAndOperatorCommandsAreMapped() {
        assertEquals("\u03B1 + \u03B2", latexToDisplay("\\alpha + \\beta"))
        assertEquals("\u2211 x", latexToDisplay("\\sum x"))
        assertEquals("\u2264", latexToDisplay("\\leq"))
    }

    @Test fun unknownCommandsStayLiteralInsteadOfFailing() {
        val rendered = latexToDisplay("\\unknowncmd{x}")
        assertTrue(rendered.contains("unknowncmd"))
    }

    @Test fun emptyInputIsEmpty() {
        assertEquals("", latexToDisplay("   "))
    }

    @Test fun inlineMathFromGeminiReplyRenders() {
        assertEquals("E = mc\u00B2", latexToDisplay(" E = mc^2 "))
        assertEquals("x = (-b \u00B1 \u221A(b\u00B2 - 4ac))/(2a)", latexToDisplay("x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}"))
    }

    // Android's ICU regex engine throws on a bare brace that the JVM accepts, so the JVM
    // test run cannot catch it by compiling the patterns; check the source form instead.
    @Test fun patternsEscapeEveryBraceForIcu() {
        LATEX_PATTERNS.forEach { regex ->
            val bare = Regex("""(?<!\\)[{}]""").find(regex.pattern.replace("\\\\", ""))
            assertTrue("bare brace in ${regex.pattern}", bare == null)
        }
    }
}
