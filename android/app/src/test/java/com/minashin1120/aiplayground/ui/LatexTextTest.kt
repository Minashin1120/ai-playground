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
}
