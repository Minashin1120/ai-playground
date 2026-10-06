package com.minashin1120.aiplayground.ui

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class MathJaxRendererTest {
    @Test fun sizeAndBaselineComeFromTheMathJaxViewBox() {
        // MathJax 3 SVG output for x^2: the viewBox is in 1/1000 em with the baseline at y = 0.
        val markup = """<svg style="vertical-align: -0.025ex;" xmlns="http://www.w3.org/2000/svg" width="2.016ex" height="2.072ex" role="img" focusable="false" viewBox="0 -905.6 891 916.6"><g stroke="currentColor" fill="currentColor"></g></svg>"""
        val svg = parseMathSvg(markup)!!
        assertEquals(0.891f, svg.widthEm, 1e-4f)
        assertEquals(0.9056f, svg.ascentEm, 1e-4f)
        assertEquals(0.011f, svg.depthEm, 1e-4f)
        assertEquals(0.9166f, svg.heightEm, 1e-4f)
    }

    @Test fun unusableOutputFallsBackToTheApproximation() {
        assertNull(parseMathSvg(""))
        assertNull(parseMathSvg("<svg></svg>"))
        assertNull(parseMathSvg("""<svg viewBox="0 0 0 10"></svg>"""))
        assertNull(parseMathSvg("""<svg viewBox="0 -10 10 x"></svg>"""))
    }

    @Test fun formulasMatchTheTextXHeightLikeWebMathJax() {
        // Noto Sans JP x-height (0.543em) over the MathJax TeX font's (0.442em).
        assertEquals(1.2285f, WEB_MATH_SCALE, 1e-3f)
        assertEquals(0.436f, TEXT_CENTER_EM, 1e-4f)
    }
}
