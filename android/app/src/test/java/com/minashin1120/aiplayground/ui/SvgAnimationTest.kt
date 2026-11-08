package com.minashin1120.aiplayground.ui

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class SvgAnimationTest {
    @Test fun detectsSmilAnimationElements() {
        assertTrue(svgHasAnimation("""<svg><circle r="4"><animate attributeName="r" values="4;9" dur="1s" repeatCount="indefinite"/></circle></svg>"""))
        assertTrue(svgHasAnimation("""<svg><rect width="4" height="4"><animateTransform attributeName="transform" type="rotate" from="0" to="360" dur="2s"/></rect></svg>"""))
        assertTrue(svgHasAnimation("""<svg><circle r="4"><set attributeName="fill" to="red" begin="1s"/></circle></svg>"""))
        assertTrue(svgHasAnimation("""<svg><path d="M0 0"><animateMotion dur="2s" path="M0 0 L9 9"/></path></svg>"""))
        assertTrue(svgHasAnimation("""<svg><svg:animate xmlns:svg="http://www.w3.org/2000/svg" attributeName="x"/></svg>"""))
    }

    @Test fun detectsCssAnimationInStyleBlocksAndAttributes() {
        assertTrue(svgHasAnimation("""<svg><style>@keyframes spin{to{transform:rotate(1turn)}} .a{animation:spin 1s linear infinite}</style><rect class="a"/></svg>"""))
        assertTrue(svgHasAnimation("""<svg><style>.a{animation-name:pulse}</style><rect class="a"/></svg>"""))
        assertTrue(svgHasAnimation("""<svg><circle style="animation: glow 2s ease-in-out infinite" r="4"/></svg>"""))
    }

    @Test fun stillDrawingStaysOnTheComposePath() {
        assertFalse(svgHasAnimation("""<svg viewBox="0 0 10 10"><rect width="10" height="10" fill="#f00"/></svg>"""))
        assertFalse(svgHasAnimation("""<svg><text x="0" y="9">set animate animation: none</text></svg>"""))
        assertFalse(svgHasAnimation("""<svg><style>.a{fill:red;transition:fill 1s}</style><rect class="a"/></svg>"""))
    }

    @Test fun animatedDocumentIsAnImageDataUriWithTheSvgNamespace() {
        val uri = animatedSvgDataUri("""<svg viewBox="0 0 10 10"><rect width="10" height="10"/></svg>""")
        assertTrue(uri.startsWith("data:image/svg+xml;charset=utf-8,"))
        assertTrue(uri.contains("xmlns=%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22"))
        // An existing root namespace is kept once, and scripts stay inside the image document.
        val existing = animatedSvgDataUri("""<svg xmlns="http://www.w3.org/2000/svg"><rect/></svg>""")
        assertTrue(existing.indexOf("xmlns=%22").let { it >= 0 && existing.indexOf("xmlns=%22", it + 1) == -1 })
        val html = animatedSvgHtml("""<svg viewBox="0 0 1 1"><rect/></svg>""")
        assertTrue(html.contains("<img src=\"data:image/svg+xml;charset=utf-8,"))
    }

    @Test fun xlinkNamespaceIsAddedOnlyWhenUsed() {
        val uri = animatedSvgDataUri("""<svg viewBox="0 0 10 10"><use xlink:href="#a"/></svg>""")
        assertTrue(uri.contains("xmlns:xlink=%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxlink%22"))
        val plain = animatedSvgDataUri("""<svg viewBox="0 0 10 10"><rect/></svg>""")
        assertFalse(plain.contains("xmlns:xlink"))
    }
}
