package com.minashin1120.aiplayground.ui

import org.json.JSONTokener
import org.junit.Assert.*
import org.junit.Test

class HtmlCodePreviewTest {
    @Test fun generatedCodeCannotCloseTheTrustedScriptAndPreservesItsCharacters() {
        val code = """<style>body {color:#123456;width:100%}</style><h1>日本語 😀</h1></script><script>alert('test')</script>""" + "\n\u2028\u2029"
        val document = buildHtmlPreviewDocument(code, "https://example.com/", "")
        val argument = document.substringAfter("frame.srcdoc=sanitizeHtmlForPreview(").substringBefore(");\n")
        assertEquals(code, JSONTokener(argument).nextValue())
        assertEquals(1, Regex("</script>", RegexOption.IGNORE_CASE).findAll(document).count())
        assertFalse(document.contains("allow-same-origin"))
        assertTrue(document.contains("sandbox=\"allow-scripts allow-forms allow-modals allow-popups\""))
        assertTrue(document.contains("referrerpolicy=\"no-referrer\""))
    }

    @Test fun previewOriginNeverUsesTheAccountServerOrigin() {
        assertEquals("html-preview.invalid", java.net.URI(HTML_PREVIEW_URL).host)
        assertEquals("https", java.net.URI(HTML_PREVIEW_URL).scheme)
        val document = buildHtmlPreviewDocument("<h1>test</h1>", "https://example.com/", "")
        assertTrue(document.contains("HTMLプレビューを読み込めませんでした"))
    }
}
