package com.minashin1120.aiplayground.data

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class CanvasBlocksTest {
    @Test fun fencesMoveToTheCanvasLikeWeb() {
        val data = parseCanvasMarkdown("前\n```js\nconst a = 1;\n```\n後")
        assertEquals("前\n$CANVAS_PLACEHOLDER_LINE\n後", data.renderText)
        assertEquals(listOf(CanvasBlock(0, "js", "const a = 1;", false)), data.blocks)

        val open = parseCanvasMarkdown("```py\nprint(1)").blocks.single()
        assertTrue(open.open)
        assertEquals("print(1)", open.code)

        val nested = parseCanvasMarkdown("~~~\n```\ninner\n```\n~~~").blocks.single()
        assertEquals("", nested.lang)
        assertEquals("```\ninner\n```", nested.code)
        assertFalse(nested.open)

        val indented = parseCanvasMarkdown("  ```html\n<p>x</p>\n  ```").blocks.single()
        assertEquals("html", indented.lang)
        assertFalse(indented.open)

        // A longer closing fence does not close the block (Web compares the exact fence).
        val unclosed = parseCanvasMarkdown("```\na\n````").blocks.single()
        assertTrue(unclosed.open)
        assertEquals("a\n````", unclosed.code)
    }

    @Test fun selectionFollowsTheLatestBlockUnlessPicked() {
        val blocks = listOf(CanvasBlock(0, "js", "a", false), CanvasBlock(1, "py", "b", false))
        assertEquals(CanvasSelection(), nextCanvasSelection(emptyList(), CanvasSelection(1, true)))
        assertEquals(CanvasSelection(1), nextCanvasSelection(blocks, CanvasSelection(0)))
        assertEquals(CanvasSelection(0, true), nextCanvasSelection(blocks, CanvasSelection(0, true)))
        assertEquals(CanvasSelection(1), nextCanvasSelection(blocks, CanvasSelection(5, true)))
    }

    @Test fun titleStatusAndPreview() {
        val js = CanvasBlock(0, "js", "\n\n  const   x = 1  \n", false)
        val html = CanvasBlock(0, "html", "<p>", false)
        assertEquals("Canvas Preview: js", canvasTitle(listOf(js), 0))
        assertEquals("HTML Canvas Preview #1/2", canvasTitle(listOf(html, js.copy(index = 1)), 0))
        assertEquals("Canvas Preview: text", canvasTitle(listOf(CanvasBlock(0, "", "x", false)), 0))
        assertEquals("Canvasで表示中", canvasTitle(emptyList(), -1))
        assertTrue(isCanvasHtml("", "<!DOCTYPE html><p>"))
        assertFalse(isCanvasHtml("js", "<html>"))
        assertEquals("const x = 1", canvasBlockPreview(js))
        assertEquals("空のコードブロック", canvasBlockPreview(CanvasBlock(0, "", "  \n", false)))
        assertEquals("コードブロックを待機中", canvasStatus(null))
        assertEquals("コードブロックを生成中", canvasStatus(js.copy(open = true)))
        assertEquals("コードブロックをプレビューしています", canvasStatus(js))
    }
}
