package com.minashin1120.aiplayground.ui

import com.minashin1120.aiplayground.data.CANVAS_PLACEHOLDER_LINE
import com.minashin1120.aiplayground.data.parseCanvasMarkdown
import org.junit.Assert.assertEquals
import org.junit.Test

class CanvasMarkdownTest {
    @Test fun canvasModeShowsAPlaceholderForEachFence() {
        assertEquals(
            listOf(MarkdownBlock.Paragraph("前"), MarkdownBlock.CanvasPlaceholder, MarkdownBlock.Paragraph("後")),
            parseMarkdownBlocks(parseCanvasMarkdown("前\n```js\nx\n```\n後").renderText),
        )
        assertEquals(listOf(MarkdownBlock.Paragraph("本文"), MarkdownBlock.CanvasPlaceholder),
            parseMarkdownBlocks("本文\n$CANVAS_PLACEHOLDER_LINE"))
    }
}
