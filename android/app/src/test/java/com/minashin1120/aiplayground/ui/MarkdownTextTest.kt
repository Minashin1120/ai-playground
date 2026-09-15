package com.minashin1120.aiplayground.ui

import org.junit.Assert.*
import org.junit.Test

class MarkdownTextTest {
    @Test fun parsesHeadingsQuotesListsAndCodeWithoutExecutingMarkup() {
        val blocks = parseMarkdownBlocks("""# 見出し

> 引用
- 項目

```kotlin
println("安全")
```
""")
        assertTrue(blocks[0] is MarkdownBlock.Heading)
        assertTrue(blocks[1] is MarkdownBlock.Quote)
        assertTrue(blocks[2] is MarkdownBlock.ListItem)
        val code = blocks[3] as MarkdownBlock.Code
        assertEquals("kotlin", code.language)
        assertEquals("println(\"安全\")", code.text)
    }

    @Test fun unclosedFenceRemainsABoundedCodeBlock() {
        val blocks = parseMarkdownBlocks("```text\n<not-html>")
        assertEquals("<not-html>", (blocks.single() as MarkdownBlock.Code).text)
    }

    @Test fun onlyAbsoluteHttpLinksAreAllowed() {
        assertEquals("https://example.com/docs", safeWebUrl("https://example.com/docs"))
        assertNull(safeWebUrl("javascript:alert(1)"))
        assertNull(safeWebUrl("https:///missing-host"))
    }

    @Test fun tablesAreParsedIntoHeadersAndRows() {
        val blocks = parseMarkdownBlocks("""
| 名前 | 値 |
| --- | --- |
| a | 1 |
| b | 2 |
""")
        val table = blocks.single() as MarkdownBlock.Table
        assertEquals(listOf("名前", "値"), table.headers)
        assertEquals(listOf(listOf("a", "1"), listOf("b", "2")), table.rows)
    }

    @Test fun displayMathIsCapturedAsItsOwnBlock() {
        val blocks = parseMarkdownBlocks("$$\nE = mc^2\n$$")
        val math = blocks.single() as MarkdownBlock.Math
        assertTrue(math.display)
        assertEquals("E = mc^2", math.tex)
        val bracketed = parseMarkdownBlocks("\\[a^2 + b^2 = c^2\\]")
        assertTrue((bracketed.single() as MarkdownBlock.Math).display)
    }

    @Test fun attachmentsBecomeImageBlocksAndForeignImagesStayText() {
        val blocks = parseMarkdownBlocks("![図](/files/123/pic.png)")
        assertEquals(MarkdownBlock.Image("123/pic.png", "図"), blocks.single())
        val foreign = parseMarkdownBlocks("![x](https://example.org/a.png)")
        assertTrue(foreign.single() is MarkdownBlock.Paragraph)
    }
}
