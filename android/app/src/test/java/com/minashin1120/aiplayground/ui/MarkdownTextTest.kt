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
        assertEquals(MarkdownBlock.Heading(1, "見出し"), blocks[0])
        val quote = blocks[1] as MarkdownBlock.Quote
        assertEquals(listOf(MarkdownBlock.Paragraph("引用")), quote.blocks)
        val list = blocks[2] as MarkdownBlock.ListBlock
        assertFalse(list.ordered)
        assertEquals(listOf(MarkdownBlock.Paragraph("項目")), list.items.single().blocks)
        val code = blocks[3] as MarkdownBlock.Code
        assertEquals("kotlin", code.language)
        assertEquals("println(\"安全\")", code.text)
    }

    @Test fun orderedListsKeepNumbersAndNestedItems() {
        val blocks = parseMarkdownBlocks("""1. 軽量です
2. `suspend` 関数
   - 入れ子の項目
3. 構造化
""")
        val list = blocks.single() as MarkdownBlock.ListBlock
        assertTrue(list.ordered)
        assertEquals(1, list.start)
        assertEquals(3, list.items.size)
        val nested = list.items[1].blocks[1] as MarkdownBlock.ListBlock
        assertEquals(listOf(MarkdownBlock.Paragraph("入れ子の項目")), nested.items.single().blocks)
        assertEquals(5, (parseMarkdownBlocks("5. five\n6. six").single() as MarkdownBlock.ListBlock).start)
    }

    @Test fun multiLineQuotesFormOneBlock() {
        val quote = parseMarkdownBlocks("> 1行目\n> 2行目").single() as MarkdownBlock.Quote
        assertEquals(listOf(MarkdownBlock.Paragraph("1行目\n2行目")), quote.blocks)
    }

    @Test fun taskItemsAreRecognised() {
        val list = parseMarkdownBlocks("- [x] done\n- [ ] todo").single() as MarkdownBlock.ListBlock
        assertEquals(listOf(true, false), list.items.map { it.checked })
    }

    @Test fun chatErrorBecomesAnErrorBoxAndPyexecIsHidden() {
        val blocks = parseMarkdownBlocks("本文\n\n```pyexec\n{\"code\":\"1\"}\n```\n\n```chat_error\nrate_limit\n```")
        assertEquals(listOf(MarkdownBlock.Paragraph("本文"), MarkdownBlock.ChatError("rate_limit")), blocks)
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

    @Test fun setextHeadingsAndRules() {
        assertEquals(MarkdownBlock.Heading(1, "Title"), parseMarkdownBlocks("Title\n===").single())
        assertEquals(MarkdownBlock.Rule, parseMarkdownBlocks("***").single())
    }

    @Test fun codeDownloadNamesFollowWeb() {
        assertEquals("code.kt", codeDownloadName("kotlin"))
        assertEquals("code.py", codeDownloadName("Python"))
        assertEquals("code.txt", codeDownloadName(""))
        assertEquals("Dockerfile", codeDownloadName("dockerfile"))
        assertEquals("code.txt", codeDownloadName("objective-c"))
        assertEquals("code.vue", codeDownloadName("vue"))
    }

    @Test fun tableColumnsStretchLikeAnAutoLayoutTable() {
        assertArrayEquals(intArrayOf(100, 200), tableColumnWidths(intArrayOf(10, 20), intArrayOf(50, 100), 300))
        val shrunk = tableColumnWidths(intArrayOf(40, 40), intArrayOf(200, 200), 200)
        assertEquals(200, shrunk.sum())
        assertArrayEquals(intArrayOf(80, 90), tableColumnWidths(intArrayOf(80, 90), intArrayOf(300, 300), 100))
    }
}
