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
}
