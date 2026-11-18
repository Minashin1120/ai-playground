package com.minashin1120.aiplayground.ui

import com.minashin1120.aiplayground.data.ChatMessage
import com.minashin1120.aiplayground.data.MathStepsState
import com.minashin1120.aiplayground.data.buildMathStepViews
import com.minashin1120.aiplayground.data.parseMathSteps
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class QuoteNavigationTest {
    private val rows = listOf(
        ChatMessage("101", "user", "最初の質問"),
        ChatMessage("102", "assistant", "最初の回答"),
        ChatMessage("103", "user", "次の質問"),
    )

    @Test fun aQuoteOpenedFromAStoredMessageTargetsThatRow() {
        val target = resolveQuoteTarget(rows, sourceKey = "102", sourceMessageId = 102, liveKey = null, liveVisible = false)
        assertEquals(QuoteTarget.Message(1), target)
    }

    @Test fun aStableConversationKeyStillWorksAfterSyncReplacesTheMessageId() {
        // Sync replaced the pending row's id; the key the quote captured is the conversation key.
        val synced = listOf(rows[0], rows[1].copy(id = "9002", conversationKey = "102"), rows[2])
        val target = resolveQuoteTarget(synced, sourceKey = "102", sourceMessageId = null, liveKey = null, liveVisible = false)
        assertEquals(QuoteTarget.Message(1), target)
    }

    @Test fun theNumericIdIsAFallbackWhenTheKeyIsGone() {
        val target = resolveQuoteTarget(rows, sourceKey = "missing", sourceMessageId = 103, liveKey = null, liveVisible = false)
        assertEquals(QuoteTarget.Message(2), target)
    }

    @Test fun aQuoteTakenFromTheStreamingAnswerTargetsTheLiveRow() {
        val target = resolveQuoteTarget(rows, sourceKey = "live-4", sourceMessageId = null, liveKey = "live-4", liveVisible = true)
        assertEquals(QuoteTarget.Live, target)
    }

    @Test fun aLiveKeyThatIsNoLongerVisibleDoesNotScroll() {
        val target = resolveQuoteTarget(rows, sourceKey = "live-4", sourceMessageId = null, liveKey = "live-4", liveVisible = false)
        assertEquals(QuoteTarget.Missing, target)
    }

    @Test fun anUnknownSourceDoesNotScroll() {
        val target = resolveQuoteTarget(rows, sourceKey = "missing", sourceMessageId = null, liveKey = null, liveVisible = false)
        assertEquals(QuoteTarget.Missing, target)
    }

    @Test fun aSentQuoteStripFindsTheNearestPrecedingMessageThatContainsIt() {
        val messages = listOf(
            ChatMessage("101", "assistant", "同じ説明です。**太字**の部分。"),
            ChatMessage("102", "user", "同じ説明です。太字の部分。"),
            ChatMessage("103", "assistant", "同じ説明です。太字の部分。"),
        )
        // The nearest preceding message wins (102, not 101), and Markdown markers do not block the match.
        org.junit.Assert.assertEquals(1, findQuotedMessageIndex(messages, beforeIndex = 2, quote = "太字の部分") ?: -1)
    }

    @Test fun aQuoteMadeFromRenderedTextStillFindsTheStoredMarkdown() {
        val messages = listOf(
            ChatMessage("101", "assistant",
                "使い方は次のとおりです。\n\n- 項目Aを選ぶ\n- 項目Bを押す\n\n`code` も使えます。"),
            ChatMessage("102", "user", "ありがとう"),
        )
        // The selection copies the rendered list bullets and drops the backticks.
        val quote = "• 項目Aを選ぶ • 項目Bを押す code も使えます。"
        org.junit.Assert.assertEquals(0, findQuotedMessageIndex(messages, beforeIndex = 2, quote = quote) ?: -1)
    }

    @Test fun aQuoteAroundAFormulaStillFindsTheMessageWithItsLatex() {
        val messages = listOf(
            ChatMessage("101", "assistant",
                "次の式を計算します。\\frac{-b \\pm \\sqrt{b^2-4ac}}{2a} を使って解きます。"),
        )
        // The reader selected the approximation drawn by the formula renderer, not the LaTeX source.
        val quote = "次の式を計算します。x = (−b ± √(b² − 4ac)) / 2a を使って解きます。"
        org.junit.Assert.assertEquals(0, findQuotedMessageIndex(messages, beforeIndex = 1, quote = quote) ?: -1)
    }

    @Test fun aComposerQuoteWithoutASourceIdUsesTheQuoteTextBeforeTheEditedMessage() {
        val messages = listOf(
            ChatMessage("101", "assistant", "引用される説明です。"),
            ChatMessage("102", "user", "質問です。"),
            ChatMessage("103", "assistant", "別の説明です。"),
        )
        org.junit.Assert.assertEquals(0, findQuoteTextIndex(messages, beforeIndex = 1, quote = "引用される説明です。") ?: -1)
        // The edited message itself is not a source, so searching only before row 1 finds nothing.
        org.junit.Assert.assertNull(findQuoteTextIndex(messages, beforeIndex = 0, quote = "引用される説明です。"))
    }

    @Test fun anUnrelatedQuoteDoesNotFuzzyMatchAnotherAnswer() {
        val messages = listOf(
            ChatMessage("101", "assistant", "今日の天気は晴れで、気温は25度です。"),
            ChatMessage("102", "user", "質問です。"),
        )
        org.junit.Assert.assertNull(findQuotedMessageIndex(messages, beforeIndex = 2, quote = "明日の予定を教えてください。"))
    }

    @Test fun aSentQuoteStripFromAnotherBranchStaysInert() {
        val messages = listOf(ChatMessage("101", "user", "この会話にはない引用元"))
        org.junit.Assert.assertNull(findQuotedMessageIndex(messages, beforeIndex = 0, quote = "別の分岐の文章"))
    }

    @Test fun theFirstMessageCannotQuoteItself() {
        val messages = listOf(ChatMessage("101", "user", "自分の本文"))
        org.junit.Assert.assertNull(findQuotedMessageIndex(messages, beforeIndex = 0, quote = "自分の本文"))
    }

    @Test fun theQuoteRangeIsTheSelectedTextInAPlainBubble() {
        val text = "前の文です。引用されたテキストです。後の文です。"
        val range = findQuoteRange(text, "引用されたテキストです。")
        assertEquals("引用されたテキストです。", text.substring(range!!.first, range.last + 1))
    }

    @Test fun theQuoteRangeSkipsMarkdownMarkersAndWrappedLines() {
        val text = "同じ **太字** の部分と\n改行した部分"
        val range = findQuoteRange(text, "太字 の部分と 改行した部分")
        assertEquals("太字** の部分と\n改行した部分", text.substring(range!!.first, range.last + 1))
    }

    @Test fun aQuoteThatIsNotInTheBubbleHasNoRange() {
        org.junit.Assert.assertNull(findQuoteRange("別の文章", "引用元にない文"))
    }

    @Test fun theJumpScrollsUntilTheQuoteStartsBelowTheTopBar() {
        // The quote was laid out 900px below the top of the window, the list starts at 100px and the
        // margin is 12px, so the content has to move up by 788px.
        assertEquals(788f, quoteScrollDelta(quoteTop = 900f, viewportTop = 100f, margin = 12f), 0.001f)
    }

    @Test fun aQuoteAlreadyUnderTheTopBarDoesNotScroll() {
        assertEquals(0f, quoteScrollDelta(quoteTop = 112f, viewportTop = 100f, margin = 12f), 0.001f)
    }

    @Test fun aFormulaOnlyQuoteMatchesTheApproximationTheReaderCopied() {
        val tex = "\\frac{1}{2}"
        val messages = listOf(
            ChatMessage("101", "assistant", "前の説明です。\n\n$$${tex}$$\n\n続きです。"),
            ChatMessage("102", "user", "質問です。"),
        )
        val quote = latexToDisplay(tex)
        assertEquals("(1)/(2)", quote)
        assertEquals(0, findQuotedMessageIndex(messages, beforeIndex = 2, quote = quote) ?: -1)
        assertTrue(readerQuoteText("値は \\($tex\\) です。").contains(quote))
    }

    @Test fun aQuoteFromALaterMathStepOpensThatStep() {
        val tex = "\\frac{a}{b}"
        val content = "## ステップ 1 導入\n前置きです。\n\n## ステップ 2 計算\n計算すると \\($tex\\) です。"
        val quote = "計算すると ${latexToDisplay(tex)} です。"
        val messages = listOf(
            ChatMessage("101", "assistant", content),
            ChatMessage("102", "user", "質問です。"),
        )
        assertEquals(0, findQuotedMessageIndex(messages, beforeIndex = 2, quote = quote) ?: -1)
        val views = buildMathStepViews(parseMathSteps(content), emptyMap())
        assertEquals(1, findMathQuoteTarget(views, emptyMap(), quote, current = 0)!!.index)
        // The step already on screen wins when it contains the same text.
        assertEquals(1, findMathQuoteTarget(views, emptyMap(), quote, current = 1)!!.index)
    }

    @Test fun anEasierMathStepQuoteFindsTheMessageAndTheOriginalCanBeRevealed() {
        val original = "## ステップ 1 元の説明\nむずかしい原文です。"
        val easier = "やさしい説明です。これは引用です。"
        val message = ChatMessage(
            "101", "assistant", original,
            mathStepsState = MathStepsState(simplified = mapOf(0 to easier)),
        )
        val messages = listOf(message, ChatMessage("102", "user", "質問です。"))
        assertEquals(0, findQuotedMessageIndex(messages, beforeIndex = 2, quote = "これは引用です。") ?: -1)
        val views = buildMathStepViews(parseMathSteps(original), emptyMap())
        val hidden = findMathQuoteTarget(views, mapOf(0 to easier), "むずかしい原文です。", current = 0)
        assertEquals(0, hidden!!.index)
        assertTrue(hidden.revealOriginal)
        assertFalse(findMathQuoteTarget(views, mapOf(0 to easier), "これは引用です。", current = 0)!!.revealOriginal)
    }

    @Test fun codeFencesStayLiteralAndDiagramsBecomeTheFigureMark() {
        val fenced = "```\n\\frac{1}{2}\n```"
        assertEquals(fenced, readerQuoteText(fenced))
        val drawn = "前\n```mathsvg\n<svg></svg>\n```\n後"
        val reader = readerQuoteText(drawn)
        assertTrue(reader.contains("［図］"))
        assertFalse(reader.contains("svg"))
        assertTrue(reader.contains("前"))
        assertTrue(reader.contains("後"))
    }
}
