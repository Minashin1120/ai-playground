package com.minashin1120.aiplayground.ui

import com.minashin1120.aiplayground.data.ChatMessage
import org.junit.Assert.assertEquals
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
}
