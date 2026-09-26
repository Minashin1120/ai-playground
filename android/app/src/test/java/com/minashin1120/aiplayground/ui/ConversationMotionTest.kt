package com.minashin1120.aiplayground.ui

import com.minashin1120.aiplayground.data.ChatMessage
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ConversationMotionTest {
    private val history = listOf(ChatMessage("10", "user", "前の質問"), ChatMessage("11", "assistant", "前の回答"))

    @Test
    fun openedHistoryDoesNotPlayEntryMotion() {
        val keys = ConversationKeyTracker()
        keys.update(history, streaming = false, liveVisible = false)
        assertFalse(keys.consumeFresh("10"))
        assertFalse(keys.consumeFresh("11"))
    }

    @Test
    fun storedMessagesTakeOverStreamedRows() {
        val keys = ConversationKeyTracker()
        keys.update(history, streaming = false, liveVisible = false)
        keys.update(history, streaming = true, liveVisible = true)
        val liveKey = keys.liveKey
        assertTrue(keys.consumeFresh(liveKey))

        val local = ChatMessage("local-abc", "user", "こんにちは")
        val withLocal = history + local
        keys.update(withLocal, streaming = true, liveVisible = true)
        assertTrue(keys.consumeFresh("local-abc"))
        assertFalse(keys.consumeFresh("local-abc"))

        val stored = history + ChatMessage("12", "user", "こんにちは") + ChatMessage("13", "assistant", "回答です")
        keys.update(stored, streaming = true, liveVisible = true)
        assertEquals("local-abc", keys.keyOf(stored[2]))
        assertEquals(liveKey, keys.keyOf(stored[3]))
        assertTrue(keys.liveConsumed)
        assertFalse(keys.consumeFresh("local-abc"))
        assertFalse(keys.consumeFresh(liveKey))
        // The next stream gets its own key so it never collides with the reply that took the old one.
        assertTrue(keys.liveKey != liveKey)

        keys.update(stored, streaming = false, liveVisible = false)
        keys.update(stored, streaming = true, liveVisible = true)
        assertFalse(keys.liveConsumed)
        assertTrue(keys.consumeFresh(keys.liveKey))
    }

    @Test
    fun rowKeysStayUniqueWhenACarriedKeyMeetsItsOriginalRow() {
        val keys = ConversationKeyTracker()
        val local = ChatMessage("local-abc", "user", "こんにちは")
        keys.update(history + local, streaming = false, liveVisible = false)
        val stored = history + ChatMessage("12", "user", "こんにちは")
        keys.update(stored, streaming = false, liveVisible = false)
        assertEquals("local-abc", keys.keyAt(2, stored[2]))
        // A stale snapshot or a retry can show the local row next to the stored one it handed its key to.
        val mixed = history + local + stored[2]
        keys.update(mixed, streaming = false, liveVisible = false)
        val rowKeys = mixed.indices.map { keys.keyAt(it, mixed[it]) }
        assertEquals(rowKeys.size, rowKeys.toSet().size)
    }

    @Test
    fun duplicatedMessageIdsGetDistinctRowKeys() {
        val keys = ConversationKeyTracker()
        val duplicated = history + history[1]
        keys.update(duplicated, streaming = false, liveVisible = false)
        val rowKeys = duplicated.indices.map { keys.keyAt(it, duplicated[it]) }
        assertEquals(rowKeys.size, rowKeys.toSet().size)
        assertEquals("11", rowKeys[1])
    }

    @Test
    fun branchSwitchMessagesAreFresh() {
        val keys = ConversationKeyTracker()
        keys.update(history, streaming = false, liveVisible = false)
        val switched = listOf(history[0], ChatMessage("20", "assistant", "別の回答"))
        keys.update(switched, streaming = false, liveVisible = false)
        assertEquals("20", keys.keyOf(switched[1]))
        assertTrue(keys.consumeFresh("20"))
    }

    @Test
    fun staggerDelayIsCapped() {
        assertEquals(0, staggerDelay(0))
        assertEquals(80, staggerDelay(2))
        assertEquals(320, staggerDelay(50))
    }
}
