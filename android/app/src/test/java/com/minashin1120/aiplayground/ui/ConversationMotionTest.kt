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
    fun branchSwitchMessagesAreFresh() {
        val keys = ConversationKeyTracker()
        keys.update(history, streaming = false, liveVisible = false)
        val switched = listOf(history[0], ChatMessage("20", "assistant", "別の回答"))
        keys.update(switched, streaming = false, liveVisible = false)
        assertEquals("20", keys.keyOf(switched[1]))
        assertTrue(keys.consumeFresh("20"))
    }

    @Test
    fun typingDotsPeakInTurn() {
        assertEquals(1f, typingDotAlpha(0.5f, 0), 0.001f)
        assertEquals(0.3f, typingDotAlpha(0f, 0), 0.001f)
        assertTrue(typingDotAlpha(0.5f + 1f / 3f, 1) > 0.99f)
        for (i in 0..20) {
            val alpha = typingDotAlpha(i / 20f, 2)
            assertTrue(alpha in 0.3f..1f)
        }
    }

    @Test
    fun settingsTabsSlideTowardTheChosenTab() {
        val tabs = listOf("一般", "表示", "データ")
        assertEquals(1, settingsTabDirection(tabs, "一般", "データ"))
        assertEquals(-1, settingsTabDirection(tabs, "データ", "表示"))
        assertEquals(1, settingsTabDirection(tabs, "不明", "表示"))
    }

    @Test
    fun staggerDelayIsCapped() {
        assertEquals(0, staggerDelay(0))
        assertEquals(80, staggerDelay(2))
        assertEquals(320, staggerDelay(50))
    }
}
