package com.minashin1120.aiplayground

import com.minashin1120.aiplayground.data.ThreadItem
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class BubbleNotificationsTest {
    @Test
    fun blankThreadIdOpensNewChat() {
        assertNull(bubbleThreadId(null))
        assertNull(bubbleThreadId("  "))
    }

    @Test
    fun threadIdIsTrimmedBeforeItIsPlacedInIntent() {
        assertEquals("thread-1", bubbleThreadId("  thread-1  "))
    }

    @Test
    fun emptyTitleUsesNewChatLabel() {
        assertEquals("新しいチャット", bubbleTitle(null))
        assertEquals("新しいチャット", bubbleTitle(ThreadItem("t1", " ", "model")))
    }

    @Test
    fun titleIsPreservedForCurrentThread() {
        assertEquals("仕事の相談", bubbleTitle(ThreadItem("t1", "仕事の相談", "model")))
    }
}
