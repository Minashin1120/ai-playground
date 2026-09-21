package com.minashin1120.aiplayground.ui

import org.junit.Assert.assertEquals
import org.junit.Test

class ChatTransitionVeilTest {
    @Test
    fun veilIsInvisibleAtBothEnds() {
        assertEquals(0f, chatTransitionVeilAlpha(0f, newChat = false), 0.001f)
        assertEquals(0f, chatTransitionVeilAlpha(1f, newChat = false), 0.001f)
        assertEquals(0f, chatTransitionVeilAlpha(0f, newChat = true), 0.001f)
        assertEquals(0f, chatTransitionVeilAlpha(1f, newChat = true), 0.001f)
    }

    @Test
    fun historyVeilMatchesWebKeyframes() {
        assertEquals(0.22f, chatTransitionVeilAlpha(0.30f, newChat = false), 0.001f)
        assertEquals(0.72f, chatTransitionVeilAlpha(0.52f, newChat = false), 0.001f)
    }

    @Test
    fun newChatVeilPeaksEarlierAndHigher() {
        assertEquals(0.58f, chatTransitionVeilAlpha(0.36f, newChat = true), 0.001f)
        assertEquals(0.29f, chatTransitionVeilAlpha(0.18f, newChat = true), 0.001f)
    }

    @Test
    fun progressIsClampedOutsideTheAnimation() {
        assertEquals(0f, chatTransitionVeilAlpha(-1f, newChat = false), 0.001f)
        assertEquals(0f, chatTransitionVeilAlpha(2f, newChat = false), 0.001f)
        assertEquals(0f, chatTransitionVeilAlpha(-1f, newChat = true), 0.001f)
        assertEquals(0f, chatTransitionVeilAlpha(2f, newChat = true), 0.001f)
    }
}
