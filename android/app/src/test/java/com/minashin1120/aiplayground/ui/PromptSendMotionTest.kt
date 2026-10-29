package com.minashin1120.aiplayground.ui

import androidx.compose.ui.geometry.Rect
import org.junit.Assert.*
import org.junit.Test

class PromptSendMotionTest {
    @Test fun reducedMotionAndPausedReaderDoNotCreateAFlight() {
        val motion = PromptSendMotion()
        motion.source = Rect(0f, 400f, 300f, 450f)
        motion.begin("local-a", reduce = true)
        assertNull(motion.id)
        motion.following = false
        motion.begin("local-b", reduce = false)
        assertNull(motion.id)
    }

    @Test fun cancellationDropsDrawingAndCoordinates() {
        val motion = PromptSendMotion()
        motion.source = Rect(0f, 400f, 300f, 450f)
        motion.begin("local-a", reduce = false)
        assertEquals("local-a", motion.id)
        assertEquals(motion.source, motion.origin)
        motion.target = Rect(100f, 100f, 300f, 150f)
        motion.ready = true
        motion.finish()
        assertNull(motion.id)
        assertNull(motion.layer)
        assertNull(motion.target)
        assertFalse(motion.ready)
    }

    @Test fun bubbleHandoffDoesNotCrossfadeTwoTranslucentCopies() {
        assertFalse(promptSendShowsBubble(0f))
        assertFalse(promptSendShowsBubble(0.8f))
        assertFalse(promptSendShowsBubble(0.99f))
        assertTrue(promptSendShowsBubble(1f))
    }

    @Test fun tallBubbleArrivesInsideTheConversationViewport() {
        val origin = Rect(20f, 600f, 300f, 750f)
        val viewport = Rect(0f, 50f, 320f, 590f)
        val start = promptSendClip(origin, viewport, 0f)
        assertTrue(start.contains(origin.topLeft))
        assertEquals(750f, start.bottom)
        assertEquals(viewport, promptSendClip(origin, viewport, 1f))
        assertEquals(670f, promptSendClip(origin, viewport, 0.5f).bottom)
    }
}
