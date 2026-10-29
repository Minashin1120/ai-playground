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

    @Test fun realBubbleOnlyRevealsDuringTheLastSeventyMilliseconds() {
        assertEquals(0f, promptSendReveal(0f), 0.001f)
        assertEquals(0f, promptSendReveal(0.8f), 0.001f)
        assertEquals(0.5f, promptSendReveal(0.9f), 0.001f)
        assertEquals(1f, promptSendReveal(1f), 0.001f)
    }
}
