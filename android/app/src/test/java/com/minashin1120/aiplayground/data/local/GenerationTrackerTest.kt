package com.minashin1120.aiplayground.data.local

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class GenerationTrackerTest {
    @Test fun onlyImageAndVideoAnswersKeepTheAppAlive() {
        assertTrue(needsKeepAlive("image"))
        assertTrue(needsKeepAlive("video"))
        assertFalse(needsKeepAlive("chat"))
        assertFalse(needsKeepAlive("tts"))
        assertFalse(needsKeepAlive("transcription"))
    }

    @Test fun theHoldLastsUntilEveryGenerationHasEnded() {
        val tracker = GenerationTracker()
        val image = tracker.begin("image")
        val video = tracker.begin("video")
        assertEquals(listOf("image", "video"), tracker.active.value)

        image.close()
        assertEquals(listOf("video"), tracker.active.value)
        video.close()
        assertTrue(tracker.active.value.isEmpty())
    }

    @Test fun closingTwiceDoesNotEndAnotherGeneration() {
        val tracker = GenerationTracker()
        val first = tracker.begin("image")
        tracker.begin("image")

        first.close()
        first.close()
        assertEquals(listOf("image"), tracker.active.value)
    }

    @Test fun noneNeverHoldsAnything() {
        GenerationKeepAlive.None.begin("video").close()
    }
}
