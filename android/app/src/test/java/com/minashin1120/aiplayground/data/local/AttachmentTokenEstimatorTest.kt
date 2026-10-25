package com.minashin1120.aiplayground.data.local

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class AttachmentTokenEstimatorTest {
    @Test fun gemini3UsesDefaultMediaResolutionRates() {
        assertEquals(1120, AttachmentTokenEstimator.mediaTokens("gemini-3.1-pro-preview", "image/png"))
        assertEquals(1680, AttachmentTokenEstimator.mediaTokens("gemini-3-pro-preview", "application/pdf", pages = 3))
        assertEquals(63, AttachmentTokenEstimator.mediaTokens("gemini-3-pro-preview", "audio/wav", seconds = 2.5))
        assertEquals(273, AttachmentTokenEstimator.mediaTokens("gemini-3-pro-preview", "video/mp4", seconds = 2.5))
        assertEquals(210, AttachmentTokenEstimator.mediaTokens("gemini-3-pro-preview", "video/mp4", seconds = 2.5, hasAudio = false))
    }

    @Test fun legacyUsesTilesAndLegacyMediaRates() {
        assertEquals(258, AttachmentTokenEstimator.mediaTokens("gemini-2.5-flash", "image/jpeg", 384, 384))
        assertEquals(1032, AttachmentTokenEstimator.mediaTokens("gemini-2.5-flash", "image/jpeg", 1024, 1024))
        assertEquals(516, AttachmentTokenEstimator.mediaTokens("gemini-2.5-flash", "application/pdf", pages = 2))
        assertEquals(80, AttachmentTokenEstimator.mediaTokens("gemini-2.5-flash", "audio/wav", seconds = 2.5))
        assertEquals(658, AttachmentTokenEstimator.mediaTokens("gemini-2.5-flash", "video/mp4", seconds = 2.5))
        assertNull(AttachmentTokenEstimator.mediaTokens("gpt-5", "image/png"))
    }
}
