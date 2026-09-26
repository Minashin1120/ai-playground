package com.minashin1120.aiplayground.ui

import com.minashin1120.aiplayground.data.batchFormatTime
import com.minashin1120.aiplayground.data.batchProviderLabel
import com.minashin1120.aiplayground.data.batchStateShort
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.time.ZoneId

class ConfirmDialogsTest {
    @Test
    fun geminiLocalPythonModeMatchesWeb() {
        assertTrue(isGeminiLocalPythonMode("gemini-3.6-flash", hasAudio = true, hasVideo = false, python = true))
        assertFalse(isGeminiLocalPythonMode("gemini-3.6-flash", hasAudio = true, hasVideo = false, python = false))
        assertFalse(isGeminiLocalPythonMode("gemini-3.6-flash", hasAudio = false, hasVideo = false, python = true))
        assertFalse(isGeminiLocalPythonMode("gemini-3.1-flash-image", hasAudio = true, hasVideo = true, python = true))
        assertFalse(isGeminiLocalPythonMode("gpt-5.5", hasAudio = true, hasVideo = true, python = true))
    }

    @Test
    fun mcpArgumentsArePrettyPrintedWhenJson() {
        assertEquals("{\"a\": 1}", mcpArgsPreview("{\"a\":1}").replace("\n", "").replace("  ", ""))
        assertEquals("not json", mcpArgsPreview("not json"))
    }

    @Test
    fun batchLabelsAndTimeMatchWeb() {
        assertEquals("結果取得中", batchStateShort("job_state_finalizing"))
        assertEquals("確認中", batchStateShort(""))
        assertEquals("xAI", batchProviderLabel("XAI"))
        assertEquals("Batch", batchProviderLabel(""))
        // Naive server times are UTC.
        assertEquals("09/26 14:05", batchFormatTime("2026-09-26T05:05:00", ZoneId.of("Asia/Tokyo")))
        assertEquals("09/26 14:05", batchFormatTime("2026-09-26T05:05:00+00:00", ZoneId.of("Asia/Tokyo")))
        assertEquals("", batchFormatTime(""))
    }
}
