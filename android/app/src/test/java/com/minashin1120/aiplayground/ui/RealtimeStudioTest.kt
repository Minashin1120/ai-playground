package com.minashin1120.aiplayground.ui

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class RealtimeStudioTest {
    @Test
    fun voiceFallsBackToProviderDefaultWhenPreviousVoiceDoesNotApply() {
        val options = RealtimeOptions(voice = "alloy")

        assertEquals("alloy", resolvedRealtimeVoice("gpt-realtime-2", options))
        assertEquals("Ara", resolvedRealtimeVoice("grok-voice-fast-1.0", options))
        assertEquals("Kore", resolvedRealtimeVoice("gemini-3.1-flash-live-preview", options))
        assertEquals("Kore", resolvedRealtimeVoice("gemini-3.1-flash-live-preview", RealtimeOptions(voice = "Kore")))
    }

    @Test
    fun transcriptionAndTranslationModelsHaveNoVoiceChoice() {
        assertTrue(realtimeVoices("gemini-3.5-transcribe-live").isEmpty())
        assertTrue(realtimeVoices("grok-voice-transcribe-2.0").isEmpty())
        assertTrue(realtimeVoices("gemini-3.5-live-translate-preview").isEmpty())
    }

    @Test
    fun thinkingLevelsFollowTheWebPanel() {
        assertEquals(listOf("low", "medium", "high"), realtimeThinkingLevels("gemini-3.8-live-extended-thinking"))
        assertTrue(realtimeThinkingLevels("gemini-3.8-live").isEmpty())
        assertTrue(realtimeThinkingLevels("gpt-realtime-2").isEmpty())
        assertEquals("medium", resolvedRealtimeThinking("gemini-3.8-live-extended-thinking", RealtimeOptions(thinkingLevel = "minimal")))
        assertEquals("minimal", resolvedRealtimeThinking("gemini-3.1-flash-live-preview", RealtimeOptions()))
        assertEquals("high", resolvedRealtimeThinking("gemini-3.1-flash-live-preview", RealtimeOptions(thinkingLevel = "high")))
    }

    @Test
    fun dockLabelsAndNotesFollowTheWeb() {
        assertEquals("Speech-to-Speech Live", realtimeModeLabel("gpt-realtime-2"))
        assertEquals("Realtime Speech-to-Text", realtimeModeLabel("gemini-3.5-transcribe-live"))
        assertEquals("Realtime Translation", realtimeModeLabel("gemini-3.5-live-translate-preview"))
        assertEquals("OpenAI Realtimeは24kHz PCM固定", realtimeNote("gpt-realtime-2"))
        assertEquals("xAIはPCMサンプルレート変更可", realtimeNote("grok-voice-fast-1.0"))
        assertEquals("Gemini Liveは音声速度変更非対応", realtimeNote("gemini-3.1-flash-live-preview"))
        assertEquals("xai", realtimeProvider("grok-voice-fast-1.0"))
    }
}
