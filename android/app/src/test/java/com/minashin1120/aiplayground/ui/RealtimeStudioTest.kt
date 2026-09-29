package com.minashin1120.aiplayground.ui

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class RealtimeStudioTest {
    @Test
    fun voiceFallsBackToProviderDefaultWhenPreviousVoiceDoesNotApply() {
        val options = RealtimeOptions(voice = "alloy")

        assertEquals("alloy", resolvedRealtimeVoice("gpt-realtime-2", options))
        assertEquals("alloy", resolvedRealtimeVoice("gpt-realtime-2.1", options))
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
        assertEquals("OpenAI Realtimeは24kHz PCM固定（Reasoningで推論の強さを指定）", realtimeNote("gpt-realtime-2"))
        assertEquals("OpenAI Realtimeは24kHz PCM固定", realtimeNote("gpt-realtime-1.5"))
        assertEquals("Speech-to-Speech Live", realtimeModeLabel("gpt-realtime-2.1-mini"))
        assertEquals("OpenAI Realtimeは24kHz PCM固定（Reasoningで推論の強さを指定）", realtimeNote("gpt-realtime-2.1"))
        assertEquals("openai", realtimeProvider("gpt-realtime-2.1"))
        assertEquals("openai", realtimeProvider("gpt-live-1"))
    }

    @Test
    fun reasoningEffortAndGptLiveVoicesFollowTheWeb() {
        assertEquals("low", resolvedRealtimeReasoning("gpt-realtime-2.1", RealtimeOptions()))
        assertEquals("xhigh", resolvedRealtimeReasoning("gpt-realtime-2.1-mini", RealtimeOptions(reasoningEffort = "xhigh")))
        assertEquals(null, resolvedRealtimeReasoning("gpt-realtime-1.5", RealtimeOptions()))
        assertEquals(null, resolvedRealtimeReasoning("gpt-live-1", RealtimeOptions()))
        assertEquals("marin", resolvedRealtimeVoice("gpt-live-1", RealtimeOptions()))
        assertEquals("quartz", resolvedRealtimeVoice("gpt-live-1", RealtimeOptions(voice = "quartz")))
        assertEquals("alloy", resolvedRealtimeVoice("gpt-realtime-2.1", RealtimeOptions(voice = "quartz")))
        assertEquals("xAIはPCMサンプルレート変更可", realtimeNote("grok-voice-fast-1.0"))
        assertEquals("Gemini Liveは音声速度変更非対応", realtimeNote("gemini-3.1-flash-live-preview"))
        assertEquals("xai", realtimeProvider("grok-voice-fast-1.0"))
    }

    @Test
    fun openAiTranslationUsesTargetLanguageInsteadOfVoice() {
        assertTrue(isRealtimeTranslation("gpt-realtime-translate"))
        assertTrue(realtimeVoices("gpt-realtime-translate").isEmpty())
        assertEquals("Realtime Translation", realtimeModeLabel("gpt-realtime-translate"))
        assertEquals("話した内容を選択した言語へリアルタイムで音声翻訳（24kHz PCM・音声選択不可）", realtimeNote("gpt-realtime-translate"))
    }

    @Test
    fun realtimeWhisperIsTranscriptionOnly() {
        assertTrue(isRealtimeTranscription("gpt-realtime-whisper"))
        assertTrue(realtimeVoices("gpt-realtime-whisper").isEmpty())
        assertEquals("Realtime Speech-to-Text", realtimeModeLabel("gpt-realtime-whisper"))
    }

    @Test
    fun gemini25NativeAudioHasNoThinkingLevelAndGrokRatesAreSupported() {
        assertTrue(realtimeThinkingLevels("gemini-2.5-flash-native-audio-preview-12-2025").isEmpty())
        assertTrue(22050 in GROK_PCM_RATES)
        assertTrue(21050 !in GROK_PCM_RATES)
    }
}
