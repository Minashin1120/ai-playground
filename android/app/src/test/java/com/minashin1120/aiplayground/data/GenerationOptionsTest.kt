package com.minashin1120.aiplayground.data

import org.junit.Assert.*
import org.junit.Test

class GenerationOptionsTest {
    private fun fields(model: String, values: Map<String, String> = emptyMap()) =
        generationPanels(model, values).flatMap { it.fields }.associateBy { it.key }

    @Test fun hiddenProviderOptionsAndArbitraryKeysCannotReachTheRequest() {
        val body = generationOptionsPayload("gpt-image-2", mapOf(
            "image_quality" to "high", "grok_image_count" to "10", "model" to "other", "thread_id" to "other"))
        assertEquals("high", body.getString("image_quality"))
        assertFalse(body.has("grok_image_count"))
        assertFalse(body.has("model"))
        assertFalse(body.has("thread_id"))
    }

    @Test fun invalidNumbersAreRejectedAndUnknownChoicesFallBackToTheDefault() {
        for (value in listOf("0", "11", "1.5", "NaN", "Infinity", "oops")) {
            assertTrue(runCatching { generationOptionsPayload("grok-imagine-image", mapOf("grok_image_count" to value)) }.isFailure)
        }
        assertEquals("jpeg", generationOptionsPayload("gpt-image-2", mapOf("image_format" to "exe")).getString("image_format"))
    }

    @Test fun booleanOptionsRemainJsonBooleansAndBlankOptionalNumbersAreOmitted() {
        val ocr = generationOptionsPayload("mistral-ocr-4-0", mapOf("ocr_extract_header" to "true"))
        assertEquals(true, ocr.get("ocr_extract_header"))
        assertEquals(true, ocr.get("ocr_include_image_base64"))
        val chat = generationOptionsPayload("grok-4", emptyMap())
        assertFalse(chat.has("xai_temperature"))
        assertEquals(true, chat.get("xai_parallel_tool_calls"))
        assertFalse(chat.has("enable_file_creation"))
        assertFalse(chat.has("thinking_level"))
    }

    @Test fun videoRulesFollowTheWeb() {
        val veo = fields("veo-3.1-generate-preview")
        assertTrue(veo.getValue("gemini_video_resolution").disabledValues.isEmpty())
        assertEquals(12.0, veo.getValue("gemini_video_duration").max!!, 0.0)
        assertEquals(setOf("4K"), fields("veo-3.1-fast-generate-preview").getValue("gemini_video_resolution").disabledValues)
        assertTrue(fields("gemini-omni-1.1-flash").getValue("gemini_video_duration").hidden)
        // A disabled 4K falls back to 1080p like the Web select.
        assertEquals("1080p", generationOptionsPayload("veo-3.1-lite-generate-preview", mapOf("gemini_video_resolution" to "4K")).getString("gemini_video_resolution"))
        assertEquals("720p", generationOptionsPayload("grok-imagine-video", mapOf("grok_video_resolution" to "1080p")).getString("grok_video_resolution"))
        assertEquals("1080p", generationOptionsPayload("grok-imagine-video-1.5", mapOf("grok_video_resolution" to "1080p")).getString("grok_video_resolution"))
    }

    @Test fun imageRulesFollowTheWeb() {
        assertTrue(fields("gpt-image-2", mapOf("image_format" to "png")).getValue("image_compression").hidden)
        assertFalse(fields("gpt-image-2").getValue("image_compression").hidden)
        assertEquals("1K", generationOptionsPayload("gemini-3.1-flash-lite-image", mapOf("gemini_image_size" to "4K")).getString("gemini_image_size"))
        assertTrue(fields("grok-imagine-image").getValue("grok_image_resolution").hidden)
        assertFalse(fields("grok-imagine-image-2.0").getValue("grok_image_quality").hidden)
        assertFalse(fields("grok-4.20").getValue("xai_logprobs").enabled)
        assertEquals(false, generationOptionsPayload("grok-4.20", mapOf("xai_logprobs" to "true")).get("xai_logprobs"))
    }

    @Test fun ttsVoicesAndSpeedFollowTheProvider() {
        assertEquals("Kore", generationOptionsPayload("gemini-2.5-flash-preview-tts", emptyMap()).getString("tts_voice"))
        // A voice from another provider's list falls back to this provider's default.
        assertEquals("Kore", generationOptionsPayload("gemini-2.5-flash-preview-tts", mapOf("tts_voice" to "alloy")).getString("tts_voice"))
        val google = fields("google-tts")
        assertEquals(listOf("auto", "custom"), google.getValue("tts_voice").options.map { it.first })
        assertTrue(google.getValue("tts_voice_custom").hidden)
        assertFalse(fields("google-tts", mapOf("tts_voice" to "custom")).getValue("tts_voice_custom").hidden)
        assertEquals("ja-JP", generationOptionsPayload("google-tts", emptyMap()).getString("tts_language"))
        assertEquals(2.0, google.getValue("tts_speed").max!!, 0.0)
        assertEquals("1.5", generationOptionsPayload("grok-tts", mapOf("tts_speed" to "3")).getString("tts_speed"))
        assertFalse(fields("gemini-2.5-flash-preview-tts").getValue("tts_speed").enabled)
        assertEquals("openai", ttsProvider("gpt-4o-mini-tts"))
    }

    @Test fun musicAndLyriaRealtimePanels() {
        assertEquals(false, generationOptionsPayload("lyria-3", emptyMap()).get("music_instrumental"))
        assertTrue(generationPanels("lyria-realtime-exp").single { it.studioBar }.fields.isEmpty())
        assertFalse(generationOptionsPayload("lyria-realtime-exp", emptyMap()).has("music_instrumental"))
    }
}
