package com.minashin1120.aiplayground.data

import org.junit.Assert.*
import org.junit.Test

class GenerationOptionsTest {
    private fun model(id: String, mode: String) = ModelInfo(id, id, "test", "Test", mode, setOf("thinking"), false, true)

    @Test fun hiddenProviderOptionsAndArbitraryKeysCannotReachTheRequest() {
        val body = generationOptionsPayload(model("gpt-image-2", "image"), mapOf(
            "image_quality" to "high", "grok_image_count" to "10", "model" to "other", "thread_id" to "other"))
        assertEquals("high", body.getString("image_quality"))
        assertFalse(body.has("grok_image_count"))
        assertFalse(body.has("model"))
        assertFalse(body.has("thread_id"))
    }

    @Test fun invalidNumbersAndChoicesAreRejectedBeforeSend() {
        for (value in listOf("0", "11", "1.5", "NaN", "Infinity", "oops")) {
            assertTrue(runCatching { generationOptionsPayload(model("grok-imagine-image", "image"), mapOf("grok_image_count" to value)) }.isFailure)
        }
        assertTrue(runCatching { generationOptionsPayload(model("gpt-image-2", "image"), mapOf("image_format" to "exe")) }.isFailure)
    }

    @Test fun booleanOptionsRemainJsonBooleansAndBlankOptionalNumbersAreOmitted() {
        val ocr = generationOptionsPayload(model("mistral-ocr-4-0", "ocr"), mapOf("ocr_extract_header" to "true"))
        assertEquals(true, ocr.get("ocr_extract_header"))
        assertEquals(true, ocr.get("ocr_include_image_base64"))
        val chat = generationOptionsPayload(model("grok-4", "chat"), emptyMap())
        assertFalse(chat.has("xai_temperature"))
        assertEquals(true, chat.get("enable_file_creation"))
    }
}
