package com.minashin1120.aiplayground.data.direct

import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import java.time.ZoneId
import java.time.ZonedDateTime

class SystemPromptBuilderTest {
    private val defaults = ServerlessDefaults(JSONObject()
        .put("coding_mode_system_prompt", "[Coding Mode]")
        .put("auto_system_prompt_notices_config", JSONObject()
            .put("python", JSONObject().put("text", "PYTHON NOTICE"))
            .put("mathjax", JSONObject().put("text", "Use MathJax"))
            .put("openai_search", JSONObject().put("text", "SEARCH NOTICE"))
            .put("attachment_names", JSONObject().put("text", "添付: {{attachment_names}}"))))
    private val now = ZonedDateTime.of(2026, 9, 27, 10, 0, 0, 0, ZoneId.of("Asia/Tokyo"))
    private val thread = SystemPromptBuilder.ThreadSettings("このチャットでは短く", true)

    @Test fun ordersPartsLikeTheServer() {
        val prefs = JSONObject().put("global_system_prompt_effective", "GLOBAL").put("global_system_prompt_uses_time_fallback", false)
            .put("system_prompt", "USER").put("system_prompt_enabled", true)
        val body = JSONObject().put("system_prompt", "GEM").put("enable_system_prompt", true).put("enable_python", true)
        val prompt = SystemPromptBuilder.build(body, prefs, thread, defaults, "gemini", now)
        assertEquals("PYTHON NOTICE\n\nGEM\n\nGLOBAL\n\nUSER\n\n[Chat Specific Instructions]:\nこのチャットでは短く\n\nUse MathJax", prompt)
    }

    @Test fun timeNoticeReplacesAnEmptyGlobalPromptAndSearchNoticeIsProviderSpecific() {
        val prefs = JSONObject().put("global_system_prompt_uses_time_fallback", true)
        val body = JSONObject().put("enable_search", true).put("coding_mode", true)
        val prompt = SystemPromptBuilder.build(body, prefs, SystemPromptBuilder.ThreadSettings("", true), defaults, "openai", now)
        assertTrue(prompt.startsWith("SEARCH NOTICE\n\nCurrent time: 2026-09-27 10:00:00 "))
        assertTrue(prompt.contains("(UTC+0900)"))
        assertTrue(prompt.contains("[Coding Mode]"))
        val gemini = SystemPromptBuilder.build(body, prefs, SystemPromptBuilder.ThreadSettings("", true), defaults, "gemini", now)
        assertFalse(gemini.contains("SEARCH NOTICE"))
    }

    @Test fun chatsWithoutGlobalInstructionsSkipGlobalAndUserPrompts() {
        val prefs = JSONObject().put("global_system_prompt_effective", "GLOBAL").put("global_system_prompt_uses_time_fallback", false)
            .put("system_prompt", "USER").put("apply_auto_system_prompt_notices", false)
        val body = JSONObject().put("enable_system_prompt", true)
        val prompt = SystemPromptBuilder.build(body, prefs, SystemPromptBuilder.ThreadSettings("ONLY", false), defaults, "gemini", now)
        assertEquals("ONLY", prompt)
    }

    @Test fun attachmentNamesUseTheTemplate() {
        assertEquals("添付: 画像1: a.png\n画像2: b.jpg",
            SystemPromptBuilder.attachmentNamesBlock(listOf("local/a.png", "b.jpg"), JSONObject(), defaults))
        assertEquals("", SystemPromptBuilder.attachmentNamesBlock(emptyList(), JSONObject(), defaults))
    }
}
