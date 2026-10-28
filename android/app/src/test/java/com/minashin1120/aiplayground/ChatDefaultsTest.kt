package com.minashin1120.aiplayground

import com.minashin1120.aiplayground.data.*
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [35])
class ChatDefaultsTest {
    private fun state(useLast: Boolean = false): ChatState {
        val models = listOf("saved", "default", "last").map {
            ModelInfo(it, it, "", "", "chat", emptySet(), false, true)
        }
        val preferences = parsePreferences(JSONObject()
            .put("default_model", "default").put("last_model", "last")
            .put("use_last_chat_settings", useLast)
            .put("default_enable_thinking", true).put("default_enable_search", true)
            .put("default_enable_system_prompt", true).put("default_enable_mcp", false)
            .put("default_thinking_level", "high").put("default_thinking_budget", 2048)
            .put("last_enable_thinking", false).put("last_enable_search", false)
            .put("last_thinking_level", "low").put("last_thinking_budget", 512))
        return ChatState(account = Account(1, "account", models, "default", false),
            model = "saved", preferences = preferences)
    }

    @Test fun startupUsesAccountDefaultsInsteadOfTheDeviceModel() {
        val applied = state().withNewChatDefaults()
        assertEquals("default", applied.model)
        assertTrue(applied.enableThinking)
        assertTrue(applied.enableSearch)
        assertTrue(applied.enableSystemPrompt)
        assertFalse(applied.enableMcp)
        assertEquals("2048", applied.chipValues["thinking_budget"])
    }

    @Test fun lastChatPreferencesApplyOnlyWhenEnabledAndAvailable() {
        val initial = state(true)
        val applied = initial.withNewChatDefaults()
        assertEquals("last", applied.model)
        assertFalse(applied.enableThinking)
        assertFalse(applied.enableSearch)
        assertEquals("low", applied.chipValues["thinking_level"])
        assertEquals("512", applied.chipValues["thinking_budget"])
        assertEquals("default", initial.copy(preferences = initial.preferences!!.copy(lastModel = ""))
            .withNewChatDefaults().model)
    }

    @Test fun openChatsAndActiveAnswersAreNotResetByLatePreferences() {
        val open = state().copy(selected = ThreadItem("thread", "Chat", "saved"))
        assertSame(open, open.withNewChatDefaults())
        val active = state().copy(streaming = true)
        assertSame(active, active.withNewChatDefaults())
        assertEquals("saved", state().withNewChatDefaults(preserveSelectedModel = true).model)
    }

    @Test fun unavailableDefaultKeepsASelectableFallbackAndMissingCacheIsSafe() {
        val initial = state()
        assertEquals("saved", initial.copy(preferences = initial.preferences!!.copy(defaultModel = "removed"))
            .withNewChatDefaults().model)
        val uncached = initial.copy(preferences = null)
        assertSame(uncached, uncached.withNewChatDefaults())
    }

    @Test fun newChatClearsModelForcedSystemPromptMemoryBeforeApplyingDefaults() {
        val applied = state().copy(sysPromptSuppressed = true, sysPromptRestore = true)
            .withNewChatDefaults()
        assertFalse(applied.sysPromptSuppressed)
        assertFalse(applied.sysPromptRestore)
        assertTrue(applied.enableSystemPrompt)
    }
}
