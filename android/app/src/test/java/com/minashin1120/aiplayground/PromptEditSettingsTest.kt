package com.minashin1120.aiplayground

import com.minashin1120.aiplayground.data.ChatMessage
import com.minashin1120.aiplayground.data.ThreadItem
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [35])
class PromptEditSettingsTest {
    @Test fun cancelRestoresUnsentSettingsRatherThanTheEditedOrLatestPromptsSettings() {
        val before = ChatState(model = "chosen-before", enableThinking = true, enableSearch = false,
            enablePromptCache = true, enableSystemPrompt = true, sysPromptRestore = true,
            chipValues = mapOf("thinking_level" to "low", "reasoning_effort" to "medium"),
            generationValues = mapOf("tts_speed" to "1.25"), canvasMode = true, visionModel = "vision-before")
        val saved = PromptEditSettings.capture(before, modelChosenManually = true)
        val latest = ChatMessage("32", "user", "latest prompt")
        val edited = before.copy(model = "chosen-during-edit", enableThinking = false, enableSearch = true,
            enablePromptCache = false, enableSystemPrompt = false, sysPromptRestore = false,
            chipValues = mapOf("thinking_level" to "high", "reasoning_effort" to "high"),
            generationValues = mapOf("tts_speed" to "0.5"), canvasMode = false, visionModel = "vision-during-edit",
            selected = ThreadItem("thread", "Chat", "model"), messages = listOf(latest))
        val restored = saved.restore(edited)
        assertEquals("chosen-before", restored.model)
        assertTrue(restored.enableThinking)
        assertFalse(restored.enableSearch)
        assertTrue(restored.enablePromptCache)
        assertTrue(restored.enableSystemPrompt)
        assertTrue(restored.sysPromptRestore)
        assertEquals("low", restored.chipValues["thinking_level"])
        assertEquals("medium", restored.chipValues["reasoning_effort"])
        assertEquals("1.25", restored.generationValues["tts_speed"])
        assertTrue(restored.canvasMode)
        assertEquals("vision-before", restored.visionModel)
        assertTrue(saved.modelChosenManually)
        assertSame(edited.messages, restored.messages)
        assertEquals(edited.selected, restored.selected)
    }

    @Test fun settingsSnapshotDoesNotKeepMutableMapsOrRestoreDraftsAndHistory() {
        val chips = mutableMapOf("thinking_level" to "low")
        val generation = mutableMapOf("tts_speed" to "1.25")
        val before = ChatState(draft = "unsent text", chipValues = chips, generationValues = generation)
        val saved = PromptEditSettings.capture(before, modelChosenManually = false)
        chips["thinking_level"] = "high"
        generation["tts_speed"] = "0.5"
        val current = ChatState(draft = "", messages = listOf(ChatMessage("1", "user", "stored text")))
        val restored = saved.restore(current)
        assertEquals("low", restored.chipValues["thinking_level"])
        assertEquals("1.25", restored.generationValues["tts_speed"])
        assertEquals("", restored.draft)
        assertSame(current.messages, restored.messages)
        assertFalse(saved.modelChosenManually)
    }
}
