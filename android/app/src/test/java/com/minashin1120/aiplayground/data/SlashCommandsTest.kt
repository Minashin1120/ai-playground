package com.minashin1120.aiplayground.data

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class SlashCommandsTest {
    @Test fun minimalCommandsOnlyShowInMinimalMode() {
        assertEquals(listOf("settings"), visibleSlashCommands("", minimalMode = false).map { it.id })
        val all = visibleSlashCommands("", minimalMode = true).map { it.id }
        assertTrue(all.containsAll(listOf("settings", "options", "thinking", "thinking-high", "effort", "tempchat")))
        assertFalse("fast" in all)
    }

    @Test fun filterMatchesLabelOrDescription() {
        assertEquals(listOf("search"), visibleSlashCommands("search", true).map { it.id })
        // Descriptions count too: "Canvas" appears in /canvas only.
        assertEquals(listOf("canvas"), visibleSlashCommands("Canvas", true).map { it.id })
        assertEquals(listOf("thinking-mid"), visibleSlashCommands(slashSuggestionFilter("/thinking mid"), true).map { it.id })
    }

    @Test fun tokenAndStrippingFollowTheWeb() {
        assertEquals("settings", slashCommandToken("/settingsデフォルト"))
        assertNull(slashCommandToken("hello"))
        assertEquals("デフォルト", stripSlashCommand("/settingsデフォルト"))
        assertEquals("high", stripSlashCommand("/effort high"))
        assertEquals("th", slashPaletteFilter("/th"))
        assertNull(slashPaletteFilter("hi /th"))
    }

    @Test fun toggleArgumentsAcceptJapanese() {
        assertEquals(SlashToggle.ON, parseSlashToggle("オン"))
        assertEquals(SlashToggle.OFF, parseSlashToggle("無効"))
        assertEquals(SlashToggle.TOGGLE, parseSlashToggle(""))
        assertEquals(SlashToggle.INVALID, parseSlashToggle("maybe"))
        assertEquals("medium", SLASH_THINKING_LEVELS["mid"])
    }

    @Test fun aiSettingsHelpers() {
        assertEquals("ON", formatAiSettingValue(true))
        assertEquals("未設定", formatAiSettingValue(null))
        assertEquals("3", formatAiSettingValue(3.0))
        assertEquals("Gemini API Key", apiKeyInfoFor("veo-3.1-generate-preview")?.label)
        assertEquals("xai_key", apiKeyInfoFor("grok-4")?.keyField)
        assertTrue(X_LINK_PATTERN.containsMatchIn("見て https://x.com/user/status/1"))
        assertFalse(X_LINK_PATTERN.containsMatchIn("https://example.com/x"))
    }
}
