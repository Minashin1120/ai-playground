package com.minashin1120.aiplayground.ui

import com.minashin1120.aiplayground.data.SECRET_MASK
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/** Web settings search (`filterSettings` / `getSectionSnippet`) and form helpers. */
class SettingsSearchTest {
    private fun card(title: String?, search: String) = SettingsCardSpec(SettingsTab.General, "k", title, search) {}

    @Test fun matchingIsCaseInsensitiveOverTitleAndText() {
        assertTrue(card("送信設定", "Enterで送信").matches("enter"))
        assertTrue(card("送信設定", "").matches("送信"))
        assertFalse(card("送信設定", "Enterで送信").matches(""))
        assertFalse(card("送信設定", "Enterで送信").matches("MCP"))
    }

    @Test fun snippetKeepsTwentyFiveCharactersBeforeAndThirtyFiveAfter() {
        val text = "a".repeat(40) + "needle" + "b".repeat(50)
        val snippet = settingsSnippet(text, "NEEDLE")
        assertEquals("…" + "a".repeat(25) + "needle" + "b".repeat(35) + "…", snippet)
        assertEquals("", settingsSnippet("abc", "zzz"))
    }

    @Test fun temporaryChatTimeoutFollowsWebNormalization() {
        assertEquals(90, normalizeTempChatTimeout("90"))
        assertEquals(10, normalizeTempChatTimeout("3"))
        assertEquals(10, normalizeTempChatTimeout(""))
        assertEquals(3600, normalizeTempChatTimeout("99999"))
    }

    @Test fun apiKeyPreviewMasksLikeWeb() {
        assertEquals("", maskApiKeyPreview(""))
        assertEquals(SECRET_MASK, maskApiKeyPreview("short"))
        assertEquals("sk-a...wxyz", maskApiKeyPreview("sk-abcdefghwxyz"))
    }
}
