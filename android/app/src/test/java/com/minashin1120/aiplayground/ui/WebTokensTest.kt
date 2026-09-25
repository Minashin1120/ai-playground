package com.minashin1120.aiplayground.ui

import androidx.compose.ui.graphics.Color
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class WebTokensTest {
    @Test
    fun normalizesHexLikeWeb() {
        assertEquals("#aabbcc", normalizeWebHex("ABC"))
        assertEquals("#0dd4bf", normalizeWebHex(" #0DD4BF "))
        assertNull(normalizeWebHex("#12345"))
        assertNull(normalizeWebHex(""))
    }

    @Test
    fun derivesThemeRampWithWebRounding() {
        // deriveTheme('#0dd4bf') in chat_core part01: mix toward black/white with Math.round.
        val ramp = deriveWebTheme(null)
        assertEquals(Color(13, 212, 191), ramp.t500)
        assertEquals(Color(11, 174, 157), ramp.t600)
        assertEquals(Color(9, 144, 130), ramp.t700)
        assertEquals(Color(122, 231, 220), ramp.t300)
        assertEquals(Color(182, 242, 236), ramp.t200)
        assertEquals(ramp, deriveWebTheme("not a color"))
    }

    @Test
    fun lightThemeUsesDarkerAccentForText() {
        val light = webPalette(light = true, themeColor = "#38bdf8")
        assertEquals(light.theme.t700, light.theme300)
        assertEquals(light.theme.t600, light.theme200)
        val dark = webPalette(light = false, themeColor = "#38bdf8")
        assertEquals(dark.theme.t300, dark.theme300)
    }

    @Test
    fun lightThemeRemapsTailwindUtilities() {
        val light = webPalette(light = true, themeColor = null)
        val dark = webPalette(light = false, themeColor = null)
        assertEquals(Color(0xFFE7EDF6), light.twBg(Tw.gray700, 0.5f))
        assertEquals(Tw.gray700.copy(alpha = 0.5f), dark.twBg(Tw.gray700, 0.5f))
        assertEquals(Color(0xFF131C2E), light.twText(Tw.white))
        assertEquals(Color(0xFFCDD7E4), light.twBorder(Tw.gray600))
        assertEquals(Tw.cyan200, dark.twText(Tw.cyan200))
    }
}
