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
        assertEquals(Color(0xFF031413), ramp.textInverse)
        assertEquals(ramp, deriveWebTheme("not a color"))
    }

    @Test
    fun derivesDarkThemeColorsWithWhiteContrastAndLiftedRamp() {
        val blackRamp = deriveWebTheme("#000000")
        assertEquals(Color.Black, blackRamp.t500)
        assertEquals(Color.White, blackRamp.textInverse)
        assertEquals(Color(31, 31, 31), blackRamp.t600)
        assertEquals(Color(56, 56, 56), blackRamp.t700)
        assertEquals(Color(204, 204, 204), blackRamp.t300)
        assertEquals(Color(235, 235, 235), blackRamp.t200)
        assertEquals(Color.White.copy(alpha = 0.18f), blackRamp.bubbleBorder)
        assertEquals(Color.White.copy(alpha = 0.28f), blackRamp.btnBorder)
        assertEquals(Color(46, 46, 46), blackRamp.btnBgStart)
        assertEquals(Color(71, 71, 71), blackRamp.btnBgEnd)
        assertEquals(Color.White.copy(alpha = 0.09f), blackRamp.tabActiveBg)
        assertEquals(Color.White.copy(alpha = 0.20f), blackRamp.tabActiveBorder)

        val darkPalette = webPalette(light = false, themeColor = "#000000")
        assertEquals(Color.White, darkPalette.textInverse)
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
