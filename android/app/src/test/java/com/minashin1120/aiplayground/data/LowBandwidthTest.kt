package com.minashin1120.aiplayground.data

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class LowBandwidthTest {
    @Test
    fun detectsLikeWebAutoMode() {
        val fast = detectLowBandwidth(LowBandwidthSignal(saveData = false, effectiveType = "4g", downlinkMbps = 10.0))
        assertFalse(fast.enabled)
        assertEquals("回線:4g", fast.reason)

        val saver = detectLowBandwidth(LowBandwidthSignal(saveData = true, effectiveType = "4g", downlinkMbps = 10.0))
        assertTrue(saver.enabled)
        assertEquals("データ節約 / 回線:4g", saver.reason)

        val slow = detectLowBandwidth(LowBandwidthSignal(saveData = false, effectiveType = "3g", downlinkMbps = 0.4))
        assertTrue(slow.enabled)
        assertEquals("回線:3g / 下り:0.4Mbps", slow.reason)

        assertEquals(LowBandwidthDetection(false, ""), detectLowBandwidth(null))
    }

    @Test
    fun mapsDownstreamEstimateToChromeBuckets() {
        assertEquals("", effectiveConnectionType(0))
        assertEquals("slow-2g", effectiveConnectionType(40))
        assertEquals("2g", effectiveConnectionType(60))
        assertEquals("3g", effectiveConnectionType(500))
        assertEquals("4g", effectiveConnectionType(20000))
        assertEquals(1.25, roundedDownlinkMbps(1240), 0.0001)
    }

    @Test
    fun cyclesPreferenceAndFormatsMessages() {
        assertEquals("on", nextLowBandwidthPreference("auto"))
        assertEquals("off", nextLowBandwidthPreference("on"))
        assertEquals("auto", nextLowBandwidthPreference("off"))
        assertEquals("auto", normalizeLowBandwidthPreference("weird"))
        assertTrue(effectiveLowBandwidth("on", auto = false))
        assertFalse(effectiveLowBandwidth("off", auto = true))
        assertTrue(effectiveLowBandwidth("auto", auto = true))
        assertEquals("低速回線モードをONにしました [手動] (回線:4g)", lowBandwidthToast(true, "on", "回線:4g"))
        assertEquals("低速回線モードをOFFにしました [自動]", lowBandwidthToast(false, "auto", ""))
    }
}
