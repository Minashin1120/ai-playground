package com.minashin1120.aiplayground.data

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class BotTelemetryTest {
    @Test fun ordinaryTappingIsNeverReported() {
        val telemetry = BotTelemetry()
        var now = 1_000L
        repeat(10) {
            assertFalse(telemetry.recordTap(now))
            assertNull(telemetry.takeReport(now, force = false))
            now += 900
        }
    }

    @Test fun scrollingAloneIsNeverReported() {
        val telemetry = BotTelemetry()
        var now = 1_000L
        repeat(200) {
            telemetry.recordMove(now)
            assertNull(telemetry.takeReport(now, force = false))
            now += 16
        }
    }

    @Test fun rapidTapsForceAReportInTheWebPayloadShape() {
        val telemetry = BotTelemetry()
        var force = false
        for (i in 0 until 5) force = telemetry.recordTap(1_000L + i * 50L)
        assertTrue(force)
        val payload = telemetry.takeReport(1_200L, force = true)!!
        assertEquals(5, payload.getInt("clicks"))
        assertEquals(4, payload.getInt("fast_clicks"))
        assertEquals(5, payload.getInt("click_burst"))
        assertEquals(0, payload.getInt("keys"))
        assertEquals(0.0, payload.getDouble("pointer_speed_max"), 0.0)
        // The window starts over after a report.
        assertEquals(0, telemetry.stats(1_300L).getInt("clicks"))
    }

    @Test fun theFirstInputIsNotReportedAsAHighEventRate() {
        val telemetry = BotTelemetry()
        telemetry.recordTap(1_000L)
        assertNull(telemetry.takeReport(1_000L, force = false))
    }

    @Test fun checksAreSpacedUnlessForced() {
        val telemetry = BotTelemetry()
        for (i in 0 until 9) telemetry.recordTap(1_000L + i * 200L)
        // Within 3 s of the first input: not checked yet (the window keeps counting).
        assertNull(telemetry.takeReport(2_700L, force = false))
        assertNotNull(telemetry.takeReport(4_100L, force = false))
        for (i in 0 until 9) telemetry.recordTap(4_200L + i * 200L)
        assertNull(telemetry.takeReport(6_000L, force = false))
        assertNotNull(telemetry.takeReport(7_200L, force = false))
    }

    @Test fun eightSendsWithinThreeSecondsReachTheLockThreshold() {
        val counter = SendSpamCounter()
        val counts = (0 until 8).map { counter.register(10_000L + it * 300L) }
        assertEquals(SendSpamCounter.LOCK_THRESHOLD, counts.last())
        assertEquals(1, counter.register(20_000L))
        counter.reset()
        assertEquals(1, counter.register(20_100L))
    }
}
