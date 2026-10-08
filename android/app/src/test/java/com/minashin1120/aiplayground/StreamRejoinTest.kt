package com.minashin1120.aiplayground

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class StreamRejoinTest {
    private val after = 30_000L

    @Test fun aShortAbsenceNeverRejoins() {
        assertFalse(shouldRejoinStream(awayMs = 5_000, keptRunning = false, quietMs = 60_000, afterMs = after))
    }

    @Test fun anAnswerStillArrivingWhileTheAppKeptRunningIsReadOn() {
        assertFalse(shouldRejoinStream(awayMs = 120_000, keptRunning = true, quietMs = 1_000, afterMs = after))
    }

    @Test fun aStoppedServiceOrAQuietStreamIsRejoined() {
        assertTrue(shouldRejoinStream(awayMs = 120_000, keptRunning = false, quietMs = 1_000, afterMs = after))
        assertTrue(shouldRejoinStream(awayMs = 120_000, keptRunning = true, quietMs = 45_000, afterMs = after))
    }
}
