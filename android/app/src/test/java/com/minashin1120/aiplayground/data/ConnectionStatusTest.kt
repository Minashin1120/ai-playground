package com.minashin1120.aiplayground.data

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ConnectionStatusTest {
    @Test fun disconnectedStatesUseTheFastProbeInterval() {
        assertEquals(2_000L, ConnectionStatus.OFFLINE.probeIntervalMillis())
        assertEquals(2_000L, ConnectionStatus.UNSTABLE.probeIntervalMillis())
        assertEquals(5_000L, ConnectionStatus.ONLINE.probeIntervalMillis())
    }

    @Test fun serverResponsesMatchTheWebRetryModes() {
        assertEquals(ConnectionStatus.MAINTENANCE, connectionStatusForHttp(503))
        assertEquals(ConnectionStatus.SERVER_DOWN, connectionStatusForHttp(502))
        assertEquals(ConnectionStatus.SERVER_DOWN, connectionStatusForHttp(524))
        assertEquals(null, connectionStatusForHttp(500))
    }

    @Test fun onlyUnavailableStatesAreDisconnected() {
        assertTrue(ConnectionStatus.SERVER_DOWN.isDisconnected())
        assertFalse(ConnectionStatus.UNKNOWN.isDisconnected())
        assertFalse(ConnectionStatus.ONLINE.isDisconnected())
    }
}
