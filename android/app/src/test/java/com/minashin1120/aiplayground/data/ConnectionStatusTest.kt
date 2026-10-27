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
        assertFalse(ConnectionStatus.UNSTABLE.isDisconnected())
    }

    @Test fun oneProbeFailureDoesNotBlockWorkingRequests() {
        val health = ConnectionProbeHealth()
        assertEquals(null, health.failed(10, usableNetwork = true))
        assertEquals(ConnectionStatus.UNSTABLE, health.failed(20, usableNetwork = true))
    }

    @Test fun successfulTrafficOverridesAConcurrentProbeTimeout() {
        val health = ConnectionProbeHealth()
        health.failed(10, usableNetwork = true)
        health.succeeded(25)
        assertEquals(null, health.failed(20, usableNetwork = true))
        assertEquals(null, health.failed(30, usableNetwork = true))
        assertEquals(ConnectionStatus.UNSTABLE, health.failed(40, usableNetwork = true))
        health.succeeded(50)
        assertEquals(null, health.failed(60, usableNetwork = true))
    }

    @Test fun lostNetworkIsReportedImmediately() {
        val health = ConnectionProbeHealth()
        assertEquals(ConnectionStatus.OFFLINE, health.failed(10, usableNetwork = false))
    }
}
