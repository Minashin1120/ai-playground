package com.minashin1120.aiplayground.ui

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class DrawerStartStateTest {
    @Test
    fun startingAlwaysShowsSpinnerCover() {
        assertTrue(shouldCoverPhoneHistoryUntilClosed(starting = true, showThreads = false, wideLayout = false, drawerSettledClosed = false))
        assertTrue(shouldCoverPhoneHistoryUntilClosed(starting = true, showThreads = true, wideLayout = false, drawerSettledClosed = true))
    }

    @Test
    fun phoneKeepsSpinnerUntilDrawerHasSettledClosed() {
        assertTrue(shouldCoverPhoneHistoryUntilClosed(starting = false, showThreads = true, wideLayout = false, drawerSettledClosed = false))
        assertFalse(shouldCoverPhoneHistoryUntilClosed(starting = false, showThreads = true, wideLayout = false, drawerSettledClosed = true))
    }

    @Test
    fun tabletDoesNotCoverPermanentSidePaneAfterStart() {
        assertFalse(shouldCoverPhoneHistoryUntilClosed(starting = false, showThreads = true, wideLayout = true, drawerSettledClosed = false))
    }

    @Test
    fun pairingScreenIsNotCoveredAfterStart() {
        assertFalse(shouldCoverPhoneHistoryUntilClosed(starting = false, showThreads = false, wideLayout = false, drawerSettledClosed = false))
    }
}
