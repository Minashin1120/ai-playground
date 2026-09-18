package com.minashin1120.aiplayground.ui

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assert.assertEquals
import org.junit.Test
import androidx.compose.ui.unit.dp

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

    @Test
    fun layoutPolicyMatchesWebBreakpointAndPhoneOrientation() {
        assertEquals(PlaygroundLayoutClass.Phone, playgroundLayoutClass(412.dp, 915.dp))
        assertEquals(PlaygroundLayoutClass.LandscapePhone, playgroundLayoutClass(700.dp, 412.dp))
        assertEquals(PlaygroundLayoutClass.Tablet, playgroundLayoutClass(768.dp, 1024.dp))
    }
}
