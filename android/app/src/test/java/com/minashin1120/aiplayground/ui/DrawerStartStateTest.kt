package com.minashin1120.aiplayground.ui

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class DrawerStartStateTest {
    @Test
    fun phoneWithAccountClosesHistoryDrawer() {
        assertTrue(shouldForceHistoryDrawerClosed(showThreads = true, wideLayout = false))
    }

    @Test
    fun tabletKeepsPermanentSidePane() {
        assertFalse(shouldForceHistoryDrawerClosed(showThreads = true, wideLayout = true))
    }

    @Test
    fun pairingScreenDoesNotForceDrawer() {
        assertFalse(shouldForceHistoryDrawerClosed(showThreads = false, wideLayout = false))
        assertFalse(shouldForceHistoryDrawerClosed(showThreads = false, wideLayout = true))
    }
}
