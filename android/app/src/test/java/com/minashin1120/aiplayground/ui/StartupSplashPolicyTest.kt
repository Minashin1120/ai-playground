package com.minashin1120.aiplayground.ui

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class StartupSplashPolicyTest {
    @Test
    fun freshLaunchPlaysTheIntro() {
        assertTrue(shouldPlayStartupAnimation(hasSavedState = false, shownInProcess = false))
        assertTrue(shouldPlayStartupAnimation(hasSavedState = false, shownInProcess = true))
    }

    @Test
    fun taskRestoredAfterProcessDeathPlaysTheIntro() {
        // The system delivers saved state for a restored task, but this process has not shown the intro yet.
        assertTrue(shouldPlayStartupAnimation(hasSavedState = true, shownInProcess = false))
    }

    @Test
    fun configurationChangeInTheSameProcessDoesNotReplayTheIntro() {
        assertFalse(shouldPlayStartupAnimation(hasSavedState = true, shownInProcess = true))
    }
}
