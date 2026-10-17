package com.minashin1120.aiplayground.ui

/**
 * Rotation and other configuration changes recreate the activity with saved state in the same process; replaying the
 * intro there would make the screen flicker. A task that the system restored after killing the process is a cold start,
 * however, so it must play the intro even though the activity also receives saved state.
 */
internal fun shouldPlayStartupAnimation(hasSavedState: Boolean, shownInProcess: Boolean): Boolean =
    !hasSavedState || !shownInProcess
