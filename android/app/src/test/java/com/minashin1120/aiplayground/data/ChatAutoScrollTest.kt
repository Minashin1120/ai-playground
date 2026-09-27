package com.minashin1120.aiplayground.data

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ChatAutoScrollTest {
    @Test fun scrollingUpPausesUntilTheUserComesBackNearTheBottom() {
        val follow = ChatAutoScroll(thresholdPx = 64f)
        assertTrue(follow.following)
        follow.onUserScroll(towardTop = true, canScrollBackward = true)
        assertFalse(follow.following)
        // Near the bottom without scrolling down again (e.g. content shrank) stays paused, like Web.
        follow.onPosition(10f)
        assertFalse(follow.following)
        follow.onUserScroll(towardTop = false, canScrollBackward = true)
        follow.onPosition(200f)
        assertFalse(follow.following)
        assertTrue(follow.showsJumpButton(200f))
        follow.onPosition(40f)
        assertTrue(follow.following)
        assertFalse(follow.showsJumpButton(40f))
    }

    @Test fun contentGrowthWhileFollowingKeepsFollowing() {
        val follow = ChatAutoScroll(thresholdPx = 64f)
        follow.onPosition(500f)
        assertTrue(follow.following)
        assertFalse(follow.showsJumpButton(500f))
    }

    @Test fun anUpwardDragAtTheTopDoesNotPause() {
        val follow = ChatAutoScroll(thresholdPx = 64f)
        follow.onUserScroll(towardTop = true, canScrollBackward = false)
        assertTrue(follow.following)
    }

    @Test fun sendingOrTheJumpButtonResumes() {
        val follow = ChatAutoScroll(thresholdPx = 64f)
        follow.onUserScroll(towardTop = true, canScrollBackward = true)
        follow.resume()
        assertTrue(follow.following)
        follow.onPosition(300f)
        assertTrue(follow.following)
    }
}
