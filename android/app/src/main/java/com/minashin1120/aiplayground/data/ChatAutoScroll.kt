package com.minashin1120.aiplayground.data

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue

/**
 * Web `part05` chat auto-scroll (`userAutoScroll`): the conversation follows its bottom while content
 * arrives. Scrolling up pauses that; scrolling back down to within [thresholdPx] of the bottom resumes it.
 * Sending, reconnecting to an answer and the 一番下へ button resume it at once.
 */
class ChatAutoScroll(private val thresholdPx: Float) {
    /** Web `userAutoScroll`. */
    var following by mutableStateOf(true)
        private set
    private var paused = false
    private var resumeArmed = false

    /**
     * A user drag. [towardTop] when the content moves down to show older messages; that pauses following
     * only when the list can actually scroll that way. Dragging back down arms the resume.
     */
    fun onUserScroll(towardTop: Boolean, canScrollBackward: Boolean) {
        if (towardTop) {
            if (!canScrollBackward) return
            paused = true
            resumeArmed = false
            following = false
        } else if (paused) {
            resumeArmed = true
        }
    }

    /** After the list moved, with the distance left to its very bottom. */
    fun onPosition(distanceToBottom: Float) {
        val near = distanceToBottom <= thresholdPx
        if (paused) {
            if (resumeArmed && near) resume() else following = false
        } else if (near) {
            following = true
        }
    }

    /** Web `resumeChatAutoScroll` / `scrollToBottom(true)`. */
    fun resume() {
        paused = false
        resumeArmed = false
        following = true
    }

    /** Web `syncScrollToBottomButton`: shown while not following and away from the bottom. */
    fun showsJumpButton(distanceToBottom: Float): Boolean = !following && distanceToBottom > thresholdPx
}
