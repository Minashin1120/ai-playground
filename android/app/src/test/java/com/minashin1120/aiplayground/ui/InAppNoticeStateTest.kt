package com.minashin1120.aiplayground.ui

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Test

class InAppNoticeStateTest {
    @Test
    fun newestGoesAboveOlderMessagesWithoutExtendingTheirDeadlines() {
        var now = 1_000L
        val state = InAppNoticeState { now }
        state.add("保存されました")
        val first = state.notices.single()
        now += 2_000L
        state.add("ファイル名を変更しました")

        assertEquals(listOf("ファイル名を変更しました", "保存されました"), state.notices.map { it.message })
        assertEquals(first.expiresAtMillis, state.notices.last().expiresAtMillis)
        now = 6_000L
        state.expire()
        assertNull(state.notices.first().exitDirection)
        assertEquals(-1, state.notices.last().exitDirection)
        now += PlaygroundMotion.SHORT
        state.expire()
        assertEquals(listOf("ファイル名を変更しました"), state.notices.map { it.message })
        now = 8_000L
        state.expire()
        now += PlaygroundMotion.SHORT
        state.expire()
        assertTrue(state.notices.isEmpty())
    }

    @Test
    fun identicalMessagesRemainIndependentAndOnlyTheSwipedMessageCloses() {
        val state = InAppNoticeState { 0L }
        repeat(2) { state.add("保存されました") }
        val newest = state.notices.first().id
        val older = state.notices.last().id
        assertTrue(newest != older)

        state.dismiss(newest, 1)
        assertEquals(1, state.notices.first().exitDirection)
        assertNull(state.notices.last().exitDirection)
        state.finishExit(newest)
        assertEquals(older, state.notices.single().id)
        state.finishExit(older)
        assertEquals(1, state.notices.size)
    }

    @Test
    fun expiredMessageWaitsForAnActiveDragWithoutExtendingItsDeadline() {
        var now = 0L
        val state = InAppNoticeState { now }
        state.add("通知")
        val notice = state.notices.single()
        state.setDragging(notice.id, true)
        assertNull(state.nextDeadline)
        now = 7_000L
        state.expire()
        assertNull(state.notices.single().exitDirection)

        state.setDragging(notice.id, false)
        state.expire()
        assertEquals(-1, state.notices.single().exitDirection)
        assertEquals(notice.expiresAtMillis, state.notices.single().expiresAtMillis)
    }

    @Test
    fun hiddenMessagesAreRemovedEvenWithoutComposedExitAnimations() {
        var now = 0L
        val state = InAppNoticeState { now }
        repeat(50) { state.add("通知 $it") }
        now = NOTICE_DURATION_MILLIS
        state.expire()
        assertTrue(state.notices.all { it.exitDirection == -1 })
        now += PlaygroundMotion.SHORT
        state.expire()
        assertTrue(state.notices.isEmpty())
        assertNull(state.nextDeadline)
    }

    @Test
    fun aNewMessageDuringExitSurvivesTheOlderMessagesRemoval() {
        var now = 0L
        val state = InAppNoticeState { now }
        state.add("閉じる通知")
        val closingId = state.notices.single().id
        state.dismiss(closingId)
        now = 80L
        state.add("新着")
        val incomingId = state.notices.first().id
        now = PlaygroundMotion.SHORT.toLong()
        state.expire()
        assertEquals(incomingId, state.notices.single().id)
        assertNull(state.notices.single().exitDirection)
        assertEquals(5_080L, state.notices.single().expiresAtMillis)
        state.finishExit(closingId)
        assertEquals(1, state.notices.size)
    }

    @Test
    fun nestedWindowsHandBackTheSameMessagesAndDeadlines() {
        val state = InAppNoticeState { 1_000L }
        val main = Any()
        val settings = Any()
        val confirmation = Any()
        state.add("通知")
        val original = state.notices.single()
        state.registerWindow(main)
        state.registerWindow(settings)
        state.registerWindow(confirmation)
        assertSame(confirmation, state.activeWindow)
        state.unregisterWindow(confirmation)
        assertSame(settings, state.activeWindow)
        state.focusWindow(main)
        assertSame(main, state.activeWindow)
        state.unregisterWindow(settings)
        assertSame(main, state.activeWindow)
        assertEquals(original, state.notices.single())
        state.unregisterWindow(main)
        assertNull(state.activeWindow)
    }

    @Test
    fun restoreKeepsOrderAndRemainingTimeAndDropsDismissedOrExpiredMessages() {
        var now = 0L
        val state = InAppNoticeState { now }
        state.add("期限切れ")
        now = 2_000L
        state.add("残す通知")
        state.add("閉じた通知")
        state.dismiss(state.notices.first().id)
        val saved = state.save()
        now = 5_500L
        val restored = InAppNoticeState.restore(saved) { now }
        assertEquals("残す通知", restored.notices.single().message)
        assertEquals(7_000L, restored.notices.single().expiresAtMillis)
        assertTrue(restored.notices.single().entered)
        assertFalse(restored.notices.single().dragging)
        val restoredId = restored.notices.single().id
        restored.add("新着")
        assertTrue(restored.notices.first().id > restoredId)
    }

    @Test
    fun accessibilityCanKeepAMessageUntilExplicitDismissalWithoutOverflow() {
        val state = InAppNoticeState { 50_000L }
        state.add("通知", Long.MAX_VALUE)
        state.expire()
        assertEquals(Long.MAX_VALUE, state.notices.single().expiresAtMillis)
        assertNull(state.notices.single().exitDirection)
    }

    @Test
    fun verticalSwipesRespectDistanceVelocityAndDensity() {
        assertNull(noticeSwipeDirection(20f, 100f, 1f))
        assertEquals(-1, noticeSwipeDirection(-45f, 0f, 1f))
        assertEquals(1, noticeSwipeDirection(45f, 0f, 1f))
        assertEquals(-1, noticeSwipeDirection(-10f, -900f, 1f))
        assertEquals(1, noticeSwipeDirection(10f, 900f, 1f))
        assertNull(noticeSwipeDirection(45f, 900f, 2f))
        assertEquals(1, noticeSwipeDirection(90f, 0f, 2f))
    }
}
