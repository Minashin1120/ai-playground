package com.minashin1120.aiplayground.ui

import androidx.compose.animation.EnterTransition
import androidx.compose.animation.ExitTransition
import androidx.compose.animation.core.SnapSpec
import androidx.compose.animation.core.TweenSpec
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotSame
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Test

class PlaygroundMotionTest {
    @Test
    fun reducedMotionCollapsesEveryTransition() {
        assertSame(EnterTransition.None, expandFadeIn(reduce = true))
        assertSame(ExitTransition.None, shrinkFadeOut(reduce = true))
        assertSame(EnterTransition.None, popIn(reduce = true))
        assertSame(ExitTransition.None, popOut(reduce = true))
        assertSame(EnterTransition.None, modalEnter(fullScreen = true, reduce = true))
        assertSame(EnterTransition.None, modalEnter(fullScreen = false, reduce = true))
        assertSame(ExitTransition.None, modalExit(fullScreen = true, reduce = true))
        assertSame(ExitTransition.None, modalExit(fullScreen = false, reduce = true))
    }

    @Test
    fun normalMotionAnimates() {
        assertNotSame(EnterTransition.None, expandFadeIn(reduce = false))
        assertNotSame(ExitTransition.None, shrinkFadeOut(reduce = false))
        assertNotSame(EnterTransition.None, modalEnter(fullScreen = true, reduce = false))
        assertNotSame(ExitTransition.None, modalExit(fullScreen = false, reduce = false))
    }

    @Test
    fun motionTweenSnapsWhenReduced() {
        assertTrue(motionTween<Float>(reduce = true) is SnapSpec<*>)
        val spec = motionTween<Float>(reduce = false)
        assertTrue(spec is TweenSpec<*>)
        assertEquals(PlaygroundMotion.MEDIUM, (spec as TweenSpec<*>).durationMillis)
    }

    @Test
    fun predictiveBackScaleIsClamped() {
        assertEquals(1f, predictiveBackScale(0f), 0.0001f)
        assertEquals(0.92f, predictiveBackScale(1f), 0.0001f)
        assertEquals(0.92f, predictiveBackScale(3f), 0.0001f)
        assertEquals(1f, predictiveBackScale(-1f), 0.0001f)
    }
}
