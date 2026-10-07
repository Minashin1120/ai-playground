package com.minashin1120.aiplayground.ui

import org.junit.Assert.assertEquals
import org.junit.Test

class InAppCameraZoomTest {
    @Test
    fun presetsIncludeUltraWideWhenAvailable() {
        assertEquals(listOf(0.6f, 1f, 2f, 5f), cameraZoomPresets(0.6f, 10f))
    }

    @Test
    fun presetsStayWithinRange() {
        assertEquals(listOf(1f, 2f), cameraZoomPresets(1f, 4f))
        assertEquals(listOf(1f), cameraZoomPresets(1f, 1f))
    }

    @Test
    fun activePresetFollowsCurrentZoom() {
        val presets = listOf(0.6f, 1f, 2f, 5f)
        assertEquals(0.6f, activeCameraZoomPreset(presets, 0.8f), 0.001f)
        assertEquals(1f, activeCameraZoomPreset(presets, 1f), 0.001f)
        assertEquals(2f, activeCameraZoomPreset(presets, 3.4f), 0.001f)
        assertEquals(5f, activeCameraZoomPreset(presets, 8f), 0.001f)
    }

    @Test
    fun ratioIsFormattedLikeSystemCamera() {
        assertEquals("1×", formatCameraZoomRatio(1f))
        assertEquals("0.6×", formatCameraZoomRatio(0.6f))
        assertEquals("2.4×", formatCameraZoomRatio(2.43f))
        assertEquals("5×", formatCameraZoomRatio(4.98f))
    }
}
