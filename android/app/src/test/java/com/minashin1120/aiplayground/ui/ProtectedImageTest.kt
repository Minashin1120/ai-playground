package com.minashin1120.aiplayground.ui

import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import org.junit.Assert.assertEquals
import org.junit.Test

class ProtectedImageTest {
    @Test fun wideImagesFillTheWidth() {
        val (width, height) = fittedImageSize(1600, 800, 320.dp, Dp.Infinity)
        assertEquals(320f, width.value, 0.01f)
        assertEquals(160f, height.value, 0.01f)
    }

    @Test fun tallImagesKeepTheWidthWithoutAHeightCap() {
        val (width, height) = fittedImageSize(900, 1600, 320.dp, Dp.Infinity)
        assertEquals(320f, width.value, 0.01f)
        assertEquals(568.89f, height.value, 0.01f)
    }

    @Test fun heightCapShrinksTheFrameToTheImage() {
        val (width, height) = fittedImageSize(900, 1600, 400.dp, 400.dp)
        assertEquals(225f, width.value, 0.01f)
        assertEquals(400f, height.value, 0.01f)
    }
}
