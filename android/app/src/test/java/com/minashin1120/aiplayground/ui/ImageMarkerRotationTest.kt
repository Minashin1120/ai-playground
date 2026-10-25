package com.minashin1120.aiplayground.ui

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import com.minashin1120.aiplayground.data.MarkerPoint
import com.minashin1120.aiplayground.data.MarkerRect
import com.minashin1120.aiplayground.data.rotateMarkerPoint
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import org.robolectric.annotation.GraphicsMode

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34], manifest = Config.NONE)
@GraphicsMode(GraphicsMode.Mode.NATIVE)
class ImageMarkerRotationTest {
    private fun source(): Bitmap = Bitmap.createBitmap(4, 2, Bitmap.Config.ARGB_8888).apply {
        eraseColor(Color.WHITE)
        setPixel(0, 0, Color.RED)
        setPixel(3, 0, Color.GREEN)
        setPixel(0, 1, Color.BLUE)
        setPixel(3, 1, Color.YELLOW)
    }

    private fun render(source: Bitmap, ops: List<MarkerOp>, crop: MarkerRect? = null): Bitmap {
        val bytes = renderEdited(source, ops, crop, source.width to source.height)
        return requireNotNull(BitmapFactory.decodeByteArray(bytes, 0, bytes.size))
    }

    @Test fun rightRotationSavesPixelsWithSwappedDimensions() {
        val source = source()
        val output = render(source, listOf(MarkerOp.RotateOp(1)))
        assertEquals(2, output.width)
        assertEquals(4, output.height)
        assertEquals(Color.BLUE, output.getPixel(0, 0))
        assertEquals(Color.RED, output.getPixel(1, 0))
        assertEquals(Color.YELLOW, output.getPixel(0, 3))
        assertEquals(Color.GREEN, output.getPixel(1, 3))
        assertFalse(source.isRecycled)
        assertEquals(Color.RED, source.getPixel(0, 0))
        output.recycle()
        source.recycle()
    }

    @Test fun cropAndAnnotationsRotateTogetherAndClearPreservesOrientation() {
        val source = source()
        val paint = MarkerOp.StrokeOp(listOf(MarkerPoint(0f, 0f), MarkerPoint(4f, 0f)), 2f,
            androidx.compose.ui.graphics.Color.Magenta)
        val output = render(source, listOf(paint, MarkerOp.RotateOp(-1)), MarkerRect(0f, 0f, 2f, 2f))
        assertEquals(2, output.width)
        assertEquals(2, output.height)
        assertEquals(Color.MAGENTA, output.getPixel(0, 0))
        val cleared = render(source, listOf(paint, MarkerOp.RotateOp(-1), MarkerOp.ClearOp))
        assertEquals(2, cleared.width)
        assertEquals(4, cleared.height)
        assertEquals(Color.GREEN, cleared.getPixel(0, 0))
        assertEquals(Color.YELLOW, cleared.getPixel(1, 0))
        output.recycle()
        cleared.recycle()
        source.recycle()
    }

    @Test fun undoingRotationRestoresSavedOrientation() {
        val source = source()
        val ops = mutableListOf<MarkerOp>(MarkerOp.RotateOp(1), MarkerOp.RotateOp(1))
        ops.removeAt(ops.lastIndex)
        val once = render(source, ops)
        assertEquals(Color.RED, once.getPixel(1, 0))
        ops.removeAt(ops.lastIndex)
        val original = render(source, ops)
        assertEquals(4, original.width)
        assertEquals(Color.RED, original.getPixel(0, 0))
        once.recycle()
        original.recycle()
        source.recycle()
    }

    @Test fun rotatedTouchCoordinatesMapBackToTheSameEditPoint() {
        val point = MarkerPoint(23f, 41f)
        for (turns in -4..4) {
            val rotated = rotateMarkerPoint(point, 120f, 80f, turns)
            val odd = turns % 2 != 0
            val recovered = rotateMarkerPoint(rotated, if (odd) 80f else 120f, if (odd) 120f else 80f, -turns)
            assertEquals(point.x, recovered.x, 0.001f)
            assertEquals(point.y, recovered.y, 0.001f)
        }
        assertEquals(MarkerPoint(80f, 0f), rotateMarkerPoint(MarkerPoint(0f, 0f), 120f, 80f, 1))
        assertEquals(MarkerPoint(0f, 120f), rotateMarkerPoint(MarkerPoint(0f, 0f), 120f, 80f, -1))
    }
}
