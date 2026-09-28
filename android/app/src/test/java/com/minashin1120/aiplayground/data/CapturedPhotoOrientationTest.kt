package com.minashin1120.aiplayground.data

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.exifinterface.media.ExifInterface
import java.io.File
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.rules.TemporaryFolder
import org.robolectric.RobolectricTestRunner

@RunWith(RobolectricTestRunner::class)
class CapturedPhotoOrientationTest {
    @get:Rule val temporaryFolder = TemporaryFolder()

    @Test fun rotatesExifPixelsBeforeUpload() {
        val file = photo(ExifInterface.ORIENTATION_ROTATE_90)

        normalizeCapturedPhotoOrientation(file)

        val image = BitmapFactory.decodeFile(file.path)
        assertEquals(120, image.width)
        assertEquals(80, image.height)
        assertEquals(ExifInterface.ORIENTATION_NORMAL,
            ExifInterface(file).getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL))
        assertBlue(image.getPixel(20, 20)) // Source bottom-left moves to top-left.
        image.recycle()
    }

    @Test fun preservesFrontCameraMirrorDirection() {
        val file = photo(ExifInterface.ORIENTATION_FLIP_HORIZONTAL)

        normalizeCapturedPhotoOrientation(file)

        val image = BitmapFactory.decodeFile(file.path)
        assertEquals(80, image.width)
        assertEquals(120, image.height)
        assertGreen(image.getPixel(20, 20)) // Source top-right moves to top-left.
        image.recycle()
    }

    @Test fun combinesFlipAndRotationFromExif() {
        val file = photo(ExifInterface.ORIENTATION_TRANSPOSE)

        normalizeCapturedPhotoOrientation(file)

        val image = BitmapFactory.decodeFile(file.path)
        assertEquals(120, image.width)
        assertEquals(80, image.height)
        assertRed(image.getPixel(20, 20)) // Transpose keeps source top-left at top-left.
        image.recycle()
    }

    @Test fun leavesAlreadyUprightPhotoUnchanged() {
        val file = photo(ExifInterface.ORIENTATION_NORMAL)
        val original = file.readBytes()

        normalizeCapturedPhotoOrientation(file)

        assertArrayEquals(original, file.readBytes())
    }

    private fun photo(orientation: Int): File {
        val file = temporaryFolder.newFile("camera_${orientation}.jpg")
        val image = Bitmap.createBitmap(80, 120, Bitmap.Config.ARGB_8888)
        for (y in 0 until 120) for (x in 0 until 80) {
            image.setPixel(x, y, when {
                x < 40 && y < 60 -> 0xffff0000.toInt()
                x >= 40 && y < 60 -> 0xff00ff00.toInt()
                x < 40 -> 0xff0000ff.toInt()
                else -> 0xffffff00.toInt()
            })
        }
        file.outputStream().use { assertTrue(image.compress(Bitmap.CompressFormat.JPEG, 100, it)) }
        image.recycle()
        ExifInterface(file).apply {
            setAttribute(ExifInterface.TAG_ORIENTATION, orientation.toString())
            saveAttributes()
        }
        return file
    }

    private fun assertBlue(color: Int) {
        assertTrue(android.graphics.Color.blue(color) > 180)
        assertTrue(android.graphics.Color.red(color) < 80)
    }

    private fun assertGreen(color: Int) {
        assertTrue(android.graphics.Color.green(color) > 180)
        assertTrue(android.graphics.Color.red(color) < 80)
    }

    private fun assertRed(color: Int) {
        assertTrue(android.graphics.Color.red(color) > 180)
        assertTrue(android.graphics.Color.green(color) < 80)
    }
}
