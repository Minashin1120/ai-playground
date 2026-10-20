package com.minashin1120.aiplayground.data

import android.app.Application
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import androidx.core.content.FileProvider
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment
import org.robolectric.annotation.Config
import org.robolectric.annotation.GraphicsMode
import java.io.File
import java.util.UUID

@RunWith(RobolectricTestRunner::class)
@GraphicsMode(GraphicsMode.Mode.NATIVE)
@Config(sdk = [35])
class ImageSplitCompressionTest {
    private val context: Application get() = RuntimeEnvironment.getApplication()

    @Before fun resetPreferences() {
        // Robolectric gives every test a different cache directory; AndroidX's static
        // provider strategy otherwise keeps a root from a previous sandbox.
        val cache = FileProvider::class.java.getDeclaredField("sCache").apply { isAccessible = true }
        (cache.get(null) as MutableMap<*, *>).clear()
        context.getSharedPreferences("image_split_compression", Context.MODE_PRIVATE).edit().clear().commit()
    }

    @Test fun defaultsAndSavedValuesAreIndependentFromComposerAndTranslation() {
        assertEquals(CompressionSettings(), loadSplitCompression(context))
        val composer = context.getSharedPreferences("navigation", Context.MODE_PRIVATE)
        val translation = context.getSharedPreferences("selection_translation", Context.MODE_PRIVATE)
        saveCompressionSettings(composer, CompressionSettings(maxDimension = 4096))
        saveCompressionSettings(translation, CompressionSettings(maxDimension = 1024))
        val selected = CompressionSettings(enabled = false, maxSizeMB = 2f, maxDimension = 512,
            outputType = "image/webp", formatOnly = true)
        saveSplitCompression(context, selected)
        assertEquals(selected, loadSplitCompression(context))
        assertEquals(4096, compressionSettingsFrom(composer).maxDimension)
        assertEquals(1024, compressionSettingsFrom(translation).maxDimension)
    }

    private fun source(): File {
        val dir = File(context.cacheDir, "shared/split_test/${UUID.randomUUID()}").apply { mkdirs() }
        val file = File(dir, "photo.png")
        val bitmap = Bitmap.createBitmap(100, 68, Bitmap.Config.ARGB_8888)
        bitmap.eraseColor(Color.rgb(18, 171, 52))
        bitmap.setPixel(0, 0, Color.TRANSPARENT)
        file.outputStream().use { bitmap.compress(Bitmap.CompressFormat.PNG, 100, it) }
        bitmap.recycle()
        return file
    }

    @Test fun disabledCompressionPreservesPixelsAndOriginalCropDimensions() {
        val source = source()
        val outputs = renderImageSplit(context, splitShareUri(context, source),
            ImageSplitOptions(pieces = 4, overlapPct = 0, includeOriginal = false, markPieces = false),
            File(source.parentFile, "out"), CompressionSettings(enabled = false))
        assertEquals(4, outputs.size)
        val bitmap = outputs.first().open().use { BitmapFactory.decodeStream(it) }!!
        assertEquals(50, bitmap.width)
        assertEquals(34, bitmap.height)
        assertEquals(Color.TRANSPARENT, bitmap.getPixel(0, 0))
        assertEquals(Color.rgb(18, 171, 52), bitmap.getPixel(1, 0))
        assertTrue(outputs.all { it.mime == "image/png" && it.name.endsWith(".png") })
        bitmap.recycle()
    }

    @Test fun captionDoesNotFlattenTransparencyInTheImage() {
        val source = source()
        val outputs = renderImageSplit(context, splitShareUri(context, source),
            ImageSplitOptions(pieces = 4, overlapPct = 0, includeOriginal = false, markPieces = true),
            File(source.parentFile, "out"), CompressionSettings(enabled = false))
        val bitmap = outputs.first().open().use { BitmapFactory.decodeStream(it) }!!
        val header = bitmap.height - 34
        assertTrue(header >= 12)
        assertEquals(Color.TRANSPARENT, bitmap.getPixel(0, header))
        assertEquals(Color.rgb(18, 171, 52), bitmap.getPixel(1, header))
        bitmap.recycle()
    }

    @Test fun formatOnlyUsesSharedConversionAndKeepsDimensionsAndReadableNames() {
        val source = source()
        val outputs = renderImageSplit(context, splitShareUri(context, source),
            ImageSplitOptions(pieces = 4, overlapPct = 0, includeOriginal = false, markPieces = false),
            File(source.parentFile, "out"), CompressionSettings(outputType = "image/jpeg", formatOnly = true))
        assertEquals(listOf("photo_part1of4_r1c1.jpg", "photo_part2of4_r1c2.jpg",
            "photo_part3of4_r2c1.jpg", "photo_part4of4_r2c2.jpg"), outputs.map { it.name })
        outputs.forEach { output ->
            assertEquals("image/jpeg", output.mime)
            val bitmap = output.open().use { BitmapFactory.decodeStream(it) }!!
            assertEquals(50, bitmap.width)
            assertEquals(34, bitmap.height)
            bitmap.recycle()
        }
    }

    @Test fun resizedPiecesAreEncodedDirectlyWithoutIntermediatePngFiles() {
        val source = source()
        val directory = File(source.parentFile, "out")
        val outputs = renderImageSplit(context, splitShareUri(context, source),
            ImageSplitOptions(pieces = 4, overlapPct = 0, includeOriginal = false, markPieces = false),
            directory, CompressionSettings(maxDimension = 32, outputType = "image/jpeg"))
        assertEquals(4, directory.walkTopDown().count { it.isFile })
        outputs.forEach { output ->
            assertTrue(output.name.endsWith(".jpg"))
            val bitmap = output.open().use { BitmapFactory.decodeStream(it) }!!
            assertEquals(32, bitmap.width)
            assertEquals(21, bitmap.height)
            bitmap.recycle()
        }
    }

    @Test fun formatOnlyOriginalKeepsLosslessFullSizePieces() {
        val source = source()
        val outputs = renderImageSplit(context, splitShareUri(context, source),
            ImageSplitOptions(pieces = 4, overlapPct = 0, includeOriginal = false, markPieces = false),
            File(source.parentFile, "out"), CompressionSettings(maxDimension = 32, formatOnly = true))
        val bitmap = outputs.first().open().use { BitmapFactory.decodeStream(it) }!!
        assertEquals(50, bitmap.width)
        assertEquals(34, bitmap.height)
        assertEquals(Color.TRANSPARENT, bitmap.getPixel(0, 0))
        bitmap.recycle()
    }

    @Test fun sharedSourceRemainsIntactAfterDrawingOverviewAndAllPieces() {
        val source = source()
        val outputs = renderImageSplit(context, splitShareUri(context, source),
            ImageSplitOptions(pieces = 4, overlapPct = 0, includeOriginal = true,
                markOriginal = true, markPieces = false),
            File(source.parentFile, "out"), CompressionSettings(enabled = false))
        assertEquals(5, outputs.size)
        val overview = outputs.first().open().use { BitmapFactory.decodeStream(it) }!!
        assertEquals(100, overview.width)
        assertEquals(68, overview.height)
        overview.recycle()
        outputs.drop(1).forEach { output ->
            val bitmap = output.open().use { BitmapFactory.decodeStream(it) }!!
            assertEquals(50, bitmap.width)
            assertEquals(34, bitmap.height)
            // Overview badges and lines must never be drawn onto the shared source.
            assertEquals(Color.rgb(18, 171, 52), bitmap.getPixel(25, 17))
            bitmap.recycle()
        }
    }

    @Test fun oversizedPngShrinksWithoutChangingItsLosslessFormat() {
        val bitmap = Bitmap.createBitmap(512, 512, Bitmap.Config.ARGB_8888)
        val random = java.util.Random(1234)
        val pixels = IntArray(512 * 512) { Color.rgb(random.nextInt(256), random.nextInt(256), random.nextInt(256)) }
        bitmap.setPixels(pixels, 0, 512, 0, 0, 512, 512)
        val target = File(context.cacheDir, "noise.png")
        writeCompressedBitmap(bitmap, target, Bitmap.CompressFormat.PNG,
            CompressionSettings(maxSizeMB = 0.05f))
        assertTrue(bitmap.isRecycled)
        val result = BitmapFactory.decodeFile(target.path)!!
        // Preserve the existing four-shrink limit even if the byte budget cannot be reached.
        assertEquals(208, result.width)
        assertEquals(208, result.height)
        result.recycle()
    }

}
