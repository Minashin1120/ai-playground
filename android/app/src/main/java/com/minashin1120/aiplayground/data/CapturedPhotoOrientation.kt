package com.minashin1120.aiplayground.data

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import androidx.exifinterface.media.ExifInterface
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

private const val CAPTURED_PHOTO_MAX_PIXELS = 25_000_000L
private const val CAPTURED_PHOTO_MAX_SAMPLE = 16
private const val CAPTURED_PHOTO_JPEG_QUALITY = 98

/** CameraX stores rotation in JPEG EXIF. Re-encoding for upload discards that tag. */
internal fun normalizeCapturedPhotoOrientation(file: File) {
    val exif = ExifInterface(file)
    val rotation = exif.rotationDegrees
    val flipped = exif.isFlipped
    if (rotation == 0 && !flipped) return

    val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
    BitmapFactory.decodeFile(file.path, bounds)
    if (bounds.outWidth <= 0 || bounds.outHeight <= 0) throw IOException("Cannot decode captured photo")
    // Keep every pixel of ordinary camera output; shrink only beyond the pixel budget or when memory runs out.
    var sample = 1
    while (bounds.outWidth.toLong() * bounds.outHeight / (sample.toLong() * sample) > CAPTURED_PHOTO_MAX_PIXELS) sample *= 2
    val matrix = Matrix().apply {
        if (flipped) postScale(-1f, 1f)
        if (rotation != 0) postRotate(rotation.toFloat())
    }
    while (true) {
        try {
            writeOriented(file, sample, matrix)
            return
        } catch (error: OutOfMemoryError) {
            if (sample >= CAPTURED_PHOTO_MAX_SAMPLE) throw IOException("Cannot decode captured photo")
            sample *= 2
        }
    }
}

private fun writeOriented(file: File, sample: Int, matrix: Matrix) {
    val source = BitmapFactory.decodeFile(file.path, BitmapFactory.Options().apply { inSampleSize = sample })
        ?: throw IOException("Cannot decode captured photo")
    val target = File(file.parentFile, "${file.name}.rotated")
    try {
        val oriented = Bitmap.createBitmap(source, 0, 0, source.width, source.height, matrix, true)
        try {
            FileOutputStream(target).use { output ->
                if (!oriented.compress(Bitmap.CompressFormat.JPEG, CAPTURED_PHOTO_JPEG_QUALITY, output)) throw IOException("Cannot save captured photo")
            }
        } finally {
            if (oriented !== source) oriented.recycle()
        }
        if (!target.renameTo(file)) throw IOException("Cannot replace captured photo")
    } finally {
        source.recycle()
        target.delete()
    }
}
