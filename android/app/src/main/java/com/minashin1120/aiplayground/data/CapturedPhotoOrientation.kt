package com.minashin1120.aiplayground.data

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import androidx.exifinterface.media.ExifInterface
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

/** CameraX stores rotation in JPEG EXIF. Re-encoding for upload discards that tag. */
internal fun normalizeCapturedPhotoOrientation(file: File) {
    val exif = ExifInterface(file)
    val rotation = exif.rotationDegrees
    val flipped = exif.isFlipped
    if (rotation == 0 && !flipped) return

    val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
    BitmapFactory.decodeFile(file.path, bounds)
    if (bounds.outWidth <= 0 || bounds.outHeight <= 0) throw IOException("Cannot decode captured photo")
    // Limit the temporary bitmap for high-resolution camera sensors.
    var sample = 1
    while (bounds.outWidth.toLong() * bounds.outHeight / (sample.toLong() * sample) > 12_000_000L) sample *= 2
    val source = BitmapFactory.decodeFile(file.path, BitmapFactory.Options().apply { inSampleSize = sample })
        ?: throw IOException("Cannot decode captured photo")
    val matrix = Matrix().apply {
        if (flipped) postScale(-1f, 1f)
        if (rotation != 0) postRotate(rotation.toFloat())
    }
    val target = File(file.parentFile, "${file.name}.rotated")
    try {
        val oriented = Bitmap.createBitmap(source, 0, 0, source.width, source.height, matrix, true)
        try {
            FileOutputStream(target).use { output ->
                if (!oriented.compress(Bitmap.CompressFormat.JPEG, 95, output)) throw IOException("Cannot save captured photo")
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
