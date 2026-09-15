package com.minashin1120.aiplayground.data

import android.content.Context
import android.content.SharedPreferences
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import java.io.File
import java.io.FileOutputStream

/** Mirrors the Web image-compression modal defaults (1 MiB / 1920 px). */
data class CompressionSettings(
    val enabled: Boolean = true,
    val maxSizeMB: Float = 1.0f,
    val maxDimension: Int = 1920,
    val outputType: String = "original",
    val formatOnly: Boolean = false,
) {
    val maxSizeBytes: Long get() = (maxSizeMB * 1024f * 1024f).toLong().coerceAtLeast(64L * 1024L)
}

data class CompressedImage(val file: File, val name: String, val mime: String)

fun compressionSettingsFrom(prefs: SharedPreferences): CompressionSettings = CompressionSettings(
    enabled = prefs.getBoolean("compression_enabled", true),
    maxSizeMB = prefs.getFloat("compression_max_size_mb", 1.0f).coerceIn(0.05f, 50f),
    maxDimension = prefs.getInt("compression_max_dim", 1920).coerceIn(256, 8192),
    outputType = prefs.getString("compression_output_type", "original") ?: "original",
    formatOnly = prefs.getBoolean("compression_format_only", false),
)

/**
 * Resizes and re-encodes a picked image so its dimensions and byte size fit the
 * user's settings. Returns null for non-images, animated GIFs, or files that
 * need no work, so the original upload path is used unchanged.
 */
@Suppress("DEPRECATION")
fun compressImage(context: Context, uri: Uri, name: String, mime: String, settings: CompressionSettings): CompressedImage? {
    if (!settings.enabled || !mime.startsWith("image/") || mime == "image/gif") return null
    return try {
        val resolver = context.contentResolver
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        resolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, bounds) }
        val sourceWidth = bounds.outWidth
        val sourceHeight = bounds.outHeight
        if (sourceWidth <= 0 || sourceHeight <= 0) return null
        // Format-only with "original" means the Web keeps the file untouched.
        if (settings.formatOnly && settings.outputType == "original") return null

        val (format, extension) = resolveOutputType(settings.outputType, mime)
        val sample = if (settings.formatOnly) 1 else sampleSize(sourceWidth, sourceHeight, settings.maxDimension)
        var bitmap = resolver.openInputStream(uri)?.use {
            BitmapFactory.decodeStream(it, null, BitmapFactory.Options().apply { inSampleSize = sample })
        } ?: return null

        if (!settings.formatOnly) {
            val longest = maxOf(bitmap.width, bitmap.height)
            if (longest > settings.maxDimension) {
                val scale = settings.maxDimension.toFloat() / longest
                bitmap = scaleBitmap(bitmap, (bitmap.width * scale).toInt(), (bitmap.height * scale).toInt())
            }
        }

        val directory = File(context.cacheDir, "compressed").apply { mkdirs() }
        val stem = name.substringBeforeLast('.').ifBlank { "image" }
            .filter { it.isLetterOrDigit() || it == '-' || it == '_' }.take(32).ifBlank { "image" }
        val target = File(directory, "$stem.$extension")

        var current = bitmap
        var quality = if (settings.formatOnly) 100 else 90
        var shrinkAttempts = 0
        while (true) {
            FileOutputStream(target).use { output -> current.compress(format, quality, output) }
            if (settings.formatOnly || target.length() <= settings.maxSizeBytes) break
            if (quality > 35) { quality -= 15; continue }
            if (shrinkAttempts >= 4) break
            shrinkAttempts++
            current = scaleBitmap(current, (current.width * 0.8).toInt(), (current.height * 0.8).toInt())
            quality = 80
        }
        if (current !== bitmap) current.recycle()
        bitmap.recycle()
        if (target.length() <= 0) null else CompressedImage(target, "$stem.$extension", mimeForExtension(extension))
    } catch (e: Exception) {
        null
    }
}

private fun sampleSize(width: Int, height: Int, maxDimension: Int): Int {
    if (maxDimension <= 0) return 1
    var sample = 1
    while (width / (sample * 2) >= maxDimension || height / (sample * 2) >= maxDimension) sample *= 2
    return sample
}

private fun scaleBitmap(source: Bitmap, width: Int, height: Int): Bitmap {
    val safeWidth = width.coerceAtLeast(1)
    val safeHeight = height.coerceAtLeast(1)
    if (safeWidth == source.width && safeHeight == source.height) return source
    val scaled = Bitmap.createScaledBitmap(source, safeWidth, safeHeight, true)
    if (scaled !== source) source.recycle()
    return scaled
}

private fun resolveOutputType(outputType: String, sourceMime: String): Pair<Bitmap.CompressFormat, String> = when {
    outputType.contains("png", ignoreCase = true) -> Bitmap.CompressFormat.PNG to "png"
    outputType.contains("webp", ignoreCase = true) -> Bitmap.CompressFormat.WEBP to "webp"
    outputType.contains("jpeg", ignoreCase = true) || outputType.contains("jpg", ignoreCase = true) ->
        Bitmap.CompressFormat.JPEG to "jpg"
    sourceMime.contains("png", ignoreCase = true) -> Bitmap.CompressFormat.PNG to "png"
    sourceMime.contains("webp", ignoreCase = true) -> Bitmap.CompressFormat.WEBP to "webp"
    else -> Bitmap.CompressFormat.JPEG to "jpg"
}

private fun mimeForExtension(extension: String): String = when (extension.lowercase()) {
    "png" -> "image/png"
    "webp" -> "image/webp"
    else -> "image/jpeg"
}
