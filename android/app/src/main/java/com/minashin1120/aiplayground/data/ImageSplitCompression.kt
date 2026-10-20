package com.minashin1120.aiplayground.data

import android.content.Context
import java.io.File

/** The composer's compression schema and defaults, saved independently for image splitting. */
private const val SPLIT_COMPRESSION_PREFS = "image_split_compression"

internal fun loadSplitCompression(context: Context): CompressionSettings =
    compressionSettingsFrom(context.getSharedPreferences(SPLIT_COMPRESSION_PREFS, Context.MODE_PRIVATE))

internal fun saveSplitCompression(context: Context, settings: CompressionSettings) {
    saveCompressionSettings(context.getSharedPreferences(SPLIT_COMPRESSION_PREFS, Context.MODE_PRIVATE), settings)
}

/** Compresses the unchanged original; rendered pieces use direct bitmap encoding instead. */
internal fun compressSplitOutput(context: Context, output: SplitOutput, settings: CompressionSettings, directory: File): SplitOutput {
    val compressed = compressImage(context, output.uploadUri, output.name, output.mime, settings) ?: return output
    directory.mkdirs()
    val name = output.name.substringBeforeLast('.') + "." + compressed.name.substringAfterLast('.')
    val file = File(directory, name)
    compressed.file.copyTo(file, overwrite = true)
    return SplitOutput(name, compressed.mime, splitShareUri(context, file)) { file.inputStream() }
}
