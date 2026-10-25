package com.minashin1120.aiplayground.data.local

import android.graphics.BitmapFactory
import android.graphics.pdf.PdfRenderer
import android.media.MediaDataSource
import android.media.MediaMetadataRetriever
import android.os.ParcelFileDescriptor
import com.minashin1120.aiplayground.data.direct.AttachmentExtractor
import java.io.File
import kotlin.math.ceil

internal data class AttachmentTokenEstimate(val tokens: Int = 0, val reason: String = "ok")

/** Mirrors server/token_utils.py media estimates; these are estimates, not provider usage. */
internal object AttachmentTokenEstimator {
    internal fun mediaTokens(model: String, mime: String, width: Int = 0, height: Int = 0,
        pages: Int = 0, seconds: Double = 0.0, hasAudio: Boolean = true): Int? {
        if (!model.contains("gemini", ignoreCase = true)) return null
        val major = Regex("gemini-(\\d+)(?:[.-]|$)", RegexOption.IGNORE_CASE).find(model)?.groupValues?.get(1)?.toIntOrNull()
        val legacy = major != null && major < 3
        return when {
            mime.startsWith("image/") -> if (!legacy) 1120 else if (width <= 384 && height <= 384) 258
                else 258 * ceil(width / 768.0).toInt() * ceil(height / 768.0).toInt()
            mime == "application/pdf" -> pages * if (legacy) 258 else 560
            mime.startsWith("audio/") -> ceil(seconds * if (legacy) 32 else 25).toInt()
            mime.startsWith("video/") -> if (legacy) ceil(seconds * 263).toInt()
                else maxOf(1, ceil(seconds).toInt()) * 70 + if (hasAudio) ceil(seconds * 25).toInt() else 0
            else -> null
        }
    }

    fun estimate(model: String, name: String, mime: String, bytes: ByteArray, temporaryFile: () -> File): AttachmentTokenEstimate {
        val attachment = AttachmentExtractor.prepare(name, mime, bytes)
        val type = attachment.mime
        if (model.contains("gemini", ignoreCase = true) && attachment.bytes != null) {
            val tokens = when {
                type.startsWith("image/") -> {
                    val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
                    BitmapFactory.decodeByteArray(bytes, 0, bytes.size, bounds)
                    mediaTokens(model, type, maxOf(0, bounds.outWidth), maxOf(0, bounds.outHeight))
                }
                type == "application/pdf" -> {
                    val file = temporaryFile()
                    try {
                        file.writeBytes(bytes)
                        ParcelFileDescriptor.open(file, ParcelFileDescriptor.MODE_READ_ONLY).use { fd ->
                            PdfRenderer(fd).use { pdf -> mediaTokens(model, type, pages = pdf.pageCount) }
                        }
                    } finally { file.delete() }
                }
                type.startsWith("audio/") || type.startsWith("video/") -> {
                    val source = object : MediaDataSource() {
                        override fun getSize(): Long = bytes.size.toLong()
                        override fun close() = Unit
                        override fun readAt(position: Long, buffer: ByteArray, offset: Int, size: Int): Int {
                            if (position < 0 || position >= bytes.size) return -1
                            val count = minOf(size, bytes.size - position.toInt())
                            bytes.copyInto(buffer, offset, position.toInt(), position.toInt() + count)
                            return count
                        }
                    }
                    val retriever = MediaMetadataRetriever()
                    try {
                        retriever.setDataSource(source)
                        val duration = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toDoubleOrNull()
                            ?: return AttachmentTokenEstimate(reason = "error")
                        if (!duration.isFinite() || duration <= 0) return AttachmentTokenEstimate(reason = "error")
                        val hasVideo = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_HAS_VIDEO) == "yes"
                        val hasAudio = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_HAS_AUDIO) == "yes"
                        mediaTokens(model, if (hasVideo) "video/" else "audio/", seconds = duration / 1000, hasAudio = hasAudio)
                    } finally { retriever.release(); source.close() }
                }
                else -> null
            }
            if (tokens != null) return AttachmentTokenEstimate(tokens)
        }
        val text = attachment.text ?: return AttachmentTokenEstimate(reason = "non_text")
        return AttachmentTokenEstimate(LocalChatBackend.estimateTokens(text))
    }
}
