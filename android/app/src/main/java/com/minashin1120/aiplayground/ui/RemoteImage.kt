package com.minashin1120.aiplayground.ui

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.LruCache
import androidx.compose.foundation.Image
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/** Loads same-origin attachment bytes with the Android Bearer token, or null on failure. */
typealias FileBytesLoader = suspend (reference: String, thumbnail: Boolean, limit: Long) -> ByteArray?

internal object ProtectedImageCache {
    private val cache = object : LruCache<String, Bitmap>(24 * 1024 * 1024) {
        override fun sizeOf(key: String, value: Bitmap): Int = value.byteCount
    }

    fun get(key: String): Bitmap? = cache.get(key)
    fun put(key: String, value: Bitmap) { cache.put(key, value) }
}

private fun decodeScaled(bytes: ByteArray, maxWidth: Int): Bitmap? {
    val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
    BitmapFactory.decodeByteArray(bytes, 0, bytes.size, bounds)
    var sample = 1
    val width = bounds.outWidth
    if (width > 0) {
        while (width / sample > maxWidth * 2) sample *= 2
    }
    val options = BitmapFactory.Options().apply { inSampleSize = sample }
    return runCatching { BitmapFactory.decodeByteArray(bytes, 0, bytes.size, options) }.getOrNull()
}

/**
 * Shows a protected image preview. Only same-origin references reach the loader, so
 * the Bearer token is never attached to a provider or third-party URL.
 */
@Composable
fun ProtectedImage(
    reference: String,
    loader: FileBytesLoader?,
    onOpen: (String) -> Unit,
    modifier: Modifier = Modifier,
    thumbnail: Boolean = true,
    contentDescription: String? = null,
) {
    val key = "$reference|${if (thumbnail) "t" else "f"}"
    var bitmap by remember(key) { mutableStateOf(ProtectedImageCache.get(key)) }
    var failed by remember(key) { mutableStateOf(false) }
    LaunchedEffect(key) {
        if (bitmap != null || loader == null) return@LaunchedEffect
        val bytes = runCatching { loader(reference, thumbnail, 8L * 1024 * 1024) }.getOrNull()
        val decoded = bytes?.let { data -> withContext(Dispatchers.Default) { decodeScaled(data, 1280) } }
        if (decoded != null) {
            ProtectedImageCache.put(key, decoded)
            bitmap = decoded
        } else {
            failed = true
        }
    }
    Surface(
        shape = RoundedCornerShape(12.dp),
        color = MaterialTheme.colorScheme.surfaceVariant,
        modifier = modifier.clip(RoundedCornerShape(12.dp)).clickable { onOpen(reference) },
    ) {
        Box(
            Modifier.fillMaxWidth().heightIn(min = 120.dp, max = 320.dp),
            contentAlignment = Alignment.Center,
        ) {
            val image = bitmap
            when {
                image != null -> Image(
                    image.asImageBitmap(), contentDescription,
                    modifier = Modifier.fillMaxWidth(), contentScale = ContentScale.Fit,
                )
                failed -> Text(
                    "画像を読み込めませんでした", style = MaterialTheme.typography.labelSmall,
                    modifier = Modifier.padding(16.dp),
                )
                else -> CircularProgressIndicator(Modifier.size(24.dp), strokeWidth = 2.dp)
            }
        }
    }
}
