package com.minashin1120.aiplayground.ui

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.LruCache
import androidx.compose.animation.Crossfade
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
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
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
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
    onOpen: ((String) -> Unit)? = null,
    modifier: Modifier = Modifier,
    thumbnail: Boolean = true,
    compact: Boolean = true,
    maxDecodeWidth: Int = 1280,
    limitBytes: Long = 8L * 1024 * 1024,
    contentDescription: String? = null,
    shape: androidx.compose.ui.graphics.Shape = RoundedCornerShape(12.dp),
    contentScale: ContentScale = ContentScale.Fit,
    background: Color? = null,
) {
    val key = "$reference|${if (thumbnail) "t" else "f"}|$maxDecodeWidth"
    var bitmap by remember(key) { mutableStateOf(ProtectedImageCache.get(key)) }
    var failed by remember(key) { mutableStateOf(false) }
    LaunchedEffect(key) {
        if (bitmap != null || loader == null) return@LaunchedEffect
        val bytes = runCatching { loader(reference, thumbnail, limitBytes) }.getOrNull()
        val decoded = bytes?.let { data -> withContext(Dispatchers.Default) { decodeScaled(data, maxDecodeWidth) } }
        if (decoded != null) {
            ProtectedImageCache.put(key, decoded)
            bitmap = decoded
        } else {
            failed = true
        }
    }
    val click = if (onOpen != null) Modifier.clickable { onOpen(reference) } else Modifier
    Surface(
        shape = shape,
        color = background ?: MaterialTheme.colorScheme.surfaceVariant,
        modifier = modifier.clip(shape).then(click),
    ) {
        Box(
            if (compact) Modifier.fillMaxWidth().heightIn(min = 120.dp, max = 320.dp)
            else Modifier.fillMaxSize(),
            contentAlignment = Alignment.Center,
        ) {
            val image = bitmap
            val phase = when {
                image != null -> ImageLoadPhase.Loaded
                failed -> ImageLoadPhase.Failed
                else -> ImageLoadPhase.Loading
            }
            // Web `libShimmer` → `thumb-pop`: a soft sweep while loading, then a cross-fade to the image.
            Crossfade(phase, animationSpec = motionTween(LocalReduceMotion.current), label = "protected image") { shown ->
                when (shown) {
                    ImageLoadPhase.Loaded -> image?.let {
                        Image(
                            it.asImageBitmap(), contentDescription,
                            modifier = if (compact) Modifier.fillMaxWidth() else Modifier.fillMaxSize(), contentScale = contentScale,
                        )
                    }
                    ImageLoadPhase.Failed -> Text(
                        "画像を読み込めませんでした", style = MaterialTheme.typography.labelSmall,
                        modifier = Modifier.padding(16.dp),
                    )
                    ImageLoadPhase.Loading -> LoadingShimmer(
                        if (compact) Modifier.fillMaxWidth().height(120.dp) else Modifier.fillMaxSize(),
                    )
                }
            }
        }
    }
}

private enum class ImageLoadPhase { Loading, Loaded, Failed }

/** Diagonal highlight sweeping across the placeholder; static under reduced motion. */
@Composable
private fun LoadingShimmer(modifier: Modifier) {
    val reduce = LocalReduceMotion.current
    val base = MaterialTheme.colorScheme.surfaceVariant
    val highlight = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.08f)
    val transition = rememberInfiniteTransition(label = "image shimmer")
    val sweep by transition.animateFloat(
        initialValue = -1f,
        targetValue = 2f,
        animationSpec = infiniteRepeatable(tween(1_300, easing = LinearEasing)),
        label = "image shimmer sweep",
    )
    Canvas(modifier) {
        drawRect(base)
        if (!reduce) {
            val center = size.width * sweep
            drawRect(
                Brush.linearGradient(
                    colors = listOf(Color.Transparent, highlight, Color.Transparent),
                    start = Offset(center - size.width * 0.4f, 0f),
                    end = Offset(center + size.width * 0.4f, size.height),
                ),
            )
        }
    }
}
