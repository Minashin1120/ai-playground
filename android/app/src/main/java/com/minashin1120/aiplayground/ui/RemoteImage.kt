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
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.min
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.CancellationException
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
    /** Pixel size of the decoded image once it is available (the viewer uses it to find the drawn area). */
    onImageSize: ((IntSize) -> Unit)? = null,
    /** The image could not be loaded (the viewer stops ignoring taps for its loading state). */
    onLoadFailed: (() -> Unit)? = null,
    /**
     * Sizes the frame to the image's aspect ratio (Web `.chat-image` with `height: auto`) within [fitMaxWidth]
     * and [fitMaxHeight], so no letterbox is left around the image and only the image itself opens the viewer.
     */
    fitToImage: Boolean = false,
    fitMaxWidth: Dp = Dp.Infinity,
    fitMaxHeight: Dp = Dp.Infinity,
    /** Where the fitted frame sits in the available width (Web `object-fit: contain` centers it). */
    fitAlignment: Alignment = Alignment.TopStart,
) {
    val key = "$reference|${if (thumbnail) "t" else "f"}|$maxDecodeWidth"
    var bitmap by remember(key) { mutableStateOf(ProtectedImageCache.get(key)) }
    LaunchedEffect(bitmap) { bitmap?.let { onImageSize?.invoke(IntSize(it.width, it.height)) } }
    var failed by remember(key) { mutableStateOf(false) }
    val loadFailed = rememberUpdatedState(onLoadFailed)
    LaunchedEffect(key) {
        if (bitmap != null) return@LaunchedEffect
        if (loader == null) {
            failed = true
            loadFailed.value?.invoke()
            return@LaunchedEffect
        }
        val bytes = try { loader(reference, thumbnail, limitBytes) }
            catch (e: CancellationException) { throw e }
            catch (e: Exception) { null }
        val decoded = bytes?.let { data -> withContext(Dispatchers.Default) { decodeScaled(data, maxDecodeWidth) } }
        if (decoded != null) {
            ProtectedImageCache.put(key, decoded)
            bitmap = decoded
        } else {
            failed = true
            loadFailed.value?.invoke()
        }
    }
    val click = if (onOpen != null) Modifier.clickable { onOpen(reference) } else Modifier
    if (fitToImage) {
        BoxWithConstraints(modifier, contentAlignment = fitAlignment) {
            val width = min(maxWidth, fitMaxWidth).takeIf { it != Dp.Infinity } ?: 320.dp
            // Web leaves the height to the image; the finite cap only keeps extreme panoramas within layout limits.
            val height = min(min(maxHeight, fitMaxHeight), MAX_FITTED_HEIGHT)
            val image = bitmap
            val frame = if (image != null && image.width > 0 && image.height > 0) {
                val (w, h) = fittedImageSize(image.width, image.height, width, height)
                Modifier.size(w, h)
            } else {
                Modifier.width(width).heightIn(min = 120.dp)
            }
            ImageFrame(
                frame.clip(shape).then(click), shape, background, image, failed, contentDescription,
                inner = Modifier.fillMaxSize(), imageModifier = Modifier.fillMaxSize(), contentScale = ContentScale.Fit,
                loadingModifier = Modifier.fillMaxWidth().height(120.dp),
            )
        }
    } else {
        ImageFrame(
            modifier.clip(shape).then(click), shape, background, bitmap, failed, contentDescription,
            inner = if (compact) Modifier.fillMaxWidth().heightIn(min = 120.dp, max = 320.dp) else Modifier.fillMaxSize(),
            imageModifier = if (compact) Modifier.fillMaxWidth() else Modifier.fillMaxSize(),
            contentScale = contentScale,
            loadingModifier = if (compact) Modifier.fillMaxWidth().height(120.dp) else Modifier.fillMaxSize(),
        )
    }
}

private val MAX_FITTED_HEIGHT = 4000.dp

/** Largest size with the image's aspect ratio that fits in [maxWidth] × [maxHeight] (the width is filled first). */
internal fun fittedImageSize(imageWidth: Int, imageHeight: Int, maxWidth: Dp, maxHeight: Dp): Pair<Dp, Dp> {
    val aspect = imageWidth.toFloat() / imageHeight
    var width = maxWidth
    var height = width / aspect
    if (height > maxHeight) {
        height = maxHeight
        width = height * aspect
    }
    return width to height
}

@Composable
private fun ImageFrame(
    modifier: Modifier,
    shape: androidx.compose.ui.graphics.Shape,
    background: Color?,
    image: Bitmap?,
    failed: Boolean,
    contentDescription: String?,
    inner: Modifier,
    imageModifier: Modifier,
    contentScale: ContentScale,
    loadingModifier: Modifier,
) {
    Surface(
        shape = shape,
        color = background ?: MaterialTheme.colorScheme.surfaceVariant,
        modifier = modifier,
    ) {
        Box(inner, contentAlignment = Alignment.Center) {
            val phase = when {
                image != null -> ImageLoadPhase.Loaded
                failed -> ImageLoadPhase.Failed
                else -> ImageLoadPhase.Loading
            }
            // Web `libShimmer` → `thumb-pop`: a soft sweep while loading, then a cross-fade to the image.
            Crossfade(phase, animationSpec = motionTween(LocalReduceMotion.current), label = "protected image") { shown ->
                when (shown) {
                    ImageLoadPhase.Loaded -> image?.let {
                        Image(it.asImageBitmap(), contentDescription, modifier = imageModifier, contentScale = contentScale)
                    }
                    ImageLoadPhase.Failed -> Text(
                        "画像を読み込めませんでした", style = MaterialTheme.typography.labelSmall,
                        modifier = Modifier.padding(16.dp),
                    )
                    ImageLoadPhase.Loading -> LoadingShimmer(loadingModifier)
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
