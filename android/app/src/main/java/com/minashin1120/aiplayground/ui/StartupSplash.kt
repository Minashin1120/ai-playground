package com.minashin1120.aiplayground.ui

import android.app.Activity
import android.content.Context
import android.content.ContextWrapper
import android.os.Build
import androidx.compose.animation.core.CubicBezierEasing
import androidx.compose.animation.core.EaseIn
import androidx.compose.animation.core.EaseOut
import androidx.compose.animation.core.Easing
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.LinearEasing
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.setValue
import androidx.compose.runtime.withFrameNanos
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.BlendMode
import androidx.compose.ui.graphics.BlurEffect
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.CompositingStrategy
import androidx.compose.ui.graphics.GraphicsLayerScope
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.RenderEffect
import androidx.compose.ui.graphics.TileMode
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.withTransform
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.unit.sp
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.pow
import kotlin.math.sin

// Timeline (ms). Everything is derived from one frame clock so the layers never drift apart. The intro plays from the
// first frame; the logo then idles until startup has finished, and only then does the dive start.
private const val LOGO_IN_START = 60f
private const val LOGO_IN_END = 960f
private const val TITLE_IN_END = 800f
private const val SPARKLE_START = 380f
private const val SPARKLE_END = 1_380f
private const val DIVE_EARLIEST = 1_050f
// The zoom is exponential in time (maxScale^progress with linear progress): every frame multiplies the scale by
// the same factor, so the dive starts at once and keeps accelerating on screen.
private const val DIVE_MS = 300f
private const val FADE_TAIL_MS = 150f

private const val TITLE = "AI Playground"

// The animation is drawn at 60 Hz even on 120 Hz panels: three full-screen layers plus blur do not fit in an 8 ms frame on mid-range GPUs.
private const val SPLASH_REFRESH_RATE = 60f
// Frames slower than this (after the warm-up) count as dropped; a few of them switch the blur off for the rest of the animation.
private const val WARMUP_FRAMES = 6
private const val SLOW_FRAME_NANOS = 26_000_000L
private const val SLOW_FRAME_LIMIT = 3

private const val BLOOM_STEPS = 5
private const val BLOOM_SPREAD = 0.045f
private const val BLOOM_ALPHA = 0.2f

private val Backdrop = Color(0xFF05070F)
private val Indigo = Color(0xFF5356D8)
private val Teal = Color(0xFF0DD4BF)

// Same 108-unit geometry as `res/drawable/ic_playground_mark.xml`, drawn as vector paths so it stays sharp at any zoom.
private const val VIEWPORT = 108f
private val MarkCenter = Offset(54f, 48f)
private const val BODY_HALF_WIDTH = 29f
private const val BODY_HALF_HEIGHT = 21.5f

private val BubblePath = Path().apply {
    moveTo(25f, 27f)
    lineTo(83f, 27f)
    lineTo(83f, 70f)
    lineTo(49f, 70f)
    lineTo(33f, 83f)
    lineTo(33f, 70f)
    lineTo(25f, 70f)
    close()
}

private val SparklePath = Path().apply {
    moveTo(54f, 32f)
    lineTo(58f, 44f)
    lineTo(70f, 48f)
    lineTo(58f, 52f)
    lineTo(54f, 64f)
    lineTo(50f, 52f)
    lineTo(38f, 48f)
    lineTo(50f, 44f)
    close()
}

/** Overshooting ease-out used for the logo arrival. */
private val EaseOutBack: Easing = CubicBezierEasing(0.34f, 1.56f, 0.64f, 1f)

/** One frame of the timeline: [t] is the time since the first frame, [diveAt] when the dive began (NaN until startup has finished). */
private class SplashTime(val t: Float, val diveAt: Float) {
    /** Progress of a dive phase that starts [offset] ms into the dive and lasts [length] ms; 0 while the logo is still waiting. */
    fun dive(offset: Float = 0f, length: Float = DIVE_MS, easing: Easing = LinearEasing): Float =
        if (diveAt.isNaN()) 0f else phase(t, diveAt + offset, diveAt + offset + length, easing)
}

private class Twinkle(val angle: Float, val distance: Float, val size: Float, val phase: Float, val color: Color)

private val Twinkles = listOf(
    Twinkle(-2.45f, 0.78f, 0.55f, 0.0f, Color.White),
    Twinkle(-0.75f, 0.92f, 0.38f, 1.7f, Teal),
    Twinkle(0.35f, 0.86f, 0.62f, 3.1f, Color.White),
    Twinkle(1.25f, 0.70f, 0.30f, 4.4f, Teal),
    Twinkle(2.35f, 0.96f, 0.48f, 2.2f, Color.White),
    Twinkle(3.05f, 0.64f, 0.28f, 5.3f, Teal),
    Twinkle(-1.60f, 0.62f, 0.26f, 0.9f, Color.White),
)

/** Screen-space placement of the 108-unit mark: size, position and the zoom needed for its body to cover the screen. */
private class SplashLayout(width: Float, height: Float) {
    val unit = width * 0.5f / VIEWPORT
    val left = (width - VIEWPORT * unit) / 2f
    val top = height * 0.42f - VIEWPORT * unit / 2f
    val pivot = Offset(left + MarkCenter.x * unit, top + MarkCenter.y * unit)
    val logoPx = VIEWPORT * unit
    val diagonal = hypot(width, height)
    val maxScale = max(
        width / 2f / (BODY_HALF_WIDTH * unit),
        (height / 2f + abs(pivot.y - height / 2f)) / (BODY_HALF_HEIGHT * unit),
    ) * 1.15f
}

private fun phase(t: Float, from: Float, to: Float, easing: Easing = LinearEasing): Float =
    easing.transform(((t - from) / (to - from)).coerceIn(0f, 1f))

/** Blur is a RenderEffect (API 31+); older devices, and devices that cannot keep up (`lowQuality`), skip it and keep the rest of the motion. */
private fun GraphicsLayerScope.blurEffect(radiusDp: Float, lowQuality: Boolean): RenderEffect? =
    if (!lowQuality && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S && radiusDp > 0.5f) {
        val radius = radiusDp * density
        BlurEffect(radius, radius, TileMode.Decal)
    } else {
        null
    }

private fun DrawScope.drawBubble(layout: SplashLayout, zoom: Float, color: Color, blendMode: BlendMode = BlendMode.SrcOver) {
    withTransform({
        scale(zoom, zoom, layout.pivot)
        translate(layout.left, layout.top)
        scale(layout.unit, layout.unit, Offset.Zero)
    }) {
        drawPath(path = BubblePath, color = color, blendMode = blendMode)
    }
}

private fun DrawScope.drawSparkle(layout: SplashLayout, zoom: Float, rotation: Float, pulse: Float, color: Color) {
    withTransform({
        scale(zoom, zoom, layout.pivot)
        translate(layout.left, layout.top)
        scale(layout.unit, layout.unit, Offset.Zero)
        rotate(rotation, MarkCenter)
        scale(pulse, pulse, MarkCenter)
    }) {
        drawPath(path = SparklePath, color = color)
    }
}

private fun Context.findHostActivity(): Activity? {
    var current: Context? = this
    while (current is ContextWrapper) {
        if (current is Activity) return current
        current = current.baseContext
    }
    return null
}

/** Asks the window for [SPLASH_REFRESH_RATE] while the splash is on screen and gives the previous request back afterwards. */
@Composable
private fun CapRefreshRate() {
    val view = LocalView.current
    DisposableEffect(view) {
        val window = view.context.findHostActivity()?.window
        val previous = window?.attributes?.preferredRefreshRate ?: 0f
        window?.let { it.attributes = it.attributes.apply { preferredRefreshRate = SPLASH_REFRESH_RATE } }
        onDispose {
            window?.let { it.attributes = it.attributes.apply { preferredRefreshRate = previous } }
        }
    }
}

private fun DrawScope.drawOrb(color: Color, center: Offset, radius: Float, alpha: Float) {
    if (alpha <= 0.001f) return
    drawCircle(
        brush = Brush.radialGradient(listOf(color.copy(alpha = alpha), Color.Transparent), center = center, radius = radius),
        radius = radius,
        center = center,
    )
}

/**
 * Startup animation: soft light blooms drift behind the logo while it blurs into focus, the title slides up
 * under it and the sparkle spins; the logo keeps idling until [ready] (startup has finished), then the camera
 * dives through the speech bubble (the bubble opens into a window onto the app) with a shock ring. Blur is used
 * for the arrival and the dive; the glow and the ring are layered shapes because a full-screen blur per frame is
 * what made the animation stutter on 120 Hz devices.
 */
@Composable
internal fun StartupSplash(enabled: Boolean, ready: Boolean, onFinished: () -> Unit = {}) {
    if (!enabled) {
        val currentOnFinished by rememberUpdatedState(onFinished)
        LaunchedEffect(Unit) { currentOnFinished() }
        return
    }

    // Not saveable: a process-restored cold start must play the intro even though the previous
    // activity had already finished it (rememberSaveable would restore `false` and skip it).
    var visible by remember { mutableStateOf(true) }
    if (!visible) return

    var elapsed by remember { mutableFloatStateOf(0f) }
    var diveAt by remember { mutableFloatStateOf(Float.NaN) }
    var lowQuality by remember { mutableStateOf(false) }
    val currentReady by rememberUpdatedState(ready)
    val currentOnFinished by rememberUpdatedState(onFinished)
    val textMeasurer = rememberTextMeasurer()
    val titleStyle = TextStyle(fontFamily = WebFonts.sans, fontWeight = FontWeight.Bold, fontSize = 28.sp, color = Color.White)
    CapRefreshRate()
    LaunchedEffect(Unit) {
        var start = 0L
        var previous = 0L
        var frames = 0
        var slow = 0
        while (true) {
            val now = withFrameNanos { it }
            if (start == 0L) start = now
            if (!lowQuality && previous != 0L && ++frames > WARMUP_FRAMES && now - previous > SLOW_FRAME_NANOS && ++slow >= SLOW_FRAME_LIMIT) {
                lowQuality = true
            }
            previous = now
            val t = (now - start) / 1_000_000f
            elapsed = t
            if (diveAt.isNaN() && currentReady && t >= DIVE_EARLIEST) diveAt = t
            if (!diveAt.isNaN() && t >= diveAt + DIVE_MS + FADE_TAIL_MS) break
        }
        visible = false
        currentOnFinished()
    }

    Box(
        Modifier
            .fillMaxSize()
            .graphicsLayer {
                compositingStrategy = CompositingStrategy.Offscreen
                alpha = 1f - SplashTime(elapsed, diveAt).dive(DIVE_MS - 100f, 100f + FADE_TAIL_MS, EaseOut)
            },
    ) {
        // 1. Backdrop: dark base, drifting light blooms and twinkling stars.
        Canvas(Modifier.fillMaxSize()) {
            val time = SplashTime(elapsed, diveAt)
            val t = time.t
            val layout = SplashLayout(size.width, size.height)
            val glow = phase(t, 0f, 900f, EaseOut)
            val zoom = time.dive()
            val drift = t / 1_000f * 2.2f
            val reach = max(size.width, size.height)
            drawRect(Backdrop)
            drawOrb(
                Indigo,
                Offset(size.width / 2f + cos(drift) * size.width * 0.16f, layout.pivot.y - size.height * 0.05f + sin(drift * 1.3f) * size.height * 0.04f),
                reach * 0.55f * (1f + zoom * 0.4f),
                0.5f * glow,
            )
            drawOrb(
                Teal,
                Offset(size.width / 2f - cos(drift * 0.8f + 1f) * size.width * 0.2f, layout.pivot.y + size.height * 0.1f + sin(drift) * size.height * 0.04f),
                reach * 0.42f * (1f + zoom * 0.4f),
                0.28f * glow,
            )
            for (twinkle in Twinkles) {
                val burst = 1f + zoom * 5f
                val angle = twinkle.angle + drift * 0.12f
                val center = Offset(
                    layout.pivot.x + cos(angle) * twinkle.distance * layout.logoPx * burst,
                    layout.pivot.y + sin(angle) * twinkle.distance * layout.logoPx * 0.8f * burst,
                )
                val pulse = 0.5f + 0.5f * sin(t / 1_000f * 5f + twinkle.phase)
                val alpha = glow * (0.25f + 0.75f * pulse) * (1f - zoom)
                val starScale = twinkle.size * layout.unit
                withTransform({
                    translate(center.x, center.y)
                    scale(starScale, starScale, Offset.Zero)
                    translate(-MarkCenter.x, -MarkCenter.y)
                }) {
                    drawPath(path = SparklePath, color = twinkle.color.copy(alpha = alpha))
                }
            }
        }

        // 2. Bloom: the bubble silhouette, stacked in growing copies, glowing around the sharp logo (no blur: it is drawn every frame).
        Canvas(Modifier.fillMaxSize()) {
            val time = SplashTime(elapsed, diveAt)
            val t = time.t
            val layout = SplashLayout(size.width, size.height)
            val arrive = phase(t, LOGO_IN_START, LOGO_IN_END, EaseOutBack)
            val zoom = time.dive()
            val breathe = 0.85f + 0.15f * sin(t / 1_000f * 4f)
            val strength = 0.85f * phase(t, 120f, 800f, EaseOut) * breathe * (1f - time.dive(length = 200f))
            if (strength > 0.001f) {
                val scale = (0.55f + 0.45f * arrive) * layout.maxScale.pow(zoom)
                for (i in 0 until BLOOM_STEPS) {
                    drawBubble(layout, scale * (1.04f + BLOOM_SPREAD * i), Indigo.copy(alpha = strength * BLOOM_ALPHA))
                }
            }
        }

        // 3. Logo: blurs into focus with a springy scale, spins its sparkle, then blurs out as the camera dives through.
        Canvas(
            Modifier
                .fillMaxSize()
                .graphicsLayer {
                    val time = SplashTime(elapsed, diveAt)
                    val t = time.t
                    val arrivalBlur = (1f - phase(t, LOGO_IN_START, LOGO_IN_END, EaseOut)) * 28f
                    val diveBlur = time.dive(easing = EaseIn) * 14f
                    renderEffect = blurEffect(arrivalBlur + diveBlur, lowQuality)
                },
        ) {
            val time = SplashTime(elapsed, diveAt)
            val t = time.t
            val layout = SplashLayout(size.width, size.height)
            val arrive = phase(t, LOGO_IN_START, LOGO_IN_END, EaseOutBack)
            val zoom = layout.maxScale.pow(time.dive())
            val total = (0.55f + 0.45f * arrive) * zoom
            val appear = phase(t, LOGO_IN_START, 520f, EaseOut)
            val spin = phase(t, SPARKLE_START, SPARKLE_END, FastOutSlowInEasing)
            val pulse = 1f + 0.3f * sin(PI.toFloat() * spin)
            drawBubble(layout, total, Color.White.copy(alpha = appear))
            drawSparkle(layout, total, 180f * spin, pulse, Indigo.copy(alpha = appear))
        }

        // 4. Title: slides up from below from the first frame, then fades as the dive begins.
        Canvas(Modifier.fillMaxSize()) {
            val time = SplashTime(elapsed, diveAt)
            val t = time.t
            val layout = SplashLayout(size.width, size.height)
            val alpha = phase(t, 0f, 500f, EaseOut) * (1f - time.dive(length = 150f, easing = EaseOut))
            if (alpha > 0.001f) {
                // Measured every frame (the measurer caches it) so the Web font replaces the fallback once it has loaded.
                val title = textMeasurer.measure(TITLE, titleStyle)
                val rise = (1f - phase(t, 0f, TITLE_IN_END, FastOutSlowInEasing)) * 56f * density
                val sink = time.dive(easing = EaseIn) * 24f * density
                drawText(
                    title,
                    topLeft = Offset((size.width - title.size.width) / 2f, layout.top + 92f * layout.unit + rise + sink),
                    alpha = alpha,
                )
            }
        }

        // 5. Window: cuts the bubble out of everything above so the app shows through the zooming bubble.
        Canvas(Modifier.fillMaxSize()) {
            val time = SplashTime(elapsed, diveAt)
            val layout = SplashLayout(size.width, size.height)
            val dive = time.dive()
            val open = (dive / 0.25f).coerceIn(0f, 1f)
            if (open > 0f) {
                val zoom = layout.maxScale.pow(dive)
                // Wider, fainter copies first give the window a soft edge.
                drawBubble(layout, zoom * 1.1f, Color.Black.copy(alpha = open * 0.3f), BlendMode.DstOut)
                drawBubble(layout, zoom * 1.05f, Color.Black.copy(alpha = open * 0.45f), BlendMode.DstOut)
                drawBubble(layout, zoom, Color.Black.copy(alpha = open), BlendMode.DstOut)
            }
        }

        // 6. Shock ring: a soft ring (a wide faint stroke under a narrow one) racing outwards when the dive begins.
        Canvas(Modifier.fillMaxSize()) {
            val time = SplashTime(elapsed, diveAt)
            val layout = SplashLayout(size.width, size.height)
            val ring = time.dive(length = DIVE_MS + FADE_TAIL_MS, easing = EaseOut)
            if (ring > 0f && ring < 1f) {
                val radius = layout.logoPx * 0.3f + layout.diagonal * 0.8f * ring
                val width = layout.unit * (6f - 4f * ring)
                drawCircle(color = Teal.copy(alpha = 0.2f * (1f - ring)), radius = radius, center = layout.pivot, style = Stroke(width = width * 2.5f))
                drawCircle(color = Teal.copy(alpha = 0.5f * (1f - ring)), radius = radius, center = layout.pivot, style = Stroke(width = width))
            }
        }
    }
}
