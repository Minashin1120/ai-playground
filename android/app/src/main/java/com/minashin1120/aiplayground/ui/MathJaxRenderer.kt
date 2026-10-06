package com.minashin1120.aiplayground.ui

import android.annotation.SuppressLint
import android.content.Context
import android.os.Handler
import android.os.Looper
import android.util.LruCache
import android.webkit.JavascriptInterface
import android.webkit.RenderProcessGoneDetail
import android.webkit.WebResourceRequest
import android.webkit.WebResourceResponse
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.text.InlineTextContent
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.translate
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.layout.onPlaced
import androidx.compose.ui.layout.positionInParent
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.Placeholder
import androidx.compose.ui.text.PlaceholderVerticalAlign
import androidx.compose.ui.text.TextLayoutResult
import androidx.compose.ui.unit.Constraints
import androidx.compose.ui.unit.TextUnit
import androidx.compose.ui.unit.em
import androidx.compose.ui.unit.sp
import androidx.compose.ui.unit.takeOrElse
import com.caverock.androidsvg.SVG
import org.json.JSONObject
import java.io.ByteArrayInputStream
import kotlin.math.abs

/**
 * One formula typeset by MathJax as SVG. Sizes are in em of MathJax's TeX font (its viewBox is in
 * 1/1000 em, with the baseline at y = 0).
 */
internal data class MathSvg(val markup: String, val widthEm: Float, val ascentEm: Float, val depthEm: Float) {
    val heightEm: Float get() = ascentEm + depthEm
}

private val VIEW_BOX = Regex("""viewBox="(-?[\d.eE+-]+)\s+(-?[\d.eE+-]+)\s+(-?[\d.eE+-]+)\s+(-?[\d.eE+-]+)"""")

/** Reads the size and baseline of a MathJax `<svg>`, or null when it is not a usable drawing. */
internal fun parseMathSvg(markup: String): MathSvg? {
    if (markup.isBlank() || markup.length > MAX_SVG_LENGTH) return null
    val box = VIEW_BOX.find(markup)?.groupValues ?: return null
    val minY = box[2].toFloatOrNull() ?: return null
    val width = box[3].toFloatOrNull() ?: return null
    val height = box[4].toFloatOrNull() ?: return null
    if (width <= 0f || height <= 0f || !width.isFinite() || !height.isFinite()) return null
    return MathSvg(markup, width / 1000f, -minY / 1000f, (minY + height) / 1000f)
}

/**
 * Web's MathJax (CHTML, `matchFontHeight`) scales formulas so their x-height matches the text:
 * Noto Sans JP 0.543em over MathJax TeX 0.442em.
 */
internal const val WEB_MATH_SCALE = 0.543f / 0.442f

/**
 * Compose's `PlaceholderVerticalAlign.TextCenter` puts a placeholder's center this far above the
 * baseline: half of Noto Sans JP's hhea ascent (1.160em) minus descent (0.288em).
 */
internal const val TEXT_CENTER_EM = (1.160f - 0.288f) / 2f

private const val MAX_SVG_LENGTH = 400_000
private const val MAX_TEX_LENGTH = 20_000

/**
 * Typesets TeX with the MathJax 3 bundled in `assets/mathjax` (the version Web loads from its CDN),
 * running in one off-screen WebView that never shows HTML or reaches the network. Results are cached
 * for the process; the WebView is released after a quiet minute and recreated on demand.
 */
@SuppressLint("StaticFieldLeak")
internal object MathJaxRenderer {
    private const val HOST_URL = "file:///android_asset/mathjax/host.html"
    private const val ASSET_PREFIX = "file:///android_asset/mathjax/"
    private const val STARTUP_TIMEOUT_MS = 30_000L
    private const val RENDER_TIMEOUT_MS = 20_000L
    private const val IDLE_RELEASE_MS = 60_000L

    private val main = Handler(Looper.getMainLooper())
    private val cache = LruCache<String, MathSvg>(512)
    private val failed = LruCache<String, Boolean>(256)
    private val waiting = HashMap<String, MutableList<(MathSvg?) -> Unit>>()
    private val queued = ArrayList<String>()
    private var webView: WebView? = null
    private var ready = false
    private var unavailable = false
    private var generation = 0

    private fun key(tex: String, display: Boolean) = (if (display) "D:" else "I:") + tex

    fun cached(tex: String, display: Boolean): MathSvg? = cache.get(key(tex, display))

    fun hasFailed(tex: String, display: Boolean): Boolean =
        unavailable || failed.get(key(tex, display)) == true || tex.isBlank() || tex.length > MAX_TEX_LENGTH

    /** Calls [onResult] on the main thread with the SVG, or null when MathJax cannot set it. */
    fun request(context: Context, tex: String, display: Boolean, onResult: (MathSvg?) -> Unit) {
        val id = key(tex, display)
        cache.get(id)?.let { onResult(it); return }
        if (hasFailed(tex, display)) { onResult(null); return }
        waiting[id]?.let { it += onResult; return }
        waiting[id] = mutableListOf(onResult)
        main.removeCallbacks(releaseIdle)
        if (webView == null && !start(context.applicationContext)) return
        if (ready) send(id) else queued += id
    }

    @SuppressLint("SetJavaScriptEnabled", "AddJavascriptInterface")
    private fun start(context: Context): Boolean {
        val view = runCatching { WebView(context) }.getOrNull()
        if (view == null) { giveUp(); return false }
        val token = ++generation
        view.settings.apply {
            javaScriptEnabled = true
            blockNetworkLoads = true
            allowFileAccess = false
            allowContentAccess = false
            domStorageEnabled = false
        }
        view.webViewClient = object : WebViewClient() {
            override fun shouldOverrideUrlLoading(view: WebView, request: WebResourceRequest): Boolean = true

            override fun shouldInterceptRequest(view: WebView, request: WebResourceRequest): WebResourceResponse? =
                if (request.url.toString().startsWith(ASSET_PREFIX)) null
                else WebResourceResponse("text/plain", "utf-8", ByteArrayInputStream(ByteArray(0)))

            override fun onRenderProcessGone(view: WebView, detail: RenderProcessGoneDetail): Boolean {
                main.post { if (token == generation) reset(failPending = true) }
                return true
            }
        }
        view.addJavascriptInterface(Bridge(token), "AndroidMathJax")
        webView = view
        ready = false
        main.postDelayed({ if (token == generation && !ready) giveUp() }, STARTUP_TIMEOUT_MS)
        view.loadUrl(HOST_URL)
        return true
    }

    private fun send(id: String) {
        val view = webView ?: return
        val display = id.startsWith("D:")
        val tex = id.substring(2)
        view.evaluateJavascript("aipRender(${JSONObject.quote(id)}, ${JSONObject.quote(tex)}, $display);", null)
        val token = generation
        main.postDelayed({ if (token == generation && waiting.containsKey(id)) finish(id, null) }, RENDER_TIMEOUT_MS)
    }

    private fun finish(id: String, svg: MathSvg?) {
        if (svg != null) cache.put(id, svg) else failed.put(id, true)
        waiting.remove(id)?.forEach { it(svg) }
        if (waiting.isEmpty()) {
            main.removeCallbacks(releaseIdle)
            main.postDelayed(releaseIdle, IDLE_RELEASE_MS)
        }
    }

    private val releaseIdle = Runnable { if (waiting.isEmpty()) reset(failPending = false) }

    private fun reset(failPending: Boolean) {
        generation++
        ready = false
        queued.clear()
        webView?.let { view -> runCatching { view.stopLoading(); view.destroy() } }
        webView = null
        if (failPending) waiting.keys.toList().forEach { finish(it, null) }
    }

    /** No WebView on this device (or MathJax never started): keep the readable approximation. */
    private fun giveUp() {
        unavailable = true
        reset(failPending = true)
    }

    /** Called by host.html on the WebView's bridge thread; everything is handed back to the main thread. */
    private class Bridge(private val token: Int) {
        private val renderer = MathJaxRenderer

        @JavascriptInterface
        fun ready() {
            renderer.main.post {
                if (token != renderer.generation) return@post
                renderer.ready = true
                renderer.queued.toList().also { renderer.queued.clear() }.forEach { renderer.send(it) }
            }
        }

        @JavascriptInterface
        fun done(id: String, markup: String) {
            renderer.main.post {
                if (token == renderer.generation && renderer.waiting.containsKey(id)) renderer.finish(id, parseMathSvg(markup))
            }
        }

        @JavascriptInterface
        fun broken() {
            renderer.main.post { if (token == renderer.generation) renderer.giveUp() }
        }
    }
}

internal sealed interface MathState {
    data object Loading : MathState
    data class Ready(val svg: MathSvg) : MathState
    data object Failed : MathState
}

/** The MathJax rendering of [tex]; [MathState.Loading] and [MathState.Failed] show the text approximation. */
@Composable
internal fun rememberMathState(tex: String, display: Boolean): MathState {
    val context = LocalContext.current
    var state by remember(tex, display) {
        mutableStateOf(
            MathJaxRenderer.cached(tex, display)?.let { MathState.Ready(it) }
                ?: if (MathJaxRenderer.hasFailed(tex, display)) MathState.Failed else MathState.Loading,
        )
    }
    DisposableEffect(tex, display) {
        var active = true
        if (state == MathState.Loading) {
            MathJaxRenderer.request(context, tex, display) { svg ->
                if (active) state = svg?.let { MathState.Ready(it) } ?: MathState.Failed
            }
        }
        onDispose { active = false }
    }
    return state
}

/** Parses the SVG once per color; MathJax draws with `currentColor`, which becomes the text color. */
@Composable
internal fun rememberMathDrawing(svg: MathSvg, color: Color): SVG? = remember(svg, color) {
    val hex = String.format(java.util.Locale.ROOT, "#%06X", color.toArgb() and 0xFFFFFF)
    runCatching { SVG.getFromString(svg.markup.replace("currentColor", hex)) }.getOrNull()
}

/** Draws [drawing] with its top-left corner at ([left], [top]) and 1em = [emPx] (TeX font em). */
internal fun DrawScope.drawMath(drawing: SVG, svg: MathSvg, emPx: Float, left: Float, top: Float, alpha: Float) {
    val width = svg.widthEm * emPx
    val height = svg.heightEm * emPx
    if (width <= 0f || height <= 0f) return
    translate(left, top) {
        val canvas = drawContext.canvas.nativeCanvas
        val layer = if (alpha < 1f) canvas.saveLayerAlpha(0f, 0f, width, height, (alpha * 255).toInt()) else canvas.save()
        runCatching { drawing.renderToCanvas(canvas, android.graphics.RectF(0f, 0f, width, height)) }
        canvas.restoreToCount(layer)
    }
}

/** A display formula as its own box (Web `mjx-container[display]`), sized from the text's font size. */
@Composable
internal fun MathDisplayImage(svg: MathSvg, drawing: SVG, color: Color, fontSize: TextUnit, modifier: Modifier = Modifier) {
    val density = LocalDensity.current
    val emPx = with(density) { fontSize.takeOrElse { 16.sp }.toPx() } * WEB_MATH_SCALE
    val width = with(density) { (svg.widthEm * emPx).toDp() }
    val height = with(density) { (svg.heightEm * emPx).toDp() }
    Canvas(modifier.size(width, height)) { drawMath(drawing, svg, emPx, 0f, 0f, color.alpha) }
}

/**
 * The placeholder for one formula inside text. Compose can only center a placeholder on the text, so the
 * box is tall enough around that center and the drawing is then moved onto the line's baseline.
 */
internal fun inlineMathContent(
    item: InlineMath,
    svg: MathSvg,
    color: Color,
    availableWidth: Int,
    fontPx: Float,
    textLayout: () -> TextLayoutResult?,
): InlineTextContent {
    // A display formula wider than the text shrinks to fit instead of running off the bubble.
    val natural = svg.widthEm * WEB_MATH_SCALE * fontPx
    val scale = if (item.display && availableWidth != Constraints.Infinity && natural > availableWidth) {
        WEB_MATH_SCALE * availableWidth / natural
    } else WEB_MATH_SCALE
    val half = maxOf(svg.ascentEm * scale - TEXT_CENTER_EM, svg.depthEm * scale + TEXT_CENTER_EM)
    // Web `mjx-container[display]` keeps 1em above and below a display formula.
    val margin = if (item.display) 1f else 0f
    val placeholder = Placeholder((svg.widthEm * scale).em, (2 * half + 2 * margin).em, PlaceholderVerticalAlign.TextCenter)
    return InlineTextContent(placeholder) { InlineMathDrawing(svg, color, scale, item.start, textLayout) }
}

@Composable
private fun InlineMathDrawing(svg: MathSvg, color: Color, scale: Float, start: Int, textLayout: () -> TextLayoutResult?) {
    val drawing = rememberMathDrawing(svg, color) ?: return
    var top by remember { mutableFloatStateOf(Float.NaN) }
    Canvas(Modifier.fillMaxSize().onPlaced { top = it.positionInParent().y }) {
        val mathEm = size.width / svg.widthEm
        // Where TextCenter puts the baseline for Noto Sans JP; the laid-out line's baseline when known.
        val estimated = size.height / 2f + TEXT_CENTER_EM * mathEm / scale
        val result = textLayout()
        val baseline = if (result != null && !top.isNaN() && start < result.layoutInput.text.length) {
            (result.getLineBaseline(result.getLineForOffset(start)) - top).takeIf { abs(it - estimated) < size.height / 2f }
        } else null
        drawMath(drawing, svg, mathEm, 0f, (baseline ?: estimated) - svg.ascentEm * mathEm, color.alpha)
    }
}
