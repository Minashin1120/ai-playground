package com.minashin1120.aiplayground.ui

import android.graphics.Color
import android.view.View
import android.view.ViewGroup
import android.webkit.RenderProcessGoneDetail
import android.webkit.WebResourceError
import android.webkit.WebResourceRequest
import android.webkit.WebResourceResponse
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.compose.foundation.layout.size
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.key
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.compose.LocalLifecycleOwner
import java.io.ByteArrayInputStream

/**
 * Inline SVG is drawn by Compose (AndroidSVG) when it is a still drawing. AndroidSVG does not run
 * SMIL (`<animate>` and friends) or CSS animations, while the browser on Web does. Animated markup
 * is therefore shown in a locked-down WebView, the same way Web draws a fenced SVG through `<img>`.
 */

private val SMIL_ANIMATION = Regex(
    """<\s*(?:[A-Za-z_][\w.-]*:)?(?:animate|animateTransform|animateMotion|animateColor|set|discard)\b""",
    RegexOption.IGNORE_CASE,
)
/** Active content the animated WebView path must not receive (the SVG is also loaded as an `<img>`). */
private val UNSAFE_MARKUP = Regex(
    """<\s*(?:script|foreignObject|iframe|object|embed)\b|javascript\s*:|(?:^|[\s;"'])on[a-z]+\s*=""",
    RegexOption.IGNORE_CASE,
)
/** External references never load in the app; only data URIs and fragment (`#id`) links are kept. */
private val EXTERNAL_REFERENCE = Regex(
    """(?:href|src)\s*=\s*["']\s*(?:https?:)?//|url\(\s*["']?\s*(?:https?:)?//""",
    RegexOption.IGNORE_CASE,
)
private val STYLE_BLOCK = Regex("""<style\b[^>]*>([\s\S]*?)</style\s*>""", RegexOption.IGNORE_CASE)
private val STYLE_ATTRIBUTE = Regex("""\bstyle\s*=\s*(["'])([\s\S]*?)\1""", RegexOption.IGNORE_CASE)
private val KEYFRAMES = Regex("""@(?:-[A-Za-z]+-)?keyframes\b""", RegexOption.IGNORE_CASE)
private val ANIMATION_PROPERTY = Regex("""(?:^|[;{\s])animation(?:-[A-Za-z-]+)?\s*:""", RegexOption.IGNORE_CASE)
private val ROOT_SVG_TAG = Regex("""<svg\b([^>]*)>""", RegexOption.IGNORE_CASE)
private const val SVG_NAMESPACE = """ xmlns="http://www.w3.org/2000/svg""""
private const val XLINK_NAMESPACE = """ xmlns:xlink="http://www.w3.org/1999/xlink""""
private const val HEX = "0123456789ABCDEF"

/** True when the drawing contains declarative animation that AndroidSVG would freeze. */
internal fun svgHasAnimation(markup: String): Boolean {
    if (SMIL_ANIMATION.containsMatchIn(markup)) return true
    if (STYLE_BLOCK.findAll(markup).any { block ->
            val css = block.groupValues[1]
            KEYFRAMES.containsMatchIn(css) || ANIMATION_PROPERTY.containsMatchIn(css)
        }) return true
    return STYLE_ATTRIBUTE.findAll(markup).any { ANIMATION_PROPERTY.containsMatchIn(it.groupValues[2]) }
}

/**
 * The animated path is only used for markup that cannot run scripts, embed foreign HTML or reach
 * the network. Anything else falls back to the still AndroidSVG rendering (scripts are ignored).
 */
internal fun svgAnimationSafe(markup: String): Boolean =
    !UNSAFE_MARKUP.containsMatchIn(markup) && !EXTERNAL_REFERENCE.containsMatchIn(markup)

/**
 * Web `buildSvgCodeRenderHtml`: the drawing is a data-URI `<img>`, so scripts and external
 * references stay outside the app. The root needs the SVG namespace when the model omitted it, and
 * the xlink namespace when the drawing uses `xlink:`.
 */
internal fun animatedSvgDataUri(markup: String): String {
    var attrs = ROOT_SVG_TAG.find(markup)?.groupValues?.get(1) ?: ""
    if (!Regex("""\sxmlns\s*=""").containsMatchIn(attrs)) attrs = SVG_NAMESPACE + attrs
    if (markup.contains("xlink:") && !Regex("""\sxmlns:xlink\s*=""").containsMatchIn(attrs)) {
        attrs = XLINK_NAMESPACE + attrs
    }
    val root = ROOT_SVG_TAG.find(markup)
    val normalized = if (root == null) markup else {
        markup.substring(0, root.range.first) + "<svg" + attrs + ">" + markup.substring(root.range.last + 1)
    }
    return "data:image/svg+xml;charset=utf-8," + percentEncode(normalized)
}

internal fun animatedSvgHtml(markup: String): String =
    """<!doctype html><html><head><meta charset="utf-8">
        <meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
        <style>html,body{margin:0;padding:0;width:100%;height:100%;overflow:hidden;background:transparent}
        img{display:block;width:100%;height:100%;object-fit:contain;-webkit-user-select:none;user-select:none}</style>
        </head><body><img src="${animatedSvgDataUri(markup)}" alt="SVG"></body></html>"""

private fun percentEncode(text: String): String {
    val bytes = text.toByteArray(Charsets.UTF_8)
    val out = StringBuilder(bytes.size + bytes.size / 2)
    for (byte in bytes) {
        val value = byte.toInt() and 0xFF
        val unreserved = value in 'A'.code..'Z'.code || value in 'a'.code..'z'.code ||
            value in '0'.code..'9'.code || value == '-'.code || value == '.'.code ||
            value == '_'.code || value == '~'.code
        if (unreserved) out.append(value.toChar()) else out.append('%').append(HEX[value ushr 4]).append(HEX[value and 0x0F])
    }
    return out.toString()
}

/**
 * Draws one animated SVG. The WebView never runs scripts, never reaches the network or local files,
 * and the animation is paused while the app is not started. It is destroyed with the message row.
 */
@Composable
internal fun AnimatedSvg(markup: String, widthPx: Int, heightPx: Int, onError: () -> Unit = {}) {
    val density = LocalDensity.current
    val document = remember(markup) { animatedSvgHtml(markup) }
    val holder = remember { SvgWebViewHolder() }
    val lifecycle = LocalLifecycleOwner.current.lifecycle
    var failed by remember(markup) { mutableStateOf(false) }

    LaunchedEffect(failed) { if (failed) onError() }

    DisposableEffect(lifecycle) {
        val observer = LifecycleEventObserver { _, event ->
            when (event) {
                Lifecycle.Event.ON_START -> holder.view?.onResume()
                Lifecycle.Event.ON_STOP -> holder.view?.onPause()
                else -> Unit
            }
        }
        lifecycle.addObserver(observer)
        onDispose { lifecycle.removeObserver(observer) }
    }

    key(markup, widthPx, heightPx) {
        if (failed) return@key
        AndroidView(
            modifier = Modifier.size(
                with(density) { widthPx.toDp() },
                with(density) { heightPx.toDp() },
            ),
            factory = { viewContext ->
                WebView(viewContext).apply {
                    layoutParams = ViewGroup.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT)
                    setBackgroundColor(Color.TRANSPARENT)
                    overScrollMode = View.OVER_SCROLL_NEVER
                    isVerticalScrollBarEnabled = false
                    isHorizontalScrollBarEnabled = false
                    isScrollContainer = false
                    isEnabled = false
                    isFocusable = false
                    isFocusableInTouchMode = false
                    isLongClickable = false
                    settings.apply {
                        javaScriptEnabled = false
                        blockNetworkLoads = true
                        allowFileAccess = false
                        allowContentAccess = false
                        domStorageEnabled = false
                    }
                    webViewClient = SvgAnimationWebViewClient(holder) { failed = true }
                    loadDataWithBaseURL(null, document, "text/html", "utf-8", null)
                    holder.view = this
                }
            },
            onRelease = { view ->
                holder.view = null
                runCatching {
                    view.stopLoading()
                    view.destroy()
                }
            },
        )
    }
}

private class SvgWebViewHolder(var view: WebView? = null)

private class SvgAnimationWebViewClient(
    private val holder: SvgWebViewHolder,
    private val onFailed: () -> Unit,
) : WebViewClient() {
    override fun shouldOverrideUrlLoading(view: WebView, request: WebResourceRequest): Boolean = true

    override fun shouldInterceptRequest(view: WebView, request: WebResourceRequest): WebResourceResponse? =
        if (request.url.scheme == "data") null
        else WebResourceResponse("text/plain", "utf-8", ByteArrayInputStream(ByteArray(0)))

    override fun onReceivedError(view: WebView, request: WebResourceRequest, error: WebResourceError) {
        if (request.isForMainFrame) onFailed()
    }

    override fun onRenderProcessGone(view: WebView, detail: RenderProcessGoneDetail): Boolean {
        holder.view = null
        onFailed()
        return true
    }
}
