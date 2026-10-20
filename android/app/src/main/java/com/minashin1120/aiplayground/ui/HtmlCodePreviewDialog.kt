package com.minashin1120.aiplayground.ui

import android.annotation.SuppressLint
import android.webkit.CookieManager
import android.webkit.WebChromeClient
import android.webkit.WebResourceError
import android.webkit.WebResourceRequest
import android.webkit.WebResourceResponse
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.getValue
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.viewinterop.AndroidView
import com.minashin1120.aiplayground.data.ServerOrigin
import org.json.JSONObject
import java.io.ByteArrayInputStream

internal const val HTML_PREVIEW_URL = "https://html-preview.invalid/preview"

/** Same language and doctype detection as Web `isHtmlPreviewCandidate`. */
internal fun isHtmlPreviewCandidate(language: String, code: String): Boolean {
    val token = language.trim().lowercase()
    return token in setOf("html", "htm", "xhtml") ||
        (token.isEmpty() && Regex("<!doctype\\s+html", RegexOption.IGNORE_CASE).containsMatchIn(code))
}

// Escape '<' so a code block containing </script> cannot escape the trusted shell.
private fun scriptString(value: String): String = JSONObject.quote(value)
    .replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")

@SuppressLint("SetJavaScriptEnabled") // Explicit preview action; code runs in an opaque sandbox, without a native bridge.
@Composable
internal fun HtmlCodePreviewDialog(code: String, onDismiss: () -> Unit) {
    val context = LocalContext.current
    val base = ServerOrigin.base + "/"
    val document = remember(code, base) {
        val security = context.assets.open("html-preview-security.js").bufferedReader().use { it.readText() }
        buildHtmlPreviewDocument(code, base, security)
    }
    var loadFailed by remember { mutableStateOf(false) }

    PlaygroundDialog(
        onDismissRequest = onDismiss,
        title = { Text("プレビュー") },
        confirmButton = {},
        fillHeight = true,
        padBody = false,
        showFooter = false,
        text = {
            if (loadFailed) {
                Text("HTMLプレビューを読み込めませんでした。閉じてからもう一度お試しください。")
            } else AndroidView(
                modifier = Modifier.fillMaxSize(),
                factory = { viewContext ->
                    WebView(viewContext).apply {
                        settings.apply {
                            javaScriptEnabled = true
                            allowFileAccess = false
                            allowContentAccess = false
                            setSupportMultipleWindows(true)
                            mediaPlaybackRequiresUserGesture = true
                        }
                        CookieManager.getInstance().setAcceptThirdPartyCookies(this, false)
                        webChromeClient = WebChromeClient()
                        // Do not use the app's authenticated HTTP client, cookies, or a JavaScript bridge.
                        // Serve the shell from a reserved origin, separate from the account server.
                        // Only the initial main-frame request can receive this document.
                        webViewClient = object : WebViewClient() {
                            override fun shouldOverrideUrlLoading(view: WebView, request: WebResourceRequest): Boolean =
                                request.isForMainFrame || request.url.scheme !in setOf("https", "http", "about", "data", "blob")

                            override fun shouldInterceptRequest(view: WebView, request: WebResourceRequest): WebResourceResponse? {
                                if (request.isForMainFrame && request.url.toString() == HTML_PREVIEW_URL) {
                                    return WebResourceResponse("text/html", "UTF-8", ByteArrayInputStream(document.toByteArray(Charsets.UTF_8)))
                                }
                                val host = request.url.host.orEmpty().lowercase()
                                if (host == "html-preview.invalid" || host == "polyfill.io" || host.endsWith(".polyfill.io") ||
                                    request.url.scheme !in setOf("https", "http", "about", "data", "blob")) {
                                    return WebResourceResponse("text/plain", "UTF-8", ByteArrayInputStream(ByteArray(0)))
                                }
                                return null
                            }
                            override fun onReceivedError(view: WebView, request: WebResourceRequest, error: WebResourceError) {
                                if (request.isForMainFrame) loadFailed = true
                            }
                        }
                        loadUrl(HTML_PREVIEW_URL)
                    }
                },
                onRelease = { view -> view.stopLoading(); view.destroy() },
            )
        },
    )
}

/** The outer shell is trusted; generated HTML only ever enters the sandboxed srcdoc. */
internal fun buildHtmlPreviewDocument(code: String, base: String, security: String): String =
    """<!doctype html><html><head><meta charset="utf-8">
        <meta name="referrer" content="no-referrer">
        <meta name="viewport" content="width=device-width,initial-scale=1">
        <style>html,body,iframe{width:100%;height:100%;margin:0;border:0;background:#fff}body{overflow:hidden}#status{position:absolute;inset:0;padding:16px;font:14px system-ui;box-sizing:border-box}</style>
        </head><body><div id="status">HTMLプレビューを読み込んでいます…</div><iframe id="preview" sandbox="allow-scripts allow-forms allow-modals allow-popups" referrerpolicy="no-referrer"></iframe>
        <script>
        const status=document.getElementById('status');
        const frame=document.getElementById('preview');
        const failed=()=>{status.textContent='HTMLプレビューを読み込めませんでした。閉じてからもう一度お試しください。';};
        window.addEventListener('error',failed);
        const timeout=setTimeout(failed,15000);
        frame.addEventListener('load',()=>{if(frame.hasAttribute('srcdoc')){clearTimeout(timeout);status.remove();}});
        const previewBase=${scriptString(base)};
        $security
        frame.srcdoc=sanitizeHtmlForPreview(${scriptString(code)});
        </script></body></html>""".trimIndent()
