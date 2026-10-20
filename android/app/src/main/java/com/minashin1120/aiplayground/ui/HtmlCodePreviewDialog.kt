package com.minashin1120.aiplayground.ui

import android.annotation.SuppressLint
import android.webkit.CookieManager
import android.webkit.WebChromeClient
import android.webkit.WebResourceRequest
import android.webkit.WebResourceResponse
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.viewinterop.AndroidView
import com.minashin1120.aiplayground.data.ServerOrigin
import org.json.JSONObject
import java.io.ByteArrayInputStream

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
        """<!doctype html><html><head><meta charset="utf-8">
            <meta name="referrer" content="no-referrer">
            <meta name="viewport" content="width=device-width,initial-scale=1">
            <style>html,body,iframe{width:100%;height:100%;margin:0;border:0;background:#fff}body{overflow:hidden}</style>
            </head><body><iframe id="preview" sandbox="allow-scripts allow-forms allow-modals allow-popups" referrerpolicy="no-referrer"></iframe>
            <script>const previewBase=${scriptString(base)};
            $security
            document.getElementById('preview').srcdoc=sanitizeHtmlForPreview(${scriptString(code)});
            </script></body></html>""".trimIndent()
    }
    PlaygroundDialog(
        onDismissRequest = onDismiss,
        title = { Text("プレビュー") },
        confirmButton = {},
        fillHeight = true,
        padBody = false,
        showFooter = false,
        text = {
        AndroidView(
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
                    // The outer document has no server origin; iframe navigation cannot replace it.
                    webViewClient = object : WebViewClient() {
                        override fun shouldOverrideUrlLoading(view: WebView, request: WebResourceRequest): Boolean =
                            request.isForMainFrame || request.url.scheme !in setOf("https", "http", "about", "data", "blob")

                        override fun shouldInterceptRequest(view: WebView, request: WebResourceRequest): WebResourceResponse? {
                            val host = request.url.host.orEmpty().lowercase()
                            if (host == "polyfill.io" || host.endsWith(".polyfill.io") ||
                                request.url.scheme !in setOf("https", "http", "about", "data", "blob")) {
                                return WebResourceResponse("text/plain", "UTF-8", ByteArrayInputStream(ByteArray(0)))
                            }
                            return null
                        }
                    }
                    loadDataWithBaseURL(null, document, "text/html", "UTF-8", null)
                }
            },
            onRelease = { view -> view.stopLoading(); view.destroy() },
        )
        },
    )
}
