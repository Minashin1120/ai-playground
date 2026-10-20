package com.minashin1120.aiplayground.ui

import android.graphics.Color
import android.view.View
import android.view.ViewGroup
import android.view.inspector.WindowInspector
import android.webkit.WebView
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.SdkSuppress
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.concurrent.atomic.AtomicReference

@RunWith(AndroidJUnit4::class)
// Run both a stable baseline and the SDK 37 environment from the report.
@SdkSuppress(minSdkVersion = 29)
class HtmlPreviewDeviceTest {
    @get:Rule val compose = createComposeRule()

    @Test fun htmlPreviewActuallyPaintsInsideTheComposeDialog() {
        compose.setContent {
            PlaygroundTheme {
                HtmlCodePreviewDialog("""<!doctype html><html><head><style>html,body{margin:0;width:100%;height:100%;background:rgb(0,255,0)}</style></head><body><h1>Preview 日本語</h1><script>setInterval(()=>{parent.postMessage('preview-painted','*');parent.postMessage({type:'preview-child',color:getComputedStyle(document.body).backgroundColor,w:innerWidth,h:innerHeight},'*')},100)</script></body></html>""") {}
            }
        }
        println("HTML_PREVIEW_DEVICE sdk=${android.os.Build.VERSION.SDK_INT} heap=${Runtime.getRuntime().maxMemory()}")
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        assertTrue("Device credential storage is not ready", instrumentation.targetContext.getSystemService(android.os.UserManager::class.java).isUserUnlocked)
        var webView: WebView? = null
        compose.waitUntil(15000) {
            instrumentation.runOnMainSync {
                fun find(view: View): WebView? {
                    if (view is WebView) return view
                    if (view is ViewGroup) for (i in 0 until view.childCount) find(view.getChildAt(i))?.let { return it }
                    return null
                }
                webView = WindowInspector.getGlobalWindowViews().firstNotNullOfOrNull { find(it) }
            }
            webView != null
        }
        val state = AtomicReference("")
        instrumentation.runOnMainSync {
            webView!!.evaluateJavascript("window.addEventListener('message',e=>{if(e.data==='preview-painted')window.previewPainted=true;if(e.data?.type==='preview-child')window.previewChild=e.data});'listening'", null)
        }
        try {
            compose.waitUntil(20000) {
                instrumentation.runOnMainSync {
                    webView!!.evaluateJavascript("if(!window.previewListener){window.previewListener=true;window.addEventListener('message',e=>{if(e.data==='preview-painted')window.previewPainted=true;if(e.data?.type==='preview-child')window.previewChild=e.data})};JSON.stringify({ready:window.previewPainted===true,w:innerWidth,h:innerHeight,status:document.getElementById('status')?.textContent,frames:frames.length,url:location.href,child:window.previewChild,frameHeight:document.getElementById('preview')?.getBoundingClientRect().height})") { state.set(it) }
                }
                println("HTML_PREVIEW_STATE=${state.get()}")
                state.get().contains("ready\\\":true")
            }
        } finally {
            val bitmap = instrumentation.uiAutomation.takeScreenshot()
            File(instrumentation.targetContext.getExternalFilesDir(null), "html-preview-attempt.png").outputStream().use {
                bitmap.compress(android.graphics.Bitmap.CompressFormat.PNG,100,it)
            }
            println("HTML_PREVIEW_FINAL=${state.get()}")
        }
        // WebView loading uses real time; finish the independent Compose animation clock as well.
        compose.mainClock.advanceTimeBy(1000)
        compose.waitForIdle()
        var lastBitmap: android.graphics.Bitmap? = null
        fun sample(): Int {
            val location = IntArray(2)
            var width = 0
            var height = 0
            instrumentation.runOnMainSync {
                webView!!.getLocationOnScreen(location)
                width = webView!!.width
                height = webView!!.height
            }
            assertTrue("WebView has no viewport", width > 100 && height > 100)
            val bitmap = instrumentation.uiAutomation.takeScreenshot()
            lastBitmap?.recycle()
            lastBitmap = bitmap
            return bitmap.getPixel(location[0] + width / 2, location[1] + height / 2)
        }
        fun isGreen(pixel: Int): Boolean = Color.green(pixel) > 230 && Color.red(pixel) < 20 && Color.blue(pixel) < 20
        fun waitForPaint(): Int {
            val deadline = android.os.SystemClock.uptimeMillis() + 3000
            var pixel: Int
            do {
                pixel = sample()
                if (isGreen(pixel)) break
                android.os.SystemClock.sleep(100)
            } while (android.os.SystemClock.uptimeMillis() < deadline)
            return pixel
        }
        fun save(name: String) {
            val values = android.content.ContentValues().apply {
                put(android.provider.MediaStore.Images.Media.DISPLAY_NAME, "html-preview-$name.png")
                put(android.provider.MediaStore.Images.Media.MIME_TYPE, "image/png")
                put(android.provider.MediaStore.Images.Media.RELATIVE_PATH, "Pictures/html-preview-device")
            }
            val resolver = instrumentation.targetContext.contentResolver
            val uri = resolver.insert(android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)!!
            resolver.openOutputStream(uri)!!.use { lastBitmap!!.compress(android.graphics.Bitmap.CompressFormat.PNG,100,it) }
        }
        instrumentation.runOnMainSync {
            println("HTML_PREVIEW_VIEW hardware=${webView!!.isHardwareAccelerated} layer=${webView!!.layerType} size=${webView!!.width}x${webView!!.height}")
        }
        val baseline = waitForPaint()
        println("HTML_PREVIEW_PAINT baseline=$baseline")
        save("baseline")
        if (!isGreen(baseline)) {
            instrumentation.runOnMainSync {
                webView!!.visibility = View.INVISIBLE
                webView!!.visibility = View.VISIBLE
                webView!!.onResume()
                webView!!.invalidate()
            }
            val visible = waitForPaint()
            println("HTML_PREVIEW_PAINT visibility=$visible")
            save("visibility")
            if (!isGreen(visible)) {
                instrumentation.runOnMainSync { webView!!.reload() }
                val reloaded = waitForPaint()
                println("HTML_PREVIEW_PAINT reload=$reloaded")
                save("reload")
                if (!isGreen(reloaded)) {
                    instrumentation.runOnMainSync {
                        webView!!.layoutParams = ViewGroup.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT)
                        webView!!.requestLayout()
                        webView!!.reload()
                    }
                    val layout = waitForPaint()
                    println("HTML_PREVIEW_PAINT matchParent=$layout")
                    save("layout")
                    if (!isGreen(layout)) {
                        instrumentation.runOnMainSync {
                            webView!!.webViewClient = android.webkit.WebViewClient()
                            webView!!.loadDataWithBaseURL("https://control.invalid/", "<html><body style='margin:0;background:rgb(0,255,0)'><h1>Direct control</h1></body></html>", "text/html", "UTF-8", null)
                        }
                        println("HTML_PREVIEW_PAINT direct=${waitForPaint()}")
                        save("direct")
                    }
                }
            }
        }
        lastBitmap?.recycle()
        assertTrue("WebView content was not painted: $baseline", isGreen(baseline))
    }
}
