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
                HtmlCodePreviewDialog("""<!doctype html><html><head><style>html,body{margin:0;width:100%;height:100%;background:rgb(0,255,0)}</style></head><body><h1>Preview 日本語</h1><script>setInterval(()=>parent.postMessage('preview-painted','*'),100)</script></body></html>""") {}
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
            webView!!.evaluateJavascript("window.addEventListener('message',e=>{if(e.data==='preview-painted')window.previewPainted=true});'listening'", null)
        }
        try {
            compose.waitUntil(20000) {
                instrumentation.runOnMainSync {
                    webView!!.evaluateJavascript("if(!window.previewListener){window.previewListener=true;window.addEventListener('message',e=>{if(e.data==='preview-painted')window.previewPainted=true})};JSON.stringify({ready:window.previewPainted===true,w:innerWidth,h:innerHeight,status:document.getElementById('status')?.textContent,frames:frames.length,url:location.href})") { state.set(it) }
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
        val location = IntArray(2)
        var width = 0
        var height = 0
        instrumentation.runOnMainSync {
            webView!!.getLocationOnScreen(location)
            width = webView!!.width
            height = webView!!.height
        }
        val screenshot = instrumentation.uiAutomation.takeScreenshot()
        val file = File(instrumentation.targetContext.getExternalFilesDir(null), "html-preview.png")
        file.outputStream().use { screenshot.compress(android.graphics.Bitmap.CompressFormat.PNG,100,it) }
        println("HTML_PREVIEW_STATE=${state.get()} view=${width}x$height location=${location.toList()}")
        assertTrue("WebView has no viewport", width > 100 && height > 100)
        val pixel = screenshot.getPixel(location[0]+width/2,location[1]+height/2)
        assertEquals("WebView content was not painted", Color.GREEN,pixel)
    }
}
