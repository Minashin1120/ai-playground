package com.minashin1120.aiplayground

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import androidx.browser.customtabs.CustomTabsIntent
import androidx.core.content.FileProvider
import com.minashin1120.aiplayground.ui.PlaygroundScreen

class MainActivity : ComponentActivity() {
    private val model: ChatViewModel by viewModels()
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            PlaygroundScreen(model, onWeb = { pairing ->
                val url = BuildConfig.BASE_URL + if (pairing) "android/connect" else ""
                try { CustomTabsIntent.Builder().build().launchUrl(this, Uri.parse(url)) }
                catch (_: Exception) { model.notify("ブラウザーを開けません。ブラウザーをインストールして再試行してください。") }
            }, onFile = { reference ->
                model.openFile(reference) { file, mime ->
                    val uri = FileProvider.getUriForFile(this, "$packageName.files", file)
                    val intent = Intent(Intent.ACTION_VIEW).setDataAndType(uri, mime)
                        .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                    try { startActivity(Intent.createChooser(intent, "添付を開く")) }
                    catch (_: Exception) { model.notify("この添付を開けるアプリが見つかりません。") }
                }
            })
        }
    }
    override fun onStart() { super.onStart(); model.setForeground(true) }
    override fun onStop() { model.setForeground(false); super.onStop() }
}
