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
import androidx.compose.runtime.mutableStateOf
import com.minashin1120.aiplayground.data.AppUpdate
import com.minashin1120.aiplayground.data.AppUpdateChecker
import com.minashin1120.aiplayground.ui.PlaygroundScreen

class MainActivity : ComponentActivity() {
    private val model: ChatViewModel by viewModels()
    private val updateChecker = AppUpdateChecker()
    private val availableUpdate = mutableStateOf<AppUpdate?>(null)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        updateChecker.check(BuildConfig.VERSION_NAME) { update ->
            runOnUiThread {
                if (!isFinishing && !isDestroyed) availableUpdate.value = update
            }
        }
        setContent {
            PlaygroundScreen(model, appUpdate = availableUpdate.value,
                onDismissUpdate = { availableUpdate.value = null }, onOpenUpdate = { update ->
                    availableUpdate.value = null
                    openExternalUrl(update.releaseUrl)
                }, onWeb = { path ->
                val safePath = path.takeIf { it.startsWith('/') && !it.startsWith("//") } ?: "/"
                val url = BuildConfig.BASE_URL.trimEnd('/') + safePath
                openExternalUrl(url)
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

    private fun openExternalUrl(url: String) {
        try { CustomTabsIntent.Builder().build().launchUrl(this, Uri.parse(url)) }
        catch (_: Exception) { model.notify("ブラウザーを開けません。ブラウザーをインストールして再試行してください。") }
    }

    override fun onStart() { super.onStart(); model.setForeground(true) }
    override fun onStop() { model.setForeground(false); super.onStop() }
    override fun onDestroy() { updateChecker.cancel(); super.onDestroy() }
}
