package com.minashin1120.aiplayground

import android.content.ClipData
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.os.Build
import android.provider.Settings
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.browser.customtabs.CustomTabsIntent
import androidx.core.content.FileProvider
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.minashin1120.aiplayground.ui.PlaygroundScreen

class MainActivity : ComponentActivity() {
    private val model: ChatViewModel by viewModels()
    private val updateModel: AppUpdateViewModel by viewModels()
    private val changelogModel: AppChangelogViewModel by viewModels()
    private val installerLauncher = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) {
        updateModel.installerClosed()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        handleBubbleIntent(intent)
        setContent {
            val updateState = updateModel.state.collectAsStateWithLifecycle().value
            val changelogState = changelogModel.state.collectAsStateWithLifecycle().value
            PlaygroundScreen(model, appUpdate = updateState,
                appChangelog = changelogState,
                playStartupAnimation = savedInstanceState == null,
                onDismissUpdate = updateModel::dismiss,
                onDownloadUpdate = updateModel::startDownload,
                onCancelDownload = updateModel::cancelDownload,
                onRetryUpdate = updateModel::retryDownload,
                onInstallUpdate = ::installUpdate,
                onCheckForUpdate = { updateModel.check(BuildConfig.VERSION_NAME) },
                onOpenChangelog = changelogModel::load,
                onRetryChangelog = changelogModel::load,
                onOpenBubble = { createChatBubble(this, model.state.value.selected) },
                onWeb = { path ->
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

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        handleBubbleIntent(intent)
    }

    private fun handleBubbleIntent(intent: Intent?) {
        if (intent?.action != Intent.ACTION_VIEW || !intent.hasExtra(EXTRA_BUBBLE_THREAD_ID)) return
        val threadId = intent.getStringExtra(EXTRA_BUBBLE_THREAD_ID)
        intent.removeExtra(EXTRA_BUBBLE_THREAD_ID)
        model.openBubbleTarget(threadId)
    }

    private fun installUpdate() {
        val file = updateModel.state.value.readyFile
        if (file == null || !file.isFile) {
            updateModel.installFailed("更新ファイルが見つかりません。もう一度ダウンロードしてください。")
            return
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O && !packageManager.canRequestPackageInstalls()) {
            updateModel.awaitInstallPermission()
            try {
                startActivity(Intent(Settings.ACTION_MANAGE_UNKNOWN_APP_SOURCES, Uri.parse("package:$packageName")))
            } catch (_: Exception) {
                updateModel.installFailed("Androidのインストール設定を開けません。端末の設定から許可してください。")
            }
            return
        }
        val uri = FileProvider.getUriForFile(this, "$packageName.files", file)
        val intent = Intent(Intent.ACTION_INSTALL_PACKAGE)
            .setDataAndType(uri, "application/vnd.android.package-archive")
            .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        intent.clipData = ClipData.newRawUri("AI Playground update", uri)
        updateModel.markInstalling()
        try {
            installerLauncher.launch(intent)
        } catch (_: Exception) {
            updateModel.installFailed("Androidのインストーラーを開けません。端末の設定を確認してください。")
        }
    }

    private fun openExternalUrl(url: String) {
        try { CustomTabsIntent.Builder().build().launchUrl(this, Uri.parse(url)) }
        catch (_: Exception) { model.notify("ブラウザーを開けません。ブラウザーをインストールして再試行してください。") }
    }

    override fun onStart() { super.onStart(); model.setForeground(true) }
    override fun onStop() { model.setForeground(false); super.onStop() }
    override fun onResume() {
        super.onResume()
        updateModel.check(BuildConfig.VERSION_NAME)
        if (updateModel.state.value.phase == AppUpdatePhase.AwaitingInstallPermission &&
            (Build.VERSION.SDK_INT < Build.VERSION_CODES.O || packageManager.canRequestPackageInstalls())) {
            installUpdate()
        }
    }
}
