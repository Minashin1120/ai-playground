package com.minashin1120.aiplayground

import android.content.ClipData
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.os.Build
import android.os.Parcelable
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
        handleAuthIntent(intent)
        handleShareIntent(intent)
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
                onOpenBubble = { openBubble() },
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
                    catch (ignored: Exception) { model.notify("この添付を開けるアプリが見つかりません。") }
                }
            })
        }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        handleBubbleIntent(intent)
        handleAuthIntent(intent)
        handleShareIntent(intent)
    }

    private fun handleBubbleIntent(intent: Intent?) {
        if (intent?.action != Intent.ACTION_VIEW || !intent.hasExtra(EXTRA_BUBBLE_THREAD_ID)) return
        val threadId = intent.getStringExtra(EXTRA_BUBBLE_THREAD_ID)
        intent.removeExtra(EXTRA_BUBBLE_THREAD_ID)
        model.openBubbleTarget(threadId)
    }

    private fun handleAuthIntent(intent: Intent?) {
        val data = intent?.data ?: return
        if (data.scheme != "https" || data.host != "ai.minashin1120.com" || data.path != "/android/auth/callback") return
        intent?.data = null
        data.getQueryParameter("integrity_ticket")?.let { model.integrityTurnstileComplete(it); return }
        data.getQueryParameter("turnstile_ticket")?.let { model.completeSessionTurnstile(it); return }
        data.getQueryParameter("linked")?.let { model.linkCompleted(it); return }
        when (data.getQueryParameter("error")) {
            // Same messages as the Web settings link flow.
            "google_already_linked" -> { model.notify("この Google アカウントは既に他のユーザーに紐付けられています。"); return }
            "minashin_already_linked" -> { model.notify("この Minashin アカウントは既に他のユーザーに紐付けられています。"); return }
        }
        data.getQueryParameter("code")?.let { model.exchangeNativeCode(it); return }
        data.getQueryParameter("error")?.let { model.notify("外部ログインに失敗しました。($it)") }
    }

    /** Handles files shared from other apps via the system share sheet (Intent.ACTION_SEND[_MULTIPLE]). */
    private fun handleShareIntent(intent: Intent?) {
        val uris: List<Uri> = when (intent?.action) {
            Intent.ACTION_SEND -> intent.parcelableExtraCompat<Uri>(Intent.EXTRA_STREAM)?.let { listOf(it) } ?: emptyList()
            Intent.ACTION_SEND_MULTIPLE -> intent.parcelableArrayListExtraCompat<Uri>(Intent.EXTRA_STREAM) ?: emptyList()
            else -> emptyList()
        }
        if (uris.isEmpty()) return
        intent?.action = null
        intent?.removeExtra(Intent.EXTRA_STREAM)
        model.upload(uris)
    }

    private fun openBubble() {
        when (createChatBubble(this, model.state.value.selected)) {
            ChatBubbleResult.ANDROID_17_USER_ACTION -> {
                model.notify("Android 17では、ホーム画面でAI Playgroundのアイコンを長押しし、「バブルに追加」を選択してください。")
            }
            ChatBubbleResult.SETTINGS_REQUIRED -> {
                try {
                    startActivity(Intent(Settings.ACTION_APP_NOTIFICATION_SETTINGS).apply {
                        putExtra(Settings.EXTRA_APP_PACKAGE, packageName)
                    })
                } catch (ignored: Exception) {
                    model.notify("Androidの通知設定を開けません。設定からAI Playgroundのバブルを許可してください。")
                }
            }
            ChatBubbleResult.POSTED -> Unit
        }
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
            } catch (ignored: Exception) {
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
        } catch (ignored: Exception) {
            updateModel.installFailed("Androidのインストーラーを開けません。端末の設定を確認してください。")
        }
    }

    private fun openExternalUrl(url: String) {
        try { CustomTabsIntent.Builder().build().launchUrl(this, Uri.parse(url)) }
        catch (ignored: Exception) { model.notify("ブラウザーを開けません。ブラウザーをインストールして再試行してください。") }
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

@Suppress("DEPRECATION")
private inline fun <reified T : Parcelable> Intent.parcelableExtraCompat(name: String): T? =
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) getParcelableExtra(name, T::class.java)
    else getParcelableExtra(name)

@Suppress("DEPRECATION")
private inline fun <reified T : Parcelable> Intent.parcelableArrayListExtraCompat(name: String): ArrayList<T>? =
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) getParcelableArrayListExtra(name, T::class.java)
    else getParcelableArrayListExtra(name)
