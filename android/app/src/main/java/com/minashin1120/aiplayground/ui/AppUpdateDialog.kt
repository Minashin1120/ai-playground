package com.minashin1120.aiplayground.ui

import androidx.compose.material3.AlertDialog
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.ui.unit.dp
import androidx.compose.ui.Modifier
import com.minashin1120.aiplayground.AppUpdatePhase
import com.minashin1120.aiplayground.AppUpdateUiState

@Composable
fun AppUpdateDialog(
    state: AppUpdateUiState,
    onDismiss: () -> Unit,
    onDownload: () -> Unit,
    onCancelDownload: () -> Unit,
    onRetry: () -> Unit,
    onInstall: () -> Unit,
) {
    val update = state.update ?: return
    val downloading = state.phase == AppUpdatePhase.Downloading
    val installing = state.phase == AppUpdatePhase.Installing
    val progress = state.totalBytes?.takeIf { it > 0L }?.let {
        (state.downloadedBytes.toFloat() / it).coerceIn(0f, 1f)
    }
    AlertDialog(
        onDismissRequest = { if (downloading) onCancelDownload() else onDismiss() },
        title = {
            Text(
                when (state.phase) {
                    AppUpdatePhase.Downloading -> "Android版をダウンロード中"
                    AppUpdatePhase.Ready -> "更新ファイルの準備ができました"
                    AppUpdatePhase.AwaitingInstallPermission -> "インストール許可が必要です"
                    AppUpdatePhase.Installing -> "Android版を更新しています"
                    AppUpdatePhase.Error -> "Android版を更新できません"
                    AppUpdatePhase.UpToDate -> "Android版は最新です"
                    else -> "新しいAndroid版があります"
                }
            )
        },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("AI Playground ${update.versionName} が利用できます。")
                when (state.phase) {
                    AppUpdatePhase.Available -> Text("GitHubからAPKを直接取得して、Androidの標準インストーラーで更新します。")
                    AppUpdatePhase.Downloading -> {
                        Text(formatProgress(state.downloadedBytes, state.totalBytes))
                        if (progress == null) LinearProgressIndicator(Modifier.fillMaxWidth())
                        else LinearProgressIndicator(progress = { progress }, modifier = Modifier.fillMaxWidth())
                    }
                    AppUpdatePhase.Ready -> Text("取得したAPKの検証が完了しました。インストールを開始できます。")
                    AppUpdatePhase.AwaitingInstallPermission -> Text("Android設定で「このアプリからのインストール」を許可してください。戻るとインストールを続けます。")
                    AppUpdatePhase.Installing -> Text("Androidの確認画面が表示されます。画面を閉じた場合は、もう一度インストールを選べます。")
                    AppUpdatePhase.Error -> Text(state.errorMessage ?: "通信状態を確認して、もう一度お試しください。")
                    AppUpdatePhase.UpToDate -> Unit
                    AppUpdatePhase.Checking -> Unit
                }
            }
        },
        confirmButton = {
            when (state.phase) {
                AppUpdatePhase.Available -> TextButton(onClick = onDownload) { Text("ダウンロード") }
                AppUpdatePhase.Downloading -> TextButton(onClick = onCancelDownload) { Text("キャンセル") }
                AppUpdatePhase.Ready -> TextButton(onClick = onInstall) { Text("インストール") }
                AppUpdatePhase.AwaitingInstallPermission -> TextButton(onClick = onInstall) { Text("設定を開く") }
                AppUpdatePhase.Installing -> Unit
                AppUpdatePhase.Error -> TextButton(onClick = onRetry) { Text("再試行") }
                AppUpdatePhase.UpToDate -> Unit
                AppUpdatePhase.Checking -> Unit
            }
        },
        dismissButton = {
            if (!installing) TextButton(onClick = { if (downloading) onCancelDownload() else onDismiss() }) { Text("後で") }
        },
    )
}

private fun formatProgress(downloaded: Long, total: Long?): String {
    if (total == null || total <= 0L) return "${formatBytes(downloaded)}を取得しました。"
    val percent = (downloaded * 100L / total).coerceIn(0L, 100L)
    return "$percent%（${formatBytes(downloaded)} / ${formatBytes(total)}）"
}

private fun formatBytes(bytes: Long): String = when {
    bytes >= 1024L * 1024L -> "%.1f MB".format(bytes / (1024f * 1024f))
    bytes >= 1024L -> "%.1f KB".format(bytes / 1024f)
    else -> "$bytes B"
}
