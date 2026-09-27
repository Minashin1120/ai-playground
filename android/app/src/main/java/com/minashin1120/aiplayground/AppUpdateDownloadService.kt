package com.minashin1120.aiplayground

import android.Manifest
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import android.os.SystemClock
import androidx.core.app.NotificationCompat
import androidx.core.app.ServiceCompat
import androidx.core.content.ContextCompat
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch

const val APP_UPDATE_CHANNEL_ID = "app_update"
const val EXTRA_SHOW_APP_UPDATE = "com.minashin1120.aiplayground.SHOW_APP_UPDATE"
private const val PROGRESS_NOTIFICATION_ID = 0x5550
private const val RESULT_NOTIFICATION_ID = 0x5551
private const val ACTION_START = "com.minashin1120.aiplayground.action.START_APP_UPDATE_DOWNLOAD"
private const val ACTION_CANCEL = "com.minashin1120.aiplayground.action.CANCEL_APP_UPDATE_DOWNLOAD"

fun createAppUpdateNotificationChannel(context: Context) {
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
        val manager = context.getSystemService(NotificationManager::class.java)
        manager.createNotificationChannel(NotificationChannel(
            APP_UPDATE_CHANNEL_ID,
            "アプリ更新",
            NotificationManager.IMPORTANCE_LOW,
        ).apply { description = "Android版の更新ファイルのダウンロード状況を表示します" })
    }
}

/** Removes the "ready" or "failed" notification once the app shows the update dialog itself. */
fun clearAppUpdateResultNotification(context: Context) {
    context.getSystemService(NotificationManager::class.java).cancel(RESULT_NOTIFICATION_ID)
}

/**
 * Keeps the process in the foreground while [AppUpdateDownloadManager] downloads the APK, so the
 * user can use other apps meanwhile. The service only mirrors the manager state into a
 * notification; the download itself belongs to the manager.
 */
class AppUpdateDownloadService : Service() {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    private var watchJob: Job? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val manager = (application as PlaygroundApplication).appUpdates
        when (intent?.action) {
            ACTION_START -> {
                // startForegroundService() requires this call even when the download already ended.
                ServiceCompat.startForeground(
                    this, PROGRESS_NOTIFICATION_ID, progressNotification(manager.state.value),
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC else 0,
                )
                clearAppUpdateResultNotification(this)
                if (watchJob?.isActive != true) watchJob = scope.launch { watch(manager) }
            }
            ACTION_CANCEL -> {
                manager.cancelDownload()
                if (watchJob?.isActive != true) stopSelf()
            }
            else -> if (watchJob?.isActive != true) stopSelf()
        }
        // A killed process loses the download, so there is nothing to restart.
        return START_NOT_STICKY
    }

    private suspend fun watch(manager: AppUpdateDownloadManager) {
        var lastPercent = -1L
        var lastPostedAt = 0L
        var finished: AppUpdateUiState
        do {
            finished = waitUntilFinished(manager) { state, percent, now ->
                if (percent != lastPercent && now - lastPostedAt >= 500L) {
                    lastPercent = percent
                    lastPostedAt = now
                    postNotification(PROGRESS_NOTIFICATION_ID, progressNotification(state))
                }
            }
            // A retry started right as the previous download ended reuses this running service.
        } while (manager.state.value.phase == AppUpdatePhase.Downloading)
        ServiceCompat.stopForeground(this, ServiceCompat.STOP_FOREGROUND_REMOVE)
        // In the foreground the dialog reappears by itself; outside the app, leave a way back.
        if (!(application as PlaygroundApplication).isInForeground) {
            when (finished.phase) {
                AppUpdatePhase.Ready -> postNotification(RESULT_NOTIFICATION_ID, resultNotification(
                    "更新ファイルの準備ができました", "タップしてAI Playground ${finished.update?.versionName.orEmpty()} をインストールします。"))
                AppUpdatePhase.Error -> postNotification(RESULT_NOTIFICATION_ID, resultNotification(
                    "Android版を更新できません", finished.errorMessage ?: "更新ファイルを取得できませんでした。"))
                else -> Unit
            }
        }
        stopSelf()
    }

    /** Posting a notification for every buffer read would be throttled by Android, so callers rate-limit. */
    private suspend fun waitUntilFinished(
        manager: AppUpdateDownloadManager,
        onProgress: (AppUpdateUiState, Long, Long) -> Unit,
    ): AppUpdateUiState = manager.state.first { state ->
        if (state.phase != AppUpdatePhase.Downloading) return@first true
        val percent = state.totalBytes?.takeIf { it > 0L }?.let { state.downloadedBytes * 100L / it } ?: -1L
        onProgress(state, percent, SystemClock.elapsedRealtime())
        false
    }

    private fun progressNotification(state: AppUpdateUiState): Notification {
        val total = state.totalBytes?.takeIf { it > 0L }
        val percent = total?.let { (state.downloadedBytes * 100L / it).coerceIn(0L, 100L).toInt() }
        val cancel = PendingIntent.getService(
            this, 1, Intent(this, AppUpdateDownloadService::class.java).setAction(ACTION_CANCEL),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        return NotificationCompat.Builder(this, APP_UPDATE_CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_playground)
            .setContentTitle("Android版をダウンロード中")
            .setContentText(percent?.let { "$it%" } ?: "AI Playground ${state.update?.versionName.orEmpty()}")
            .setProgress(100, percent ?: 0, percent == null)
            .setContentIntent(openAppIntent())
            .addAction(0, "キャンセル", cancel)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .setSilent(true)
            .setForegroundServiceBehavior(NotificationCompat.FOREGROUND_SERVICE_IMMEDIATE)
            .build()
    }

    private fun resultNotification(title: String, text: String): Notification =
        NotificationCompat.Builder(this, APP_UPDATE_CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_playground)
            .setContentTitle(title)
            .setContentText(text)
            .setStyle(NotificationCompat.BigTextStyle().bigText(text))
            .setContentIntent(openAppIntent())
            .setAutoCancel(true)
            .build()

    /** Opens the app on the update dialog; the verified file is read from the manager, never the intent. */
    private fun openAppIntent(): PendingIntent {
        val intent = Intent(this, MainActivity::class.java)
            .addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP)
            .putExtra(EXTRA_SHOW_APP_UPDATE, true)
        return PendingIntent.getActivity(this, 0, intent, PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE)
    }

    private fun postNotification(id: Int, notification: Notification) {
        if (Build.VERSION.SDK_INT >= 33 && ContextCompat.checkSelfPermission(
                this, Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) return
        getSystemService(NotificationManager::class.java).notify(id, notification)
    }

    // Android 15+ limits dataSync services; stop the notification but let the in-app download go on.
    override fun onTimeout(startId: Int, fgsType: Int) {
        ServiceCompat.stopForeground(this, ServiceCompat.STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    override fun onDestroy() {
        scope.cancel()
        super.onDestroy()
    }

    companion object {
        /** Returns false when Android refuses the foreground service; the download still runs in the app. */
        fun start(context: Context): Boolean = try {
            ContextCompat.startForegroundService(
                context, Intent(context, AppUpdateDownloadService::class.java).setAction(ACTION_START),
            )
            true
        } catch (_: Exception) {
            false
        }
    }
}
