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
import android.os.PowerManager
import android.os.SystemClock
import androidx.core.app.NotificationCompat
import androidx.core.app.ServiceCompat
import androidx.core.content.ContextCompat
import com.minashin1120.aiplayground.data.local.GenerationKeepAlive
import com.minashin1120.aiplayground.data.local.GenerationTracker
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch

const val GENERATION_CHANNEL_ID = "generation"
private const val GENERATION_NOTIFICATION_ID = 0x5552
private const val ACTION_START = "com.minashin1120.aiplayground.action.START_GENERATION"
/** Upper bound for the CPU wake lock, in case the service is never told to stop. */
private const val GENERATION_WAKE_LOCK_MS = 60L * 60 * 1000

fun createGenerationNotificationChannel(context: Context) {
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
        val manager = context.getSystemService(NotificationManager::class.java)
        manager.createNotificationChannel(NotificationChannel(
            GENERATION_CHANNEL_ID,
            "回答の生成",
            NotificationManager.IMPORTANCE_LOW,
        ).apply { description = "回答を生成している間、アプリを離れても処理を続けるために表示します" })
    }
}

/**
 * Keeps the process in the foreground while an answer is generated or received, so Android neither
 * kills the app nor cuts its connection when the user switches to another app or turns the screen off.
 * The generation itself runs in the chat screen; the service only follows [tracker].
 */
class GenerationService : Service() {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    private var watchJob: Job? = null
    private var latestStartId = 0
    private var wakeLock: PowerManager.WakeLock? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        latestStartId = startId
        // startForegroundService() requires this call even when the generation already ended.
        try {
            ServiceCompat.startForeground(
                this, GENERATION_NOTIFICATION_ID, notification(tracker.active.value),
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC else 0,
            )
        } catch (_: Exception) {
            // Android refused the foreground service; the generation still runs while the app is open.
            stopSelf(startId)
            return START_NOT_STICKY
        }
        if (foregroundSince == 0L) foregroundSince = SystemClock.elapsedRealtime()
        holdCpu()
        if (watchJob?.isActive != true) watchJob = scope.launch { watch() }
        // A killed process loses the generation, so there is nothing to restart.
        return START_NOT_STICKY
    }

    private suspend fun watch() {
        tracker.active.collect { modes ->
            if (modes.isEmpty()) {
                leaveForeground()
                // Only stops when no newer start request is waiting.
                stopSelf(latestStartId)
                watchJob?.cancel()
            } else {
                postNotification(notification(modes))
            }
        }
    }

    private fun notification(modes: List<String>): Notification {
        val title = when {
            "video" in modes -> "動画を生成中"
            "image" in modes -> "画像を生成中"
            else -> "回答を生成中"
        }
        val open = PendingIntent.getActivity(
            this, 0, Intent(this, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        return NotificationCompat.Builder(this, GENERATION_CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_playground)
            .setContentTitle(title)
            .setContentText("完了するまでアプリを終了しないでください")
            .setProgress(0, 0, true)
            .setContentIntent(open)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .setSilent(true)
            .setForegroundServiceBehavior(NotificationCompat.FOREGROUND_SERVICE_IMMEDIATE)
            .build()
    }

    private fun postNotification(notification: Notification) {
        if (Build.VERSION.SDK_INT >= 33 && ContextCompat.checkSelfPermission(
                this, Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) return
        getSystemService(NotificationManager::class.java).notify(GENERATION_NOTIFICATION_ID, notification)
    }

    /** The screen going off must not suspend the CPU while the answer streams in. */
    private fun holdCpu() {
        if (wakeLock?.isHeld == true) return
        wakeLock = try {
            getSystemService(PowerManager::class.java)
                .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "AIPlayground:generation")
                .apply { setReferenceCounted(false); acquire(GENERATION_WAKE_LOCK_MS) }
        } catch (_: Exception) { null }
    }

    private fun leaveForeground() {
        foregroundSince = 0L
        wakeLock?.let { lock -> if (lock.isHeld) runCatching { lock.release() } }
        wakeLock = null
        ServiceCompat.stopForeground(this, ServiceCompat.STOP_FOREGROUND_REMOVE)
    }

    // Android 15+ limits dataSync services; stop the notification but let the generation go on in the app.
    override fun onTimeout(startId: Int, fgsType: Int) {
        leaveForeground()
        stopSelf()
    }

    override fun onDestroy() {
        leaveForeground()
        scope.cancel()
        super.onDestroy()
    }

    companion object {
        /** The generations running in this process; the service lives while this is not empty. */
        private val tracker = GenerationTracker()

        /** `elapsedRealtime` when the service entered the foreground, 0 while it is not there. */
        @Volatile private var foregroundSince = 0L

        /** The service has kept the app running without a break since [time] (`elapsedRealtime`). */
        fun keptRunningSince(time: Long): Boolean = foregroundSince.let { it != 0L && it <= time }

        /** Starts the foreground service for one generation; close the result when the answer is done. */
        fun keepAlive(context: Context): GenerationKeepAlive {
            val app = context.applicationContext
            return GenerationKeepAlive { mode ->
                val hold = tracker.begin(mode)
                // If Android refuses the foreground service the generation still runs while the app is open.
                try {
                    ContextCompat.startForegroundService(app, Intent(app, GenerationService::class.java).setAction(ACTION_START))
                } catch (_: Exception) {
                }
                hold
            }
        }
    }
}
