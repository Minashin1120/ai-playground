package com.minashin1120.aiplayground

import android.app.Activity
import android.app.Application
import android.os.Bundle
import com.minashin1120.aiplayground.data.ActivityLog
import com.minashin1120.aiplayground.data.AppUpdateDownloader
import com.minashin1120.aiplayground.data.Diagnostics
import com.minashin1120.aiplayground.ui.useQuoteSelectionToolbar

class PlaygroundApplication : Application() {
    /** Owns the APK update so a download survives the dialog, the activity and leaving the app. */
    val appUpdates: AppUpdateDownloadManager by lazy {
        val downloader = AppUpdateDownloader()
        AppUpdateDownloadManager(cacheDir, { update, directory, onProgress -> downloader.download(update, directory, onProgress) })
    }

    private var startedActivities = 0
    val isInForeground: Boolean get() = startedActivities > 0

    override fun onCreate() {
        super.onCreate()
        ActivityLog.init(this)
        Diagnostics.init(this)
        useQuoteSelectionToolbar()
        createBatchNotificationChannel(this)
        createAppUpdateNotificationChannel(this)
        createGenerationNotificationChannel(this)
        createAnswerNotificationChannel(this)
        createToolbarNotificationChannel(this)
        refreshToolbarNotification(this)
        ActivityLog.log("app.start", "version" to BuildConfig.VERSION_NAME, "sdk" to android.os.Build.VERSION.SDK_INT,
            "device" to android.os.Build.MODEL, "memory" to Diagnostics.memory())
        registerActivityLifecycleCallbacks(object : ActivityLifecycleCallbacks {
            // ログの収集を強化: which screen (activity) came and went.
            private fun record(event: String, activity: Activity) =
                ActivityLog.log("activity.$event", "activity" to activity.javaClass.simpleName)
            override fun onActivityStarted(activity: Activity) { startedActivities++; record("started", activity) }
            override fun onActivityStopped(activity: Activity) {
                startedActivities = (startedActivities - 1).coerceAtLeast(0)
                record("stopped", activity)
            }
            override fun onActivityCreated(activity: Activity, savedInstanceState: Bundle?) {
                ActivityLog.log("activity.created", "activity" to activity.javaClass.simpleName,
                    "restored" to (savedInstanceState != null), "action" to activity.intent?.action)
            }
            override fun onActivityResumed(activity: Activity) = record("resumed", activity)
            override fun onActivityPaused(activity: Activity) = record("paused", activity)
            override fun onActivitySaveInstanceState(activity: Activity, outState: Bundle) = Unit
            override fun onActivityDestroyed(activity: Activity) = record("destroyed", activity)
        })
    }
}
