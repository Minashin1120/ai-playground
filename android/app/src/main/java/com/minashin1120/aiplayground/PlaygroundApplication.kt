package com.minashin1120.aiplayground

import android.app.Activity
import android.app.Application
import android.os.Bundle
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
        Diagnostics.init(this)
        useQuoteSelectionToolbar()
        createBatchNotificationChannel(this)
        createAppUpdateNotificationChannel(this)
        createGenerationNotificationChannel(this)
        createToolbarNotificationChannel(this)
        refreshToolbarNotification(this)
        registerActivityLifecycleCallbacks(object : ActivityLifecycleCallbacks {
            override fun onActivityStarted(activity: Activity) { startedActivities++ }
            override fun onActivityStopped(activity: Activity) { startedActivities = (startedActivities - 1).coerceAtLeast(0) }
            override fun onActivityCreated(activity: Activity, savedInstanceState: Bundle?) = Unit
            override fun onActivityResumed(activity: Activity) = Unit
            override fun onActivityPaused(activity: Activity) = Unit
            override fun onActivitySaveInstanceState(activity: Activity, outState: Bundle) = Unit
            override fun onActivityDestroyed(activity: Activity) = Unit
        })
    }
}
