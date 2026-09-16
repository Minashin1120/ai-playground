package com.minashin1120.aiplayground

import android.app.Application

class PlaygroundApplication : Application() {
    override fun onCreate() {
        super.onCreate()
        createBatchNotificationChannel(this)
    }
}
