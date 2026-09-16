package com.minashin1120.aiplayground

import android.Manifest
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import androidx.core.app.NotificationCompat
import androidx.core.content.ContextCompat
import com.minashin1120.aiplayground.data.BatchJob
import com.minashin1120.aiplayground.data.batchStateLabel

const val BATCH_CHANNEL_ID = "batch_completion"

fun createBatchNotificationChannel(context: Context) {
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
        val manager = context.getSystemService(NotificationManager::class.java)
        manager.createNotificationChannel(NotificationChannel(
            BATCH_CHANNEL_ID,
            "Batch処理の完了",
            NotificationManager.IMPORTANCE_DEFAULT,
        ).apply { description = "AI PlaygroundのBatch処理が完了したときに通知します" })
    }
}

fun notifyBatchCompletion(context: Context, job: BatchJob) {
    if (Build.VERSION.SDK_INT >= 33 && ContextCompat.checkSelfPermission(
            context, Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) return
    val intent = Intent(context, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP)
    val pending = PendingIntent.getActivity(context, job.id.hashCode(), intent,
        PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE)
    val notification = NotificationCompat.Builder(context, BATCH_CHANNEL_ID)
        .setSmallIcon(com.minashin1120.aiplayground.R.drawable.ic_playground)
        .setContentTitle("Batch処理: ${batchStateLabel(job)}")
        .setContentText(job.threadTitle)
        .setStyle(NotificationCompat.BigTextStyle().bigText("${job.threadTitle}\n${job.model} · ${batchStateLabel(job)}"))
        .setContentIntent(pending)
        .setAutoCancel(true)
        .build()
    context.getSystemService(NotificationManager::class.java).notify(job.id.hashCode(), notification)
}
