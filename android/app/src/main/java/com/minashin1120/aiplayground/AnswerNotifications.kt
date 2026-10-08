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

const val ANSWER_CHANNEL_ID = "answer_completion"
private const val ANSWER_NOTIFICATION_ID = 0x5553

fun createAnswerNotificationChannel(context: Context) {
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
        context.getSystemService(NotificationManager::class.java).createNotificationChannel(NotificationChannel(
            ANSWER_CHANNEL_ID,
            "回答の完了",
            NotificationManager.IMPORTANCE_DEFAULT,
        ).apply { description = "アプリを離れている間に回答の生成が終わったときに通知します" })
    }
}

fun answerNotificationTitle(failed: Boolean): String =
    if (failed) "回答の生成でエラーが発生しました" else "回答が完了しました"

/** Only one answer streams at a time, so a newer notification replaces the previous one. */
fun notifyAnswerFinished(context: Context, chatTitle: String, failed: Boolean) {
    if (Build.VERSION.SDK_INT >= 33 && ContextCompat.checkSelfPermission(
            context, Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) return
    // The finished chat stays open in the app, so the notification only brings the app back.
    val open = PendingIntent.getActivity(context, ANSWER_NOTIFICATION_ID,
        Intent(context, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP),
        PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE)
    val notification = NotificationCompat.Builder(context, ANSWER_CHANNEL_ID)
        .setSmallIcon(R.drawable.ic_playground)
        .setContentTitle(answerNotificationTitle(failed))
        .apply { if (chatTitle.isNotBlank()) setContentText(chatTitle) }
        .setContentIntent(open)
        .setAutoCancel(true)
        .build()
    try { context.getSystemService(NotificationManager::class.java).notify(ANSWER_NOTIFICATION_ID, notification) }
    catch (_: Exception) { }
}

/** Back in the app, the answer is on screen and the notification has nothing more to say. */
fun clearAnswerNotification(context: Context) {
    try { context.getSystemService(NotificationManager::class.java).cancel(ANSWER_NOTIFICATION_ID) }
    catch (_: Exception) { }
}
