package com.minashin1120.aiplayground

import android.Manifest
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import androidx.core.app.NotificationCompat
import androidx.core.content.ContextCompat

const val TOOLBAR_CHANNEL_ID = "toolbar"
const val TOOLBAR_NOTIFICATION_ID = 4102

private const val TOOLBAR_PREFS = "toolbar_notification"
private const val TOOLBAR_ENABLED_KEY = "enabled"

/** One button of the notification-shade toolbar. Add new toolbar features here. */
internal class ToolbarAction(val label: String, val iconRes: Int, val target: Class<*>)

/** Same entry as the Quick Settings tile: a translucent screen that keeps the current app behind it. */
internal val TOOLBAR_ACTIONS: List<ToolbarAction> = listOf(
    ToolbarAction("画像を分割", R.drawable.fa_solid_image, ImageSplitOverlayActivity::class.java),
)

fun createToolbarNotificationChannel(context: Context) {
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
        context.getSystemService(NotificationManager::class.java).createNotificationChannel(NotificationChannel(
            TOOLBAR_CHANNEL_ID,
            "ツールバー",
            NotificationManager.IMPORTANCE_LOW,
        ).apply {
            description = "通知パネルにAI Playgroundのツールバーを常に表示します"
            setShowBadge(false)
            lockscreenVisibility = Notification.VISIBILITY_SECRET
        })
    }
}

fun isToolbarNotificationEnabled(context: Context): Boolean =
    context.getSharedPreferences(TOOLBAR_PREFS, Context.MODE_PRIVATE).getBoolean(TOOLBAR_ENABLED_KEY, false)

private fun canPostNotifications(context: Context): Boolean =
    Build.VERSION.SDK_INT < 33 ||
        ContextCompat.checkSelfPermission(context, Manifest.permission.POST_NOTIFICATIONS) == PackageManager.PERMISSION_GRANTED

/** Remembers the choice and posts or removes the toolbar. Returns false when the notification permission is missing. */
fun setToolbarNotificationEnabled(context: Context, enabled: Boolean): Boolean {
    if (enabled && !canPostNotifications(context)) return false
    context.getSharedPreferences(TOOLBAR_PREFS, Context.MODE_PRIVATE).edit().putBoolean(TOOLBAR_ENABLED_KEY, enabled).apply()
    refreshToolbarNotification(context)
    return true
}

private fun actionIntent(context: Context, index: Int): PendingIntent {
    val intent = Intent(context, TOOLBAR_ACTIONS[index].target).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP)
    return PendingIntent.getActivity(context, 100 + index, intent, PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE)
}

/**
 * Posts the toolbar when it is switched on (and permitted), otherwise removes it. Safe to call repeatedly.
 * Tapping the notification body runs the first action, so it works as a button too.
 */
fun refreshToolbarNotification(context: Context) {
    val manager = context.getSystemService(NotificationManager::class.java)
    if (!isToolbarNotificationEnabled(context) || !canPostNotifications(context)) {
        manager.cancel(TOOLBAR_NOTIFICATION_ID)
        return
    }
    createToolbarNotificationChannel(context)
    val builder = NotificationCompat.Builder(context, TOOLBAR_CHANNEL_ID)
        .setSmallIcon(R.drawable.ic_playground)
        .setContentTitle("AI Playground")
        .setContentText("タップして${TOOLBAR_ACTIONS.first().label}")
        .setContentIntent(actionIntent(context, 0))
        .setOngoing(true)
        .setOnlyAlertOnce(true)
        .setShowWhen(false)
        .setPriority(NotificationCompat.PRIORITY_LOW)
        .setVisibility(NotificationCompat.VISIBILITY_SECRET)
    TOOLBAR_ACTIONS.forEachIndexed { index, action ->
        builder.addAction(action.iconRes, action.label, actionIntent(context, index))
    }
    try {
        manager.notify(TOOLBAR_NOTIFICATION_ID, builder.build())
    } catch (ignored: SecurityException) {
        // The permission was revoked between the check and the post; the next refresh removes it.
    }
}

/** Restores the toolbar after a reboot or an app update, when the user has it switched on. */
class ToolbarBootReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action == Intent.ACTION_BOOT_COMPLETED || intent.action == Intent.ACTION_MY_PACKAGE_REPLACED) {
            refreshToolbarNotification(context)
        }
    }
}
