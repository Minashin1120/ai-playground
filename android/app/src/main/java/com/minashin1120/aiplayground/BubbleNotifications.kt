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
import androidx.core.app.Person
import androidx.core.content.ContextCompat
import androidx.core.content.pm.ShortcutInfoCompat
import androidx.core.content.pm.ShortcutManagerCompat
import androidx.core.graphics.drawable.IconCompat
import com.minashin1120.aiplayground.data.ThreadItem

const val CHAT_BUBBLE_PARENT_CHANNEL_ID = "chat_bubble"
const val CHAT_BUBBLE_CHANNEL_ID = "chat_bubble_conversation"
const val CHAT_BUBBLE_NOTIFICATION_ID = 4101
const val CHAT_BUBBLE_SHORTCUT_ID = "chat_bubble"
const val EXTRA_BUBBLE_THREAD_ID = "com.minashin1120.aiplayground.extra.BUBBLE_THREAD_ID"

private const val BUBBLE_SHORTCUT_CATEGORY = "android.shortcut.conversation"

enum class ChatBubbleResult {
    POSTED,
    SETTINGS_REQUIRED,
}

internal fun bubbleThreadId(value: String?): String? = value?.trim()?.takeIf { it.isNotEmpty() }

internal fun bubbleTitle(thread: ThreadItem?): String =
    thread?.title?.trim()?.takeIf { it.isNotEmpty() } ?: "新しいチャット"

fun createChatBubble(context: Context, thread: ThreadItem?): ChatBubbleResult {
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU &&
        ContextCompat.checkSelfPermission(context, Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) {
        return ChatBubbleResult.SETTINGS_REQUIRED
    }

    val notificationManager = context.getSystemService(NotificationManager::class.java)
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S &&
        notificationManager.bubblePreference == NotificationManager.BUBBLE_PREFERENCE_NONE) {
        return ChatBubbleResult.SETTINGS_REQUIRED
    }
    createChatBubbleChannel(context)
    val threadId = bubbleThreadId(thread?.id)
    val title = bubbleTitle(thread)
    val targetIntent = Intent(context, MainActivity::class.java).apply {
        action = Intent.ACTION_VIEW
        putExtra(EXTRA_BUBBLE_THREAD_ID, threadId)
        addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP)
    }
    val pendingIntent = PendingIntent.getActivity(
        context,
        CHAT_BUBBLE_NOTIFICATION_ID,
        targetIntent,
        PendingIntent.FLAG_UPDATE_CURRENT or bubblePendingIntentMutability(),
    )

    try {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            val shortcut = ShortcutInfoCompat.Builder(context, CHAT_BUBBLE_SHORTCUT_ID)
                .setShortLabel(title.take(30))
                .setLongLabel("AI Playground: $title".take(80))
                .setIcon(IconCompat.createWithResource(context, R.drawable.ic_playground_mark))
                .setIntent(targetIntent)
                .setLongLived(true)
                .setCategories(setOf(BUBBLE_SHORTCUT_CATEGORY))
                .build()
            ShortcutManagerCompat.pushDynamicShortcut(context, shortcut)
        }

        postChatNotification(context, title, pendingIntent, bubbleMetadata = true)
        return ChatBubbleResult.POSTED
    } catch (_: RuntimeException) {
        // Some OEM notification providers reject bubble metadata even when the
        // app and OS support it. Keep the user action useful without crashing.
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N_MR1) {
            ShortcutManagerCompat.removeDynamicShortcuts(context, listOf(CHAT_BUBBLE_SHORTCUT_ID))
        }
        runCatching { postChatNotification(context, title, pendingIntent, bubbleMetadata = false) }
        return ChatBubbleResult.POSTED
    }
}

fun cancelChatBubble(context: Context) {
    context.getSystemService(NotificationManager::class.java)
        .cancel(CHAT_BUBBLE_NOTIFICATION_ID)
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N_MR1) {
        ShortcutManagerCompat.removeDynamicShortcuts(context, listOf(CHAT_BUBBLE_SHORTCUT_ID))
    }
}

private fun createChatBubbleChannel(context: Context) {
    if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return
    val notificationManager = context.getSystemService(NotificationManager::class.java)
    val parentChannel = NotificationChannel(
        CHAT_BUBBLE_PARENT_CHANNEL_ID,
        "チャットバブル",
        NotificationManager.IMPORTANCE_DEFAULT,
    ).apply {
        description = "AI Playgroundのチャットをバブルで開きます"
    }
    notificationManager.createNotificationChannel(parentChannel)
    val conversationChannel = NotificationChannel(
        CHAT_BUBBLE_CHANNEL_ID,
        "現在のチャット",
        NotificationManager.IMPORTANCE_DEFAULT,
    ).apply {
        description = "AI Playgroundの現在のチャットをバブルで開きます"
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) setAllowBubbles(true)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            setConversationId(CHAT_BUBBLE_PARENT_CHANNEL_ID, CHAT_BUBBLE_SHORTCUT_ID)
        }
    }
    notificationManager.createNotificationChannel(conversationChannel)
}

private fun bubblePendingIntentMutability(): Int =
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) PendingIntent.FLAG_MUTABLE
    else PendingIntent.FLAG_IMMUTABLE

private fun postChatNotification(
    context: Context,
    title: String,
    pendingIntent: PendingIntent,
    bubbleMetadata: Boolean,
) {
    val metadata = when {
        !bubbleMetadata -> null
        Build.VERSION.SDK_INT >= Build.VERSION_CODES.R ->
            NotificationCompat.BubbleMetadata.Builder(CHAT_BUBBLE_SHORTCUT_ID)
                .setDesiredHeight(640)
                .setAutoExpandBubble(true)
                .build()
        Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q ->
            NotificationCompat.BubbleMetadata.Builder(
                pendingIntent,
                IconCompat.createWithResource(context, R.drawable.ic_playground_mark),
            )
                .setDesiredHeight(640)
                .setAutoExpandBubble(true)
                .build()
        else -> null
    }
    val person = Person.Builder().setName("AI Playground").setImportant(true).build()
    val messagingStyle = NotificationCompat.MessagingStyle(person)
        .addMessage(NotificationCompat.MessagingStyle.Message("タップしてチャットを開く", System.currentTimeMillis(), person))
    val notification = NotificationCompat.Builder(context, CHAT_BUBBLE_CHANNEL_ID)
        .setSmallIcon(R.drawable.ic_playground)
        .setContentTitle(title)
        .setContentText("タップしてチャットを開く")
        .setContentIntent(pendingIntent)
        .setCategory(NotificationCompat.CATEGORY_MESSAGE)
        .addPerson(person)
        .setStyle(messagingStyle)
        .setOnlyAlertOnce(true)
        .setAutoCancel(false)
        .setOngoing(false)
        .apply {
            if (bubbleMetadata && Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                setShortcutId(CHAT_BUBBLE_SHORTCUT_ID)
            }
            metadata?.let { setBubbleMetadata(it) }
        }
        .build()
    context.getSystemService(NotificationManager::class.java)
        .notify(CHAT_BUBBLE_NOTIFICATION_ID, notification)
}
