package com.minashin1120.aiplayground

import android.app.NotificationManager
import android.content.Context
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment
import org.robolectric.Shadows.shadowOf
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [32])
class ToolbarNotificationTest {
    private val context: Context get() = RuntimeEnvironment.getApplication()
    private val manager: NotificationManager get() = context.getSystemService(NotificationManager::class.java)

    @Test fun firstToolbarActionIsImageSplit() {
        assertEquals("画像を分割", TOOLBAR_ACTIONS.first().label)
        assertEquals(ImageSplitOverlayActivity::class.java, TOOLBAR_ACTIONS.first().target)
    }

    @Test fun toolbarIsOffUntilSwitchedOn() {
        assertFalse(isToolbarNotificationEnabled(context))
        assertNull(shadowOf(manager).getNotification(TOOLBAR_NOTIFICATION_ID))
    }

    @Test fun switchingOnPostsAnOngoingNotificationWithEveryAction() {
        assertTrue(setToolbarNotificationEnabled(context, true))
        val notification = shadowOf(manager).getNotification(TOOLBAR_NOTIFICATION_ID)
        assertNotNull(notification)
        assertTrue(notification.flags and android.app.Notification.FLAG_ONGOING_EVENT != 0)
        assertEquals(TOOLBAR_ACTIONS.map { it.label }, notification.actions.map { it.title.toString() })
    }

    @Test fun tappingTheNotificationRunsTheFirstAction() {
        setToolbarNotificationEnabled(context, true)
        val notification = shadowOf(manager).getNotification(TOOLBAR_NOTIFICATION_ID)
        val tapped = shadowOf(notification.contentIntent).savedIntent
        assertEquals(TOOLBAR_ACTIONS.first().target.name, tapped.component?.className)
    }

    @Test fun switchingOffRemovesTheNotification() {
        setToolbarNotificationEnabled(context, true)
        assertTrue(setToolbarNotificationEnabled(context, false))
        assertFalse(isToolbarNotificationEnabled(context))
        assertNull(shadowOf(manager).getNotification(TOOLBAR_NOTIFICATION_ID))
    }
}
