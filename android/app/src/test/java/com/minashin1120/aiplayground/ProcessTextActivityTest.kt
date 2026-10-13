package com.minashin1120.aiplayground

import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.Robolectric
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment
import org.robolectric.Shadows.shadowOf
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class ProcessTextActivityTest {
    @Test fun textSelectionMenuOffersPromptAttach() {
        val pm = RuntimeEnvironment.getApplication().packageManager
        val intent = Intent(Intent.ACTION_PROCESS_TEXT).setType("text/plain")
            .addCategory(Intent.CATEGORY_DEFAULT)
        val ours = pm.queryIntentActivities(intent, PackageManager.MATCH_DEFAULT_ONLY)
            .filter { it.activityInfo.name == ProcessTextActivity::class.java.name }
        assertEquals(1, ours.size)
        assertEquals("プロンプトとして添付", ours.single().loadLabel(pm).toString())
    }

    @Test fun shareSheetDoesNotUseTheSelectionEntry() {
        val pm = RuntimeEnvironment.getApplication().packageManager
        val intent = Intent(Intent.ACTION_SEND).setType("text/plain").addCategory(Intent.CATEGORY_DEFAULT)
        val names = pm.queryIntentActivities(intent, PackageManager.MATCH_DEFAULT_ONLY)
            .map { it.activityInfo.name }
        assertTrue(ProcessTextActivity::class.java.name !in names)
    }

    @Test fun forwardsSelectionIntoTheSharePathWithoutReplacingIt() {
        val activity = Robolectric.buildActivity(ProcessTextActivity::class.java, processText(" 選んだ文 ")).create().get()
        assertTrue(activity.isFinishing)
        assertEquals(Activity.RESULT_CANCELED, shadowOf(activity).resultCode)
        val next = shadowOf(activity).nextStartedActivity
        assertEquals(MainActivity::class.java.name, next.component?.className)
        assertEquals(Intent.ACTION_SEND, next.action)
        assertEquals("text/plain", next.type)
        assertEquals("選んだ文", next.getStringExtra(Intent.EXTRA_TEXT))
        assertTrue(next.flags and Intent.FLAG_ACTIVITY_NEW_TASK != 0)
        assertTrue(next.flags and Intent.FLAG_ACTIVITY_CLEAR_TOP != 0)
        assertTrue(next.flags and Intent.FLAG_ACTIVITY_SINGLE_TOP != 0)
    }

    @Test fun blankSelectionDoesNotOpenTheApp() {
        val activity = Robolectric.buildActivity(ProcessTextActivity::class.java, processText(" \n ")).create().get()
        assertTrue(activity.isFinishing)
        assertEquals(Activity.RESULT_CANCELED, shadowOf(activity).resultCode)
        assertNull(shadowOf(activity).nextStartedActivity)
    }

    private fun processText(text: String) = Intent(Intent.ACTION_PROCESS_TEXT).apply {
        type = "text/plain"
        putExtra(Intent.EXTRA_PROCESS_TEXT, text)
    }
}
