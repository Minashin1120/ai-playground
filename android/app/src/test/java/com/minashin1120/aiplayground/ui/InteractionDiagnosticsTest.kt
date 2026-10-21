package com.minashin1120.aiplayground.ui

import android.os.Looper
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.BasicText
import androidx.compose.runtime.SideEffect
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.performTouchInput
import androidx.compose.ui.test.swipeLeft
import androidx.compose.ui.unit.dp
import com.minashin1120.aiplayground.data.ActivityLog
import org.junit.After
import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.Shadows.shadowOf
import org.robolectric.annotation.Config
import java.time.Duration
import java.util.UUID

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [35])
class InteractionDiagnosticsTest {
    @get:Rule val compose = createComposeRule()
    @get:Rule val temporary = TemporaryFolder()
    @After fun stopLogging() { ActivityLog.setEnabled(false) }

    @Test fun observingASwipePreservesScrollingAndRecordsItsResultWithoutText() {
        ActivityLog.initFile(temporary.newFile(), enabledAtStart = true)
        var observedScroll: androidx.compose.foundation.ScrollState? = null
        compose.setContent {
            val scroll = rememberScrollState()
            val trace = rememberInteractionTrace("math_block")
            SideEffect { observedScroll = scroll }
            Box(Modifier.width(120.dp).height(80.dp).testTag("formula")
                .observeInteraction(trace, scroll).horizontalScroll(scroll)) {
                BasicText("PRIVATE_PROMPT_ANSWER_TITLE_FILENAME_KEY", Modifier.width(800.dp))
            }
        }
        compose.onNodeWithTag("formula").performTouchInput { swipeLeft(durationMillis = 200) }
        compose.waitForIdle()
        assertTrue("observer must not consume the horizontal drag", observedScroll!!.value > 0)
        val entries = ActivityLog.recent()
        val events = (0 until entries.length()).map { entries.getJSONObject(it) }
        val start = events.first { it.optString("interaction_stage") == "press" && it.optString("phase") == "start" }
        val end = events.first { it.optString("interaction_stage") == "press" && it.optString("phase") == "done" }
        assertEquals(start.getString("op"), end.getString("op"))
        assertEquals(start.getString("gesture"), end.getString("gesture"))
        assertTrue(end.getInt("scroll_max") > 0)
        assertTrue(end.getInt("scrolled_px") > 0)
        assertFalse(entries.toString().contains("PRIVATE_PROMPT_ANSWER_TITLE_FILENAME_KEY"))
    }

    @Test fun delayedSnapshotCapturesToolbarResultAndStopsWhenDisposedOrDisabled() {
        ActivityLog.initFile(temporary.newFile(), enabledAtStart = true)
        val trace = InteractionTrace("message")
        trace.afterRelease(UUID.randomUUID().toString(), null)
        trace.toolbarRequests = 1
        trace.toolbarCreates = 1
        trace.actionModeActive = true
        shadowOf(Looper.getMainLooper()).idleFor(Duration.ofMillis(450))
        val snapshot = ActivityLog.recent().getJSONObject(0)
        assertEquals("settled", snapshot.getString("interaction_stage"))
        assertTrue(snapshot.getBoolean("action_mode_active"))
        assertEquals(1, snapshot.getInt("toolbar_requests"))

        ActivityLog.clear()
        trace.afterRelease(UUID.randomUUID().toString(), null)
        trace.active = false
        shadowOf(Looper.getMainLooper()).idleFor(Duration.ofMillis(450))
        assertEquals(0, ActivityLog.recent().length())

        trace.active = true
        trace.afterRelease(UUID.randomUUID().toString(), null)
        ActivityLog.setEnabled(false)
        shadowOf(Looper.getMainLooper()).idleFor(Duration.ofMillis(450))
        assertEquals(0, ActivityLog.recent().length())
    }
}
