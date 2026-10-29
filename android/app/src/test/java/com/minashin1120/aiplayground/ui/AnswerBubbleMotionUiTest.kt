package com.minashin1120.aiplayground.ui

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.size
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.unit.dp
import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [35])
class AnswerBubbleMotionUiTest {
    @get:Rule val rule = createComposeRule()
    private val prompt = PromptSendMotion()
    private val answers = AnswerBubbleMotions()
    private var revision by mutableStateOf(0)
    private val progress get() = answers.entry("live-0", animate = true)

    private fun render(animate: Boolean = true, reduce: Boolean = false) {
        rule.mainClock.autoAdvance = false
        prompt.source = Rect(0f, 500f, 300f, 550f)
        prompt.begin("local-send", reduce)
        rule.setContent {
            CompositionLocalProvider(LocalReduceMotion provides reduce, LocalPromptSendMotion provides prompt) {
                // A persisted answer reuses the live row's stable key, even if its content is recomposed.
                key(revision) {
                    ConversationMessageEntry("live-0", answer = true, animate = animate, answers = answers) {
                        Box(Modifier.size(200.dp, 80.dp)) { Text("回答バブル") }
                    }
                }
            }
        }
        repeat(4) { rule.mainClock.advanceTimeByFrame(); rule.waitForIdle() }
    }

    @Test fun answerWaitsForPromptAndSlidesOnlyAfterArrival() {
        render()
        rule.mainClock.advanceTimeBy(500L)
        rule.waitForIdle()
        rule.runOnIdle { assertEquals(0f, progress.value, 0f) }
        rule.onNodeWithText("回答バブル").assertDoesNotExist()
        rule.runOnIdle { prompt.finish() }
        repeat(4) { rule.mainClock.advanceTimeByFrame(); rule.waitForIdle() }
        rule.mainClock.advanceTimeBy(100L)
        rule.waitForIdle()
        rule.runOnIdle { assertTrue(progress.value > 0f && progress.value < 1f) }
        rule.onNodeWithText("回答バブル").assertExists()
        rule.mainClock.advanceTimeBy(400L)
        rule.waitForIdle()
        rule.runOnIdle { assertEquals(1f, progress.value, 0f) }
    }

    @Test fun fastStoredAnswerKeepsWaitingAndDoesNotReplayAfterHandoff() {
        render()
        val original = progress
        rule.runOnIdle { revision++ }
        repeat(4) { rule.mainClock.advanceTimeByFrame(); rule.waitForIdle() }
        rule.runOnIdle { assertSame(original, progress); assertEquals(0f, progress.value, 0f) }
        rule.runOnIdle { prompt.finish() }
        repeat(4) { rule.mainClock.advanceTimeByFrame(); rule.waitForIdle() }
        rule.mainClock.advanceTimeBy(500L)
        rule.waitForIdle()
        rule.runOnIdle { assertEquals(1f, progress.value, 0f); revision++ }
        repeat(4) { rule.mainClock.advanceTimeByFrame(); rule.waitForIdle() }
        rule.runOnIdle { assertEquals(1f, progress.value, 0f) }
    }

    @Test fun existingHistoryIsNotHiddenByANewSend() {
        render(animate = false)
        rule.runOnIdle { assertEquals(1f, progress.value, 0f) }
        rule.onNodeWithText("回答バブル").assertExists()
    }

    @Test fun reducedMotionShowsTheAnswerImmediately() {
        render(reduce = true)
        rule.runOnIdle { assertEquals(1f, progress.value, 0f) }
        rule.onNodeWithText("回答バブル").assertExists()
    }
}
