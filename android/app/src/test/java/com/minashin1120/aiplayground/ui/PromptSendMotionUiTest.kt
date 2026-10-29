package com.minashin1120.aiplayground.ui

import androidx.compose.foundation.layout.*
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.isRoot
import androidx.compose.ui.test.onFirst
import androidx.compose.ui.unit.dp
import com.github.takahirom.roborazzi.captureRoboImage
import com.minashin1120.aiplayground.ChatState
import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import org.robolectric.annotation.GraphicsMode

@RunWith(RobolectricTestRunner::class)
@GraphicsMode(GraphicsMode.Mode.NATIVE)
@Config(sdk = [35], qualifiers = "w411dp-h923dp-xxhdpi")
class PromptSendMotionUiTest {
    @get:Rule val rule = createComposeRule()
    private var state by mutableStateOf(ChatState())
    private lateinit var motion: PromptSendMotion
    private val composedFlights = mutableListOf<Triple<String?, String?, Float>>()

    private var hostHeight by mutableStateOf(800)

    private fun render(longPrompt: Boolean = false) {
        rule.mainClock.autoAdvance = false
        rule.setContent {
            CompositionLocalProvider(LocalReduceMotion provides false) {
                Box(Modifier.height(hostHeight.dp)) {
                PromptSendMotionHost(state) {
                    motion = checkNotNull(LocalPromptSendMotion.current)
                    composedFlights += Triple(state.promptSendMotionId, motion.id, motion.progress.value)
                    SideEffect {
                        motion.source = Rect(0f, 600f, 300f, 650f)
                        motion.viewport = Rect(0f, 0f, 1000f, 1000f)
                    }
                    if (state.promptSendMotionId != null) {
                        Column(Modifier.wrapContentHeight(unbounded = true).padding(20.dp).width(200.dp).then(promptSendBubbleModifier(checkNotNull(state.promptSendMotionId)))) {
                            Text(if (longPrompt) (1..80).joinToString("\n") { "長いプロンプト $it" } else "本文\n2行目")
                            Text("引用と添付ファイル")
                        }
                    }
                }
                }
            }
        }
        rule.mainClock.advanceTimeByFrame()
        rule.waitForIdle()
    }

    @Test fun drawingIsRecordedAndReleasedAfterTheFlight() {
        render()
        rule.runOnIdle { state = state.copy(promptSendMotionId = "local-test") }
        repeat(4) { rule.mainClock.advanceTimeByFrame(); rule.waitForIdle() }
        // Robolectric advances composition/layout independently of Android's draw pass.
        // Exercise the actual drawWithContent/GraphicsLayer recording before checking it.
        rule.onAllNodes(isRoot()).onFirst().captureRoboImage("build/outputs/roborazzi/prompt_send_flight.png")
        rule.mainClock.advanceTimeByFrame()
        rule.waitForIdle()
        rule.runOnIdle {
            assertEquals("local-test", motion.id)
            assertTrue(motion.ready)
            assertNotNull(motion.layer)
            hostHeight = 500
        }
        repeat(2) { rule.mainClock.advanceTimeByFrame(); rule.waitForIdle() }
        rule.runOnIdle { assertEquals("local-test", motion.id) }
        rule.mainClock.advanceTimeBy(450L)
        rule.waitForIdle()
        rule.runOnIdle { assertNull(motion.id); assertNull(motion.layer) }
    }

    @Test fun keyboardResizeKeepsTheFlightForALongPrompt() {
        // A height-only root resize is what adjustResize/IME animation delivers.
        render(longPrompt = true)
        rule.runOnIdle { state = state.copy(promptSendMotionId = "local-test") }
        repeat(4) { rule.mainClock.advanceTimeByFrame(); rule.waitForIdle() }
        rule.onAllNodes(isRoot()).onFirst().captureRoboImage("build/outputs/roborazzi/prompt_send_long.png")
        rule.mainClock.advanceTimeByFrame()
        rule.waitForIdle()
        rule.runOnIdle { assertTrue(motion.ready); hostHeight = 500 }
        repeat(2) { rule.mainClock.advanceTimeByFrame(); rule.waitForIdle() }
        rule.runOnIdle { assertEquals("local-test", motion.id); assertNotNull(motion.layer) }
        rule.mainClock.advanceTimeBy(450L)
        rule.waitForIdle()
        rule.runOnIdle { assertNull(motion.id); assertNull(motion.layer) }
    }

    @Test fun firstCompositionHidesTheDestinationForInitialAndSubsequentSends() {
        render(longPrompt = true)
        fun assertPreparedBeforeRow(request: String) {
            val first = composedFlights.first { it.first == request }
            assertEquals("The row must be hidden in its first composition", request, first.second)
            assertEquals("A new send must start at the composer", 0f, first.third, 0f)
        }
        rule.runOnIdle { state = state.copy(promptSendMotionId = "local-test") }
        rule.mainClock.advanceTimeByFrame()
        rule.waitForIdle()
        rule.runOnIdle { assertPreparedBeforeRow("local-test") }
        rule.onAllNodes(isRoot()).onFirst().captureRoboImage("build/outputs/roborazzi/prompt_send_first_frame.png")
        rule.mainClock.advanceTimeBy(500L)
        rule.waitForIdle()
        rule.runOnIdle {
            assertNull(motion.id)
            assertEquals(1f, motion.progress.value, 0f)
            state = state.copy(promptSendMotionId = "local-second")
        }
        rule.mainClock.advanceTimeByFrame()
        rule.waitForIdle()
        rule.runOnIdle { assertPreparedBeforeRow("local-second") }
        rule.onAllNodes(isRoot()).onFirst().captureRoboImage("build/outputs/roborazzi/prompt_send_second_first_frame.png")
        rule.mainClock.advanceTimeBy(500L)
        rule.waitForIdle()
        rule.runOnIdle { assertNull(motion.id) }
    }

    @Test fun navigationCancelsAndDoesNotReplayTheRequest() {
        render()
        rule.runOnIdle { state = state.copy(promptSendMotionId = "local-test") }
        repeat(4) { rule.mainClock.advanceTimeByFrame(); rule.waitForIdle() }
        rule.onAllNodes(isRoot()).onFirst().captureRoboImage("build/outputs/roborazzi/prompt_send_cancel.png")
        rule.runOnIdle { state = state.copy(chatNavigationId = state.chatNavigationId + 1L) }
        rule.mainClock.advanceTimeByFrame()
        rule.waitForIdle()
        rule.runOnIdle { assertNull(motion.id); assertNull(motion.layer) }
        rule.mainClock.advanceTimeBy(500L)
        rule.waitForIdle()
        rule.runOnIdle { assertNull(motion.id) }
    }
}
