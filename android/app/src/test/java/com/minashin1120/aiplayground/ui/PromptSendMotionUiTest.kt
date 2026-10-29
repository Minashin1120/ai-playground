package com.minashin1120.aiplayground.ui

import androidx.compose.foundation.layout.*
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.unit.dp
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

    private fun render() {
        rule.mainClock.autoAdvance = false
        rule.setContent {
            CompositionLocalProvider(LocalReduceMotion provides false) {
                PromptSendMotionHost(state) {
                    motion = checkNotNull(LocalPromptSendMotion.current)
                    SideEffect {
                        motion.source = Rect(0f, 600f, 300f, 650f)
                        motion.viewport = Rect(0f, 0f, 1000f, 1000f)
                    }
                    if (state.promptSendMotionId != null) {
                        Column(Modifier.padding(20.dp).width(200.dp).then(promptSendBubbleModifier("local-test"))) {
                            Text("本文\n2行目")
                            Text("引用と添付ファイル")
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
        rule.runOnIdle {
            assertEquals("local-test", motion.id)
            assertTrue(motion.ready)
            assertNotNull(motion.layer)
        }
        rule.mainClock.advanceTimeBy(450L)
        rule.waitForIdle()
        rule.runOnIdle { assertNull(motion.id); assertNull(motion.layer) }
    }

    @Test fun navigationCancelsAndDoesNotReplayTheRequest() {
        render()
        rule.runOnIdle { state = state.copy(promptSendMotionId = "local-test") }
        repeat(4) { rule.mainClock.advanceTimeByFrame(); rule.waitForIdle() }
        rule.runOnIdle { state = state.copy(chatNavigationId = state.chatNavigationId + 1L) }
        rule.mainClock.advanceTimeByFrame()
        rule.waitForIdle()
        rule.runOnIdle { assertNull(motion.id); assertNull(motion.layer) }
        rule.mainClock.advanceTimeBy(500L)
        rule.waitForIdle()
        rule.runOnIdle { assertNull(motion.id) }
    }
}
