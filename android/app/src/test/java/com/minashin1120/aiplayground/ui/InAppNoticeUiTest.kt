package com.minashin1120.aiplayground.ui

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxScope
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.semantics.SemanticsActions
import androidx.compose.ui.test.assertCountEquals
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.junit4.StateRestorationTester
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onAllNodesWithText
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performSemanticsAction
import androidx.compose.ui.test.performTextInput
import androidx.compose.ui.test.performTouchInput
import androidx.compose.ui.test.swipeDown
import androidx.compose.ui.test.swipeUp
import androidx.compose.ui.unit.Density
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotSame
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import org.robolectric.annotation.GraphicsMode

@RunWith(RobolectricTestRunner::class)
@GraphicsMode(GraphicsMode.Mode.NATIVE)
@Config(sdk = [35], qualifiers = "w411dp-h923dp-xxhdpi")
class InAppNoticeUiTest {
    @get:Rule val rule = createComposeRule()
    private lateinit var state: InAppNoticeState

    private fun render(
        dark: Boolean = false,
        reduce: Boolean = false,
        largeText: Boolean = false,
        content: @Composable BoxScope.() -> Unit = {},
    ) {
        rule.mainClock.autoAdvance = false
        rule.setContent {
            val density = LocalDensity.current
            CompositionLocalProvider(
                LocalDensity provides if (largeText) Density(density.density, 1.8f) else density,
                LocalReduceMotion provides reduce,
            ) {
                PlaygroundTheme(darkTheme = dark) {
                    InAppNoticeProvider {
                        state = checkNotNull(LocalInAppNotices.current)
                        Box(Modifier.fillMaxSize(), content = content)
                    }
                }
            }
        }
        settle()
    }

    private fun settle() {
        rule.mainClock.advanceTimeBy(500L)
        rule.waitForIdle()
    }

    @Test
    fun newestAppearsAboveOlderAndBothSwipeDirectionsDismissOnlyTheirOwnMessage() {
        render()
        rule.runOnIdle { state.add("古い通知", 60_000L) }
        settle()
        rule.runOnIdle { state.add("新しい通知", 60_000L) }
        settle()
        val newestY = rule.onNodeWithText("新しい通知").fetchSemanticsNode().boundsInRoot.top
        val olderY = rule.onNodeWithText("古い通知").fetchSemanticsNode().boundsInRoot.top
        assertTrue(newestY < olderY)

        // Continue beyond the original card bounds, as a finger can when dismissing a banner.
        rule.onNodeWithText("新しい通知").performTouchInput { swipeUp(endY = -height.toFloat()) }
        settle()
        rule.onNodeWithText("新しい通知").assertDoesNotExist()
        rule.onNodeWithText("古い通知").assertIsDisplayed()
        rule.onNodeWithText("古い通知").performTouchInput { swipeDown(endY = height * 3f) }
        settle()
        rule.onNodeWithText("古い通知").assertDoesNotExist()
    }

    @Test
    fun openingAndClosingNestedDialogsKeepsOneNoticeLayerAndDoesNotStealTextInput() {
        var settings by mutableStateOf(false)
        var confirm by mutableStateOf(false)
        var input by mutableStateOf("")
        render {
            if (settings) PlaygroundDialog(
                onDismissRequest = { settings = false },
                title = { Text("設定") },
                text = { OutlinedTextField(input, { input = it }, Modifier.testTag("notice_test_input")) },
                confirmButton = {},
            )
            if (confirm) BrowserConfirmDialog("確認") { confirm = false }
        }
        lateinit var original: InAppNotice
        val mainWindow = state.activeWindow
        rule.runOnIdle {
            state.add("モーダルの前面の通知", 60_000L)
            original = state.notices.single()
            settings = true
        }
        settle()
        val settingsWindow = state.activeWindow
        assertNotSame(mainWindow, settingsWindow)
        rule.onAllNodesWithText(original.message).assertCountEquals(1)
        rule.onNodeWithTag("notice_test_input").performClick().performTextInput("入力できます")
        rule.runOnIdle {
            assertEquals("入力できます", input)
            confirm = true
        }
        settle()
        assertNotSame(settingsWindow, state.activeWindow)
        rule.onAllNodesWithText(original.message).assertCountEquals(1)
        rule.runOnIdle { confirm = false }
        settle()
        assertSame(settingsWindow, state.activeWindow)
        rule.runOnIdle { settings = false }
        settle()
        assertSame(mainWindow, state.activeWindow)
        rule.onAllNodesWithText(original.message).assertCountEquals(1)
        rule.runOnIdle { assertEquals(original.expiresAtMillis, state.notices.single().expiresAtMillis) }
    }

    @OptIn(ExperimentalMaterial3Api::class)
    @Test
    fun bottomSheetAlsoRegistersItsWindowAndHandsBackToTheMainScreen() {
        var sheet by mutableStateOf(false)
        render {
            if (sheet) ModalBottomSheet(onDismissRequest = { sheet = false }) {
                NoticeWindowHost()
                Text("添付シート")
            }
        }
        val mainWindow = state.activeWindow
        rule.runOnIdle { state.add("シート上の通知", 60_000L); sheet = true }
        settle()
        assertNotSame(mainWindow, state.activeWindow)
        rule.onAllNodesWithText("シート上の通知").assertCountEquals(1)
        rule.runOnIdle { sheet = false }
        settle()
        assertSame(mainWindow, state.activeWindow)
        rule.onNodeWithText("シート上の通知").assertIsDisplayed()
    }

    @Test
    fun noticeAreaDoesNotBlockButtonsElsewhereAndSupportsAccessibleDismissWithMotionDisabled() {
        var clicked = false
        render(reduce = true) {
            Button(onClick = { clicked = true }, modifier = Modifier.align(Alignment.BottomCenter)) { Text("画面の操作") }
        }
        rule.runOnIdle { state.add("読み上げられる通知", 60_000L) }
        settle()
        rule.onNodeWithText("画面の操作").performClick()
        rule.runOnIdle { assertTrue(clicked) }
        rule.onNodeWithText("読み上げられる通知").performSemanticsAction(SemanticsActions.Dismiss) { it() }
        settle()
        rule.onNodeWithText("読み上げられる通知").assertDoesNotExist()
    }

    @Test
    fun lightThemeSupportsLongMultilineMessagesAndLargeText() {
        checkLongMessage(dark = false)
    }

    @Test
    fun darkThemeSupportsLongMultilineMessagesAndLargeText() {
        checkLongMessage(dark = true)
    }

    private fun checkLongMessage(dark: Boolean) {
        val message = "通知は省略せず、文字サイズを大きくした場合も折り返して表示します。\nモーダルの前面でも読むことができます。"
        render(dark = dark, largeText = true)
        rule.runOnIdle { state.add(message, 60_000L) }
        settle()
        rule.onNodeWithText(message).assertIsDisplayed()
        rule.onAllNodesWithText(message).assertCountEquals(1)
    }

    @Test
    fun savedInstanceStateKeepsMessagesAndTheirOriginalExpiryAfterRecreation() {
        rule.mainClock.autoAdvance = false
        val restoration = StateRestorationTester(rule)
        restoration.setContent {
            PlaygroundTheme {
                InAppNoticeProvider {
                    state = checkNotNull(LocalInAppNotices.current)
                    Box(Modifier.fillMaxSize())
                }
            }
        }
        settle()
        rule.runOnIdle { state.add("回転後も表示する通知", 60_000L) }
        settle()
        val original = state.notices.single()
        restoration.emulateSavedInstanceStateRestore()
        settle()
        rule.onAllNodesWithText(original.message).assertCountEquals(1)
        rule.runOnIdle {
            assertEquals(original.id, state.notices.single().id)
            assertEquals(original.expiresAtMillis, state.notices.single().expiresAtMillis)
        }
    }
}
