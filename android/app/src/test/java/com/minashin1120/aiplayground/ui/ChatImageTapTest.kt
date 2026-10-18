package com.minashin1120.aiplayground.ui

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.width
import androidx.compose.ui.Modifier
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onAllNodesWithContentDescription
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.performClick
import androidx.compose.ui.unit.dp
import com.minashin1120.aiplayground.data.ChatMessage
import org.junit.Assert.assertEquals
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import org.robolectric.annotation.GraphicsMode
import java.util.Base64

/**
 * A tap on an image in the chat opens the image viewer. The attachment grid is inside the bubble's
 * selection container, so this checks that the grid's own click still reaches the cell.
 */
@RunWith(RobolectricTestRunner::class)
@GraphicsMode(GraphicsMode.Mode.NATIVE)
@Config(sdk = [35], qualifiers = "w411dp-h923dp-xxhdpi")
class ChatImageTapTest {
    @get:Rule val rule = createComposeRule()

    /** A 1×1 red PNG, so the protected image reaches its loaded state in the test. */
    private val png = Base64.getDecoder().decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC",
    )

    @Test
    fun tappingAGridImageOpensTheViewer() {
        var opened: String? = null
        // More than four images use the grid's auto-fill branch, like the reported 19 generated images.
        val files = listOf("1/a.jpg", "1/b.jpg", "1/c.jpg", "1/d.jpg", "1/e.jpg")
        rule.setContent {
            PlaygroundTheme(darkTheme = true) {
                Box(Modifier.width(411.dp)) {
                    MessageBubble(
                        ChatMessage("1", "assistant", "画像です。", files = files, encrypted = false),
                        onFile = { opened = it },
                        loader = { _, _, _ -> png },
                        actions = MessageActions(),
                        controlsVisible = false,
                        onToggleControls = {},
                    )
                }
            }
        }
        rule.waitUntil(5_000) {
            rule.onAllNodesWithContentDescription("a.jpg", useUnmergedTree = true).fetchSemanticsNodes().isNotEmpty()
        }
        rule.onNodeWithContentDescription("a.jpg", useUnmergedTree = true).performClick()
        rule.waitForIdle()
        assertEquals("1/a.jpg", opened)
    }
}
