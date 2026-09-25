package com.minashin1120.aiplayground.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import com.github.takahirom.roborazzi.captureRoboImage
import com.minashin1120.aiplayground.ChatState
import com.minashin1120.aiplayground.data.Gem
import com.minashin1120.aiplayground.data.ThreadItem
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import org.robolectric.annotation.GraphicsMode

/** Same fixtures as the Web reference harness (threads t1–t5, two Gems, t1 selected). */
@RunWith(RobolectricTestRunner::class)
@GraphicsMode(GraphicsMode.Mode.NATIVE)
@Config(sdk = [34], qualifiers = "w360dp-h800dp-xxhdpi")
class SidebarScreenshotTest {
    private val threads = listOf(
        ThreadItem("t1", "Kotlinのコルーチン入門", "gemini-3.6-flash", isBookmarked = true),
        ThreadItem("t2", "旅行の計画を立てる", "gpt-5.5"),
        ThreadItem("t3", "", "gemini-3.6-flash"),
        ThreadItem("t4", "一時的な相談", "gemini-3.6-flash", isTemporary = true),
        ThreadItem("t5", "とても長いタイトルのチャットがサイドバーでどのように省略されるかを確認するための例", "claude-opus-5"),
    )
    private val gems = listOf(
        Gem("g1", "翻訳アシスタント", "日英翻訳", "あなたは翻訳者です。", ""),
        Gem("g2", "コードレビュー", "", "レビューしてください。", ""),
    )
    private val state = ChatState(starting = false, threads = threads, gems = gems, selected = threads.first())
    private val actions = SidebarActions(
        onNavigate = {}, onChangelog = {}, onHistory = {}, onLowBandwidth = {}, onSettings = {}, onLibrary = {},
        onNewChat = {}, onBatch = {}, onPdf = {}, onExternal = {}, onSearch = {}, onOpenThread = {}, onBookmark = {},
        onRenameThread = {}, onDeleteThread = {}, onMoreThreads = {}, onRefreshThreads = { it() }, onChooseGem = {},
        onNewGem = {}, onEditGem = {}, onDeleteGem = {}, onRefreshGems = { it() }, onHelp = {}, onLegal = {},
        onAlphaInfo = {}, onLogout = {},
    )

    @Test
    fun sidebarDark() = captureRoboImage("build/outputs/roborazzi/sidebar_dark_phone.png") { Frame(dark = true) }

    @Test
    fun sidebarLight() = captureRoboImage("build/outputs/roborazzi/sidebar_light_phone.png") { Frame(dark = false) }

    @Test
    fun lowBandwidthPillMatchesWeb() {
        assertEquals("低速回線モード (自動): 回線:3g", lowBandwidthPillText("auto", "回線:3g"))
        assertEquals("低速回線モード (手動)", lowBandwidthPillText("on", ""))
    }

    @Composable
    private fun Frame(dark: Boolean) {
        PlaygroundTheme(darkTheme = dark) {
            Box(Modifier.fillMaxSize().background(LocalWebPalette.current.bg1)) {
                Column { Sidebar(state, "V1.16.0", actions) }
            }
        }
    }
}
