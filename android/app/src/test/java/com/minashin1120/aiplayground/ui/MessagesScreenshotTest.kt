package com.minashin1120.aiplayground.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.github.takahirom.roborazzi.captureRoboImage
import com.minashin1120.aiplayground.data.ChatMessage
import com.minashin1120.aiplayground.data.buildTokenTotals
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import org.robolectric.annotation.GraphicsMode

/** The conversation fixture of the Web reference harness (thread t1). */
@RunWith(RobolectricTestRunner::class)
@GraphicsMode(GraphicsMode.Mode.NATIVE)
@Config(sdk = [34], qualifiers = "w360dp-h2400dp-xxhdpi")
class MessagesScreenshotTest {
    private val answer = """Kotlinのコルーチンは、**非同期処理**を簡潔に書くための仕組みです。

## 主な特徴

1. 軽量なスレッドのように動作します
2. `suspend` 関数で中断・再開できます
   - 入れ子の項目
3. 構造化された並行性を持ちます

> 公式ドキュメントも参照してください。
> 2行目の引用です。

```kotlin
fun main() = runBlocking {
    launch { delay(1000L); println("World!") }
    println("Hello")
}
```

| 機能 | 説明 |
|---|---|
| launch | 結果を返さない |
| async | 結果を返す |

詳しくは [公式サイト](https://kotlinlang.org) をご覧ください。"""

    private val messages = listOf(
        ChatMessage("101", "user", "Kotlinのコルーチンについて *簡単に* 教えてください。\n2行目です。", encrypted = false),
        ChatMessage("102", "assistant", answer, thought = "ユーザーはコルーチンの概要を求めている。", parentId = 101, model = "gemini-3.6-flash",
            tokens = 1523, tokensIn = 210, tokensOut = 1313, tokensContent = 900, tokensThought = 413, encrypted = true),
        ChatMessage("103", "user", "ありがとう。asyncの例もお願いします。", parentId = 102, quote = "構造化された並行性を持ちます",
            gemName = "翻訳アシスタント", encrypted = false),
        ChatMessage("104", "assistant", "`async` は `Deferred` を返します。\n\n```kotlin\nval x = async { 1 + 2 }\nprintln(x.await())\n```",
            parentId = 103, model = "gemini-3.6-flash", tokens = 320, tokensIn = 1600, tokensOut = 120, tokensThought = 0,
            gemName = "翻訳アシスタント", encrypted = false),
    )

    @Test
    fun conversationDark() = captureRoboImage("build/outputs/roborazzi/messages_dark_phone.png") { Frame(dark = true) }

    @Test
    fun conversationLight() = captureRoboImage("build/outputs/roborazzi/messages_light_phone.png") { Frame(dark = false) }

    @Test
    fun totalsMatchWebFixture() = assertEquals(1843, buildTokenTotals(messages).total)

    @Composable
    private fun Frame(dark: Boolean) {
        PlaygroundTheme(darkTheme = dark) {
            val web = LocalWebPalette.current
            Column(Modifier.background(web.bg1)) {
                TotalTokenBar(buildTokenTotals(messages), buildTokenTotals(messages)) {}
                Column(
                    Modifier.fillMaxWidth().verticalScroll(rememberScrollState()).padding(12.dp),
                    verticalArrangement = Arrangement.spacedBy(20.dp),
                ) {
                    messages.forEach { message ->
                        MessageBubble(message, onFile = {}, loader = null, actions = MessageActions(),
                            controlsVisible = message.id == "101", onToggleControls = {})
                    }
                }
            }
        }
    }
}
