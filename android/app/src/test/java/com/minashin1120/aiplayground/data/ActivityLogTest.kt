package com.minashin1120.aiplayground.data

import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.nio.file.Files

class ActivityLogTest {
    private val dir: File = Files.createTempDirectory("activity-log").toFile()
    private val file = File(dir, "activity_log/events.jsonl")

    @After fun tearDown() {
        ActivityLog.setEnabled(false)
        dir.deleteRecursively()
    }

    @Test fun recordsOnlyWhileEnabledAndKeepsTheLastHour() {
        ActivityLog.initFile(file, enabledAtStart = false)
        ActivityLog.log("ignored")
        ActivityLog.drain()
        assertFalse(file.exists())

        ActivityLog.initFile(file, enabledAtStart = true)
        file.parentFile?.mkdirs()
        val old = System.currentTimeMillis() - ActivityLog.WINDOW_MS - 60_000
        file.writeText("$old\t{\"t\":$old,\"ev\":\"old\"}\n")
        ActivityLog.log("http", "method" to "GET", "path" to "/api/threads", "status" to 200, "skipped" to null)

        val entries = ActivityLog.recent()
        assertEquals(1, entries.length())
        val entry = entries.getJSONObject(0)
        assertEquals("http", entry.getString("ev"))
        assertEquals("/api/threads", entry.getString("path"))
        assertEquals(200, entry.getInt("status"))
        assertFalse(entry.has("skipped"))
        assertEquals(2, ActivityLog.stats().first)
    }

    @Test fun longValuesAreShortenedAndClearRemovesTheFile() {
        ActivityLog.initFile(file, enabledAtStart = true)
        ActivityLog.log("error", "message" to "x".repeat(2000))
        val message = ActivityLog.recent().getJSONObject(0).getString("message")
        assertTrue(message.length < 600)
        assertTrue(message.endsWith("…(2000)"))

        ActivityLog.clear()
        assertFalse(file.exists())
        assertEquals(0, ActivityLog.recent().length())
    }

    @Test fun turningOffDeletesTheLog() {
        ActivityLog.initFile(file, enabledAtStart = true)
        ActivityLog.log("click")
        ActivityLog.drain()
        assertTrue(file.exists())
        ActivityLog.setEnabled(false)
        assertFalse(ActivityLog.enabled)
        assertFalse(file.exists())
        ActivityLog.log("after")
        assertEquals(0, ActivityLog.recent().length())
    }

    @Test fun feedbackSentTextNamesTheLogsAndTheChatCopy() {
        assertEquals("フィードバックを送信しました", feedbackSentText(null, true, false, false))
        assertEquals("フィードバックと直近1時間のログ（3件）を送信しました", feedbackSentText(3, true, false, false))
        assertEquals("フィードバックとチャットのコピーを送信しました", feedbackSentText(null, true, true, true))
        assertEquals("フィードバック、直近1時間のログ（2件）、チャットのコピーを送信しました", feedbackSentText(2, true, true, true))
        assertEquals("フィードバックを送信しました（ログとチャットのコピーは保存できませんでした）", feedbackSentText(2, false, true, false))
        assertEquals("フィードバックと画像（2枚）を送信しました", feedbackSentText(null, true, false, false, 2, true))
        assertEquals("フィードバックを送信しました（画像は保存できませんでした）", feedbackSentText(null, true, false, false, 2, false))
    }
}
