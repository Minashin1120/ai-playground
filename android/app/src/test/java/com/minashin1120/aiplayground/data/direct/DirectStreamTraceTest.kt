package com.minashin1120.aiplayground.data.direct

import com.minashin1120.aiplayground.data.ActivityLog
import com.minashin1120.aiplayground.data.LogPolicy
import okio.Buffer
import okio.Source
import okio.Timeout
import okio.buffer
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import java.io.File
import java.nio.file.Files
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import kotlin.concurrent.thread

class DirectStreamTraceTest {
    @Test fun distinguishesHeaderWaitFromBytesAndEmptyLinesBeforeThinking() {
        var now = 0L
        val logs = mutableListOf<Map<String, Any>>()
        val trace = DirectStreamTrace({ now }, { true }, { logs += it }, monitor = false, dispatch = { it() })
        trace.stage(2)
        now = 10_000
        trace.snapshot()
        assertEquals(2, logs.last()["from"])
        assertEquals(0L, logs.last()["bytes"])
        assertEquals(10_000L, logs.last()["wait_ms"])
        trace.stage(3)
        val secret = "PRIVATE_THINKING_PROMPT_ANSWER_KEY"
        val stream = ": $secret\n\nevent: response.reasoning_summary_text.delta\ndata: {\"type\":\"response.reasoning_summary_text.delta\",\"delta\":\"$secret\"}\n\n"
        val received = mutableListOf<String>()
        readSse(trace.source(Buffer().writeUtf8(stream)), onLine = trace::line) { event ->
            trace.event(DirectStreamTrace.category(event))
            received += event.data
            trace.stage(5)
            true
        }
        trace.finish("done")
        assertTrue(received.single().contains(secret)) // Diagnostics must not change the actual stream.
        assertEquals(1L, logs.last()["count"])
        assertEquals(2L, logs.last()["rows"])
        assertEquals(1L, logs.last()["pieces"])
        assertEquals(1L, logs.last()["messages"])
        assertEquals(stream.toByteArray().size.toLong(), logs.last()["bytes"])
        assertTrue(logs.any { it["reason_code"] == 131 })
        assertTrue(logs.any { it["reason_code"] == 132 })
        assertTrue(logs.any { it["reason_code"] == 133 && it["type"] == "thought" })
        assertFalse(logs.toString().contains(secret))
        logs.forEach { fields ->
            val safe = LogPolicy.entry(JSONObject(fields).put("ev", "direct"))!!
            assertEquals(fields.size + 1, safe.length())
        }
    }

    @Test fun distinguishesKeepalivesFromEventsWhileTheNextReadWaits() {
        val entered = CountDownLatch(1)
        val release = CountDownLatch(1)
        val logs = java.util.Collections.synchronizedList(mutableListOf<Map<String, Any>>())
        val trace = DirectStreamTrace({ 10L }, { true }, { logs += it }, monitor = false, dispatch = { it() })
        val source = object : Source {
            var reads = 0
            override fun timeout() = Timeout.NONE
            override fun close() { release.countDown() }
            override fun read(sink: Buffer, byteCount: Long): Long {
                if (reads++ == 0) { sink.writeUtf8(": heartbeat\n\n"); return 13 }
                entered.countDown()
                check(release.await(5, TimeUnit.SECONDS))
                return -1
            }
        }
        val reader = thread {
            readSse(trace.source(source.buffer()), onLine = trace::line) { error("No event expected") }
        }
        try {
            assertTrue(entered.await(5, TimeUnit.SECONDS))
            trace.snapshot()
            assertEquals(4, logs.last()["from"])
            assertEquals(13L, logs.last()["bytes"])
            assertEquals(0L, logs.last()["count"])
            assertEquals(1L, logs.last()["rows"])
            assertEquals(1L, logs.last()["pieces"])
        } finally {
            release.countDown(); reader.join(5_000); trace.finish("cancelled")
        }
        assertFalse(reader.isAlive)
    }

    @Test fun consentRevocationAndTerminalStatesStopCollectionAndLoggerFailuresAreHarmless() {
        for (phase in listOf("done", "error", "cancelled")) {
            var enabled = false
            val logs = mutableListOf<Map<String, Any>>()
            val disabled = DirectStreamTrace({ 0L }, { enabled }, { logs += it }, monitor = false, dispatch = { it() })
            disabled.event("thought"); disabled.snapshot(); disabled.finish(phase)
            assertTrue(logs.isEmpty())
            enabled = true
            val trace = DirectStreamTrace({ 0L }, { enabled }, { logs += it }, monitor = false, dispatch = { it() })
            trace.finish(phase)
            assertEquals(phase, logs.last()["phase"])
            val count = logs.size
            trace.snapshot(); trace.event("thought"); trace.finish(phase)
            assertEquals(count, logs.size)
            val revoked = DirectStreamTrace({ 0L }, { enabled }, { logs += it }, monitor = false, dispatch = { it() })
            enabled = false
            revoked.snapshot()
            enabled = true
            val before = logs.size
            revoked.event("content"); revoked.snapshot(); revoked.finish(phase)
            assertEquals(before, logs.size)
        }
        val throwing = DirectStreamTrace({ 0L }, { true }, { error("PRIVATE_EXCEPTION") }, monitor = false, dispatch = { it() })
        readSse(throwing.source(Buffer().writeUtf8("data: ok\n\n"))) { assertEquals("ok", it.data); true }
        throwing.finish("done")
    }

    @Test fun unknownProviderStringsAndPayloadContentsNeverReachSavedLogs() {
        val dir = Files.createTempDirectory("stream-log-test").toFile()
        val file = File(dir, "events.jsonl")
        val secret = "PRIVATE_THINKING_ANSWER_TITLE_FILE_API_KEY"
        try {
            ActivityLog.initFile(file, enabledAtStart = true)
            val generation = ActivityLog.collectionGeneration
            val trace = DirectStreamTrace({ 0L },
                { ActivityLog.enabled && ActivityLog.collectionGeneration == generation },
                { fields -> ActivityLog.log("direct", *fields.toList().toTypedArray()) }, monitor = false, dispatch = { it() })
            for (event in listOf(
                SseEvent(secret, secret),
                SseEvent(null, JSONObject().put("type", secret).put("delta", secret).toString()),
                SseEvent(null, JSONObject().put("type", "response.reasoning_summary_text.delta").put("delta", secret).toString()),
            )) trace.event(DirectStreamTrace.category(event))
            trace.snapshot()
            ActivityLog.drain()
            assertFalse(file.readText().contains(secret))
            assertTrue(ActivityLog.recent().toString().contains("thought"))
            // Account/log replacement changes the generation even if logging remains enabled.
            ActivityLog.initFile(File(dir, "other.jsonl"), enabledAtStart = true)
            trace.snapshot(); trace.finish("done")
            ActivityLog.drain()
            assertEquals(0, ActivityLog.recent().length())
        } finally {
            ActivityLog.setEnabled(false)
            dir.deleteRecursively()
        }
    }
}
