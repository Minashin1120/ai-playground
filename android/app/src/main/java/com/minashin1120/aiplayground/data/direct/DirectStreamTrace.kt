package com.minashin1120.aiplayground.data.direct

import okio.Buffer
import okio.BufferedSource
import okio.ForwardingSource
import okio.buffer
import org.json.JSONObject
import java.util.concurrent.Executors
import java.util.concurrent.ScheduledFuture
import java.util.concurrent.TimeUnit

/** Counts and fixed categories only. Neither the emitter nor the timer receives stream contents. */
internal class DirectStreamTrace(
    private val elapsed: () -> Long,
    private val enabled: () -> Boolean,
    private val emit: (Map<String, Any>) -> Unit,
    monitor: Boolean = true,
    private val dispatch: (() -> Unit) -> Unit = { task -> scheduler.execute(task) },
) {
    private val lock = Any()
    private var active = enabled()
    private var stage = 0
    private var changedAt = elapsed()
    private var lastByteAt: Long? = null
    private var bytes = 0L
    private var reads = 0L
    private var events = 0L
    private var blanks = 0L
    private var comments = 0L
    private var thoughts = 0L
    private var contents = 0L
    private var readStarted = false
    private val categories = HashSet<String>()
    private var timer: ScheduledFuture<*>? = null

    init {
        snapshot("start")
        if (active && monitor) timer = scheduler.scheduleAtFixedRate({ snapshot("start") }, 10, 10, TimeUnit.SECONDS)
    }

    private fun allowed(): Boolean {
        if (!enabled()) { active = false; timer?.cancel(false) }
        return active
    }

    private fun record(code: Int, phase: String, type: String = "status") {
        val now = elapsed()
        // This map is the entire logging boundary: no data, names, URLs, errors, or headers.
        val fields = mapOf("reason_code" to code, "phase" to phase, "type" to type,
            "ms" to now, "from" to stage, "wait_ms" to (now - changedAt),
            "quiet_ms" to (lastByteAt?.let { now - it } ?: now),
            "bytes" to bytes, "chunks" to reads, "count" to events, "rows" to blanks,
            "pieces" to comments, "messages" to thoughts, "nodes" to contents)
        // File pruning/locking in the logger must never hold up the network reader.
        runCatching { dispatch { if (enabled()) runCatching { emit(fields) } } }
    }

    fun stage(value: Int) = synchronized(lock) {
        if (allowed() && stage != value) { stage = value; changedAt = elapsed() }
    }

    fun snapshot(phase: String = "start") = synchronized(lock) {
        if (allowed()) record(130, phase)
    }

    fun finish(phase: String) = synchronized(lock) {
        timer?.cancel(false)
        if (allowed()) record(130, phase)
        active = false
    }

    fun source(source: BufferedSource): BufferedSource = object : ForwardingSource(source) {
        override fun read(sink: Buffer, byteCount: Long): Long {
            synchronized(lock) {
                if (allowed()) {
                    stage(4)
                    if (!readStarted) { readStarted = true; record(131, "start") }
                }
            }
            val count = super.read(sink, byteCount)
            synchronized(lock) {
                if (allowed()) {
                    if (count > 0) {
                        val first = bytes == 0L
                        bytes += count; reads++; lastByteAt = elapsed()
                        if (first) record(132, "done")
                    }
                    stage(5)
                }
            }
            return count
        }
    }.buffer()

    /** Called with a fixed line class, never with the line itself. */
    fun line(kind: Int): Unit = synchronized(lock) {
        if (allowed()) when (kind) { 0 -> blanks++; 1 -> comments++ }
    }

    fun event(category: String) = synchronized(lock) {
        if (allowed()) {
            events++
            val safe = when (category) { "thought" -> "thought"; "content" -> "content"; else -> "message" }
            if (safe == "thought") thoughts++
            if (safe == "content") contents++
            stage(6)
            if (categories.add(safe)) record(133, "done", safe)
        }
    }

    companion object {
        private val scheduler = Executors.newSingleThreadScheduledExecutor { task ->
            Thread(task, "direct-stream-diagnostics").apply { isDaemon = true }
        }

        /** Only exact known event types become a category. Arbitrary provider strings never escape. */
        fun category(event: SseEvent): String {
            val type = event.event ?: if (event.data.length <= 16_384)
                runCatching { JSONObject(event.data).optString("type") }.getOrNull() else null
            return when (type) {
                "response.reasoning_summary_text.delta", "response.reasoning_text.delta",
                "response.reasoning_text.done", "response.reasoning_summary_text.done",
                "response.reasoning_summary_part.done" -> "thought"
                "response.output_text.delta", "response.output_text.done" -> "content"
                else -> "message"
            }
        }
    }
}
