package com.minashin1120.aiplayground.data

import org.json.JSONObject
import kotlin.math.pow
import kotlin.math.sqrt

/**
 * Web `botTelemetry` (chat_core part04) for touch input: counts taps (primary pointer down) and
 * sampled moves, and reports to `/api/bot-telemetry` only when the window looks automated.
 * Keys and pointer speed are not measured (the soft keyboard is another window, and a fling is
 * faster than any mouse), so their fields stay at the Web "nothing seen" values.
 * Times are monotonic milliseconds (`MotionEvent.eventTime`).
 */
class BotTelemetry {
    private var windowStart = -1L
    private var lastSend = -1L
    private var clicks = 0
    private var moves = 0
    private var fastClicks = 0
    private val clickTimes = ArrayDeque<Long>()
    private val clickIntervals = ArrayDeque<Double>()
    private var lastClick = 0L
    private var lastMoveSample = Long.MIN_VALUE / 2

    /** Returns true when a forced report is due (Web: `fastClicks >= 4` sends immediately). */
    @Synchronized fun recordTap(now: Long): Boolean {
        begin(now)
        clicks += 1
        if (lastClick > 0) {
            val delta = (now - lastClick).toDouble()
            clickIntervals.addLast(delta)
            if (clickIntervals.size > 10) clickIntervals.removeFirst()
            if (delta < 120) fastClicks += 1
        }
        lastClick = now
        clickTimes.addLast(now)
        while (clickTimes.isNotEmpty() && now - clickTimes.first() > 2000) clickTimes.removeFirst()
        return fastClicks >= 4
    }

    /** Web `recordMove`: at most one sample per 80 ms. */
    @Synchronized fun recordMove(now: Long) {
        begin(now)
        if (now - lastMoveSample < 80) return
        lastMoveSample = now
        moves += 1
    }

    /** Web `computeStats`, in the `/api/bot-telemetry` payload shape. */
    @Synchronized fun stats(now: Long): JSONObject {
        val windowMs = maxOf(1L, now - (if (windowStart < 0) now else windowStart))
        var avgClick = 0.0
        var clickCv = 1.0
        if (clickIntervals.size >= 3) {
            val mean = clickIntervals.average()
            val variance = clickIntervals.sumOf { (it - mean).pow(2) } / clickIntervals.size
            avgClick = mean
            clickCv = if (mean > 0) sqrt(variance) / mean else 1.0
        }
        return JSONObject()
            .put("window_ms", windowMs)
            .put("clicks", clicks).put("keys", 0).put("moves", moves)
            .put("fast_clicks", fastClicks).put("fast_keys", 0)
            .put("click_burst", clickTimes.size).put("key_burst", 0)
            .put("avg_click_ms", avgClick).put("click_cv", clickCv)
            .put("event_rate", (clicks + moves) / (windowMs / 1000.0))
            .put("pointer_speed_max", 0).put("pointer_speed_avg", 0)
            .put("source", "android")
    }

    /**
     * Web `send`: checked at most once per 3 s unless forced (the first check comes 3 s after the
     * first input, like the Web interval after page load), never for an empty window, and only for
     * a suspicious window. Returns the payload to send (the window starts over), or null.
     */
    @Synchronized fun takeReport(now: Long, force: Boolean): JSONObject? {
        if (!force && now - lastSend < 3000) return null
        lastSend = now
        val payload = stats(now)
        if (clicks + moves == 0) return null
        if (!isSuspicious(payload)) return null
        resetWindow(now)
        return payload
    }

    private fun begin(now: Long) {
        if (windowStart >= 0) return
        windowStart = now
        if (lastSend < 0) lastSend = now
    }

    private fun resetWindow(now: Long) {
        windowStart = now
        clicks = 0
        moves = 0
        fastClicks = 0
        clickTimes.clear()
        clickIntervals.clear()
    }

    companion object {
        /** Web `isSuspicious` (the key and pointer-speed rules never match touch input). */
        fun isSuspicious(payload: JSONObject): Boolean {
            if (payload.optInt("fast_clicks") >= 4) return true
            if (payload.optInt("click_burst") >= 8) return true
            if (payload.optDouble("event_rate", 0.0) >= 20) return true
            val avgClick = payload.optDouble("avg_click_ms", 0.0)
            return avgClick > 0 && avgClick < 160 && payload.optDouble("click_cv", 1.0) < 0.08
        }
    }
}

/** Web `registerSendButtonSpam`: send presses in the last 3 s (8 or more locks the account). */
class SendSpamCounter {
    private val presses = ArrayDeque<Long>()

    @Synchronized fun register(now: Long): Int {
        presses.addLast(now)
        while (presses.isNotEmpty() && now - presses.first() > 3000) presses.removeFirst()
        return presses.size
    }

    @Synchronized fun reset() = presses.clear()

    companion object {
        const val LOCK_THRESHOLD = 8
    }
}
