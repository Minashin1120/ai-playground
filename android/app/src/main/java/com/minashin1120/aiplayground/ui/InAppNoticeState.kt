package com.minashin1120.aiplayground.ui

import android.os.SystemClock
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.runtime.saveable.listSaver

internal const val NOTICE_DURATION_MILLIS = 5_000L

/** Deadlines belong to messages, rather than to whichever window currently displays them. */
internal data class InAppNotice(
    val id: Long,
    val message: String,
    val expiresAtMillis: Long,
    val exitDirection: Int? = null,
    val dragging: Boolean = false,
    val entered: Boolean = false,
    val removeAtMillis: Long? = null,
)

internal class InAppNoticeState(private val nowMillis: () -> Long = SystemClock::elapsedRealtime) {
    val notices = mutableStateListOf<InAppNotice>()
    private var nextId = 0L
    private val windows = mutableListOf<Any>()
    var activeWindow by mutableStateOf<Any?>(null)
        private set

    val nextDeadline: Long?
        get() = notices.mapNotNull {
            if (it.exitDirection != null) it.removeAtMillis else if (!it.dragging) it.expiresAtMillis else null
        }.minOrNull()

    fun add(message: String, durationMillis: Long = NOTICE_DURATION_MILLIS) {
        val now = nowMillis()
        val duration = durationMillis.coerceAtLeast(0L)
        // Accessibility may request Long.MAX_VALUE (keep controls until explicitly dismissed).
        val deadline = if (duration >= Long.MAX_VALUE - now) Long.MAX_VALUE else now + duration
        notices.add(0, InAppNotice(++nextId, message, deadline))
    }

    fun expire() {
        val now = nowMillis()
        // Off-screen lazy items have no animation coroutine; they must still leave the collection.
        notices.removeAll { it.removeAtMillis?.let { deadline -> deadline <= now } == true }
        notices.indices.forEach { index ->
            val notice = notices[index]
            if (notice.exitDirection == null && !notice.dragging && notice.expiresAtMillis <= now) {
                notices[index] = notice.copy(exitDirection = -1, removeAtMillis = now + PlaygroundMotion.SHORT)
            }
        }
    }

    fun dismiss(id: Long, direction: Int = -1) {
        update(id) { notice ->
            if (notice.exitDirection == null) notice.copy(
                exitDirection = if (direction > 0) 1 else -1, dragging = false,
                removeAtMillis = nowMillis() + PlaygroundMotion.SHORT,
            )
            else notice
        }
    }

    fun setDragging(id: Long, dragging: Boolean) {
        update(id) { it.copy(dragging = dragging) }
    }

    fun finishEnter(id: Long) {
        update(id) { it.copy(entered = true) }
    }

    fun finishExit(id: Long) {
        notices.removeAll { it.id == id && it.exitDirection != null }
    }

    private fun update(id: Long, transform: (InAppNotice) -> InAppNotice) {
        val index = notices.indexOfFirst { it.id == id }
        if (index >= 0) notices[index] = transform(notices[index])
    }

    fun registerWindow(window: Any) {
        windows.add(window)
        activeWindow = window
    }

    fun focusWindow(window: Any) {
        if (window in windows) {
            windows.remove(window)
            windows.add(window)
            activeWindow = window
        }
    }

    fun unregisterWindow(window: Any) {
        windows.remove(window)
        if (activeWindow === window) activeWindow = windows.lastOrNull()
    }

    /** Save visible messages only; closing messages should not reappear after rotation. */
    fun save(): List<Any> = buildList {
        add(nextId)
        notices.filter { it.exitDirection == null && it.expiresAtMillis > nowMillis() }.forEach {
            add(it.id)
            add(it.message)
            add(it.expiresAtMillis)
        }
    }

    companion object {
        val Saver = listSaver<InAppNoticeState, Any>(save = { it.save() }, restore = { restore(it) })

        fun restore(values: List<Any>, nowMillis: () -> Long = SystemClock::elapsedRealtime): InAppNoticeState =
            InAppNoticeState(nowMillis).apply {
                nextId = values.first() as Long
                values.drop(1).chunked(3).forEach { row ->
                    val deadline = row[2] as Long
                    if (deadline > nowMillis()) notices.add(InAppNotice(row[0] as Long, row[1] as String, deadline, entered = true))
                }
            }
    }
}
