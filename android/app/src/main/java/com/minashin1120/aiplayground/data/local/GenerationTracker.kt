package com.minashin1120.aiplayground.data.local

import java.util.concurrent.atomic.AtomicBoolean
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update

/** Answers of these catalog modes take minutes, so the process must not be killed while they run. */
fun needsKeepAlive(mode: String): Boolean = mode == "image" || mode == "video"

/** Held while a long answer is generated on the device; [begin] returns the handle that ends it. */
fun interface GenerationKeepAlive {
    fun begin(mode: String): AutoCloseable

    companion object {
        val None = GenerationKeepAlive { AutoCloseable { } }
    }
}

/** The long answers running right now, one entry (the catalog mode) per answer. */
class GenerationTracker {
    private val running = MutableStateFlow<List<String>>(emptyList())
    val active: StateFlow<List<String>> get() = running

    fun begin(mode: String): AutoCloseable {
        running.update { it + mode }
        val closed = AtomicBoolean(false)
        return AutoCloseable {
            // Closing twice must not end another answer's hold.
            if (closed.compareAndSet(false, true)) running.update { list ->
                val at = list.indexOf(mode)
                if (at < 0) list else list.toMutableList().apply { removeAt(at) }
            }
        }
    }
}
