package com.minashin1120.aiplayground.data.direct

import okio.BufferedSource

/** One Server-Sent Event: the optional `event:` name and the joined `data:` lines. */
data class SseEvent(val event: String?, val data: String)

/**
 * Reads Server-Sent Events from [source] (the Playground server's realtime/Lyria streams and the AI
 * providers' streaming APIs). `data:` lines of one block are joined with `\n`; a blank line ends the
 * block. [onEvent] returns false to stop reading early (for example after a terminal `[DONE]`).
 */
fun readSse(source: BufferedSource, maxLine: Long = 8L * 1024 * 1024, onEvent: (SseEvent) -> Boolean) {
    var data = StringBuilder()
    var event: String? = null
    while (!source.exhausted()) {
        val line = source.readUtf8LineStrict(maxLine)
        when {
            line.startsWith("data:") -> data.append(line.removePrefix("data:").removePrefix(" ")).append('\n')
            line.startsWith("event:") -> event = line.removePrefix("event:").trim()
            line.startsWith(":") -> Unit
            line.isBlank() -> {
                if (data.isNotEmpty()) {
                    val payload = data.toString().trimEnd('\n')
                    val name = event
                    data = StringBuilder()
                    event = null
                    if (!onEvent(SseEvent(name, payload))) return
                } else event = null
            }
        }
    }
    if (data.isNotEmpty()) onEvent(SseEvent(event, data.toString().trimEnd('\n')))
}
