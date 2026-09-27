package com.minashin1120.aiplayground.data.direct

import okio.Buffer
import org.junit.Assert.*
import org.junit.Test

class SseReaderTest {
    private fun events(text: String): List<SseEvent> {
        val out = mutableListOf<SseEvent>()
        readSse(Buffer().writeUtf8(text)) { out += it; true }
        return out
    }

    @Test fun joinsDataLinesAndKeepsEventNames() {
        val parsed = events(": comment\nevent: message_start\ndata: {\"a\":1}\n\ndata: line1\ndata: line2\n\n")
        assertEquals(listOf(SseEvent("message_start", "{\"a\":1}"), SseEvent(null, "line1\nline2")), parsed)
    }

    @Test fun deliversTheLastBlockWithoutTrailingBlankLine() {
        assertEquals(listOf(SseEvent(null, "[DONE]")), events("data: [DONE]"))
    }

    @Test fun eventNameDoesNotLeakIntoTheNextBlock() {
        val parsed = events("event: ping\n\ndata: x\n\n")
        assertEquals(listOf(SseEvent(null, "x")), parsed)
    }

    @Test fun stopsWhenTheCallbackReturnsFalse() {
        val out = mutableListOf<String>()
        readSse(Buffer().writeUtf8("data: 1\n\ndata: 2\n\n")) { out += it.data; false }
        assertEquals(listOf("1"), out)
    }
}
