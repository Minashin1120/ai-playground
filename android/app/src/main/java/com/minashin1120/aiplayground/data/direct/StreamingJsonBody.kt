package com.minashin1120.aiplayground.data.direct

import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody
import okio.BufferedSink
import okio.ForwardingSink
import okio.blackholeSink
import okio.buffer
import org.json.JSONArray
import org.json.JSONObject

/** Writes base64-heavy requests incrementally, without copying the whole JSON into a String and byte[]. */
internal class StreamingJsonBody(private val payload: JSONObject) : RequestBody() {
    override fun contentType() = "application/json; charset=utf-8".toMediaType()
    override fun contentLength(): Long {
        var size = 0L
        val counter = object : ForwardingSink(blackholeSink()) {
            override fun write(source: okio.Buffer, byteCount: Long) {
                size += byteCount
                super.write(source, byteCount)
            }
        }.buffer()
        writeTo(counter)
        counter.flush()
        return size
    }
    override fun writeTo(sink: BufferedSink) { writeValue(sink, payload) }

    private fun writeValue(sink: BufferedSink, value: Any?) {
        when (value) {
            null, JSONObject.NULL -> sink.writeUtf8("null")
            is JSONObject -> {
                sink.writeByte('{'.code)
                val keys = value.keys()
                var comma = false
                while (keys.hasNext()) {
                    if (comma) sink.writeByte(','.code)
                    comma = true
                    val key = keys.next()
                    writeString(sink, key)
                    sink.writeByte(':'.code)
                    writeValue(sink, value.get(key))
                }
                sink.writeByte('}'.code)
            }
            is JSONArray -> {
                sink.writeByte('['.code)
                for (index in 0 until value.length()) {
                    if (index > 0) sink.writeByte(','.code)
                    writeValue(sink, value.opt(index))
                }
                sink.writeByte(']'.code)
            }
            is Number -> sink.writeUtf8(JSONObject.numberToString(value))
            is Boolean -> sink.writeUtf8(value.toString())
            else -> writeString(sink, value.toString())
        }
    }
    private fun writeString(sink: BufferedSink, value: String) {
        sink.writeByte('"'.code)
        // Quote small slices only; keep surrogate pairs together at the slice boundary.
        var start = 0
        while (start < value.length) {
            var end = minOf(start + 4096, value.length)
            if (end < value.length && value[end - 1].isHighSurrogate()) end--
            val quoted = JSONObject.quote(value.substring(start, end))
            sink.writeUtf8(quoted, 1, quoted.length - 1)
            start = end
        }
        sink.writeByte('"'.code)
    }
}
