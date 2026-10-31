package com.minashin1120.aiplayground.data

import mockwebserver3.MockResponse
import mockwebserver3.MockWebServer
import okhttp3.*
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import java.io.IOException
import java.net.UnknownHostException
import java.util.Collections

class NetworkTraceTest {
    @Test fun queuedCallsAreDistinguishedFromDnsAndConnectDelays() {
        val entries = Collections.synchronizedList(mutableListOf<Map<String, Any?>>())
        val entered = java.util.concurrent.CountDownLatch(1)
        val release = java.util.concurrent.CountDownLatch(1)
        val queued = java.util.concurrent.CountDownLatch(1)
        val finished = java.util.concurrent.CountDownLatch(2)
        MockWebServer().use { server ->
            server.start()
            repeat(2) { server.enqueue(MockResponse.Builder().body("{}").build()) }
            val client = OkHttpClient.Builder().dispatcher(Dispatcher().apply { maxRequestsPerHost = 1 })
                .eventListenerFactory {
                    NetworkTrace(context = 1, emit = { entry ->
                        entries.add(entry)
                        if (entry["reason_code"] == 111 && entry["phase"] == "start") queued.countDown()
                    })
                }.build()
            val first = client.newCall(Request.Builder().url(server.url("/first")).build())
            val second = client.newCall(Request.Builder().url(server.url("/second")).build())
            fun callback(block: Boolean) = object : Callback {
                override fun onFailure(call: Call, e: IOException) { finished.countDown() }
                override fun onResponse(call: Call, response: Response) {
                    try {
                        response.use {
                            if (block) { entered.countDown(); release.await(5, java.util.concurrent.TimeUnit.SECONDS) }
                            it.body.string()
                        }
                    } finally { finished.countDown() }
                }
            }
            try {
                first.enqueue(callback(true))
                assertTrue(entered.await(5, java.util.concurrent.TimeUnit.SECONDS))
                second.enqueue(callback(false))
                assertTrue(queued.await(2, java.util.concurrent.TimeUnit.SECONDS))
                release.countDown()
                assertTrue(finished.await(5, java.util.concurrent.TimeUnit.SECONDS))
                val queueEntries = entries.filter { it["reason_code"] == 111 }
                assertEquals(listOf("start", "done"), queueEntries.map { it["phase"] })
                assertEquals(1, queueEntries.map { it["op"] }.distinct().size)
            } finally {
                release.countDown(); first.cancel(); second.cancel(); client.connectionPool.evictAll()
            }
        }
    }

    @Test fun recordsTransportStagesAndReuseWithoutRecordingRequestOrResponseSecrets() {
        val entries = Collections.synchronizedList(mutableListOf<Map<String, Any?>>())
        MockWebServer().use { server ->
            server.start()
            val client = OkHttpClient.Builder().eventListenerFactory {
                NetworkTrace(context = 1, emit = { entries.add(it) })
            }.build()
            repeat(2) {
                server.enqueue(MockResponse.Builder().addHeader("X-Secret", "private-response").body("private-body").build())
                client.newCall(Request.Builder().url(server.url("/private-file?key=private-key"))
                    .header("Authorization", "private-token").build()).execute().use { it.body.string() }
            }
            val calls = entries.groupBy { it["op"] }.values.toList()
            assertEquals(2, calls.size)
            assertTrue(calls[0].any { it["reason_code"] == 102 })
            assertFalse(calls[1].any { it["reason_code"] == 102 })
            assertTrue(calls[1].any { it["reason_code"] == 104 })
            assertTrue(calls.all { call -> call.any { it["reason_code"] == 108 && it["bytes"] == 12L } })
            for (entry in entries) {
                val raw = JSONObject(entry).put("ev", "http")
                val safe = requireNotNull(LogPolicy.entry(raw))
                assertEquals(raw.getInt("reason_code"), safe.getInt("reason_code"))
                assertEquals(raw.getString("op"), safe.getString("op"))
                assertFalse(raw.toString().contains("private-"))
                assertFalse(raw.has("host"))
                assertFalse(raw.has("path"))
            }
            client.connectionPool.evictAll()
        }
    }

    @Test fun dnsFailurePreservesItsStageAndClassificationButNotTheExceptionText() {
        val entries = mutableListOf<Map<String, Any?>>()
        val client = OkHttpClient.Builder().dns { throw UnknownHostException("private-host-and-token") }
            .eventListenerFactory { NetworkTrace(context = 2, emit = { entries.add(it) }) }.build()
        try { client.newCall(Request.Builder().url("https://private.invalid/").build()).execute(); fail("Expected DNS failure") }
        catch (_: UnknownHostException) { }
        val end = entries.last()
        assertEquals("error", end["phase"])
        assertEquals(1, end["error_code"])
        assertEquals(101, end["to"])
        assertFalse(entries.toString().contains("private"))
    }

    @Test fun cancellationAndLoggingFailureDoNotInterfereWithCallLifecycle() {
        val entries = mutableListOf<Map<String, Any?>>()
        val call = OkHttpClient().newCall(Request.Builder().url("https://example.com/").build())
        val listener = NetworkTrace(context = 3, emit = { entries.add(it) })
        listener.callStart(call)
        call.cancel()
        listener.canceled(call)
        listener.callFailed(call, IOException("private-cancellation"))
        assertEquals("cancelled", entries.last()["phase"])
        val broken = NetworkTrace(context = 3, emit = { error("private-sink-error") })
        broken.callStart(call)
        broken.callFailed(call, IOException())
        assertEquals(1, NetworkTrace.failureCode(IOException("private", UnknownHostException("private"))))
        assertEquals(3, NetworkTrace.failureCode(javax.net.ssl.SSLHandshakeException("private")))
    }
}
