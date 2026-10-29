package com.minashin1120.aiplayground.data.direct

import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.async
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.withTimeout
import mockwebserver3.MockResponse
import mockwebserver3.MockWebServer
import okhttp3.RequestBody
import okhttp3.MediaType
import okio.BufferedSink
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import java.io.IOException
import javax.net.ssl.SSLException

class DirectHttpTest {
    private val http = DirectHttp(allowedHosts = emptySet(), allowCleartextLocalhost = true)

    @Test fun sslReadFailureDoesNotReplayPostAndManualRetryUsesNewConnection() = runBlocking {
        MockWebServer().use { server ->
            server.start()
            repeat(2) { server.enqueue(MockResponse.Builder().body("{}").build()) }
            val request = http.request(server.url("/").toString(), emptyMap()).post(http.jsonBody(JSONObject())).build()
            try {
                http.execute(request) { response ->
                    response.body.string() // Return the connection to the pool before injecting a read failure.
                    throw IOException("wrapper", SSLException("fake sensitive failure"))
                }
                fail("Expected SSL failure")
            } catch (expected: IOException) {
                assertTrue(isSslFailure(expected))
            }
            assertEquals(1, server.requestCount)
            assertEquals("{}", http.execute(request) { it.body.string() })
            assertEquals(2, server.requestCount)
            assertNotEquals(server.takeRequest().connectionIndex, server.takeRequest().connectionIndex)
        }
    }

    @Test fun sslFailureBeforeResponseIsPropagatedWithoutRetry() = runBlocking {
        MockWebServer().use { server ->
            server.start()
            var writes = 0
            val body = object : RequestBody() {
                override fun contentType(): MediaType? = null
                override fun contentLength() = 1L
                override fun writeTo(sink: BufferedSink) {
                    writes++
                    throw SSLException("fake TLS write failure")
                }
            }
            val request = http.request(server.url("/").toString(), emptyMap()).post(body).build()
            try {
                http.execute(request) { fail("No response expected") }
                fail("Expected SSL failure")
            } catch (expected: SSLException) {
                assertEquals(1, writes)
            }
        }
    }

    @Test fun sslClassificationDoesNotMatchArbitraryErrorTextOrLoopOnCauses() {
        assertFalse(isSslFailure(IOException("SSLException SSLV3_ALERT_BAD_RECORD_MAC")))
        val first = IOException("first")
        val second = IOException("second", first)
        first.initCause(second)
        assertFalse(isSslFailure(first))
    }

    @Test fun cancellingAReadRemainsCancellationAndDoesNotReplayPost() = runBlocking {
        MockWebServer().use { server ->
            server.start()
            server.enqueue(MockResponse.Builder().body("{}").build())
            val entered = CompletableDeferred<Unit>()
            val released = java.util.concurrent.CountDownLatch(1)
            val request = http.request(server.url("/").toString(), emptyMap()).post(http.jsonBody(JSONObject())).build()
            val reading = async {
                http.execute(request) {
                    entered.complete(Unit)
                    check(released.await(5, java.util.concurrent.TimeUnit.SECONDS))
                    it.body.string()
                }
            }
            try {
                withTimeout(5_000) { entered.await() }
                reading.cancelAndJoin()
                assertTrue(reading.isCancelled)
                assertEquals(1, server.requestCount)
            } finally {
                released.countDown()
            }
        }
    }
}
