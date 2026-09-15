package com.minashin1120.aiplayground.data

import kotlinx.coroutines.runBlocking
import mockwebserver3.MockResponse
import mockwebserver3.MockWebServer
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import java.io.ByteArrayInputStream
import java.io.IOException

class PlaygroundApiTest {
    @Test fun bearerDoesNotBecomeBrowserCookie() = runBlocking {
        MockWebServer().use { server ->
            server.start()
            server.enqueue(MockResponse.Builder().addHeader("Content-Type", "application/json")
                .addHeader("Set-Cookie", "session=browser; Path=/").body("{}").build())
            server.enqueue(MockResponse.Builder().addHeader("Content-Type", "application/json").body("{}").build())
            val api = PlaygroundApi(server.url("/"))
            api.get("/api/mobile/v1/me", "test-token")
            api.post("/api/mobile/v1/device", JSONObject())
            val first = server.takeRequest()
            assertEquals("Bearer test-token", first.headers["Authorization"])
            assertNull(first.headers["Cookie"])
            val second = server.takeRequest()
            assertNull(second.headers["Authorization"])
            assertNull(second.headers["Cookie"])
        }
    }

    @Test fun redirectsAreNotFollowed() = runBlocking {
        MockWebServer().use { server ->
            server.start()
            server.enqueue(MockResponse.Builder().code(302).addHeader("Location", "https://example.org/").build())
            val failure = runCatching { PlaygroundApi(server.url("/")).get("/api/threads", "test") }.exceptionOrNull()
            assertTrue(failure is ApiException)
            assertEquals(302, (failure as ApiException).status)
            assertEquals(1, server.requestCount)
        }
    }

    @Test fun streamRequiresTerminalEvent() = runBlocking {
        MockWebServer().use { server ->
            server.start()
            val api = PlaygroundApi(server.url("/"))
            server.enqueue(MockResponse.Builder().addHeader("Content-Type", "application/x-ndjson")
                .body("{\"type\":\"content\",\"content\":\"こんにちは\"}\n{\"type\":\"done\"}\n").build())
            val events = mutableListOf<String>()
            api.stream("/chat_stream", JSONObject(), "test") { events.add(it.getString("type")) }
            assertEquals(listOf("content", "done"), events)
            server.enqueue(MockResponse.Builder().addHeader("Content-Type", "application/x-ndjson")
                .body("{\"type\":\"content\"}\n").build())
            assertTrue(runCatching { api.stream("/chat_stream", JSONObject(), "test") {} }.exceptionOrNull() is IOException)
        }
    }

    @Test fun retryAfterIsPreserved() = runBlocking {
        MockWebServer().use { server ->
            server.start()
            server.enqueue(MockResponse.Builder().code(429).addHeader("Content-Type", "application/json")
                .addHeader("Retry-After", "17").body("{\"error\":\"slow_down\"}").build())
            val failure = runCatching { PlaygroundApi(server.url("/")).get("/api/threads", "test") }.exceptionOrNull()
            assertEquals(17L, (failure as ApiException).retryAfter)
        }
    }

    @Test fun externalPathsAreRejected() {
        MockWebServer().use { server ->
            server.start()
            val api = PlaygroundApi(server.url("/"))
            for (path in listOf("https://example.org/", "//example.org/")) {
                assertTrue(runCatching { api.url(path) }.exceptionOrNull() is IllegalArgumentException)
            }
        }
    }

    @Test fun responseSizeIsBounded() {
        assertEquals("日本語", readBoundedUtf8(ByteArrayInputStream("日本語".toByteArray()), 9))
        assertTrue(runCatching { readBoundedUtf8(ByteArrayInputStream(ByteArray(10)), 9) }.exceptionOrNull() is IOException)
    }

    @Test fun historyParsesLegacyAndMultipleAttachments() {
        val history = JSONObject("""{"messages":[{"id":1,"role":"user","content":null,"image_url":"one.txt"},{"id":2,"role":"assistant","content":"回答","image_url":"[\"a.png\",\"b.png\"]"}]}""")
        val rows = parseMessages(history)
        assertEquals("", rows[0].content)
        assertEquals(listOf("one.txt"), rows[0].files)
        assertEquals(listOf("a.png", "b.png"), rows[1].files)
    }
}
