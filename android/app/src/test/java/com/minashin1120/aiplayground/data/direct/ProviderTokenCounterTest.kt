package com.minashin1120.aiplayground.data.direct

import kotlinx.coroutines.runBlocking
import mockwebserver3.MockResponse
import mockwebserver3.MockWebServer
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test

class ProviderTokenCounterTest {
    private val http = DirectHttp(allowedHosts = emptySet(), allowCleartextLocalhost = true)
    private fun request(model: String, attachments: List<DirectAttachment> = emptyList()) =
        DirectRequest(model, "fake-key", "", listOf(DirectTurn("user", "入力", attachments)), JSONObject())

    @Test fun countersUseOfficialEndpointsAndCacheResults() = runBlocking {
        val cases = listOf(
            Triple("gemini-3-pro-preview", "gemini", """{"totalTokens":31}"""),
            Triple("claude-sonnet-4-6", "anthropic", """{"input_tokens":31}"""),
            Triple("gpt-5", "openai", """{"input_tokens":31}"""),
            Triple("grok-4", "xai", """{"token_ids":[1,2,3]}"""),
            Triple("kimi-k2", "kimi", """{"data":{"total_tokens":31}}"""),
        )
        for ((model, provider, response) in cases) MockWebServer().use { server ->
            server.start()
            server.enqueue(MockResponse.Builder().body(response).build())
            val counter = ProviderTokenCounter(http, mapOf(provider to server.url("/").toString().trimEnd('/')))
            val expected = if (provider == "xai") 3 else 31
            assertEquals(expected, counter.count(request(model)))
            assertEquals(expected, counter.count(request(model)))
            assertEquals(1, server.requestCount)
            val sent = server.takeRequest()
            val path = when (provider) {
                "gemini" -> "/v1beta/models/$model:countTokens"
                "anthropic" -> "/v1/messages/count_tokens"
                "xai" -> "/v1/tokenize-text"
                "kimi" -> "/v1/tokenizers/estimate-token-count"
                else -> "/v1/responses/input_tokens"
            }
            assertEquals(path, sent.target)
            val body = JSONObject(sent.body!!.utf8())
            assertFalse(body.has("stream"))
            assertFalse(body.has("max_tokens"))
            assertFalse(body.has("store"))
        }
    }

    @Test fun imagesAreSentToTheCounterWithoutGeneratingOrUploading() = runBlocking {
        MockWebServer().use { server ->
            server.start()
            server.enqueue(MockResponse.Builder().body("""{"input_tokens":1120}""").build())
            val counter = ProviderTokenCounter(http, mapOf("openai" to server.url("/").toString().trimEnd('/')))
            val f = DirectAttachment("test.png", "image/png", byteArrayOf(1, 2, 3))
            assertEquals(1120, counter.count(request("gpt-5", listOf(f))))
            val body = server.takeRequest().body!!.utf8()
            assertTrue(body.contains("input_image"))
            assertTrue(body.contains("data:image/png;base64,AQID"))
        }
    }

    @Test fun unsupportedMediaIsNotReportedAsCountedAndInvalidCountsStayUnknown() = runBlocking {
        val counter = ProviderTokenCounter(http)
        val image = listOf(DirectAttachment("test.png", "image/png", byteArrayOf(1)))
        assertNull(counter.count(request("grok-4", image)))
        assertNull(counter.count(request("deepseek-chat")))
        MockWebServer().use { server ->
            server.start()
            server.enqueue(MockResponse.Builder().body("""{"input_tokens":-1}""").build())
            val bad = ProviderTokenCounter(http, mapOf("openai" to server.url("/").toString().trimEnd('/')))
            assertNull(bad.count(request("gpt-5")))
        }
    }
}
