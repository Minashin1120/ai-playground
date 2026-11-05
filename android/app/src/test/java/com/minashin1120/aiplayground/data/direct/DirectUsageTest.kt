package com.minashin1120.aiplayground.data.direct

import com.minashin1120.aiplayground.data.ModelUsageLedger
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import mockwebserver3.MockResponse
import mockwebserver3.MockWebServer
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test

class DirectUsageTest {
    private val http = DirectHttp(allowedHosts = emptySet(), allowCleartextLocalhost = true)

    @Test fun repeatedUsageEventsDoNotDuplicateARequestButFollowupsAddCosts() = runBlocking {
        MockWebServer().use { server ->
            server.start()
            for (i in 1..2) {
                val json = """{"type":"response.completed","response":{"id":"reply_$i","usage":{"input_tokens":10010,"output_tokens":1000,"output_tokens_details":{"reasoning_tokens":900}}}}"""
                server.enqueue(MockResponse.Builder().addHeader("Content-Type", "text/event-stream")
                    .body("data: $json\n\ndata: $json\n\ndata: [DONE]\n\n").build())
            }
            val ledger = ModelUsageLedger()
            withContext(UsageContext(ledger, "gpt-6.1-sol")) {
                repeat(2) { http.postSse(server.url("/v1/responses").toString(), emptyMap(), JSONObject().put("model", "gpt-6.1-sol")) { true } }
            }
            assertEquals(2, ledger.snapshot().length())
            assertEquals(20020, ledger.total("input_tokens"))
            assertEquals(2000, ledger.total("output_tokens"))
            assertEquals(1800, ledger.total("thought_tokens"))
        }
    }

    @Test fun countTokensAndRetrievalDoNotBecomeExtraGenerationCalls() = runBlocking {
        MockWebServer().use { server ->
            server.start()
            repeat(3) { server.enqueue(MockResponse.Builder().body("""{"usage":{"input_tokens":100,"output_tokens":20}}""").build()) }
            val ledger = ModelUsageLedger()
            withContext(UsageContext(ledger, "gpt-6.1-sol")) {
                http.postJson(server.url("/v1/responses").toString(), emptyMap(), JSONObject().put("model", "gpt-6.1-sol"))
                http.getJson(server.url("/v1/responses/answer").toString(), emptyMap())
                http.postJson(server.url("/v1/responses/input_tokens").toString(), emptyMap(), JSONObject().put("model", "gpt-6.1-sol"))
            }
            assertEquals(1, ledger.snapshot().length())
        }
    }
}
