package com.minashin1120.aiplayground.data

import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test

class ModelUsageLedgerTest {
    @Test fun cumulativeUpdatesAndSeparateRequestsHaveDifferentAccounting() {
        val ledger = ModelUsageLedger()
        for (i in 1..2) {
            val call = ledger.begin("gpt-6.1-sol", "openai")
            val event = JSONObject("""{"response":{"id":"reply_$i","usage":{"input_tokens":10010,"output_tokens":1000,"output_tokens_details":{"reasoning_tokens":900}}}}""")
            call.observe(event); call.observe(event); call.finished = true
        }
        assertEquals(20020, ledger.total("input_tokens"))
        assertEquals(2000, ledger.total("output_tokens"))
        assertEquals(1800, ledger.total("thought_tokens"))
        assertEquals(2, ledger.snapshot().length())
        assertFalse(ledger.snapshot().toString().contains("reply_"))
    }

    @Test fun rejectedAndUnknownRequestsAreDistinguished() {
        val ledger = ModelUsageLedger()
        ledger.begin("gpt-6.1-sol", "openai").rejected = true
        ledger.begin("gpt-6.1-sol", "openai")
        assertEquals(1, ledger.snapshot().length())
        assertFalse(ledger.snapshot().getJSONObject(0).getBoolean("complete"))
    }

    @Test fun geminiAndClaudeUsageKeepThinkingAndCacheCountsCorrectly() {
        val ledger = ModelUsageLedger()
        ledger.begin("gemini-3.6-flash", "google").apply {
            observe(JSONObject("""{"usageMetadata":{"promptTokenCount":100,"candidatesTokenCount":20,"thoughtsTokenCount":10}}"""))
            finished = true
        }
        assertEquals(30, ledger.total("output_tokens"))
        assertEquals(10, ledger.total("thought_tokens"))
        val claude = ledger.begin("claude-sonnet-4-6", "anthropic")
        claude.observe(JSONObject("""{"message":{"usage":{"input_tokens":100,"cache_read_input_tokens":20,"cache_creation_input_tokens":30,"output_tokens":3}}}"""))
        claude.observe(JSONObject("""{"usage":{"output_tokens":1000}}""")); claude.finished = true
        assertEquals(250, ledger.total("input_tokens"))
        assertEquals(1030, ledger.total("output_tokens"))
    }
}
